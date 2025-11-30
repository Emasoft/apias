"""
Real-time status update pipeline with atomic snapshots.

This module implements a hybrid status update system that combines stable polling
with event-driven wake-up for critical errors. Worker threads publish status updates
through the event bus, and the main thread renders from atomic snapshots.

Key Design Principles:
- Atomic snapshots: TUI renders from immutable copy (no lock during render)
- Hybrid timing: 50ms polling baseline + instant wake-up for critical errors
- Thread-safe: All state updates protected by _task_lock
- History tracking: Keep last 5 status messages per task
- Lock hierarchy: StatusPipeline._task_lock is Level 2 (after ErrorCollector._lock)

Architecture:
    Worker Thread → publish(StatusEvent) → EventBus → StatusPipeline
                                                       ↓
    Main Thread ← get_snapshot() ← StatusPipeline ← wait_for_update()

Usage:
    # Initialize
    status_pipeline = StatusPipeline(event_bus)

    # Worker thread updates status
    status_pipeline.update_status(
        task_id=1,
        state=URLState.PROCESSING,
        message="Processing chunk 2/5",
        progress_pct=40.0
    )

    # Main thread waits and renders
    critical = status_pipeline.wait_for_update(timeout=0.05)
    snapshot = status_pipeline.get_snapshot()
    render_tui(snapshot)
"""

import copy
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from apias.batch_tui import URLState, URLTask
from apias.event_system import CircuitBreakerEvent, ErrorEvent, EventBus, StatusEvent

logger = logging.getLogger(__name__)


# ============================================================================
# Status Pipeline
# ============================================================================


@dataclass
class TaskSnapshot:
    """
    Immutable snapshot of a task's state at a point in time.

    This is returned by get_snapshot() and can be safely used by TUI
    without holding locks. All fields are copied from URLTask.

    Design Note: We use a separate snapshot class instead of deep-copying
    URLTask to make the immutability contract explicit.
    """

    task_id: int
    url: str
    state: URLState
    progress_pct: float
    size_in: int
    size_out: int
    cost: float
    duration: float
    start_time: Optional[float]
    error: str
    current_chunk: int
    total_chunks: int
    status_message: str
    # Status history: last 5 (timestamp, message) tuples
    status_history: List[Tuple[datetime, str]] = field(default_factory=list)


class StatusPipeline:
    """
    Hybrid status update system: 50ms polling + event wake-up.

    This class manages the real-time status of all tasks being processed,
    providing atomic snapshots for TUI rendering and instant wake-up for
    critical events (circuit breaker, quota exceeded, auth failure).

    Thread Safety:
    - Worker threads: Call update_status() to publish events (lock-free)
    - Event handlers: Update task state under _task_lock
    - Main thread: Call get_snapshot() to get immutable copy (lock-free read)

    Performance:
    - Normal operation: TUI polls at 20 FPS (50ms) with minimal overhead
    - Critical events: Instant wake-up via threading.Event (no polling delay)
    - Snapshot: O(n) deep copy where n = number of tasks (typically <100)

    Lock Hierarchy (Level 2):
    - Never acquire ErrorCollector._lock while holding StatusPipeline._task_lock
    - OK to acquire StatusPipeline._task_lock while holding ErrorCollector._lock
    """

    def __init__(self, event_bus: EventBus):
        """
        Initialize status pipeline.

        Args:
            event_bus: EventBus instance for pub/sub communication
        """
        self._event_bus = event_bus

        # Task state dictionary: task_id -> URLTask
        # WHY Dict: Fast O(1) lookup by task_id, typical size <100 tasks
        self._tasks: Dict[int, URLTask] = {}

        # Lock for task state updates (Level 2 in lock hierarchy)
        # WHY: Protects _tasks dict from concurrent modifications
        self._task_lock = threading.Lock()

        # Critical event flag: Set when circuit breaker trips or auth fails
        # WHY threading.Event: Allows instant wake-up of main thread without polling
        self._critical_event = threading.Event()

        # Subscribe to relevant events
        event_bus.subscribe(StatusEvent, self._on_status_event)
        event_bus.subscribe(CircuitBreakerEvent, self._on_circuit_breaker)
        event_bus.subscribe(ErrorEvent, self._on_error_event)

        logger.debug("StatusPipeline initialized")

    def initialize_tasks(self, urls: List[str]) -> None:
        """
        Initialize task state for a list of URLs.

        Called once at the start of batch processing.

        Args:
            urls: List of URLs to process

        Design Note: This creates URLTask objects before worker threads start,
        so no locking is needed. All tasks start in PENDING state.
        """
        with self._task_lock:
            self._tasks = {
                task_id: URLTask(task_id=task_id, url=url, state=URLState.PENDING)
                for task_id, url in enumerate(urls, start=1)
            }

        logger.info(f"Initialized {len(urls)} tasks in StatusPipeline")

    def update_status(
        self,
        task_id: int,
        state: URLState,
        message: str = "",
        progress_pct: float = 0.0,
        **kwargs,
    ) -> None:
        """
        Update task status (called by worker threads).

        This is the main entry point for worker threads to update task status.
        It publishes a StatusEvent to the event bus, which is then processed
        by _on_status_event() in the main thread.

        Args:
            task_id: Task ID (1-based)
            state: New task state
            message: Status message (e.g., "⚠️ AI service timeout. Retrying...")
            progress_pct: Progress percentage (0.0-100.0)
            **kwargs: Additional fields (size_in, size_out, cost, error, chunks)

        Design Note: This method is lock-free and non-blocking. Worker threads
        can call it frequently without worrying about blocking.

        Example:
            status_pipeline.update_status(
                task_id=1,
                state=URLState.PROCESSING,
                message="Processing chunk 2/5",
                progress_pct=40.0,
                size_in=12000,
                current_chunk=2,
                total_chunks=5
            )
        """
        # Publish event to bus (lock-free, non-blocking)
        self._event_bus.publish(
            StatusEvent(
                task_id=task_id,
                state=state,
                message=message,
                progress_pct=progress_pct,
                extras=kwargs,
            )
        )

    def _on_status_event(self, event: StatusEvent) -> None:
        """
        Event handler: Update task state atomically.

        Called by EventBus.dispatch() in the main thread when a StatusEvent
        is published by a worker thread.

        Args:
            event: StatusEvent with task update data

        Thread Safety: Runs in main thread, uses _task_lock for atomicity.
        """
        with self._task_lock:
            task = self._tasks.get(event.task_id)
            if not task:
                logger.warning(f"Status update for unknown task {event.task_id}")
                return

            # Update basic fields
            task.state = event.state
            task.status_message = event.message
            # WHY clamp: Prevents progress overflow from buggy callers
            # Defense in depth - batch_tui also clamps, but catch issues at source
            task.progress_pct = max(0.0, min(100.0, event.progress_pct))

            # Update optional fields from extras
            if "size_in" in event.extras:
                task.size_in = event.extras["size_in"]
            if "size_out" in event.extras:
                task.size_out = event.extras["size_out"]
            if "cost" in event.extras:
                task.cost = event.extras["cost"]
            if "duration" in event.extras:
                task.duration = event.extras["duration"]
            if "error" in event.extras:
                task.error = event.extras["error"]
            if "current_chunk" in event.extras:
                task.current_chunk = event.extras["current_chunk"]
            if "total_chunks" in event.extras:
                task.total_chunks = event.extras["total_chunks"]

            # Keep status history (last 5 messages)
            # WHY: Allows TUI to show recent status changes for debugging
            if event.message:
                # WHY: URLTask.status_history is now a proper dataclass field (no hasattr needed)
                task.status_history.append((event.timestamp, event.message))
                if len(task.status_history) > 5:
                    task.status_history.pop(0)  # Remove oldest

            logger.debug(
                f"Updated task {event.task_id}: {event.state.name} "
                f"({event.progress_pct:.1f}%) - {event.message}"
            )

    def _on_error_event(self, event: ErrorEvent) -> None:
        """
        Event handler: Update task with error information.

        Called when an ErrorEvent is published for a specific task.
        Updates the task's error field and status message.

        Args:
            event: ErrorEvent with error details

        Thread Safety: Runs in main thread, uses _task_lock for atomicity.
        """
        if event.task_id is None:
            # Global error not associated with a task
            return

        with self._task_lock:
            task = self._tasks.get(event.task_id)
            if not task:
                logger.warning(f"Error event for unknown task {event.task_id}")
                return

            # Update error message (preserves previous value if already set)
            if not task.error:
                task.error = event.message

            # Add to status history
            # WHY: URLTask.status_history is now a proper dataclass field (no hasattr needed)
            task.status_history.append((event.timestamp, f"❌ {event.message}"))
            if len(task.status_history) > 5:
                task.status_history.pop(0)

            logger.debug(f"Recorded error for task {event.task_id}: {event.message}")

    def _on_circuit_breaker(self, event: CircuitBreakerEvent) -> None:
        """
        Critical event handler: Wake up main thread immediately.

        When the circuit breaker trips, we need to stop processing immediately
        and show a dialog to the user. This sets the critical event flag to
        wake up the main thread from wait_for_update().

        Args:
            event: CircuitBreakerEvent with trip details

        Design Note: This handler just sets a flag; the actual circuit breaker
        dialog is shown by the main thread after it wakes up.
        """
        self._critical_event.set()
        logger.critical(
            f"Circuit breaker tripped: {event.reason}. "
            f"Main thread will wake up immediately."
        )

    def get_snapshot(self) -> Dict[int, TaskSnapshot]:
        """
        Get atomic snapshot of all tasks for rendering.

        Returns a deep copy of all task state, allowing the TUI to render
        without holding locks or worrying about concurrent modifications.

        Returns:
            Dict mapping task_id -> TaskSnapshot (immutable)

        Performance: O(n) where n = number of tasks (typically <100)
        Typical cost: <1ms for 100 tasks

        Thread Safety: Safe to call from any thread. TUI can render the
        returned snapshot without worrying about race conditions.

        Design Note: We return TaskSnapshot instead of URLTask to make
        the immutability contract explicit and avoid accidental mutations.
        """
        with self._task_lock:
            snapshot = {}
            for task_id, task in self._tasks.items():
                # Create immutable snapshot
                snapshot[task_id] = TaskSnapshot(
                    task_id=task.task_id,
                    url=task.url,
                    state=task.state,
                    progress_pct=task.progress_pct,
                    size_in=task.size_in,
                    size_out=task.size_out,
                    cost=task.cost,
                    duration=task.duration,
                    start_time=task.start_time,
                    error=task.error,
                    current_chunk=task.current_chunk,
                    total_chunks=task.total_chunks,
                    status_message=task.status_message,
                    # Deep copy history to ensure immutability
                    # WHY: URLTask.status_history is now a proper dataclass field (no getattr needed)
                    status_history=copy.deepcopy(task.status_history),
                )

        logger.debug(f"Created snapshot of {len(snapshot)} tasks")
        return snapshot

    def wait_for_update(self, timeout: float = 0.05) -> bool:
        """
        Wait for critical event or timeout (hybrid polling).

        This implements the hybrid polling strategy:
        - Normal operation: Wait up to 50ms (20 FPS baseline)
        - Critical event: Return immediately when circuit breaker trips

        Args:
            timeout: Maximum wait time in seconds (default 50ms = 20 FPS)

        Returns:
            True if critical event occurred, False if timeout

        Design Note: The main thread calls this in a loop to implement
        responsive TUI updates while remaining efficient.

        Example:
            while not done:
                critical = status_pipeline.wait_for_update(timeout=0.05)
                if critical:
                    # Circuit breaker tripped - handle immediately
                    show_circuit_breaker_dialog()
                    break

                # Normal update - render TUI
                snapshot = status_pipeline.get_snapshot()
                render_tui(snapshot)
        """
        # Wait for critical event with timeout
        # WHY threading.Event.wait(): Efficient blocking with instant wake-up
        critical = self._critical_event.wait(timeout)

        if critical:
            logger.debug("Critical event detected, waking up main thread")

        return critical

    def clear_critical_flag(self) -> None:
        """
        Clear the critical event flag.

        Called after handling a critical event to reset the flag for
        future events.

        Design Note: This is necessary because threading.Event.wait()
        returns immediately if the flag is set, even after timeout.
        """
        self._critical_event.clear()
        logger.debug("Cleared critical event flag")

    def get_task(self, task_id: int) -> Optional[URLTask]:
        """
        Get a single task by ID.

        Returns a reference to the actual URLTask object, not a snapshot.
        Caller should not modify the returned object without holding _task_lock.

        Args:
            task_id: Task ID to retrieve

        Returns:
            URLTask if found, None otherwise

        Design Note: This is for backward compatibility with code that needs
        direct task access. New code should use get_snapshot() instead.
        """
        with self._task_lock:
            return self._tasks.get(task_id)

    def get_stats(self) -> Dict[str, int]:
        """
        Get pipeline statistics for monitoring.

        Returns:
            Dict with:
            - total: Total number of tasks
            - pending: Tasks in PENDING state
            - active: Tasks in SCRAPING/PROCESSING state
            - complete: Tasks in COMPLETE state
            - failed: Tasks in FAILED state
        """
        with self._task_lock:
            stats = {
                "total": len(self._tasks),
                "pending": sum(
                    1 for t in self._tasks.values() if t.state == URLState.PENDING
                ),
                "active": sum(
                    1
                    for t in self._tasks.values()
                    if t.state in (URLState.SCRAPING, URLState.PROCESSING)
                ),
                "complete": sum(
                    1 for t in self._tasks.values() if t.state == URLState.COMPLETE
                ),
                "failed": sum(
                    1 for t in self._tasks.values() if t.state == URLState.FAILED
                ),
            }

        return stats
