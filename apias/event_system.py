"""
Event-driven communication system for APIAS.

This module provides a decoupled event bus architecture allowing components to
communicate without direct dependencies. All error handling, status updates, and
user dialogs flow through this event system.

Key Design Principles:
- Lock-free publishing: Worker threads publish events without blocking
- Type-safe subscriptions: Subscribe to specific event types
- Bounded processing: dispatch() has timeout to prevent blocking
- Thread-safe: All operations safe for concurrent use

Architecture:
    Worker Thread → publish(Event) → EventBus Queue → dispatch() → Subscribers

Usage:
    # Initialize
    event_bus = EventBus()

    # Subscribe
    event_bus.subscribe(ErrorEvent, handle_error)

    # Publish (from worker thread)
    event_bus.publish(ErrorEvent(...))

    # Process (from main thread) - use centralized timeout from config.py
    from apias.config import EVENT_DISPATCH_TIMEOUT
    event_bus.dispatch(timeout=EVENT_DISPATCH_TIMEOUT)
"""

import logging
import queue
import threading
import time
import uuid
from abc import ABC
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar

logger = logging.getLogger(__name__)


# ============================================================================
# Event Base Classes
# ============================================================================


@dataclass
class Event(ABC):
    """
    Base class for all events in the system.

    All events have a timestamp and unique ID for tracking and debugging.
    Subclasses should be dataclasses with specific fields for their context.

    Design Note: Using dataclass for immutability and automatic __repr__.
    Note: ABC is used for type hierarchy even without abstract methods.
    Events should be created once and never modified after publishing.
    """

    timestamp: datetime = field(default_factory=datetime.now)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))


# TypeVar for generic event handling in subscribe()
E = TypeVar("E", bound=Event)


# ============================================================================
# Status Events
# ============================================================================


# Import URLState from batch_tui to avoid duplicate enum definitions
# WHY: Having two URLState enums caused mypy type errors in status_pipeline.py
# where batch_tui.URLState was assigned to StatusEvent.state (event_system.URLState)
from apias.batch_tui import URLState


@dataclass
class StatusEvent(Event):
    """
    Status update from a worker thread processing a URL.

    Published when worker thread wants to update task status in TUI.
    Replaces direct calls to batch_tui.update_task().

    Fields:
        task_id: Numeric task identifier (1-based)
        state: Current processing state
        message: Status message to display (e.g., "⚠️ AI service timeout. Retrying...")
        progress_pct: Progress percentage (0.0-100.0)
        extras: Additional fields (size_in, size_out, cost, error, chunk info)
    """

    # WHY fields before Event fields: Python dataclass requires non-default args before default args
    # Event has timestamp and event_id with defaults, so StatusEvent required fields must come first
    task_id: int = field(kw_only=True)
    state: URLState = field(kw_only=True)
    message: str = field(default="", kw_only=True)
    progress_pct: float = field(default=0.0, kw_only=True)
    extras: Dict[str, Any] = field(default_factory=dict, kw_only=True)


# ============================================================================
# Error Events
# ============================================================================


class ErrorCategory(Enum):
    """
    SINGLE SOURCE OF TRUTH for error classification across APIAS.

    These categories determine:
    - Circuit breaker thresholds (per-category in YAML config)
    - Error recoverability (via RECOVERABLE_CATEGORIES set below)
    - User messaging (different guidance per category)
    - Summary reporting icons and descriptions

    DRY PRINCIPLE: This enum is THE definitive error classification.
    DO NOT create duplicate ErrorCategory enums elsewhere.
    Import this: `from apias.event_system import ErrorCategory, RECOVERABLE_CATEGORIES`
    """

    # Success state
    NONE = auto()  # No error - successful operation

    # API Errors
    QUOTA_EXCEEDED = auto()  # Insufficient API quota - FATAL
    RATE_LIMIT = auto()  # 429 rate limit - FATAL (short-term)
    API_TIMEOUT = auto()  # Request timeout - RECOVERABLE
    AUTHENTICATION = auto()  # Auth failure - FATAL
    INVALID_API_KEY = auto()  # Bad API key - FATAL

    # Network Errors
    CONNECTION_ERROR = auto()  # Network connection failed - RECOVERABLE
    SERVER_ERROR = auto()  # 500/502/503 - RECOVERABLE

    # Content Errors
    INVALID_RESPONSE = auto()  # LLM returned invalid format - RECOVERABLE
    SOURCE_NOT_FOUND = auto()  # 404 - NON-RECOVERABLE (skip URL)
    PARSE_ERROR = auto()  # HTML parsing failed - RECOVERABLE
    XML_VALIDATION = auto()  # XML validation failed - RECOVERABLE

    # Unknown
    UNKNOWN = auto()  # Unclassified error


# =============================================================================
# RECOVERABLE CATEGORIES - Single source of truth for retry logic
# =============================================================================
# IMPORTANT: RATE_LIMIT is NOT recoverable - hitting API limits means we should
# stop immediately to avoid wasting time and potentially being banned.
# Only transient errors (timeout, connection, server error) are recoverable.
# DRY: Import this set - DO NOT recreate elsewhere.
RECOVERABLE_CATEGORIES: frozenset[ErrorCategory] = frozenset(
    {
        ErrorCategory.API_TIMEOUT,
        ErrorCategory.CONNECTION_ERROR,
        ErrorCategory.SERVER_ERROR,
        # Parse/validation errors can be retried with different input
        ErrorCategory.PARSE_ERROR,
        ErrorCategory.XML_VALIDATION,
        ErrorCategory.INVALID_RESPONSE,
    }
)


@dataclass
class ErrorEvent(Event):
    """
    Error occurrence from any part of the system.

    Published when any error occurs (network, API, scraping, validation).
    Replaces direct calls to error_tracker.record().

    Fields:
        category: Error classification (determines circuit breaker logic)
        message: Human-readable error description
        task_id: Task number where error occurred (None for global errors)
        url: URL being processed (None for non-URL errors)
        exception_type: Exception class name (e.g., "RateLimitError")
        exception_traceback: Full traceback string for debugging
        context: Additional error context (retry count, chunk number, etc.)
        recoverable: Whether error is transient (from RECOVERABLE_CATEGORIES)
    """

    # WHY kw_only: Allows required fields after Event's default fields without ordering issues
    category: ErrorCategory = field(kw_only=True)
    message: str = field(kw_only=True)
    task_id: int | None = field(default=None, kw_only=True)
    url: str | None = field(default=None, kw_only=True)
    exception_type: str | None = field(default=None, kw_only=True)
    exception_traceback: str | None = field(default=None, kw_only=True)
    context: Dict[str, Any] = field(default_factory=dict, kw_only=True)
    recoverable: bool = field(default=True, kw_only=True)


# ============================================================================
# Circuit Breaker Events
# ============================================================================


@dataclass
class CircuitBreakerEvent(Event):
    """
    Circuit breaker has tripped - processing must stop.

    Published when circuit breaker decides to stop processing due to
    repeated errors. This triggers immediate wake-up of main thread
    and shows user dialog.

    Fields:
        reason: Why circuit tripped (e.g., "3 consecutive RATE_LIMIT errors")
        affected_tasks: List of task IDs still pending when circuit tripped
        trigger_category: Error category that caused the trip
        consecutive_counts: Snapshot of consecutive error counts at trip time
    """

    # WHY kw_only: Allows required fields after Event's default fields without ordering issues
    reason: str = field(kw_only=True)
    affected_tasks: List[int] = field(default_factory=list, kw_only=True)
    trigger_category: ErrorCategory | None = field(default=None, kw_only=True)
    consecutive_counts: Dict[ErrorCategory, int] = field(
        default_factory=dict, kw_only=True
    )


# ============================================================================
# Dialog Events
# ============================================================================


class DialogType(Enum):
    """Types of dialogs that can be shown to user"""

    CIRCUIT_BREAKER = auto()  # Processing paused dialog
    ERROR_SUMMARY = auto()  # End-of-session error summary
    CONFIRMATION = auto()  # User confirmation needed
    INFO = auto()  # Informational message


class DialogPriority(Enum):
    """Priority for dialog queue (lower value = higher priority)"""

    CRITICAL = 0  # Circuit breaker, auth failure
    HIGH = 1  # Error summary with failures
    NORMAL = 2  # General info
    LOW = 3  # Success confirmation


@dataclass
class DialogEvent(Event):
    """
    Request to show a dialog to the user.

    Published when system needs to show user a dialog. Dialog manager
    queues these by priority and shows them after TUI stops.

    Fields:
        dialog_type: What kind of dialog to show
        priority: Queue priority (CRITICAL shown first)
        context: Dialog-specific data (error details, file paths, etc.)
    """

    # WHY kw_only: Allows required fields after Event's default fields without ordering issues
    dialog_type: DialogType = field(kw_only=True)
    priority: DialogPriority = field(kw_only=True)
    context: Dict[str, Any] = field(default_factory=dict, kw_only=True)


# ============================================================================
# Event Bus
# ============================================================================


class EventBus:
    """
    Thread-safe event bus using pub/sub pattern.

    Architecture:
    - Publishers: call publish() from any thread (lock-free, non-blocking)
    - Subscribers: register handlers for specific event types
    - Dispatcher: main thread calls dispatch() periodically to process events

    Thread Safety:
    - publish(): Lock-free (queue.Queue is thread-safe)
    - subscribe(): Uses _lock to protect _subscribers dict
    - dispatch(): Runs in main thread only (no concurrent dispatch)

    Performance:
    - publish(): O(1), non-blocking
    - dispatch(): O(n) where n = events processed per timeout
    - Typical: 10-50 events per dispatch() call

    Usage:
        bus = EventBus()

        # Subscribe (main thread, during setup)
        bus.subscribe(ErrorEvent, my_error_handler)

        # Publish (worker thread, during processing)
        bus.publish(ErrorEvent(category=..., message=...))

        # Dispatch (main thread, in event loop)
        count = bus.dispatch(timeout=0.05)  # Process for up to 50ms
    """

    def __init__(self, max_queue_size: int = 10000):
        """
        Initialize event bus.

        Args:
            max_queue_size: Maximum events in queue before blocking.
                           Default 10000 should never be reached in practice.
                           If reached, publish() will raise queue.Full.
        """
        # Subscriber registry: EventType -> List[handler_function]
        self._subscribers: Dict[Type[Event], List[Callable[[Event], None]]] = (
            defaultdict(list)
        )

        # Thread-safe event queue (FIFO)
        # WHY queue.Queue: Lock-free for publishers, thread-safe, bounded
        self._event_queue: queue.Queue[Event] = queue.Queue(maxsize=max_queue_size)

        # Lock for subscriber registration (NOT needed for queue operations)
        self._lock = threading.Lock()

        # Lock for metrics counters (needed for thread-safe increment)
        # WHY separate lock: Avoids contention with subscriber registration
        self._stats_lock = threading.Lock()

        # Metrics for monitoring
        self._published_count = 0
        self._dispatched_count = 0
        self._dropped_count = 0  # If queue full

        logger.debug(f"EventBus initialized with max_queue_size={max_queue_size}")

    def publish(self, event: Event) -> None:
        """
        Publish an event to the bus (lock-free, non-blocking).

        This method is called from worker threads and must be fast.
        Events are queued and processed later by dispatch().

        Args:
            event: Event instance to publish

        Raises:
            queue.Full: If event queue is full (should never happen with 10k limit)

        Design Note: We use put_nowait() to fail fast if queue is full.
        This is better than blocking worker threads.
        """
        try:
            self._event_queue.put_nowait(event)
            # THREAD SAFETY: Increment counter under lock
            # WHY: publish() is called from multiple worker threads
            with self._stats_lock:
                self._published_count += 1

            # Log at TRACE level (only if DEBUG enabled)
            logger.debug(
                f"Published {type(event).__name__} (id={event.event_id[:8]}..., queue_size={self._event_queue.qsize()})"
            )

        except queue.Full:
            # Queue full - this should NEVER happen with 10k limit
            # If it does, we're publishing faster than we can process
            # THREAD SAFETY: Increment counter under lock
            with self._stats_lock:
                self._dropped_count += 1
            logger.error(
                f"Event queue FULL! Dropped {type(event).__name__}. Published={self._published_count}, Dropped={self._dropped_count}"
            )
            raise

    def subscribe(self, event_type: Type[E], handler: Callable[[E], None]) -> None:
        """
        Subscribe to an event type.

        When events of this type are dispatched, handler will be called.
        Multiple handlers can subscribe to the same event type.

        Args:
            event_type: Event class to subscribe to (e.g., ErrorEvent)
            handler: Function to call: handler(event: E) -> None

        Thread Safety: Uses lock to protect _subscribers dict.
        Usually called during initialization, not performance-critical.

        Example:
            def handle_error(event: ErrorEvent):
                print(f"Error: {event.message}")

            bus.subscribe(ErrorEvent, handle_error)
        """
        with self._lock:
            # Cast handler to base type for storage (callers use specific types)
            self._subscribers[event_type].append(handler)  # type: ignore[arg-type]
            logger.debug(f"Subscribed {handler.__name__} to {event_type.__name__}")

    def dispatch(self, timeout: float = 0.01) -> int:
        """
        Process pending events by calling subscribers.

        This should be called from the main thread in the event loop.
        Processes events until timeout expires or queue is empty.

        Args:
            timeout: Maximum time in seconds to spend processing (default 10ms)

        Returns:
            Number of events processed

        Design Note: We process events until timeout to avoid blocking
        the main loop indefinitely. If queue has many events, we'll
        process them over multiple dispatch() calls.

        Thread Safety: Should only be called from main thread.
        Handlers execute sequentially (no concurrent handler execution).
        """
        count = 0
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            try:
                # Non-blocking get
                event = self._event_queue.get_nowait()

                # Dispatch to subscribers
                self._dispatch_to_subscribers(event)

                count += 1
                # THREAD SAFETY: Increment counter under lock
                # WHY: get_stats() might be called from other threads
                with self._stats_lock:
                    self._dispatched_count += 1

            except queue.Empty:
                # No more events to process
                break

        if count > 0:
            logger.debug(
                f"Dispatched {count} events (total={self._dispatched_count}, queue_remaining={self._event_queue.qsize()})"
            )

        return count

    def _dispatch_to_subscribers(self, event: Event) -> None:
        """
        Call all handlers subscribed to this event type.

        Handlers are called sequentially in registration order.
        If a handler raises an exception, we log it and continue
        to other handlers (fail-safe behavior).

        Args:
            event: Event to dispatch
        """
        event_type = type(event)

        # THREAD SAFETY FIX: Make a defensive copy of handlers list
        # WHY: While dict.get() is atomic in CPython, iterating over the returned
        # list is NOT protected. If subscribe() is called during iteration,
        # the list could be modified mid-iteration causing race conditions.
        # DO NOT rely on GIL for thread safety - it only guarantees atomic operations,
        # not iteration safety over mutable collections.
        with self._lock:
            handlers = list(self._subscribers.get(event_type, []))

        if not handlers:
            logger.debug(
                f"No subscribers for {event_type.__name__} (id={event.event_id[:8]}...)"
            )
            return

        # Call each handler
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                # Handler failed - log but continue to other handlers
                # This ensures one broken handler doesn't stop all event processing
                logger.error(
                    f"Handler {handler.__name__} failed for {event_type.__name__}: {e}",
                    exc_info=True,  # Include traceback in log
                )

    def get_stats(self) -> Dict[str, int]:
        """
        Get event bus statistics for monitoring.

        Returns:
            Dict with:
            - published: Total events published
            - dispatched: Total events dispatched
            - dropped: Events dropped due to queue full
            - pending: Events currently in queue

        Thread Safety: Safe to call from any thread.
        """
        # THREAD SAFETY: Read counters under lock for consistent snapshot
        # WHY: Without lock, could read partially-updated values
        with self._stats_lock:
            return {
                "published": self._published_count,
                "dispatched": self._dispatched_count,
                "dropped": self._dropped_count,
                "pending": self._event_queue.qsize(),
            }

    def clear(self) -> int:
        """
        Clear all pending events from queue.

        Used for testing or emergency reset.

        Returns:
            Number of events cleared
        """
        count = 0
        while not self._event_queue.empty():
            try:
                self._event_queue.get_nowait()
                count += 1
            except queue.Empty:
                break

        logger.info(f"Cleared {count} pending events from queue")
        return count
