"""
Thread-safe message queue for TUI-safe output.

All log messages and status updates flow through this queue
to prevent console output from corrupting the TUI display.

DESIGN NOTES:
- Queue operations are non-blocking to prevent deadlocks in batch processing
- Both pending queue and deferred list are bounded to prevent memory exhaustion
- Callbacks are called with error protection but errors are logged (not silent)
- All public methods are thread-safe

DO NOT:
- Remove thread locks - causes race conditions in multi-threaded batch mode
- Remove size limits - causes memory exhaustion on large jobs
- Swallow callback exceptions silently - always log for debugging
- Block on put() - causes deadlocks when queue is full

Usage:
    from apias.message_queue import get_message_queue, MessageLevel, TUIMessage

    queue = get_message_queue()
    queue.put(TUIMessage(MessageLevel.WARNING, "Rate limit approaching"))

    # In TUI render loop:
    for msg in queue.drain():
        display_message(msg)
"""

import logging
import queue
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Callable, Final, List, Optional

# Module-level logger for tracing message flow
logger = logging.getLogger(__name__)

# =============================================================================
# CENTRALIZED CONSTANTS - Single source of truth for queue configuration
# =============================================================================
DEFAULT_MAX_QUEUE_SIZE: Final[int] = 1000  # Max pending messages
DEFAULT_MAX_DEFERRED: Final[int] = 500  # Max deferred (persistent) messages


class MessageLevel(Enum):
    """Message priority/type for display formatting."""

    DEBUG = auto()
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    SUCCESS = auto()
    STATUS = auto()  # Task status updates
    PROGRESS = auto()  # Progress bar updates


@dataclass
class TUIMessage:
    """Single message for TUI display."""

    level: MessageLevel
    text: str
    task_id: int | None = None
    timestamp: datetime = field(default_factory=datetime.now)
    persist: bool = False  # If True, show in deferred summary


class TUIMessageQueue:
    """
    Thread-safe message queue that integrates with TUI managers.

    Features:
    - Non-blocking put operations (drops oldest if full)
    - Persistent messages stored for summary display (bounded)
    - Callback registration for real-time handling (with error logging)
    - Thread-safe for multi-threaded batch processing

    MEMORY SAFETY:
    - Queue is bounded by max_size to prevent memory exhaustion
    - Deferred list is bounded by max_deferred
    - When limits reached, oldest messages are dropped (FIFO)

    THREAD SAFETY:
    - All mutable state protected by _lock
    - Queue operations use thread-safe queue.Queue
    """

    def __init__(
        self,
        max_size: int = DEFAULT_MAX_QUEUE_SIZE,
        max_deferred: int = DEFAULT_MAX_DEFERRED,
    ) -> None:
        """
        Initialize message queue.

        Args:
            max_size: Maximum number of pending messages to buffer
            max_deferred: Maximum number of deferred messages to store
        """
        self._max_deferred = max_deferred
        self._queue: queue.Queue[TUIMessage] = queue.Queue(maxsize=max_size)
        self._deferred: List[TUIMessage] = []
        self._deferred_dropped = 0  # Count of dropped deferred messages
        self._lock = threading.Lock()
        self._callbacks: List[Callable[[TUIMessage], None]] = []
        self._callback_lock = threading.Lock()  # Separate lock for callbacks

        logger.debug(
            f"TUIMessageQueue initialized: max_size={max_size}, max_deferred={max_deferred}"
        )

    def put(self, message: TUIMessage) -> None:
        """
        Add a message to the queue (non-blocking).

        If the queue is full, drops the oldest message to make room.
        Never blocks - critical for preventing deadlocks in batch processing.

        Args:
            message: The message to queue
        """
        try:
            self._queue.put_nowait(message)
        except queue.Full:
            # Drop oldest message if queue is full - never block
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(message)
            except queue.Empty:
                # Race condition: another thread drained - just put
                try:
                    self._queue.put_nowait(message)
                except queue.Full:
                    # Still full after another drain attempt - log and drop
                    logger.warning(
                        f"Message queue full, dropping message: {message.text[:50]}"
                    )

        # Store persistent messages for summary (bounded)
        if message.persist:
            with self._lock:
                # Enforce bounded deferred list
                if len(self._deferred) >= self._max_deferred:
                    self._deferred.pop(0)  # Drop oldest (FIFO)
                    self._deferred_dropped += 1
                    if self._deferred_dropped == 1:
                        logger.warning(
                            f"Deferred list full (max={self._max_deferred}), "
                            "dropping oldest deferred messages"
                        )
                self._deferred.append(message)

        # Notify callbacks (for real-time handling)
        # Use separate lock to prevent deadlock if callback calls back into queue
        with self._callback_lock:
            callbacks_copy = list(self._callbacks)

        for callback in callbacks_copy:
            try:
                callback(message)
            except Exception as e:
                # Log callback errors instead of swallowing silently
                # DO NOT re-raise - one bad callback shouldn't break the queue
                logger.warning(
                    f"TUI callback error (callback will continue to be called): {e}"
                )

    def drain(self) -> List[TUIMessage]:
        """
        Get all pending messages (clears queue).

        Thread-safe but not atomic with put() - messages added during
        drain may or may not be included.

        Returns:
            List of all messages that were in the queue
        """
        messages = []
        # Drain until empty - this is atomic per message but not for entire drain
        while True:
            try:
                messages.append(self._queue.get_nowait())
            except queue.Empty:
                break

        if messages:
            logger.debug(f"Drained {len(messages)} messages from queue")
        return messages

    def get_deferred(self) -> List[TUIMessage]:
        """
        Get messages marked for deferred display.

        Returns:
            Copy of the deferred messages list (safe to iterate)
        """
        with self._lock:
            return list(self._deferred)

    def clear_deferred(self) -> None:
        """Clear the deferred messages list."""
        with self._lock:
            count = len(self._deferred)
            self._deferred.clear()
            self._deferred_dropped = 0
        if count > 0:
            logger.debug(f"Cleared {count} deferred messages")

    def register_callback(self, callback: Callable[[TUIMessage], None]) -> None:
        """
        Register callback for real-time message handling.

        Callbacks are called synchronously on put() so they should be fast.
        Slow callbacks will delay message processing.

        Args:
            callback: Function called for each message added
        """
        with self._callback_lock:
            self._callbacks.append(callback)
        logger.debug(
            f"Registered TUI callback: {callback.__name__ if hasattr(callback, '__name__') else 'anonymous'}"
        )

    def unregister_callback(self, callback: Callable[[TUIMessage], None]) -> None:
        """
        Remove a previously registered callback.

        Args:
            callback: The callback to remove
        """
        with self._callback_lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)
                logger.debug("Unregistered TUI callback")

    @property
    def pending_count(self) -> int:
        """Get number of messages waiting in queue."""
        return self._queue.qsize()

    @property
    def deferred_count(self) -> int:
        """Get number of deferred messages."""
        with self._lock:
            return len(self._deferred)

    @property
    def deferred_dropped_count(self) -> int:
        """Get number of deferred messages that were dropped due to limit."""
        with self._lock:
            return self._deferred_dropped


# Global instance for module-level access
_global_queue: TUIMessageQueue | None = None
_global_lock = threading.Lock()


def get_message_queue() -> TUIMessageQueue:
    """
    Get or create the global message queue.

    Thread-safe singleton access.

    Returns:
        The global TUIMessageQueue instance
    """
    global _global_queue
    if _global_queue is None:
        with _global_lock:
            # Double-check locking pattern
            if _global_queue is None:
                _global_queue = TUIMessageQueue()
    return _global_queue


def reset_message_queue() -> None:
    """
    Reset the global message queue.

    Creates a new empty queue. Useful for testing.
    """
    global _global_queue
    with _global_lock:
        _global_queue = TUIMessageQueue()


# Convenience functions for common message types


def queue_info(text: str, task_id: int | None = None) -> None:
    """Queue an info-level message."""
    get_message_queue().put(
        TUIMessage(level=MessageLevel.INFO, text=text, task_id=task_id)
    )


def queue_warning(text: str, task_id: int | None = None, persist: bool = True) -> None:
    """Queue a warning-level message (persisted by default)."""
    get_message_queue().put(
        TUIMessage(
            level=MessageLevel.WARNING,
            text=text,
            task_id=task_id,
            persist=persist,
        )
    )


def queue_error(text: str, task_id: int | None = None, persist: bool = True) -> None:
    """Queue an error-level message (persisted by default)."""
    get_message_queue().put(
        TUIMessage(
            level=MessageLevel.ERROR,
            text=text,
            task_id=task_id,
            persist=persist,
        )
    )


def queue_success(text: str, task_id: int | None = None) -> None:
    """Queue a success-level message."""
    get_message_queue().put(
        TUIMessage(level=MessageLevel.SUCCESS, text=text, task_id=task_id)
    )


def queue_status(text: str, task_id: int | None = None) -> None:
    """Queue a status update message."""
    get_message_queue().put(
        TUIMessage(level=MessageLevel.STATUS, text=text, task_id=task_id)
    )


def queue_progress(text: str, task_id: int | None = None) -> None:
    """Queue a progress update message."""
    get_message_queue().put(
        TUIMessage(level=MessageLevel.PROGRESS, text=text, task_id=task_id)
    )
