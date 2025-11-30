"""
Comprehensive unit tests for apias/event_system.py

Tests cover EventBus core functionality, thread safety, dispatch behavior,
and Event class creation with realistic, non-mocked test scenarios.

All tests use real threading, real queues, and real event processing.
No mocks, no conceptual tests - only real functional tests.
"""

import queue as queue_module
import threading
import time
from typing import List

import pytest

from apias.event_system import (
    CircuitBreakerEvent,
    DialogEvent,
    DialogPriority,
    DialogType,
    ErrorCategory,
    ErrorEvent,
    Event,
    EventBus,
    StatusEvent,
    URLState,
)

# ============================================================================
# EventBus Core Functionality (5 tests)
# ============================================================================


def test_basic_publish_subscribe_flow():
    """Test basic publish/subscribe flow - publish event and verify handler called."""
    bus = EventBus()
    received_events: List[ErrorEvent] = []

    def handler(event: ErrorEvent) -> None:
        received_events.append(event)

    # Subscribe handler
    bus.subscribe(ErrorEvent, handler)

    # Publish event
    error = ErrorEvent(
        category=ErrorCategory.API_TIMEOUT,
        message="Request timeout after 30s",
        task_id=42,
        url="https://example.com",
        exception_type="TimeoutError",
        recoverable=True,
    )
    bus.publish(error)

    # Dispatch to process
    count = bus.dispatch(timeout=1.0)

    # Verify
    assert count == 1, "Should have dispatched exactly 1 event"
    assert len(received_events) == 1, "Handler should have been called once"
    assert received_events[0] is error, (
        "Handler should receive the exact event published"
    )
    assert received_events[0].category == ErrorCategory.API_TIMEOUT
    assert received_events[0].message == "Request timeout after 30s"
    assert received_events[0].task_id == 42


def test_multiple_handlers_for_same_event_type():
    """Test multiple handlers for same event type - all handlers should be called."""
    bus = EventBus()
    handler1_calls: List[StatusEvent] = []
    handler2_calls: List[StatusEvent] = []
    handler3_calls: List[StatusEvent] = []

    def handler1(event: StatusEvent) -> None:
        handler1_calls.append(event)

    def handler2(event: StatusEvent) -> None:
        handler2_calls.append(event)

    def handler3(event: StatusEvent) -> None:
        handler3_calls.append(event)

    # Subscribe 3 handlers to same event type
    bus.subscribe(StatusEvent, handler1)
    bus.subscribe(StatusEvent, handler2)
    bus.subscribe(StatusEvent, handler3)

    # Publish single event
    status = StatusEvent(
        task_id=1,
        state=URLState.PROCESSING,
        message="Processing chunk 3/10",
        progress_pct=30.0,
        extras={"chunk": 3, "total_chunks": 10},
    )
    bus.publish(status)

    # Dispatch
    count = bus.dispatch(timeout=1.0)

    # Verify all handlers called
    assert count == 1, "Should have dispatched 1 event"
    assert len(handler1_calls) == 1, "Handler1 should be called once"
    assert len(handler2_calls) == 1, "Handler2 should be called once"
    assert len(handler3_calls) == 1, "Handler3 should be called once"
    assert handler1_calls[0] is status
    assert handler2_calls[0] is status
    assert handler3_calls[0] is status


def test_type_safe_subscriptions():
    """Test type-safe subscriptions - handler for ErrorEvent should NOT be called for StatusEvent."""
    bus = EventBus()
    error_handler_calls: List[ErrorEvent] = []
    status_handler_calls: List[StatusEvent] = []

    def error_handler(event: ErrorEvent) -> None:
        error_handler_calls.append(event)

    def status_handler(event: StatusEvent) -> None:
        status_handler_calls.append(event)

    # Subscribe each handler to specific event type
    bus.subscribe(ErrorEvent, error_handler)
    bus.subscribe(StatusEvent, status_handler)

    # Publish StatusEvent only (using kw_only arguments)
    status = StatusEvent(
        task_id=5,
        state=URLState.COMPLETE,
        message="Processing complete",
        progress_pct=100.0,
    )
    bus.publish(status)

    # Dispatch
    count = bus.dispatch(timeout=1.0)

    # Verify only status_handler called, error_handler NOT called
    assert count == 1, "Should have dispatched 1 event"
    assert len(status_handler_calls) == 1, "StatusEvent handler should be called"
    assert len(error_handler_calls) == 0, (
        "ErrorEvent handler should NOT be called for StatusEvent"
    )
    assert status_handler_calls[0] is status


def test_event_queuing():
    """Test event queuing - publish 10 events, verify all processed by dispatch()."""
    bus = EventBus()
    received_events: List[Event] = []

    def handler(event: Event) -> None:
        received_events.append(event)

    # Subscribe to multiple event types
    bus.subscribe(StatusEvent, handler)
    bus.subscribe(ErrorEvent, handler)
    bus.subscribe(CircuitBreakerEvent, handler)

    # Publish 10 events of different types
    events_published = []
    for i in range(10):
        if i % 3 == 0:
            event = StatusEvent(
                task_id=i, state=URLState.PROCESSING, message=f"Status {i}"
            )
        elif i % 3 == 1:
            event = ErrorEvent(category=ErrorCategory.API_TIMEOUT, message=f"Error {i}")
        else:
            event = CircuitBreakerEvent(
                reason=f"Circuit breaker {i}", affected_tasks=[i]
            )

        bus.publish(event)
        events_published.append(event)

    # Verify queue has 10 events
    stats_before = bus.get_stats()
    assert stats_before["published"] == 10, "Should have published 10 events"
    assert stats_before["pending"] == 10, "Should have 10 pending events in queue"
    assert stats_before["dispatched"] == 0, "No events dispatched yet"

    # Dispatch with long timeout to process all
    count = bus.dispatch(timeout=5.0)

    # Verify all events processed
    assert count == 10, "Should have dispatched all 10 events"
    assert len(received_events) == 10, "All 10 events should be received"
    assert received_events == events_published, (
        "Events should be received in FIFO order"
    )

    stats_after = bus.get_stats()
    assert stats_after["dispatched"] == 10, "Should have dispatched 10 events"
    assert stats_after["pending"] == 0, "Queue should be empty"


def test_get_stats_returns_correct_metrics():
    """Test get_stats() returns correct published, dispatched, pending counts."""
    bus = EventBus()
    received_events: List[ErrorEvent] = []

    def handler(event: ErrorEvent) -> None:
        received_events.append(event)

    bus.subscribe(ErrorEvent, handler)

    # Initial state
    stats = bus.get_stats()
    assert stats["published"] == 0, "Initially no events published"
    assert stats["dispatched"] == 0, "Initially no events dispatched"
    assert stats["pending"] == 0, "Initially no events pending"
    assert stats["dropped"] == 0, "Initially no events dropped"

    # Publish 5 events
    for i in range(5):
        bus.publish(
            ErrorEvent(category=ErrorCategory.PARSE_ERROR, message=f"Parse error {i}")
        )

    stats = bus.get_stats()
    assert stats["published"] == 5, "Should show 5 events published"
    assert stats["dispatched"] == 0, "No events dispatched yet"
    assert stats["pending"] == 5, "Should have 5 events pending"

    # Dispatch 3 events (with short timeout)
    bus.dispatch(timeout=0.001)  # Process first few quickly

    stats = bus.get_stats()
    assert stats["published"] == 5, "Published count unchanged"
    # Dispatched should be > 0 and <= 5 (depends on timeout)
    dispatched_count = stats["dispatched"]
    pending_count = stats["pending"]
    assert dispatched_count + pending_count == 5, "Total should equal published"
    assert dispatched_count > 0, "Should have dispatched at least 1 event"

    # Dispatch remaining
    bus.dispatch(timeout=5.0)

    stats = bus.get_stats()
    assert stats["published"] == 5, "Published count unchanged"
    assert stats["dispatched"] == 5, "All events dispatched"
    assert stats["pending"] == 0, "Queue should be empty"


# ============================================================================
# EventBus Thread Safety (4 tests)
# ============================================================================


def test_concurrent_publishing_from_worker_threads():
    """Test concurrent publishing from 10 worker threads (each publishes 100 events, verify all 1000 processed)."""
    bus = EventBus()
    received_events: List[ErrorEvent] = []
    lock = threading.Lock()

    def handler(event: ErrorEvent) -> None:
        with lock:
            received_events.append(event)

    bus.subscribe(ErrorEvent, handler)

    # Worker thread function
    def worker(worker_id: int) -> None:
        for i in range(100):
            event = ErrorEvent(
                category=ErrorCategory.CONNECTION_ERROR,
                message=f"Worker {worker_id} event {i}",
                task_id=worker_id * 100 + i,
                recoverable=True,
            )
            bus.publish(event)

    # Start 10 worker threads
    threads = []
    for worker_id in range(10):
        thread = threading.Thread(target=worker, args=(worker_id,))
        thread.start()
        threads.append(thread)

    # Wait for all workers to finish publishing
    for thread in threads:
        thread.join()

    # Verify all events published
    stats = bus.get_stats()
    assert stats["published"] == 1000, (
        "Should have published 1000 events from 10 workers"
    )
    assert stats["pending"] == 1000, "All events should be pending"

    # Dispatch all events
    count = bus.dispatch(timeout=10.0)

    # Verify all events processed
    assert count == 1000, "Should have dispatched all 1000 events"
    assert len(received_events) == 1000, "All 1000 events should be received"

    stats = bus.get_stats()
    assert stats["dispatched"] == 1000, "Should have dispatched 1000 events"
    assert stats["pending"] == 0, "Queue should be empty"


def test_queue_full_handling():
    """Test queue.Full handling - create EventBus with max_queue_size=10, publish 20 events, verify dropped_count."""
    bus = EventBus(max_queue_size=10)

    # Publish events until queue full
    successful_publishes = 0
    dropped_count = 0

    for i in range(20):
        try:
            event = ErrorEvent(category=ErrorCategory.UNKNOWN, message=f"Event {i}")
            bus.publish(event)
            successful_publishes += 1
        except queue_module.Full:
            dropped_count += 1

    # Verify queue behavior
    stats = bus.get_stats()
    assert successful_publishes == 10, (
        "Should successfully publish exactly 10 events (queue size)"
    )
    assert dropped_count == 10, "Should have 10 failed publish attempts (queue full)"
    assert stats["published"] == 10, "Published count should be 10"
    assert stats["dropped"] == 10, "Dropped count should be 10"
    assert stats["pending"] == 10, "Queue should be full with 10 events"


def test_handler_exception_isolation():
    """Test handler exception isolation - handler1 raises exception, handler2 should still be called."""
    bus = EventBus()
    handler1_calls: List[StatusEvent] = []
    handler2_calls: List[StatusEvent] = []

    def failing_handler(event: StatusEvent) -> None:
        handler1_calls.append(event)
        raise RuntimeError("Handler1 intentionally failed!")

    def normal_handler(event: StatusEvent) -> None:
        handler2_calls.append(event)

    # Subscribe both handlers (failing handler first)
    bus.subscribe(StatusEvent, failing_handler)
    bus.subscribe(StatusEvent, normal_handler)

    # Publish event
    status = StatusEvent(
        task_id=99, state=URLState.FAILED, message="Test failure isolation"
    )
    bus.publish(status)

    # Dispatch (should not raise exception even though handler1 fails)
    count = bus.dispatch(timeout=1.0)

    # Verify both handlers called despite handler1 exception
    assert count == 1, "Event should be dispatched successfully"
    assert len(handler1_calls) == 1, "Failing handler should be called"
    assert len(handler2_calls) == 1, (
        "Normal handler should still be called despite handler1 exception"
    )
    assert handler1_calls[0] is status
    assert handler2_calls[0] is status


def test_clear_removes_all_pending_events():
    """Test clear() removes all pending events from queue."""
    bus = EventBus()

    # Publish 50 events without dispatching
    for i in range(50):
        event = ErrorEvent(category=ErrorCategory.PARSE_ERROR, message=f"Event {i}")
        bus.publish(event)

    # Verify 50 pending
    stats_before = bus.get_stats()
    assert stats_before["pending"] == 50, "Should have 50 pending events"
    assert stats_before["dispatched"] == 0, "No events dispatched yet"

    # Clear queue
    cleared = bus.clear()

    # Verify all cleared
    assert cleared == 50, "clear() should return 50 (events removed)"

    stats_after = bus.get_stats()
    assert stats_after["pending"] == 0, "Queue should be empty after clear()"
    assert stats_after["dispatched"] == 0, "Dispatched count unchanged by clear()"
    assert stats_after["published"] == 50, "Published count unchanged by clear()"


# ============================================================================
# EventBus dispatch() Behavior (3 tests)
# ============================================================================


def test_dispatch_timeout():
    """Test dispatch() timeout - queue 1000 events, call dispatch(timeout=0.01), verify partial processing."""
    bus = EventBus()
    received_events: List[ErrorEvent] = []

    def slow_handler(event: ErrorEvent) -> None:
        """Handler that takes a bit of time to process."""
        time.sleep(0.001)  # 1ms per event
        received_events.append(event)

    bus.subscribe(ErrorEvent, handler=slow_handler)

    # Publish 1000 events
    for i in range(1000):
        bus.publish(ErrorEvent(category=ErrorCategory.UNKNOWN, message=f"Event {i}"))

    stats_before = bus.get_stats()
    assert stats_before["pending"] == 1000, "Should have 1000 pending events"

    # Dispatch with short timeout (0.01s = 10ms)
    # With 1ms per event, we should process roughly 10 events (but allow variance)
    count = bus.dispatch(timeout=0.01)

    # Verify partial processing (not all 1000 events)
    assert 0 < count < 1000, f"Should have processed partial events, got {count}"
    assert len(received_events) == count, "Received events should match dispatch count"

    stats_after = bus.get_stats()
    assert stats_after["dispatched"] == count, (
        "Dispatched count should match events processed"
    )
    assert stats_after["pending"] == 1000 - count, (
        "Pending should be reduced by dispatched count"
    )


def test_dispatch_processes_until_queue_empty():
    """Test dispatch() processes until queue empty - queue 50 events, call dispatch(timeout=10), verify all 50 processed."""
    bus = EventBus()
    received_events: List[CircuitBreakerEvent] = []

    def handler(event: CircuitBreakerEvent) -> None:
        received_events.append(event)

    bus.subscribe(CircuitBreakerEvent, handler)

    # Publish 50 events
    for i in range(50):
        event = CircuitBreakerEvent(
            reason=f"Circuit breaker test {i}",
            affected_tasks=[i],
            trigger_category=ErrorCategory.RATE_LIMIT,
            consecutive_counts={ErrorCategory.RATE_LIMIT: i},
        )
        bus.publish(event)

    # Dispatch with long timeout (10s - more than enough to process 50 events)
    count = bus.dispatch(timeout=10.0)

    # Verify all 50 processed
    assert count == 50, "Should have dispatched all 50 events"
    assert len(received_events) == 50, "All 50 events should be received"

    stats = bus.get_stats()
    assert stats["dispatched"] == 50, "Should have dispatched 50 events"
    assert stats["pending"] == 0, (
        "Queue should be empty (dispatch stopped early when queue empty)"
    )


def test_dispatch_returns_event_count():
    """Test dispatch() returns correct event count."""
    bus = EventBus()
    processed_events: List[Event] = []

    def handler(event: Event) -> None:
        processed_events.append(event)

    bus.subscribe(StatusEvent, handler)
    bus.subscribe(ErrorEvent, handler)

    # Publish mixed events
    for i in range(15):
        if i % 2 == 0:
            bus.publish(
                StatusEvent(task_id=i, state=URLState.PROCESSING, message=f"Status {i}")
            )
        else:
            bus.publish(
                ErrorEvent(category=ErrorCategory.UNKNOWN, message=f"Error {i}")
            )

    # Dispatch and verify count
    count = bus.dispatch(timeout=5.0)

    assert count == 15, "dispatch() should return 15 (events processed)"
    assert len(processed_events) == 15, "Should have received all 15 events"

    # Second dispatch with empty queue
    count2 = bus.dispatch(timeout=1.0)
    assert count2 == 0, "dispatch() should return 0 when queue empty"


# ============================================================================
# Event Classes (3 tests)
# ============================================================================


def test_status_event_creation_with_all_fields():
    """Test StatusEvent creation with all fields."""
    event = StatusEvent(
        task_id=42,
        state=URLState.PROCESSING,
        message="Processing chunk 7/10",
        progress_pct=70.0,
        extras={
            "chunk": 7,
            "total_chunks": 10,
            "size_in": 1024,
            "size_out": 512,
            "cost": 0.05,
        },
    )

    # Verify all fields
    assert event.task_id == 42
    assert event.state == URLState.PROCESSING
    assert event.message == "Processing chunk 7/10"
    assert event.progress_pct == 70.0
    assert event.extras["chunk"] == 7
    assert event.extras["total_chunks"] == 10
    assert event.extras["size_in"] == 1024

    # Verify Event base class fields auto-populated
    assert event.event_id is not None, "event_id should be auto-generated"
    assert len(event.event_id) == 36, "event_id should be UUID string"
    assert event.timestamp is not None, "timestamp should be auto-generated"


def test_error_event_creation_with_all_fields_and_defaults():
    """Test ErrorEvent creation with all fields and defaults."""
    # Test with all fields
    event_full = ErrorEvent(
        category=ErrorCategory.QUOTA_EXCEEDED,
        message="API quota exceeded. Please upgrade plan.",
        task_id=123,
        url="https://example.com/article",
        exception_type="QuotaExceededError",
        exception_traceback="Traceback (most recent call last):\n  File ...",
        context={"retry_count": 3, "quota_limit": 1000, "quota_used": 1000},
        recoverable=False,
    )

    assert event_full.category == ErrorCategory.QUOTA_EXCEEDED
    assert event_full.message == "API quota exceeded. Please upgrade plan."
    assert event_full.task_id == 123
    assert event_full.url == "https://example.com/article"
    assert event_full.exception_type == "QuotaExceededError"
    assert event_full.exception_traceback.startswith("Traceback")
    assert event_full.context["retry_count"] == 3
    assert event_full.recoverable is False

    # Test with defaults (optional fields)
    event_minimal = ErrorEvent(
        category=ErrorCategory.CONNECTION_ERROR,
        message="Network connection failed",
    )

    assert event_minimal.category == ErrorCategory.CONNECTION_ERROR
    assert event_minimal.message == "Network connection failed"
    assert event_minimal.task_id is None, "task_id should default to None"
    assert event_minimal.url is None, "url should default to None"
    assert event_minimal.exception_type is None, "exception_type should default to None"
    assert event_minimal.exception_traceback is None, (
        "exception_traceback should default to None"
    )
    assert event_minimal.context == {}, "context should default to empty dict"
    assert event_minimal.recoverable is True, "recoverable should default to True"


def test_circuit_breaker_and_dialog_event_creation():
    """Test CircuitBreakerEvent and DialogEvent creation."""
    # CircuitBreakerEvent
    circuit_event = CircuitBreakerEvent(
        reason="3 consecutive RATE_LIMIT errors",
        affected_tasks=[1, 2, 3, 5, 8, 13],
        trigger_category=ErrorCategory.RATE_LIMIT,
        consecutive_counts={
            ErrorCategory.RATE_LIMIT: 3,
            ErrorCategory.API_TIMEOUT: 1,
        },
    )

    assert circuit_event.reason == "3 consecutive RATE_LIMIT errors"
    assert circuit_event.affected_tasks == [1, 2, 3, 5, 8, 13]
    assert circuit_event.trigger_category == ErrorCategory.RATE_LIMIT
    assert circuit_event.consecutive_counts[ErrorCategory.RATE_LIMIT] == 3
    assert circuit_event.event_id is not None

    # CircuitBreakerEvent with defaults
    circuit_minimal = CircuitBreakerEvent(reason="System overload")
    assert circuit_minimal.reason == "System overload"
    assert circuit_minimal.affected_tasks == [], (
        "affected_tasks should default to empty list"
    )
    assert circuit_minimal.trigger_category is None, (
        "trigger_category should default to None"
    )
    assert circuit_minimal.consecutive_counts == {}, (
        "consecutive_counts should default to empty dict"
    )

    # DialogEvent
    dialog_event = DialogEvent(
        dialog_type=DialogType.CIRCUIT_BREAKER,
        priority=DialogPriority.CRITICAL,
        context={
            "error_category": "RATE_LIMIT",
            "consecutive_count": 3,
            "affected_tasks": 15,
            "suggestion": "Wait 60 seconds before retrying",
        },
    )

    assert dialog_event.dialog_type == DialogType.CIRCUIT_BREAKER
    assert dialog_event.priority == DialogPriority.CRITICAL
    assert dialog_event.context["error_category"] == "RATE_LIMIT"
    assert dialog_event.context["consecutive_count"] == 3

    # DialogEvent with defaults
    dialog_minimal = DialogEvent(
        dialog_type=DialogType.INFO,
        priority=DialogPriority.LOW,
    )
    assert dialog_minimal.dialog_type == DialogType.INFO
    assert dialog_minimal.priority == DialogPriority.LOW
    assert dialog_minimal.context == {}, "context should default to empty dict"
    assert dialog_minimal.event_id is not None
