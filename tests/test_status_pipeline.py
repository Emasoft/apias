"""
Comprehensive unit tests for apias/status_pipeline.py

Tests cover StatusPipeline initialization, status updates, error events, critical events,
atomic snapshots, thread safety, and utility methods with realistic, non-mocked scenarios.

All tests use real EventBus, real URLTask objects, and real threading.
No mocks for internal logic - only real functional tests.

Test Coverage:
- StatusPipeline Initialization (2 tests)
- Status Updates (5 tests)
- Error Events (2 tests)
- Critical Events (3 tests)
- Atomic Snapshots (3 tests)
- Thread Safety (2 tests)
- Utility Methods (1 test)

Total: 18 tests
"""

import threading
import time
from datetime import datetime
from typing import Dict, List

import pytest

from apias.batch_tui import URLState, URLTask
from apias.event_system import (
    CircuitBreakerEvent,
    ErrorCategory,
    ErrorEvent,
    EventBus,
    StatusEvent,
)
from apias.status_pipeline import StatusPipeline, TaskSnapshot

# ============================================================================
# StatusPipeline Initialization (2 tests)
# ============================================================================


def test_status_pipeline_initialization():
    """Test StatusPipeline initialization - verify event bus subscription and empty task dict."""
    bus = EventBus()
    pipeline = StatusPipeline(bus)

    # Verify event bus is stored
    assert pipeline._event_bus is bus, "Should store EventBus reference"

    # Verify task dictionary is empty
    assert isinstance(pipeline._tasks, dict), "Should have empty tasks dict"
    assert len(pipeline._tasks) == 0, "Tasks dict should be empty on init"

    # Verify lock exists
    assert hasattr(pipeline, "_task_lock"), "Should have task lock"
    assert pipeline._task_lock is not None, "Task lock should not be None"

    # Verify critical event flag exists and is not set
    assert isinstance(pipeline._critical_event, threading.Event), (
        "Should have critical event flag"
    )
    assert not pipeline._critical_event.is_set(), (
        "Critical event flag should start unset"
    )


def test_initialize_tasks_creates_url_task_objects():
    """Test initialize_tasks() creates URLTask objects with correct task_id, url, and PENDING state."""
    bus = EventBus()
    pipeline = StatusPipeline(bus)

    # Initialize with realistic URLs
    urls = [
        "https://example.com/page1",
        "https://example.com/page2",
        "https://example.com/page3",
        "https://example.com/page4",
        "https://example.com/page5",
    ]

    pipeline.initialize_tasks(urls)

    # Verify all tasks created
    assert len(pipeline._tasks) == 5, "Should create 5 tasks"

    # Verify each task has correct properties
    for task_id, expected_url in enumerate(urls, start=1):
        task = pipeline._tasks[task_id]
        assert isinstance(task, URLTask), f"Task {task_id} should be URLTask instance"
        assert task.task_id == task_id, f"Task {task_id} should have correct task_id"
        assert task.url == expected_url, f"Task {task_id} should have correct URL"
        assert task.state == URLState.PENDING, (
            f"Task {task_id} should start in PENDING state"
        )
        assert task.progress_pct == 0.0, f"Task {task_id} should start with 0% progress"


# ============================================================================
# Status Updates (5 tests)
# ============================================================================


def test_update_status_publishes_status_event():
    """Test update_status() publishes StatusEvent - verify event appears in queue."""
    bus = EventBus()
    pipeline = StatusPipeline(bus)

    # Initialize a task
    pipeline.initialize_tasks(["https://example.com/test"])

    # Update status
    pipeline.update_status(
        task_id=1,
        state=URLState.PROCESSING,
        message="Processing chunk 2/5",
        progress_pct=40.0,
        size_in=12000,
        current_chunk=2,
        total_chunks=5,
    )

    # Verify event was published to queue (without dispatching)
    assert not bus._event_queue.empty(), "Event queue should not be empty"
    assert bus._event_queue.qsize() == 1, "Should have exactly 1 event in queue"


def test_on_status_event_updates_task_state():
    """Test _on_status_event() updates task state, message, and progress."""
    bus = EventBus()
    pipeline = StatusPipeline(bus)

    # Initialize task
    pipeline.initialize_tasks(["https://example.com/test"])

    # Update status (publishes event)
    pipeline.update_status(
        task_id=1,
        state=URLState.PROCESSING,
        message="Processing chunk 2/5",
        progress_pct=40.0,
    )

    # Dispatch event to trigger handler
    dispatched = bus.dispatch(timeout=1.0)
    assert dispatched == 1, "Should have dispatched 1 event"

    # Verify task was updated
    task = pipeline._tasks[1]
    assert task.state == URLState.PROCESSING, "Task state should be updated"
    assert task.status_message == "Processing chunk 2/5", (
        "Status message should be updated"
    )
    assert task.progress_pct == 40.0, "Progress should be updated"


def test_on_status_event_updates_optional_fields():
    """Test _on_status_event() updates optional fields (size_in, size_out, cost, chunks)."""
    bus = EventBus()
    pipeline = StatusPipeline(bus)

    # Initialize task
    pipeline.initialize_tasks(["https://example.com/test"])

    # Update with all optional fields
    pipeline.update_status(
        task_id=1,
        state=URLState.PROCESSING,
        message="Processing...",
        progress_pct=50.0,
        size_in=15000,
        size_out=8000,
        cost=0.025,
        duration=3.5,
        current_chunk=3,
        total_chunks=6,
    )

    # Dispatch event
    bus.dispatch(timeout=1.0)

    # Verify all optional fields updated
    task = pipeline._tasks[1]
    assert task.size_in == 15000, "size_in should be updated"
    assert task.size_out == 8000, "size_out should be updated"
    assert task.cost == 0.025, "cost should be updated"
    assert task.duration == 3.5, "duration should be updated"
    assert task.current_chunk == 3, "current_chunk should be updated"
    assert task.total_chunks == 6, "total_chunks should be updated"


def test_on_status_event_maintains_status_history():
    """Test _on_status_event() maintains status history - last 5 messages, FIFO."""
    bus = EventBus()
    pipeline = StatusPipeline(bus)

    # Initialize task
    pipeline.initialize_tasks(["https://example.com/test"])

    # Send 7 status updates (should keep only last 5)
    messages = [
        "Message 1",
        "Message 2",
        "Message 3",
        "Message 4",
        "Message 5",
        "Message 6",
        "Message 7",
    ]

    for i, msg in enumerate(messages):
        pipeline.update_status(
            task_id=1,
            state=URLState.PROCESSING,
            message=msg,
            progress_pct=float(i * 10),
        )
        bus.dispatch(timeout=1.0)
        time.sleep(0.01)  # Ensure different timestamps

    # Verify history has exactly 5 items (FIFO - oldest removed)
    task = pipeline._tasks[1]
    assert hasattr(task, "status_history"), "Task should have status_history attribute"
    assert len(task.status_history) == 5, "Should keep exactly 5 messages"

    # Verify it's the last 5 messages (first 2 should be removed)
    history_messages = [msg for timestamp, msg in task.status_history]
    assert history_messages == messages[-5:], (
        "Should keep last 5 messages in FIFO order"
    )

    # Verify timestamps are present and increasing
    timestamps = [ts for ts, msg in task.status_history]
    assert all(isinstance(ts, datetime) for ts in timestamps), (
        "All timestamps should be datetime objects"
    )
    assert timestamps == sorted(timestamps), (
        "Timestamps should be in chronological order"
    )


def test_on_status_event_ignores_unknown_task_id():
    """Test _on_status_event() ignores unknown task_id without crashing."""
    bus = EventBus()
    pipeline = StatusPipeline(bus)

    # Initialize only task 1
    pipeline.initialize_tasks(["https://example.com/test"])

    # Update non-existent task 999
    pipeline.update_status(
        task_id=999,
        state=URLState.PROCESSING,
        message="This should be ignored",
        progress_pct=50.0,
    )

    # Dispatch - should not crash
    dispatched = bus.dispatch(timeout=1.0)
    assert dispatched == 1, "Should have dispatched 1 event (even though ignored)"

    # Verify task 1 was not affected
    task = pipeline._tasks[1]
    assert task.state == URLState.PENDING, "Task 1 should remain PENDING"
    assert task.status_message == "", "Task 1 should have no status message"

    # Verify no task 999 was created
    assert 999 not in pipeline._tasks, "Should not create task for unknown ID"


# ============================================================================
# Error Events (2 tests)
# ============================================================================


def test_on_error_event_updates_task_error_field():
    """Test _on_error_event() updates task error field."""
    bus = EventBus()
    pipeline = StatusPipeline(bus)

    # Initialize task
    pipeline.initialize_tasks(["https://example.com/test"])

    # Publish error event
    error = ErrorEvent(
        category=ErrorCategory.API_TIMEOUT,
        message="Request timeout after 30s",
        task_id=1,
        url="https://example.com/test",
        exception_type="TimeoutError",
        recoverable=True,
    )
    bus.publish(error)

    # Dispatch event
    bus.dispatch(timeout=1.0)

    # Verify error field was set
    task = pipeline._tasks[1]
    assert task.error == "Request timeout after 30s", "Error field should be updated"


def test_on_error_event_adds_to_status_history():
    """Test _on_error_event() adds error to status history."""
    bus = EventBus()
    pipeline = StatusPipeline(bus)

    # Initialize task
    pipeline.initialize_tasks(["https://example.com/test"])

    # Publish error event
    error = ErrorEvent(
        category=ErrorCategory.SERVER_ERROR,
        message="LLM API returned 500 error",
        task_id=1,
        url="https://example.com/test",
        exception_type="APIError",
        recoverable=True,
    )
    bus.publish(error)

    # Dispatch event
    bus.dispatch(timeout=1.0)

    # Verify error was added to history
    task = pipeline._tasks[1]
    assert hasattr(task, "status_history"), "Task should have status_history"
    assert len(task.status_history) == 1, "Should have 1 history entry"

    timestamp, message = task.status_history[0]
    assert isinstance(timestamp, datetime), "Timestamp should be datetime"
    assert "❌" in message, "Error message should have error emoji"
    assert "LLM API returned 500 error" in message, "Should contain error text"


# ============================================================================
# Critical Events (3 tests)
# ============================================================================


def test_on_circuit_breaker_sets_critical_event_flag():
    """Test _on_circuit_breaker() sets critical event flag for instant wake-up."""
    bus = EventBus()
    pipeline = StatusPipeline(bus)

    # Verify flag starts unset
    assert not pipeline._critical_event.is_set(), "Flag should start unset"

    # Publish circuit breaker event
    cb_event = CircuitBreakerEvent(
        reason="Too many API errors (5 in 60s)",
        affected_tasks=[],
        trigger_category=ErrorCategory.API_TIMEOUT,
    )
    bus.publish(cb_event)

    # Dispatch event
    bus.dispatch(timeout=1.0)

    # Verify critical flag is now set
    assert pipeline._critical_event.is_set(), (
        "Critical event flag should be set after circuit breaker"
    )


def test_wait_for_update_returns_false_on_timeout():
    """Test wait_for_update() returns False on timeout (no critical event)."""
    bus = EventBus()
    pipeline = StatusPipeline(bus)

    # Wait with short timeout (no critical event)
    start = time.time()
    critical = pipeline.wait_for_update(timeout=0.1)
    elapsed = time.time() - start

    # Verify it waited and returned False
    assert not critical, "Should return False when no critical event"
    assert elapsed >= 0.1, "Should have waited at least 100ms"
    assert elapsed < 0.2, "Should not have waited much longer than timeout"


def test_wait_for_update_returns_true_on_critical_event():
    """Test wait_for_update() returns True on critical event - instant wake-up."""
    bus = EventBus()
    pipeline = StatusPipeline(bus)

    # Trigger critical event in background thread after 50ms
    def trigger_circuit_breaker():
        time.sleep(0.05)  # Wait 50ms
        cb_event = CircuitBreakerEvent(
            reason="Circuit breaker test",
            affected_tasks=[],
            trigger_category=ErrorCategory.API_TIMEOUT,
        )
        bus.publish(cb_event)
        bus.dispatch(timeout=1.0)

    # Start background thread
    t = threading.Thread(target=trigger_circuit_breaker, daemon=True)
    t.start()

    # Wait for update with long timeout (should wake up early)
    start = time.time()
    critical = pipeline.wait_for_update(timeout=1.0)  # 1 second timeout
    elapsed = time.time() - start

    # Verify it woke up early on critical event
    assert critical, "Should return True when critical event occurs"
    assert elapsed < 0.2, (
        f"Should wake up quickly (got {elapsed:.3f}s), not wait full timeout"
    )

    # Clean up
    t.join(timeout=1.0)


# ============================================================================
# Atomic Snapshots (3 tests)
# ============================================================================


def test_get_snapshot_returns_task_snapshot_dict():
    """Test get_snapshot() returns dict of TaskSnapshot objects, not URLTask."""
    bus = EventBus()
    pipeline = StatusPipeline(bus)

    # Initialize tasks
    urls = [
        "https://example.com/page1",
        "https://example.com/page2",
        "https://example.com/page3",
    ]
    pipeline.initialize_tasks(urls)

    # Update one task
    pipeline.update_status(
        task_id=2,
        state=URLState.PROCESSING,
        message="Processing...",
        progress_pct=50.0,
    )
    bus.dispatch(timeout=1.0)

    # Get snapshot
    snapshot = pipeline.get_snapshot()

    # Verify snapshot structure
    assert isinstance(snapshot, dict), "Snapshot should be a dict"
    assert len(snapshot) == 3, "Should have 3 task snapshots"

    # Verify all entries are TaskSnapshot, not URLTask
    for task_id, task_snap in snapshot.items():
        assert isinstance(task_snap, TaskSnapshot), (
            f"Task {task_id} should be TaskSnapshot"
        )
        assert not isinstance(task_snap, URLTask), (
            f"Task {task_id} should NOT be URLTask"
        )
        assert task_snap.task_id == task_id, "task_id should match dict key"
        assert task_snap.url == urls[task_id - 1], "URL should match"


def test_get_snapshot_returns_deep_copy():
    """Test get_snapshot() returns deep copy - modifying snapshot doesn't affect original."""
    bus = EventBus()
    pipeline = StatusPipeline(bus)

    # Initialize task
    pipeline.initialize_tasks(["https://example.com/test"])

    # Update task with history
    for i in range(3):
        pipeline.update_status(
            task_id=1,
            state=URLState.PROCESSING,
            message=f"Message {i + 1}",
            progress_pct=float(i * 30),
        )
        bus.dispatch(timeout=1.0)
        time.sleep(0.01)

    # Get snapshot
    snapshot = pipeline.get_snapshot()

    # Modify snapshot
    snapshot[1].progress_pct = 999.0  # This should NOT affect original

    # Verify original task unchanged
    original_task = pipeline._tasks[1]
    assert original_task.progress_pct != 999.0, (
        "Original task should not be affected by snapshot modification"
    )

    # Get new snapshot and verify it still has original value
    new_snapshot = pipeline.get_snapshot()
    assert new_snapshot[1].progress_pct != 999.0, (
        "New snapshot should have original value, not modified value"
    )


def test_get_snapshot_includes_status_history():
    """Test get_snapshot() includes deep copy of status history."""
    bus = EventBus()
    pipeline = StatusPipeline(bus)

    # Initialize task
    pipeline.initialize_tasks(["https://example.com/test"])

    # Add status updates
    messages = ["Starting", "Processing chunk 1", "Processing chunk 2", "Almost done"]
    for msg in messages:
        pipeline.update_status(
            task_id=1,
            state=URLState.PROCESSING,
            message=msg,
            progress_pct=25.0,
        )
        bus.dispatch(timeout=1.0)
        time.sleep(0.01)

    # Get snapshot
    snapshot = pipeline.get_snapshot()

    # Verify history is included
    assert hasattr(snapshot[1], "status_history"), (
        "Snapshot should include status_history"
    )
    assert len(snapshot[1].status_history) == 4, "Should have 4 history entries"

    # Verify history has correct structure
    for timestamp, message in snapshot[1].status_history:
        assert isinstance(timestamp, datetime), "Timestamp should be datetime"
        assert isinstance(message, str), "Message should be string"
        assert message in messages, "Message should be one we sent"

    # Verify it's a deep copy (modifying snapshot history doesn't affect original)
    snapshot[1].status_history.clear()
    new_snapshot = pipeline.get_snapshot()
    assert len(new_snapshot[1].status_history) == 4, (
        "Original history should be unchanged"
    )


# ============================================================================
# Thread Safety (2 tests)
# ============================================================================


def test_concurrent_update_status_from_multiple_threads():
    """Test concurrent update_status() from 10 threads - each updates 10 tasks, verify all 100 updates."""
    bus = EventBus()
    pipeline = StatusPipeline(bus)

    # Initialize 10 tasks
    urls = [f"https://example.com/page{i}" for i in range(1, 11)]
    pipeline.initialize_tasks(urls)

    # Counters for verification
    total_updates = 0
    lock = threading.Lock()

    def worker(thread_id: int):
        """Each thread updates all 10 tasks once"""
        nonlocal total_updates
        for task_id in range(1, 11):
            pipeline.update_status(
                task_id=task_id,
                state=URLState.PROCESSING,
                message=f"Update from thread {thread_id}",
                progress_pct=float(thread_id * 10),
            )
            with lock:
                total_updates += 1

    # Start 10 worker threads
    threads = []
    for i in range(10):
        t = threading.Thread(target=worker, args=(i,), daemon=True)
        t.start()
        threads.append(t)

    # Wait for all threads to finish
    for t in threads:
        t.join(timeout=5.0)

    # Verify all 100 updates were made
    assert total_updates == 100, "Should have made 100 updates (10 threads × 10 tasks)"

    # Dispatch all events (100 events in queue)
    dispatched_count = 0
    for _ in range(100):
        count = bus.dispatch(timeout=0.1)
        dispatched_count += count
        if count == 0:
            break

    assert dispatched_count == 100, (
        f"Should have dispatched all 100 events (got {dispatched_count})"
    )

    # Verify all tasks were updated (at least once)
    for task_id in range(1, 11):
        task = pipeline._tasks[task_id]
        assert task.state == URLState.PROCESSING, (
            f"Task {task_id} should be in PROCESSING state"
        )
        assert "Update from thread" in task.status_message, (
            f"Task {task_id} should have update message"
        )


def test_get_snapshot_during_concurrent_updates():
    """Test get_snapshot() during concurrent updates - no race conditions."""
    bus = EventBus()
    pipeline = StatusPipeline(bus)

    # Initialize tasks
    urls = [f"https://example.com/page{i}" for i in range(1, 21)]
    pipeline.initialize_tasks(urls)

    snapshots_taken: List[Dict] = []
    snapshot_lock = threading.Lock()
    stop_flag = threading.Event()

    def updater():
        """Continuously update tasks"""
        counter = 0
        while not stop_flag.is_set():
            for task_id in range(1, 21):
                pipeline.update_status(
                    task_id=task_id,
                    state=URLState.PROCESSING,
                    message=f"Update {counter}",
                    progress_pct=float(counter % 100),
                )
                bus.dispatch(timeout=0.01)
            counter += 1
            time.sleep(0.001)

    def snapshot_taker():
        """Take snapshots while updates are happening"""
        for _ in range(50):  # Take 50 snapshots
            snapshot = pipeline.get_snapshot()
            with snapshot_lock:
                snapshots_taken.append(snapshot)
            time.sleep(0.01)

    # Start updater thread
    updater_thread = threading.Thread(target=updater, daemon=True)
    updater_thread.start()

    # Start snapshot taker thread
    snapshot_thread = threading.Thread(target=snapshot_taker, daemon=True)
    snapshot_thread.start()

    # Wait for snapshot taker to finish
    snapshot_thread.join(timeout=10.0)

    # Stop updater
    stop_flag.set()
    updater_thread.join(timeout=2.0)

    # Verify snapshots were taken successfully
    assert len(snapshots_taken) == 50, "Should have taken 50 snapshots"

    # Verify all snapshots are valid (no corruption from race conditions)
    for i, snapshot in enumerate(snapshots_taken):
        assert len(snapshot) == 20, f"Snapshot {i} should have 20 tasks"
        for task_id, task_snap in snapshot.items():
            assert isinstance(task_snap, TaskSnapshot), (
                f"Snapshot {i} task {task_id} should be TaskSnapshot"
            )
            assert 1 <= task_snap.task_id <= 20, (
                f"Snapshot {i} task {task_id} should have valid task_id"
            )
            assert 0.0 <= task_snap.progress_pct <= 100.0, (
                f"Snapshot {i} task {task_id} should have valid progress"
            )


# ============================================================================
# Utility Methods (1 test)
# ============================================================================


def test_get_stats_returns_correct_counts():
    """Test get_stats() returns correct counts (total, pending, active, complete, failed)."""
    bus = EventBus()
    pipeline = StatusPipeline(bus)

    # Initialize 10 tasks
    urls = [f"https://example.com/page{i}" for i in range(1, 11)]
    pipeline.initialize_tasks(urls)

    # Update tasks to different states:
    # 2 PENDING (task 1, 2)
    # 3 SCRAPING (task 3, 4, 5)
    # 2 PROCESSING (task 6, 7)
    # 2 COMPLETE (task 8, 9)
    # 1 FAILED (task 10)

    # Tasks 1-2: Keep as PENDING (no update)

    # Tasks 3-5: SCRAPING
    for task_id in [3, 4, 5]:
        pipeline.update_status(task_id=task_id, state=URLState.SCRAPING)
        bus.dispatch(timeout=1.0)

    # Tasks 6-7: PROCESSING
    for task_id in [6, 7]:
        pipeline.update_status(task_id=task_id, state=URLState.PROCESSING)
        bus.dispatch(timeout=1.0)

    # Tasks 8-9: COMPLETE
    for task_id in [8, 9]:
        pipeline.update_status(task_id=task_id, state=URLState.COMPLETE)
        bus.dispatch(timeout=1.0)

    # Task 10: FAILED
    pipeline.update_status(task_id=10, state=URLState.FAILED)
    bus.dispatch(timeout=1.0)

    # Get stats
    stats = pipeline.get_stats()

    # Verify counts
    assert stats["total"] == 10, "Should have 10 total tasks"
    assert stats["pending"] == 2, "Should have 2 pending tasks"
    assert stats["active"] == 5, (
        "Should have 5 active tasks (3 SCRAPING + 2 PROCESSING)"
    )
    assert stats["complete"] == 2, "Should have 2 complete tasks"
    assert stats["failed"] == 1, "Should have 1 failed task"

    # Verify sum
    assert (
        stats["pending"] + stats["active"] + stats["complete"] + stats["failed"]
        == stats["total"]
    ), "Stats should sum to total"
