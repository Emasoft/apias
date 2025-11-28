"""
Comprehensive unit tests for apias/dialog_manager.py

Tests cover DialogManager initialization, event subscriptions, dialog queueing,
priority ordering, dialog rendering, and integration with EventBus.

All tests use real EventBus and Rich Console instances (no mocks).
Console is used in record mode for testing rendered output.

Test Count: 15 tests total
- DialogManager Initialization: 2 tests
- Dialog Queueing: 5 tests
- Priority Ordering: 3 tests
- Dialog Rendering: 4 tests
- Integration with Events: 1 test
"""

import time
from datetime import datetime
from pathlib import Path
from typing import List
from unittest.mock import patch

import pytest
from rich.console import Console

from apias.dialog_manager import DialogManager
from apias.event_system import (
    CircuitBreakerEvent,
    DialogEvent,
    DialogPriority,
    DialogType,
    ErrorCategory,
    ErrorEvent,
    EventBus,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def event_bus():
    """EventBus instance for testing event subscriptions."""
    return EventBus()


@pytest.fixture
def console():
    """Rich console in record mode for testing output."""
    return Console(record=True, width=80)


@pytest.fixture
def dialog_manager(event_bus, console):
    """DialogManager with test console and event bus."""
    return DialogManager(event_bus, console)


@pytest.fixture
def temp_output_dir(tmp_path):
    """Temporary output directory for dialog file references."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def session_log(tmp_path):
    """Temporary session.log file for error investigation."""
    log_file = tmp_path / "session.log"
    log_file.write_text("Sample session log\n")
    return log_file


# ============================================================================
# DialogManager Initialization (2 tests)
# ============================================================================


def test_dialog_manager_initialization(dialog_manager):
    """Test DialogManager initialization - verify event subscriptions and empty queue."""
    # Verify queue is empty at initialization
    assert dialog_manager.get_pending_count() == 0, "Queue should be empty at initialization"

    # Verify internal state
    assert dialog_manager._dialog_counter == 0, "Counter should start at 0"
    assert dialog_manager._dialog_queue.empty(), "Priority queue should be empty"


def test_event_bus_subscription_to_events(event_bus, console):
    """Test event bus subscription to CircuitBreakerEvent and DialogEvent."""
    # Create DialogManager (subscriptions happen in __init__)
    dialog_manager = DialogManager(event_bus, console)

    # Publish CircuitBreakerEvent - should trigger _queue_circuit_breaker_dialog
    circuit_event = CircuitBreakerEvent(
        reason="Too many API_TIMEOUT errors (5 consecutive)",
        affected_tasks=[1, 2, 3],
        trigger_category=ErrorCategory.API_TIMEOUT,
        consecutive_counts={ErrorCategory.API_TIMEOUT: 5},
        timestamp=datetime.now(),
    )
    event_bus.publish(circuit_event)
    event_bus.dispatch(timeout=0.5)

    # Verify dialog was queued
    assert dialog_manager.get_pending_count() == 1, "CircuitBreakerEvent should trigger dialog queue"

    # Publish DialogEvent - should trigger _queue_dialog_event
    dialog_event = DialogEvent(
        priority=DialogPriority.NORMAL,
        dialog_type=DialogType.INFO,
        context={"message": "Processing complete", "title": "Success"},
    )
    event_bus.publish(dialog_event)
    event_bus.dispatch(timeout=0.5)

    # Verify second dialog was queued
    assert dialog_manager.get_pending_count() == 2, "DialogEvent should trigger dialog queue"


# ============================================================================
# Dialog Queueing (5 tests)
# ============================================================================


def test_queue_circuit_breaker_dialog_with_critical_priority(dialog_manager, event_bus):
    """Test _queue_circuit_breaker_dialog() queues CRITICAL dialog with correct context fields."""
    # Create CircuitBreakerEvent with full context
    circuit_event = CircuitBreakerEvent(
        reason="API quota exceeded (429 Too Many Requests)",
        affected_tasks=[10, 20, 30, 40],
        trigger_category=ErrorCategory.QUOTA_EXCEEDED,
        consecutive_counts={ErrorCategory.QUOTA_EXCEEDED: 3},
        timestamp=datetime(2025, 11, 28, 12, 30, 45),
    )

    # Publish event to trigger dialog queueing
    event_bus.publish(circuit_event)
    event_bus.dispatch(timeout=0.5)

    # Verify dialog was queued
    assert dialog_manager.get_pending_count() == 1, "CircuitBreakerEvent should queue 1 dialog"

    # Verify internal queue structure (priority, counter, dialog)
    priority_value, counter, dialog = dialog_manager._dialog_queue.get_nowait()

    assert priority_value == DialogPriority.CRITICAL.value, "Circuit breaker should have CRITICAL priority"
    assert counter == 0, "First dialog should have counter=0"
    assert dialog.priority == DialogPriority.CRITICAL
    assert dialog.dialog_type == DialogType.CIRCUIT_BREAKER

    # Verify context fields match event
    assert dialog.context["reason"] == "API quota exceeded (429 Too Many Requests)"
    assert dialog.context["affected_tasks"] == [10, 20, 30, 40]
    assert dialog.context["trigger_category"] == ErrorCategory.QUOTA_EXCEEDED
    assert dialog.context["consecutive_counts"] == {ErrorCategory.QUOTA_EXCEEDED: 3}
    assert dialog.context["timestamp"] == datetime(2025, 11, 28, 12, 30, 45)


def test_queue_dialog_event_with_generic_priority(dialog_manager, event_bus):
    """Test _queue_dialog_event() queues generic dialog with priority from event."""
    # Create DialogEvent with HIGH priority
    dialog_event = DialogEvent(
        priority=DialogPriority.HIGH,
        dialog_type=DialogType.INFO,
        context={"message": "Authentication required", "title": "Auth Failure"},
    )

    # Publish event
    event_bus.publish(dialog_event)
    event_bus.dispatch(timeout=0.5)

    # Verify dialog queued
    assert dialog_manager.get_pending_count() == 1

    # Verify queue structure
    priority_value, counter, dialog = dialog_manager._dialog_queue.get_nowait()

    assert priority_value == DialogPriority.HIGH.value, "Should use priority from event"
    assert dialog.priority == DialogPriority.HIGH
    assert dialog.dialog_type == DialogType.INFO
    assert dialog.context["message"] == "Authentication required"
    assert dialog.context["title"] == "Auth Failure"


def test_queue_error_summary_high_priority_with_errors(dialog_manager, temp_output_dir, session_log):
    """Test queue_error_summary() queues HIGH priority when total_errors > 0."""
    # Create error breakdown with real errors
    error_breakdown = {
        ErrorCategory.API_TIMEOUT: 5,
        ErrorCategory.CONNECTION_ERROR: 3,
        ErrorCategory.QUOTA_EXCEEDED: 1,
    }

    recent_errors = [
        ErrorEvent(
            category=ErrorCategory.API_TIMEOUT,
            message="Request timeout after 30s",
            task_id=1,
            url="https://example.com/api/v1",
            exception_type="TimeoutError",
            recoverable=True,
        ),
        ErrorEvent(
            category=ErrorCategory.CONNECTION_ERROR,
            message="Connection refused",
            task_id=2,
            url="https://example.com/api/v2",
            exception_type="ConnectionError",
            recoverable=True,
        ),
    ]

    # Queue error summary with errors
    dialog_manager.queue_error_summary(
        total_errors=9,
        error_breakdown=error_breakdown,
        recent_errors=recent_errors,
        output_dir=temp_output_dir,
        session_log=session_log,
    )

    # Verify HIGH priority for errors
    assert dialog_manager.get_pending_count() == 1

    priority_value, counter, dialog = dialog_manager._dialog_queue.get_nowait()

    assert priority_value == DialogPriority.HIGH.value, "Error summary with errors should be HIGH priority"
    assert dialog.dialog_type == DialogType.ERROR_SUMMARY
    assert dialog.context["total_errors"] == 9
    assert dialog.context["error_breakdown"] == error_breakdown
    assert len(dialog.context["recent_errors"]) == 2


def test_queue_error_summary_normal_priority_no_errors(dialog_manager, temp_output_dir, session_log):
    """Test queue_error_summary() queues NORMAL priority when total_errors = 0."""
    # Queue error summary with NO errors
    dialog_manager.queue_error_summary(
        total_errors=0,
        error_breakdown={},
        recent_errors=[],
        output_dir=temp_output_dir,
        session_log=session_log,
    )

    # Verify NORMAL priority for success case
    assert dialog_manager.get_pending_count() == 1

    priority_value, counter, dialog = dialog_manager._dialog_queue.get_nowait()

    assert priority_value == DialogPriority.NORMAL.value, "Error summary with no errors should be NORMAL priority"
    assert dialog.context["total_errors"] == 0
    assert dialog.context["error_breakdown"] == {}
    assert dialog.context["recent_errors"] == []


def test_get_pending_count_returns_correct_count(dialog_manager, event_bus):
    """Test get_pending_count() returns correct count after queueing multiple dialogs."""
    # Initially empty
    assert dialog_manager.get_pending_count() == 0

    # Queue 3 dialogs
    for i in range(3):
        dialog_event = DialogEvent(
            priority=DialogPriority.NORMAL,
            dialog_type=DialogType.INFO,
            context={"message": f"Dialog {i}"},
        )
        event_bus.publish(dialog_event)

    event_bus.dispatch(timeout=0.5)

    # Verify count
    assert dialog_manager.get_pending_count() == 3, "Should have 3 pending dialogs"

    # Remove one dialog
    dialog_manager._dialog_queue.get_nowait()

    # Verify count decremented
    assert dialog_manager.get_pending_count() == 2, "Should have 2 pending dialogs after removing one"


def test_clear_pending_dialogs_empties_queue(dialog_manager, event_bus):
    """Test clear_pending_dialogs() empties queue and returns count."""
    # Queue 5 dialogs
    for i in range(5):
        dialog_event = DialogEvent(
            priority=DialogPriority.NORMAL,
            dialog_type=DialogType.INFO,
            context={"message": f"Dialog {i}"},
        )
        event_bus.publish(dialog_event)

    event_bus.dispatch(timeout=0.5)

    # Verify queued
    assert dialog_manager.get_pending_count() == 5

    # Clear queue
    cleared_count = dialog_manager.clear_pending_dialogs()

    # Verify cleared
    assert cleared_count == 5, "Should return count of cleared dialogs"
    assert dialog_manager.get_pending_count() == 0, "Queue should be empty after clear"


# ============================================================================
# Priority Ordering (3 tests)
# ============================================================================


def test_priority_ordering_critical_high_normal_low(dialog_manager, event_bus):
    """Test priority ordering - queue NORMAL, CRITICAL, LOW, HIGH -> show in order: CRITICAL, HIGH, NORMAL, LOW."""
    # Queue dialogs in random priority order
    priorities = [
        DialogPriority.NORMAL,
        DialogPriority.CRITICAL,
        DialogPriority.LOW,
        DialogPriority.HIGH,
    ]

    for priority in priorities:
        dialog_event = DialogEvent(
            priority=priority,
            dialog_type=DialogType.INFO,
            context={"message": f"Priority {priority.name}"},
        )
        event_bus.publish(dialog_event)

    event_bus.dispatch(timeout=0.5)

    # Verify 4 dialogs queued
    assert dialog_manager.get_pending_count() == 4

    # Extract dialogs in priority order
    extracted_priorities = []
    while not dialog_manager._dialog_queue.empty():
        priority_value, counter, dialog = dialog_manager._dialog_queue.get_nowait()
        extracted_priorities.append(dialog.priority)

    # Verify correct priority order
    assert extracted_priorities == [
        DialogPriority.CRITICAL,
        DialogPriority.HIGH,
        DialogPriority.NORMAL,
        DialogPriority.LOW,
    ], "Dialogs should be ordered by priority: CRITICAL -> HIGH -> NORMAL -> LOW"


def test_fifo_ordering_for_same_priority(dialog_manager, event_bus):
    """Test FIFO ordering for same priority - queue 3 NORMAL dialogs -> show in FIFO order."""
    # Queue 3 NORMAL priority dialogs
    for i in range(3):
        dialog_event = DialogEvent(
            priority=DialogPriority.NORMAL,
            dialog_type=DialogType.INFO,
            context={"message": f"Dialog {i}", "order": i},
        )
        event_bus.publish(dialog_event)

    event_bus.dispatch(timeout=0.5)

    # Extract dialogs
    extracted_orders = []
    while not dialog_manager._dialog_queue.empty():
        priority_value, counter, dialog = dialog_manager._dialog_queue.get_nowait()
        extracted_orders.append(dialog.context["order"])

    # Verify FIFO order (0, 1, 2)
    assert extracted_orders == [0, 1, 2], "Same priority dialogs should be shown in FIFO order"


def test_counter_increments_for_stable_ordering(dialog_manager, event_bus):
    """Test counter increments for stable ordering."""
    # Queue 5 dialogs
    for i in range(5):
        dialog_event = DialogEvent(
            priority=DialogPriority.NORMAL,
            dialog_type=DialogType.INFO,
            context={"message": f"Dialog {i}"},
        )
        event_bus.publish(dialog_event)

    event_bus.dispatch(timeout=0.5)

    # Verify counter incremented
    assert dialog_manager._dialog_counter == 5, "Counter should increment for each queued dialog"

    # Extract counters from queue
    counters = []
    while not dialog_manager._dialog_queue.empty():
        priority_value, counter, dialog = dialog_manager._dialog_queue.get_nowait()
        counters.append(counter)

    # Verify sequential counters
    assert counters == [0, 1, 2, 3, 4], "Counters should be sequential for stable ordering"


# ============================================================================
# Dialog Rendering (4 tests)
# ============================================================================


def test_show_pending_dialogs_processes_all_dialogs(dialog_manager, event_bus, temp_output_dir, session_log):
    """Test show_pending_dialogs() processes all dialogs - queue 5 dialogs, verify all shown, queue empty."""
    # Queue 5 dialogs with different priorities
    priorities = [
        DialogPriority.NORMAL,
        DialogPriority.HIGH,
        DialogPriority.LOW,
        DialogPriority.CRITICAL,
        DialogPriority.NORMAL,
    ]

    for i, priority in enumerate(priorities):
        dialog_event = DialogEvent(
            priority=priority,
            dialog_type=DialogType.INFO,
            context={"message": f"Dialog {i}", "title": f"Test Dialog {i}"},
        )
        event_bus.publish(dialog_event)

    event_bus.dispatch(timeout=0.5)

    # Verify 5 dialogs queued
    assert dialog_manager.get_pending_count() == 5

    # Show all pending dialogs (mock input to avoid blocking)
    with patch("builtins.input", return_value=""):
        dialog_manager.show_pending_dialogs(output_dir=temp_output_dir, session_log=session_log)

    # Verify queue is empty
    assert dialog_manager.get_pending_count() == 0, "All dialogs should be processed and queue emptied"


def test_render_circuit_breaker_prints_panel(dialog_manager, console, event_bus, temp_output_dir, session_log):
    """Test _render_circuit_breaker() prints circuit breaker panel with Rich console."""
    # Queue circuit breaker dialog
    circuit_event = CircuitBreakerEvent(
        reason="Too many API_TIMEOUT errors (5 consecutive)",
        affected_tasks=[1, 2, 3, 4, 5],
        trigger_category=ErrorCategory.API_TIMEOUT,
        consecutive_counts={ErrorCategory.API_TIMEOUT: 5},
        timestamp=datetime.now(),
    )
    event_bus.publish(circuit_event)
    event_bus.dispatch(timeout=0.5)

    # Show dialog (mock input to avoid blocking)
    with patch("builtins.input", return_value=""):
        dialog_manager.show_pending_dialogs(output_dir=temp_output_dir, session_log=session_log)

    # Verify console output contains expected strings
    output = console.export_text()

    assert "Processing Paused" in output, "Should contain circuit breaker title"
    assert "Processing has been paused" in output, "Should contain explanation message"
    assert "Too many API_TIMEOUT errors" in output, "Should contain reason"
    assert "Affected Tasks: 5 tasks" in output, "Should show affected tasks count"
    assert "Next Steps" in output, "Should show next steps section"
    assert "session.log" in output or "Session Log" in output, "Should reference session log"


def test_render_error_summary_prints_error_table(dialog_manager, console, temp_output_dir, session_log):
    """Test _render_error_summary() prints error table with category counts."""
    # Queue error summary with breakdown
    error_breakdown = {
        ErrorCategory.API_TIMEOUT: 10,
        ErrorCategory.CONNECTION_ERROR: 5,
        ErrorCategory.QUOTA_EXCEEDED: 2,
    }

    recent_errors = [
        ErrorEvent(
            category=ErrorCategory.API_TIMEOUT,
            message="Request timeout after 30s",
            task_id=1,
            url="https://example.com",
            exception_type="TimeoutError",
            recoverable=True,
        ),
    ]

    dialog_manager.queue_error_summary(
        total_errors=17,
        error_breakdown=error_breakdown,
        recent_errors=recent_errors,
        output_dir=temp_output_dir,
        session_log=session_log,
    )

    # Show dialog
    with patch("builtins.input", return_value=""):
        dialog_manager.show_pending_dialogs(output_dir=temp_output_dir, session_log=session_log)

    # Verify console output
    output = console.export_text()

    assert "Error Summary" in output, "Should contain error summary title"
    assert "17 Total Errors" in output, "Should show total error count"
    assert "API_TIMEOUT" in output, "Should show API_TIMEOUT category"
    assert "10" in output, "Should show API_TIMEOUT count"
    assert "CONNECTION_ERROR" in output, "Should show CONNECTION_ERROR category"
    assert "5" in output, "Should show CONNECTION_ERROR count"
    assert "QUOTA_EXCEEDED" in output, "Should show QUOTA_EXCEEDED category"
    assert "2" in output, "Should show QUOTA_EXCEEDED count"


def test_render_info_prints_simple_panel(dialog_manager, console, event_bus):
    """Test _render_info() prints simple panel for info dialogs."""
    # Queue info dialog
    dialog_event = DialogEvent(
        priority=DialogPriority.NORMAL,
        dialog_type=DialogType.INFO,
        context={"message": "Processing complete! All tasks finished successfully.", "title": "✅ Success"},
    )
    event_bus.publish(dialog_event)
    event_bus.dispatch(timeout=0.5)

    # Show dialog
    with patch("builtins.input", return_value=""):
        dialog_manager.show_pending_dialogs()

    # Verify console output
    output = console.export_text()

    assert "Success" in output, "Should contain title"
    assert "Processing complete" in output, "Should contain message"
    assert "All tasks finished successfully" in output, "Should show full message"


# ============================================================================
# Integration with Events (1 test)
# ============================================================================


def test_circuit_breaker_event_triggers_dialog_queue(event_bus, console):
    """Test CircuitBreakerEvent triggers dialog queue - publish event, verify dialog queued with correct context."""
    # Create DialogManager (subscribes to events)
    dialog_manager = DialogManager(event_bus, console)

    # Verify queue is empty initially
    assert dialog_manager.get_pending_count() == 0

    # Publish CircuitBreakerEvent
    circuit_event = CircuitBreakerEvent(
        reason="Authentication failure - API key invalid",
        affected_tasks=[100, 101, 102],
        trigger_category=ErrorCategory.AUTHENTICATION,
        consecutive_counts={ErrorCategory.AUTHENTICATION: 3},
        timestamp=datetime(2025, 11, 28, 14, 45, 30),
    )
    event_bus.publish(circuit_event)

    # Dispatch event to trigger handler
    event_bus.dispatch(timeout=0.5)

    # Verify dialog was queued
    assert dialog_manager.get_pending_count() == 1, "CircuitBreakerEvent should trigger dialog queueing"

    # Verify dialog has correct context
    priority_value, counter, dialog = dialog_manager._dialog_queue.get_nowait()

    assert dialog.priority == DialogPriority.CRITICAL, "Circuit breaker dialog should be CRITICAL"
    assert dialog.dialog_type == DialogType.CIRCUIT_BREAKER
    assert dialog.context["reason"] == "Authentication failure - API key invalid"
    assert dialog.context["affected_tasks"] == [100, 101, 102]
    assert dialog.context["trigger_category"] == ErrorCategory.AUTHENTICATION
    assert dialog.context["consecutive_counts"] == {ErrorCategory.AUTHENTICATION: 3}
    assert dialog.context["timestamp"] == datetime(2025, 11, 28, 14, 45, 30)
