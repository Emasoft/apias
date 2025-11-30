"""
Unit tests for apias/batch_tui.py module.

Coverage: 70 tests covering URLState, URLTask, BatchStats, and BatchTUIManager classes.

Test Categories:
- Category 1: URLState Enum (5 tests)
- Category 2: URLTask Dataclass (5 tests)
- Category 3: BatchStats Dataclass (5 tests)
- Category 4: BatchTUIManager Initialization (8 tests)
- Category 5: BatchTUIManager Scroll Methods (6 tests)
- Category 6: BatchTUIManager State Management (6 tests)
- Category 7: Dashboard Creation Methods (5 tests)
- Category 8: Update Methods Edge Cases (5 tests)
- Category 9: Summary Methods (5 tests)
- Category 10: Navigation Edge Cases (5 tests)
- Category 11: Extended Summary Methods (15 tests)
  - show_final_summary: 5 tests
  - prompt_retry_failed: 4 tests
  - show_circuit_breaker_dialog: 3 tests
  - show_session_summary: 3 tests

All tests use realistic data and execute actual code paths.
External dependencies (Rich Console, Live, stdin) are mocked where necessary.
"""

import io
import shutil
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import List
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel

from apias.batch_tui import BatchStats, BatchTUIManager, URLState, URLTask
from apias.terminal_utils import BaseTUIManager, ProcessState

# =============================================================================
# Category 1: URLState Enum Tests (5 tests)
# =============================================================================


class TestURLStateEnum:
    """Tests for URLState enumeration and its get_symbol method."""

    def test_urlstate_has_all_expected_states(self) -> None:
        """Verify URLState enum contains all required processing states."""
        # WHY: Ensures all expected states exist for URL processing pipeline
        expected_states = {
            "PENDING",
            "SCRAPING",
            "PROCESSING",
            "MERGING_CHUNKS",
            "COMPLETE",
            "FAILED",
        }
        actual_states = {state.name for state in URLState}
        assert actual_states == expected_states, (
            f"Missing or extra states. Expected: {expected_states}, Got: {actual_states}"
        )

    def test_urlstate_get_symbol_pending(self) -> None:
        """Verify get_symbol returns a non-empty string for PENDING state."""
        # WHY: PENDING state must have a visual indicator for TUI display
        symbol = URLState.PENDING.get_symbol()
        assert isinstance(symbol, str), "Symbol must be a string"
        assert len(symbol) > 0, "Symbol must not be empty"

    def test_urlstate_get_symbol_complete(self) -> None:
        """Verify get_symbol returns a non-empty string for COMPLETE state."""
        # WHY: COMPLETE state needs visual checkmark or success indicator
        symbol = URLState.COMPLETE.get_symbol()
        assert isinstance(symbol, str), "Symbol must be a string"
        assert len(symbol) > 0, "Symbol must not be empty"

    def test_urlstate_get_symbol_failed(self) -> None:
        """Verify get_symbol returns a non-empty string for FAILED state."""
        # WHY: FAILED state needs visual error indicator for user attention
        symbol = URLState.FAILED.get_symbol()
        assert isinstance(symbol, str), "Symbol must be a string"
        assert len(symbol) > 0, "Symbol must not be empty"

    def test_urlstate_get_symbol_processing(self) -> None:
        """Verify get_symbol returns a non-empty string for PROCESSING state."""
        # WHY: PROCESSING state needs progress indicator for active tasks
        symbol = URLState.PROCESSING.get_symbol()
        assert isinstance(symbol, str), "Symbol must be a string"
        assert len(symbol) > 0, "Symbol must not be empty"


# =============================================================================
# Category 2: URLTask Dataclass Tests (5 tests)
# =============================================================================


class TestURLTaskDataclass:
    """Tests for URLTask dataclass creation and field manipulation."""

    def test_urltask_creation_defaults(self) -> None:
        """Verify URLTask creates with correct default values for optional fields."""
        # WHY: Default values are critical for task initialization before processing
        task = URLTask(task_id=1, url="https://example.com/api")

        assert task.task_id == 1
        assert task.url == "https://example.com/api"
        assert task.state == URLState.PENDING
        assert task.progress_pct == 0.0
        assert task.size_in == 0
        assert task.size_out == 0
        assert task.cost == 0.0
        assert task.duration == 0.0
        assert task.start_time is None
        assert task.error == ""
        assert task.current_chunk == 0
        assert task.total_chunks == 0
        assert task.status_message == ""

    def test_urltask_creation_full(self) -> None:
        """Verify URLTask accepts all fields when explicitly specified."""
        # WHY: Full field specification needed for task restoration and updates
        start = time.time()
        status_hist: List[tuple[datetime, str]] = [
            (datetime.now(), "Started processing")
        ]

        task = URLTask(
            task_id=42,
            url="https://api.example.com/v2/docs",
            state=URLState.PROCESSING,
            progress_pct=65.5,
            size_in=10240,
            size_out=5120,
            cost=0.0025,
            duration=15.7,
            start_time=start,
            error="",
            current_chunk=3,
            total_chunks=5,
            status_message="Processing chunk 3/5",
            status_history=status_hist,
        )

        assert task.task_id == 42
        assert task.url == "https://api.example.com/v2/docs"
        assert task.state == URLState.PROCESSING
        assert task.progress_pct == 65.5
        assert task.size_in == 10240
        assert task.size_out == 5120
        assert task.cost == 0.0025
        assert task.duration == 15.7
        assert task.start_time == start
        assert task.current_chunk == 3
        assert task.total_chunks == 5
        assert task.status_message == "Processing chunk 3/5"
        assert len(task.status_history) == 1

    def test_urltask_state_transition(self) -> None:
        """Verify URLTask state field can be changed to track processing progress."""
        # WHY: State transitions are core to progress tracking
        task = URLTask(task_id=1, url="https://example.com")

        assert task.state == URLState.PENDING

        task.state = URLState.SCRAPING
        assert task.state == URLState.SCRAPING

        task.state = URLState.PROCESSING
        assert task.state == URLState.PROCESSING

        task.state = URLState.COMPLETE
        assert task.state == URLState.COMPLETE

    def test_urltask_progress_update(self) -> None:
        """Verify URLTask progress_pct field updates correctly."""
        # WHY: Progress percentage drives TUI progress bar display
        task = URLTask(task_id=1, url="https://example.com")

        assert task.progress_pct == 0.0

        task.progress_pct = 25.5
        assert task.progress_pct == 25.5

        task.progress_pct = 100.0
        assert task.progress_pct == 100.0

    def test_urltask_status_history(self) -> None:
        """Verify URLTask status_history field is a mutable list."""
        # WHY: Status history enables debugging by tracking state changes
        task = URLTask(task_id=1, url="https://example.com")

        assert isinstance(task.status_history, list)
        assert len(task.status_history) == 0

        # Add status entries
        task.status_history.append((datetime.now(), "Started"))
        task.status_history.append((datetime.now(), "Retrying..."))

        assert len(task.status_history) == 2
        assert task.status_history[0][1] == "Started"
        assert task.status_history[1][1] == "Retrying..."


# =============================================================================
# Category 3: BatchStats Dataclass Tests (5 tests)
# =============================================================================


class TestBatchStatsDataclass:
    """Tests for BatchStats dataclass creation and field values."""

    def test_batchstats_creation_defaults(self) -> None:
        """Verify BatchStats creates with correct default values."""
        # WHY: Default stats are used when initializing new batch processing
        stats = BatchStats()

        assert stats.total_urls == 0
        assert stats.pending == 0
        assert stats.scraping == 0
        assert stats.processing == 0
        assert stats.completed == 0
        assert stats.failed == 0
        assert stats.total_cost == 0.0
        assert isinstance(stats.start_time, datetime)

    def test_batchstats_total_urls(self) -> None:
        """Verify BatchStats total_urls field stores URL count correctly."""
        # WHY: total_urls is essential for progress percentage calculation
        stats = BatchStats(total_urls=100, pending=100)

        assert stats.total_urls == 100
        assert stats.pending == 100

    def test_batchstats_counts_sum(self) -> None:
        """Verify state counts do not exceed total_urls (invariant check)."""
        # WHY: Ensures stats consistency - sum of states cannot exceed total
        stats = BatchStats(
            total_urls=10,
            pending=2,
            scraping=2,
            processing=3,
            completed=2,
            failed=1,
        )

        state_sum = (
            stats.pending
            + stats.scraping
            + stats.processing
            + stats.completed
            + stats.failed
        )
        assert state_sum <= stats.total_urls, (
            f"State sum {state_sum} exceeds total_urls {stats.total_urls}"
        )

    def test_batchstats_total_cost(self) -> None:
        """Verify BatchStats total_cost field stores accumulated cost."""
        # WHY: Cost tracking is critical for API usage monitoring
        stats = BatchStats(total_cost=0.0)
        assert stats.total_cost == 0.0

        stats.total_cost = 0.0125
        assert stats.total_cost == 0.0125

        stats.total_cost += 0.0050
        assert abs(stats.total_cost - 0.0175) < 1e-10

    def test_batchstats_start_time(self) -> None:
        """Verify BatchStats start_time is a datetime instance."""
        # WHY: start_time is required for elapsed time and ETA calculations
        stats = BatchStats()

        assert isinstance(stats.start_time, datetime)
        # start_time should be recent (within last second)
        time_diff = (datetime.now() - stats.start_time).total_seconds()
        assert time_diff < 1.0, "start_time should be set to current time"


# =============================================================================
# Category 4: BatchTUIManager Initialization Tests (8 tests)
# =============================================================================


class TestBatchTUIManagerInit:
    """Tests for BatchTUIManager initialization and configuration."""

    @patch("apias.terminal_utils.detect_terminal_capabilities")
    def test_batchtuimanager_init_no_tui(self, mock_detect: MagicMock) -> None:
        """Verify BatchTUIManager initializes with no_tui=True correctly."""
        # WHY: no_tui mode is required for headless/CI environments
        mock_detect.return_value = {"unicode": True, "colors": True}

        urls = ["https://example.com/page1", "https://example.com/page2"]
        manager = BatchTUIManager(urls=urls, no_tui=True)

        assert manager.no_tui is True
        assert len(manager.tasks) == 2

    @patch("apias.terminal_utils.detect_terminal_capabilities")
    def test_batchtuimanager_init_quiet(self, mock_detect: MagicMock) -> None:
        """Verify quiet=True implies no_tui=True."""
        # WHY: quiet mode should suppress all TUI output
        mock_detect.return_value = {"unicode": True, "colors": True}

        urls = ["https://example.com"]
        manager = BatchTUIManager(urls=urls, quiet=True)

        assert manager.quiet is True
        assert manager.no_tui is True, "quiet=True should set no_tui=True"

    @patch("apias.terminal_utils.detect_terminal_capabilities")
    def test_batchtuimanager_init_default_stats(self, mock_detect: MagicMock) -> None:
        """Verify BatchTUIManager initializes stats with correct URL counts."""
        # WHY: Stats must reflect initial state (all URLs pending)
        mock_detect.return_value = {"unicode": True, "colors": True}

        urls = [
            "https://example.com/1",
            "https://example.com/2",
            "https://example.com/3",
        ]
        manager = BatchTUIManager(urls=urls, no_tui=True)

        assert manager.stats.total_urls == 3
        assert manager.stats.pending == 3
        assert manager.stats.scraping == 0
        assert manager.stats.processing == 0
        assert manager.stats.completed == 0
        assert manager.stats.failed == 0

    @patch("apias.terminal_utils.detect_terminal_capabilities")
    def test_batchtuimanager_init_empty_tasks(self, mock_detect: MagicMock) -> None:
        """Verify BatchTUIManager with empty URL list has empty tasks dict."""
        # WHY: Edge case - empty batch should not cause errors
        mock_detect.return_value = {"unicode": True, "colors": True}

        manager = BatchTUIManager(urls=[], no_tui=True)

        assert len(manager.tasks) == 0
        assert manager.stats.total_urls == 0
        assert manager.stats.pending == 0

    @patch("apias.terminal_utils.detect_terminal_capabilities")
    def test_batchtuimanager_init_scroll_offset(self, mock_detect: MagicMock) -> None:
        """Verify scroll_offset starts at 0."""
        # WHY: Initial scroll position should be at the top
        mock_detect.return_value = {"unicode": True, "colors": True}

        urls = ["https://example.com"]
        manager = BatchTUIManager(urls=urls, no_tui=True)

        assert manager.scroll_offset == 0

    @patch("apias.terminal_utils.detect_terminal_capabilities")
    def test_batchtuimanager_init_task_lock(self, mock_detect: MagicMock) -> None:
        """Verify _task_lock is a threading.Lock instance."""
        # WHY: Thread safety requires proper locking for task updates
        mock_detect.return_value = {"unicode": True, "colors": True}

        urls = ["https://example.com"]
        manager = BatchTUIManager(urls=urls, no_tui=True)

        assert isinstance(manager._task_lock, type(threading.Lock()))

    @patch("apias.terminal_utils.detect_terminal_capabilities")
    def test_batchtuimanager_init_scroll_lock(self, mock_detect: MagicMock) -> None:
        """Verify _scroll_lock is a threading.Lock instance."""
        # WHY: Thread safety requires proper locking for scroll operations
        mock_detect.return_value = {"unicode": True, "colors": True}

        urls = ["https://example.com"]
        manager = BatchTUIManager(urls=urls, no_tui=True)

        assert isinstance(manager._scroll_lock, type(threading.Lock()))

    @patch("apias.terminal_utils.detect_terminal_capabilities")
    def test_batchtuimanager_inherits_base(self, mock_detect: MagicMock) -> None:
        """Verify BatchTUIManager inherits from BaseTUIManager."""
        # WHY: Inheritance provides common TUI functionality
        mock_detect.return_value = {"unicode": True, "colors": True}

        urls = ["https://example.com"]
        manager = BatchTUIManager(urls=urls, no_tui=True)

        assert isinstance(manager, BaseTUIManager)
        # Verify inherited attributes exist
        assert hasattr(manager, "process_state")
        assert hasattr(manager, "console")
        assert hasattr(manager, "_state_lock")


# =============================================================================
# Category 5: BatchTUIManager Scroll Methods Tests (6 tests)
# =============================================================================


class TestBatchTUIManagerScroll:
    """Tests for BatchTUIManager scroll functionality."""

    @patch("apias.terminal_utils.detect_terminal_capabilities")
    def test_scroll_debounce_respects_threshold(self, mock_detect: MagicMock) -> None:
        """Verify _should_debounce_scroll returns True within debounce window."""
        # WHY: Debouncing prevents accidental rapid scrolling from key repeat
        mock_detect.return_value = {"unicode": True, "colors": True}

        urls = ["https://example.com"]
        manager = BatchTUIManager(urls=urls, no_tui=True)

        # First call should NOT debounce (sets initial time)
        result1 = manager._should_debounce_scroll()
        assert result1 is False, "First scroll should not be debounced"

        # Immediate second call SHOULD debounce (within threshold)
        result2 = manager._should_debounce_scroll()
        assert result2 is True, "Rapid second scroll should be debounced"

    @patch("apias.terminal_utils.detect_terminal_capabilities")
    def test_scroll_up_decrements_offset(self, mock_detect: MagicMock) -> None:
        """Verify _on_scroll_up decreases scroll_offset when > 0."""
        # WHY: Scroll up should move viewport towards beginning of list
        mock_detect.return_value = {"unicode": True, "colors": True}

        urls = [f"https://example.com/{i}" for i in range(20)]
        manager = BatchTUIManager(urls=urls, no_tui=True)

        # Set scroll_offset > 0 first
        manager.scroll_offset = 5
        manager._last_scroll_time = 0  # Clear debounce

        manager._on_scroll_up()

        assert manager.scroll_offset == 4

    @patch("apias.terminal_utils.detect_terminal_capabilities")
    def test_scroll_up_stops_at_zero(self, mock_detect: MagicMock) -> None:
        """Verify scroll_offset does not go negative when at 0."""
        # WHY: Negative scroll offset would cause display bugs
        mock_detect.return_value = {"unicode": True, "colors": True}

        urls = ["https://example.com"]
        manager = BatchTUIManager(urls=urls, no_tui=True)

        assert manager.scroll_offset == 0
        manager._last_scroll_time = 0  # Clear debounce

        manager._on_scroll_up()

        assert manager.scroll_offset == 0, "scroll_offset should not go negative"

    @patch("apias.terminal_utils.detect_terminal_capabilities")
    def test_scroll_down_increments_offset(self, mock_detect: MagicMock) -> None:
        """Verify _on_scroll_down increases scroll_offset."""
        # WHY: Scroll down should move viewport towards end of list
        mock_detect.return_value = {"unicode": True, "colors": True}

        # Create manager with many URLs to allow scrolling
        urls = [f"https://example.com/{i}" for i in range(50)]
        manager = BatchTUIManager(urls=urls, no_tui=True)

        initial_offset = manager.scroll_offset
        manager._last_scroll_time = 0  # Clear debounce

        manager._on_scroll_down()

        assert manager.scroll_offset == initial_offset + 1

    @patch("apias.terminal_utils.detect_terminal_capabilities")
    def test_page_up_decrements_by_page(self, mock_detect: MagicMock) -> None:
        """Verify _on_page_up changes scroll by page size (visible tasks count)."""
        # WHY: PageUp should jump by a full page for faster navigation
        mock_detect.return_value = {"unicode": True, "colors": True}

        urls = [f"https://example.com/{i}" for i in range(100)]
        manager = BatchTUIManager(urls=urls, no_tui=True)

        # Set scroll offset to middle of list
        manager.scroll_offset = 50
        manager._last_scroll_time = 0  # Clear debounce

        # Get page size for comparison
        page_size = manager._get_visible_tasks_count()

        manager._on_page_up()

        # scroll_offset should decrease by page_size (but not below 0)
        expected = max(0, 50 - page_size)
        assert manager.scroll_offset == expected

    @patch("apias.terminal_utils.detect_terminal_capabilities")
    def test_home_resets_to_zero(self, mock_detect: MagicMock) -> None:
        """Verify _on_home sets scroll_offset to 0."""
        # WHY: Home key should jump to beginning of task list
        mock_detect.return_value = {"unicode": True, "colors": True}

        urls = [f"https://example.com/{i}" for i in range(50)]
        manager = BatchTUIManager(urls=urls, no_tui=True)

        # Set scroll offset to non-zero
        manager.scroll_offset = 25
        manager._last_scroll_time = 0  # Clear debounce

        manager._on_home()

        assert manager.scroll_offset == 0


# =============================================================================
# Category 6: BatchTUIManager State Management Tests (6 tests)
# =============================================================================


class TestBatchTUIManagerStateManagement:
    """Tests for BatchTUIManager task and state management methods."""

    @patch("apias.terminal_utils.detect_terminal_capabilities")
    def test_update_task_creates_new_task(self, mock_detect: MagicMock) -> None:
        """Verify update_task handles updates for tasks created during init."""
        # WHY: Tasks are pre-created during init, update_task should modify them
        mock_detect.return_value = {"unicode": True, "colors": True}

        urls = ["https://example.com/page1"]
        manager = BatchTUIManager(urls=urls, no_tui=True)

        # Task 1 was created during init
        assert 1 in manager.tasks
        assert manager.tasks[1].state == URLState.PENDING

        # Update existing task
        manager.update_task(
            task_id=1,
            state=URLState.SCRAPING,
            progress_pct=10.0,
        )

        assert manager.tasks[1].state == URLState.SCRAPING
        assert manager.tasks[1].progress_pct == 10.0

    @patch("apias.terminal_utils.detect_terminal_capabilities")
    def test_update_task_updates_existing(self, mock_detect: MagicMock) -> None:
        """Verify update_task modifies existing task with new values."""
        # WHY: Task updates drive TUI display changes during processing
        mock_detect.return_value = {"unicode": True, "colors": True}

        urls = ["https://example.com"]
        manager = BatchTUIManager(urls=urls, no_tui=True)

        # Update task through processing stages
        manager.update_task(
            task_id=1,
            state=URLState.SCRAPING,
            progress_pct=0.0,
        )

        manager.update_task(
            task_id=1,
            state=URLState.PROCESSING,
            progress_pct=50.0,
            size_in=1024,
            cost=0.001,
        )

        task = manager.tasks[1]
        assert task.state == URLState.PROCESSING
        assert task.progress_pct == 50.0
        assert task.size_in == 1024
        assert task.cost == 0.001

    @patch("apias.terminal_utils.detect_terminal_capabilities")
    def test_update_stats_increments_completed(self, mock_detect: MagicMock) -> None:
        """Verify _update_stats increments completed count on state transition."""
        # WHY: Stats tracking is essential for progress display
        mock_detect.return_value = {"unicode": True, "colors": True}

        urls = ["https://example.com"]
        manager = BatchTUIManager(urls=urls, no_tui=True)

        assert manager.stats.completed == 0

        # Transition from PROCESSING to COMPLETE
        manager._update_stats(
            old_state=URLState.PROCESSING,
            new_state=URLState.COMPLETE,
            cost=0.002,
        )

        assert manager.stats.completed == 1

    @patch("apias.terminal_utils.detect_terminal_capabilities")
    def test_update_stats_calculates_total_cost(self, mock_detect: MagicMock) -> None:
        """Verify _update_stats accumulates cost on completion."""
        # WHY: Total cost tracking is critical for API usage monitoring
        mock_detect.return_value = {"unicode": True, "colors": True}

        urls = ["https://example.com/1", "https://example.com/2"]
        manager = BatchTUIManager(urls=urls, no_tui=True)

        assert manager.stats.total_cost == 0.0

        # First task completes
        manager._update_stats(
            old_state=URLState.PROCESSING,
            new_state=URLState.COMPLETE,
            cost=0.0015,
        )

        assert abs(manager.stats.total_cost - 0.0015) < 1e-10

        # Second task completes
        manager._update_stats(
            old_state=URLState.PROCESSING,
            new_state=URLState.COMPLETE,
            cost=0.0025,
        )

        assert abs(manager.stats.total_cost - 0.0040) < 1e-10

    @patch("apias.terminal_utils.detect_terminal_capabilities")
    def test_get_effective_elapsed_returns_float(self, mock_detect: MagicMock) -> None:
        """Verify get_effective_elapsed returns a positive float."""
        # WHY: Elapsed time is required for ETA calculation
        mock_detect.return_value = {"unicode": True, "colors": True}

        urls = ["https://example.com"]
        manager = BatchTUIManager(urls=urls, no_tui=True)

        elapsed = manager.get_effective_elapsed()

        assert isinstance(elapsed, float)
        assert elapsed >= 0.0

    @patch("apias.terminal_utils.detect_terminal_capabilities")
    def test_visible_tasks_count_positive(self, mock_detect: MagicMock) -> None:
        """Verify _get_visible_tasks_count returns a positive integer."""
        # WHY: Visible count determines scrolling bounds and display
        mock_detect.return_value = {"unicode": True, "colors": True}

        urls = ["https://example.com"]
        manager = BatchTUIManager(urls=urls, no_tui=True)

        visible_count = manager._get_visible_tasks_count()

        assert isinstance(visible_count, int)
        assert visible_count >= 1, "At least 1 task should be visible"


# =============================================================================
# Category 7: Dashboard Creation Methods Tests (5 tests)
# =============================================================================


class TestDashboardCreation:
    """Tests for dashboard and panel creation methods."""

    @patch("apias.terminal_utils.detect_terminal_capabilities")
    def test_create_waiting_dashboard_returns_renderable(
        self, mock_detect: MagicMock
    ) -> None:
        """Verify _create_waiting_dashboard returns a Rich Panel renderable."""
        # WHY: Waiting dashboard must be a valid Rich renderable for Live display
        mock_detect.return_value = {"unicode": True, "colors": True}

        urls = ["https://example.com/page1", "https://example.com/page2"]
        manager = BatchTUIManager(urls=urls, no_tui=True)

        dashboard = manager._create_waiting_dashboard()

        # Verify it returns a Panel instance
        assert isinstance(dashboard, Panel), "Waiting dashboard must be a Panel"
        # Panel should have a title
        assert dashboard.title is not None, "Panel should have a title"

    @patch("apias.terminal_utils.detect_terminal_capabilities")
    def test_create_dashboard_returns_renderable(self, mock_detect: MagicMock) -> None:
        """Verify _create_dashboard returns a Rich Layout renderable."""
        # WHY: Main dashboard must be a valid Rich Layout for TUI display
        mock_detect.return_value = {"unicode": True, "colors": True}

        urls = ["https://example.com/api"]
        manager = BatchTUIManager(urls=urls, no_tui=True)

        dashboard = manager._create_dashboard()

        # Verify it returns a Layout instance
        assert isinstance(dashboard, Layout), "Dashboard must be a Layout"

    @patch("apias.terminal_utils.detect_terminal_capabilities")
    def test_create_dashboard_with_tasks(self, mock_detect: MagicMock) -> None:
        """Verify _create_dashboard handles multiple tasks with different states."""
        # WHY: Dashboard must render correctly with tasks in various states
        mock_detect.return_value = {"unicode": True, "colors": True}

        urls = [f"https://example.com/page{i}" for i in range(5)]
        manager = BatchTUIManager(urls=urls, no_tui=True)

        # Set tasks to different states
        manager.tasks[1].state = URLState.COMPLETE
        manager.tasks[1].progress_pct = 100.0
        manager.tasks[2].state = URLState.PROCESSING
        manager.tasks[2].progress_pct = 50.0
        manager.tasks[3].state = URLState.FAILED
        manager.tasks[3].error = "Test error"

        dashboard = manager._create_dashboard()

        # Verify layout has expected sections
        assert isinstance(dashboard, Layout)
        # Layout should be created without errors regardless of task states

    @patch("apias.terminal_utils.detect_terminal_capabilities")
    def test_create_stats_panel_returns_panel(self, mock_detect: MagicMock) -> None:
        """Verify _create_stats_panel returns a Rich Panel with statistics."""
        # WHY: Stats panel displays overall batch progress and must be a valid Panel
        mock_detect.return_value = {"unicode": True, "colors": True}

        urls = [
            "https://example.com/1",
            "https://example.com/2",
            "https://example.com/3",
        ]
        manager = BatchTUIManager(urls=urls, no_tui=True)

        # Simulate some processing progress
        manager.stats.completed = 1
        manager.stats.pending = 1
        manager.stats.processing = 1
        manager.stats.total_cost = 0.0025

        panel = manager._create_stats_panel()

        assert isinstance(panel, Panel), "Stats panel must be a Panel"
        assert panel.title is not None, "Stats panel should have a title"

    @patch("apias.terminal_utils.detect_terminal_capabilities")
    def test_create_urls_panel_returns_panel(self, mock_detect: MagicMock) -> None:
        """Verify _create_urls_panel returns a Rich Panel with URL list."""
        # WHY: URLs panel shows scrollable task list and must be a valid Panel
        mock_detect.return_value = {"unicode": True, "colors": True}

        urls = [f"https://api.example.com/docs/{i}" for i in range(10)]
        manager = BatchTUIManager(urls=urls, no_tui=True)

        # Simulate realistic task progress
        manager.tasks[1].state = URLState.COMPLETE
        manager.tasks[1].progress_pct = 100.0
        manager.tasks[1].size_in = 5120
        manager.tasks[1].size_out = 2048
        manager.tasks[2].state = URLState.PROCESSING
        manager.tasks[2].progress_pct = 45.0
        manager.tasks[2].current_chunk = 2
        manager.tasks[2].total_chunks = 4

        panel = manager._create_urls_panel(available_height=30)

        assert isinstance(panel, Panel), "URLs panel must be a Panel"
        assert panel.title is not None, "URLs panel should have a title"


# =============================================================================
# Category 8: Update Methods Edge Cases Tests (5 tests)
# =============================================================================


class TestUpdateMethodsEdgeCases:
    """Tests for edge cases in update_task and _update_stats methods."""

    @patch("apias.terminal_utils.detect_terminal_capabilities")
    def test_update_task_with_error_state(self, mock_detect: MagicMock) -> None:
        """Verify update_task correctly handles FAILED state with error message."""
        # WHY: Error state must preserve error message for user feedback
        mock_detect.return_value = {"unicode": True, "colors": True}

        urls = ["https://example.com/api"]
        manager = BatchTUIManager(urls=urls, no_tui=True)

        error_message = "OpenAI API rate limit exceeded"
        manager.update_task(
            task_id=1,
            state=URLState.FAILED,
            progress_pct=30.0,
            error=error_message,
            status_message="Error: Rate limit exceeded. Aborting.",
        )

        task = manager.tasks[1]
        assert task.state == URLState.FAILED
        assert task.error == error_message
        assert "Aborting" in task.status_message
        assert manager.stats.failed == 1

    @patch("apias.terminal_utils.detect_terminal_capabilities")
    def test_update_task_progress_100_sets_complete(
        self, mock_detect: MagicMock
    ) -> None:
        """Verify update_task at 100% progress with COMPLETE state updates correctly."""
        # WHY: Completion must update duration and set final state
        mock_detect.return_value = {"unicode": True, "colors": True}

        urls = ["https://example.com"]
        manager = BatchTUIManager(urls=urls, no_tui=True)

        # Start task
        manager.update_task(task_id=1, state=URLState.SCRAPING, progress_pct=0.0)
        # Ensure start_time is set
        manager.tasks[1].start_time = time.time() - 5.0  # 5 seconds ago

        # Complete task
        manager.update_task(
            task_id=1,
            state=URLState.COMPLETE,
            progress_pct=100.0,
            size_in=10240,
            size_out=4096,
            cost=0.0015,
        )

        task = manager.tasks[1]
        assert task.state == URLState.COMPLETE
        assert task.progress_pct == 100.0
        assert task.duration > 0, "Duration should be set on completion"
        assert manager.stats.completed == 1

    @patch("apias.terminal_utils.detect_terminal_capabilities")
    def test_update_stats_failed_count(self, mock_detect: MagicMock) -> None:
        """Verify _update_stats increments failed count on FAILED transition."""
        # WHY: Failed count tracking is essential for retry decisions
        mock_detect.return_value = {"unicode": True, "colors": True}

        urls = ["https://example.com/1", "https://example.com/2"]
        manager = BatchTUIManager(urls=urls, no_tui=True)

        assert manager.stats.failed == 0

        # Transition from PROCESSING to FAILED
        manager._update_stats(
            old_state=URLState.PROCESSING,
            new_state=URLState.FAILED,
            cost=None,
        )

        assert manager.stats.failed == 1

        # Another failure
        manager._update_stats(
            old_state=URLState.SCRAPING,
            new_state=URLState.FAILED,
            cost=None,
        )

        assert manager.stats.failed == 2

    @patch("apias.terminal_utils.detect_terminal_capabilities")
    def test_update_stats_with_zero_cost(self, mock_detect: MagicMock) -> None:
        """Verify _update_stats handles zero cost correctly (not None)."""
        # WHY: Zero cost is valid (cached response), should not skip accumulation logic
        mock_detect.return_value = {"unicode": True, "colors": True}

        urls = ["https://example.com"]
        manager = BatchTUIManager(urls=urls, no_tui=True)
        manager.stats.total_cost = 0.001  # Pre-existing cost

        # Complete with zero cost (e.g., cached response)
        manager._update_stats(
            old_state=URLState.PROCESSING,
            new_state=URLState.COMPLETE,
            cost=0.0,  # Zero, not None
        )

        # Cost should remain 0.001 (0.001 + 0.0)
        assert abs(manager.stats.total_cost - 0.001) < 1e-10
        assert manager.stats.completed == 1

    @patch("apias.terminal_utils.detect_terminal_capabilities")
    def test_render_snapshot_returns_none_for_no_tui(
        self, mock_detect: MagicMock
    ) -> None:
        """Verify render_snapshot exits early when no_tui is True."""
        # WHY: No TUI mode should not waste CPU on rendering
        mock_detect.return_value = {"unicode": True, "colors": True}

        urls = ["https://example.com"]
        manager = BatchTUIManager(urls=urls, no_tui=True)

        # Create a mock snapshot
        @dataclass
        class MockTaskSnapshot:
            task_id: int = 1
            url: str = "https://example.com"
            state: URLState = URLState.PROCESSING
            progress_pct: float = 50.0
            size_in: int = 1024
            size_out: int = 512
            cost: float = 0.001
            duration: float = 0.0
            start_time: float | None = None
            error: str = ""
            current_chunk: int = 0
            total_chunks: int = 0
            status_message: str = ""

        snapshot = {1: MockTaskSnapshot()}

        # Should not raise and should exit early
        result = manager.render_snapshot(snapshot)
        assert result is None  # Method returns None


# =============================================================================
# Category 9: Summary Methods Tests (5 tests)
# =============================================================================


class TestSummaryMethods:
    """Tests for summary and display methods."""

    @patch("apias.terminal_utils.detect_terminal_capabilities")
    def test_show_final_summary_no_tui_mode(self, mock_detect: MagicMock) -> None:
        """Verify show_final_summary uses simple text output in no_tui mode."""
        # WHY: no_tui mode must produce readable text output without Rich
        mock_detect.return_value = {"unicode": True, "colors": True}

        urls = ["https://example.com/1", "https://example.com/2"]
        manager = BatchTUIManager(urls=urls, no_tui=True)

        # Set up completed stats
        manager.stats.completed = 2
        manager.stats.pending = 0
        manager.stats.total_cost = 0.003

        # Capture stdout
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            manager.show_final_summary(output_dir="/tmp/test_output")

        output = captured.getvalue()
        assert "BATCH PROCESSING COMPLETE" in output
        assert "Success Rate" in output

    @patch("apias.terminal_utils.detect_terminal_capabilities")
    def test_show_session_summary_returns_none(self, mock_detect: MagicMock) -> None:
        """Verify show_session_summary executes without errors."""
        # WHY: Session summary must be displayable in both TUI and no_tui modes
        mock_detect.return_value = {"unicode": True, "colors": True}

        urls = ["https://example.com"]
        manager = BatchTUIManager(urls=urls, no_tui=True)

        # Capture stdout
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            result = manager.show_session_summary(
                session_log_path="/tmp/session.log", error_count=2
            )

        assert result is None  # Method returns None
        output = captured.getvalue()
        assert "session.log" in output or "Session log" in output

    @patch("apias.terminal_utils.detect_terminal_capabilities")
    def test_print_simple_summary_outputs_text(self, mock_detect: MagicMock) -> None:
        """Verify _print_simple_summary outputs text statistics."""
        # WHY: Simple summary must work without Rich dependencies
        mock_detect.return_value = {"unicode": True, "colors": True}

        urls = [
            "https://example.com/1",
            "https://example.com/2",
            "https://example.com/3",
        ]
        manager = BatchTUIManager(urls=urls, no_tui=True)

        manager.stats.completed = 2
        manager.stats.failed = 1
        manager.stats.total_cost = 0.0045

        # Capture stdout
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            manager._print_simple_summary(output_dir="/tmp/output")

        output = captured.getvalue()
        assert "BATCH PROCESSING COMPLETE" in output
        assert "2/3" in output  # Success rate
        assert "Failed: 1" in output
        assert "/tmp/output" in output

    @patch("apias.terminal_utils.detect_terminal_capabilities")
    def test_print_simple_circuit_breaker_outputs_text(
        self, mock_detect: MagicMock
    ) -> None:
        """Verify _print_simple_circuit_breaker outputs error information."""
        # WHY: Circuit breaker message must inform user about stop reason
        mock_detect.return_value = {"unicode": True, "colors": True}

        urls = ["https://example.com/1", "https://example.com/2"]
        manager = BatchTUIManager(urls=urls, no_tui=True)

        manager.stats.completed = 1
        manager.stats.total_urls = 2

        # Capture stdout
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            manager._print_simple_circuit_breaker(
                trigger_reason="OpenAI quota exceeded", output_dir="/tmp/batch_output"
            )

        output = captured.getvalue()
        assert "PROCESSING STOPPED" in output
        assert "OpenAI quota exceeded" in output
        assert "1/2" in output  # Progress saved
        assert "resume" in output.lower()

    @patch("apias.terminal_utils.detect_terminal_capabilities")
    def test_stop_live_display_clears_live(self, mock_detect: MagicMock) -> None:
        """Verify stop_live_display sets live to None and stops listener."""
        # WHY: Stopping display must clean up resources properly
        mock_detect.return_value = {"unicode": True, "colors": True}

        urls = ["https://example.com"]
        manager = BatchTUIManager(urls=urls, no_tui=True)

        # Simulate an active live display
        mock_live = MagicMock()
        manager.live = mock_live

        manager.stop_live_display()

        mock_live.stop.assert_called_once()
        assert manager.live is None


# =============================================================================
# Category 10: Navigation Edge Cases Tests (5 tests)
# =============================================================================


class TestNavigationEdgeCases:
    """Tests for navigation edge cases in scroll methods."""

    @patch("apias.terminal_utils.detect_terminal_capabilities")
    def test_on_end_with_many_tasks(self, mock_detect: MagicMock) -> None:
        """Verify _on_end scrolls to correct position with many tasks."""
        # WHY: End key must position viewport at last page of tasks
        mock_detect.return_value = {"unicode": True, "colors": True}

        urls = [f"https://example.com/{i}" for i in range(100)]
        manager = BatchTUIManager(urls=urls, no_tui=True)
        manager._last_scroll_time = 0  # Clear debounce

        manager._on_end()

        # scroll_offset should be at max position
        visible = manager._get_visible_tasks_count()
        expected_offset = max(0, len(urls) - visible)
        assert manager.scroll_offset == expected_offset

    @patch("apias.terminal_utils.detect_terminal_capabilities")
    def test_page_down_respects_bounds(self, mock_detect: MagicMock) -> None:
        """Verify _on_page_down does not scroll past end of task list."""
        # WHY: Scrolling past bounds would show empty content
        mock_detect.return_value = {"unicode": True, "colors": True}

        urls = [f"https://example.com/{i}" for i in range(20)]
        manager = BatchTUIManager(urls=urls, no_tui=True)

        # Set offset near the end
        visible = manager._get_visible_tasks_count()
        manager.scroll_offset = max(0, len(urls) - visible - 1)
        manager._last_scroll_time = 0  # Clear debounce

        initial_offset = manager.scroll_offset
        manager._on_page_down()

        # Should not exceed max offset
        max_offset = max(0, len(urls) - visible)
        assert manager.scroll_offset <= max_offset

    @patch("apias.terminal_utils.detect_terminal_capabilities")
    def test_scroll_down_with_empty_tasks(self, mock_detect: MagicMock) -> None:
        """Verify _on_scroll_down handles empty task list gracefully."""
        # WHY: Empty task list is valid state (edge case)
        mock_detect.return_value = {"unicode": True, "colors": True}

        urls: List[str] = []  # Empty URL list
        manager = BatchTUIManager(urls=urls, no_tui=True)
        manager._last_scroll_time = 0  # Clear debounce

        # Should not raise
        manager._on_scroll_down()

        # Offset should remain at 0
        assert manager.scroll_offset == 0

    @patch("apias.terminal_utils.detect_terminal_capabilities")
    def test_get_visible_tasks_count_respects_terminal_height(
        self, mock_detect: MagicMock
    ) -> None:
        """Verify _get_visible_tasks_count adapts to terminal height."""
        # WHY: Visible count must be calculated dynamically for different terminals
        mock_detect.return_value = {"unicode": True, "colors": True}

        urls = ["https://example.com"]
        manager = BatchTUIManager(urls=urls, no_tui=True)

        # Create a mock console size object for tall terminal
        mock_size_tall = MagicMock()
        mock_size_tall.height = 50
        mock_size_tall.width = 120

        # Patch the console.size property to return our mock
        with patch.object(
            type(manager.console),
            "size",
            new_callable=PropertyMock,
            return_value=mock_size_tall,
        ):
            visible_count_tall = manager._get_visible_tasks_count()

        # Create mock for shorter terminal
        mock_size_short = MagicMock()
        mock_size_short.height = 20
        mock_size_short.width = 120

        with patch.object(
            type(manager.console),
            "size",
            new_callable=PropertyMock,
            return_value=mock_size_short,
        ):
            visible_count_short = manager._get_visible_tasks_count()

        # Taller terminal should show more tasks
        assert visible_count_tall >= visible_count_short
        assert visible_count_tall >= 1
        assert visible_count_short >= 1

    @patch("apias.terminal_utils.detect_terminal_capabilities")
    def test_scroll_debounce_after_threshold_passes(
        self, mock_detect: MagicMock
    ) -> None:
        """Verify scroll is allowed after debounce threshold time passes."""
        # WHY: Debounce should only block rapid consecutive scrolls, not all scrolls
        mock_detect.return_value = {"unicode": True, "colors": True}

        urls = [f"https://example.com/{i}" for i in range(20)]
        manager = BatchTUIManager(urls=urls, no_tui=True)

        # First scroll - should NOT debounce
        result1 = manager._should_debounce_scroll()
        assert result1 is False

        # Immediate second call - SHOULD debounce
        result2 = manager._should_debounce_scroll()
        assert result2 is True

        # Simulate time passing beyond debounce threshold
        # SCROLL_DEBOUNCE_SECONDS is typically 0.05-0.1 seconds
        time.sleep(0.15)

        # Third call after delay - should NOT debounce
        result3 = manager._should_debounce_scroll()
        assert result3 is False


# =============================================================================
# Category 11: Extended Summary Methods Tests (15 tests)
# =============================================================================


class TestShowFinalSummaryExtended:
    """Extended tests for show_final_summary method covering various task states."""

    @patch("apias.terminal_utils.detect_terminal_capabilities")
    def test_show_final_summary_with_all_completed(
        self, mock_detect: MagicMock
    ) -> None:
        """Verify show_final_summary displays 100% success when all tasks complete."""
        # WHY: Success case should show 100% rate and no failed URLs section
        mock_detect.return_value = {"unicode": True, "colors": True}

        urls = [
            "https://example.com/1",
            "https://example.com/2",
            "https://example.com/3",
        ]
        manager = BatchTUIManager(urls=urls, no_tui=True)

        # Set all tasks to COMPLETE state
        for _task_id, task in manager.tasks.items():
            task.state = URLState.COMPLETE
            task.progress_pct = 100.0
            task.size_in = 5120
            task.size_out = 2048
            task.cost = 0.001
            task.duration = 5.0

        # Update stats to reflect completed state
        manager.stats.completed = 3
        manager.stats.pending = 0
        manager.stats.failed = 0
        manager.stats.total_cost = 0.003

        # Capture stdout
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            manager.show_final_summary(output_dir="/tmp/output")

        output = captured.getvalue()
        assert "BATCH PROCESSING COMPLETE" in output
        assert "3/3" in output  # All completed
        assert "100.0%" in output  # 100% success rate
        assert "Failed: 0" in output

    @patch("apias.terminal_utils.detect_terminal_capabilities")
    def test_show_final_summary_with_some_failed(self, mock_detect: MagicMock) -> None:
        """Verify show_final_summary displays partial success with failed URLs."""
        # WHY: Mixed results must show accurate failure count and success rate
        mock_detect.return_value = {"unicode": True, "colors": True}

        urls = [
            "https://example.com/1",
            "https://example.com/2",
            "https://example.com/3",
            "https://example.com/4",
        ]
        manager = BatchTUIManager(urls=urls, no_tui=True)

        # Set mixed task states
        manager.tasks[1].state = URLState.COMPLETE
        manager.tasks[1].progress_pct = 100.0
        manager.tasks[1].cost = 0.001
        manager.tasks[2].state = URLState.COMPLETE
        manager.tasks[2].progress_pct = 100.0
        manager.tasks[2].cost = 0.001
        manager.tasks[3].state = URLState.FAILED
        manager.tasks[3].error = "API rate limit exceeded"
        manager.tasks[4].state = URLState.FAILED
        manager.tasks[4].error = "Network timeout"

        # Update stats
        manager.stats.completed = 2
        manager.stats.pending = 0
        manager.stats.failed = 2
        manager.stats.total_cost = 0.002

        # Capture stdout
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            manager.show_final_summary(output_dir="/tmp/output")

        output = captured.getvalue()
        assert "BATCH PROCESSING COMPLETE" in output
        assert "2/4" in output  # 2 of 4 completed
        assert "50.0%" in output  # 50% success rate
        assert "Failed: 2" in output

    @patch("apias.terminal_utils.detect_terminal_capabilities")
    def test_show_final_summary_with_all_failed(self, mock_detect: MagicMock) -> None:
        """Verify show_final_summary handles 0% success rate correctly."""
        # WHY: All failures must be handled without division errors
        mock_detect.return_value = {"unicode": True, "colors": True}

        urls = ["https://example.com/1", "https://example.com/2"]
        manager = BatchTUIManager(urls=urls, no_tui=True)

        # Set all tasks to FAILED state
        for _task_id, task in manager.tasks.items():
            task.state = URLState.FAILED
            task.error = "Connection refused"
            task.progress_pct = 0.0

        # Update stats
        manager.stats.completed = 0
        manager.stats.pending = 0
        manager.stats.failed = 2
        manager.stats.total_cost = 0.0

        # Capture stdout
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            manager.show_final_summary(output_dir="/tmp/output")

        output = captured.getvalue()
        assert "BATCH PROCESSING COMPLETE" in output
        assert "0/2" in output  # None completed
        assert "0.0%" in output  # 0% success rate
        assert "Failed: 2" in output

    @patch("apias.terminal_utils.detect_terminal_capabilities")
    def test_show_final_summary_calculates_totals(self, mock_detect: MagicMock) -> None:
        """Verify show_final_summary correctly accumulates total cost from stats."""
        # WHY: Total cost tracking is critical for billing awareness
        mock_detect.return_value = {"unicode": True, "colors": True}

        urls = [
            "https://example.com/1",
            "https://example.com/2",
            "https://example.com/3",
        ]
        manager = BatchTUIManager(urls=urls, no_tui=True)

        # Set completed tasks with different costs
        manager.tasks[1].state = URLState.COMPLETE
        manager.tasks[1].cost = 0.0012
        manager.tasks[1].size_in = 10240
        manager.tasks[1].size_out = 4096
        manager.tasks[2].state = URLState.COMPLETE
        manager.tasks[2].cost = 0.0023
        manager.tasks[2].size_in = 15360
        manager.tasks[2].size_out = 6144
        manager.tasks[3].state = URLState.COMPLETE
        manager.tasks[3].cost = 0.0015
        manager.tasks[3].size_in = 8192
        manager.tasks[3].size_out = 3072

        # Update stats with total cost
        manager.stats.completed = 3
        manager.stats.pending = 0
        manager.stats.failed = 0
        manager.stats.total_cost = 0.0050  # Accumulated cost

        # Capture stdout
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            manager.show_final_summary(output_dir="/tmp/output")

        output = captured.getvalue()
        # Verify total cost is displayed
        assert "$0.00500" in output or "0.00500" in output
        assert "BATCH PROCESSING COMPLETE" in output

    @patch("apias.terminal_utils.detect_terminal_capabilities")
    def test_show_final_summary_quiet_mode(self, mock_detect: MagicMock) -> None:
        """Verify show_final_summary in quiet mode outputs minimal text."""
        # WHY: quiet mode should suppress verbose output but still show summary
        mock_detect.return_value = {"unicode": True, "colors": True}

        urls = ["https://example.com/1"]
        # quiet=True implies no_tui=True
        manager = BatchTUIManager(urls=urls, quiet=True)

        manager.tasks[1].state = URLState.COMPLETE
        manager.stats.completed = 1
        manager.stats.pending = 0
        manager.stats.total_cost = 0.001

        # Capture stdout
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            manager.show_final_summary(output_dir="/tmp/output")

        output = captured.getvalue()
        # Should still show basic summary info even in quiet mode
        assert "BATCH PROCESSING COMPLETE" in output
        assert "1/1" in output


class TestPromptRetryFailedExtended:
    """Extended tests for prompt_retry_failed method."""

    @patch("apias.terminal_utils.detect_terminal_capabilities")
    def test_prompt_retry_failed_returns_list(self, mock_detect: MagicMock) -> None:
        """Verify prompt_retry_failed returns a list of URLs when user confirms."""
        # WHY: Return type must be list for iteration in retry logic
        mock_detect.return_value = {"unicode": True, "colors": True}

        urls = ["https://example.com/1", "https://example.com/2"]
        manager = BatchTUIManager(urls=urls, no_tui=False)

        # Set one task as failed
        manager.tasks[2].state = URLState.FAILED
        manager.tasks[2].error = "API error"

        # Mock stdin to simulate user typing 'y'
        with patch("builtins.input", return_value="y"):
            with patch.object(manager.console, "print"):  # Suppress output
                with patch("sys.stdin") as mock_stdin:
                    mock_stdin.isatty.return_value = True
                    result = manager.prompt_retry_failed()

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == "https://example.com/2"

    @patch("apias.terminal_utils.detect_terminal_capabilities")
    def test_prompt_retry_failed_no_failed_returns_empty(
        self, mock_detect: MagicMock
    ) -> None:
        """Verify prompt_retry_failed returns empty list when no tasks failed."""
        # WHY: No failures means nothing to retry
        mock_detect.return_value = {"unicode": True, "colors": True}

        urls = ["https://example.com/1", "https://example.com/2"]
        manager = BatchTUIManager(urls=urls, no_tui=False)

        # All tasks completed successfully
        for task in manager.tasks.values():
            task.state = URLState.COMPLETE

        result = manager.prompt_retry_failed()

        assert isinstance(result, list)
        assert len(result) == 0

    @patch("apias.terminal_utils.detect_terminal_capabilities")
    def test_prompt_retry_failed_no_tui_returns_empty(
        self, mock_detect: MagicMock
    ) -> None:
        """Verify prompt_retry_failed returns empty list in no_tui mode."""
        # WHY: Non-interactive mode should not prompt and return empty
        mock_detect.return_value = {"unicode": True, "colors": True}

        urls = ["https://example.com/1"]
        manager = BatchTUIManager(urls=urls, no_tui=True)

        # Set task as failed
        manager.tasks[1].state = URLState.FAILED
        manager.tasks[1].error = "Connection error"

        result = manager.prompt_retry_failed()

        # Should return empty list without prompting
        assert isinstance(result, list)
        assert len(result) == 0

    @patch("apias.terminal_utils.detect_terminal_capabilities")
    def test_prompt_retry_failed_with_multiple_failures(
        self, mock_detect: MagicMock
    ) -> None:
        """Verify prompt_retry_failed returns all failed URLs when user confirms."""
        # WHY: Multiple failures must all be included in retry list
        mock_detect.return_value = {"unicode": True, "colors": True}

        urls = [
            "https://example.com/1",
            "https://example.com/2",
            "https://example.com/3",
            "https://example.com/4",
        ]
        manager = BatchTUIManager(urls=urls, no_tui=False)

        # Set multiple tasks as failed
        manager.tasks[1].state = URLState.COMPLETE
        manager.tasks[2].state = URLState.FAILED
        manager.tasks[2].error = "Rate limit"
        manager.tasks[3].state = URLState.FAILED
        manager.tasks[3].error = "Timeout"
        manager.tasks[4].state = URLState.FAILED
        manager.tasks[4].error = "Server error"

        # Mock stdin to simulate user typing 'y'
        with patch("builtins.input", return_value="y"):
            with patch.object(manager.console, "print"):  # Suppress output
                with patch("sys.stdin") as mock_stdin:
                    mock_stdin.isatty.return_value = True
                    result = manager.prompt_retry_failed()

        assert isinstance(result, list)
        assert len(result) == 3
        assert "https://example.com/2" in result
        assert "https://example.com/3" in result
        assert "https://example.com/4" in result


class TestShowCircuitBreakerDialogExtended:
    """Extended tests for show_circuit_breaker_dialog method."""

    @patch("apias.terminal_utils.detect_terminal_capabilities")
    def test_show_circuit_breaker_dialog_executes(self, mock_detect: MagicMock) -> None:
        """Verify show_circuit_breaker_dialog executes without error in no_tui mode."""
        # WHY: Circuit breaker must be displayable regardless of terminal mode
        mock_detect.return_value = {"unicode": True, "colors": True}

        urls = [
            "https://example.com/1",
            "https://example.com/2",
            "https://example.com/3",
        ]
        manager = BatchTUIManager(urls=urls, no_tui=True)

        manager.stats.completed = 1
        manager.stats.failed = 2

        # Capture stdout
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            manager.show_circuit_breaker_dialog(
                trigger_reason="API quota exceeded", output_dir="/tmp/output"
            )

        output = captured.getvalue()
        assert "PROCESSING STOPPED" in output
        assert "API quota exceeded" in output

    @patch("apias.terminal_utils.detect_terminal_capabilities")
    def test_show_circuit_breaker_dialog_no_tui_prints_simple(
        self, mock_detect: MagicMock
    ) -> None:
        """Verify show_circuit_breaker_dialog uses simple text in no_tui mode."""
        # WHY: no_tui mode must produce readable text without Rich formatting
        mock_detect.return_value = {"unicode": True, "colors": True}

        urls = ["https://example.com/1", "https://example.com/2"]
        manager = BatchTUIManager(urls=urls, no_tui=True)

        manager.stats.completed = 1
        manager.stats.total_urls = 2

        # Capture stdout
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            manager.show_circuit_breaker_dialog(
                trigger_reason="Network failure", output_dir="/tmp/output_dir"
            )

        output = captured.getvalue()
        assert "PROCESSING STOPPED" in output
        assert "Network failure" in output
        assert "1/2" in output
        assert "resume" in output.lower()
        assert "/tmp/output_dir" in output

    @patch("apias.terminal_utils.detect_terminal_capabilities")
    def test_show_circuit_breaker_dialog_with_reason(
        self, mock_detect: MagicMock
    ) -> None:
        """Verify show_circuit_breaker_dialog displays the trigger reason correctly."""
        # WHY: Failure reason must be clearly visible for user troubleshooting
        mock_detect.return_value = {"unicode": True, "colors": True}

        urls = ["https://example.com/1"]
        manager = BatchTUIManager(urls=urls, no_tui=True)

        manager.stats.completed = 0
        manager.stats.failed = 1
        manager.stats.total_urls = 1

        trigger_reason = "OpenAI API key invalid or expired"

        # Capture stdout
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            manager.show_circuit_breaker_dialog(
                trigger_reason=trigger_reason, output_dir="/tmp/test_out"
            )

        output = captured.getvalue()
        assert trigger_reason in output
        assert "PROCESSING STOPPED" in output


class TestShowSessionSummaryExtended:
    """Extended tests for show_session_summary method."""

    @patch("apias.terminal_utils.detect_terminal_capabilities")
    def test_show_session_summary_executes(self, mock_detect: MagicMock) -> None:
        """Verify show_session_summary executes without errors in no_tui mode."""
        # WHY: Session summary must complete without raising exceptions
        mock_detect.return_value = {"unicode": True, "colors": True}

        urls = ["https://example.com/1"]
        manager = BatchTUIManager(urls=urls, no_tui=True)

        # Capture stdout
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            result = manager.show_session_summary(
                session_log_path="/tmp/session.log", error_count=0
            )

        # Method returns None
        assert result is None
        output = captured.getvalue()
        assert "session.log" in output

    @patch("apias.terminal_utils.detect_terminal_capabilities")
    def test_show_session_summary_with_stats(self, mock_detect: MagicMock) -> None:
        """Verify show_session_summary displays stats with error count."""
        # WHY: Error count must be visible for user awareness
        mock_detect.return_value = {"unicode": True, "colors": True}

        urls = ["https://example.com/1", "https://example.com/2"]
        manager = BatchTUIManager(urls=urls, no_tui=True)

        manager.stats.completed = 1
        manager.stats.failed = 1

        # Capture stdout
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            manager.show_session_summary(
                session_log_path="/home/user/output/session.log", error_count=5
            )

        output = captured.getvalue()
        assert "session.log" in output
        assert "5" in output  # Error count should appear
        assert "error" in output.lower()

    @patch("apias.terminal_utils.detect_terminal_capabilities")
    def test_show_session_summary_quiet_mode(self, mock_detect: MagicMock) -> None:
        """Verify show_session_summary respects quiet flag (via no_tui)."""
        # WHY: quiet mode implies no_tui and should use simple output
        mock_detect.return_value = {"unicode": True, "colors": True}

        urls = ["https://example.com/1"]
        # quiet=True implies no_tui=True
        manager = BatchTUIManager(urls=urls, quiet=True)

        manager.stats.completed = 1

        # Capture stdout
        captured = io.StringIO()
        with patch("sys.stdout", captured):
            manager.show_session_summary(
                session_log_path="/var/log/session.log", error_count=0
            )

        output = captured.getvalue()
        # Should output session log path even in quiet mode
        assert "session.log" in output
