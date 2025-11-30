"""
Tests for APIAS TUI components.

Tests terminal utilities, base TUI manager, and state management.
"""

import threading
import time

import pytest

from apias.terminal_utils import (
    BaseTUIManager,
    ProcessState,
    Symbols,
    calculate_eta,
    format_duration,
    format_size,
    truncate_url,
)


class TestFormatDuration:
    """Tests for format_duration utility function."""

    def test_format_zero_seconds(self) -> None:
        """Zero seconds formats with one decimal place."""
        # format_duration uses f"{seconds:.1f}s" for sub-minute durations
        assert format_duration(0) == "0.0s"

    def test_format_seconds_only(self) -> None:
        """Durations under 60s show seconds with one decimal place."""
        assert format_duration(45) == "45.0s"
        assert format_duration(1) == "1.0s"

    def test_format_minutes_and_seconds(self) -> None:
        """Durations under 1h show minutes and integer seconds."""
        assert format_duration(65) == "1m 5s"
        assert format_duration(120) == "2m 0s"
        assert format_duration(3599) == "59m 59s"

    def test_format_hours_minutes(self) -> None:
        """Durations over 1h show hours and minutes (no seconds)."""
        # format_duration only shows hours and minutes for durations >= 1h
        assert format_duration(3600) == "1h 0m"
        assert format_duration(3661) == "1h 1m"
        assert format_duration(7325) == "2h 2m"

    def test_format_fractional_seconds(self) -> None:
        """Fractional seconds are shown with one decimal place."""
        assert format_duration(45.7) == "45.7s"
        assert format_duration(65.9) == "1m 5s"  # Minutes mode uses int()

    def test_negative_seconds_raises_value_error(self) -> None:
        """Negative seconds raise ValueError (fail-fast on bug detection)."""
        # WHY: Negative durations indicate clock skew, race condition, or logic error
        # The fail-fast approach ensures bugs are caught immediately
        with pytest.raises(ValueError, match="negative seconds"):
            format_duration(-1)
        with pytest.raises(ValueError, match="negative seconds"):
            format_duration(-0.1)
        with pytest.raises(ValueError, match="negative seconds"):
            format_duration(-3600)


class TestFormatSize:
    """Tests for format_size utility function."""

    def test_format_bytes(self) -> None:
        """Sizes under 1KB show bytes."""
        # Note: format_size uses compact format without spaces (e.g., "0B" not "0 B")
        assert format_size(0) == "0B"
        assert format_size(512) == "512B"
        assert format_size(1023) == "1023B"

    def test_format_kilobytes(self) -> None:
        """Sizes in KB range format correctly."""
        assert format_size(1024) == "1.0KB"
        assert format_size(1536) == "1.5KB"
        assert format_size(10240) == "10.0KB"

    def test_format_megabytes(self) -> None:
        """Sizes in MB range format correctly."""
        assert format_size(1048576) == "1.0MB"
        assert format_size(5242880) == "5.0MB"

    def test_format_gigabytes(self) -> None:
        """Sizes in GB range format correctly."""
        assert format_size(1073741824) == "1.0GB"

    def test_negative_bytes_raises_value_error(self) -> None:
        """Negative bytes raise ValueError (fail-fast on bug detection)."""
        # WHY: Negative sizes indicate data corruption, integer overflow, or logic error
        # The fail-fast approach ensures bugs are caught immediately
        with pytest.raises(ValueError, match="negative bytes_size"):
            format_size(-1)
        with pytest.raises(ValueError, match="negative bytes_size"):
            format_size(-1024)
        with pytest.raises(ValueError, match="negative bytes_size"):
            format_size(-1048576)


class TestCalculateEta:
    """Tests for calculate_eta - verifies it NEVER returns negative values."""

    def test_eta_returns_none_for_zero_progress(self) -> None:
        """ETA is None when progress is 0% (can't estimate)."""
        result = calculate_eta(0, 10.0)
        assert result is None

    def test_eta_returns_none_for_100_percent(self) -> None:
        """ETA is None when progress is 100% (already done)."""
        result = calculate_eta(100, 10.0)
        assert result is None

    def test_eta_returns_none_for_negative_progress(self) -> None:
        """ETA is None for invalid negative progress."""
        result = calculate_eta(-5, 10.0)
        assert result is None

    def test_eta_normal_calculation(self) -> None:
        """ETA calculates correctly for normal progress."""
        # 50% done in 10 seconds = 10 more seconds expected
        result = calculate_eta(50, 10.0)
        assert result is not None
        assert abs(result - 10.0) < 0.01  # Allow small float error

    def test_eta_never_returns_negative(self) -> None:
        """ETA output is ALWAYS >= 0 (critical invariant for format_duration)."""
        # This test verifies the source never generates negatives
        test_cases = [
            (1, 0.1),  # Very early progress
            (50, 10),  # Normal progress
            (99, 100),  # Almost done
            (99.9, 1000),  # Nearly complete
            (0.1, 0.001),  # Tiny progress, tiny elapsed
        ]
        for progress, elapsed in test_cases:
            result = calculate_eta(progress, elapsed)
            if result is not None:
                assert (
                    result >= 0
                ), f"calculate_eta({progress}, {elapsed}) returned negative: {result}"

    def test_eta_with_edge_case_values(self) -> None:
        """ETA handles edge cases without producing negatives."""
        # Very small progress - should return large but non-negative ETA
        result = calculate_eta(0.001, 1.0)
        if result is not None:
            assert result >= 0
            assert result > 99000  # Should be a very large estimate

        # Progress very close to 100
        result = calculate_eta(99.999, 100.0)
        if result is not None:
            assert result >= 0
            assert result < 0.01  # Should be nearly zero

    def test_negative_elapsed_raises_value_error(self) -> None:
        """Negative elapsed time raises ValueError (fail-fast on bug detection)."""
        # WHY: Negative elapsed indicates clock skew, pause tracking bug, or logic error
        # The fail-fast approach ensures bugs are caught immediately
        with pytest.raises(ValueError, match="negative elapsed_seconds"):
            calculate_eta(50, -1.0)
        with pytest.raises(ValueError, match="negative elapsed_seconds"):
            calculate_eta(25, -0.001)
        with pytest.raises(ValueError, match="negative elapsed_seconds"):
            calculate_eta(99, -100.0)


class TestTruncateUrl:
    """Tests for truncate_url utility function."""

    def test_short_url_unchanged(self) -> None:
        """URLs shorter than max_length are unchanged."""
        url = "https://example.com"
        assert truncate_url(url, 50) == url

    def test_long_url_truncated(self) -> None:
        """URLs longer than max_length are truncated with ellipsis in middle."""
        url = "https://example.com/very/long/path/to/some/resource"
        result = truncate_url(url, 30)
        # Smart truncation keeps beginning and end of path with ... in middle
        assert "..." in result
        # Result should be approximately max_length (smart truncation may vary slightly)
        assert len(result) <= 35  # Allow some flexibility for smart truncation

    def test_exact_length_url(self) -> None:
        """URLs exactly max_length are unchanged."""
        url = "https://example.com/path"
        assert truncate_url(url, len(url)) == url


class TestSymbols:
    """Tests for Symbols class constants."""

    def test_spinner_frames_are_strings(self) -> None:
        """Spinner frames are non-empty string lists."""
        # Use get_spinner_frames() method to get the appropriate frames
        frames = Symbols.get_spinner_frames()
        assert isinstance(frames, list)
        assert all(isinstance(f, str) for f in frames)
        assert len(frames) > 0

    def test_status_symbols_defined(self) -> None:
        """Status indicator symbols are defined as tuples (unicode, ascii)."""
        # Symbols are defined as tuples (unicode_version, ascii_version)
        assert isinstance(Symbols.COMPLETE, tuple)
        assert isinstance(Symbols.FAILED, tuple)
        assert isinstance(Symbols.RETRY, tuple)
        assert isinstance(Symbols.PENDING, tuple)
        assert isinstance(Symbols.PROCESSING, tuple)


class TestProcessState:
    """Tests for ProcessState enum."""

    def test_all_states_defined(self) -> None:
        """All expected process states are defined."""
        # ProcessState uses auto() so values are integers, check by name
        state_names = [s.name for s in ProcessState]
        assert "WAITING" in state_names
        assert "RUNNING" in state_names
        assert "PAUSED" in state_names
        assert "STOPPED" in state_names

    def test_state_comparison(self) -> None:
        """Process states can be compared."""
        # Use variables to avoid mypy's literal type narrowing
        running: ProcessState = ProcessState.RUNNING
        paused: ProcessState = ProcessState.PAUSED
        waiting: ProcessState = ProcessState.WAITING
        assert running != paused
        assert waiting == ProcessState.WAITING


class TestBaseTUIManager:
    """Tests for BaseTUIManager class."""

    def test_init_default_values(self) -> None:
        """BaseTUIManager initializes with correct defaults."""
        manager = BaseTUIManager()
        assert manager.no_tui is False
        assert manager.quiet is False
        assert manager.process_state == ProcessState.WAITING

    def test_init_with_no_tui(self) -> None:
        """BaseTUIManager respects no_tui flag."""
        manager = BaseTUIManager(no_tui=True)
        assert manager.no_tui is True

    def test_init_quiet_implies_no_tui(self) -> None:
        """Setting quiet=True automatically sets no_tui=True."""
        manager = BaseTUIManager(quiet=True)
        assert manager.quiet is True
        assert manager.no_tui is True

    def test_process_state_property_thread_safe(self) -> None:
        """Process state property uses lock for thread safety."""
        manager = BaseTUIManager()
        results: list[ProcessState] = []

        def read_state() -> None:
            for _ in range(100):
                results.append(manager.process_state)

        threads = [threading.Thread(target=read_state) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All reads should succeed without errors
        assert len(results) == 500
        assert all(s == ProcessState.WAITING for s in results)

    def test_set_running_state(self) -> None:
        """Can transition from WAITING to RUNNING using property setter."""
        manager = BaseTUIManager()
        # Use the property setter to change state
        manager.process_state = ProcessState.RUNNING
        assert manager.process_state == ProcessState.RUNNING

    def test_pause_from_running(self) -> None:
        """Can transition from RUNNING to PAUSED."""
        manager = BaseTUIManager()
        manager.process_state = ProcessState.RUNNING
        manager.process_state = ProcessState.PAUSED
        assert manager.process_state == ProcessState.PAUSED

    def test_resume_from_paused(self) -> None:
        """Can transition from PAUSED back to RUNNING."""
        manager = BaseTUIManager()
        manager.process_state = ProcessState.RUNNING
        manager.process_state = ProcessState.PAUSED
        manager.process_state = ProcessState.RUNNING
        assert manager.process_state == ProcessState.RUNNING

    def test_stop_from_any_state(self) -> None:
        """Can transition to STOPPED from any state."""
        for initial_state in [
            ProcessState.WAITING,
            ProcessState.RUNNING,
            ProcessState.PAUSED,
        ]:
            manager = BaseTUIManager()
            manager.process_state = initial_state
            manager.process_state = ProcessState.STOPPED
            assert manager.process_state == ProcessState.STOPPED

    def test_is_paused_property(self) -> None:
        """is_paused property returns correct value."""
        manager = BaseTUIManager()
        assert manager.is_paused is False

        manager.process_state = ProcessState.RUNNING
        assert manager.is_paused is False

        manager.process_state = ProcessState.PAUSED
        assert manager.is_paused is True

    def test_is_running_property(self) -> None:
        """is_running property returns correct value for different states."""
        # Test WAITING state
        manager1 = BaseTUIManager()
        assert manager1.is_running is False

        # Test RUNNING state
        manager2 = BaseTUIManager()
        manager2.process_state = ProcessState.RUNNING
        assert manager2.is_running is True

        # Test PAUSED state
        manager3 = BaseTUIManager()
        manager3.process_state = ProcessState.PAUSED
        assert manager3.is_running is False

    def test_pause_time_tracking_via_on_space(self) -> None:
        """Pause duration is tracked correctly via _on_space_pressed."""
        manager = BaseTUIManager()
        # First space press: WAITING -> RUNNING
        manager._on_space_pressed()
        current_state: ProcessState = manager.process_state
        assert current_state == ProcessState.RUNNING

        # Second space press: RUNNING -> PAUSED (starts pause timer)
        manager._on_space_pressed()
        current_state = manager.process_state
        assert current_state == ProcessState.PAUSED
        time.sleep(0.1)

        # Third space press: PAUSED -> RUNNING (adds pause duration)
        manager._on_space_pressed()
        current_state = manager.process_state
        assert current_state == ProcessState.RUNNING

        # Check that pause duration was recorded
        assert manager._total_pause_duration >= 0.1

    def test_get_effective_elapsed_subtracts_pause(self) -> None:
        """get_effective_elapsed subtracts pause duration from total elapsed."""
        from datetime import datetime

        manager = BaseTUIManager()
        start_time = datetime.now()

        # First space: WAITING -> RUNNING
        manager._on_space_pressed()
        time.sleep(0.05)

        # Second space: RUNNING -> PAUSED
        manager._on_space_pressed()
        time.sleep(0.1)

        # Third space: PAUSED -> RUNNING
        manager._on_space_pressed()
        time.sleep(0.05)

        effective = manager.get_effective_elapsed(start_time)
        total_elapsed = (datetime.now() - start_time).total_seconds()

        # Effective should be less than total by approximately the pause duration
        assert effective < total_elapsed
        assert effective < 0.15  # Should be around 0.1s (excluding pause)


class TestBaseTUIManagerConcurrency:
    """Concurrency tests for BaseTUIManager."""

    def test_concurrent_state_transitions(self) -> None:
        """Multiple threads can safely transition states using property setter."""
        manager = BaseTUIManager()
        manager.process_state = ProcessState.RUNNING
        errors: list[Exception] = []

        def toggle_pause() -> None:
            try:
                for _ in range(50):
                    if manager.process_state == ProcessState.RUNNING:
                        manager.process_state = ProcessState.PAUSED
                    elif manager.process_state == ProcessState.PAUSED:
                        manager.process_state = ProcessState.RUNNING
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=toggle_pause) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No errors should occur
        assert len(errors) == 0
        # State should be valid
        assert manager.process_state in [ProcessState.RUNNING, ProcessState.PAUSED]

    def test_concurrent_elapsed_calculation(self) -> None:
        """Multiple threads can safely calculate elapsed time."""
        from datetime import datetime

        manager = BaseTUIManager()
        manager.process_state = ProcessState.RUNNING
        start_time = datetime.now()
        results: list[float] = []

        def get_elapsed() -> None:
            for _ in range(100):
                results.append(manager.get_effective_elapsed(start_time))

        threads = [threading.Thread(target=get_elapsed) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All results should be valid positive numbers
        assert len(results) == 500
        assert all(r >= 0 for r in results)
