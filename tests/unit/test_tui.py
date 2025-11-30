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
                assert result >= 0, (
                    f"calculate_eta({progress}, {elapsed}) returned negative: {result}"
                )

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


# ==============================================================================
# NEW TESTS: 20 additional tests for coverage gaps
# ==============================================================================


class TestTerminalDetection:
    """Tests for terminal capability detection functions."""

    def test_detect_terminal_capabilities_returns_dataclass(self) -> None:
        """detect_terminal_capabilities returns a TerminalCapabilities dataclass."""
        from apias.terminal_utils import (
            TerminalCapabilities,
            detect_terminal_capabilities,
        )

        result = detect_terminal_capabilities()
        assert isinstance(result, TerminalCapabilities)

    def test_detect_terminal_capabilities_has_color_key(self) -> None:
        """TerminalCapabilities includes supports_colors attribute."""
        from apias.terminal_utils import detect_terminal_capabilities

        result = detect_terminal_capabilities()
        assert hasattr(result, "supports_colors")
        assert isinstance(result.supports_colors, bool)

    def test_detect_terminal_capabilities_has_unicode_key(self) -> None:
        """TerminalCapabilities includes supports_unicode attribute."""
        from apias.terminal_utils import detect_terminal_capabilities

        result = detect_terminal_capabilities()
        assert hasattr(result, "supports_unicode")
        assert isinstance(result.supports_unicode, bool)

    def test_is_terminal_returns_bool(self) -> None:
        """TerminalCapabilities.is_interactive is a boolean."""
        from apias.terminal_utils import detect_terminal_capabilities

        result = detect_terminal_capabilities()
        assert hasattr(result, "is_interactive")
        assert isinstance(result.is_interactive, bool)


class TestSpinnerAnimation:
    """Tests for spinner/animation frame retrieval."""

    def test_get_spinner_frames_returns_list(self) -> None:
        """get_spinner_frames returns a list of strings."""
        frames = Symbols.get_spinner_frames()
        assert isinstance(frames, list)

    def test_spinner_frames_non_empty(self) -> None:
        """Spinner frames list is never empty."""
        frames = Symbols.get_spinner_frames()
        assert len(frames) > 0

    def test_ascii_spinner_fallback(self) -> None:
        """ASCII spinner uses pipe/slash/hyphen/backslash characters."""
        # Set ASCII mode
        Symbols.set_ascii_mode(True)
        frames = Symbols.get_spinner_frames()
        # ASCII spinner frames should be |, /, -, \
        assert frames == ["|", "/", "-", "\\"]
        # Reset to default
        Symbols.set_ascii_mode(False)

    def test_unicode_spinner_available(self) -> None:
        """Unicode spinner uses braille-like characters."""
        # Set Unicode mode
        Symbols.set_ascii_mode(False)
        frames = Symbols.get_spinner_frames()
        # Unicode spinner frames should be braille patterns
        assert len(frames) == 10
        assert all(ord(f) > 127 for f in frames)  # All are non-ASCII


class TestBaseTUIManagerAdvanced:
    """Advanced tests for BaseTUIManager functionality."""

    def test_base_tui_stop_from_running(self) -> None:
        """request_stop transitions from RUNNING to STOPPED."""
        manager = BaseTUIManager()
        manager.process_state = ProcessState.RUNNING
        manager.request_stop()
        assert manager.process_state == ProcessState.STOPPED

    def test_base_tui_process_state_setter_thread_safe(self) -> None:
        """Process state setter uses lock for thread safety."""
        manager = BaseTUIManager()
        errors: list[Exception] = []

        def write_state() -> None:
            try:
                for _ in range(100):
                    manager.process_state = ProcessState.RUNNING
                    manager.process_state = ProcessState.PAUSED
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=write_state) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No errors should occur from concurrent writes
        assert len(errors) == 0
        # State should be valid
        assert manager.process_state in [ProcessState.RUNNING, ProcessState.PAUSED]

    def test_base_tui_total_pause_duration_accumulates(self) -> None:
        """Total pause duration accumulates across multiple pause/resume cycles."""
        manager = BaseTUIManager()
        # Start running
        manager._on_space_pressed()  # WAITING -> RUNNING

        # First pause cycle
        manager._on_space_pressed()  # RUNNING -> PAUSED
        time.sleep(0.05)
        manager._on_space_pressed()  # PAUSED -> RUNNING
        first_pause = manager._total_pause_duration

        # Second pause cycle
        manager._on_space_pressed()  # RUNNING -> PAUSED
        time.sleep(0.05)
        manager._on_space_pressed()  # PAUSED -> RUNNING

        # Total should be greater than first pause alone
        assert manager._total_pause_duration > first_pause
        assert manager._total_pause_duration >= 0.1

    def test_base_tui_pause_start_tracked(self) -> None:
        """Pause start time is tracked when entering PAUSED state."""
        manager = BaseTUIManager()
        manager._on_space_pressed()  # WAITING -> RUNNING
        assert manager._pause_start_time is None

        manager._on_space_pressed()  # RUNNING -> PAUSED
        assert manager._pause_start_time is not None
        assert manager._pause_start_time > 0

    def test_base_tui_effective_elapsed_never_negative(self) -> None:
        """get_effective_elapsed never returns negative values."""
        from datetime import datetime

        manager = BaseTUIManager()
        start_time = datetime.now()

        # Even with pause tracking, effective elapsed should never be negative
        manager._on_space_pressed()  # WAITING -> RUNNING
        manager._on_space_pressed()  # RUNNING -> PAUSED
        manager._on_space_pressed()  # PAUSED -> RUNNING

        effective = manager.get_effective_elapsed(start_time)
        assert effective >= 0

    def test_base_tui_on_space_from_stopped_no_change(self) -> None:
        """Pressing space when STOPPED does not change state."""
        manager = BaseTUIManager()
        manager.process_state = ProcessState.STOPPED

        manager._on_space_pressed()  # Should not change from STOPPED
        assert manager.process_state == ProcessState.STOPPED


class TestHelperFunctions:
    """Tests for utility helper functions."""

    def test_truncate_url_preserves_protocol(self) -> None:
        """Truncated URL preserves the protocol scheme."""
        url = "https://example.com/very/long/path/to/some/deeply/nested/resource/page.html"
        result = truncate_url(url, 50)
        assert result.startswith("https://")

    def test_truncate_url_handles_empty_string(self) -> None:
        """Empty URL returns empty string."""
        result = truncate_url("", 50)
        assert result == ""

    def test_format_duration_large_values(self) -> None:
        """format_duration handles large hour values correctly."""
        # 10 hours, 30 minutes
        assert format_duration(37800) == "10h 30m"
        # 100 hours, 0 minutes
        assert format_duration(360000) == "100h 0m"
        # 24 hours exactly
        assert format_duration(86400) == "24h 0m"

    def test_format_size_terabytes(self) -> None:
        """format_size handles terabyte-range values (displays as GB)."""
        # 1 TB = 1024 GB
        one_tb = 1024 * 1024 * 1024 * 1024
        result = format_size(one_tb)
        # Should display as 1024.0GB since there's no TB format
        assert result == "1024.0GB"

    def test_calculate_eta_high_progress(self) -> None:
        """calculate_eta returns small values for high progress percentages."""
        # 99% done in 99 seconds = about 1 second remaining
        result = calculate_eta(99, 99.0)
        assert result is not None
        assert result < 2.0  # Should be approximately 1 second
        assert result >= 0

    def test_symbols_has_arrow_symbols(self) -> None:
        """Symbols class defines arrow symbols as tuples."""
        assert hasattr(Symbols, "ARROW_UP")
        assert hasattr(Symbols, "ARROW_DOWN")
        assert isinstance(Symbols.ARROW_UP, tuple)
        assert isinstance(Symbols.ARROW_DOWN, tuple)
        # Each tuple should have 2 elements (unicode, ascii)
        assert len(Symbols.ARROW_UP) == 2
        assert len(Symbols.ARROW_DOWN) == 2


# ==============================================================================
# NEW TESTS: 15 additional tests for rendering coverage gaps
# Added 2025-11-30 to cover lines 442-750 in terminal_utils.py
# ==============================================================================


class TestSymbolsProgressBar:
    """Tests for Symbols.make_progress_bar rendering method."""

    def test_make_progress_bar_zero_percent(self) -> None:
        """Progress bar at 0% shows all empty characters."""
        bar = Symbols.make_progress_bar(0, width=10)
        # At 0%, all characters should be empty bar character
        assert len(bar) == 10
        empty_char = Symbols.get(Symbols.BAR_EMPTY)
        assert bar == empty_char * 10

    def test_make_progress_bar_hundred_percent(self) -> None:
        """Progress bar at 100% shows all filled characters."""
        bar = Symbols.make_progress_bar(100, width=10)
        assert len(bar) == 10
        filled_char = Symbols.get(Symbols.BAR_FILLED)
        assert bar == filled_char * 10

    def test_make_progress_bar_fifty_percent(self) -> None:
        """Progress bar at 50% shows half filled, half empty."""
        bar = Symbols.make_progress_bar(50, width=10)
        assert len(bar) == 10
        filled_char = Symbols.get(Symbols.BAR_FILLED)
        empty_char = Symbols.get(Symbols.BAR_EMPTY)
        # 50% of 10 = 5 filled chars
        assert bar == filled_char * 5 + empty_char * 5

    def test_make_progress_bar_clamps_over_hundred(self) -> None:
        """Progress bar clamps values over 100% to prevent overflow."""
        bar = Symbols.make_progress_bar(150, width=10)
        assert len(bar) == 10
        filled_char = Symbols.get(Symbols.BAR_FILLED)
        # Should clamp to 100% = all filled
        assert bar == filled_char * 10

    def test_make_progress_bar_clamps_negative(self) -> None:
        """Progress bar clamps negative values to 0%."""
        bar = Symbols.make_progress_bar(-50, width=10)
        assert len(bar) == 10
        empty_char = Symbols.get(Symbols.BAR_EMPTY)
        # Should clamp to 0% = all empty
        assert bar == empty_char * 10


class TestSymbolsPulseAnimation:
    """Tests for Symbols pulse animation methods (used for paused state)."""

    def test_get_pulse_frames_returns_list(self) -> None:
        """get_pulse_frames returns a non-empty list of strings."""
        frames = Symbols.get_pulse_frames()
        assert isinstance(frames, list)
        assert len(frames) > 0
        assert all(isinstance(f, str) for f in frames)

    def test_get_pulse_frame_returns_string(self) -> None:
        """get_pulse_frame returns a single character from pulse frames."""
        frame = Symbols.get_pulse_frame()
        assert isinstance(frame, str)
        assert len(frame) == 1
        # The returned frame should be in the list of pulse frames
        frames = Symbols.get_pulse_frames()
        assert frame in frames

    def test_ascii_pulse_frames(self) -> None:
        """ASCII pulse frames use simple characters."""
        Symbols.set_ascii_mode(True)
        frames = Symbols.get_pulse_frames()
        # ASCII pulse should be ["*", "o", ".", "o"]
        assert frames == ["*", "o", ".", "o"]
        # Reset to default
        Symbols.set_ascii_mode(False)


class TestKeyboardListenerSetup:
    """Tests for KeyboardListener initialization and callback registration."""

    def test_keyboard_listener_init_registers_instance(self) -> None:
        """KeyboardListener registers itself for atexit cleanup."""
        from apias.terminal_utils import KeyboardListener

        initial_count = len(KeyboardListener._all_instances)
        listener = KeyboardListener()
        assert len(KeyboardListener._all_instances) == initial_count + 1
        # Cleanup
        listener.stop()
        KeyboardListener._all_instances.remove(listener)

    def test_keyboard_listener_register_callback(self) -> None:
        """register_callback stores callbacks for key events."""
        from apias.terminal_utils import KeyboardListener

        listener = KeyboardListener()
        callback_called = []

        def test_callback() -> None:
            callback_called.append(True)

        listener.register_callback("space", test_callback)
        assert "space" in listener._callbacks
        # Trigger the callback manually
        listener._trigger_callback("space")
        assert len(callback_called) == 1
        # Cleanup
        listener.stop()
        KeyboardListener._all_instances.remove(listener)

    def test_keyboard_listener_callback_case_insensitive(self) -> None:
        """Callback registration is case-insensitive for key names."""
        from apias.terminal_utils import KeyboardListener

        listener = KeyboardListener()
        callback_called = []

        def test_callback() -> None:
            callback_called.append(True)

        # Register with uppercase
        listener.register_callback("SPACE", test_callback)
        # Should be stored as lowercase
        assert "space" in listener._callbacks
        # Cleanup
        listener.stop()
        KeyboardListener._all_instances.remove(listener)


class TestSymbolsGetMethod:
    """Tests for Symbols.get() method with different modes."""

    def test_symbols_get_unicode_mode(self) -> None:
        """In Unicode mode, get() returns the first tuple element."""
        Symbols.set_ascii_mode(False)
        result = Symbols.get(Symbols.COMPLETE)
        assert result == Symbols.COMPLETE[0]  # Unicode version
        assert result == "✅"

    def test_symbols_get_ascii_mode(self) -> None:
        """In ASCII mode, get() returns the second tuple element."""
        Symbols.set_ascii_mode(True)
        result = Symbols.get(Symbols.COMPLETE)
        assert result == Symbols.COMPLETE[1]  # ASCII version
        assert result == "[OK]"
        # Reset to default
        Symbols.set_ascii_mode(False)


class TestKeyboardListenerTriggerCallback:
    """Tests for KeyboardListener._trigger_callback error handling."""

    def test_trigger_callback_handles_exception(self) -> None:
        """_trigger_callback catches and logs exceptions from callbacks."""
        from apias.terminal_utils import KeyboardListener

        listener = KeyboardListener()

        def bad_callback() -> None:
            raise RuntimeError("Callback error")

        listener.register_callback("space", bad_callback)
        # Should not raise - exception is caught and logged
        listener._trigger_callback("space")
        # Cleanup
        listener.stop()
        KeyboardListener._all_instances.remove(listener)

    def test_trigger_callback_unregistered_key_no_error(self) -> None:
        """_trigger_callback does nothing for unregistered keys."""
        from apias.terminal_utils import KeyboardListener

        listener = KeyboardListener()
        # Should not raise for unregistered key
        listener._trigger_callback("nonexistent_key")
        # Cleanup
        listener.stop()
        KeyboardListener._all_instances.remove(listener)
