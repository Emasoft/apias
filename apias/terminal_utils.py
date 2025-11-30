"""
Terminal Utilities Module for cross-platform terminal handling.

Provides:
- Terminal capability detection (Unicode, colors, size)
- Cross-platform keyboard input handling (Unix/Windows)
- ASCII fallbacks for emoji/Unicode characters
- Terminal settings management and restoration
- Process state management (waiting, running, paused, stopped)
"""

import atexit
import locale
import logging
import os
import sys
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Dict, List, Optional, Tuple
from urllib.parse import urlparse

if TYPE_CHECKING:
    from datetime import datetime as DatetimeType

import threading
import time
from dataclasses import dataclass
from enum import Enum, auto

# Import centralized constants - single source of truth for configuration values
# DO NOT hardcode terminal/timing values here - add new constants to config.py
from apias.config import (
    DEFAULT_TERMINAL_HEIGHT,
    DEFAULT_TERMINAL_WIDTH,
    KEYBOARD_POLL_INTERVAL,
    KEYBOARD_THREAD_TIMEOUT,
    URL_TRUNCATE_MAX_LENGTH,
)

logger = logging.getLogger(__name__)


class ProcessState(Enum):
    """
    Process execution states for TUI managers.

    State transitions:
        WAITING -> RUNNING (first SPACE press)
        RUNNING <-> PAUSED (subsequent SPACE presses)
        Any state -> STOPPED (Ctrl+C only)
    """

    WAITING = auto()  # Before first SPACE press - ready to start
    RUNNING = auto()  # Actively processing
    PAUSED = auto()  # Temporarily halted - press SPACE to resume
    STOPPED = auto()  # Terminated by Ctrl+C - cannot resume


# Platform-specific imports
IS_WINDOWS = sys.platform == "win32"

if IS_WINDOWS:
    try:
        import msvcrt

        HAS_MSVCRT = True
    except ImportError:
        HAS_MSVCRT = False
    HAS_TERMIOS = False
else:
    try:
        import select
        import termios
        import tty

        HAS_TERMIOS = True
    except ImportError:
        HAS_TERMIOS = False
    HAS_MSVCRT = False


@dataclass
class TerminalCapabilities:
    """Detected terminal capabilities"""

    supports_unicode: bool = True
    supports_emoji: bool = True
    supports_colors: bool = True
    supports_256_colors: bool = True
    supports_true_color: bool = False
    is_interactive: bool = True
    # Use centralized defaults - DO NOT hardcode terminal dimensions
    width: int = DEFAULT_TERMINAL_WIDTH
    height: int = DEFAULT_TERMINAL_HEIGHT


class Symbols:
    """
    Terminal symbols with ASCII fallbacks.

    Use Symbols.get() to get the appropriate symbol based on terminal capabilities.
    """

    # Status indicators
    PENDING = ("⏳", "[..]")
    PROCESSING = ("🔄", "[~~]")
    COMPLETE = ("✅", "[OK]")
    FAILED = ("❌", "[X!]")
    RETRY = ("⚠️", "[!!]")
    SCRAPING = ("🌐", "[WEB]")
    MERGING = ("🔀", "[MRG]")

    # Progress indicators
    SPINNER_FRAMES_UNICODE = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    SPINNER_FRAMES_ASCII = ["|", "/", "-", "\\"]

    # Progress bar characters
    BAR_FILLED = ("█", "#")
    BAR_EMPTY = ("░", "-")
    BAR_PARTIAL = ("▒", "=")

    # Decorative
    ROCKET = ("🚀", ">>")
    SPARKLES = ("✨", "**")
    CHART = ("📊", "[#]")
    FOLDER = ("📁", "[D]")
    FILE = ("📄", "[F]")
    LINK = ("🔗", "[L]")
    CLOCK = ("⏱️", "[T]")
    CHECK = ("✓", "v")
    CROSS = ("✗", "x")
    DOT = ("•", "*")
    ARROW_UP = ("↑", "^")
    ARROW_DOWN = ("↓", "v")

    # Control indicators
    PAUSE = ("⏸️", "||")
    PLAY = ("▶️", ">")
    STOP = ("⏹️", "[X]")
    PAUSED_INDICATOR = ("⏸", "||")  # Simpler pause for inline use

    # Informational indicators
    WARNING = ("⚠️", "[!]")  # Warning symbol for alerts
    INFO = ("ℹ️", "[i]")  # Info symbol for informational messages

    # Status animations (for paused state pulsing)
    PULSE_FRAMES_UNICODE = ["◉", "◎", "○", "◎"]
    PULSE_FRAMES_ASCII = ["*", "o", ".", "o"]

    _use_ascii: bool = False

    @classmethod
    def set_ascii_mode(cls, use_ascii: bool) -> None:
        """Set whether to use ASCII fallbacks"""
        cls._use_ascii = use_ascii

    @classmethod
    def get(cls, symbol_tuple: Tuple[str, str]) -> str:
        """Get the appropriate symbol based on current mode"""
        return symbol_tuple[1] if cls._use_ascii else symbol_tuple[0]

    @classmethod
    def get_spinner_frames(cls) -> List[str]:
        """Get spinner animation frames"""
        return (
            cls.SPINNER_FRAMES_ASCII if cls._use_ascii else cls.SPINNER_FRAMES_UNICODE
        )

    @classmethod
    def get_pulse_frames(cls) -> List[str]:
        """Get pulse animation frames (for paused state)"""
        return cls.PULSE_FRAMES_ASCII if cls._use_ascii else cls.PULSE_FRAMES_UNICODE

    @classmethod
    def get_pulse_frame(cls, time_offset: float = 0.0) -> str:
        """Get current pulse frame based on time (for paused indicator animation)"""
        frames = cls.get_pulse_frames()
        frame_idx = int((time.time() + time_offset) * 2) % len(frames)
        return frames[frame_idx]

    @classmethod
    def make_progress_bar(cls, percent: float, width: int = 30) -> str:
        """Create a progress bar string with percent capped at 0-100 range.

        Args:
            percent: Progress percentage (clamped to 0-100 to prevent overflow)
            width: Bar width in characters

        Returns:
            Progress bar string using filled/empty characters
        """
        filled_char = cls.get(cls.BAR_FILLED)
        empty_char = cls.get(cls.BAR_EMPTY)

        # WHY clamp: Prevents progress bar overflow when percent > 100
        # Can happen if progress_pct is calculated incorrectly upstream
        clamped_percent = max(0.0, min(100.0, percent))
        filled = int((clamped_percent / 100) * width)
        return filled_char * filled + empty_char * (width - filled)


class BaseTUIManager:
    """
    Base class for TUI managers with thread-safe state management.

    Provides common functionality:
    - Process state machine (WAITING -> RUNNING <-> PAUSED -> STOPPED)
    - Thread-safe state transitions with locking
    - Keyboard listener management
    - Pause time tracking for accurate ETA calculations

    Subclasses should implement:
    - _create_dashboard() -> Layout
    - _create_waiting_dashboard() -> Layout
    """

    def __init__(self, no_tui: bool = False, quiet: bool = False) -> None:
        """
        Initialize base TUI manager.

        Args:
            no_tui: If True, disable Rich TUI (for headless/scripts)
            quiet: If True, minimal output (implies no_tui)
        """
        from rich.console import Console
        from rich.live import Live

        # Quiet mode implies no_tui
        if quiet:
            no_tui = True

        self.no_tui = no_tui
        self.quiet = quiet
        self.capabilities = detect_terminal_capabilities()
        self.console = Console(force_terminal=True, legacy_windows=False)
        self.live: Live | None = None

        # Thread-safe state management
        self._state_lock = threading.Lock()
        self._process_state = ProcessState.WAITING
        self._keyboard_listener: KeyboardListener | None = None

        # Pause time tracking (thread-safe with _state_lock)
        self._pause_start_time: float | None = None
        self._total_pause_duration: float = 0.0

    @property
    def process_state(self) -> ProcessState:
        """Get current process state (thread-safe)."""
        with self._state_lock:
            return self._process_state

    @process_state.setter
    def process_state(self, value: ProcessState) -> None:
        """Set process state (thread-safe)."""
        with self._state_lock:
            self._process_state = value

    @property
    def waiting_to_start(self) -> bool:
        """Check if still waiting to start (backward compatibility)."""
        return self.process_state == ProcessState.WAITING

    @property
    def should_stop(self) -> bool:
        """Check if should stop processing (Ctrl+C was pressed)."""
        return self.process_state == ProcessState.STOPPED

    @property
    def is_paused(self) -> bool:
        """Check if processing is currently paused."""
        return self.process_state == ProcessState.PAUSED

    @property
    def is_running(self) -> bool:
        """Check if processing is currently running."""
        return self.process_state == ProcessState.RUNNING

    def _on_space_pressed(self) -> None:
        """
        Handle SPACE key press - thread-safe start/pause/resume toggle.

        State transitions:
            WAITING -> RUNNING (first press starts processing)
            RUNNING -> PAUSED (pause processing)
            PAUSED -> RUNNING (resume processing)
            STOPPED -> no change (Ctrl+C already pressed)
        """
        with self._state_lock:
            if self._process_state == ProcessState.WAITING:
                self._process_state = ProcessState.RUNNING
            elif self._process_state == ProcessState.RUNNING:
                self._process_state = ProcessState.PAUSED
                self._pause_start_time = time.time()
            elif self._process_state == ProcessState.PAUSED:
                if self._pause_start_time is not None:
                    # WHY max(0): Clock skew could theoretically make this negative
                    pause_elapsed = max(0.0, time.time() - self._pause_start_time)
                    self._total_pause_duration += pause_elapsed
                    self._pause_start_time = None
                self._process_state = ProcessState.RUNNING

    def request_stop(self) -> None:
        """
        Request to stop processing (thread-safe).
        Called by Ctrl+C signal handler.
        """
        with self._state_lock:
            if self._pause_start_time is not None:
                # WHY max(0): Clock skew could theoretically make this negative
                pause_elapsed = max(0.0, time.time() - self._pause_start_time)
                self._total_pause_duration += pause_elapsed
                self._pause_start_time = None
            self._process_state = ProcessState.STOPPED

    def wait_while_paused(self) -> bool:
        """
        Block while paused, return True if should continue, False if stopped.

        This method should be called in processing loops to respect pause state.
        Updates the display while waiting.
        """
        while self.process_state == ProcessState.PAUSED:
            # Use centralized polling interval - DO NOT hardcode timing values
            time.sleep(KEYBOARD_POLL_INTERVAL)
            if self.live:
                self.live.update(self._create_dashboard())
        return self.process_state == ProcessState.RUNNING

    def get_effective_elapsed(self, start_time: "DatetimeType") -> float:
        """
        Get elapsed time excluding pause duration.

        Args:
            start_time: datetime object for when processing started

        Returns:
            Effective elapsed seconds (excluding time spent paused)
        """
        from datetime import datetime

        total_elapsed: float = (datetime.now() - start_time).total_seconds()
        with self._state_lock:
            return total_elapsed - self._total_pause_duration

    def start_keyboard_listener(self) -> None:
        """Start the keyboard listener for SPACE/arrow key handling."""
        if self._keyboard_listener is None:
            self._keyboard_listener = KeyboardListener()
            self._keyboard_listener.register_callback("space", self._on_space_pressed)
            # Subclasses can register additional callbacks (e.g., for scrolling)
        self._keyboard_listener.start()

    def stop_keyboard_listener(self) -> None:
        """Stop the keyboard listener and clean up."""
        if self._keyboard_listener is not None:
            self._keyboard_listener.stop()
            self._keyboard_listener = None

    def _create_dashboard(self) -> Any:
        """
        Create the main processing dashboard.

        Must be implemented by subclasses.
        Returns a Rich renderable (Layout, Panel, etc.)
        """
        raise NotImplementedError("Subclasses must implement _create_dashboard()")

    def _create_waiting_dashboard(self) -> Any:
        """
        Create the waiting screen dashboard.

        Must be implemented by subclasses.
        Returns a Rich renderable (Layout, Panel, etc.)
        """
        raise NotImplementedError(
            "Subclasses must implement _create_waiting_dashboard()"
        )


def detect_terminal_capabilities() -> TerminalCapabilities:
    """
    Detect what the current terminal supports.

    Returns:
        TerminalCapabilities with detected values
    """
    caps = TerminalCapabilities()

    # Check if we're in an interactive terminal
    caps.is_interactive = sys.stdin.isatty() and sys.stdout.isatty()

    # Get terminal size
    try:
        size = os.get_terminal_size()
        caps.width = size.columns
        caps.height = size.lines
    except OSError:
        # Not a terminal or size unavailable - use centralized defaults
        # DO NOT hardcode terminal dimensions
        caps.width = DEFAULT_TERMINAL_WIDTH
        caps.height = DEFAULT_TERMINAL_HEIGHT

    # Check color support via environment variables
    term = os.environ.get("TERM", "").lower()
    colorterm = os.environ.get("COLORTERM", "").lower()

    # No colors in dumb terminal or when NO_COLOR is set
    if term == "dumb" or "NO_COLOR" in os.environ:
        caps.supports_colors = False
        caps.supports_256_colors = False
        caps.supports_true_color = False
    else:
        caps.supports_colors = True
        caps.supports_256_colors = "256color" in term or colorterm in (
            "truecolor",
            "24bit",
        )
        caps.supports_true_color = colorterm in ("truecolor", "24bit")

    # Check Unicode/emoji support
    caps.supports_unicode = _check_unicode_support()
    caps.supports_emoji = caps.supports_unicode and _check_emoji_support()

    # Configure Symbols based on capabilities
    Symbols.set_ascii_mode(not caps.supports_emoji)

    logger.debug(
        f"Terminal capabilities: unicode={caps.supports_unicode}, "
        f"emoji={caps.supports_emoji}, colors={caps.supports_colors}, "
        f"size={caps.width}x{caps.height}"
    )

    return caps


def _check_unicode_support() -> bool:
    """Check if terminal supports Unicode"""
    # Check locale
    try:
        encoding = locale.getpreferredencoding(False).lower()
        if "utf" in encoding:
            return True
    except Exception as e:
        # WHY debug not warning: This is a fallback check, failure is expected on some systems
        # DO NOT: Use pass without logging - makes debugging encoding issues impossible
        logger.debug(f"Could not get preferred encoding: {e}")

    # Check LANG environment variable
    lang = os.environ.get("LANG", "").lower()
    if "utf" in lang:
        return True

    # Check stdout encoding
    try:
        if hasattr(sys.stdout, "encoding") and sys.stdout.encoding:
            if "utf" in sys.stdout.encoding.lower():
                return True
    except Exception as e:
        # WHY debug not warning: Fallback check, sys.stdout may not have encoding attribute
        # DO NOT: Use pass without logging - makes debugging impossible
        logger.debug(f"Could not check stdout encoding: {e}")

    # Default to True on modern systems, False on Windows legacy console
    if IS_WINDOWS:
        # Check for Windows Terminal or ConEmu
        wt = os.environ.get("WT_SESSION")
        conemu = os.environ.get("ConEmuANSI")
        if wt or conemu:
            return True
        return False

    return True


def _check_emoji_support() -> bool:
    """
    Check if terminal likely supports emoji.
    This is heuristic-based since there's no reliable detection method.
    """
    # Check for known emoji-capable terminals
    term_program = os.environ.get("TERM_PROGRAM", "").lower()
    term = os.environ.get("TERM", "").lower()

    # Known good terminals
    good_terminals = [
        "iterm",
        "hyper",
        "kitty",
        "alacritty",
        "wezterm",
        "windows terminal",
        "vscode",
        "apple_terminal",
    ]

    if any(t in term_program for t in good_terminals):
        return True

    # Windows Terminal
    if os.environ.get("WT_SESSION"):
        return True

    # macOS Terminal.app
    if term_program == "apple_terminal":
        return True

    # xterm-256color on Linux often supports emoji
    if "xterm" in term and sys.platform == "linux":
        return True

    # SSH sessions may not support emoji well
    if os.environ.get("SSH_CLIENT") or os.environ.get("SSH_TTY"):
        return False

    # Default to True on modern systems
    return True


class KeyboardListener:
    """
    Cross-platform keyboard listener for non-blocking key detection.

    Supports:
    - Space bar detection (start/stop)
    - Arrow key detection (scrolling)
    - Graceful cleanup on exit

    Thread Safety:
    - All instances are tracked for atexit cleanup
    - Terminal settings restored even on abnormal exit
    """

    # Class-level tracking of all instances for atexit cleanup
    # WHY: Ensures terminal restoration even on abnormal exit
    _all_instances: ClassVar[List["KeyboardListener"]] = []
    _atexit_registered: ClassVar[bool] = False

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._old_settings: Any | None = None
        self._callbacks: Dict[str, Callable[[], None]] = {}

        # Track this instance for cleanup
        KeyboardListener._all_instances.append(self)

        # Register atexit handler once
        if not KeyboardListener._atexit_registered:
            atexit.register(KeyboardListener._cleanup_all_instances)
            KeyboardListener._atexit_registered = True

    @classmethod
    def _cleanup_all_instances(cls) -> None:
        """
        Cleanup all keyboard listener instances on exit.

        WHY atexit: Ensures terminal restoration even on abnormal exit
        DO NOT: Remove this - leaves terminal in broken state on crash
        """
        for instance in cls._all_instances:
            try:
                instance.stop()
            except Exception as e:
                logger.debug(f"Error cleaning up keyboard listener: {e}")
        cls._all_instances.clear()

    def register_callback(self, key: str, callback: Callable[[], None]) -> None:
        """
        Register a callback for a specific key.

        Args:
            key: Key identifier ("space", "up", "down", "escape", "q")
            callback: Function to call when key is pressed
        """
        self._callbacks[key.lower()] = callback

    def start(self) -> bool:
        """
        Start the keyboard listener thread.

        Returns:
            True if started successfully, False otherwise
        """
        if not sys.stdin.isatty():
            logger.debug("Not a TTY, keyboard listener disabled")
            return False

        if self._thread is not None and self._thread.is_alive():
            return True  # Already running

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        """Stop the keyboard listener and restore terminal settings"""
        self._stop_event.set()

        if self._thread is not None:
            # Use centralized timeout - DO NOT hardcode timing values
            self._thread.join(timeout=KEYBOARD_THREAD_TIMEOUT)
            self._thread = None

        self._restore_terminal()

    def _listen_loop(self) -> None:
        """Main listening loop - runs in separate thread"""
        if IS_WINDOWS:
            self._listen_windows()
        else:
            self._listen_unix()

    def _listen_unix(self) -> None:
        """Unix-specific keyboard listening using termios"""
        if not HAS_TERMIOS:
            return

        try:
            self._old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())

            while not self._stop_event.is_set():
                # Check for input with timeout - use centralized polling interval
                # DO NOT hardcode timing values
                if (
                    sys.stdin
                    in select.select([sys.stdin], [], [], KEYBOARD_POLL_INTERVAL)[0]
                ):
                    char = sys.stdin.read(1)
                    self._handle_key_unix(char)
        except (termios.error, OSError, ValueError) as e:
            logger.debug(f"Keyboard listener error: {e}")
        finally:
            self._restore_terminal()

    def _handle_key_unix(self, char: str) -> None:
        """Handle a key press on Unix"""
        if char == " ":
            self._trigger_callback("space")
        elif char == "q":
            self._trigger_callback("q")
        elif char == "\x1b":  # Escape sequence
            # Read additional chars for arrow keys and navigation keys
            try:
                next_chars = sys.stdin.read(2)
                if next_chars == "[A":
                    self._trigger_callback("up")
                elif next_chars == "[B":
                    self._trigger_callback("down")
                elif next_chars == "[C":
                    self._trigger_callback("right")
                elif next_chars == "[D":
                    self._trigger_callback("left")
                elif next_chars == "[H":
                    # WHY: Home key - scroll to beginning of list
                    self._trigger_callback("home")
                elif next_chars == "[F":
                    # WHY: End key - scroll to end of list
                    self._trigger_callback("end")
                elif next_chars == "[5":
                    # PageUp sequence: ESC[5~
                    extra = sys.stdin.read(1)  # Read the trailing '~'
                    if extra == "~":
                        self._trigger_callback("pageup")
                elif next_chars == "[6":
                    # PageDown sequence: ESC[6~
                    extra = sys.stdin.read(1)  # Read the trailing '~'
                    if extra == "~":
                        self._trigger_callback("pagedown")
                elif next_chars == "[1":
                    # Alternative Home/End: ESC[1~ (Home) or ESC[4~ (End on some terminals)
                    extra = sys.stdin.read(1)
                    if extra == "~":
                        self._trigger_callback("home")
                elif next_chars == "[4":
                    # Alternative End sequence: ESC[4~
                    extra = sys.stdin.read(1)
                    if extra == "~":
                        self._trigger_callback("end")
            except Exception as e:
                # WHY debug not warning: Keyboard input failures are common and recoverable
                # DO NOT: Use pass without logging - makes debugging input issues impossible
                logger.debug(f"Arrow key read error (Unix): {e}")

    def _listen_windows(self) -> None:
        """Windows-specific keyboard listening using msvcrt"""
        if not HAS_MSVCRT:
            return

        while not self._stop_event.is_set():
            if msvcrt.kbhit():  # type: ignore[attr-defined]
                char = msvcrt.getch()  # type: ignore[attr-defined]
                self._handle_key_windows(char)
            else:
                # Use centralized polling interval - DO NOT hardcode timing values
                self._stop_event.wait(KEYBOARD_POLL_INTERVAL)

    def _handle_key_windows(self, char: bytes) -> None:
        """Handle a key press on Windows"""
        if char == b" ":
            self._trigger_callback("space")
        elif char == b"q":
            self._trigger_callback("q")
        elif char in (b"\x00", b"\xe0"):  # Special key prefix
            try:
                special = msvcrt.getch()  # type: ignore[attr-defined]
                if special == b"H":  # Up arrow
                    self._trigger_callback("up")
                elif special == b"P":  # Down arrow
                    self._trigger_callback("down")
                elif special == b"K":  # Left arrow
                    self._trigger_callback("left")
                elif special == b"M":  # Right arrow
                    self._trigger_callback("right")
                elif special == b"I":  # PageUp (0x49)
                    self._trigger_callback("pageup")
                elif special == b"Q":  # PageDown (0x51)
                    self._trigger_callback("pagedown")
                elif special == b"G":  # Home (0x47)
                    self._trigger_callback("home")
                elif special == b"O":  # End (0x4F)
                    self._trigger_callback("end")
            except Exception as e:
                # WHY debug not warning: Keyboard input failures are common and recoverable
                # DO NOT: Use pass without logging - makes debugging input issues impossible
                logger.debug(f"Special key read error (Windows): {e}")

    def _trigger_callback(self, key: str) -> None:
        """Trigger registered callback for a key"""
        callback = self._callbacks.get(key)
        if callback:
            try:
                callback()
            except Exception as e:
                logger.warning(f"Keyboard callback error for '{key}': {e}")

    def _restore_terminal(self) -> None:
        """Restore terminal settings (Unix only).

        CRITICAL: Must always attempt restoration to avoid leaving terminal broken.
        WHY log errors: Silent failures hide terminal corruption issues.
        """
        if not IS_WINDOWS and HAS_TERMIOS and self._old_settings is not None:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)
                self._old_settings = None
            except (termios.error, OSError) as e:
                # WHY log: Terminal restoration failures can leave terminal in broken state
                # DO NOT: Silently ignore - this helps debug terminal corruption
                logger.debug(f"Failed to restore terminal settings: {e}")


def format_duration(seconds: float) -> str:
    """
    Format a duration in human-readable form.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted string like "1m 30s" or "45s" or "2h 15m"

    Raises:
        ValueError: If seconds is negative (indicates serious bug)
    """
    # FAIL-FAST: Negative durations indicate a serious bug in caller
    # WHY: Clock skew, race condition, or logic error in time calculation
    # DO NOT: Silently clamp - bugs must be caught immediately
    if seconds < 0:
        raise ValueError(
            f"format_duration received negative seconds={seconds} - indicates serious bug"
        )
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def format_size(bytes_size: int) -> str:
    """
    Format a size in bytes to human-readable form.

    Args:
        bytes_size: Size in bytes

    Returns:
        Formatted string like "1.5KB" or "2.3MB"

    Raises:
        ValueError: If bytes_size is negative (indicates serious bug)
    """
    # FAIL-FAST: Negative sizes indicate a serious bug in caller
    # WHY: Data corruption, integer overflow, or logic error
    # DO NOT: Silently clamp - bugs must be caught immediately
    if bytes_size < 0:
        raise ValueError(
            f"format_size received negative bytes_size={bytes_size} - indicates serious bug"
        )
    if bytes_size < 1024:
        return f"{bytes_size}B"
    elif bytes_size < 1024 * 1024:
        return f"{bytes_size / 1024:.1f}KB"
    elif bytes_size < 1024 * 1024 * 1024:
        return f"{bytes_size / (1024 * 1024):.1f}MB"
    else:
        return f"{bytes_size / (1024 * 1024 * 1024):.1f}GB"


def calculate_eta(progress_pct: float, elapsed_seconds: float) -> float | None:
    """
    Calculate estimated time remaining.

    Args:
        progress_pct: Current progress percentage (0-100)
        elapsed_seconds: Time elapsed so far

    Returns:
        Estimated seconds remaining, or None if cannot calculate

    Raises:
        ValueError: If elapsed_seconds is negative (indicates serious bug)
    """
    # FAIL-FAST: Negative elapsed time indicates serious bug in caller
    # WHY: Clock skew, race condition in pause tracking, or logic error
    # DO NOT: Silently produce wrong ETA - bugs must be caught immediately
    if elapsed_seconds < 0:
        raise ValueError(
            f"calculate_eta received negative elapsed_seconds={elapsed_seconds} - indicates serious bug"
        )

    if progress_pct <= 0 or progress_pct >= 100:
        return None

    estimated_total = (elapsed_seconds / progress_pct) * 100
    remaining = estimated_total - elapsed_seconds
    return float(max(0.0, remaining))


def truncate_url(url: str, max_length: int = URL_TRUNCATE_MAX_LENGTH) -> str:
    """
    Truncate a URL for display, keeping the important parts visible.

    Args:
        url: Full URL to truncate
        max_length: Maximum display length (default from config.py)

    Returns:
        Truncated URL with ellipsis if needed
    """
    if len(url) <= max_length:
        return url

    # Try to keep the domain and end of path visible
    # e.g., "https://example.com/very/long/.../page.html"
    # NOTE: urlparse is imported at module level for performance
    try:
        parsed = urlparse(url)
        domain = parsed.netloc
        path = parsed.path

        # Always show domain
        if len(domain) >= max_length - 10:
            return url[: max_length - 3] + "..."

        # Calculate space for path
        available_for_path = max_length - len(domain) - 10  # 10 for scheme and ellipsis

        if len(path) <= available_for_path:
            return url[: max_length - 3] + "..."

        # Show beginning and end of path
        half = available_for_path // 2
        truncated_path = path[:half] + "..." + path[-half:]

        return f"{parsed.scheme}://{domain}{truncated_path}"
    except Exception:
        # Fallback to simple truncation
        return url[: max_length - 3] + "..."
