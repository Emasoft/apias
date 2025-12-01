"""
Batch TUI Module for monitoring multiple URLs being processed in parallel.

Provides a scrollable dashboard that shows:
- Multiple URLs being processed concurrently (10-100+)
- Each URL gets 4 lines: header, stats, progress bar, status message (optional)
- Real-time updates as threads process URLs
- Scrollable viewport with arrow keys
- Cross-platform terminal support with ASCII fallbacks
- "PRESS SPACE TO START" waiting screen
"""

import logging
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

# TYPE_CHECKING imports to avoid circular imports at runtime
# WHY: status_pipeline imports batch_tui, so we need conditional import
if TYPE_CHECKING:
    from apias.status_pipeline import TaskSnapshot

from rich import box
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Import centralized constants - single source of truth for configuration values
# DO NOT hardcode values here - add new constants to config.py instead
from apias.config import (
    DEFAULT_TERMINAL_HEIGHT,
    DEFAULT_TERMINAL_WIDTH,
    FALLBACK_VERSION,
    KEYBOARD_POLL_INTERVAL,
    MAX_FAILED_URLS_TO_SHOW,
    SCROLL_DEBOUNCE_SECONDS,
    STATS_PROGRESS_BAR_WIDTH,
    TUI_REFRESH_FPS,
    URL_TRUNCATE_MAX_LENGTH,
)

# Import shared terminal utilities for cross-platform support
from apias.terminal_utils import (
    BaseTUIManager,
    ProcessState,
    Symbols,
    calculate_eta,
    format_duration,
)

# Module-level logger for tracing TUI operations and debugging
logger = logging.getLogger(__name__)

# Import version
try:
    from apias import __version__
except ImportError:
    # Use centralized fallback version from config.py
    # DO NOT hardcode version string here
    __version__ = FALLBACK_VERSION


class URLState(Enum):
    """States for URL processing with ASCII fallback support"""

    PENDING = "pending"
    SCRAPING = "scraping"
    PROCESSING = "processing"
    MERGING_CHUNKS = (
        "merging"  # Per-URL chunk merging (reconstructs coherent API from chunks)
    )
    COMPLETE = "complete"
    FAILED = "failed"

    def get_symbol(self) -> str:
        """Get the display symbol (emoji or ASCII) based on terminal capabilities"""
        symbol_map = {
            URLState.PENDING: Symbols.PENDING,
            URLState.SCRAPING: Symbols.SCRAPING,
            URLState.PROCESSING: Symbols.PROCESSING,
            URLState.MERGING_CHUNKS: Symbols.MERGING,
            URLState.COMPLETE: Symbols.COMPLETE,
            URLState.FAILED: Symbols.FAILED,
        }
        return Symbols.get(symbol_map[self])


@dataclass
class URLTask:
    """Status of a URL being processed"""

    task_id: int
    url: str
    state: URLState = URLState.PENDING
    progress_pct: float = 0.0
    size_in: int = 0
    size_out: int = 0
    cost: float = 0.0
    duration: float = 0.0
    start_time: float | None = None
    error: str = ""
    # Chunk tracking (for large pages split into multiple LLM requests)
    current_chunk: int = 0  # 0 means not chunked or not started
    total_chunks: int = 0  # 0 means not chunked
    # Status message for errors, retries, warnings (displayed above progress bar)
    # WHY clear terminology: "AI service" distinguishes from "website scraping" timeouts
    status_message: str = ""  # e.g., "⚠️ AI service timeout. Retrying in 3s..."
    # Status history: last 5 (timestamp, message) tuples for debugging
    # WHY: Allows TUI to show recent status changes and status_pipeline to track history
    status_history: List[Tuple[datetime, str]] = field(default_factory=list)


@dataclass
class BatchStats:
    """Overall batch processing statistics"""

    total_urls: int = 0
    pending: int = 0
    scraping: int = 0
    processing: int = 0
    completed: int = 0
    failed: int = 0
    total_cost: float = 0.0
    start_time: datetime = field(default_factory=datetime.now)


class BatchTUIManager(BaseTUIManager):
    """
    TUI manager for batch processing of multiple URLs.

    Displays a scrollable list of URLs with real-time progress.
    Each URL shows: header (task # + URL), stats row, progress bar.

    Inherits thread-safe state management from BaseTUIManager.
    """

    def __init__(
        self, urls: List[str], no_tui: bool = False, quiet: bool = False
    ) -> None:
        """
        Initialize batch TUI manager.

        Args:
            urls: List of URLs to process
            no_tui: If True, disable Rich TUI
            quiet: If True, minimal output (implies no_tui)
        """
        # Initialize base class (handles state management, console, etc.)
        super().__init__(no_tui=no_tui, quiet=quiet)

        # Batch-specific state
        self.stats = BatchStats(total_urls=len(urls), pending=len(urls))
        self.tasks: Dict[int, URLTask] = {}

        # Additional lock for task/stats updates (separate from state lock)
        self._task_lock = threading.Lock()

        # Scrolling state for URL list navigation
        # THREAD SAFETY: scroll_offset is modified from keyboard thread
        # WHY separate lock: Avoids blocking task updates during scroll
        self._scroll_lock = threading.Lock()
        self.scroll_offset = 0
        # WHY debounce: Prevents accidental rapid scrolling from key repeat
        # DO NOT: Hardcode this value - use SCROLL_DEBOUNCE_SECONDS from config.py
        self._last_scroll_time: float = 0.0

        # Initialize all URLs as pending tasks
        for idx, url in enumerate(urls, start=1):
            self.tasks[idx] = URLTask(task_id=idx, url=url)

    # Override get_effective_elapsed to use stats.start_time
    def get_effective_elapsed(self) -> float:  # type: ignore[override]
        """Get elapsed time excluding pause duration."""
        return super().get_effective_elapsed(self.stats.start_time)

    def _should_debounce_scroll(self) -> bool:
        """Check if scroll event should be debounced (thread-safe).

        Returns:
            True if scroll should be ignored (too soon after last scroll)

        WHY separate method: Avoids DRY violation - debounce logic in one place
        THREAD SAFETY: Uses _scroll_lock to protect _last_scroll_time
        DO NOT: Duplicate this logic in _on_scroll_up/_on_scroll_down
        """
        current_time = time.time()
        with self._scroll_lock:
            # Check if we're within debounce window
            if current_time - self._last_scroll_time < SCROLL_DEBOUNCE_SECONDS:
                return True  # Debounce - ignore this scroll event
            # Update last scroll time for next check
            self._last_scroll_time = current_time
            return False

    def _on_scroll_up(self) -> None:
        """Handle scroll up (registered with keyboard listener).

        THREAD SAFETY: Called from keyboard listener thread
        Uses _scroll_lock to protect scroll_offset modification
        """
        if self._should_debounce_scroll():
            return

        with self._scroll_lock:
            if self.scroll_offset > 0:
                self.scroll_offset -= 1
                logger.debug(f"Scroll up: offset now {self.scroll_offset}")

    def _get_visible_tasks_count(self) -> int:
        """Calculate how many tasks are visible based on terminal height.

        WHY separate method: Avoids hardcoding visible task count (was '5')
        Uses same calculation as _create_urls_panel for consistency
        """
        try:
            term_height = self.console.size.height
        except Exception:
            # Fallback if terminal size unavailable
            # WHY constant: Use centralized fallback, not hardcoded value
            term_height = DEFAULT_TERMINAL_HEIGHT
        # Same calculation as _create_urls_panel
        # Each task uses ~4 lines, minus space for header/footer/stats
        header_size = 3
        footer_size = 2
        stats_size = min(8, max(6, term_height // 6))
        urls_size = max(15, term_height - header_size - stats_size - footer_size - 2)
        lines_per_task = 4
        return int(max(1, (urls_size - 4) // lines_per_task))

    def _on_scroll_down(self) -> None:
        """Handle scroll down (registered with keyboard listener).

        THREAD SAFETY: Called from keyboard listener thread
        Uses _scroll_lock to protect scroll_offset modification
        """
        if self._should_debounce_scroll():
            return

        with self._scroll_lock:
            # WHY dynamic: Use actual visible count, not hardcoded value
            # This ensures scrolling stops at exactly the right position
            visible_tasks = self._get_visible_tasks_count()
            max_offset = max(0, len(self.tasks) - visible_tasks)
            if self.scroll_offset < max_offset:
                self.scroll_offset += 1
                logger.debug(f"Scroll down: offset now {self.scroll_offset}")

    def _on_page_up(self) -> None:
        """Handle PageUp key - scroll up by one page.

        THREAD SAFETY: Called from keyboard listener thread
        Uses _scroll_lock to protect scroll_offset modification
        """
        if self._should_debounce_scroll():
            return

        with self._scroll_lock:
            # WHY page_size from visible: Jump by visible tasks count for natural paging
            page_size = self._get_visible_tasks_count()
            self.scroll_offset = max(0, self.scroll_offset - page_size)
            logger.debug(f"Page up: offset now {self.scroll_offset}")

    def _on_page_down(self) -> None:
        """Handle PageDown key - scroll down by one page.

        THREAD SAFETY: Called from keyboard listener thread
        Uses _scroll_lock to protect scroll_offset modification
        """
        if self._should_debounce_scroll():
            return

        with self._scroll_lock:
            page_size = self._get_visible_tasks_count()
            max_offset = max(0, len(self.tasks) - page_size)
            self.scroll_offset = min(max_offset, self.scroll_offset + page_size)
            logger.debug(f"Page down: offset now {self.scroll_offset}")

    def _on_home(self) -> None:
        """Handle Home key - scroll to beginning.

        THREAD SAFETY: Called from keyboard listener thread
        Uses _scroll_lock to protect scroll_offset modification
        """
        if self._should_debounce_scroll():
            return

        with self._scroll_lock:
            self.scroll_offset = 0
            logger.debug("Home: scrolled to beginning")

    def _on_end(self) -> None:
        """Handle End key - scroll to end.

        THREAD SAFETY: Called from keyboard listener thread
        Uses _scroll_lock to protect scroll_offset modification
        """
        if self._should_debounce_scroll():
            return

        with self._scroll_lock:
            visible_tasks = self._get_visible_tasks_count()
            max_offset = max(0, len(self.tasks) - visible_tasks)
            self.scroll_offset = max_offset
            logger.debug(f"End: scrolled to end (offset {self.scroll_offset})")

    def start_keyboard_listener(self) -> None:
        """Start keyboard listener with scroll support."""
        super().start_keyboard_listener()
        # Register scroll callbacks
        if self._keyboard_listener:
            self._keyboard_listener.register_callback("up", self._on_scroll_up)
            self._keyboard_listener.register_callback("down", self._on_scroll_down)
            # WHY: PageUp/PageDown for faster navigation through long task lists
            self._keyboard_listener.register_callback("pageup", self._on_page_up)
            self._keyboard_listener.register_callback("pagedown", self._on_page_down)
            # WHY: Home/End for jumping to start/end of list
            self._keyboard_listener.register_callback("home", self._on_home)
            self._keyboard_listener.register_callback("end", self._on_end)

    # Note: waiting_to_start, should_stop, is_paused, is_running,
    # _on_space_pressed, request_stop, wait_while_paused are inherited
    # from BaseTUIManager with thread-safe implementation.

    def wait_for_start(self) -> None:
        """Display waiting screen until user presses SPACE"""
        if self.no_tui or not sys.stdin.isatty():
            # No TUI mode - start immediately
            self.process_state = ProcessState.RUNNING
            return

        self.start_keyboard_listener()
        self.console.clear()

        self.live = Live(
            self._create_waiting_dashboard(),
            console=self.console,
            # Use centralized constant for refresh rate - DO NOT hardcode FPS values
            refresh_per_second=TUI_REFRESH_FPS,
            screen=True,
        )
        self.live.start()

        # Wait until user presses SPACE (changes state from WAITING)
        # CRITICAL FIX: Also check should_stop to allow Ctrl+C exit during wait
        # WHY: Without this check, if signal handler fails to update process_state,
        # this becomes an infinite loop that user cannot escape.
        # DO NOT remove the should_stop check - it's the safety valve.
        while self.process_state == ProcessState.WAITING and not self.should_stop:
            # Use centralized polling interval - DO NOT hardcode timing values
            time.sleep(KEYBOARD_POLL_INTERVAL)
            if self.live:
                self.live.update(self._create_waiting_dashboard())

        # Switch to main dashboard if not stopped
        if self.live and self.process_state != ProcessState.STOPPED:
            self.live.update(self._create_dashboard())

    def _create_waiting_dashboard(self) -> Panel:
        """Create the waiting screen shown before processing starts"""
        pause = Symbols.get(Symbols.PAUSE)
        waiting_msg = Text.assemble(
            (f"{pause}  ", "bold yellow"),
            ("PRESS SPACE BAR TO START BATCH SCRAPING", "bold yellow blink"),
            ("\n\n", ""),
            (f"{self.stats.total_urls} URLs queued for processing", "dim white"),
            ("\n", ""),
            ("Press SPACE again to stop", "dim white"),
        )

        return Panel(
            waiting_msg,
            box=box.DOUBLE,
            padding=(2, 4),
            border_style="yellow",
            title=f"[bold yellow]{pause}  READY TO START[/bold yellow]",
            title_align="center",
        )

    def _create_dashboard(self) -> Layout:
        """Create the main batch processing dashboard (pause-aware)"""
        layout = Layout()

        # Calculate layout sizes
        term_height = self.console.size.height
        header_size = 3
        footer_size = 2
        stats_size = min(8, max(6, term_height // 6))
        urls_size = max(15, term_height - header_size - stats_size - footer_size - 2)

        layout.split_column(
            Layout(name="header", size=header_size),
            Layout(name="stats", size=stats_size),
            Layout(name="urls", size=urls_size),
            Layout(name="footer", size=footer_size),
        )

        # Header with version and pause state indicator
        rocket = Symbols.get(Symbols.ROCKET)
        if self.process_state == ProcessState.PAUSED:
            # Show paused indicator in header with pulsing animation
            pulse = Symbols.get_pulse_frame()
            pause_sym = Symbols.get(Symbols.PAUSE)
            header_text = f"{pause_sym} [bold yellow]PAUSED[/] {pulse} APIAS v{__version__} - Batch Processing Dashboard"
            header_style = "yellow"
        else:
            header_text = f"{rocket} [bold cyan]APIAS v{__version__}[/] - Batch Processing Dashboard"
            header_style = "cyan"

        layout["header"].update(
            Panel(
                Text.from_markup(header_text, justify="center"),
                box=box.ROUNDED,
                border_style=header_style,
            )
        )

        # Stats
        layout["stats"].update(self._create_stats_panel())

        # URLs list (scrollable)
        layout["urls"].update(self._create_urls_panel(urls_size))

        # Footer with state-aware controls
        effective_elapsed = self.get_effective_elapsed()
        clock = Symbols.get(Symbols.CLOCK)
        arrow_up = Symbols.get(Symbols.ARROW_UP)
        arrow_down = Symbols.get(Symbols.ARROW_DOWN)

        if self.process_state == ProcessState.PAUSED:
            # Paused state - show pulsing indicator and resume instructions
            pulse = Symbols.get_pulse_frame()
            play_sym = Symbols.get(Symbols.PLAY)
            footer_text = (
                f"{pulse} [bold yellow]PAUSED[/bold yellow] {pulse}  |  "
                f"SPACE: {play_sym} Resume  |  Ctrl+C: Stop & Exit  |  "
                f"Paused at: {datetime.now().strftime('%H:%M:%S')}"
            )
            footer_style = "bold yellow"
        else:
            # Running state - show pause and stop instructions
            pause_sym = Symbols.get(Symbols.PAUSE)
            stop_sym = Symbols.get(Symbols.STOP)
            # WHY: Show navigation keys including PageUp/PageDown for user discoverability
            footer_text = (
                f"{clock}  Elapsed: {int(effective_elapsed) // 60:02d}:{int(effective_elapsed) % 60:02d}  |  "
                f"Nav: {arrow_up}{arrow_down} PgUp/Dn Home/End  |  SPACE: {pause_sym} Pause  |  Ctrl+C: {stop_sym} Stop  |  "
                f"{datetime.now().strftime('%H:%M:%S')}"
            )
            footer_style = "dim"

        layout["footer"].update(Text.from_markup(footer_text, style=footer_style))

        return layout

    def _create_stats_panel(self) -> Panel:
        """Create overall statistics panel with ETA (pause-aware)"""
        # Use yellow border when paused, green when running
        border_style = (
            "yellow" if self.process_state == ProcessState.PAUSED else "green"
        )

        table = Table(show_header=False, box=None, expand=True)
        table.add_column("Metric", style="cyan", width=20)
        # WHY justify="left": Progress bar should align to left edge, not center
        table.add_column("Value", style="bold green", justify="left")

        # Calculate progress percentage
        # WHY clamp: Defensive programming - prevents display bugs if stats are inconsistent
        # Edge case: Race condition could cause completed+failed > total_urls
        raw_pct = (
            (self.stats.completed + self.stats.failed) / self.stats.total_urls * 100
            if self.stats.total_urls > 0
            else 0
        )
        # Clamp to valid range for both bar and text display
        progress_pct = max(0.0, min(100.0, raw_pct))

        # Use Symbols for progress bar (ASCII fallback)
        # NOTE: make_progress_bar also clamps internally - belt and suspenders
        # WHY constant: Use centralized bar width from config
        bar_length = STATS_PROGRESS_BAR_WIDTH
        progress_bar = Symbols.make_progress_bar(progress_pct, bar_length)

        # Calculate ETA using effective elapsed (excludes pause time)
        effective_elapsed = self.get_effective_elapsed()
        eta_seconds = calculate_eta(progress_pct, effective_elapsed)

        # Show "(paused)" in ETA when paused
        if self.process_state == ProcessState.PAUSED:
            eta_str = "[yellow](paused)[/yellow]"
        else:
            eta_str = format_duration(eta_seconds) if eta_seconds else "--"

        table.add_row("Overall Progress", f"{progress_bar} {progress_pct:.0f}%")
        table.add_row("ETA", eta_str)
        table.add_row("Total URLs", f"{self.stats.total_urls}")
        table.add_row("Pending", f"{self.stats.pending}")
        table.add_row("Scraping", f"{self.stats.scraping}")
        table.add_row("Processing", f"{self.stats.processing}")
        table.add_row("Completed", f"[green]{self.stats.completed}[/green]")
        table.add_row(
            "Failed",
            (
                f"[red]{self.stats.failed}[/red]"
                if self.stats.failed > 0
                else f"{self.stats.failed}"
            ),
        )
        table.add_row(
            "Total Cost",
            f"${self.stats.total_cost:.4f} (~{self.stats.total_cost * 100:.1f}¢)",
        )

        chart = Symbols.get(Symbols.CHART)
        return Panel(
            table,
            title=f"[bold cyan]{chart} Statistics[/bold cyan]",
            box=box.ROUNDED,
            border_style=border_style,
        )

    def _create_urls_panel(self, available_height: int) -> Panel:
        """Create scrollable URLs panel with 4 lines per URL (when status_message present).

        THREAD SAFETY: Reads scroll_offset under _scroll_lock since it's
        modified by keyboard listener thread. This prevents torn reads.
        """
        # Calculate how many complete tasks fit in available height
        # Each task uses 4 lines: header, stats, progress bar, status message (optional)
        # But we use 4 to account for worst case with status messages
        lines_per_task = 4
        visible_tasks = max(
            1, (available_height - 4) // lines_per_task
        )  # -4 for panel borders/title

        # THREAD SAFETY: Read scroll_offset under lock
        # WHY: scroll_offset is modified by keyboard thread (_on_scroll_up/down)
        # Without lock, we could read a partially-updated value (torn read)
        with self._scroll_lock:
            current_offset = self.scroll_offset

        # Get tasks to display (scrolled window)
        task_ids = sorted(self.tasks.keys())
        start_idx = current_offset
        end_idx = min(len(task_ids), start_idx + visible_tasks)
        visible_task_ids = task_ids[start_idx:end_idx]

        # THREAD SAFETY: Create snapshot of visible tasks under lock
        # WHY: update_task() modifies tasks from worker threads under _task_lock
        # Without this, we could read partially-updated task attributes (torn read)
        # DO NOT: Read task attributes outside this lock without copying first
        with self._task_lock:
            # Create shallow copies of task objects for this render cycle
            task_snapshots = {
                tid: URLTask(
                    task_id=self.tasks[tid].task_id,
                    url=self.tasks[tid].url,
                    state=self.tasks[tid].state,
                    progress_pct=self.tasks[tid].progress_pct,
                    size_in=self.tasks[tid].size_in,
                    size_out=self.tasks[tid].size_out,
                    cost=self.tasks[tid].cost,
                    duration=self.tasks[tid].duration,
                    start_time=self.tasks[tid].start_time,
                    error=self.tasks[tid].error,
                    current_chunk=self.tasks[tid].current_chunk,
                    total_chunks=self.tasks[tid].total_chunks,
                    status_message=self.tasks[tid].status_message,
                )
                for tid in visible_task_ids
            }

        # Build display
        lines = []
        for task_id in visible_task_ids:
            task = task_snapshots[task_id]

            # Line 1: Task header (task # + URL)
            # WHY constant: Use centralized URL length limit from config
            # DO NOT: Hardcode display length - use URL_TRUNCATE_MAX_LENGTH
            url_display = (
                task.url
                if len(task.url) <= URL_TRUNCATE_MAX_LENGTH
                else task.url[: URL_TRUNCATE_MAX_LENGTH - 3] + "..."
            )
            lines.append(f"[bold cyan]Task #{task_id:02d}:[/bold cyan] {url_display}")

            # Line 2: Stats row
            state_emoji = task.state.get_symbol()
            size_in_kb = task.size_in / 1024 if task.size_in > 0 else 0
            size_out_kb = task.size_out / 1024 if task.size_out > 0 else 0

            # WHY live duration: For active tasks, calculate elapsed time LIVE
            # Previously used task.duration which is only set on COMPLETE/FAILED,
            # causing "0.0s" to display until state change.
            # FIX: Show running timer for active tasks, freeze for finished tasks.
            is_finished = task.state in [URLState.COMPLETE, URLState.FAILED]
            if is_finished:
                # Task finished - use stored duration (frozen value)
                duration_str = f"{task.duration:.1f}s" if task.duration > 0 else "0.0s"
            elif task.start_time:
                # Task active - calculate live elapsed time
                # WHY max(0): Clock skew protection
                live_elapsed = max(0.0, time.time() - task.start_time)
                duration_str = f"{live_elapsed:.1f}s"
            else:
                # Task not started yet
                duration_str = "0.0s"

            # Show chunk information if page is chunked
            chunk_info = ""
            if task.total_chunks > 0:
                if task.state == URLState.PROCESSING:
                    chunk_info = f" (Chunk {task.current_chunk}/{task.total_chunks})"
                elif task.state == URLState.MERGING_CHUNKS:
                    chunk_info = " (Merging chunks)"

            stats_line = (
                f"  {state_emoji} Status: [yellow]{task.state.name}{chunk_info}[/yellow]  |  "
                f"Size: {size_in_kb:.1f}KB -> {size_out_kb:.1f}KB  |  "
                f"Cost: ${task.cost:.4f}  |  "
                f"Duration: {duration_str}"
            )
            lines.append(stats_line)

            # Line 3: Progress bar (adaptive width)
            # Calculate available width for progress bar
            term_width = self.console.size.width
            panel_padding = 6  # Panel borders and padding
            prefix = "  ["  # Progress bar prefix
            suffix = "]"  # Progress bar suffix

            # WHY clamp: Prevents percentage display exceeding 100%
            display_pct = max(0.0, min(100.0, task.progress_pct))

            # WHY check state: Freeze elapsed time when task is finished
            # Using task.duration for finished tasks prevents timer from running
            is_finished = task.state in [URLState.COMPLETE, URLState.FAILED]

            if is_finished:
                # Task finished - show final percentage (100% for complete, actual for failed)
                if task.state == URLState.COMPLETE:
                    display_pct = 100.0
                eta_str = f" {display_pct:.0f}%"
            elif task.progress_pct > 0 and task.progress_pct < 100 and task.start_time:
                # Task in progress - calculate ETA from current elapsed time
                # WHY max(0): Clock skew could theoretically make this negative
                elapsed = max(0.0, time.time() - task.start_time)
                eta_seconds = calculate_eta(display_pct, elapsed)
                eta_str = (
                    f" {display_pct:.0f}% Est: {format_duration(eta_seconds)}"
                    if eta_seconds
                    else f" {display_pct:.0f}%"
                )
            else:
                eta_str = f" {display_pct:.0f}%"

            # Add checkmark when complete using Symbols
            complete_symbol = Symbols.get(Symbols.COMPLETE)
            completion_marker = (
                f" {complete_symbol}" if task.state == URLState.COMPLETE else ""
            )

            # Calculate bar length to fill available width
            reserved_space = (
                len(prefix) + len(suffix) + len(eta_str) + 3  # +3 for emoji width
            )
            bar_length = max(20, term_width - panel_padding - reserved_space)

            # Use Symbols for progress bar (ASCII fallback)
            progress_bar = Symbols.make_progress_bar(display_pct, bar_length)

            # WHY color: Green for completed tasks, default for others
            if task.state == URLState.COMPLETE:
                progress_line = f"{prefix}[green]{progress_bar}[/green]{suffix}{eta_str}{completion_marker}"
            elif task.state == URLState.FAILED:
                progress_line = f"{prefix}[red]{progress_bar}[/red]{suffix}{eta_str}"
            else:
                progress_line = (
                    f"{prefix}{progress_bar}{suffix}{eta_str}{completion_marker}"
                )
            lines.append(progress_line)

            # Line 4: Status message (if present)
            if task.status_message:
                # Determine color based on emoji/content
                if (
                    "⚠️" in task.status_message
                    or "Retry" in task.status_message
                    or "timeout" in task.status_message.lower()
                ):
                    msg_style = "yellow"
                elif (
                    "❌" in task.status_message
                    or "Abort" in task.status_message
                    or "Failed" in task.status_message
                ):
                    msg_style = "red"
                elif (
                    "🌐" in task.status_message
                    or "proxy" in task.status_message.lower()
                ):
                    msg_style = "blue"
                else:
                    msg_style = "cyan"
                lines.append(f"  [{msg_style}]{task.status_message}[/{msg_style}]")

            # Add spacing between tasks (except last one)
            # EDGE CASE: visible_task_ids could be empty if scroll_offset is out of bounds
            if visible_task_ids and task_id != visible_task_ids[-1]:
                lines.append("")

        # Scroll indicator
        scroll_info = ""
        if len(task_ids) > visible_tasks:
            scroll_info = f" (showing {start_idx + 1}-{end_idx} of {len(task_ids)})"

        file_sym = Symbols.get(Symbols.FILE)
        content = Text.from_markup("\n".join(lines))
        return Panel(
            content,
            title=f"[bold cyan]{file_sym} URL Processing{scroll_info}[/bold cyan]",
            box=box.ROUNDED,
            height=available_height,
        )

    def start_live_display(self) -> None:
        """Start the live dashboard (after wait_for_start)"""
        if not self.no_tui and not self.live:
            try:
                self.live = Live(
                    self._create_dashboard(),
                    console=self.console,
                    # Use centralized constant for refresh rate - DO NOT hardcode FPS values
                    refresh_per_second=TUI_REFRESH_FPS,
                    screen=True,
                )
                self.live.start()
                logger.debug(f"Live display started at {TUI_REFRESH_FPS} FPS")
            except Exception as e:
                # CRITICAL FIX: Handle live display start failure gracefully
                # WHY: If Rich/Live fails to initialize (terminal issues, permissions),
                # we should fall back to no-TUI mode rather than crash the entire app.
                # DO NOT let TUI initialization failures stop processing.
                logger.error(f"Failed to start live display: {e}")
                self.live = None
                self.no_tui = True  # Fall back to no-TUI mode

    def update_task(
        self,
        task_id: int,
        state: URLState,
        progress_pct: float = 0.0,
        size_in: int | None = None,
        size_out: int | None = None,
        cost: float | None = None,
        error: str = "",
        current_chunk: int = 0,
        total_chunks: int = 0,
        status_message: str = "",
    ) -> None:
        """Update status of a URL task (thread-safe)

        Args:
            task_id: Task identifier
            state: Current state of the task
            progress_pct: Progress percentage (0-100)
            size_in: Input size in bytes
            size_out: Output size in bytes
            cost: Processing cost in USD
            error: Error message (for FAILED state)
            current_chunk: Current chunk being processed (0 if not chunked)
            total_chunks: Total number of chunks (0 if not chunked)
            status_message: Status/error/retry message to display above progress bar
                          Examples (use clear terminology to distinguish timeout sources):
                          - "Warning: AI service (OpenAI) timeout. Retrying in 3s..."
                          - "Warning: Website scraping timeout. Target site slow."
                          - "Error: Source page not found (404). Aborting task."
                          - "Retry: AI returned invalid XML. Retrying..."
        """
        # Use _task_lock for thread-safe task/stats updates
        with self._task_lock:
            task = self.tasks.get(task_id)
            if not task:
                # Log warning instead of silent failure - helps debug invalid task_id issues
                # DO NOT silently return without logging - always trace unexpected conditions
                logger.warning(f"update_task called with unknown task_id={task_id}")
                return

            # Track start time
            old_state = task.state

            # Reset for retry: When transitioning from finished state to PENDING
            # WHY: Retry needs fresh duration/start_time to track new processing time
            # BUG FIX: Without this, retried tasks would keep old duration values
            if (
                old_state in [URLState.COMPLETE, URLState.FAILED]
                and state == URLState.PENDING
            ):
                task.duration = 0.0
                task.start_time = None
                task.error = ""

            if old_state == URLState.PENDING and state in [
                URLState.SCRAPING,
                URLState.PROCESSING,
            ]:
                task.start_time = time.time()

            # Update task
            task.state = state
            # WHY clamp: Prevents progress overflow that could corrupt display
            task.progress_pct = max(0.0, min(100.0, progress_pct))
            # WHY explicit None check: 'or' operator treats 0 as falsy
            # size_in=0 is valid (empty response), should not keep old value
            # DO NOT: Use 'size_in or task.size_in' - breaks on 0
            task.size_in = size_in if size_in is not None else task.size_in
            task.size_out = size_out if size_out is not None else task.size_out
            task.cost = cost if cost is not None else task.cost
            task.error = error
            task.status_message = status_message  # Update status message

            # Update chunk tracking
            if total_chunks > 0:
                task.total_chunks = total_chunks
            if current_chunk > 0:
                task.current_chunk = current_chunk

            # Update duration - ONLY on first transition to finished state
            # WHY: Freeze duration when task completes, don't recalculate on subsequent
            # render_snapshot() calls that trigger update_task() with same state
            # BUG FIX: Previously recalculated duration on every call, causing timer
            # to keep running even after task showed as 100% complete (green bar)
            if task.start_time and state in [URLState.COMPLETE, URLState.FAILED]:
                # Only set duration on FIRST transition to finished state
                if old_state not in [URLState.COMPLETE, URLState.FAILED]:
                    # WHY max(0): Clock skew could theoretically make this negative
                    # Negative duration would break display and indicate system issue
                    task.duration = max(0.0, time.time() - task.start_time)

            # Update global stats
            self._update_stats(old_state, state, cost)

    def _update_stats(
        self, old_state: URLState, new_state: URLState, cost: float | None
    ) -> None:
        """Update global statistics when task state changes.

        WHY MERGING_CHUNKS treated as PROCESSING: No separate 'merging' counter in stats.
        MERGING is a sub-phase of processing, so we count it under 'processing'.
        DO NOT: Add separate merging counter - complicates UI for little benefit.

        CRITICAL: This must only update stats when state ACTUALLY changes.
        render_snapshot() calls update_task() at 20 FPS for all tasks.
        Without this guard, stats balloon to astronomical values (556504% success rate bug).
        """
        # WHY guard: render_snapshot calls update_task on every render (20 FPS).
        # Without this check, completed/failed/cost get incremented every frame!
        # This caused the "556504% success rate" and "$451058 cost" bugs.
        if old_state == new_state:
            return  # No state change - do NOT update stats

        # Decrement old state count
        if old_state == URLState.PENDING:
            self.stats.pending = max(0, self.stats.pending - 1)
        elif old_state == URLState.SCRAPING:
            self.stats.scraping = max(0, self.stats.scraping - 1)
        elif old_state in [URLState.PROCESSING, URLState.MERGING_CHUNKS]:
            # WHY group: MERGING_CHUNKS is a sub-phase of processing
            self.stats.processing = max(0, self.stats.processing - 1)
        elif old_state == URLState.COMPLETE:
            # WHY: Retry resets completed tasks - decrement completed count
            # NOTE: Cost is NOT subtracted here because we want to track TOTAL spending
            # even for retried tasks. The cost was real API usage that happened.
            # DO NOT subtract cost - it would misrepresent actual resource consumption
            self.stats.completed = max(0, self.stats.completed - 1)
        elif old_state == URLState.FAILED:
            # WHY: Retry resets failed tasks - decrement failed count
            # BUG FIX: Without this, retried tasks stay counted as failed
            self.stats.failed = max(0, self.stats.failed - 1)

        # Increment new state count
        if new_state == URLState.PENDING:
            # WHY: Retry resets tasks to pending - increment pending count
            # BUG FIX: Without this, retried tasks aren't counted as pending
            self.stats.pending += 1
        elif new_state == URLState.SCRAPING:
            self.stats.scraping += 1
        elif new_state in [URLState.PROCESSING, URLState.MERGING_CHUNKS]:
            # WHY group: MERGING_CHUNKS is a sub-phase of processing
            self.stats.processing += 1
        elif new_state == URLState.COMPLETE:
            self.stats.completed += 1
            # WHY check None: cost parameter changed from default 0.0 to Optional[None]
            if cost is not None:
                self.stats.total_cost += cost
        elif new_state == URLState.FAILED:
            self.stats.failed += 1

    def update_display(self) -> None:
        """Refresh the live display (called from main thread)"""
        if self.live:
            self.live.update(self._create_dashboard())

    def render_snapshot(self, snapshot: Dict[int, "TaskSnapshot"] | None) -> None:
        """
        Render a snapshot from StatusPipeline by updating internal state.

        This method bridges the gap between the event-driven StatusPipeline
        and the BatchTUIManager's internal state. It updates each task's
        state from the snapshot, then refreshes the display.

        Args:
            snapshot: Dict mapping task_id -> TaskSnapshot from StatusPipeline.
                      Can be None if get_snapshot() returns nothing.

        WHY: StatusPipeline uses event-driven updates via EventBus, but
        BatchTUIManager has its own internal task dictionary. This method
        synchronizes them for rendering.

        Thread Safety: Should only be called from main thread (same as update_display).

        Edge Cases:
        - snapshot=None: Early return (no-op)
        - no_tui=True: Early return (TUI disabled)
        - Empty snapshot: Still calls update_display() to refresh any pending changes
        """
        # CRITICAL: Early return if TUI is disabled
        # WHY: Prevents wasted CPU cycles when running in headless/quiet mode
        if self.no_tui:
            return

        # CRITICAL: Guard against None snapshot
        # WHY: get_snapshot() could theoretically return None in edge cases
        if snapshot is None:
            logger.debug("render_snapshot called with None snapshot - skipping")
            return

        # Update each task from snapshot
        # WHY: Synchronize StatusPipeline's event-driven state with BatchTUI's internal state
        for task_id, task_snap in snapshot.items():
            # Call update_task to synchronize state (handles stats updates automatically)
            self.update_task(
                task_id=task_id,
                state=task_snap.state,
                progress_pct=task_snap.progress_pct,
                size_in=task_snap.size_in,
                size_out=task_snap.size_out,
                cost=task_snap.cost,
                error=task_snap.error,
                current_chunk=task_snap.current_chunk,
                total_chunks=task_snap.total_chunks,
                status_message=task_snap.status_message,
            )

        # Refresh display with updated state
        self.update_display()

    def stop_live_display(self) -> None:
        """Stop the live display and keyboard listener"""
        # Stop keyboard listener first
        self.stop_keyboard_listener()

        if self.live:
            try:
                self.live.stop()
            except Exception as e:
                # CRITICAL FIX: Handle stop failure gracefully
                # WHY: If Rich/Live fails to stop cleanly (terminal corruption,
                # already stopped), we should log but continue, not crash.
                # DO NOT let stop failures prevent program exit.
                logger.error(f"Failed to stop live display cleanly: {e}")
            finally:
                # Always clear reference regardless of stop success
                self.live = None

    def show_final_summary(self, output_dir: str = "") -> None:
        """
        Show beautiful final batch processing summary.

        Args:
            output_dir: Output directory path
        """
        if self.no_tui:
            self._print_simple_summary(output_dir)
            return

        # Stop live display if active
        self.stop_live_display()

        # Clear screen and show summary
        self.console.clear()
        self.console.print()

        # Title
        sparkles = Symbols.get(Symbols.SPARKLES)
        self.console.print(
            Panel(
                f"{sparkles} [bold green]BATCH PROCESSING COMPLETE[/] {sparkles}",
                border_style="green",
                expand=False,
            )
        )
        self.console.print()

        # Main statistics table
        # WHY DEFAULT_TERMINAL_WIDTH: Use constant instead of hardcoded 80
        stats_table = Table(
            show_header=True,
            box=box.ROUNDED,
            expand=False,
            width=DEFAULT_TERMINAL_WIDTH,
        )
        stats_table.add_column("Metric", style="cyan", width=25)
        stats_table.add_column("Value", style="bold green", width=15)
        stats_table.add_column("Details", style="yellow", width=35)

        # Calculate success rate
        success_rate = (
            (self.stats.completed / self.stats.total_urls * 100)
            if self.stats.total_urls > 0
            else 0
        )

        # Time elapsed
        elapsed = datetime.now() - self.stats.start_time
        time_str = format_duration(elapsed.total_seconds())

        # Average cost per URL
        avg_cost = (
            self.stats.total_cost / self.stats.completed
            if self.stats.completed > 0
            else 0
        )

        # Average time per URL
        total_duration = sum(t.duration for t in self.tasks.values() if t.duration > 0)
        avg_time = (
            total_duration / self.stats.completed if self.stats.completed > 0 else 0
        )

        # WHY: Total XML output size from all completed tasks
        # Users requested visibility into how much data was generated
        total_size_out = sum(t.size_out for t in self.tasks.values() if t.size_out > 0)
        total_size_in = sum(t.size_in for t in self.tasks.values() if t.size_in > 0)

        # Format size for display
        def format_size(size_bytes: int) -> str:
            """Format bytes into human-readable string."""
            if size_bytes < 1024:
                return f"{size_bytes} B"
            elif size_bytes < 1024 * 1024:
                return f"{size_bytes / 1024:.1f} KB"
            else:
                return f"{size_bytes / (1024 * 1024):.2f} MB"

        stats_table.add_row(
            "Success Rate",
            f"{success_rate:.1f}%",
            f"{self.stats.completed}/{self.stats.total_urls} URLs",
        )
        stats_table.add_row(
            "Total Cost",
            f"${self.stats.total_cost:.5f}",
            f"~{self.stats.total_cost * 100:.1f} cents",
        )
        stats_table.add_row("Avg Cost/URL", f"${avg_cost:.5f}", "<1 cent per URL")
        stats_table.add_row("Processing Time", time_str, f"~{avg_time:.1f}s per URL")
        # WHY: Show data sizes to help users understand compression/expansion
        stats_table.add_row(
            "Data Scraped",
            format_size(total_size_in),
            f"HTML from {self.stats.completed} pages",
        )
        stats_table.add_row(
            "XML Generated",
            format_size(total_size_out),
            "Structured content output",
        )

        self.console.print(stats_table)
        self.console.print()

        # Output files panel
        if output_dir:
            # WHY DEFAULT_TERMINAL_WIDTH: Use constant instead of hardcoded 80
            files_table = Table(
                show_header=True,
                box=box.SIMPLE,
                expand=False,
                width=DEFAULT_TERMINAL_WIDTH,
            )
            files_table.add_column("Type", style="cyan", width=20)
            files_table.add_column("Location", style="blue", width=55)

            file_sym = Symbols.get(Symbols.FILE)
            link_sym = Symbols.get(Symbols.LINK)
            scrape_sym = Symbols.get(Symbols.SCRAPING)
            files_table.add_row(
                f"{file_sym} XML Output", f"{output_dir}/processed_*.xml"
            )
            files_table.add_row(
                f"{link_sym} Merged XML", f"{output_dir}/merged_output.xml"
            )
            files_table.add_row(f"{file_sym} Error Log", f"{output_dir}/error_log.txt")
            files_table.add_row(f"{scrape_sym} Scraped HTML", f"{output_dir}/*.html")

            folder_sym = Symbols.get(Symbols.FOLDER)
            self.console.print(
                Panel(
                    files_table,
                    title=f"{folder_sym} Output Files",
                    border_style="blue",
                    expand=False,
                )
            )
            self.console.print()

        # Final status
        check = Symbols.get(Symbols.CHECK)
        dot = Symbols.get(Symbols.DOT)
        if self.stats.failed == 0:
            self.console.print(
                f"[green]{check}[/green] All URLs processed successfully!"
            )
            self.console.print(f"[green]{check}[/green] XML validation passed")
            self.console.print(f"[green]{check}[/green] 0 permanent failures")
        else:
            retry_sym = Symbols.get(Symbols.RETRY)
            self.console.print(
                f"[yellow]{retry_sym}[/yellow] {self.stats.failed} URLs failed"
            )
            # Show failed URLs (limited to MAX_FAILED_URLS_TO_SHOW from config)
            # DO NOT hardcode display limits - use centralized constants
            failed_tasks = [
                t for t in self.tasks.values() if t.state == URLState.FAILED
            ]
            for task in failed_tasks[:MAX_FAILED_URLS_TO_SHOW]:
                error_msg = task.error[:50] if task.error else "Unknown error"
                self.console.print(
                    f"  [red]{dot}[/red] Task #{task.task_id}: {error_msg}"
                )
            if len(failed_tasks) > MAX_FAILED_URLS_TO_SHOW:
                remaining = len(failed_tasks) - MAX_FAILED_URLS_TO_SHOW
                self.console.print(f"  [dim]... and {remaining} more[/dim]")

        self.console.print()

    def prompt_retry_failed(self) -> List[str]:
        """
        Ask user if they want to retry failed tasks.

        Returns:
            List of URLs to retry (empty if user declines or non-interactive)

        FAIL-FAST: In non-interactive mode (no_tui, not a TTY), returns empty list.
        This prevents automatic retries in CI/CD or scripted pipelines.
        Use --retry-failed CLI flag for programmatic retry instead.

        THREAD SAFETY: Must be called from main thread after processing completes
        DO NOT: Call this while processing is still running
        """
        failed_tasks = [t for t in self.tasks.values() if t.state == URLState.FAILED]

        # EDGE CASE: No failed tasks - nothing to retry
        if not failed_tasks:
            logger.debug("prompt_retry_failed: No failed tasks to retry")
            return []

        # FAIL-FAST: Non-interactive mode - do NOT auto-retry
        # WHY: Prevents infinite retry loops in automated pipelines
        # Users should use --retry-failed flag for programmatic retry
        if self.no_tui or not sys.stdin.isatty():
            logger.info(
                f"prompt_retry_failed: {len(failed_tasks)} tasks failed, "
                "skipping retry prompt (non-interactive mode)"
            )
            return []

        retry_sym = Symbols.get(Symbols.RETRY)
        self.console.print()
        self.console.print(
            Panel(
                f"[bold yellow]{retry_sym} {len(failed_tasks)} task(s) failed. "
                f"Do you want to retry them?[/bold yellow]",
                border_style="yellow",
                expand=False,
            )
        )
        self.console.print()
        self.console.print(
            "[dim]Press [bold]Y[/bold] to retry, [bold]N[/bold] to skip: [/dim]", end=""
        )

        try:
            # WHY input(): Simple blocking read is acceptable here because
            # processing has completed and we're just waiting for user decision
            # DO NOT: Use this pattern during active processing
            response = input().strip().lower()
            if response in ["y", "yes"]:
                urls_to_retry = [t.url for t in failed_tasks]
                logger.info(f"User chose to retry {len(urls_to_retry)} failed tasks")

                # WHY explicit guidance: The caller (apias.py) will save failed URLs
                # to progress.json, and user can resume with --resume flag.
                # DO NOT: Mislead user by saying "Retrying..." when it just logs URLs
                self.console.print()
                self.console.print(
                    f"[green]{Symbols.get(Symbols.CHECK)}[/green] "
                    f"Marked {len(failed_tasks)} failed task(s) for retry."
                )
                self.console.print()
                self.console.print("[bold cyan]To retry failed URLs, run:[/bold cyan]")
                self.console.print("[dim]  apias --resume <progress.json path>[/dim]")
                self.console.print()
                return urls_to_retry
            else:
                logger.info("User declined to retry failed tasks")
                self.console.print(
                    f"[dim]{Symbols.get(Symbols.DOT)}[/dim] Skipping retry."
                )
                return []
        except EOFError:
            # Input stream closed (e.g., piped input exhausted)
            logger.warning("prompt_retry_failed: EOF received, skipping retry")
            self.console.print()
            return []
        except KeyboardInterrupt:
            # User pressed Ctrl+C - respect their intention to abort
            logger.info("prompt_retry_failed: User pressed Ctrl+C, aborting retry")
            self.console.print()
            return []
        except OSError as e:
            # I/O error (e.g., terminal closed)
            logger.error(f"prompt_retry_failed: I/O error: {e}")
            return []

    def show_circuit_breaker_dialog(
        self, trigger_reason: str, output_dir: str = ""
    ) -> None:
        """
        Show a graceful dialog when circuit breaker stops processing.

        Provides clear, user-friendly information including:
        - What happened (error reason)
        - Progress status (X/Y URLs completed)
        - Data safety reassurance (nothing lost)
        - Exact file locations (progress.json, session.log, output)
        - Exact resume command (copy-paste ready)
        - Next steps (what to do)

        Args:
            trigger_reason: The reason the circuit breaker tripped
            output_dir: Output directory for progress file info
        """
        if self.no_tui:
            self._print_simple_circuit_breaker(trigger_reason, output_dir)
            return

        # Stop live display if active
        self.stop_live_display()

        # Clear screen for clean dialog
        self.console.clear()
        self.console.print()

        # Error header
        stop_sym = Symbols.get(Symbols.STOP)
        self.console.print(
            Panel(
                f"{stop_sym} [bold red]PROCESSING PAUSED[/bold red] {stop_sym}",
                border_style="red",
                expand=False,
            )
        )
        self.console.print()

        # Reason panel
        warning = Symbols.get(Symbols.WARNING)
        self.console.print(
            Panel(
                f"[bold yellow]{warning} {trigger_reason}[/bold yellow]",
                title="[bold red]What Happened[/bold red]",
                border_style="yellow",
                expand=False,
            )
        )
        self.console.print()

        # Data safety reassurance panel
        check = Symbols.get(Symbols.CHECK)
        info_sym = Symbols.get(Symbols.INFO)  # Use INFO symbol for reassurance
        safety_info = [
            f"[green]{info_sym}[/green] [bold green]Your data is completely safe![/bold green]",
            "",
            f"[green]{check}[/green] Progress: {self.stats.completed}/{self.stats.total_urls} URLs completed successfully",
            f"[green]{check}[/green] All processed data has been saved",
            f"[green]{check}[/green] You can resume exactly where you left off",
            f"[green]{check}[/green] No credits wasted - completed work is preserved",
        ]

        self.console.print(
            Panel(
                "\n".join(safety_info),
                title=f"[bold green]{info_sym} Data Safety[/bold green]",
                border_style="green",
                expand=False,
            )
        )
        self.console.print()

        # File locations panel
        file_sym = Symbols.get(Symbols.FILE)
        folder_sym = Symbols.get(Symbols.FOLDER)

        # Build file paths
        from pathlib import Path

        progress_file = Path(output_dir) / "progress.json" if output_dir else None
        session_log = Path(output_dir) / "session.log" if output_dir else None

        file_info = []
        if progress_file:
            file_info.append(
                f"[cyan]{file_sym}[/cyan] Progress file: [bold]{progress_file}[/bold]"
            )
        if session_log:
            file_info.append(
                f"[cyan]{file_sym}[/cyan] Error log: [bold]{session_log}[/bold]"
            )
        if output_dir:
            file_info.append(
                f"[cyan]{folder_sym}[/cyan] Output directory: [bold]{output_dir}[/bold]"
            )

        if file_info:
            self.console.print(
                Panel(
                    "\n".join(file_info),
                    title="[bold cyan]File Locations[/bold cyan]",
                    border_style="cyan",
                    expand=False,
                )
            )
            self.console.print()

        # Resume command panel
        if progress_file:
            resume_cmd = f'apias --resume "{progress_file}"'
            info = Symbols.get(Symbols.INFO)
            self.console.print(
                Panel(
                    f"[bold white]{resume_cmd}[/bold white]",
                    title=f"[bold yellow]{info} Resume Command (copy this)[/bold yellow]",
                    border_style="yellow",
                    expand=False,
                )
            )
            self.console.print()

        # Next steps panel
        steps_info = [
            "[cyan]1.[/cyan] [bold]Add credits[/bold] to your OpenAI account (if quota exceeded)",
            "[cyan]2.[/cyan] [bold]Wait a few minutes[/bold] if rate limited (usually 1-5 minutes)",
            "[cyan]3.[/cyan] [bold]Check session.log[/bold] for detailed error information",
            "[cyan]4.[/cyan] [bold]Resume when ready[/bold] using the command above",
        ]

        self.console.print(
            Panel(
                "\n".join(steps_info),
                title="[bold cyan]Next Steps[/bold cyan]",
                border_style="cyan",
                expand=False,
            )
        )
        self.console.print()

        # Closing reassurance
        self.console.print("[dim]Press Enter to exit...[/dim]")
        input()  # Wait for user acknowledgment

    def show_session_summary(self, session_log_path: str, error_count: int = 0) -> None:
        """
        Show a summary panel at the end of processing with session log location.

        This provides users with:
        - Location of the session.log file for debugging
        - Error count (if any errors occurred)
        - Reassurance that all errors are logged for review
        - Final completion status

        Args:
            session_log_path: Path to the session.log file
            error_count: Number of errors that occurred (optional)
        """
        if self.no_tui:
            # In no_tui mode, just print the session log location
            print(f"\n📋 Session log: {session_log_path}")
            if error_count > 0:
                print(
                    f"⚠️  {error_count} errors occurred - check session.log for details"
                )
            return

        # Show summary panel with session log location
        self.console.print()
        file_sym = Symbols.get(Symbols.FILE)
        info_sym = Symbols.get(Symbols.INFO)
        check = Symbols.get(Symbols.CHECK)
        warning = Symbols.get(Symbols.WARNING)

        summary_lines = []

        # Completion status
        if error_count == 0:
            summary_lines.append(
                f"[green]{check}[/green] [bold green]Processing completed successfully![/bold green]"
            )
        else:
            summary_lines.append(
                f"[yellow]{warning}[/yellow] [bold yellow]Processing completed with {error_count} errors[/bold yellow]"
            )

        summary_lines.append("")

        # Session log location
        summary_lines.append(
            f"[cyan]{file_sym}[/cyan] Session log: [bold]{session_log_path}[/bold]"
        )

        # Additional info
        if error_count > 0:
            summary_lines.append("")
            summary_lines.append(
                f"[dim]{info_sym} Check session.log for detailed error information[/dim]"
            )
        else:
            summary_lines.append("")
            summary_lines.append(
                f"[dim]{info_sym} All operations logged to session.log for your records[/dim]"
            )

        self.console.print(
            Panel(
                "\n".join(summary_lines),
                title=f"[bold cyan]{info_sym} Session Summary[/bold cyan]",
                border_style="cyan" if error_count == 0 else "yellow",
                expand=False,
            )
        )
        self.console.print()

    def _print_simple_circuit_breaker(
        self, trigger_reason: str, output_dir: str = ""
    ) -> None:
        """Print simple circuit breaker message (for --no-tui mode)"""
        print("\n" + "=" * 60)
        print("PROCESSING STOPPED")
        print("=" * 60)
        print(f"Reason: {trigger_reason}")
        print(f"\nProgress saved: {self.stats.completed}/{self.stats.total_urls} URLs")
        if output_dir:
            print(f"Output directory: {output_dir}")
        print("\nTo resume: apias --resume <progress.json>")
        print("=" * 60 + "\n")

    def _print_simple_summary(self, output_dir: str = "") -> None:
        """Print simple text summary (for --no-tui mode)"""
        print("\n" + "=" * 60)
        print("BATCH PROCESSING COMPLETE")
        print("=" * 60)
        success_rate = (
            self.stats.completed / self.stats.total_urls * 100
            if self.stats.total_urls > 0
            else 0
        )
        print(
            f"Success Rate: {self.stats.completed}/{self.stats.total_urls} ({success_rate:.1f}%)"
        )
        print(f"Total Cost: ${self.stats.total_cost:.5f}")
        print(f"Failed: {self.stats.failed}")

        if output_dir:
            print(f"\nOutput Directory: {output_dir}")

        print("=" * 60 + "\n")
