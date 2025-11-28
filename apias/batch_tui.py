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
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
from enum import Enum
import threading
import sys
import time

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich import box

# Import centralized constants - single source of truth for configuration values
# DO NOT hardcode values here - add new constants to config.py instead
from apias.config import (
    TUI_REFRESH_FPS,
    MAX_FAILED_URLS_TO_SHOW,
    FALLBACK_VERSION,
    KEYBOARD_POLL_INTERVAL,
)

# Import shared terminal utilities for cross-platform support
from apias.terminal_utils import (
    Symbols,
    KeyboardListener,
    ProcessState,
    BaseTUIManager,
    detect_terminal_capabilities,
    format_duration,
    calculate_eta,
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
    start_time: Optional[float] = None
    error: str = ""
    # Chunk tracking (for large pages split into multiple LLM requests)
    current_chunk: int = 0  # 0 means not chunked or not started
    total_chunks: int = 0  # 0 means not chunked
    # Status message for errors, retries, warnings (displayed above progress bar)
    status_message: str = ""  # e.g., "⚠️ LLM timeout. Retrying in 3s..."


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
        self.scroll_offset = 0

        # Initialize all URLs as pending tasks
        for idx, url in enumerate(urls, start=1):
            self.tasks[idx] = URLTask(task_id=idx, url=url)

    # Override get_effective_elapsed to use stats.start_time
    def get_effective_elapsed(self) -> float:  # type: ignore[override]
        """Get elapsed time excluding pause duration."""
        return super().get_effective_elapsed(self.stats.start_time)

    def _on_scroll_up(self) -> None:
        """Handle scroll up (registered with keyboard listener)."""
        if self.scroll_offset > 0:
            self.scroll_offset -= 1

    def _on_scroll_down(self) -> None:
        """Handle scroll down (registered with keyboard listener)."""
        max_offset = max(0, len(self.tasks) - 5)
        if self.scroll_offset < max_offset:
            self.scroll_offset += 1

    def start_keyboard_listener(self) -> None:
        """Start keyboard listener with scroll support."""
        super().start_keyboard_listener()
        # Register scroll callbacks
        if self._keyboard_listener:
            self._keyboard_listener.register_callback("up", self._on_scroll_up)
            self._keyboard_listener.register_callback("down", self._on_scroll_down)

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
        while self.process_state == ProcessState.WAITING:
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
            footer_text = (
                f"{clock}  Elapsed: {int(effective_elapsed) // 60:02d}:{int(effective_elapsed) % 60:02d}  |  "
                f"Scroll: {arrow_up}{arrow_down}  |  SPACE: {pause_sym} Pause  |  Ctrl+C: {stop_sym} Stop & Exit  |  "
                f"Last Update: {datetime.now().strftime('%H:%M:%S')}"
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
        table.add_column("Value", style="bold green", justify="center")

        # Calculate progress (center-aligned with fixed width)
        progress_pct = (
            (self.stats.completed + self.stats.failed) / self.stats.total_urls * 100
            if self.stats.total_urls > 0
            else 0
        )

        # Use Symbols for progress bar (ASCII fallback)
        bar_length = 40
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
            f"[red]{self.stats.failed}[/red]"
            if self.stats.failed > 0
            else f"{self.stats.failed}",
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
        """Create scrollable URLs panel with 4 lines per URL (when status_message present)"""
        # Calculate how many complete tasks fit in available height
        # Each task uses 4 lines: header, stats, progress bar, status message (optional)
        # But we use 4 to account for worst case with status messages
        lines_per_task = 4
        visible_tasks = max(
            1, (available_height - 4) // lines_per_task
        )  # -4 for panel borders/title

        # Get tasks to display (scrolled window)
        task_ids = sorted(self.tasks.keys())
        start_idx = self.scroll_offset
        end_idx = min(len(task_ids), start_idx + visible_tasks)
        visible_task_ids = task_ids[start_idx:end_idx]

        # Build display
        lines = []
        for task_id in visible_task_ids:
            task = self.tasks[task_id]

            # Line 1: Task header (task # + URL)
            url_display = task.url if len(task.url) <= 80 else task.url[:77] + "..."
            lines.append(f"[bold cyan]Task #{task_id:02d}:[/bold cyan] {url_display}")

            # Line 2: Stats row
            state_emoji = task.state.get_symbol()
            size_in_kb = task.size_in / 1024 if task.size_in > 0 else 0
            size_out_kb = task.size_out / 1024 if task.size_out > 0 else 0
            duration_str = f"{task.duration:.1f}s" if task.duration > 0 else "0.0s"

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

            # Estimated time remaining using shared utility
            if task.progress_pct > 0 and task.progress_pct < 100 and task.start_time:
                elapsed = time.time() - task.start_time
                eta_seconds = calculate_eta(task.progress_pct, elapsed)
                eta_str = (
                    f" {task.progress_pct:.0f}% Est: {format_duration(eta_seconds)}"
                    if eta_seconds
                    else f" {task.progress_pct:.0f}%"
                )
            else:
                eta_str = f" {task.progress_pct:.0f}%"

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
            progress_bar = Symbols.make_progress_bar(task.progress_pct, bar_length)

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
            if task_id != visible_task_ids[-1]:
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
            self.live = Live(
                self._create_dashboard(),
                console=self.console,
                # Use centralized constant for refresh rate - DO NOT hardcode FPS values
                refresh_per_second=TUI_REFRESH_FPS,
                screen=True,
            )
            self.live.start()
            logger.debug(f"Live display started at {TUI_REFRESH_FPS} FPS")

    def update_task(
        self,
        task_id: int,
        state: URLState,
        progress_pct: float = 0.0,
        size_in: int = 0,
        size_out: int = 0,
        cost: float = 0.0,
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
                          Examples:
                          - "Warning: LLM server timeout. Retrying in 3s..."
                          - "Error: Source page not found. Aborting task."
                          - "Retry: LLM returned invalid XML. Retrying..."
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
            if old_state == URLState.PENDING and state in [
                URLState.SCRAPING,
                URLState.PROCESSING,
            ]:
                task.start_time = time.time()

            # Update task
            task.state = state
            task.progress_pct = progress_pct
            task.size_in = size_in or task.size_in
            task.size_out = size_out or task.size_out
            task.cost = cost or task.cost
            task.error = error
            task.status_message = status_message  # Update status message

            # Update chunk tracking
            if total_chunks > 0:
                task.total_chunks = total_chunks
            if current_chunk > 0:
                task.current_chunk = current_chunk

            # Update duration
            if task.start_time and state in [URLState.COMPLETE, URLState.FAILED]:
                task.duration = time.time() - task.start_time

            # Update global stats
            self._update_stats(old_state, state, cost)

    def _update_stats(
        self, old_state: URLState, new_state: URLState, cost: float
    ) -> None:
        """Update global statistics when task state changes"""
        # Decrement old state count
        if old_state == URLState.PENDING:
            self.stats.pending = max(0, self.stats.pending - 1)
        elif old_state == URLState.SCRAPING:
            self.stats.scraping = max(0, self.stats.scraping - 1)
        elif old_state == URLState.PROCESSING:
            self.stats.processing = max(0, self.stats.processing - 1)

        # Increment new state count
        if new_state == URLState.SCRAPING:
            self.stats.scraping += 1
        elif new_state == URLState.PROCESSING:
            self.stats.processing += 1
        elif new_state == URLState.COMPLETE:
            self.stats.completed += 1
            self.stats.total_cost += cost
        elif new_state == URLState.FAILED:
            self.stats.failed += 1

    def update_display(self) -> None:
        """Refresh the live display (called from main thread)"""
        if self.live:
            self.live.update(self._create_dashboard())

    def stop_live_display(self) -> None:
        """Stop the live display and keyboard listener"""
        # Stop keyboard listener first
        self.stop_keyboard_listener()

        if self.live:
            self.live.stop()
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
        stats_table = Table(show_header=True, box=box.ROUNDED, expand=False, width=80)
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

        self.console.print(stats_table)
        self.console.print()

        # Output files panel
        if output_dir:
            files_table = Table(
                show_header=True, box=box.SIMPLE, expand=False, width=80
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
