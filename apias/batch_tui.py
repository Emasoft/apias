"""
Batch TUI Module for monitoring multiple URLs being processed in parallel.

Provides a scrollable dashboard that shows:
- Multiple URLs being processed concurrently (10-100+)
- Each URL gets 3 lines: header, stats, progress bar
- Real-time updates as threads process URLs
- Scrollable viewport with arrow keys
- "PRESS SPACE TO START" waiting screen
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
from enum import Enum
import threading
import sys
import tty
import termios
import select
import time

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.progress import Progress, BarColumn, TextColumn
from rich.text import Text
from rich import box

# Import version
try:
    from apias import __version__
except ImportError:
    __version__ = "0.1.4"  # Fallback if import fails


class URLState(Enum):
    """States for URL processing"""
    PENDING = "⏳"
    SCRAPING = "🌐"
    PROCESSING = "🔄"
    COMPLETE = "✅"
    FAILED = "❌"


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


class BatchTUIManager:
    """
    TUI manager for batch processing of multiple URLs.

    Displays a scrollable list of URLs with real-time progress.
    Each URL shows: header (task # + URL), stats row, progress bar.
    """

    def __init__(self, urls: List[str], no_tui: bool = False):
        """
        Initialize batch TUI manager.

        Args:
            urls: List of URLs to process
            no_tui: If True, disable Rich TUI
        """
        self.console = Console(force_terminal=True, legacy_windows=False)
        self.no_tui = no_tui
        self.stats = BatchStats(total_urls=len(urls), pending=len(urls))
        self.tasks: Dict[int, URLTask] = {}
        self.live: Optional[Live] = None

        # Keyboard control state
        self.waiting_to_start = True
        self.should_stop = False
        self.keyboard_listener_thread: Optional[threading.Thread] = None
        self.old_terminal_settings = None

        # Scrolling state
        self.scroll_offset = 0

        # Initialize all URLs as pending tasks
        for idx, url in enumerate(urls, start=1):
            self.tasks[idx] = URLTask(task_id=idx, url=url)

    def start_keyboard_listener(self):
        """Start listening for keyboard input (SPACE and arrow keys)"""
        if not self.no_tui and sys.stdin.isatty():
            self.keyboard_listener_thread = threading.Thread(
                target=self._keyboard_listener,
                daemon=True
            )
            self.keyboard_listener_thread.start()

    def _keyboard_listener(self):
        """Listen for SPACE (start/stop) and arrow keys (scroll)"""
        try:
            if not sys.stdin.isatty():
                return

            if self.old_terminal_settings is None:
                self.old_terminal_settings = termios.tcgetattr(sys.stdin)

            tty.setcbreak(sys.stdin.fileno())
            while not self.should_stop:
                if sys.stdin in select.select([sys.stdin], [], [], 0.1)[0]:
                    char = sys.stdin.read(1)
                    if char == ' ':  # SPACE key
                        if self.waiting_to_start:
                            self.waiting_to_start = False
                        else:
                            self.should_stop = True
                            break
                    elif char == '\x1b':  # Escape sequence (arrow keys)
                        # Read next two characters for arrow keys
                        next_chars = sys.stdin.read(2)
                        if next_chars == '[A':  # Up arrow
                            self.scroll_offset = max(0, self.scroll_offset - 1)
                        elif next_chars == '[B':  # Down arrow
                            max_scroll = max(0, len(self.tasks) - 10)  # Adjust based on visible tasks
                            self.scroll_offset = min(max_scroll, self.scroll_offset + 1)
        except (termios.error, OSError):
            pass
        finally:
            # Restore terminal settings
            if self.old_terminal_settings is not None:
                try:
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_terminal_settings)
                except (termios.error, OSError):
                    pass

    def wait_for_start(self):
        """Display waiting screen until user presses SPACE"""
        if self.no_tui or not sys.stdin.isatty():
            self.waiting_to_start = False
            return

        self.start_keyboard_listener()
        self.console.clear()

        self.live = Live(
            self._create_waiting_dashboard(),
            console=self.console,
            refresh_per_second=20,  # Increased to 20 FPS for more fluid updates
            screen=True,
        )
        self.live.start()

        while self.waiting_to_start and not self.should_stop:
            time.sleep(0.1)
            if self.live:
                self.live.update(self._create_waiting_dashboard())

        if self.live and not self.should_stop:
            self.live.update(self._create_dashboard())

    def _create_waiting_dashboard(self) -> Panel:
        """Create the waiting screen shown before processing starts"""
        waiting_msg = Text.assemble(
            ("⏸️  ", "bold yellow"),
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
            title="[bold yellow]⏸️  READY TO START[/bold yellow]",
            title_align="center",
        )

    def _create_dashboard(self) -> Layout:
        """Create the main batch processing dashboard"""
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

        # Header
        layout["header"].update(
            Panel(
                Text(f"🚀 APIAS v{__version__} - Batch Processing Dashboard", justify="center", style="bold cyan"),
                box=box.ROUNDED,
            )
        )

        # Stats
        layout["stats"].update(self._create_stats_panel())

        # URLs list (scrollable)
        layout["urls"].update(self._create_urls_panel(urls_size))

        # Footer
        elapsed = datetime.now() - self.stats.start_time
        layout["footer"].update(
            Text(
                f"⏱️  Elapsed: {elapsed.seconds // 60:02d}:{elapsed.seconds % 60:02d}  |  "
                f"Scroll: ↑↓ arrows  |  Stop: SPACE or Ctrl+C  |  "
                f"Last Update: {datetime.now().strftime('%H:%M:%S')}",
                style="dim",
            )
        )

        return layout

    def _create_stats_panel(self) -> Panel:
        """Create overall statistics panel"""
        table = Table(show_header=False, box=None, expand=True)
        table.add_column("Metric", style="cyan", width=20)
        table.add_column("Value", style="bold green", justify="center")

        # Calculate progress (center-aligned with fixed width)
        progress_pct = (self.stats.completed + self.stats.failed) / self.stats.total_urls * 100 if self.stats.total_urls > 0 else 0
        bar_length = 40
        filled = int((progress_pct / 100) * bar_length)
        progress_bar = "█" * filled + "░" * (bar_length - filled)

        table.add_row("Overall Progress", f"{progress_bar} {progress_pct:.0f}%")
        table.add_row("Total URLs", f"{self.stats.total_urls}")
        table.add_row("Pending", f"{self.stats.pending}")
        table.add_row("Scraping", f"{self.stats.scraping}")
        table.add_row("Processing", f"{self.stats.processing}")
        table.add_row("Completed", f"[green]{self.stats.completed}[/green]")
        table.add_row("Failed", f"[red]{self.stats.failed}[/red]" if self.stats.failed > 0 else f"{self.stats.failed}")
        table.add_row("Total Cost", f"${self.stats.total_cost:.4f} (~{self.stats.total_cost * 100:.1f}¢)")

        return Panel(table, title="[bold cyan]📊 Statistics[/bold cyan]", box=box.ROUNDED)

    def _create_urls_panel(self, available_height: int) -> Panel:
        """Create scrollable URLs panel with 3 lines per URL"""
        # Calculate how many complete tasks (3 lines each) fit in available height
        lines_per_task = 3
        visible_tasks = max(1, (available_height - 4) // lines_per_task)  # -4 for panel borders/title

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
            state_emoji = task.state.value
            size_in_kb = task.size_in / 1024 if task.size_in > 0 else 0
            size_out_kb = task.size_out / 1024 if task.size_out > 0 else 0
            duration_str = f"{task.duration:.1f}s" if task.duration > 0 else "0.0s"

            stats_line = (
                f"  {state_emoji} Status: [yellow]{task.state.name}[/yellow]  |  "
                f"Size: {size_in_kb:.1f}KB → {size_out_kb:.1f}KB  |  "
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

            # Estimated time remaining
            if task.progress_pct > 0 and task.progress_pct < 100 and task.start_time:
                elapsed = time.time() - task.start_time
                estimated_total = (elapsed / task.progress_pct) * 100
                remaining = estimated_total - elapsed
                eta_str = f" {task.progress_pct:.0f}% Est: {remaining:.0f}s"
            else:
                eta_str = f" {task.progress_pct:.0f}%"

            # Add checkmark when complete
            completion_marker = " ✅" if task.state == URLState.COMPLETE else ""

            # Calculate bar length to fill available width
            reserved_space = len(prefix) + len(suffix) + len(eta_str) + len(completion_marker)
            bar_length = max(20, term_width - panel_padding - reserved_space)

            filled = int((task.progress_pct / 100) * bar_length)
            progress_bar = "█" * filled + "░" * (bar_length - filled)

            progress_line = f"{prefix}{progress_bar}{suffix}{eta_str}{completion_marker}"
            lines.append(progress_line)

            # Add spacing between tasks (except last one)
            if task_id != visible_task_ids[-1]:
                lines.append("")

        # Scroll indicator
        scroll_info = ""
        if len(task_ids) > visible_tasks:
            scroll_info = f" (showing {start_idx+1}-{end_idx} of {len(task_ids)})"

        content = Text.from_markup("\n".join(lines))
        return Panel(
            content,
            title=f"[bold cyan]📋 URL Processing{scroll_info}[/bold cyan]",
            box=box.ROUNDED,
            height=available_height,
        )

    def start_live_display(self):
        """Start the live dashboard (after wait_for_start)"""
        if not self.no_tui and not self.live:
            self.live = Live(
                self._create_dashboard(),
                console=self.console,
                refresh_per_second=20,  # Increased to 20 FPS for more fluid updates
                screen=True,
            )
            self.live.start()

    def update_task(self, task_id: int, state: URLState, progress_pct: float = 0.0, size_in: int = 0, size_out: int = 0, cost: float = 0.0, error: str = ""):
        """Update status of a URL task"""
        task = self.tasks.get(task_id)
        if not task:
            return

        # Track start time
        old_state = task.state
        if old_state == URLState.PENDING and state in [URLState.SCRAPING, URLState.PROCESSING]:
            task.start_time = time.time()

        # Update task
        task.state = state
        task.progress_pct = progress_pct
        task.size_in = size_in or task.size_in
        task.size_out = size_out or task.size_out
        task.cost = cost or task.cost
        task.error = error

        # Update duration
        if task.start_time and state in [URLState.COMPLETE, URLState.FAILED]:
            task.duration = time.time() - task.start_time

        # Update global stats
        self._update_stats(old_state, state, cost)

    def _update_stats(self, old_state: URLState, new_state: URLState, cost: float):
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

    def update_display(self):
        """Refresh the live display (called from main thread)"""
        if self.live:
            self.live.update(self._create_dashboard())

    def stop_live_display(self):
        """Stop the live display"""
        if self.live:
            self.live.stop()
            self.live = None

        # Restore terminal settings
        if self.old_terminal_settings is not None:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_terminal_settings)
            except (termios.error, OSError):
                pass
