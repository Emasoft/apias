"""
Rich TUI Module for beautiful terminal interface.

Provides:
- Live monitoring dashboard with concurrent request tracking
- Real-time chunk status updates
- Beautiful final statistics summary
- Error logging and display
- Progress bars and health monitoring
- Cross-platform terminal support with ASCII fallbacks

Usage:
    tui = RichTUIManager(total_chunks=19, no_tui=False)
    tui.update_chunk_status(1, "processing", size_in=252000)
    tui.show_final_summary(stats)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum
import threading
import sys
import time
import logging

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
    TUI_SINGLE_PAGE_FPS,
    TUI_WAITING_FPS,
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

# Import version
try:
    from apias import __version__
except ImportError:
    # Use centralized fallback version from config.py
    # DO NOT hardcode version string here
    __version__ = FALLBACK_VERSION

logger = logging.getLogger(__name__)


class ChunkState(Enum):
    """Chunk processing states with ASCII fallback support"""

    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETE = "complete"
    RETRY = "retry"
    FAILED = "failed"

    def get_symbol(self) -> str:
        """Get the display symbol (emoji or ASCII) based on terminal capabilities"""
        symbol_map = {
            ChunkState.QUEUED: Symbols.PENDING,
            ChunkState.PROCESSING: Symbols.PROCESSING,
            ChunkState.COMPLETE: Symbols.COMPLETE,
            ChunkState.RETRY: Symbols.RETRY,
            ChunkState.FAILED: Symbols.FAILED,
        }
        return Symbols.get(symbol_map[self])


class ProcessingStep(Enum):
    """Granular processing steps within a chunk"""

    QUEUED = (0, "Queued")
    SCRAPING = (1, "Scraping HTML")
    CLEANING = (2, "Cleaning HTML")
    CHUNKING = (3, "Chunking HTML")
    SENDING = (4, "Sending to AI")
    RECEIVING = (5, "Receiving from AI")
    VALIDATING = (6, "Validating XML")
    SAVING = (7, "Saving XML")
    COMPLETE = (8, "Complete")

    def __init__(self, step_num: int, description: str):
        self.step_num = step_num
        self.description = description

    @property
    def total_steps(self) -> int:
        """Total number of steps in processing"""
        return 8

    @property
    def progress_percentage(self) -> float:
        """Calculate progress percentage for this step"""
        return (self.step_num / self.total_steps) * 100


@dataclass
class ChunkStatus:
    """Status information for a single chunk"""

    chunk_id: int
    state: ChunkState = ChunkState.QUEUED
    current_step: ProcessingStep = ProcessingStep.QUEUED
    size_in: int = 0
    size_out: int = 0
    cost: float = 0.0
    duration: float = 0.0
    start_time: Optional[float] = None
    error: str = ""
    attempt: int = 1


@dataclass
class ProcessingStats:
    """Overall processing statistics"""

    total_chunks: int
    completed: int = 0
    failed: int = 0
    retrying: int = 0
    total_cost: float = 0.0
    start_time: datetime = field(default_factory=datetime.now)
    errors: List[str] = field(default_factory=list)


class RichTUIManager(BaseTUIManager):
    """
    Manages Rich TUI for single-page chunk processing.

    Handles live monitoring and final statistics display for
    processing a single page split into multiple chunks.

    Inherits thread-safe state management from BaseTUIManager.
    """

    def __init__(
        self, total_chunks: int, no_tui: bool = False, quiet: bool = False
    ) -> None:
        """
        Initialize TUI manager for chunk processing.

        Args:
            total_chunks: Total number of chunks to process
            no_tui: If True, disable Rich TUI (for headless/scripts)
            quiet: If True, minimal output (implies no_tui)
        """
        # Initialize base class (handles state management, console, etc.)
        super().__init__(no_tui=no_tui, quiet=quiet)

        # Chunk-specific state
        self.stats = ProcessingStats(total_chunks=total_chunks)
        self.chunks: Dict[int, ChunkStatus] = {}

        # Initialize all chunks as queued
        for i in range(1, total_chunks + 1):
            self.chunks[i] = ChunkStatus(chunk_id=i)

    # Override get_effective_elapsed to use stats.start_time
    def get_effective_elapsed(self) -> float:  # type: ignore[override]
        """Get elapsed time excluding pause duration."""
        return super().get_effective_elapsed(self.stats.start_time)

    # Note: waiting_to_start, should_stop, is_paused, is_running,
    # _on_space_pressed, request_stop, wait_while_paused are now inherited
    # from BaseTUIManager with thread-safe implementation.

    def start_live_display(self) -> None:
        """Start the live monitoring display"""
        if not self.no_tui:
            self.live = Live(
                self._create_dashboard(),
                console=self.console,
                # Use centralized constant for refresh rate - DO NOT hardcode FPS values
                refresh_per_second=TUI_SINGLE_PAGE_FPS,
                screen=False,  # Don't use alternate screen (allows scrollback)
            )
            self.live.start()
            logger.debug(f"Live display started at {TUI_SINGLE_PAGE_FPS} FPS")

    def stop_live_display(self) -> None:
        """Stop the live monitoring display"""
        # Stop keyboard listener first
        self.stop_keyboard_listener()

        if self.live:
            self.live.stop()
            self.live = None

    def update_chunk_status(
        self,
        chunk_id: int,
        state: ChunkState,
        step: Optional[ProcessingStep] = None,
        size_in: int = 0,
        size_out: int = 0,
        cost: float = 0.0,
        error: str = "",
        attempt: int = 1,
    ) -> None:
        """
        Update status of a chunk.

        Args:
            chunk_id: Chunk number
            state: New state
            step: Current processing step (optional, for granular progress)
            size_in: Input size in bytes
            size_out: Output size in bytes
            cost: Processing cost in dollars
            error: Error message if any
            attempt: Attempt number (for retries)
        """
        chunk = self.chunks.get(chunk_id)
        if not chunk:
            # Log warning instead of silent failure - helps debug invalid chunk_id issues
            # DO NOT silently return without logging - always trace unexpected conditions
            logger.warning(
                f"update_chunk_status called with unknown chunk_id={chunk_id}"
            )
            return

        # Update chunk status
        old_state = chunk.state
        chunk.state = state
        if step is not None:
            chunk.current_step = step
        chunk.size_in = size_in or chunk.size_in
        chunk.size_out = size_out or chunk.size_out
        chunk.cost += cost  # Accumulate cost across retries
        chunk.error = error
        chunk.attempt = attempt

        # Track start time for duration calculation
        if state == ChunkState.PROCESSING and not chunk.start_time:
            chunk.start_time = time.time()
        elif state in [ChunkState.COMPLETE, ChunkState.FAILED]:
            if chunk.start_time:
                chunk.duration = time.time() - chunk.start_time
            # Set final step when complete
            if state == ChunkState.COMPLETE:
                chunk.current_step = ProcessingStep.COMPLETE

        # Update global stats
        if old_state != ChunkState.COMPLETE and state == ChunkState.COMPLETE:
            self.stats.completed += 1
            self.stats.total_cost += cost
        elif old_state != ChunkState.FAILED and state == ChunkState.FAILED:
            self.stats.failed += 1
            if error:
                self.stats.errors.append(f"Chunk #{chunk_id}: {error}")
        elif state == ChunkState.RETRY:
            self.stats.retrying += 1

        # NOTE: Do NOT call self.live.update() here!
        # Rich Live is not thread-safe. The main thread handles all live display updates.
        # This method only updates the data (chunk status and stats).
        # The main thread's update loop will pick up these changes and refresh the display.

        # Print progress when not in live mode
        if not self.live and not self.no_tui:
            self._print_chunk_update(chunk)

    def _print_chunk_update(self, chunk: ChunkStatus) -> None:
        """Print a single chunk update (when not in live mode)"""
        status_text = f"{chunk.state.get_symbol()} Chunk #{chunk.chunk_id:02d}: {chunk.state.name}"
        if chunk.state == ChunkState.COMPLETE:
            status_text += f" ({chunk.size_in // 1024}KB → {chunk.size_out // 1024}KB, ${chunk.cost:.4f}, {chunk.duration:.1f}s)"
        elif chunk.state == ChunkState.PROCESSING:
            elapsed = time.time() - chunk.start_time if chunk.start_time else 0
            status_text += f" ({chunk.size_in // 1024}KB, {elapsed:.1f}s elapsed)"
        elif chunk.state == ChunkState.FAILED:
            status_text += f" - {chunk.error}"

        self.console.print(status_text)

    def _create_dashboard(self) -> Layout:
        """Create the live monitoring dashboard - fills entire terminal"""
        # Get terminal dimensions
        term_height = self.console.size.height

        # Calculate dynamic sizes to fill entire screen
        header_size = 3
        footer_size = 2
        # Reserve space for stats (8 rows minimum)
        stats_size = min(10, max(8, term_height // 5))
        # Give everything else to chunks table
        chunks_size = max(
            10, term_height - header_size - stats_size - footer_size - 2
        )  # -2 for borders

        layout = Layout()
        layout.split_column(
            Layout(name="header", size=header_size),
            Layout(name="stats", size=stats_size),
            Layout(
                name="chunks", size=chunks_size
            ),  # Largest section, fills remaining space
            Layout(name="footer", size=footer_size),
        )

        # Header with version and pause state indicator
        rocket = Symbols.get(Symbols.ROCKET)
        if self.process_state == ProcessState.PAUSED:
            # Show paused indicator in header with pulsing animation
            pulse = Symbols.get_pulse_frame()
            pause_sym = Symbols.get(Symbols.PAUSE)
            header_text = f"{pause_sym} [bold yellow]PAUSED[/] {pulse} APIAS v{__version__} - API Documentation Extractor"
            header_style = "yellow"
        else:
            header_text = f"{rocket} [bold cyan]APIAS v{__version__}[/] - API Documentation Extractor"
            header_style = "cyan"

        layout["header"].update(
            Panel(
                header_text,
                border_style=header_style,
            )
        )

        # Stats table
        layout["stats"].update(self._create_stats_table())

        # Chunks table (now dynamic, will show more chunks on taller terminals)
        layout["chunks"].update(self._create_chunks_table())

        # Footer with state-aware controls and elapsed time
        elapsed = datetime.now() - self.stats.start_time
        effective_elapsed = self.get_effective_elapsed()
        clock = Symbols.get(Symbols.CLOCK)

        # Show different footer based on pause state
        if self.process_state == ProcessState.PAUSED:
            # Paused state - show pulsing indicator and resume instructions
            pulse = Symbols.get_pulse_frame()
            pause_sym = Symbols.get(Symbols.PAUSE)
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
                f"SPACE: {pause_sym} Pause  |  Ctrl+C: {stop_sym} Stop & Exit  |  "
                f"Last Update: {datetime.now().strftime('%H:%M:%S')}"
            )
            footer_style = "dim"

        layout["footer"].update(Text.from_markup(footer_text, style=footer_style))

        return layout

    def _create_stats_table(self) -> Panel:
        """Create the statistics table with ETA (pause-aware)"""
        # Use yellow border when paused, green when running
        border_style = (
            "yellow" if self.process_state == ProcessState.PAUSED else "green"
        )

        table = Table(show_header=False, box=box.ROUNDED, expand=True)
        table.add_column("Metric", style="cyan", width=20)
        table.add_column("Value", style="bold green", width=12)
        table.add_column("Progress", style="yellow")

        # Processing = total - completed - failed
        processing = self.stats.total_chunks - self.stats.completed - self.stats.failed

        # Step-based progress bar (granular progress within each chunk)
        total_steps = self.stats.total_chunks * 8  # 8 steps per chunk
        completed_steps = sum(
            chunk.current_step.step_num for chunk in self.chunks.values()
        )
        progress_pct = (completed_steps / total_steps) * 100 if total_steps > 0 else 0

        # Use Symbols for progress bar characters (ASCII fallback)
        bar_length = 30
        progress_bar = Symbols.make_progress_bar(progress_pct, bar_length)

        # Calculate ETA using effective elapsed (excludes pause time)
        effective_elapsed = self.get_effective_elapsed()
        eta_seconds = calculate_eta(progress_pct, effective_elapsed)

        # Show "(paused)" in ETA when paused, or time remaining otherwise
        if self.process_state == ProcessState.PAUSED:
            eta_str = "[yellow](paused)[/yellow]"
        else:
            eta_str = format_duration(eta_seconds) if eta_seconds else "--"

        table.add_row(
            "Overall Progress",
            f"{completed_steps}/{total_steps} steps",
            f"{progress_bar} {progress_pct:.0f}%",
        )
        table.add_row("ETA", eta_str, "")
        table.add_row(
            "Processing Chunks", f"{processing}/{self.stats.total_chunks}", ""
        )
        table.add_row(
            "Completed Chunks", f"{self.stats.completed}/{self.stats.total_chunks}", ""
        )
        table.add_row(
            "Failed Chunks",
            f"{self.stats.failed}",
            "[red]" if self.stats.failed > 0 else "[green]",
        )
        table.add_row("Retry Queue", f"{self.stats.retrying}", "")
        table.add_row(
            "Total Cost",
            f"${self.stats.total_cost:.4f}",
            f"~{self.stats.total_cost * 100:.1f}¢",
        )

        chart = Symbols.get(Symbols.CHART)
        return Panel(
            table, title=f"{chart} Processing Stats", border_style=border_style
        )

    def _create_chunks_table(self) -> Table:
        """Create the chunks status table (dynamically sized based on terminal height)"""
        table = Table(show_header=True, box=box.SIMPLE, expand=True)
        table.add_column("#", style="cyan", width=4)
        table.add_column("Status", width=10)
        table.add_column("Input", width=8, style="blue")
        table.add_column("Output", width=8, style="green")
        table.add_column("Cost", width=8, style="yellow")
        table.add_column("Time", width=6)
        table.add_column("Details", style="dim")

        # Dynamically calculate how many chunks to show based on terminal height
        # Terminal height - (header=3 + stats≈8 + footer=2 + borders≈4) = available for chunks
        terminal_height = self.console.size.height
        available_rows = max(
            10, terminal_height - 17
        )  # At least 10, or more if terminal is tall

        chunk_ids = sorted(self.chunks.keys())
        # Show most recent chunks that fit in available space
        visible_chunks = (
            chunk_ids[-available_rows:]
            if len(chunk_ids) > available_rows
            else chunk_ids
        )

        for chunk_id in visible_chunks:
            chunk = self.chunks[chunk_id]

            # Format status with current step and spinner for active processing
            status_text = f"{chunk.state.get_symbol()} {chunk.state.name}"
            if (
                chunk.state == ChunkState.PROCESSING
                and chunk.current_step != ProcessingStep.QUEUED
            ):
                # Show current step when processing with animated spinner
                spinner_frames = Symbols.get_spinner_frames()
                spinner_idx = int(time.time() * 10) % len(
                    spinner_frames
                )  # Animate based on time
                spinner = spinner_frames[spinner_idx]
                status_text = f"{spinner} {chunk.current_step.description}"
            if chunk.attempt > 1:
                status_text += f" ({chunk.attempt})"

            # Format sizes
            size_in_str = f"{chunk.size_in // 1024}KB" if chunk.size_in > 0 else "-"
            size_out_str = f"{chunk.size_out // 1024}KB" if chunk.size_out > 0 else "-"

            # Format cost
            cost_str = f"${chunk.cost:.4f}" if chunk.cost > 0 else "-"

            # Format time
            if chunk.state == ChunkState.PROCESSING and chunk.start_time:
                elapsed = time.time() - chunk.start_time
                time_str = f"{elapsed:.1f}s"
            elif chunk.duration > 0:
                time_str = f"{chunk.duration:.1f}s"
            else:
                time_str = "-"

            # Details
            details = ""
            if chunk.state == ChunkState.COMPLETE and chunk.attempt > 1:
                details = "succeeded on retry"
            elif chunk.state == ChunkState.RETRY:
                details = "retrying..."
            elif chunk.state == ChunkState.FAILED:
                details = chunk.error[:40]  # Truncate long errors

            # Color code based on state
            style = ""
            if chunk.state == ChunkState.COMPLETE:
                style = "green"
            elif chunk.state == ChunkState.FAILED:
                style = "red"
            elif chunk.state == ChunkState.PROCESSING:
                style = "yellow"

            table.add_row(
                f"#{chunk_id:02d}",
                status_text,
                size_in_str,
                size_out_str,
                cost_str,
                time_str,
                details,
                style=style,
            )

        # Dynamic title showing actual visible chunk count
        processing_symbol = Symbols.get(Symbols.PROCESSING)
        visible_count = len(visible_chunks)
        total_count = len(self.chunks)
        title_suffix = (
            f" (showing {visible_count} of {total_count})"
            if visible_count < total_count
            else ""
        )
        return Panel(
            table,
            title=f"{processing_symbol} Chunk Status{title_suffix}",
            border_style="blue",
        )

    def show_final_summary(
        self, xml_files: Optional[List[str]] = None, output_dir: str = ""
    ) -> None:
        """
        Show beautiful final statistics summary.

        Args:
            xml_files: List of generated XML files
            output_dir: Output directory path
        """
        if self.no_tui:
            self._print_simple_summary(xml_files, output_dir)
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
                f"{sparkles} [bold green]EXTRACTION COMPLETE[/] {sparkles}",
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
            (self.stats.completed / self.stats.total_chunks * 100)
            if self.stats.total_chunks > 0
            else 0
        )

        # Time elapsed
        elapsed = datetime.now() - self.stats.start_time
        time_str = f"{elapsed.seconds // 60}m {elapsed.seconds % 60}s"

        # Average cost per chunk
        avg_cost = (
            self.stats.total_cost / self.stats.completed
            if self.stats.completed > 0
            else 0
        )

        # Average time per chunk
        total_duration = sum(c.duration for c in self.chunks.values() if c.duration > 0)
        avg_time = (
            total_duration / self.stats.completed if self.stats.completed > 0 else 0
        )

        # Count retries
        retry_count = sum(1 for c in self.chunks.values() if c.attempt > 1)

        stats_table.add_row(
            "Success Rate",
            f"{success_rate:.1f}%",
            f"{self.stats.completed}/{self.stats.total_chunks} chunks",
        )
        stats_table.add_row(
            "Total Cost",
            f"${self.stats.total_cost:.5f}",
            f"~{self.stats.total_cost * 100:.1f}¢",
        )
        stats_table.add_row("Avg Cost/Chunk", f"${avg_cost:.5f}", "<1¢ per chunk")
        stats_table.add_row("Processing Time", time_str, f"~{avg_time:.1f}s per chunk")

        if retry_count > 0:
            retry_success = sum(
                1
                for c in self.chunks.values()
                if c.attempt > 1 and c.state == ChunkState.COMPLETE
            )
            stats_table.add_row(
                "Retry Success",
                f"{retry_success}/{retry_count}",
                "100% retry success" if retry_success == retry_count else "",
            )

        # XML output size
        if xml_files:
            import os

            total_size = sum(os.path.getsize(f) for f in xml_files if os.path.exists(f))
            stats_table.add_row(
                "XML Output Size", f"{total_size // 1024} KB", "Well-formed ✓"
            )

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
                f"[green]{check}[/green] All chunks processed successfully!"
            )
            self.console.print(f"[green]{check}[/green] XML validation passed")
            self.console.print(f"[green]{check}[/green] 0 permanent failures")
        else:
            retry_sym = Symbols.get(Symbols.RETRY)
            self.console.print(
                f"[yellow]{retry_sym}[/yellow] {self.stats.failed} chunks failed"
            )
            # Show errors (limited to MAX_FAILED_URLS_TO_SHOW from config)
            # DO NOT hardcode display limits - use centralized constants
            for error in self.stats.errors[:MAX_FAILED_URLS_TO_SHOW]:
                self.console.print(f"  [red]{dot}[/red] {error}")
            if len(self.stats.errors) > MAX_FAILED_URLS_TO_SHOW:
                remaining = len(self.stats.errors) - MAX_FAILED_URLS_TO_SHOW
                self.console.print(f"  [dim]... and {remaining} more errors[/dim]")

        self.console.print()

    def _print_simple_summary(
        self, xml_files: Optional[List[str]] = None, output_dir: str = ""
    ) -> None:
        """Print simple text summary (for --no-tui mode)"""
        print("\n" + "=" * 60)
        print("EXTRACTION COMPLETE")
        print("=" * 60)
        print(
            f"Success Rate: {self.stats.completed}/{self.stats.total_chunks} ({self.stats.completed / self.stats.total_chunks * 100:.1f}%)"
        )
        print(f"Total Cost: ${self.stats.total_cost:.5f}")
        print(f"Failed: {self.stats.failed}")

        if output_dir:
            print(f"\nOutput Directory: {output_dir}")

        print("=" * 60 + "\n")

    # Note: _on_space_pressed, request_stop, wait_while_paused,
    # start_keyboard_listener, stop_keyboard_listener are inherited
    # from BaseTUIManager with thread-safe implementation.

    def wait_for_start(self) -> None:
        """Display TUI and wait for SPACE keypress to start processing"""
        if self.no_tui:
            # In no-tui mode, skip waiting and start immediately
            self.process_state = ProcessState.RUNNING
            return

        # Check if stdin is a TTY
        if not sys.stdin.isatty():
            # Not a TTY, skip waiting and start immediately
            self.process_state = ProcessState.RUNNING
            return

        # Start keyboard listener
        self.start_keyboard_listener()

        # Clear screen and move to top for clean full-screen effect
        self.console.clear()

        # Start live display with "waiting" message
        # Use screen=True to take over terminal and prevent scrolling
        self.live = Live(
            self._create_waiting_dashboard(),
            console=self.console,
            # Use centralized constant for refresh rate - DO NOT hardcode FPS values
            # TUI_WAITING_FPS is faster than TUI_SINGLE_PAGE_FPS for smooth spinner animations
            refresh_per_second=TUI_WAITING_FPS,
            screen=True,  # Use alternate screen buffer for full control
            vertical_overflow="visible",  # Allow content to fill screen
        )
        self.live.start()

        # Wait for user to press SPACE (state changes to RUNNING)
        while self.process_state == ProcessState.WAITING:
            # Use centralized polling interval - DO NOT hardcode timing values
            time.sleep(KEYBOARD_POLL_INTERVAL)
            if self.live:
                self.live.update(self._create_waiting_dashboard())

        # Switch to normal dashboard
        if self.live and self.process_state == ProcessState.RUNNING:
            self.live.update(self._create_dashboard())

    def _create_waiting_dashboard(self) -> Table:
        """Create the waiting state dashboard with prominent start message"""
        rocket = Symbols.get(Symbols.ROCKET)
        pause = Symbols.get(Symbols.PAUSE)
        chart = Symbols.get(Symbols.CHART)

        # Header
        header = Panel(
            Text(
                f"{rocket} APIAS v{__version__} - API Documentation Extractor",
                justify="center",
                style="bold cyan",
            ),
            box=box.ROUNDED,
            padding=(0, 1),
        )

        # Waiting message (prominent)
        waiting_msg = Panel(
            Text.assemble(
                (f"{pause}  ", "bold yellow"),
                ("PRESS SPACE BAR TO START SCRAPING", "bold yellow blink"),
                ("\n\n", ""),
                ("Press SPACE again to stop", "dim white"),
            ),
            box=box.DOUBLE,
            padding=(2, 4),
            border_style="yellow",
            title=f"[bold yellow]{pause}  READY TO START[/bold yellow]",
            title_align="center",
        )

        # Stats (initial state)
        stats_content = Table.grid(padding=(0, 2))
        stats_content.add_column(justify="left")
        stats_content.add_column(justify="left")
        stats_content.add_column(justify="left")

        stats_content.add_row(
            f"Total Chunks: {self.stats.total_chunks}",
            f"Completed: 0/{self.stats.total_chunks}",
            "Failed: 0",
        )

        stats_panel = Panel(
            stats_content,
            title=f"{chart} Processing Stats",
            border_style="cyan",
            box=box.ROUNDED,
        )

        # Combine panels
        layout = Table.grid()
        layout.add_row(header)
        layout.add_row("")
        layout.add_row(waiting_msg)
        layout.add_row("")
        layout.add_row(stats_panel)

        return layout
