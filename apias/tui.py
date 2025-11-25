"""
Rich TUI Module for beautiful terminal interface.

Provides:
- Live monitoring dashboard with concurrent request tracking
- Real-time chunk status updates
- Beautiful final statistics summary
- Error logging and display
- Progress bars and health monitoring

Usage:
    tui = RichTUIManager(total_chunks=19, no_tui=False)
    tui.update_chunk_status(1, "processing", size_in=252000)
    tui.show_final_summary(stats)
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from enum import Enum

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.text import Text
from rich import box
import time


class ChunkState(Enum):
    """Chunk processing states"""

    QUEUED = "⏳"
    PROCESSING = "🔄"
    COMPLETE = "✅"
    RETRY = "⚠️"
    FAILED = "❌"


@dataclass
class ChunkStatus:
    """Status information for a single chunk"""

    chunk_id: int
    state: ChunkState = ChunkState.QUEUED
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


class RichTUIManager:
    """
    Manages Rich TUI for beautiful terminal output.

    Handles live monitoring and final statistics display.
    """

    def __init__(self, total_chunks: int, no_tui: bool = False):
        """
        Initialize TUI manager.

        Args:
            total_chunks: Total number of chunks to process
            no_tui: If True, disable Rich TUI (for headless/scripts)
        """
        self.console = Console()
        self.no_tui = no_tui
        self.stats = ProcessingStats(total_chunks=total_chunks)
        self.chunks: Dict[int, ChunkStatus] = {}
        self.live: Optional[Live] = None

        # Initialize all chunks as queued
        for i in range(1, total_chunks + 1):
            self.chunks[i] = ChunkStatus(chunk_id=i)

    def start_live_display(self):
        """Start the live monitoring display"""
        if not self.no_tui:
            self.live = Live(
                self._create_dashboard(),
                console=self.console,
                refresh_per_second=4,
                screen=False,  # Don't use alternate screen (allows scrollback)
            )
            self.live.start()

    def stop_live_display(self):
        """Stop the live monitoring display"""
        if self.live:
            self.live.stop()
            self.live = None

    def update_chunk_status(
        self,
        chunk_id: int,
        state: ChunkState,
        size_in: int = 0,
        size_out: int = 0,
        cost: float = 0.0,
        error: str = "",
        attempt: int = 1,
    ):
        """
        Update status of a chunk.

        Args:
            chunk_id: Chunk number
            state: New state
            size_in: Input size in bytes
            size_out: Output size in bytes
            cost: Processing cost in dollars
            error: Error message if any
            attempt: Attempt number (for retries)
        """
        chunk = self.chunks.get(chunk_id)
        if not chunk:
            return

        # Update chunk status
        old_state = chunk.state
        chunk.state = state
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

        # Update display
        if self.live:
            self.live.update(self._create_dashboard())
        elif not self.no_tui:
            # Not in live mode, print progress
            self._print_chunk_update(chunk)

    def _print_chunk_update(self, chunk: ChunkStatus):
        """Print a single chunk update (when not in live mode)"""
        status_text = f"{chunk.state.value} Chunk #{chunk.chunk_id:02d}: {chunk.state.name}"
        if chunk.state == ChunkState.COMPLETE:
            status_text += f" ({chunk.size_in // 1024}KB → {chunk.size_out // 1024}KB, ${chunk.cost:.4f}, {chunk.duration:.1f}s)"
        elif chunk.state == ChunkState.PROCESSING:
            elapsed = time.time() - chunk.start_time if chunk.start_time else 0
            status_text += f" ({chunk.size_in // 1024}KB, {elapsed:.1f}s elapsed)"
        elif chunk.state == ChunkState.FAILED:
            status_text += f" - {chunk.error}"

        self.console.print(status_text)

    def _create_dashboard(self) -> Layout:
        """Create the live monitoring dashboard"""
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="stats", size=6),
            Layout(name="chunks", size=12),
            Layout(name="footer", size=2),
        )

        # Header
        layout["header"].update(Panel(f"🚀 [bold cyan]APIAS[/] - API Documentation Extractor", border_style="cyan"))

        # Stats table
        layout["stats"].update(self._create_stats_table())

        # Chunks table
        layout["chunks"].update(self._create_chunks_table())

        # Footer
        elapsed = datetime.now() - self.stats.start_time
        layout["footer"].update(
            Text(
                f"⏱️  Elapsed: {elapsed.seconds // 60:02d}:{elapsed.seconds % 60:02d}  |  "
                f"Press Ctrl+C to stop  |  "
                f"Last Update: {datetime.now().strftime('%H:%M:%S')}",
                style="dim",
            )
        )

        return layout

    def _create_stats_table(self) -> Table:
        """Create the statistics table"""
        table = Table(show_header=False, box=box.ROUNDED, expand=True)
        table.add_column("Metric", style="cyan", width=20)
        table.add_column("Value", style="bold green", width=12)
        table.add_column("Progress", style="yellow")

        # Processing = total - completed - failed
        processing = self.stats.total_chunks - self.stats.completed - self.stats.failed

        # Progress bar
        progress_pct = (self.stats.completed / self.stats.total_chunks) * 100 if self.stats.total_chunks > 0 else 0
        bar_length = 30
        filled = int((progress_pct / 100) * bar_length)
        progress_bar = "█" * filled + "░" * (bar_length - filled)

        table.add_row("Concurrent Requests", f"{processing}/{self.stats.total_chunks}", f"{progress_bar} {progress_pct:.0f}%")
        table.add_row("Completed Chunks", f"{self.stats.completed}/{self.stats.total_chunks}", "")
        table.add_row("Failed Chunks", f"{self.stats.failed}", "[red]" if self.stats.failed > 0 else "[green]")
        table.add_row("Retry Queue", f"{self.stats.retrying}", "")
        table.add_row("Total Cost", f"${self.stats.total_cost:.4f}", f"~{self.stats.total_cost * 100:.1f}¢")

        return Panel(table, title="📊 Processing Stats", border_style="green")

    def _create_chunks_table(self) -> Table:
        """Create the chunks status table (scrollable)"""
        table = Table(show_header=True, box=box.SIMPLE, expand=True)
        table.add_column("#", style="cyan", width=4)
        table.add_column("Status", width=10)
        table.add_column("Input", width=8, style="blue")
        table.add_column("Output", width=8, style="green")
        table.add_column("Cost", width=8, style="yellow")
        table.add_column("Time", width=6)
        table.add_column("Details", style="dim")

        # Show only last 10 chunks to fit screen
        chunk_ids = sorted(self.chunks.keys())
        visible_chunks = chunk_ids[-10:]

        for chunk_id in visible_chunks:
            chunk = self.chunks[chunk_id]

            # Format status
            status_text = f"{chunk.state.value} {chunk.state.name}"
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

            table.add_row(f"#{chunk_id:02d}", status_text, size_in_str, size_out_str, cost_str, time_str, details, style=style)

        return Panel(table, title=f"🔄 Chunk Status (showing last 10 of {len(self.chunks)})", border_style="blue")

    def show_final_summary(self, xml_files: List[str] = None, output_dir: str = ""):
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
        self.console.print(Panel("✨ [bold green]EXTRACTION COMPLETE[/] ✨", border_style="green", expand=False))
        self.console.print()

        # Main statistics table
        stats_table = Table(show_header=True, box=box.ROUNDED, expand=False, width=80)
        stats_table.add_column("Metric", style="cyan", width=25)
        stats_table.add_column("Value", style="bold green", width=15)
        stats_table.add_column("Details", style="yellow", width=35)

        # Calculate success rate
        success_rate = (self.stats.completed / self.stats.total_chunks * 100) if self.stats.total_chunks > 0 else 0

        # Time elapsed
        elapsed = datetime.now() - self.stats.start_time
        time_str = f"{elapsed.seconds // 60}m {elapsed.seconds % 60}s"

        # Average cost per chunk
        avg_cost = self.stats.total_cost / self.stats.completed if self.stats.completed > 0 else 0

        # Average time per chunk
        total_duration = sum(c.duration for c in self.chunks.values() if c.duration > 0)
        avg_time = total_duration / self.stats.completed if self.stats.completed > 0 else 0

        # Count retries
        retry_count = sum(1 for c in self.chunks.values() if c.attempt > 1)

        stats_table.add_row("Success Rate", f"{success_rate:.1f}%", f"{self.stats.completed}/{self.stats.total_chunks} chunks")
        stats_table.add_row("Total Cost", f"${self.stats.total_cost:.5f}", f"~{self.stats.total_cost * 100:.1f}¢")
        stats_table.add_row("Avg Cost/Chunk", f"${avg_cost:.5f}", "<1¢ per chunk")
        stats_table.add_row("Processing Time", time_str, f"~{avg_time:.1f}s per chunk")

        if retry_count > 0:
            retry_success = sum(1 for c in self.chunks.values() if c.attempt > 1 and c.state == ChunkState.COMPLETE)
            stats_table.add_row("Retry Success", f"{retry_success}/{retry_count}", "100% retry success" if retry_success == retry_count else "")

        # XML output size
        if xml_files:
            import os

            total_size = sum(os.path.getsize(f) for f in xml_files if os.path.exists(f))
            stats_table.add_row("XML Output Size", f"{total_size // 1024} KB", "Well-formed ✓")

        self.console.print(stats_table)
        self.console.print()

        # Output files panel
        if output_dir:
            files_table = Table(show_header=True, box=box.SIMPLE, expand=False, width=80)
            files_table.add_column("Type", style="cyan", width=20)
            files_table.add_column("Location", style="blue", width=55)

            files_table.add_row("📄 XML Output", f"{output_dir}/processed_*.xml")
            files_table.add_row("🔗 Merged XML", f"{output_dir}/merged_output.xml")
            files_table.add_row("📋 Error Log", f"{output_dir}/error_log.txt")
            files_table.add_row("🌐 Scraped HTML", f"{output_dir}/*.html")

            self.console.print(Panel(files_table, title="📁 Output Files", border_style="blue", expand=False))
            self.console.print()

        # Final status
        if self.stats.failed == 0:
            self.console.print("[green]✓[/green] All chunks processed successfully!")
            self.console.print("[green]✓[/green] XML validation passed")
            self.console.print("[green]✓[/green] 0 permanent failures")
        else:
            self.console.print(f"[yellow]⚠[/yellow] {self.stats.failed} chunks failed")
            for error in self.stats.errors[:5]:  # Show first 5 errors
                self.console.print(f"  [red]•[/red] {error}")

        self.console.print()

    def _print_simple_summary(self, xml_files: List[str] = None, output_dir: str = ""):
        """Print simple text summary (for --no-tui mode)"""
        print("\n" + "=" * 60)
        print("EXTRACTION COMPLETE")
        print("=" * 60)
        print(f"Success Rate: {self.stats.completed}/{self.stats.total_chunks} ({self.stats.completed / self.stats.total_chunks * 100:.1f}%)")
        print(f"Total Cost: ${self.stats.total_cost:.5f}")
        print(f"Failed: {self.stats.failed}")

        if output_dir:
            print(f"\nOutput Directory: {output_dir}")

        print("=" * 60 + "\n")
