"""
User-friendly dialogs without blocking TUI or worker threads.

This module implements a non-blocking dialog system that queues dialogs by priority
and shows them AFTER the TUI stops. This ensures that dialogs never interfere with
TUI rendering, and critical dialogs (circuit breaker, auth failure) are shown first.

Key Design Principles:
- Non-blocking: Dialogs are queued, not shown immediately
- Priority-based: Critical dialogs shown first (circuit breaker → error summary → info)
- Post-TUI: All dialogs shown AFTER TUI stops (no interference)
- Event-driven: Subscribe to CircuitBreakerEvent and other events
- Rich formatting: Use Rich panels for beautiful, readable dialogs

Architecture:
    Worker Thread → ErrorCollector → CircuitBreakerEvent → EventBus
                                                              ↓
    DialogManager subscribes → Queue dialog → Show after TUI stops

Usage:
    # Initialize
    dialog_manager = DialogManager(event_bus, console)

    # Worker threads trigger events (automatic)
    # CircuitBreakerEvent → dialog_manager._queue_circuit_breaker_dialog()

    # After TUI stops, show all pending dialogs
    dialog_manager.show_pending_dialogs(output_dir, session_log)
"""

import logging
import queue
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from apias.event_system import (
    CircuitBreakerEvent,
    DialogEvent,
    DialogPriority,
    DialogType,
    ErrorCategory,
    EventBus,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Dialog Request
# ============================================================================


@dataclass
class DialogRequest:
    """
    Request to show a dialog to the user.

    This is queued by priority and rendered after TUI stops.
    The priority determines the order in which dialogs are shown.
    """

    priority: DialogPriority
    dialog_type: DialogType
    context: Dict[str, Any]


# ============================================================================
# Dialog Manager
# ============================================================================


class DialogManager:
    """
    Non-blocking dialog system with priority queue.

    This class manages user dialogs that should be shown after TUI operation
    completes. Dialogs are queued by priority and rendered in order after
    the TUI stops, ensuring zero interference with live TUI rendering.

    Thread Safety:
    - queue.PriorityQueue is thread-safe (worker threads can trigger dialogs)
    - show_pending_dialogs() should only be called from main thread after TUI stops

    Priority Ordering:
    - CRITICAL (0): Circuit breaker, auth failure (shown first)
    - HIGH (1): Error summary with failures
    - NORMAL (2): General info
    - LOW (3): Success confirmation (shown last)

    Design Note: We use a priority queue instead of a regular queue to ensure
    critical dialogs (circuit breaker, quota exceeded) are shown before less
    important dialogs (success confirmation, info messages).
    """

    def __init__(self, event_bus: EventBus, console: Console):
        """
        Initialize dialog manager.

        Args:
            event_bus: EventBus instance for subscribing to events
            console: Rich Console for rendering dialogs

        Design Note: We accept Console as a parameter so tests can inject
        a test console with custom settings (width, recording, etc.).
        """
        self._event_bus = event_bus
        self._console = console

        # Priority queue: (priority_value, DialogRequest)
        # WHY PriorityQueue: Ensures critical dialogs shown first
        # Lower priority value = higher priority (CRITICAL=0 shown before LOW=3)
        self._dialog_queue: queue.PriorityQueue = queue.PriorityQueue()

        # Dialog counter for queue ordering (tie-breaker for same priority)
        # WHY: If two dialogs have same priority, show in FIFO order
        self._dialog_counter = 0

        # Subscribe to events that trigger dialogs
        event_bus.subscribe(CircuitBreakerEvent, self._queue_circuit_breaker_dialog)
        event_bus.subscribe(DialogEvent, self._queue_dialog_event)

        logger.debug("DialogManager initialized")

    def _queue_circuit_breaker_dialog(self, event: CircuitBreakerEvent) -> None:
        """
        Queue high-priority dialog for circuit breaker trip.

        This is called automatically when a CircuitBreakerEvent is published.
        The dialog is queued with CRITICAL priority to ensure it's shown first.

        Args:
            event: CircuitBreakerEvent with trip details

        Design Note: We queue the dialog instead of showing it immediately
        because the TUI may still be running. The dialog will be shown after
        the TUI stops via show_pending_dialogs().
        """
        dialog = DialogRequest(
            priority=DialogPriority.CRITICAL,
            dialog_type=DialogType.CIRCUIT_BREAKER,
            context={
                "reason": event.reason,
                "affected_tasks": event.affected_tasks,
                "trigger_category": event.trigger_category,
                "consecutive_counts": event.consecutive_counts,
                "timestamp": event.timestamp,
            },
        )

        # Queue with (priority, counter, dialog) for stable ordering
        # WHY counter: Ensures FIFO ordering for dialogs with same priority
        self._dialog_queue.put((dialog.priority.value, self._dialog_counter, dialog))
        self._dialog_counter += 1

        logger.info(f"Queued CRITICAL dialog: Circuit breaker tripped - {event.reason}")

    def _queue_dialog_event(self, event: DialogEvent) -> None:
        """
        Queue generic dialog event.

        This is called when a DialogEvent is published with arbitrary context.
        Used for error summaries, confirmations, and info messages.

        Args:
            event: DialogEvent with type, priority, and context

        Design Note: This is the generic entry point for dialogs. Specific
        event types (CircuitBreakerEvent) have dedicated handlers for richer
        context extraction.
        """
        dialog = DialogRequest(
            priority=event.priority,
            dialog_type=event.dialog_type,
            context=event.context,
        )

        self._dialog_queue.put((dialog.priority.value, self._dialog_counter, dialog))
        self._dialog_counter += 1

        logger.debug(f"Queued {event.priority.name} dialog: {event.dialog_type.name}")

    def queue_error_summary(
        self,
        total_errors: int,
        error_breakdown: Dict[ErrorCategory, int],
        recent_errors: list,
        output_dir: Path,
        session_log: Path,
    ) -> None:
        """
        Queue error summary dialog for end of session.

        This is called manually at the end of processing to summarize all
        errors that occurred during the session.

        Args:
            total_errors: Total number of errors
            error_breakdown: Count of errors per category
            recent_errors: List of recent ErrorEvent objects
            output_dir: Path to output directory
            session_log: Path to session.log file

        Design Note: This is a manual entry point (not event-driven) because
        the error summary is generated at the end of processing, not in
        response to a single event.
        """
        dialog = DialogRequest(
            priority=DialogPriority.HIGH if total_errors > 0 else DialogPriority.NORMAL,
            dialog_type=DialogType.ERROR_SUMMARY,
            context={
                "total_errors": total_errors,
                "error_breakdown": error_breakdown,
                "recent_errors": recent_errors,
                "output_dir": output_dir,
                "session_log": session_log,
            },
        )

        self._dialog_queue.put((dialog.priority.value, self._dialog_counter, dialog))
        self._dialog_counter += 1

        logger.info(f"Queued error summary dialog: {total_errors} total errors")

    def show_pending_dialogs(
        self,
        output_dir: Optional[Path] = None,
        session_log: Optional[Path] = None,
    ) -> None:
        """
        Show all pending dialogs in priority order.

        This should be called AFTER the TUI stops, from the main thread.
        Dialogs are shown in priority order (CRITICAL → HIGH → NORMAL → LOW).

        Args:
            output_dir: Path to output directory (for file references in dialogs)
            session_log: Path to session.log file (for error investigation)

        Design Note: We process dialogs until the queue is empty, which ensures
        all queued dialogs are shown even if more are queued during rendering
        (though this shouldn't happen after TUI stops).

        Thread Safety: Should only be called from main thread after TUI stops.
        """
        if self._dialog_queue.empty():
            logger.debug("No pending dialogs to show")
            return

        dialog_count = self._dialog_queue.qsize()
        logger.info(f"Showing {dialog_count} pending dialogs")

        shown_count = 0
        while not self._dialog_queue.empty():
            try:
                # Get highest priority dialog
                # Format: (priority_value, counter, DialogRequest)
                priority_value, counter, dialog = self._dialog_queue.get_nowait()

                # Render dialog
                self._render_dialog(dialog, output_dir, session_log)
                shown_count += 1

            except queue.Empty:
                # Queue is empty (shouldn't happen due to while condition, but safe)
                break

        logger.info(f"Showed {shown_count} dialogs")

    def _render_dialog(
        self,
        dialog: DialogRequest,
        output_dir: Optional[Path],
        session_log: Optional[Path],
    ) -> None:
        """
        Render dialog using Rich panels.

        Dispatches to specific dialog renderer based on dialog type.

        Args:
            dialog: DialogRequest with type and context
            output_dir: Path to output directory
            session_log: Path to session.log file

        Design Note: This is a dispatcher method that routes to specific
        renderers. Each dialog type has its own rendering logic.
        """
        if dialog.dialog_type == DialogType.CIRCUIT_BREAKER:
            self._render_circuit_breaker(dialog.context, output_dir, session_log)
        elif dialog.dialog_type == DialogType.ERROR_SUMMARY:
            self._render_error_summary(dialog.context)
        elif dialog.dialog_type == DialogType.CONFIRMATION:
            self._render_confirmation(dialog.context)
        elif dialog.dialog_type == DialogType.INFO:
            self._render_info(dialog.context)
        else:
            logger.warning(f"Unknown dialog type: {dialog.dialog_type}")

    def _render_circuit_breaker(
        self,
        context: Dict[str, Any],
        output_dir: Optional[Path],
        session_log: Optional[Path],
    ) -> None:
        """
        Render circuit breaker dialog.

        Shows user why processing stopped, which tasks were affected,
        and what actions they can take next.

        Args:
            context: Dialog context with reason, affected_tasks, etc.
            output_dir: Path to output directory
            session_log: Path to session.log file

        Design Note: This uses the existing circuit breaker dialog design
        with Rich panels and tables for maximum clarity.
        """
        reason = context.get("reason", "Unknown reason")
        affected_tasks = context.get("affected_tasks", [])
        trigger_category = context.get("trigger_category")

        # Build title with emoji
        title = "🛑 Processing Paused - Circuit Breaker Tripped"

        # Build main message
        message = Text()
        message.append(
            "Processing has been paused to prevent further errors.\n\n",
            style="bold yellow",
        )
        message.append(f"Reason: {reason}\n", style="red")

        if trigger_category:
            message.append(f"Error Category: {trigger_category.name}\n", style="dim")

        if affected_tasks:
            message.append(
                f"\nAffected Tasks: {len(affected_tasks)} tasks were not completed\n",
                style="yellow",
            )

        # Add next steps
        message.append("\n📋 Next Steps:\n", style="bold cyan")
        message.append(
            "1. Check session.log for detailed error information\n", style="dim"
        )
        message.append(
            "2. Review error_thresholds.yaml to adjust circuit breaker settings\n",
            style="dim",
        )
        message.append(
            "3. Fix the underlying issue (quota, auth, network, etc.)\n", style="dim"
        )
        message.append(
            "4. Resume processing with progress.json to continue from where you left off\n",
            style="dim",
        )

        # Add file paths if available
        if session_log:
            message.append(f"\n📄 Session Log: {session_log}\n", style="dim")
        if output_dir:
            message.append(f"📁 Output Directory: {output_dir}\n", style="dim")

        # Create panel
        panel = Panel(
            message,
            title=title,
            border_style="red",
            padding=(1, 2),
        )

        # Show panel
        self._console.print()
        self._console.print(panel)
        self._console.print()

        # Wait for user acknowledgment
        self._console.input("Press Enter to continue... ")

    def _render_error_summary(self, context: Dict[str, Any]) -> None:
        """
        Render error summary dialog.

        Shows breakdown of all errors that occurred during processing.

        Args:
            context: Dialog context with total_errors, error_breakdown, etc.

        Design Note: This creates a Rich table with error categories and counts.
        """
        total_errors = context.get("total_errors", 0)
        error_breakdown = context.get("error_breakdown", {})
        recent_errors = context.get("recent_errors", [])

        if total_errors == 0:
            # No errors - show success message
            self._console.print(
                "✅ Processing completed successfully with no errors!",
                style="bold green",
            )
            return

        # Build title
        title = f"📊 Error Summary - {total_errors} Total Errors"

        # Create error breakdown table
        table = Table(title="Error Breakdown by Category", show_header=True)
        table.add_column("Category", style="cyan")
        table.add_column("Count", justify="right", style="yellow")
        table.add_column("Recoverable", style="green")

        for category, count in sorted(
            error_breakdown.items(), key=lambda x: x[1], reverse=True
        ):
            recoverable = (
                "✅"
                if category.name.endswith("_TIMEOUT")
                or category.name.startswith("CONNECTION")
                else "❌"
            )
            table.add_row(category.name, str(count), recoverable)

        # Create panel
        panel = Panel(
            table,
            title=title,
            border_style="yellow",
            padding=(1, 2),
        )

        # Show panel
        self._console.print()
        self._console.print(panel)
        self._console.print()

        # Show recent errors sample
        if recent_errors and len(recent_errors) > 0:
            self._console.print("Recent Errors (last 5):", style="bold")
            for error_event in recent_errors[:5]:
                self._console.print(
                    f"  • {error_event.category.name}: {error_event.message}",
                    style="dim",
                )
            self._console.print()

    def _render_confirmation(self, context: Dict[str, Any]) -> None:
        """
        Render confirmation dialog.

        Asks user for confirmation (yes/no).

        Args:
            context: Dialog context with message and default

        Design Note: This is a simple confirmation dialog with yes/no input.
        """
        message = context.get("message", "Confirm action?")
        default = context.get("default", "no")

        # Show message
        self._console.print(f"\n{message}", style="bold yellow")

        # Get user input
        response = (
            self._console.input(f"Continue? (yes/no) [default: {default}]: ")
            .strip()
            .lower()
        )

        if not response:
            response = default

        # Store response in context for caller to check
        context["response"] = response in ("y", "yes")

    def _render_info(self, context: Dict[str, Any]) -> None:
        """
        Render informational dialog.

        Shows a simple info message.

        Args:
            context: Dialog context with message

        Design Note: This is a simple info dialog with no user interaction.
        """
        message = context.get("message", "")
        title = context.get("title", "ℹ️ Information")

        # Create panel
        panel = Panel(
            message,
            title=title,
            border_style="blue",
            padding=(1, 2),
        )

        # Show panel
        self._console.print()
        self._console.print(panel)
        self._console.print()

    def get_pending_count(self) -> int:
        """
        Get number of pending dialogs in queue.

        Returns:
            Number of dialogs waiting to be shown

        Design Note: Useful for testing and logging.
        """
        return self._dialog_queue.qsize()

    def clear_pending_dialogs(self) -> int:
        """
        Clear all pending dialogs from queue.

        Used for testing or emergency reset.

        Returns:
            Number of dialogs cleared

        Design Note: This is mainly for testing. In production, all dialogs
        should be shown via show_pending_dialogs().
        """
        count = 0
        while not self._dialog_queue.empty():
            try:
                self._dialog_queue.get_nowait()
                count += 1
            except queue.Empty:
                break

        logger.info(f"Cleared {count} pending dialogs from queue")
        return count
