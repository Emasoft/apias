# APIAS TUI Communication System Overhaul Plan

## Executive Summary

This plan addresses critical UX issues in the APIAS TUI (Terminal User Interface) including:
- TUI "earthquake" vibrations from uncontrolled console output
- Missing root cause analysis in failure summaries
- No data safety reassurance or resume instructions
- Infrequent progress bar updates
- Lack of circuit breaker for API quota/rate limit errors

---

## Part 1: Problem Analysis

### 1.1 Current Architecture Issues

**A. Console Output Conflicts**
```
Location: apias/apias.py (lines 294-315)
```
- `suppress_console_logging()` removes StreamHandlers during TUI
- BUT: Logger calls in exception handlers still trigger before suppression
- Direct `print()` calls in `_print_simple_summary()` bypass TUI entirely
- After `stop_live_display()`, accumulated errors flood the console

**B. Error Handling Gaps**
```
Location: apias/apias.py (lines 1583-1605)
```
- API errors (RateLimitError, APITimeoutError, etc.) are logged and re-raised
- No tracking of error CATEGORIES for summary reporting
- No circuit breaker - continues trying after quota exhausted
- Errors propagate up but lose their classification

**C. Summary Missing Root Cause**
```
Location: apias/batch_tui.py (lines 597-729)
```
- `show_final_summary()` shows success/failure counts
- Does NOT show WHY failures occurred (quota, rate limit, etc.)
- No data safety messaging
- No resume instructions

**D. Progress Update Granularity**
```
Location: apias/batch_tui.py (line 186)
```
- TUI refresh rate is 20 FPS (good)
- BUT: Progress updates only happen at state transitions
- Scraping phase has no intermediate progress signals
- Large chunks appear "stuck" during processing

---

## Part 2: Solution Architecture

### 2.1 New Module: `apias/error_handler.py`

```python
"""
Centralized error classification and circuit breaker for APIAS.

Provides:
- ErrorCategory enum for classifying errors
- CircuitBreaker for stopping on fatal errors
- SessionErrorTracker for aggregating errors across tasks
"""
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from datetime import datetime
import threading

class ErrorCategory(Enum):
    """Classification of error types for summary reporting."""
    NONE = auto()
    RATE_LIMIT = auto()           # 429 Too Many Requests
    QUOTA_EXCEEDED = auto()       # Insufficient quota
    API_TIMEOUT = auto()          # Request timeout
    CONNECTION_ERROR = auto()     # Network/connection failure
    INVALID_RESPONSE = auto()     # XML validation failed
    SOURCE_NOT_FOUND = auto()     # Page not found (404)
    AUTHENTICATION = auto()       # API key invalid
    SERVER_ERROR = auto()         # 5xx errors
    UNKNOWN = auto()              # Unclassified errors

@dataclass
class ErrorEvent:
    """Single error occurrence with full context."""
    category: ErrorCategory
    message: str
    task_id: Optional[int]
    url: Optional[str]
    timestamp: datetime = field(default_factory=datetime.now)
    recoverable: bool = True
    raw_exception: Optional[str] = None

class CircuitBreaker:
    """
    Stops processing when fatal errors occur.

    Triggers on:
    - QUOTA_EXCEEDED (immediate stop - no point continuing)
    - RATE_LIMIT with > consecutive_threshold failures
    - CONNECTION_ERROR with > consecutive_threshold failures
    """
    def __init__(
        self,
        consecutive_threshold: int = 3,
        quota_immediate_stop: bool = True
    ):
        self.consecutive_threshold = consecutive_threshold
        self.quota_immediate_stop = quota_immediate_stop
        self._consecutive_errors = 0
        self._last_error_category: Optional[ErrorCategory] = None
        self._triggered = False
        self._trigger_reason: Optional[str] = None
        self._lock = threading.Lock()

    def record_success(self) -> None:
        """Reset consecutive error count on success."""
        with self._lock:
            self._consecutive_errors = 0
            self._last_error_category = None

    def record_error(self, category: ErrorCategory) -> bool:
        """
        Record an error and check if circuit should trip.

        Returns:
            True if circuit has tripped (should stop processing)
        """
        with self._lock:
            # Immediate stop for quota exceeded
            if category == ErrorCategory.QUOTA_EXCEEDED and self.quota_immediate_stop:
                self._triggered = True
                self._trigger_reason = "API quota exceeded - no further requests possible"
                return True

            # Track consecutive errors
            if category == self._last_error_category:
                self._consecutive_errors += 1
            else:
                self._consecutive_errors = 1
                self._last_error_category = category

            # Check threshold
            if self._consecutive_errors >= self.consecutive_threshold:
                self._triggered = True
                self._trigger_reason = f"{self._consecutive_errors} consecutive {category.name} errors"
                return True

            return self._triggered

    @property
    def is_triggered(self) -> bool:
        return self._triggered

    @property
    def trigger_reason(self) -> Optional[str]:
        return self._trigger_reason

class SessionErrorTracker:
    """
    Tracks all errors across a processing session for summary reporting.
    """
    def __init__(self):
        self.errors: List[ErrorEvent] = []
        self.circuit_breaker = CircuitBreaker()
        self._lock = threading.Lock()

    def record(self, event: ErrorEvent) -> bool:
        """
        Record an error event.

        Returns:
            True if processing should stop (circuit breaker triggered)
        """
        with self._lock:
            self.errors.append(event)
        return self.circuit_breaker.record_error(event.category)

    def get_primary_failure_reason(self) -> Optional[str]:
        """
        Get the most significant reason for failures.

        Priority:
        1. Circuit breaker trigger reason
        2. Most frequent error category
        3. First error message
        """
        if self.circuit_breaker.is_triggered:
            return self.circuit_breaker.trigger_reason

        if not self.errors:
            return None

        # Count by category
        category_counts: Dict[ErrorCategory, int] = {}
        for event in self.errors:
            category_counts[event.category] = category_counts.get(event.category, 0) + 1

        # Find most common
        most_common = max(category_counts.items(), key=lambda x: x[1])
        category, count = most_common

        return f"{count} tasks failed due to {category.name.replace('_', ' ').lower()}"

    def get_error_summary(self) -> Dict[ErrorCategory, int]:
        """Get count of errors by category."""
        summary: Dict[ErrorCategory, int] = {}
        for event in self.errors:
            summary[event.category] = summary.get(event.category, 0) + 1
        return summary
```

### 2.2 New Module: `apias/message_queue.py`

```python
"""
Thread-safe message queue for TUI-safe output.

All log messages and status updates flow through this queue
to prevent console output from corrupting the TUI display.
"""
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, List, Callable
from datetime import datetime
import threading
import queue

class MessageLevel(Enum):
    """Message priority/type for display formatting."""
    DEBUG = auto()
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    SUCCESS = auto()
    STATUS = auto()      # Task status updates
    PROGRESS = auto()    # Progress bar updates

@dataclass
class TUIMessage:
    """Single message for TUI display."""
    level: MessageLevel
    text: str
    task_id: Optional[int] = None
    timestamp: datetime = field(default_factory=datetime.now)
    persist: bool = False  # If True, show in deferred summary

class TUIMessageQueue:
    """
    Thread-safe message queue that integrates with TUI managers.

    Usage:
        queue = TUIMessageQueue()
        queue.put(TUIMessage(MessageLevel.WARNING, "Rate limit approaching"))

        # In TUI render loop:
        for msg in queue.drain():
            display_message(msg)
    """
    def __init__(self, max_size: int = 1000):
        self._queue: queue.Queue[TUIMessage] = queue.Queue(maxsize=max_size)
        self._deferred: List[TUIMessage] = []
        self._lock = threading.Lock()
        self._callbacks: List[Callable[[TUIMessage], None]] = []

    def put(self, message: TUIMessage) -> None:
        """Add a message to the queue (non-blocking)."""
        try:
            self._queue.put_nowait(message)
        except queue.Full:
            # Drop oldest message if queue is full
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(message)
            except queue.Empty:
                pass

        # Store persistent messages for summary
        if message.persist:
            with self._lock:
                self._deferred.append(message)

        # Notify callbacks
        for callback in self._callbacks:
            callback(message)

    def drain(self) -> List[TUIMessage]:
        """Get all pending messages (clears queue)."""
        messages = []
        while True:
            try:
                messages.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return messages

    def get_deferred(self) -> List[TUIMessage]:
        """Get messages marked for deferred display."""
        with self._lock:
            return list(self._deferred)

    def register_callback(self, callback: Callable[[TUIMessage], None]) -> None:
        """Register callback for real-time message handling."""
        self._callbacks.append(callback)

# Global instance for module-level access
_global_queue: Optional[TUIMessageQueue] = None

def get_message_queue() -> TUIMessageQueue:
    """Get or create the global message queue."""
    global _global_queue
    if _global_queue is None:
        _global_queue = TUIMessageQueue()
    return _global_queue
```

---

## Part 3: Implementation Changes

### 3.1 Modify `apias/apias.py`

**A. Add error classification to API calls (lines ~1583-1605)**

```python
from apias.error_handler import (
    ErrorCategory, ErrorEvent, SessionErrorTracker,
    classify_openai_error
)

# Add helper function
def classify_openai_error(e: Exception) -> ErrorCategory:
    """Classify OpenAI exception into error category."""
    error_str = str(e).lower()
    error_type = type(e).__name__

    if isinstance(e, RateLimitError):
        if "quota" in error_str or "insufficient_quota" in error_str:
            return ErrorCategory.QUOTA_EXCEEDED
        return ErrorCategory.RATE_LIMIT
    elif isinstance(e, APITimeoutError):
        return ErrorCategory.API_TIMEOUT
    elif isinstance(e, APIConnectionError):
        return ErrorCategory.CONNECTION_ERROR
    elif isinstance(e, APIStatusError):
        if hasattr(e, 'status_code'):
            if e.status_code >= 500:
                return ErrorCategory.SERVER_ERROR
            elif e.status_code == 401:
                return ErrorCategory.AUTHENTICATION
        return ErrorCategory.UNKNOWN
    else:
        return ErrorCategory.UNKNOWN

# Modify make_openai_request exception handling:
except RateLimitError as e:
    category = classify_openai_error(e)
    # Record to session tracker (passed as parameter or global)
    error_tracker.record(ErrorEvent(
        category=category,
        message=str(e),
        task_id=task_id,
        url=current_url,
        recoverable=(category != ErrorCategory.QUOTA_EXCEEDED)
    ))
    logger.error(f"Rate limit exceeded (429): {str(e)}")
    raise
```

**B. Add session error tracker to main workflow**

```python
# In main_workflow(), add:
from apias.error_handler import SessionErrorTracker

error_tracker = SessionErrorTracker()

# Pass to process functions and check circuit breaker:
if error_tracker.circuit_breaker.is_triggered:
    logger.warning(f"Circuit breaker triggered: {error_tracker.circuit_breaker.trigger_reason}")
    # Graceful shutdown instead of continuing
    break
```

### 3.2 Modify `apias/batch_tui.py`

**A. Add error tracker and root cause to `BatchTUIManager.__init__`**

```python
def __init__(self, urls: List[str], no_tui: bool = False, quiet: bool = False):
    # ... existing code ...

    # Add error tracking
    from apias.error_handler import SessionErrorTracker, ErrorCategory
    self.error_tracker = SessionErrorTracker()
    self.primary_failure_reason: Optional[str] = None
```

**B. Enhance `show_final_summary()` with root cause**

```python
def show_final_summary(self, output_dir: str = "") -> None:
    # ... existing code for stats table ...

    # ADD: Root Cause Analysis Section
    if self.stats.failed > 0:
        self.console.print()

        # Get error breakdown
        error_summary = self.error_tracker.get_error_summary()
        primary_reason = self.error_tracker.get_primary_failure_reason()

        # Root cause panel
        root_cause_content = []
        root_cause_content.append(f"[bold red]Primary Cause:[/] {primary_reason or 'Unknown'}")
        root_cause_content.append("")
        root_cause_content.append("[yellow]Error Breakdown:[/]")

        for category, count in sorted(error_summary.items(), key=lambda x: -x[1]):
            if category != ErrorCategory.NONE:
                icon = self._get_error_icon(category)
                root_cause_content.append(f"  {icon} {category.name.replace('_', ' ')}: {count}")

        self.console.print(Panel(
            "\n".join(root_cause_content),
            title="[bold red]Failure Analysis[/]",
            border_style="red",
            expand=False,
        ))

    # ADD: Data Safety & Resume Section
    self.console.print()
    safety_content = []

    # Data safety assurance
    safety_content.append("[bold green]Data Safety:[/]")
    safety_content.append(f"  {Symbols.get(Symbols.CHECK)} All scraped HTML files are preserved")
    safety_content.append(f"  {Symbols.get(Symbols.CHECK)} Successfully converted XML files are saved")
    safety_content.append(f"  {Symbols.get(Symbols.CHECK)} Progress state saved to progress.json")
    safety_content.append("")

    # Resume instructions
    if self.stats.failed > 0:
        safety_content.append("[bold cyan]To Resume This Job:[/]")
        safety_content.append(f"  apias --resume {output_dir}/progress.json")
        safety_content.append("")
        safety_content.append("[dim]This will retry only the failed URLs, preserving all completed work.[/]")

    self.console.print(Panel(
        "\n".join(safety_content),
        title="[bold blue]Recovery Options[/]",
        border_style="blue",
        expand=False,
    ))

def _get_error_icon(self, category: ErrorCategory) -> str:
    """Get icon for error category."""
    icons = {
        ErrorCategory.RATE_LIMIT: "429",
        ErrorCategory.QUOTA_EXCEEDED: "$$$",
        ErrorCategory.API_TIMEOUT: "...",
        ErrorCategory.CONNECTION_ERROR: "NET",
        ErrorCategory.INVALID_RESPONSE: "XML",
        ErrorCategory.SOURCE_NOT_FOUND: "404",
        ErrorCategory.AUTHENTICATION: "KEY",
        ErrorCategory.SERVER_ERROR: "5xx",
        ErrorCategory.UNKNOWN: "???",
    }
    return f"[{icons.get(category, '???')}]"
```

### 3.3 Modify `apias/terminal_utils.py`

**A. Add message queue integration to `BaseTUIManager`**

```python
from apias.message_queue import TUIMessageQueue, TUIMessage, MessageLevel, get_message_queue

class BaseTUIManager:
    def __init__(self, no_tui: bool = False, quiet: bool = False):
        # ... existing code ...

        # Message queue for TUI-safe output
        self.message_queue = get_message_queue()
        self._pending_messages: List[TUIMessage] = []

    def queue_message(self, level: MessageLevel, text: str, persist: bool = False) -> None:
        """Queue a message for TUI display."""
        msg = TUIMessage(level=level, text=text, persist=persist)
        self.message_queue.put(msg)

    def get_pending_messages(self) -> List[TUIMessage]:
        """Get messages to display in TUI."""
        return self.message_queue.drain()
```

### 3.4 Add Progress Callbacks for Scraping

**In `apias/apias.py`, modify `web_scraper()` to emit progress:**

```python
def web_scraper(
    url: str,
    no_tui: bool = False,
    progress_callback: Optional[Callable[[str, float], None]] = None
) -> Optional[str]:
    """
    Scrape URL with optional progress callback.

    Args:
        progress_callback: Called with (status_message, progress_pct)
    """
    if progress_callback:
        progress_callback("Connecting...", 10.0)

    content = start_scraping(url, no_tui=no_tui)

    if progress_callback:
        progress_callback("Processing HTML...", 50.0)

    # ... rest of function
```

---

## Part 4: Test Plan

### 4.1 Unit Tests for Error Handler

```python
# tests/unit/test_error_handler.py

def test_classify_quota_exceeded():
    """Quota exceeded is correctly classified."""
    from openai import RateLimitError
    # Mock exception with quota message
    # Assert returns ErrorCategory.QUOTA_EXCEEDED

def test_circuit_breaker_immediate_stop_on_quota():
    """Circuit breaker triggers immediately on quota exceeded."""
    breaker = CircuitBreaker(quota_immediate_stop=True)
    assert breaker.record_error(ErrorCategory.QUOTA_EXCEEDED) == True
    assert breaker.is_triggered == True

def test_circuit_breaker_consecutive_threshold():
    """Circuit breaker triggers after N consecutive errors."""
    breaker = CircuitBreaker(consecutive_threshold=3)
    assert breaker.record_error(ErrorCategory.RATE_LIMIT) == False
    assert breaker.record_error(ErrorCategory.RATE_LIMIT) == False
    assert breaker.record_error(ErrorCategory.RATE_LIMIT) == True

def test_session_tracker_primary_reason():
    """Session tracker identifies primary failure reason."""
    tracker = SessionErrorTracker()
    tracker.record(ErrorEvent(ErrorCategory.RATE_LIMIT, "msg", None, None))
    tracker.record(ErrorEvent(ErrorCategory.RATE_LIMIT, "msg", None, None))
    tracker.record(ErrorEvent(ErrorCategory.TIMEOUT, "msg", None, None))

    reason = tracker.get_primary_failure_reason()
    assert "rate limit" in reason.lower()
```

### 4.2 Integration Tests

```python
# tests/integration/test_tui_error_display.py

def test_summary_shows_root_cause():
    """Final summary displays root cause of failures."""
    # Set up mock batch with known errors
    # Verify root cause panel is rendered

def test_circuit_breaker_stops_processing():
    """Processing stops when circuit breaker triggers."""
    # Trigger quota exceeded
    # Verify remaining tasks are not processed

def test_resume_instructions_shown():
    """Resume instructions appear in summary when failures occur."""
    # Run batch with failures
    # Verify resume command is shown
```

---

## Part 5: Implementation Order

### Phase 1: Error Infrastructure (Low Risk)
1. Create `apias/error_handler.py` with ErrorCategory, CircuitBreaker, SessionErrorTracker
2. Create `apias/message_queue.py` with TUIMessageQueue
3. Add unit tests for new modules

### Phase 2: Integration (Medium Risk)
4. Integrate error classification into `make_openai_request()`
5. Add SessionErrorTracker to `main_workflow()`
6. Pass error tracker through to BatchTUIManager

### Phase 3: TUI Enhancements (Higher Risk)
7. Modify `show_final_summary()` to show root cause
8. Add data safety and resume instructions
9. Integrate message queue into BaseTUIManager
10. Add progress callbacks to scraping

### Phase 4: Testing & Polish
11. Run full test suite
12. Manual testing with mock API
13. Manual testing with real API (rate limit scenarios)

---

## Part 6: Risk Mitigation

### 6.1 Backwards Compatibility
- All new parameters have defaults
- Error tracking is additive (doesn't change existing behavior)
- Circuit breaker is opt-in initially

### 6.2 Rollback Strategy
- Each phase can be reverted independently
- Feature flags can disable new components
- Existing logging remains as fallback

### 6.3 Performance Considerations
- Message queue uses non-blocking operations
- Error tracking uses minimal memory
- No additional API calls

---

## Acceptance Criteria

1. **No TUI Earthquakes**: Console output during TUI display is eliminated
2. **Root Cause Visible**: Summary shows WHY failures occurred (quota, rate limit, etc.)
3. **Data Safety**: Clear message that scraped data is preserved
4. **Resume Path**: Explicit command shown for resuming failed jobs
5. **Circuit Breaker**: Processing stops on quota exceeded
6. **Fluid Progress**: Progress bars update during scraping phase
7. **All Tests Pass**: Existing + new tests pass

---

## Files to Create/Modify

### New Files:
- `apias/error_handler.py` - Error classification and circuit breaker
- `apias/message_queue.py` - TUI-safe message queue
- `tests/unit/test_error_handler.py` - Unit tests for error handling
- `tests/unit/test_message_queue.py` - Unit tests for message queue

### Modified Files:
- `apias/apias.py` - Add error classification, circuit breaker integration
- `apias/batch_tui.py` - Enhanced summary with root cause, data safety
- `apias/terminal_utils.py` - Message queue integration in BaseTUIManager
- `apias/tui.py` - Message queue integration in RichTUIManager

---

## Estimated Scope

- New code: ~400 lines (2 new modules)
- Modified code: ~200 lines across existing files
- Tests: ~150 lines
- Total: ~750 lines of changes

This plan provides a comprehensive, phased approach to fixing the TUI communication issues while maintaining backwards compatibility and enabling easy rollback if needed.
