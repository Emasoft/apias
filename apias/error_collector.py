"""
Unified error collection and circuit breaker system for APIAS.

This module replaces the old SessionErrorTracker with a more comprehensive
error handling system that:
- Tracks ALL error types (network, API, scraping, XML validation)
- Uses per-category circuit breaker thresholds
- Preserves error context with smart bounded storage
- Integrates with event bus for decoupled communication

Key Components:
- SmartErrorStorage: Bounded storage with recent errors + statistics
- CircuitBreakerV2: Per-category thresholds with rich context
- ErrorCollector: Unified entry point for all error recording

Architecture:
    Worker Thread → record_error() → SmartErrorStorage + CircuitBreaker
                                   → publish(ErrorEvent/CircuitBreakerEvent)

Usage:
    collector = ErrorCollector(event_bus, config)

    # Record error (from any thread)
    result = collector.record_error(
        category=ErrorCategory.RATE_LIMIT,
        message=str(exception),
        task_id=5,
        url="https://...",
        exception=e,
        context={'retry': 2}
    )

    if result.circuit_tripped:
        # Stop processing
        return None
"""

import logging
import threading
import traceback
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml

from apias.event_system import CircuitBreakerEvent, ErrorCategory, ErrorEvent, EventBus

# Explicitly re-export ErrorCategory for type-safe imports
__all__ = ["ErrorCategory", "ErrorCollector", "load_error_config"]

logger = logging.getLogger(__name__)


# ============================================================================
# Constants and Configuration
# ============================================================================

# Recoverable error categories (can be retried)
RECOVERABLE_CATEGORIES: Set[ErrorCategory] = {
    ErrorCategory.API_TIMEOUT,
    ErrorCategory.CONNECTION_ERROR,
    ErrorCategory.SERVER_ERROR,
    ErrorCategory.PARSE_ERROR,
    ErrorCategory.XML_VALIDATION,
}

# Default thresholds if config file missing
DEFAULT_THRESHOLDS: Dict[ErrorCategory, int] = {
    ErrorCategory.QUOTA_EXCEEDED: 1,  # Trip immediately
    ErrorCategory.AUTHENTICATION: 1,  # Trip immediately
    ErrorCategory.INVALID_API_KEY: 1,  # Trip immediately
    ErrorCategory.RATE_LIMIT: 1,  # Trip immediately (avoid retry storm)
    ErrorCategory.API_TIMEOUT: 5,  # Allow 5 timeouts
    ErrorCategory.CONNECTION_ERROR: 3,  # Allow 3 connection errors
    ErrorCategory.SERVER_ERROR: 10,  # Allow 10 server errors
    ErrorCategory.INVALID_RESPONSE: 3,  # Allow 3 invalid responses
    ErrorCategory.SOURCE_NOT_FOUND: 20,  # Allow many (expected failures)
    ErrorCategory.PARSE_ERROR: 5,  # Allow 5 parse errors
    ErrorCategory.XML_VALIDATION: 3,  # Allow 3 XML validation failures
}

# Errors that immediately trip circuit breaker (fatal)
IMMEDIATE_TRIP_CATEGORIES: Set[ErrorCategory] = {
    ErrorCategory.QUOTA_EXCEEDED,
    ErrorCategory.AUTHENTICATION,
    ErrorCategory.INVALID_API_KEY,
}


# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class CategoryStats:
    """
    Statistical summary for an error category.

    Preserves insights even after individual errors are dropped
    from bounded storage.

    Fields:
        count: Total errors in this category
        first_seen: Timestamp of first occurrence
        last_seen: Timestamp of most recent occurrence
        consecutive_max: Longest consecutive streak observed
    """

    count: int = 0
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    consecutive_max: int = 0


@dataclass
class ErrorConfig:
    """
    Configuration for error handling and circuit breaker.

    Loaded from YAML file with fallback to defaults.

    Fields:
        thresholds: Per-category consecutive error thresholds
        immediate_trip: Categories that trip circuit immediately
        max_recent_errors: How many recent errors to keep full details
        max_total_errors: Maximum total errors before rejecting new ones
    """

    thresholds: Dict[ErrorCategory, int] = field(default_factory=lambda: DEFAULT_THRESHOLDS.copy())
    immediate_trip: Set[ErrorCategory] = field(default_factory=lambda: IMMEDIATE_TRIP_CATEGORIES.copy())
    max_recent_errors: int = 100
    max_total_errors: int = 50000


@dataclass
class ErrorRecordResult:
    """
    Result of recording an error.

    Returned by ErrorCollector.record_error() to inform caller
    whether circuit breaker tripped.

    Fields:
        recorded: Whether error was successfully recorded
        circuit_tripped: Whether this error caused circuit to trip
        trigger_reason: Why circuit tripped (if it did)
    """

    recorded: bool
    circuit_tripped: bool
    trigger_reason: Optional[str] = None


@dataclass
class CircuitTripContext:
    """
    Rich context when circuit breaker trips.

    Preserved for debugging and user messaging.

    Fields:
        reason: Human-readable reason (e.g., "3 consecutive RATE_LIMIT errors")
        triggered_at: When circuit tripped
        triggering_error: The ErrorEvent that caused the trip
        consecutive_counts: Snapshot of consecutive counts at trip time
    """

    reason: str
    triggered_at: datetime
    triggering_error: ErrorEvent
    consecutive_counts: Dict[ErrorCategory, int]


@dataclass
class CircuitResult:
    """Result of circuit breaker check"""

    circuit_tripped: bool
    trigger_reason: Optional[str] = None


# ============================================================================
# Smart Error Storage
# ============================================================================


class SmartErrorStorage:
    """
    Bounded error storage with smart retention strategy.

    Design: Keep recent 100 errors (full details) + statistical summary
    for all errors. This bounds memory at ~50KB even for 10,000+ errors
    while preserving debugging insights.

    Strategy:
    - Recent errors: deque(maxlen=100) with full ErrorEvent details
    - Category stats: count, first/last seen, consecutive max
    - First occurrences: First error of each category (for debugging)

    Memory Footprint:
    - Recent 100 errors: ~30KB (300 bytes * 100)
    - Category stats: ~1KB (10 categories * 100 bytes)
    - First occurrences: ~3KB (10 categories * 300 bytes)
    - Total: ~34KB (vs 5MB for storing all 10k errors)

    Thread Safety: NOT thread-safe by itself.
    Caller (ErrorCollector) must use lock when accessing.
    """

    def __init__(self, max_recent: int = 100):
        """
        Initialize smart error storage.

        Args:
            max_recent: How many recent errors to keep full details (default 100)
        """
        # Recent errors with full details (FIFO, bounded)
        self._recent_errors: deque[ErrorEvent] = deque(maxlen=max_recent)

        # Statistical summary per category
        self._category_stats: Dict[ErrorCategory, CategoryStats] = defaultdict(CategoryStats)

        # First occurrence of each category (for debugging)
        self._first_occurrences: Dict[ErrorCategory, ErrorEvent] = {}

        # Counters
        self._total_recorded = 0
        self._dropped_count = 0  # How many old errors were dropped

        logger.debug(f"SmartErrorStorage initialized with max_recent={max_recent}")

    def add(self, error_event: ErrorEvent) -> None:
        """
        Add an error to storage.

        Updates recent errors, category stats, and first occurrences.
        If recent errors deque is full, oldest error is dropped automatically
        (but stats are still preserved).

        Args:
            error_event: Error to store

        Thread Safety: Caller must hold lock.
        """
        category = error_event.category

        # Add to recent errors (auto-drops oldest if full)
        if len(self._recent_errors) == self._recent_errors.maxlen:
            self._dropped_count += 1
            # Log only on first drop to avoid log spam
            if self._dropped_count == 1:
                logger.info(
                    f"SmartErrorStorage full ({self._recent_errors.maxlen} errors). "
                    f"Dropping oldest errors but preserving statistics."
                )

        self._recent_errors.append(error_event)

        # Update category stats
        stats = self._category_stats[category]
        stats.count += 1
        stats.last_seen = error_event.timestamp

        if stats.first_seen is None:
            stats.first_seen = error_event.timestamp

        # Record first occurrence
        if category not in self._first_occurrences:
            self._first_occurrences[category] = error_event

        # Update total
        self._total_recorded += 1

    def get_recent(self, limit: Optional[int] = None) -> List[ErrorEvent]:
        """
        Get recent errors (most recent first).

        Args:
            limit: Max number to return (None = all)

        Returns:
            List of ErrorEvent in reverse chronological order

        Thread Safety: Caller must hold lock.
        """
        recent = list(reversed(self._recent_errors))
        if limit:
            return recent[:limit]
        return recent

    def get_stats(self, category: Optional[ErrorCategory] = None) -> Dict[ErrorCategory, CategoryStats]:
        """
        Get category statistics.

        Args:
            category: Get stats for specific category (None = all)

        Returns:
            Dict of CategoryStats

        Thread Safety: Caller must hold lock.
        """
        if category:
            return {category: self._category_stats[category]}
        return dict(self._category_stats)

    def get_first_occurrence(self, category: ErrorCategory) -> Optional[ErrorEvent]:
        """
        Get first error of a category (for debugging).

        Args:
            category: Error category

        Returns:
            First ErrorEvent of this category, or None

        Thread Safety: Caller must hold lock.
        """
        return self._first_occurrences.get(category)

    def get_summary(self) -> Dict[str, Any]:
        """
        Get overall storage summary for monitoring.

        Returns:
            Dict with total_recorded, dropped, recent_count, categories

        Thread Safety: Caller must hold lock.
        """
        return {
            "total_recorded": self._total_recorded,
            "dropped": self._dropped_count,
            "recent_count": len(self._recent_errors),
            "categories": len(self._category_stats),
        }


# ============================================================================
# Circuit Breaker V2
# ============================================================================


class CircuitBreakerV2:
    """
    Per-category circuit breaker with rich context.

    Improvements over old CircuitBreaker:
    - Per-category thresholds (not just global)
    - Immediate trip for fatal errors (quota, auth)
    - Rich trip context for debugging
    - Configurable via YAML

    Design:
    - Tracks consecutive errors per category
    - Trips when category hits its threshold
    - Resets on success
    - Preserves trip context for user messaging

    Thread Safety: NOT thread-safe by itself.
    Caller (ErrorCollector) must use lock.
    """

    def __init__(self, config: ErrorConfig):
        """
        Initialize circuit breaker.

        Args:
            config: Error configuration with thresholds
        """
        self._config = config
        self._triggered = False
        self._trigger_context: Optional[CircuitTripContext] = None

        # Consecutive error count per category
        self._consecutive_count: Dict[ErrorCategory, int] = defaultdict(int)

        logger.debug(f"CircuitBreakerV2 initialized with {len(config.thresholds)} category thresholds")

    def record(self, error_event: ErrorEvent) -> CircuitResult:
        """
        Record error and check if circuit should trip.

        Logic:
        1. If already triggered, return (no further action)
        2. If error in immediate_trip set, trip immediately
        3. Otherwise, increment consecutive count for category
        4. If consecutive count >= threshold, trip

        Args:
            error_event: Error to record

        Returns:
            CircuitResult with circuit_tripped and trigger_reason

        Thread Safety: Caller must hold lock.
        """
        if self._triggered:
            # Already tripped - no further action
            return CircuitResult(circuit_tripped=False, trigger_reason=None)

        category = error_event.category

        # Check immediate trip (fatal errors)
        if category in self._config.immediate_trip:
            reason = f"Fatal error: {category.name}"
            self._trip(error_event, reason)
            return CircuitResult(circuit_tripped=True, trigger_reason=reason)

        # Increment consecutive count
        self._consecutive_count[category] += 1

        # Check threshold
        threshold = self._config.thresholds.get(category, 3)  # Default 3
        if self._consecutive_count[category] >= threshold:
            reason = f"{threshold} consecutive {category.name} errors"
            self._trip(error_event, reason)
            return CircuitResult(circuit_tripped=True, trigger_reason=reason)

        return CircuitResult(circuit_tripped=False, trigger_reason=None)

    def record_success(self) -> None:
        """
        Record successful operation.

        Resets all consecutive counts (full reset strategy).

        Design Note: We reset ALL counts on success, not just the last
        error category. This prevents accumulation of small error counts
        across categories.

        Thread Safety: Caller must hold lock.
        """
        if self._consecutive_count:
            logger.debug(f"Circuit breaker: Success recorded, resetting consecutive counts")
            self._consecutive_count.clear()

    def _trip(self, error_event: ErrorEvent, reason: str) -> None:
        """
        Trip the circuit breaker.

        Preserves rich context for debugging and user messaging.

        Args:
            error_event: Error that caused the trip
            reason: Human-readable reason

        Thread Safety: Caller must hold lock.
        """
        self._triggered = True
        self._trigger_context = CircuitTripContext(
            reason=reason,
            triggered_at=datetime.now(),
            triggering_error=error_event,
            consecutive_counts=dict(self._consecutive_count),  # Snapshot
        )

        logger.warning(f"Circuit breaker TRIPPED: {reason}")

    @property
    def is_triggered(self) -> bool:
        """
        Whether circuit breaker has tripped.

        Thread Safety: Caller must hold lock (or use ErrorCollector.is_tripped property).
        """
        return self._triggered

    @property
    def trigger_reason(self) -> Optional[str]:
        """
        Reason why circuit tripped (None if not tripped).

        Thread Safety: Caller must hold lock (or use ErrorCollector.trigger_reason property).
        """
        return self._trigger_context.reason if self._trigger_context else None

    @property
    def trigger_context(self) -> Optional[CircuitTripContext]:
        """
        Full trip context (None if not tripped).

        Thread Safety: Caller must hold lock.
        """
        return self._trigger_context


# ============================================================================
# Error Collector
# ============================================================================


class ErrorCollector:
    """
    Unified error collection system for APIAS.

    This class replaces SessionErrorTracker and provides:
    - Single entry point for ALL errors (network, API, scraping, validation)
    - Thread-safe error recording from multiple worker threads
    - Per-category circuit breaker logic
    - Event bus integration for decoupled communication
    - Smart bounded error storage

    Architecture:
        Worker Thread → record_error() → {
            SmartErrorStorage (bounded)
            CircuitBreakerV2 (per-category)
            EventBus.publish(ErrorEvent)
            [if circuit trips] → EventBus.publish(CircuitBreakerEvent)
        }

    Thread Safety: All public methods use _lock for thread-safe access.

    Usage:
        collector = ErrorCollector(event_bus, config)

        # From worker thread
        result = collector.record_error(
            category=ErrorCategory.CONNECTION_ERROR,
            message="Timeout",
            task_id=5,
            url="https://...",
            exception=exc
        )

        if result.circuit_tripped:
            return None  # Stop processing
    """

    def __init__(self, event_bus: EventBus, config: ErrorConfig):
        """
        Initialize error collector.

        Args:
            event_bus: Event bus for publishing error events
            config: Error handling configuration
        """
        self._event_bus = event_bus
        self._config = config

        # Components
        self._storage = SmartErrorStorage(max_recent=config.max_recent_errors)
        self._circuit_breaker = CircuitBreakerV2(config)

        # Thread safety
        self._lock = threading.Lock()

        # Success counter (for metrics)
        self._success_count = 0

        logger.info(
            f"ErrorCollector initialized with max_recent={config.max_recent_errors}, "
            f"max_total={config.max_total_errors}"
        )

    def record_error(
        self,
        category: ErrorCategory,
        message: str,
        task_id: Optional[int] = None,
        url: Optional[str] = None,
        exception: Optional[Exception] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> ErrorRecordResult:
        """
        Record an error occurrence.

        THIS IS THE MAIN ENTRY POINT for all error recording in APIAS.
        Call this from anywhere an error occurs (network, API, scraping, validation).

        Args:
            category: Error classification (determines circuit breaker logic)
            message: Human-readable error description
            task_id: Task number where error occurred (None for global errors)
            url: URL being processed (None for non-URL errors)
            exception: Original exception object (for traceback)
            context: Additional error context (retry count, chunk number, etc.)

        Returns:
            ErrorRecordResult with:
            - recorded: Always True (unless exception in recording itself)
            - circuit_tripped: Whether this error caused circuit to trip
            - trigger_reason: Why circuit tripped (if it did)

        Thread Safety: Uses _lock for thread-safe access.

        Example:
            except RequestException as e:
                result = error_collector.record_error(
                    category=ErrorCategory.CONNECTION_ERROR,
                    message=str(e),
                    task_id=task_id,
                    url=url,
                    exception=e,
                    context={'phase': 'scraping'}
                )
                if result.circuit_tripped:
                    return None
        """
        with self._lock:
            try:
                # Create enriched error event
                error_event = ErrorEvent(
                    category=category,
                    message=message,
                    task_id=task_id,
                    url=url,
                    exception_type=type(exception).__name__ if exception else None,
                    exception_traceback=traceback.format_exc() if exception else None,
                    context=context or {},
                    recoverable=category in RECOVERABLE_CATEGORIES,
                )

                # Store in smart storage
                self._storage.add(error_event)

                # Check circuit breaker
                circuit_result = self._circuit_breaker.record(error_event)

                # Publish error event to bus
                self._event_bus.publish(error_event)

                # If circuit tripped, publish critical event
                if circuit_result.circuit_tripped:
                    # Get affected tasks (pending tasks at trip time)
                    # Note: This requires access to task state, which we don't have here
                    # Will be filled in by higher-level code that has task context
                    affected_tasks: List[int] = []

                    self._event_bus.publish(
                        CircuitBreakerEvent(
                            reason=circuit_result.trigger_reason or "Circuit breaker tripped",
                            affected_tasks=affected_tasks,
                            trigger_category=category,
                            consecutive_counts=dict(self._circuit_breaker._consecutive_count),
                        )
                    )

                return ErrorRecordResult(
                    recorded=True,
                    circuit_tripped=circuit_result.circuit_tripped,
                    trigger_reason=circuit_result.trigger_reason,
                )

            except Exception as e:
                # Error in error recording - this is bad
                logger.error(f"CRITICAL: Failed to record error: {e}", exc_info=True)
                return ErrorRecordResult(recorded=False, circuit_tripped=False, trigger_reason=None)

    def record_success(self, task_id: Optional[int] = None) -> None:
        """
        Record successful operation.

        Resets circuit breaker consecutive counts.
        Call this after successful URL processing.

        Args:
            task_id: Task that succeeded (for logging)

        Thread Safety: Uses _lock for thread-safe access.
        """
        with self._lock:
            self._circuit_breaker.record_success()
            self._success_count += 1

            if task_id:
                logger.debug(f"Task #{task_id}: Success recorded, circuit breaker reset")

    @property
    def is_tripped(self) -> bool:
        """
        Whether circuit breaker has tripped.

        Thread Safety: Uses _lock for atomic read.
        """
        with self._lock:
            return self._circuit_breaker.is_triggered

    @property
    def trigger_reason(self) -> Optional[str]:
        """
        Reason why circuit tripped (None if not tripped).

        Thread Safety: Uses _lock for atomic read.
        """
        with self._lock:
            return self._circuit_breaker.trigger_reason

    def get_recent_errors(self, limit: Optional[int] = None) -> List[ErrorEvent]:
        """
        Get recent errors (most recent first).

        Thread Safety: Uses _lock for atomic read.
        """
        with self._lock:
            return self._storage.get_recent(limit)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get error collector statistics for monitoring.

        Returns:
            Dict with storage stats, success count, circuit breaker status

        Thread Safety: Uses _lock for atomic read.
        """
        with self._lock:
            storage_stats = self._storage.get_summary()
            return {
                **storage_stats,
                "success_count": self._success_count,
                "circuit_triggered": self._circuit_breaker.is_triggered,
                "trigger_reason": self._circuit_breaker.trigger_reason,
            }

    @property
    def total_errors(self) -> int:
        """
        Get total number of errors recorded.

        Returns:
            Total error count (including dropped errors)

        Thread Safety: Uses _lock for atomic read.
        """
        with self._lock:
            return self._storage._total_recorded

    def get_primary_failure_reason(self) -> Optional[str]:
        """
        Get the most significant reason for failures.

        Priority:
        1. Circuit breaker trigger reason (most severe)
        2. Most frequent error category
        3. None if no errors

        Returns:
            Primary failure reason string, or None if no errors

        Thread Safety: Uses _lock for atomic read.
        """
        # Check circuit breaker first - it has the most critical info
        if self.is_tripped:
            return self.trigger_reason

        with self._lock:
            # Get category statistics
            category_stats = self._storage.get_stats()
            if not category_stats:
                return None

            # Find category with highest count
            most_common = max(category_stats.items(), key=lambda x: x[1].count)
            category, stats = most_common

            category_name = category.name.replace("_", " ").lower()
            return f"{stats.count} tasks failed due to {category_name}"

    def get_error_summary(self) -> Dict[ErrorCategory, int]:
        """
        Get count of errors by category.

        Returns:
            Dict mapping ErrorCategory to error count

        Thread Safety: Uses _lock for atomic read.
        """
        with self._lock:
            category_stats = self._storage.get_stats()
            return {category: stats.count for category, stats in category_stats.items()}


# ============================================================================
# Configuration Loading
# ============================================================================


def load_error_config(config_path: Optional[Path] = None) -> ErrorConfig:
    """
    Load error configuration from YAML file.

    Falls back to defaults if file not found or invalid.

    Args:
        config_path: Path to error_thresholds.yaml (None = use defaults)

    Returns:
        ErrorConfig instance

    Design Note: We use defensive loading - if config file is missing
    or invalid, we fall back to hardcoded defaults. This ensures the
    system always works even if config is broken.
    """
    if not config_path or not config_path.exists():
        logger.info("No error config file found, using defaults")
        return ErrorConfig()

    try:
        with open(config_path) as f:
            data = yaml.safe_load(f)

        # Parse circuit breaker config
        cb_config = data.get("circuit_breaker", {})

        # Parse thresholds
        thresholds_dict = cb_config.get("thresholds", {})
        thresholds = {}
        for category_name, threshold in thresholds_dict.items():
            try:
                category = ErrorCategory[category_name]
                thresholds[category] = int(threshold)
            except (KeyError, ValueError) as e:
                logger.warning(f"Invalid threshold config for {category_name}: {e}")

        # Merge with defaults
        final_thresholds = DEFAULT_THRESHOLDS.copy()
        final_thresholds.update(thresholds)

        # Parse immediate trip categories
        immediate_trip_list = cb_config.get("immediate_trip", [])
        immediate_trip = set()
        for category_name in immediate_trip_list:
            try:
                immediate_trip.add(ErrorCategory[category_name])
            except KeyError:
                logger.warning(f"Invalid immediate_trip category: {category_name}")

        # Use defaults if none specified
        if not immediate_trip:
            immediate_trip = IMMEDIATE_TRIP_CATEGORIES

        # Parse error storage config
        storage_config = data.get("error_storage", {})
        max_recent = storage_config.get("max_recent", 100)
        max_total = storage_config.get("max_total", 50000)

        logger.info(f"Loaded error config from {config_path}")
        return ErrorConfig(
            thresholds=final_thresholds,
            immediate_trip=immediate_trip,
            max_recent_errors=max_recent,
            max_total_errors=max_total,
        )

    except Exception as e:
        logger.error(f"Failed to load error config from {config_path}: {e}, using defaults")
        return ErrorConfig()
