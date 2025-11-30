"""
Centralized error classification and circuit breaker for APIAS.

Provides:
- ErrorCategory enum for classifying errors
- CircuitBreaker for stopping on fatal errors (prevents wasting API credits)
- SessionErrorTracker for aggregating errors across tasks
- classify_openai_error() helper for OpenAI exception classification

DESIGN NOTES:
- All classes are thread-safe for use in concurrent batch processing
- Error tracking has bounded memory usage (max errors configurable)
- Circuit breaker uses immediate stop for quota errors to prevent credit waste
- All state changes are logged for debugging and audit trails

DO NOT:
- Remove thread locks - this code runs in multi-threaded batch processing
- Remove max_errors limit - unbounded lists cause memory exhaustion
- Swallow exceptions silently - always log before suppressing
- Access _triggered without lock - causes race conditions
"""

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Dict, Final, List, Optional

# Module-level logger for tracing error classification and circuit breaker events
# This enables debugging of error handling behavior without code changes
logger = logging.getLogger(__name__)

# =============================================================================
# CENTRALIZED CONSTANTS - Single source of truth for all error-related config
# =============================================================================
# DO NOT scatter these values throughout the code - always reference these constants

DEFAULT_CONSECUTIVE_THRESHOLD: Final[int] = 3  # Errors before circuit trips
DEFAULT_MAX_ERRORS: Final[int] = 1000  # Max errors to track (prevents memory bloat)
DEFAULT_QUOTA_IMMEDIATE_STOP: Final[bool] = True  # Stop immediately on quota exceeded


class ErrorCategory(Enum):
    """
    Classification of error types for summary reporting.

    Each category maps to a specific class of failure that helps users
    understand WHY their job failed and what action to take.

    Categories are ordered by severity - fatal errors that cannot be
    retried should be checked first in classification logic.
    """

    NONE = auto()  # No error - successful operation
    RATE_LIMIT = auto()  # 429 Too Many Requests - can retry after delay
    QUOTA_EXCEEDED = auto()  # Insufficient quota - FATAL, must add credits
    API_TIMEOUT = auto()  # Request timeout - can retry
    CONNECTION_ERROR = auto()  # Network/connection failure - can retry
    INVALID_RESPONSE = auto()  # XML validation failed - may need different prompt
    SOURCE_NOT_FOUND = auto()  # Page not found (404) - skip this URL
    AUTHENTICATION = auto()  # API key invalid - FATAL, fix credentials
    SERVER_ERROR = auto()  # 5xx errors - can retry later
    UNKNOWN = auto()  # Unclassified errors - investigate logs


# =============================================================================
# ERROR METADATA TABLES - Centralized display info for each category
# =============================================================================
# DO NOT duplicate these strings elsewhere - always use get_error_icon/description

ERROR_ICONS: Final[Dict[ErrorCategory, str]] = {
    ErrorCategory.NONE: "OK",
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

ERROR_DESCRIPTIONS: Final[Dict[ErrorCategory, str]] = {
    ErrorCategory.NONE: "No error",
    ErrorCategory.RATE_LIMIT: "API rate limit exceeded (429)",
    ErrorCategory.QUOTA_EXCEEDED: "API quota exhausted - add credits",
    # WHY explicit "AI service": Distinguish from website/scraping timeouts
    # Users reported confusion about which service was slow
    ErrorCategory.API_TIMEOUT: "AI service (OpenAI) request timed out",
    ErrorCategory.CONNECTION_ERROR: "Network connection failed",
    ErrorCategory.INVALID_RESPONSE: "Invalid response from API",
    ErrorCategory.SOURCE_NOT_FOUND: "Source page not found (404)",
    ErrorCategory.AUTHENTICATION: "API key invalid or expired",
    ErrorCategory.SERVER_ERROR: "API server error (5xx)",
    ErrorCategory.UNKNOWN: "Unknown error occurred",
}

# Categories that are recoverable (can be retried)
# IMPORTANT: RATE_LIMIT is NOT recoverable - it means we're hitting API limits
# and should stop immediately to avoid wasting time and potentially being banned.
# Only transient errors (timeout, connection, server error) are recoverable.
RECOVERABLE_CATEGORIES: Final[frozenset[ErrorCategory]] = frozenset(
    {
        ErrorCategory.API_TIMEOUT,
        ErrorCategory.CONNECTION_ERROR,
        ErrorCategory.SERVER_ERROR,
    }
)


@dataclass
class ErrorEvent:
    """
    Single error occurrence with full context for debugging and reporting.

    Captures all relevant information at the moment of failure so that
    root cause analysis can be performed without access to the original logs.
    """

    category: ErrorCategory  # Classified error type
    message: str  # Human-readable error message
    task_id: Optional[int]  # Task number in batch (None for single-page mode)
    url: Optional[str]  # URL being processed when error occurred
    timestamp: datetime = field(default_factory=datetime.now)  # When error happened
    recoverable: bool = True  # True if operation can be retried
    raw_exception: Optional[str] = None  # Exception class name for debugging

    def is_recoverable(self) -> bool:
        """
        Check if this error type can be retried.

        Uses centralized RECOVERABLE_CATEGORIES set to ensure consistency.
        DO NOT hardcode category checks elsewhere - use this method.
        """
        return self.category in RECOVERABLE_CATEGORIES


class CircuitBreaker:
    """
    Stops processing when fatal errors occur to prevent wasting API credits.

    The circuit breaker pattern prevents continued processing when it's clear
    that requests will keep failing. This saves user money and time.

    Triggers on:
    - QUOTA_EXCEEDED: Immediate stop - no point continuing without credits
    - AUTHENTICATION: Immediate stop - all requests will fail with bad key
    - Any error type with >= consecutive_threshold consecutive failures

    THREAD SAFETY:
    - All state access is protected by _lock
    - Properties that read state also use the lock (race condition fix)

    DO NOT:
    - Read _triggered directly - always use is_triggered property with lock
    - Remove logging - circuit breaker events must be traceable
    """

    def __init__(
        self,
        consecutive_threshold: int = DEFAULT_CONSECUTIVE_THRESHOLD,
        quota_immediate_stop: bool = DEFAULT_QUOTA_IMMEDIATE_STOP,
        rate_limit_immediate_stop: bool = True,
    ) -> None:
        """
        Initialize circuit breaker.

        Args:
            consecutive_threshold: Number of consecutive errors before tripping.
                                   Uses DEFAULT_CONSECUTIVE_THRESHOLD if not specified.
            quota_immediate_stop: If True, trip immediately on quota exceeded.
                                  This prevents wasting API calls when out of credits.
            rate_limit_immediate_stop: If True, trip immediately on rate limit (429).
                                       Rate limits mean we should back off, not retry.
        """
        # Store configuration - these never change after init
        self.consecutive_threshold = consecutive_threshold
        self.quota_immediate_stop = quota_immediate_stop
        self.rate_limit_immediate_stop = rate_limit_immediate_stop

        # Internal state - always access with lock held
        self._consecutive_errors = 0
        self._last_error_category: Optional[ErrorCategory] = None
        self._triggered = False
        self._trigger_reason: Optional[str] = None

        # Thread safety lock - protects all mutable state
        # DO NOT remove - this code runs in concurrent batch processing
        self._lock = threading.Lock()

        logger.debug(
            f"CircuitBreaker initialized: threshold={consecutive_threshold}, "
            f"quota_immediate_stop={quota_immediate_stop}, "
            f"rate_limit_immediate_stop={rate_limit_immediate_stop}"
        )

    def record_success(self) -> None:
        """
        Reset consecutive error count on success.

        Called after each successful API call to reset the error streak.
        This prevents false positives from intermittent errors.
        """
        with self._lock:
            if self._consecutive_errors > 0:
                logger.debug(
                    f"CircuitBreaker: success resets error count from {self._consecutive_errors}"
                )
            self._consecutive_errors = 0
            self._last_error_category = None

    def record_error(self, category: ErrorCategory) -> bool:
        """
        Record an error and check if circuit should trip.

        Args:
            category: The error category to record

        Returns:
            True if circuit has tripped (should stop processing)
        """
        with self._lock:
            # Log every error for debugging - helps trace failure patterns
            logger.debug(f"CircuitBreaker: recording {category.name} error")

            # Immediate stop for quota exceeded - no point wasting more calls
            if category == ErrorCategory.QUOTA_EXCEEDED and self.quota_immediate_stop:
                self._triggered = True
                self._trigger_reason = "API quota exceeded - add credits to continue"
                logger.warning(f"CircuitBreaker TRIPPED: {self._trigger_reason}")
                return True

            # Immediate stop for rate limit (429) - we must back off, not retry
            if category == ErrorCategory.RATE_LIMIT and self.rate_limit_immediate_stop:
                self._triggered = True
                self._trigger_reason = (
                    "Rate limit exceeded (429) - please wait before retrying"
                )
                logger.warning(f"CircuitBreaker TRIPPED: {self._trigger_reason}")
                return True

            # Immediate stop for authentication - all calls will fail
            if category == ErrorCategory.AUTHENTICATION:
                self._triggered = True
                self._trigger_reason = "API authentication failed - check your API key"
                logger.warning(f"CircuitBreaker TRIPPED: {self._trigger_reason}")
                return True

            # Track consecutive errors of the same type
            if category == self._last_error_category:
                self._consecutive_errors += 1
            else:
                # Different error type - reset the counter but start at 1
                self._consecutive_errors = 1
                self._last_error_category = category

            # Check if we've hit the threshold
            if self._consecutive_errors >= self.consecutive_threshold:
                self._triggered = True
                category_name = category.name.replace("_", " ").lower()
                self._trigger_reason = (
                    f"{self._consecutive_errors} consecutive {category_name} errors"
                )
                logger.warning(f"CircuitBreaker TRIPPED: {self._trigger_reason}")
                return True

            # Still counting - return current triggered state (may have tripped earlier)
            return self._triggered

    def reset(self) -> None:
        """
        Reset the circuit breaker to initial state.

        Called when starting a new processing session or after user intervention.
        """
        with self._lock:
            if self._triggered:
                logger.info("CircuitBreaker: reset from triggered state")
            self._consecutive_errors = 0
            self._last_error_category = None
            self._triggered = False
            self._trigger_reason = None

    @property
    def is_triggered(self) -> bool:
        """
        Check if circuit breaker has been triggered.

        THREAD SAFETY: Uses lock to prevent race condition where another
        thread could modify _triggered between check and use.
        """
        with self._lock:
            return self._triggered

    @property
    def trigger_reason(self) -> Optional[str]:
        """
        Get the reason why the circuit breaker was triggered.

        THREAD SAFETY: Uses lock for consistent read.
        """
        with self._lock:
            return self._trigger_reason


class SessionErrorTracker:
    """
    Tracks all errors across a processing session for summary reporting.

    Thread-safe for use in multi-threaded batch processing.

    MEMORY SAFETY:
    - Errors list is bounded by max_errors to prevent memory exhaustion
    - When limit is reached, oldest errors are discarded (FIFO)

    DO NOT:
    - Remove max_errors limit - causes OOM on large batch jobs
    - Remove thread locks - causes race conditions in batch processing
    - Iterate self.errors without holding lock - use get_* methods instead
    """

    def __init__(
        self,
        consecutive_threshold: int = DEFAULT_CONSECUTIVE_THRESHOLD,
        quota_immediate_stop: bool = DEFAULT_QUOTA_IMMEDIATE_STOP,
        rate_limit_immediate_stop: bool = True,
        max_errors: int = DEFAULT_MAX_ERRORS,
    ) -> None:
        """
        Initialize session error tracker.

        Args:
            consecutive_threshold: Circuit breaker threshold
            quota_immediate_stop: Stop immediately on quota exceeded
            rate_limit_immediate_stop: Stop immediately on rate limit (429)
            max_errors: Maximum number of errors to store (prevents memory bloat)
        """
        self._max_errors = max_errors
        self._errors: List[ErrorEvent] = []  # Bounded list
        self._dropped_count = 0  # Track how many errors we had to drop

        # Separate counters for accurate reporting even when errors are dropped
        self._total_recorded = 0
        self._success_count = 0

        self.circuit_breaker = CircuitBreaker(
            consecutive_threshold=consecutive_threshold,
            quota_immediate_stop=quota_immediate_stop,
            rate_limit_immediate_stop=rate_limit_immediate_stop,
        )
        self._lock = threading.Lock()

        logger.debug(
            f"SessionErrorTracker initialized: max_errors={max_errors}, "
            f"threshold={consecutive_threshold}, rate_limit_stop={rate_limit_immediate_stop}"
        )

    def record(self, event: ErrorEvent) -> bool:
        """
        Record an error event.

        Args:
            event: The error event to record

        Returns:
            True if processing should stop (circuit breaker triggered)
        """
        with self._lock:
            self._total_recorded += 1

            # Enforce bounded error list to prevent memory exhaustion
            if len(self._errors) >= self._max_errors:
                # Drop oldest error (FIFO) to make room
                self._errors.pop(0)
                self._dropped_count += 1
                if self._dropped_count == 1:
                    # Only log once when we start dropping
                    logger.warning(
                        f"Error list full (max={self._max_errors}), "
                        "dropping oldest errors"
                    )

            self._errors.append(event)

        # Log error recording for debugging
        logger.debug(
            f"Error recorded: {event.category.name} for task {event.task_id} "
            f"(total: {self._total_recorded})"
        )

        return self.circuit_breaker.record_error(event.category)

    def record_success(self) -> None:
        """
        Record a successful operation (resets consecutive error count).

        IMPORTANT: This must be called after each successful API call to
        prevent the circuit breaker from tripping on intermittent errors.
        """
        with self._lock:
            self._success_count += 1
        self.circuit_breaker.record_success()

    def get_primary_failure_reason(self) -> Optional[str]:
        """
        Get the most significant reason for failures.

        Priority:
        1. Circuit breaker trigger reason (most severe)
        2. Most frequent error category
        3. None if no errors
        """
        # Check circuit breaker first - it has the most critical info
        if self.circuit_breaker.is_triggered:
            return self.circuit_breaker.trigger_reason

        with self._lock:
            if not self._errors:
                return None

            # Count by category using stored errors
            category_counts: Dict[ErrorCategory, int] = {}
            for event in self._errors:
                category_counts[event.category] = (
                    category_counts.get(event.category, 0) + 1
                )

        if not category_counts:
            return None

        # Find most common error category
        most_common = max(category_counts.items(), key=lambda x: x[1])
        category, count = most_common

        category_name = category.name.replace("_", " ").lower()
        return f"{count} tasks failed due to {category_name}"

    def get_error_summary(self) -> Dict[ErrorCategory, int]:
        """
        Get count of errors by category.

        Thread-safe - makes a copy while holding lock.
        """
        with self._lock:
            summary: Dict[ErrorCategory, int] = {}
            for event in self._errors:
                summary[event.category] = summary.get(event.category, 0) + 1
            return summary

    def get_errors_by_category(self, category: ErrorCategory) -> List[ErrorEvent]:
        """
        Get all errors of a specific category.

        Thread-safe - returns a copy of matching events.
        """
        with self._lock:
            return [e for e in self._errors if e.category == category]

    def clear(self) -> None:
        """Clear all recorded errors and reset circuit breaker."""
        with self._lock:
            old_count = len(self._errors)
            self._errors.clear()
            self._dropped_count = 0
            self._total_recorded = 0
            self._success_count = 0
        self.circuit_breaker.reset()
        if old_count > 0:
            logger.debug(f"SessionErrorTracker cleared {old_count} errors")

    @property
    def total_errors(self) -> int:
        """
        Get total number of errors recorded in this session.

        Returns the actual count of errors recorded, not the list length
        (which may be smaller due to max_errors limit).
        """
        with self._lock:
            return self._total_recorded

    @property
    def stored_errors(self) -> int:
        """Get number of errors currently stored (may be less than total due to limit)."""
        with self._lock:
            return len(self._errors)

    @property
    def dropped_errors(self) -> int:
        """Get number of errors that were dropped due to max_errors limit."""
        with self._lock:
            return self._dropped_count

    @property
    def success_count(self) -> int:
        """Get number of successful operations recorded."""
        with self._lock:
            return self._success_count

    @property
    def has_fatal_errors(self) -> bool:
        """Check if any fatal (non-recoverable) errors occurred."""
        with self._lock:
            return any(not e.recoverable for e in self._errors)

    # Legacy property for backward compatibility
    @property
    def errors(self) -> List[ErrorEvent]:
        """
        Get a copy of the errors list.

        DEPRECATED: Use get_error_summary() or get_errors_by_category() instead.
        This property returns a copy to prevent external mutation.
        """
        with self._lock:
            return list(self._errors)


def classify_openai_error(exception: Exception) -> ErrorCategory:
    """
    Classify an OpenAI exception into an error category.

    Args:
        exception: The exception to classify

    Returns:
        The appropriate ErrorCategory for the exception
    """
    # Import here to avoid circular imports and optional dependency
    try:
        from openai import (
            APIConnectionError,
            APIStatusError,
            APITimeoutError,
            AuthenticationError,
            RateLimitError,
        )
    except ImportError:
        # If openai is not installed, return UNKNOWN
        return ErrorCategory.UNKNOWN

    error_str = str(exception).lower()

    # Check for quota exceeded first (it's a subtype of RateLimitError)
    if isinstance(exception, RateLimitError):
        if "quota" in error_str or "insufficient_quota" in error_str:
            return ErrorCategory.QUOTA_EXCEEDED
        return ErrorCategory.RATE_LIMIT

    if isinstance(exception, APITimeoutError):
        return ErrorCategory.API_TIMEOUT

    if isinstance(exception, APIConnectionError):
        return ErrorCategory.CONNECTION_ERROR

    if isinstance(exception, AuthenticationError):
        return ErrorCategory.AUTHENTICATION

    if isinstance(exception, APIStatusError):
        if hasattr(exception, "status_code"):
            if exception.status_code >= 500:
                return ErrorCategory.SERVER_ERROR
            elif exception.status_code == 401:
                return ErrorCategory.AUTHENTICATION
            elif exception.status_code == 404:
                return ErrorCategory.SOURCE_NOT_FOUND
        return ErrorCategory.UNKNOWN

    # Check string content for common patterns
    if "timeout" in error_str:
        return ErrorCategory.API_TIMEOUT
    if "connection" in error_str or "network" in error_str:
        return ErrorCategory.CONNECTION_ERROR
    if "rate limit" in error_str or "429" in error_str:
        return ErrorCategory.RATE_LIMIT
    if "authentication" in error_str or "api key" in error_str:
        return ErrorCategory.AUTHENTICATION
    if "xml" in error_str or "validation" in error_str:
        return ErrorCategory.INVALID_RESPONSE
    if "not found" in error_str or "404" in error_str:
        return ErrorCategory.SOURCE_NOT_FOUND

    return ErrorCategory.UNKNOWN


def get_error_icon(category: ErrorCategory) -> str:
    """
    Get a display icon/label for an error category.

    Uses centralized ERROR_ICONS dict - DO NOT duplicate icon definitions.

    Args:
        category: The error category

    Returns:
        A short string label for the category
    """
    # Use centralized constant - single source of truth
    return ERROR_ICONS.get(category, "???")


def get_error_description(category: ErrorCategory) -> str:
    """
    Get a human-readable description for an error category.

    Uses centralized ERROR_DESCRIPTIONS dict - DO NOT duplicate descriptions.

    Args:
        category: The error category

    Returns:
        A descriptive string for the category
    """
    # Use centralized constant - single source of truth
    return ERROR_DESCRIPTIONS.get(category, "Unknown error")


def is_recoverable_error(category: ErrorCategory) -> bool:
    """
    Check if an error category represents a recoverable error.

    Uses centralized RECOVERABLE_CATEGORIES - DO NOT hardcode checks.

    Args:
        category: The error category to check

    Returns:
        True if the error can be retried, False if fatal
    """
    return category in RECOVERABLE_CATEGORIES
