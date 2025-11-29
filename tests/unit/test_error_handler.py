"""
Tests for APIAS error_handler module.

Tests error classification, circuit breaker behavior, and session tracking.
"""

import pytest

from apias.error_handler import (
    CircuitBreaker,
    ErrorCategory,
    ErrorEvent,
    SessionErrorTracker,
    classify_openai_error,
    get_error_description,
    get_error_icon,
)


class TestErrorCategory:
    """Tests for ErrorCategory enum."""

    def test_all_categories_defined(self) -> None:
        """All expected error categories are defined."""
        expected = [
            "NONE",
            "RATE_LIMIT",
            "QUOTA_EXCEEDED",
            "API_TIMEOUT",
            "CONNECTION_ERROR",
            "INVALID_RESPONSE",
            "SOURCE_NOT_FOUND",
            "AUTHENTICATION",
            "SERVER_ERROR",
            "UNKNOWN",
        ]
        actual = [c.name for c in ErrorCategory]
        for name in expected:
            assert name in actual, f"Missing category: {name}"

    def test_categories_are_unique(self) -> None:
        """Each category has a unique value."""
        values = [c.value for c in ErrorCategory]
        assert len(values) == len(set(values))


class TestErrorEvent:
    """Tests for ErrorEvent dataclass."""

    def test_create_basic_event(self) -> None:
        """Can create an error event with required fields."""
        event = ErrorEvent(
            category=ErrorCategory.RATE_LIMIT,
            message="Rate limit exceeded",
            task_id=1,
            url="https://example.com",
        )
        assert event.category == ErrorCategory.RATE_LIMIT
        assert event.message == "Rate limit exceeded"
        assert event.task_id == 1
        assert event.url == "https://example.com"
        assert event.recoverable is True  # Default
        assert event.timestamp is not None

    def test_create_non_recoverable_event(self) -> None:
        """Can create a non-recoverable error event."""
        event = ErrorEvent(
            category=ErrorCategory.QUOTA_EXCEEDED,
            message="Quota exhausted",
            task_id=None,
            url=None,
            recoverable=False,
        )
        assert event.recoverable is False


class TestCircuitBreaker:
    """Tests for CircuitBreaker class."""

    def test_initial_state_not_triggered(self) -> None:
        """Circuit breaker starts in non-triggered state."""
        breaker = CircuitBreaker()
        assert breaker.is_triggered is False
        assert breaker.trigger_reason is None

    def test_immediate_stop_on_quota_exceeded(self) -> None:
        """Circuit breaker triggers immediately on quota exceeded."""
        breaker = CircuitBreaker(quota_immediate_stop=True)
        result = breaker.record_error(ErrorCategory.QUOTA_EXCEEDED)
        assert result is True
        assert breaker.is_triggered is True
        assert "quota exceeded" in breaker.trigger_reason.lower()

    def test_no_immediate_stop_when_disabled(self) -> None:
        """Circuit breaker doesn't trigger immediately when disabled."""
        breaker = CircuitBreaker(quota_immediate_stop=False, consecutive_threshold=3)
        result = breaker.record_error(ErrorCategory.QUOTA_EXCEEDED)
        assert result is False
        assert breaker.is_triggered is False

    def test_immediate_stop_on_rate_limit(self) -> None:
        """Circuit breaker triggers immediately on rate limit (429)."""
        breaker = CircuitBreaker(rate_limit_immediate_stop=True)
        result = breaker.record_error(ErrorCategory.RATE_LIMIT)
        assert result is True
        assert breaker.is_triggered is True
        assert "rate limit" in breaker.trigger_reason.lower()

    def test_no_immediate_rate_limit_stop_when_disabled(self) -> None:
        """Circuit breaker doesn't trigger immediately on rate limit when disabled."""
        breaker = CircuitBreaker(
            rate_limit_immediate_stop=False, consecutive_threshold=3
        )
        result = breaker.record_error(ErrorCategory.RATE_LIMIT)
        assert result is False
        assert breaker.is_triggered is False

    def test_consecutive_threshold_triggers(self) -> None:
        """Circuit breaker triggers after consecutive threshold reached."""
        # Use API_TIMEOUT which is not an immediate-stop error
        breaker = CircuitBreaker(consecutive_threshold=3)

        assert breaker.record_error(ErrorCategory.API_TIMEOUT) is False
        assert breaker.record_error(ErrorCategory.API_TIMEOUT) is False
        assert breaker.record_error(ErrorCategory.API_TIMEOUT) is True
        assert breaker.is_triggered is True
        assert "3 consecutive" in breaker.trigger_reason.lower()

    def test_different_errors_reset_count(self) -> None:
        """Different error types reset the consecutive count."""
        # Use API_TIMEOUT and CONNECTION_ERROR which are not immediate-stop errors
        breaker = CircuitBreaker(consecutive_threshold=3)

        breaker.record_error(ErrorCategory.API_TIMEOUT)
        breaker.record_error(ErrorCategory.API_TIMEOUT)
        # Different error type resets count
        breaker.record_error(ErrorCategory.CONNECTION_ERROR)
        breaker.record_error(ErrorCategory.CONNECTION_ERROR)

        assert breaker.is_triggered is False

    def test_success_resets_count(self) -> None:
        """Successful operation resets consecutive error count."""
        # Use API_TIMEOUT which is not an immediate-stop error
        breaker = CircuitBreaker(consecutive_threshold=3)

        breaker.record_error(ErrorCategory.API_TIMEOUT)
        breaker.record_error(ErrorCategory.API_TIMEOUT)
        breaker.record_success()
        breaker.record_error(ErrorCategory.API_TIMEOUT)
        breaker.record_error(ErrorCategory.API_TIMEOUT)

        assert breaker.is_triggered is False

    def test_reset_clears_state(self) -> None:
        """Reset method clears all state."""
        # Use API_TIMEOUT which is not an immediate-stop error
        breaker = CircuitBreaker(consecutive_threshold=2)
        breaker.record_error(ErrorCategory.API_TIMEOUT)
        breaker.record_error(ErrorCategory.API_TIMEOUT)
        assert breaker.is_triggered is True

        breaker.reset()

        assert breaker.is_triggered is False
        assert breaker.trigger_reason is None


class TestSessionErrorTracker:
    """Tests for SessionErrorTracker class."""

    def test_initial_state_empty(self) -> None:
        """Session tracker starts with no errors."""
        tracker = SessionErrorTracker()
        assert tracker.total_errors == 0
        assert tracker.has_fatal_errors is False
        assert tracker.get_error_summary() == {}

    def test_record_single_error(self) -> None:
        """Can record a single error event."""
        tracker = SessionErrorTracker()
        event = ErrorEvent(
            category=ErrorCategory.RATE_LIMIT,
            message="Rate limit",
            task_id=1,
            url="http://example.com",
        )
        tracker.record(event)

        assert tracker.total_errors == 1
        assert ErrorCategory.RATE_LIMIT in tracker.get_error_summary()

    def test_record_multiple_errors(self) -> None:
        """Can record multiple error events."""
        tracker = SessionErrorTracker()

        for i in range(3):
            tracker.record(
                ErrorEvent(
                    category=ErrorCategory.RATE_LIMIT,
                    message=f"Error {i}",
                    task_id=i,
                    url=None,
                )
            )

        tracker.record(
            ErrorEvent(
                category=ErrorCategory.API_TIMEOUT,
                message="Timeout",
                task_id=3,
                url=None,
            )
        )

        assert tracker.total_errors == 4
        summary = tracker.get_error_summary()
        assert summary[ErrorCategory.RATE_LIMIT] == 3
        assert summary[ErrorCategory.API_TIMEOUT] == 1

    def test_primary_failure_reason_most_common(self) -> None:
        """Primary failure reason is the most common category."""
        # Disable rate_limit_immediate_stop so we can test error counting
        tracker = SessionErrorTracker(
            consecutive_threshold=10, rate_limit_immediate_stop=False
        )

        # Add 3 timeout errors (use timeout instead of rate_limit since rate_limit now stops immediately)
        for i in range(3):
            tracker.record(
                ErrorEvent(
                    category=ErrorCategory.API_TIMEOUT,
                    message="",
                    task_id=i,
                    url=None,
                )
            )

        # Add 1 connection error
        tracker.record(
            ErrorEvent(
                category=ErrorCategory.CONNECTION_ERROR,
                message="",
                task_id=3,
                url=None,
            )
        )

        reason = tracker.get_primary_failure_reason()
        assert "3 tasks" in reason
        assert "timeout" in reason.lower()

    def test_primary_failure_reason_circuit_breaker(self) -> None:
        """Primary failure reason uses circuit breaker reason when triggered."""
        tracker = SessionErrorTracker(consecutive_threshold=2)

        tracker.record(
            ErrorEvent(
                category=ErrorCategory.QUOTA_EXCEEDED,
                message="Quota",
                task_id=1,
                url=None,
            )
        )

        reason = tracker.get_primary_failure_reason()
        assert "quota exceeded" in reason.lower()

    def test_has_fatal_errors(self) -> None:
        """Correctly identifies fatal (non-recoverable) errors."""
        tracker = SessionErrorTracker()

        # Add recoverable error
        tracker.record(
            ErrorEvent(
                category=ErrorCategory.RATE_LIMIT,
                message="",
                task_id=1,
                url=None,
                recoverable=True,
            )
        )
        assert tracker.has_fatal_errors is False

        # Add non-recoverable error
        tracker.record(
            ErrorEvent(
                category=ErrorCategory.QUOTA_EXCEEDED,
                message="",
                task_id=2,
                url=None,
                recoverable=False,
            )
        )
        assert tracker.has_fatal_errors is True

    def test_get_errors_by_category(self) -> None:
        """Can filter errors by category."""
        tracker = SessionErrorTracker()

        tracker.record(
            ErrorEvent(
                category=ErrorCategory.RATE_LIMIT,
                message="RL1",
                task_id=1,
                url=None,
            )
        )
        tracker.record(
            ErrorEvent(
                category=ErrorCategory.API_TIMEOUT,
                message="TO1",
                task_id=2,
                url=None,
            )
        )
        tracker.record(
            ErrorEvent(
                category=ErrorCategory.RATE_LIMIT,
                message="RL2",
                task_id=3,
                url=None,
            )
        )

        rate_limit_errors = tracker.get_errors_by_category(ErrorCategory.RATE_LIMIT)
        assert len(rate_limit_errors) == 2
        assert all(e.category == ErrorCategory.RATE_LIMIT for e in rate_limit_errors)

    def test_clear_resets_tracker(self) -> None:
        """Clear method resets all state."""
        tracker = SessionErrorTracker()

        tracker.record(
            ErrorEvent(
                category=ErrorCategory.QUOTA_EXCEEDED,
                message="",
                task_id=1,
                url=None,
            )
        )
        assert tracker.total_errors == 1
        assert tracker.circuit_breaker.is_triggered is True

        tracker.clear()

        assert tracker.total_errors == 0
        assert tracker.circuit_breaker.is_triggered is False


class TestClassifyOpenAIError:
    """Tests for classify_openai_error function."""

    def test_unknown_for_generic_exception(self) -> None:
        """Generic exceptions are classified as UNKNOWN."""
        result = classify_openai_error(Exception("Some error"))
        assert result == ErrorCategory.UNKNOWN

    def test_timeout_from_string_content(self) -> None:
        """Timeout errors detected from string content."""
        result = classify_openai_error(Exception("Connection timeout occurred"))
        assert result == ErrorCategory.API_TIMEOUT

    def test_connection_from_string_content(self) -> None:
        """Connection errors detected from string content."""
        result = classify_openai_error(Exception("Network connection failed"))
        assert result == ErrorCategory.CONNECTION_ERROR

    def test_rate_limit_from_string_content(self) -> None:
        """Rate limit errors detected from string content."""
        result = classify_openai_error(Exception("Rate limit exceeded, 429"))
        assert result == ErrorCategory.RATE_LIMIT

    def test_authentication_from_string_content(self) -> None:
        """Authentication errors detected from string content."""
        result = classify_openai_error(Exception("Invalid API key"))
        assert result == ErrorCategory.AUTHENTICATION

    def test_not_found_from_string_content(self) -> None:
        """Not found errors detected from string content."""
        result = classify_openai_error(Exception("Resource not found, 404"))
        assert result == ErrorCategory.SOURCE_NOT_FOUND


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_error_icon_all_categories(self) -> None:
        """All categories have icons defined."""
        for category in ErrorCategory:
            icon = get_error_icon(category)
            assert isinstance(icon, str)
            assert len(icon) > 0

    def test_get_error_description_all_categories(self) -> None:
        """All categories have descriptions defined."""
        for category in ErrorCategory:
            desc = get_error_description(category)
            assert isinstance(desc, str)
            assert len(desc) > 0

    def test_specific_icons(self) -> None:
        """Specific categories have expected icons."""
        assert get_error_icon(ErrorCategory.RATE_LIMIT) == "429"
        assert get_error_icon(ErrorCategory.QUOTA_EXCEEDED) == "$$$"
        assert get_error_icon(ErrorCategory.SERVER_ERROR) == "5xx"
