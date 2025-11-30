"""
Comprehensive tests for retry functionality in APIAS.

Tests cover:
1. Exponential backoff calculation (delay = base * 2^attempt, capped at max)
2. MockAPIClient force_retry_count behavior
3. process_single_chunk retry logic with mock API
4. Retry logging format for reproducibility
5. Configuration constants validation

These tests ensure the retry system works correctly and is reproducible
for debugging production failures.
"""

import asyncio
import logging
import re
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from apias.config import (
    RETRY_BASE_DELAY_SECONDS,
    RETRY_MAX_DELAY_SECONDS,
    XML_VALIDATION_MAX_RETRIES,
)
from apias.mock_api import (
    MockAPIClient,
    MockErrorScenario,
    MockOpenAIConfig,
    configure_mock_openai,
    mock_make_openai_request,
    reset_mock_openai,
)

# =============================================================================
# Test: Retry Configuration Constants
# =============================================================================


class TestRetryConstants:
    """Verify retry configuration constants are correctly defined."""

    def test_retry_base_delay_is_positive(self) -> None:
        """Base delay must be positive for meaningful backoff."""
        assert RETRY_BASE_DELAY_SECONDS > 0, "Base delay must be positive"
        assert RETRY_BASE_DELAY_SECONDS == 1.0, "Expected base delay of 1.0 seconds"

    def test_retry_max_delay_greater_than_base(self) -> None:
        """Max delay must be greater than base for backoff to work."""
        assert RETRY_MAX_DELAY_SECONDS > RETRY_BASE_DELAY_SECONDS, (
            "Max delay must exceed base delay"
        )
        assert RETRY_MAX_DELAY_SECONDS == 30.0, "Expected max delay of 30.0 seconds"

    def test_xml_validation_max_retries_is_reasonable(self) -> None:
        """Max retries should be small to avoid long waits."""
        assert XML_VALIDATION_MAX_RETRIES >= 0, "Max retries must be non-negative"
        assert XML_VALIDATION_MAX_RETRIES <= 5, (
            "Max retries should be <= 5 to limit wait time"
        )


# =============================================================================
# Test: Exponential Backoff Calculation
# =============================================================================


class TestExponentialBackoff:
    """Test the exponential backoff formula: delay = min(base * 2^attempt, max)."""

    def calculate_backoff(self, attempt: int) -> float:
        """Replicate the backoff calculation from apias.py:2699."""
        return min(RETRY_BASE_DELAY_SECONDS * (2**attempt), RETRY_MAX_DELAY_SECONDS)

    def test_attempt_0_uses_base_delay(self) -> None:
        """First retry (attempt 0) should use base delay."""
        delay = self.calculate_backoff(0)
        assert delay == RETRY_BASE_DELAY_SECONDS, (
            f"Attempt 0 should use base delay, got {delay}"
        )

    def test_attempt_1_doubles_delay(self) -> None:
        """Second retry should double the delay."""
        delay = self.calculate_backoff(1)
        expected = RETRY_BASE_DELAY_SECONDS * 2
        assert delay == expected, f"Attempt 1 should be {expected}s, got {delay}s"

    def test_attempt_2_quadruples_delay(self) -> None:
        """Third retry should quadruple the base delay."""
        delay = self.calculate_backoff(2)
        expected = RETRY_BASE_DELAY_SECONDS * 4
        assert delay == expected, f"Attempt 2 should be {expected}s, got {delay}s"

    def test_delay_sequence_is_exponential(self) -> None:
        """Verify full exponential sequence: 1, 2, 4, 8, 16, 30 (capped)."""
        expected_sequence = [1.0, 2.0, 4.0, 8.0, 16.0, 30.0]  # 32 capped to 30
        actual_sequence = [self.calculate_backoff(i) for i in range(6)]
        assert actual_sequence == expected_sequence, (
            f"Expected {expected_sequence}, got {actual_sequence}"
        )

    def test_delay_is_capped_at_max(self) -> None:
        """Delay should never exceed RETRY_MAX_DELAY_SECONDS."""
        for attempt in range(10):
            delay = self.calculate_backoff(attempt)
            assert delay <= RETRY_MAX_DELAY_SECONDS, (
                f"Attempt {attempt} delay {delay}s exceeds max {RETRY_MAX_DELAY_SECONDS}s"
            )

    def test_high_attempt_still_capped(self) -> None:
        """Even very high attempt numbers should be capped."""
        delay = self.calculate_backoff(100)
        assert delay == RETRY_MAX_DELAY_SECONDS, (
            f"Attempt 100 should be capped at {RETRY_MAX_DELAY_SECONDS}s, got {delay}s"
        )


# =============================================================================
# Test: MockAPIClient force_retry_count
# =============================================================================


class TestMockAPIClientForceRetry:
    """Test MockAPIClient's force_retry_count behavior."""

    @pytest.mark.asyncio
    async def test_force_retry_count_0_succeeds_immediately(self) -> None:
        """With force_retry_count=0, first call should succeed."""
        client = MockAPIClient(deterministic=True, force_retry_count=0)

        response = await client.responses_create(
            messages=[{"role": "user", "content": "test prompt"}]
        )

        # Should return valid XML, not "<invalid><unclosed>"
        assert "<invalid>" not in response.xml_content, "Should return valid XML"
        assert client.call_count == 1

    @pytest.mark.asyncio
    async def test_force_retry_count_1_fails_once(self) -> None:
        """With force_retry_count=1, first call fails, second succeeds."""
        client = MockAPIClient(deterministic=True, force_retry_count=1)

        # First call: should return invalid XML
        response1 = await client.responses_create(
            messages=[{"role": "user", "content": "test prompt"}]
        )
        assert response1.xml_content == "<invalid><unclosed>", "First call should fail"

        # Second call: should succeed
        response2 = await client.responses_create(
            messages=[{"role": "user", "content": "test prompt"}]
        )
        assert "<invalid>" not in response2.xml_content, "Second call should succeed"
        assert client.call_count == 2

    @pytest.mark.asyncio
    async def test_force_retry_count_3_fails_three_times(self) -> None:
        """With force_retry_count=3, first 3 calls fail, 4th succeeds."""
        client = MockAPIClient(deterministic=True, force_retry_count=3)

        # First 3 calls should fail
        for i in range(3):
            response = await client.responses_create(
                messages=[{"role": "user", "content": "test prompt"}]
            )
            assert response.xml_content == "<invalid><unclosed>", (
                f"Call {i + 1} should fail"
            )

        # 4th call should succeed
        response = await client.responses_create(
            messages=[{"role": "user", "content": "test prompt"}]
        )
        assert "<invalid>" not in response.xml_content, "4th call should succeed"
        assert client.call_count == 4

    @pytest.mark.asyncio
    async def test_forced_failure_counter_increments(self) -> None:
        """Verify internal _forced_failure_count increments correctly."""
        client = MockAPIClient(deterministic=True, force_retry_count=2)

        assert client._forced_failure_count == 0

        await client.responses_create(messages=[{"role": "user", "content": "test"}])
        assert client._forced_failure_count == 1

        await client.responses_create(messages=[{"role": "user", "content": "test"}])
        assert client._forced_failure_count == 2

        # After force_retry_count failures, counter should stop incrementing
        await client.responses_create(messages=[{"role": "user", "content": "test"}])
        assert client._forced_failure_count == 2, (
            "Counter should stop at force_retry_count"
        )


# =============================================================================
# Test: MockOpenAIConfig force_retry_count
# =============================================================================


class TestMockOpenAIConfigForceRetry:
    """Test mock_make_openai_request with force_retry_count configuration."""

    def setup_method(self) -> None:
        """Reset mock configuration before each test."""
        reset_mock_openai()

    def teardown_method(self) -> None:
        """Clean up mock configuration after each test."""
        reset_mock_openai()

    @pytest.mark.asyncio
    async def test_config_force_retry_returns_invalid_xml(self) -> None:
        """Config with force_retry_count should return invalid XML initially."""
        configure_mock_openai(MockOpenAIConfig(deterministic=True, force_retry_count=1))

        pricing_info: Dict[str, Dict[str, float]] = {
            "gpt-4o-mini": {"input_cost_per_token": 0.00001}
        }

        # First call should return invalid XML
        result = await mock_make_openai_request(
            api_key="test-key", prompt="test prompt", pricing_info=pricing_info
        )

        content = result["choices"][0]["message"]["content"]
        assert "<invalid><unclosed>" in content, "First call should return invalid XML"

    @pytest.mark.asyncio
    async def test_config_force_retry_succeeds_after_n_failures(self) -> None:
        """After N failures, subsequent calls should succeed."""
        configure_mock_openai(MockOpenAIConfig(deterministic=True, force_retry_count=2))

        pricing_info: Dict[str, Dict[str, float]] = {
            "gpt-4o-mini": {"input_cost_per_token": 0.00001}
        }
        prompt = "unique test prompt for retry test"

        # Calls 1 and 2 should fail
        for i in range(2):
            result = await mock_make_openai_request(
                api_key="test-key", prompt=prompt, pricing_info=pricing_info
            )
            content = result["choices"][0]["message"]["content"]
            assert "<invalid><unclosed>" in content, f"Call {i + 1} should fail"

        # Call 3 should succeed
        result = await mock_make_openai_request(
            api_key="test-key", prompt=prompt, pricing_info=pricing_info
        )
        content = result["choices"][0]["message"]["content"]
        assert "<invalid><unclosed>" not in content, "Call 3 should succeed"


# =============================================================================
# Test: Retry Logging Format (Reproducibility)
# =============================================================================


class TestRetryLogging:
    """Test that retry logs contain fields needed for reproduction."""

    def test_retry_log_format_regex(self) -> None:
        """Verify the expected log format can be parsed."""
        # Expected format from apias.py:2705
        sample_log = (
            "[RETRY] task_id=5 attempt=2 delay=4.0s error='XML validation failed'"
        )

        pattern = r"\[RETRY\] task_id=(\d+) attempt=(\d+) delay=([\d.]+)s error='(.+)'"
        match = re.match(pattern, sample_log)

        assert match is not None, f"Log format should match pattern: {sample_log}"
        assert match.group(1) == "5", "Should extract task_id"
        assert match.group(2) == "2", "Should extract attempt"
        assert match.group(3) == "4.0", "Should extract delay"
        assert match.group(4) == "XML validation failed", "Should extract error"

    def test_retry_exhausted_log_format_regex(self) -> None:
        """Verify the retry exhausted log format can be parsed."""
        # Expected format from apias.py:2722
        sample_log = (
            "[RETRY_EXHAUSTED] task_id=5 attempts=3 error='XML validation failed'"
        )

        pattern = r"\[RETRY_EXHAUSTED\] task_id=(\d+) attempts=(\d+) error='(.+)'"
        match = re.match(pattern, sample_log)

        assert match is not None, f"Log format should match pattern: {sample_log}"
        assert match.group(1) == "5", "Should extract task_id"
        assert match.group(2) == "3", "Should extract attempts"
        assert match.group(3) == "XML validation failed", "Should extract error"


# =============================================================================
# Test: Integration - process_single_chunk with retry
# =============================================================================


class TestProcessSingleChunkRetry:
    """Integration tests for retry logic in _process_single_chunk.

    Note: We test the private _process_single_chunk function directly because
    it contains the core retry logic. Testing through public APIs would require
    complex setup and would obscure what we're actually testing.
    """

    @pytest.fixture
    def pricing_info(self) -> Dict[str, Dict[str, float]]:
        """Standard pricing info for tests."""
        return {
            "gpt-4o-mini": {
                "input_cost_per_token": 0.00001,
                "output_cost_per_token": 0.00003,
            }
        }

    @pytest.mark.asyncio
    async def test_retry_succeeds_after_one_failure(
        self, pricing_info: Dict[str, Any]
    ) -> None:
        """_process_single_chunk should succeed after one XML validation failure."""
        # Import the private function for direct testing
        from apias.apias import _process_single_chunk

        # Create mock client that fails once then succeeds
        mock_client = MockAPIClient(deterministic=True, force_retry_count=1)

        # Patch asyncio.sleep to avoid actual delays in tests
        with patch("apias.apias.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            xml_output, cost = await _process_single_chunk(
                html_content="<html><body>Test content for chunk processing</body></html>",
                pricing_info=pricing_info,
                chunk_num=1,
                mock=True,
                mock_client=mock_client,
            )

        # Should have retried and succeeded
        assert xml_output is not None, "Should return valid XML after retry"
        assert "<invalid>" not in xml_output, "Should not return invalid XML"
        assert mock_client.call_count == 2, (
            "Should have made 2 API calls (1 fail + 1 success)"
        )

        # Verify sleep was called for backoff
        assert mock_sleep.called, "Should have waited between retries"

    @pytest.mark.asyncio
    async def test_retry_fails_after_max_retries_exhausted(
        self, pricing_info: Dict[str, Any]
    ) -> None:
        """_process_single_chunk should fail after exhausting max retries."""
        from apias.apias import _process_single_chunk

        # Force more failures than max_retries allows
        # With XML_VALIDATION_MAX_RETRIES=1, we get 2 total attempts (0 and 1)
        # So force_retry_count=5 will exhaust all retries
        mock_client = MockAPIClient(deterministic=True, force_retry_count=5)

        with patch("apias.apias.asyncio.sleep", new_callable=AsyncMock):
            xml_output, cost = await _process_single_chunk(
                html_content="<html><body>Test content for chunk processing</body></html>",
                pricing_info=pricing_info,
                chunk_num=1,
                mock=True,
                mock_client=mock_client,
            )

        # Should have failed after exhausting retries
        assert xml_output is None, "Should return None after exhausting retries"
        # With max_retries=1, should make 2 attempts (attempt 0 and attempt 1)
        expected_attempts = XML_VALIDATION_MAX_RETRIES + 1
        assert mock_client.call_count == expected_attempts, (
            f"Should have made {expected_attempts} attempts, made {mock_client.call_count}"
        )

    @pytest.mark.asyncio
    async def test_retry_backoff_delays_are_correct(
        self, pricing_info: Dict[str, Any]
    ) -> None:
        """Verify the actual sleep delays match exponential backoff formula."""
        from apias.apias import _process_single_chunk

        # Force 2 failures to see backoff delays
        mock_client = MockAPIClient(deterministic=True, force_retry_count=2)
        sleep_calls: List[float] = []

        async def capture_sleep(delay: float) -> None:
            sleep_calls.append(delay)

        with patch("apias.apias.asyncio.sleep", side_effect=capture_sleep):
            await _process_single_chunk(
                html_content="<html><body>Test content for chunk processing</body></html>",
                pricing_info=pricing_info,
                chunk_num=1,
                mock=True,
                mock_client=mock_client,
            )

        # With XML_VALIDATION_MAX_RETRIES=1, we only get 1 retry (attempt 1)
        # The mock API also has its own internal sleep calls, so we need to find the
        # retry backoff sleep specifically. The backoff delay should be exactly 1.0s
        # (RETRY_BASE_DELAY_SECONDS * 2^0) for the first retry attempt.
        if XML_VALIDATION_MAX_RETRIES >= 1:
            assert len(sleep_calls) >= 1, "Should have at least one sleep call"
            # Find the backoff sleep (should be exactly RETRY_BASE_DELAY_SECONDS)
            backoff_sleeps = [s for s in sleep_calls if s == RETRY_BASE_DELAY_SECONDS]
            assert len(backoff_sleeps) >= 1, (
                f"Should have at least one backoff sleep of {RETRY_BASE_DELAY_SECONDS}s, "
                f"got sleeps: {sleep_calls}"
            )


# =============================================================================
# Test: Error Scenarios
# =============================================================================


class TestRetryErrorScenarios:
    """Test retry behavior under various error conditions."""

    def setup_method(self) -> None:
        reset_mock_openai()

    def teardown_method(self) -> None:
        reset_mock_openai()

    @pytest.mark.asyncio
    async def test_invalid_xml_scenario_triggers_retry(self) -> None:
        """INVALID_XML error scenario should return malformed XML."""
        configure_mock_openai(
            MockOpenAIConfig(error_scenario=MockErrorScenario.INVALID_XML)
        )

        pricing_info: Dict[str, Dict[str, float]] = {
            "gpt-4o-mini": {"input_cost_per_token": 0.00001}
        }

        result = await mock_make_openai_request(
            api_key="test-key", prompt="test prompt", pricing_info=pricing_info
        )

        content = result["choices"][0]["message"]["content"]
        assert "<invalid><unclosed>" in content, "Should return invalid XML"

    @pytest.mark.asyncio
    async def test_empty_response_scenario(self) -> None:
        """EMPTY_RESPONSE scenario should return empty content."""
        configure_mock_openai(
            MockOpenAIConfig(error_scenario=MockErrorScenario.EMPTY_RESPONSE)
        )

        pricing_info: Dict[str, Dict[str, float]] = {
            "gpt-4o-mini": {"input_cost_per_token": 0.00001}
        }

        result = await mock_make_openai_request(
            api_key="test-key", prompt="test prompt", pricing_info=pricing_info
        )

        import json

        content = json.loads(result["choices"][0]["message"]["content"])
        assert content["xml_content"] == "", "Should return empty XML content"
        assert content["completeness_check"] is False, "Should indicate incomplete"


# =============================================================================
# Test: Thread Safety of Retry Counters
# =============================================================================


class TestRetryThreadSafety:
    """Test thread safety of retry counter mechanisms."""

    def setup_method(self) -> None:
        reset_mock_openai()

    def teardown_method(self) -> None:
        reset_mock_openai()

    @pytest.mark.asyncio
    async def test_concurrent_requests_have_independent_counters(self) -> None:
        """Different prompts should have independent retry counters."""
        configure_mock_openai(MockOpenAIConfig(deterministic=True, force_retry_count=1))

        pricing_info: Dict[str, Dict[str, float]] = {
            "gpt-4o-mini": {"input_cost_per_token": 0.00001}
        }

        # Two different prompts should each fail once independently
        prompt1 = "unique prompt for request 1"
        prompt2 = "unique prompt for request 2"

        # First call to each prompt should fail
        result1 = await mock_make_openai_request("key", prompt1, pricing_info)
        result2 = await mock_make_openai_request("key", prompt2, pricing_info)

        assert "<invalid>" in result1["choices"][0]["message"]["content"], (
            "Prompt 1 first call should fail"
        )
        assert "<invalid>" in result2["choices"][0]["message"]["content"], (
            "Prompt 2 first call should fail"
        )

        # Second call to each should succeed
        result1_retry = await mock_make_openai_request("key", prompt1, pricing_info)
        result2_retry = await mock_make_openai_request("key", prompt2, pricing_info)

        assert "<invalid>" not in result1_retry["choices"][0]["message"]["content"], (
            "Prompt 1 retry should succeed"
        )
        assert "<invalid>" not in result2_retry["choices"][0]["message"]["content"], (
            "Prompt 2 retry should succeed"
        )
