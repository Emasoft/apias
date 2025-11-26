"""Tests for AsyncOpenAI integration"""

import asyncio
import time
from typing import Any, Dict, List, Tuple
import pytest  # type: ignore[import-not-found]
from unittest.mock import AsyncMock, Mock, patch
from openai import AsyncOpenAI, APITimeoutError, RateLimitError, APIConnectionError  # type: ignore[import-not-found]
from apias.apias import (
    make_openai_request,
    call_openai_api,
    call_llm_to_convert_html_to_xml,
    _process_single_chunk,
)


# Fixtures


@pytest.fixture  # type: ignore[misc]
def mock_pricing_info() -> Dict[str, Dict[str, float]]:
    """Sample pricing information for testing"""
    return {
        "gpt-5-nano-2025-08-07": {
            "input_cost_per_token": 0.0000001,
            "output_cost_per_token": 0.0000002,
        }
    }


@pytest.fixture  # type: ignore[misc]
def sample_html_content() -> str:
    """Sample HTML content for testing"""
    return """
    <html>
        <body>
            <h1>API Documentation</h1>
            <p>This is test documentation</p>
        </body>
    </html>
    """


@pytest.fixture  # type: ignore[misc]
def mock_openai_response() -> Mock:
    """Mock OpenAI API response"""
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = '{"xml_output": "<api>Test</api>"}'
    mock_response.usage = Mock()
    mock_response.usage.prompt_tokens = 100
    mock_response.usage.completion_tokens = 50
    mock_response.model = "gpt-5-nano-2025-08-07"
    return mock_response


# Tests for make_openai_request


@pytest.mark.asyncio  # type: ignore[misc]
async def test_make_openai_request_success(
    mock_pricing_info: Dict[str, Dict[str, float]], mock_openai_response: Mock
) -> None:
    """Test successful OpenAI API request"""
    with patch("apias.apias.AsyncOpenAI") as mock_client_class:
        # Setup mock
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client
        mock_client.chat.completions.create = AsyncMock(
            return_value=mock_openai_response
        )

        # Execute
        result = await make_openai_request(
            api_key="test_key",
            prompt="Test prompt",
            pricing_info=mock_pricing_info,
            model="gpt-5-nano-2025-08-07",
        )

        # Verify
        assert result is not None
        assert "choices" in result
        assert "request_cost" in result
        assert result["request_cost"] > 0
        assert "message" in result["choices"][0]
        mock_client.chat.completions.create.assert_called_once()


@pytest.mark.asyncio  # type: ignore[misc]
async def test_make_openai_request_timeout(
    mock_pricing_info: Dict[str, Dict[str, float]],
) -> None:
    """Test OpenAI API request timeout handling"""
    with patch("apias.apias.AsyncOpenAI") as mock_client_class:
        # Setup mock to raise timeout
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client
        mock_request = Mock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=APITimeoutError(request=mock_request)
        )

        # Execute and verify exception
        with pytest.raises(APITimeoutError):
            await make_openai_request(
                api_key="test_key",
                prompt="Test prompt",
                pricing_info=mock_pricing_info,
            )


@pytest.mark.asyncio  # type: ignore[misc]
async def test_make_openai_request_rate_limit(
    mock_pricing_info: Dict[str, Dict[str, float]],
) -> None:
    """Test OpenAI API rate limit handling"""
    with patch("apias.apias.AsyncOpenAI") as mock_client_class:
        # Setup mock to raise rate limit error
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client
        mock_response = Mock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=RateLimitError(
                "Rate limit exceeded", response=mock_response, body=None
            )
        )

        # Execute and verify exception
        with pytest.raises(RateLimitError):
            await make_openai_request(
                api_key="test_key",
                prompt="Test prompt",
                pricing_info=mock_pricing_info,
            )


@pytest.mark.asyncio  # type: ignore[misc]
async def test_make_openai_request_connection_error(
    mock_pricing_info: Dict[str, Dict[str, float]],
) -> None:
    """Test OpenAI API connection error handling"""
    with patch("apias.apias.AsyncOpenAI") as mock_client_class:
        # Setup mock to raise connection error
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client
        mock_client.chat.completions.create = AsyncMock(
            side_effect=APIConnectionError(request=Mock())
        )

        # Execute and verify exception
        with pytest.raises(APIConnectionError):
            await make_openai_request(
                api_key="test_key",
                prompt="Test prompt",
                pricing_info=mock_pricing_info,
            )


# Tests for call_openai_api


@pytest.mark.asyncio  # type: ignore[misc]
async def test_call_openai_api_success(
    mock_pricing_info: Dict[str, Dict[str, float]],
) -> None:
    """Test successful call_openai_api execution"""
    with patch("apias.apias.get_openai_api_key") as mock_get_key:
        with patch("apias.apias.make_openai_request") as mock_request:
            # Setup mocks
            mock_get_key.return_value = "test_key"
            mock_request.return_value = {
                "choices": [
                    {
                        "message": {
                            "content": '{"xml_content": "<api>Test</api>", "document_type": "MODULE"}'
                        }
                    }
                ],
                "request_cost": 0.001,
            }

            # Execute
            xml_output, cost = await call_openai_api(
                prompt="Test prompt", pricing_info=mock_pricing_info
            )

            # Verify
            assert xml_output == "<api>Test</api>"
            assert cost == 0.001
            mock_request.assert_called_once()


@pytest.mark.asyncio  # type: ignore[misc]
async def test_call_openai_api_no_output(
    mock_pricing_info: Dict[str, Dict[str, float]],
) -> None:
    """Test call_openai_api with no XML output"""
    with patch("apias.apias.get_openai_api_key") as mock_get_key:
        with patch("apias.apias.make_openai_request") as mock_request:
            # Setup mocks
            mock_get_key.return_value = "test_key"
            mock_request.return_value = {
                "choices": [
                    {
                        "message": {
                            "content": '{"xml_content": "", "document_type": "MODULE"}'
                        }
                    }
                ],
                "request_cost": 0.001,
            }

            # Execute
            xml_output, cost = await call_openai_api(
                prompt="Test prompt", pricing_info=mock_pricing_info
            )

            # Verify
            assert xml_output == ""
            assert cost == 0.001


# Tests for _process_single_chunk


@pytest.mark.asyncio  # type: ignore[misc]
async def test_process_single_chunk_success(
    mock_pricing_info: Dict[str, Dict[str, float]],
) -> None:
    """Test successful single chunk processing"""
    with patch("apias.apias.call_openai_api") as mock_api:
        with patch("apias.apias.extract_xml_from_input") as mock_extract:
            # Setup mocks
            mock_api.return_value = ("<xml>Test</xml>", 0.001)
            mock_extract.return_value = "<xml>Test</xml>"

            # Execute
            xml_output, cost = await _process_single_chunk(
                html_content="Test content",
                pricing_info=mock_pricing_info,
                chunk_num=1,
            )

            # Verify
            assert xml_output == "<xml>Test</xml>"
            assert cost == 0.001


@pytest.mark.asyncio  # type: ignore[misc]
async def test_process_single_chunk_failure(
    mock_pricing_info: Dict[str, Dict[str, float]],
) -> None:
    """Test single chunk processing failure with retry logic"""
    with patch("apias.apias.call_openai_api") as mock_api:
        with patch("apias.apias.extract_xml_from_input") as mock_extract:
            # Setup mocks - call_openai_api returns None for xml_output
            # The function retries once on failure, so cost is 2x
            mock_api.return_value = (None, 0.001)
            mock_extract.return_value = None

            # Execute
            xml_output, cost = await _process_single_chunk(
                html_content="Test content",
                pricing_info=mock_pricing_info,
                chunk_num=1,
            )

            # Verify - cost is 0.002 because function retries once on failure
            assert xml_output is None
            assert cost == 0.002


# Tests for asyncio.gather parallelism


@pytest.mark.asyncio  # type: ignore[misc]
async def test_asyncio_gather_parallelism() -> None:
    """Test that asyncio.gather actually runs tasks in parallel"""
    execution_times: List[Tuple[int, float, float]] = []

    async def mock_task(task_id: int, delay: float) -> int:
        """Mock task that records execution time"""
        start = time.time()
        await asyncio.sleep(delay)
        end = time.time()
        execution_times.append((task_id, start, end))
        return task_id

    # Run 3 tasks in parallel
    tasks = [mock_task(i, 0.1) for i in range(3)]
    results = await asyncio.gather(*tasks)

    # Verify all tasks completed
    assert len(results) == 3

    # Verify parallel execution: all tasks should start within a short window
    start_times = [t[1] for t in execution_times]
    max_start_difference = max(start_times) - min(start_times)

    # If truly parallel, all should start within ~0.01s of each other
    assert max_start_difference < 0.05, "Tasks did not execute in parallel"


# Tests for timeout calculation


@pytest.mark.asyncio  # type: ignore[misc]
async def test_make_openai_request_timeout_calculation(
    mock_pricing_info: Dict[str, Dict[str, float]], mock_openai_response: Mock
) -> None:
    """Test that timeout is properly calculated based on prompt length"""
    with patch("apias.apias.AsyncOpenAI") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client
        mock_client.chat.completions.create = AsyncMock(
            return_value=mock_openai_response
        )

        # Test with short prompt
        short_prompt = "x" * 100
        await make_openai_request(
            api_key="test_key",
            prompt=short_prompt,
            pricing_info=mock_pricing_info,
        )

        # Verify AsyncOpenAI was created with timeout parameter
        assert mock_client_class.called
        call_kwargs = mock_client_class.call_args.kwargs
        assert "timeout" in call_kwargs
        assert call_kwargs["timeout"] > 0

        # Test with long prompt
        mock_client_class.reset_mock()
        long_prompt = "x" * 100000
        await make_openai_request(
            api_key="test_key",
            prompt=long_prompt,
            pricing_info=mock_pricing_info,
        )

        # Verify timeout is capped at max value (1200 seconds)
        call_kwargs = mock_client_class.call_args.kwargs
        assert call_kwargs["timeout"] <= 1200


# Tests for max_retries configuration


@pytest.mark.asyncio  # type: ignore[misc]
async def test_make_openai_request_retry_configuration(
    mock_pricing_info: Dict[str, Dict[str, float]], mock_openai_response: Mock
) -> None:
    """Test that max_retries is properly configured"""
    with patch("apias.apias.AsyncOpenAI") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client
        mock_client.chat.completions.create = AsyncMock(
            return_value=mock_openai_response
        )

        # Execute
        await make_openai_request(
            api_key="test_key",
            prompt="Test prompt",
            pricing_info=mock_pricing_info,
        )

        # Verify AsyncOpenAI was created with max_retries=2
        assert mock_client_class.called
        call_kwargs = mock_client_class.call_args.kwargs
        assert "max_retries" in call_kwargs
        assert call_kwargs["max_retries"] == 2
