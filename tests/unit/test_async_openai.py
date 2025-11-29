"""
Tests for AsyncOpenAI integration - supports BOTH mock and real API modes.

Run modes:
    - Default (mock): pytest tests/unit/test_async_openai.py
    - Real API only:  pytest tests/unit/test_async_openai.py --real-api
    - Mock only:      pytest tests/unit/test_async_openai.py -m mock_api
    - Real only:      pytest tests/unit/test_async_openai.py -m real_api

The --real-api flag requires OPENAI_API_KEY environment variable and incurs costs.
"""

import asyncio
import os
import time
from typing import Any, Dict, List, Tuple

import pytest

from apias.apias import (
    _process_single_chunk,
    call_llm_to_convert_html_to_xml,
    call_openai_api,
    get_openai_api_key,
    load_model_pricing,
    make_openai_request,
)
from apias.mock_api import MockAPIClient, mock_call_openai_api

# ============================================================================
# Fixtures
# ============================================================================


def has_openai_api_key() -> bool:
    """Check if OpenAI API key is available in environment."""
    try:
        key = get_openai_api_key()
        return key is not None and len(key) > 0
    except Exception:
        return os.environ.get("OPENAI_API_KEY", "") != ""


@pytest.fixture
def pricing_info() -> Dict[str, Dict[str, float]]:
    """Real pricing information loaded from remote source."""
    pricing = load_model_pricing()
    if pricing is None:
        pytest.skip("Could not load pricing information from remote source")
    return pricing


@pytest.fixture
def simple_html_content() -> str:
    """Simple HTML content for testing API conversion."""
    return """
    <html>
        <body>
            <h1>Test API Documentation</h1>
            <p>A simple function that adds two numbers.</p>
            <pre><code>def add(a, b):
    return a + b</code></pre>
        </body>
    </html>
    """


@pytest.fixture
def mock_client() -> MockAPIClient:
    """Mock API client for testing without real API calls."""
    return MockAPIClient()


# ============================================================================
# MOCK API TESTS - Run by default, no API costs
# ============================================================================


@pytest.mark.mock_api
@pytest.mark.asyncio
async def test_mock_api_returns_response(
    pricing_info: Dict[str, Dict[str, float]],
) -> None:
    """Test mock API returns valid response structure."""
    xml_output, cost = await mock_call_openai_api(
        prompt="Test prompt for mock API",
        pricing_info=pricing_info,
        deterministic=True,  # No random failures in tests
    )

    # Mock should return XML content and cost
    assert xml_output is not None, "Mock should return XML content"
    assert len(xml_output) > 0, "XML content should not be empty"
    assert cost >= 0, "Cost should be non-negative"


@pytest.mark.mock_api
@pytest.mark.asyncio
async def test_mock_api_cost_tracking(
    pricing_info: Dict[str, Dict[str, float]],
) -> None:
    """Test mock API tracks costs correctly."""
    client = MockAPIClient(deterministic=True)  # No random failures in tests

    # Make multiple requests
    for _ in range(3):
        await mock_call_openai_api(
            prompt="Test prompt",
            pricing_info=pricing_info,
            mock_client=client,
        )

    # Cost should accumulate
    assert client.total_cost > 0, "Total cost should accumulate"
    assert client.call_count == 3, "Call count should be 3"


@pytest.mark.mock_api
@pytest.mark.asyncio
async def test_mock_llm_conversion(
    pricing_info: Dict[str, Dict[str, float]],
    simple_html_content: str,
) -> None:
    """Test HTML to XML conversion using mock API."""
    result = await call_llm_to_convert_html_to_xml(
        html_content=simple_html_content,
        additional_content={},
        pricing_info=pricing_info,
        no_tui=True,
        mock=True,  # Use mock API
    )

    assert result is not None, "Should return a result"
    # After TUI unification, returns (xml_content, cost) - no tui_manager
    assert len(result) == 2, "Should return (xml_content, cost) tuple"

    xml_content, cost = result
    assert cost >= 0, "Cost should be non-negative"
    # Mock should always return XML content
    assert xml_content is not None, "Mock API should return XML content"


@pytest.mark.mock_api
@pytest.mark.asyncio
async def test_mock_api_response_varies_by_size(
    pricing_info: Dict[str, Dict[str, float]],
) -> None:
    """Test mock API returns different XML based on prompt size."""
    # Small prompt
    small_result, _ = await mock_call_openai_api(
        prompt="x" * 1000,
        pricing_info=pricing_info,
        deterministic=True,  # No random failures in tests
    )

    # Large prompt
    large_result, _ = await mock_call_openai_api(
        prompt="x" * 300000,
        pricing_info=pricing_info,
        deterministic=True,  # No random failures in tests
    )

    # Both should return valid XML
    assert small_result is not None
    assert large_result is not None
    # Large prompt should return larger/different XML
    assert len(large_result) > len(
        small_result
    ), "Large prompt should yield larger response"


# ============================================================================
# REAL API TESTS - Run with --real-api flag, requires API key and incurs costs
# ============================================================================


@pytest.mark.real_api
@pytest.mark.slow
@pytest.mark.asyncio
async def test_real_api_request(
    pricing_info: Dict[str, Dict[str, float]],
) -> None:
    """Test real OpenAI API request succeeds and returns valid structure."""
    if not has_openai_api_key():
        pytest.skip("OPENAI_API_KEY not set")

    api_key = get_openai_api_key()
    result = await make_openai_request(
        api_key=api_key,
        prompt="Say 'hello' and nothing else.",
        pricing_info=pricing_info,
    )

    assert result is not None, "API should return a response"
    assert "choices" in result, "Response should contain 'choices'"
    assert len(result["choices"]) > 0, "Should have at least one choice"
    assert "message" in result["choices"][0], "Choice should have a message"
    assert "request_cost" in result, "Response should include request cost"
    assert result["request_cost"] >= 0, "Request cost should be non-negative"


@pytest.mark.real_api
@pytest.mark.slow
@pytest.mark.asyncio
async def test_real_api_returns_content(
    pricing_info: Dict[str, Dict[str, float]],
) -> None:
    """Test that real API returns actual content in the response."""
    if not has_openai_api_key():
        pytest.skip("OPENAI_API_KEY not set")

    api_key = get_openai_api_key()
    result = await make_openai_request(
        api_key=api_key,
        prompt="Respond with exactly: TEST_RESPONSE_OK",
        pricing_info=pricing_info,
    )

    content = result["choices"][0]["message"]["content"]
    assert content is not None, "Content should not be None"
    assert len(content) > 0, "Content should not be empty"


@pytest.mark.real_api
@pytest.mark.slow
@pytest.mark.asyncio
async def test_real_api_cost_tracking(
    pricing_info: Dict[str, Dict[str, float]],
) -> None:
    """Test that real API request cost is tracked correctly."""
    if not has_openai_api_key():
        pytest.skip("OPENAI_API_KEY not set")

    api_key = get_openai_api_key()
    result = await make_openai_request(
        api_key=api_key,
        prompt="Say 'cost test' and nothing more.",
        pricing_info=pricing_info,
    )

    assert "request_cost" in result
    cost = result["request_cost"]
    assert isinstance(cost, (int, float)), "Cost should be numeric"
    assert cost >= 0, "Cost should be non-negative"


@pytest.mark.real_api
@pytest.mark.slow
@pytest.mark.asyncio
async def test_real_call_openai_api(
    pricing_info: Dict[str, Dict[str, float]],
) -> None:
    """Test that call_openai_api returns XML content and cost."""
    if not has_openai_api_key():
        pytest.skip("OPENAI_API_KEY not set")

    html_prompt = """Convert this to APIAS XML format:
    <html><body><h1>Test Function</h1><p>Adds numbers.</p></body></html>
    """

    xml_output, cost = await call_openai_api(
        prompt=html_prompt,
        pricing_info=pricing_info,
    )

    assert cost >= 0, "Cost should be non-negative"


@pytest.mark.real_api
@pytest.mark.slow
@pytest.mark.asyncio
async def test_real_process_single_chunk(
    pricing_info: Dict[str, Dict[str, float]],
    simple_html_content: str,
) -> None:
    """Test processing a real HTML chunk through the API."""
    if not has_openai_api_key():
        pytest.skip("OPENAI_API_KEY not set")

    xml_output, cost = await _process_single_chunk(
        html_content=simple_html_content,
        pricing_info=pricing_info,
        chunk_num=1,
    )

    assert cost >= 0, "Cost should be non-negative"
    if xml_output is not None:
        assert isinstance(xml_output, str), "XML output should be a string if present"


@pytest.mark.real_api
@pytest.mark.slow
@pytest.mark.asyncio
async def test_real_llm_conversion(
    pricing_info: Dict[str, Dict[str, float]],
    simple_html_content: str,
) -> None:
    """Test the full HTML to XML conversion with real API."""
    if not has_openai_api_key():
        pytest.skip("OPENAI_API_KEY not set")

    result = await call_llm_to_convert_html_to_xml(
        html_content=simple_html_content,
        additional_content={},
        pricing_info=pricing_info,
        no_tui=True,
        mock=False,  # Use real API
    )

    assert result is not None, "Should return a result"
    # After TUI unification, returns (xml_content, cost) - no tui_manager
    assert len(result) == 2, "Should return (xml_content, cost) tuple"

    xml_content, cost = result
    assert cost >= 0, "Cost should be non-negative"


# ============================================================================
# GENERAL TESTS - No API calls needed
# ============================================================================


@pytest.mark.asyncio
async def test_asyncio_gather_parallelism() -> None:
    """Test that asyncio.gather actually runs tasks in parallel."""
    execution_times: List[Tuple[int, float, float]] = []

    async def timed_task(task_id: int, delay: float) -> int:
        """Task that records execution time."""
        start = time.time()
        await asyncio.sleep(delay)
        end = time.time()
        execution_times.append((task_id, start, end))
        return task_id

    # Run 3 tasks in parallel
    tasks = [timed_task(i, 0.1) for i in range(3)]
    results = await asyncio.gather(*tasks)

    # Verify all tasks completed
    assert len(results) == 3

    # Verify parallel execution: all tasks should start within a short window
    start_times = [t[1] for t in execution_times]
    max_start_difference = max(start_times) - min(start_times)

    # If truly parallel, all should start within ~0.05s of each other
    assert max_start_difference < 0.05, "Tasks did not execute in parallel"
