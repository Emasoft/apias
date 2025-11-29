"""
Mock API module for testing without spending tokens.

Provides two levels of mocking:
1. MockAPIClient - Low-level TUI simulation with variable delays
2. mock_make_openai_request - High-level drop-in replacement for make_openai_request()

Features:
- Deterministic responses based on prompt content hash
- Exact response structure matching real OpenAI API
- Configurable error simulation (rate limits, timeouts, malformed responses)
- Realistic cost calculation
- Support for testing all "real_api" tests without actual API calls
"""

import asyncio
import hashlib
import json
import random
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple


class MockAPIClient:
    """Mock OpenAI API client that simulates realistic behavior"""

    def __init__(self, deterministic: bool = False) -> None:
        """
        Initialize mock client.

        Args:
            deterministic: If True, disables random failures and jitter for predictable testing
        """
        self.total_cost = 0.0
        self.call_count = 0
        self.deterministic = deterministic

    async def responses_create(self, **kwargs: Any) -> "MockResponse":
        """
        Simulate OpenAI API call with realistic delays and occasional failures.

        Simulates:
        - Variable latency (300ms - 2500ms) unless deterministic mode
        - ~10% failure rate on first attempt unless deterministic mode
        - Realistic cost calculation
        """
        self.call_count += 1

        # Extract prompt size to calculate realistic delay and cost
        messages = kwargs.get("messages", [])
        prompt = messages[0]["content"] if messages else ""
        prompt_size = len(prompt)

        # Simulate realistic API latency based on prompt size
        base_delay = 0.3  # 300ms minimum
        size_factor = prompt_size / 100000  # Longer prompts take longer

        if self.deterministic:
            # Predictable delay for testing
            delay = min(base_delay + size_factor, 2.5)
        else:
            # Random jitter for realistic simulation
            random_jitter = random.uniform(0, 0.5)
            delay = min(base_delay + size_factor + random_jitter, 2.5)

        await asyncio.sleep(delay)

        # Simulate occasional failures (~10% on first attempt) unless deterministic
        if not self.deterministic and random.random() < 0.1:
            raise MockAPIException("Simulated API failure - will retry")

        # Calculate realistic cost (based on GPT-4 pricing approximation)
        # Input: ~$0.01 per 1K tokens, Output: ~$0.03 per 1K tokens
        input_tokens = prompt_size // 4  # Rough estimate: 4 chars per token
        output_tokens = random.randint(1000, 5000)  # Varies by chunk
        cost = (input_tokens * 0.00001) + (output_tokens * 0.00003)
        self.total_cost += cost

        # Generate mock XML response
        xml_content = self._generate_mock_xml(prompt_size)

        return MockResponse(xml_content=xml_content, cost=cost)

    def _generate_mock_xml(self, prompt_size: int) -> str:
        """Generate realistic mock XML based on prompt size"""

        # Determine content type based on size
        if prompt_size > 200000:
            # Large chunk - probably contains a class with many methods
            return """<MODULE>
<NAME>textual.app</NAME>
<CLASS>
<NAME>App</NAME>
<CLASS_DESCRIPTION>Bases: Generic[ReturnType], DOMNode. The base class for Textual Applications.</CLASS_DESCRIPTION>
<CLASS_API>
<METHOD>
<NAME>action_add_class</NAME>
<SIGNATURE>async action_add_class(selector, class_name)</SIGNATURE>
<RETURN_TYPE>None</RETURN_TYPE>
<DESCRIPTION>Add a CSS class to selected widgets.</DESCRIPTION>
<PARAMETERS>
<PARAMETER>
<NAME>selector</NAME>
<TYPE>str</TYPE>
<DESCRIPTION>CSS selector to target widgets.</DESCRIPTION>
<DEFAULT>required</DEFAULT>
</PARAMETER>
<PARAMETER>
<NAME>class_name</NAME>
<TYPE>str</TYPE>
<DESCRIPTION>The class name to add.</DESCRIPTION>
<DEFAULT>required</DEFAULT>
</PARAMETER>
</PARAMETERS>
</METHOD>
<FIELD>
<NAME>ALLOW_IN_MAXIMIZED_VIEW</NAME>
<SIGNATURE>ALLOW_IN_MAXIMIZED_VIEW = 'Footer'</SIGNATURE>
<DESCRIPTION>The default value of Screen.ALLOW_IN_MAXIMIZED_VIEW.</DESCRIPTION>
</FIELD>
</CLASS_API>
</CLASS>
</MODULE>"""
        elif prompt_size > 50000:
            # Medium chunk - maybe a class or module section
            return """<CLASS>
<NAME>ActionError</NAME>
<CLASS_DESCRIPTION>Bases: Exception. Base class for exceptions relating to actions.</CLASS_DESCRIPTION>
</CLASS>"""
        else:
            # Small chunk - variables or simple elements
            return """<VARIABLE modifiers="module-attribute" name="ScreenType">
<SIGNATURE>ScreenType = TypeVar('ScreenType', bound=Screen)</SIGNATURE>
<DESCRIPTION>Type var for a Screen, used in get_screen.</DESCRIPTION>
</VARIABLE>"""


class MockResponse:
    """Mock API response object"""

    def __init__(self, xml_content: str, cost: float) -> None:
        self.xml_content = xml_content
        self.cost = cost
        # Wrap in JSON structure to match real API
        # Properly escape XML content for JSON to handle newlines and special characters
        escaped_xml = json.dumps(xml_content)
        self.content = type(
            "Content",
            (),
            {
                "text": f'{{"xml_content": {escaped_xml}, "document_type": "MODULE", "completeness_check": true}}'
            },
        )()


class MockAPIException(Exception):
    """Mock API exception for simulating failures"""

    pass


async def mock_call_openai_api(
    prompt: str,
    pricing_info: dict[str, Any],
    mock_client: Optional[MockAPIClient] = None,
    deterministic: bool = False,
) -> Tuple[Optional[str], float]:
    """
    Mock version of call_openai_api that simulates realistic behavior.

    Args:
        prompt: The prompt to send (used for size calculations)
        pricing_info: Pricing information (ignored in mock)
        mock_client: Optional shared mock client for cost tracking
        deterministic: If True, disables random failures for predictable testing

    Returns:
        Tuple of (xml_content, cost)
    """
    if mock_client is None:
        mock_client = MockAPIClient(deterministic=deterministic)

    try:
        response = await mock_client.responses_create(
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )

        result = json.loads(response.content.text)
        xml_content = result.get("xml_content", "")

        return xml_content, response.cost

    except MockAPIException as e:
        # Simulate retry-able failure
        raise ValueError(f"Mock API error: {e}") from e
    except Exception as e:
        # Unexpected error
        raise RuntimeError(f"Mock API unexpected error: {e}") from e


# ============================================================================
# High-Level Mock for make_openai_request()
# ============================================================================


class MockErrorScenario(Enum):
    """Error scenarios that can be simulated by the mock."""

    NONE = auto()  # No error - normal response
    RATE_LIMIT = auto()  # 429 rate limit error
    TIMEOUT = auto()  # API timeout
    CONNECTION_ERROR = auto()  # Network connection failed
    MALFORMED_JSON = auto()  # Response is not valid JSON
    INVALID_XML = auto()  # XML content is malformed
    EMPTY_RESPONSE = auto()  # Empty content in response
    QUOTA_EXCEEDED = auto()  # Insufficient quota


@dataclass
class MockOpenAIConfig:
    """
    Configuration for mock OpenAI behavior.

    Attributes:
        deterministic: If True, same prompt always gives same response
        error_scenario: Which error to simulate (or NONE for success)
        delay_seconds: Simulated API latency (0 for instant)
        model: Model name to return in response
        custom_responses: Dict mapping prompt hashes to custom XML responses
    """

    deterministic: bool = True
    error_scenario: MockErrorScenario = MockErrorScenario.NONE
    delay_seconds: float = 0.0
    model: str = "gpt-5-nano-2025-08-07"
    custom_responses: Dict[str, str] = field(default_factory=dict)


# WHY global: Allows tests to configure mock behavior before running
_mock_config: MockOpenAIConfig = MockOpenAIConfig()


def configure_mock_openai(config: MockOpenAIConfig) -> None:
    """
    Configure mock OpenAI behavior for testing.

    Args:
        config: MockOpenAIConfig instance with desired settings

    Example:
        configure_mock_openai(MockOpenAIConfig(
            error_scenario=MockErrorScenario.RATE_LIMIT,
            delay_seconds=0.1
        ))
    """
    global _mock_config
    _mock_config = config


def reset_mock_openai() -> None:
    """Reset mock configuration to defaults."""
    global _mock_config
    _mock_config = MockOpenAIConfig()


def _generate_deterministic_xml(prompt: str) -> str:
    """
    Generate deterministic XML based on prompt content hash.

    Same prompt always produces same XML output, making tests reproducible.
    The XML content varies based on what the prompt is asking for.
    """
    # Use first 8 chars of hash for deterministic selection
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:8]
    hash_int = int(prompt_hash, 16)

    # Detect what kind of content the prompt is asking for
    prompt_lower = prompt.lower()

    if "test_response_ok" in prompt_lower or "respond with exactly" in prompt_lower:
        # Test asking for specific response
        return "TEST_RESPONSE_OK"

    if "hello" in prompt_lower and len(prompt) < 100:
        # Simple greeting test
        return "hello"

    if "cost test" in prompt_lower:
        # Cost tracking test
        return "cost test"

    if "html" in prompt_lower and "xml" in prompt_lower:
        # HTML to XML conversion request
        templates = [
            _XML_TEMPLATE_MODULE,
            _XML_TEMPLATE_CLASS,
            _XML_TEMPLATE_FUNCTION,
            _XML_TEMPLATE_VARIABLE,
        ]
        # Select template based on prompt hash for determinism
        template_idx = hash_int % len(templates)
        return templates[template_idx]

    # Default: return a realistic XML structure based on prompt size
    if len(prompt) > 200000:
        return _XML_TEMPLATE_MODULE
    elif len(prompt) > 50000:
        return _XML_TEMPLATE_CLASS
    elif len(prompt) > 10000:
        return _XML_TEMPLATE_FUNCTION
    else:
        return _XML_TEMPLATE_VARIABLE


# XML templates for deterministic responses
_XML_TEMPLATE_MODULE = """<MODULE>
<NAME>textual.app</NAME>
<CLASS>
<NAME>App</NAME>
<CLASS_DESCRIPTION>Bases: Generic[ReturnType], DOMNode. The base class for Textual Applications.</CLASS_DESCRIPTION>
<CLASS_API>
<METHOD>
<NAME>action_add_class</NAME>
<SIGNATURE>async action_add_class(selector, class_name)</SIGNATURE>
<RETURN_TYPE>None</RETURN_TYPE>
<DESCRIPTION>Add a CSS class to selected widgets.</DESCRIPTION>
<PARAMETERS>
<PARAMETER>
<NAME>selector</NAME>
<TYPE>str</TYPE>
<DESCRIPTION>CSS selector to target widgets.</DESCRIPTION>
<DEFAULT>required</DEFAULT>
</PARAMETER>
<PARAMETER>
<NAME>class_name</NAME>
<TYPE>str</TYPE>
<DESCRIPTION>The class name to add.</DESCRIPTION>
<DEFAULT>required</DEFAULT>
</PARAMETER>
</PARAMETERS>
</METHOD>
</CLASS_API>
</CLASS>
</MODULE>"""

_XML_TEMPLATE_CLASS = """<CLASS>
<NAME>Widget</NAME>
<CLASS_DESCRIPTION>A visual element in the TUI application.</CLASS_DESCRIPTION>
<CLASS_API>
<METHOD>
<NAME>refresh</NAME>
<SIGNATURE>def refresh(self) -> None</SIGNATURE>
<RETURN_TYPE>None</RETURN_TYPE>
<DESCRIPTION>Request a refresh of the widget.</DESCRIPTION>
</METHOD>
</CLASS_API>
</CLASS>"""

_XML_TEMPLATE_FUNCTION = """<FUNCTION>
<NAME>add</NAME>
<SIGNATURE>def add(a: int, b: int) -> int</SIGNATURE>
<RETURN_TYPE>int</RETURN_TYPE>
<DESCRIPTION>Add two numbers together.</DESCRIPTION>
<PARAMETERS>
<PARAMETER>
<NAME>a</NAME>
<TYPE>int</TYPE>
<DESCRIPTION>First number to add.</DESCRIPTION>
</PARAMETER>
<PARAMETER>
<NAME>b</NAME>
<TYPE>int</TYPE>
<DESCRIPTION>Second number to add.</DESCRIPTION>
</PARAMETER>
</PARAMETERS>
</FUNCTION>"""

_XML_TEMPLATE_VARIABLE = """<VARIABLE modifiers="module-attribute" name="DEFAULT_TIMEOUT">
<SIGNATURE>DEFAULT_TIMEOUT = 30</SIGNATURE>
<DESCRIPTION>Default timeout in seconds for API requests.</DESCRIPTION>
</VARIABLE>"""


async def mock_make_openai_request(
    api_key: str,
    prompt: str,
    pricing_info: Dict[str, Dict[str, float]],
    model: str = "gpt-5-nano-2025-08-07",
) -> Dict[str, Any]:
    """
    Mock replacement for make_openai_request() that returns identical structure.

    This function is a drop-in replacement that can be used to patch
    make_openai_request in tests, allowing "real_api" tests to run
    without actual API calls.

    Args:
        api_key: API key (ignored in mock)
        prompt: The prompt being sent
        pricing_info: Pricing info for cost calculation
        model: Model name (used for pricing lookup)

    Returns:
        Dict matching exact structure of real make_openai_request():
        {
            "id": str,
            "object": str,
            "created": int,
            "model": str,
            "choices": [{"index": int, "message": {...}, "finish_reason": str}],
            "usage": {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int},
            "request_cost": float,
            "finish_reason": str
        }

    Raises:
        Various exceptions based on configured error scenario
    """
    global _mock_config

    # Simulate delay if configured
    if _mock_config.delay_seconds > 0:
        await asyncio.sleep(_mock_config.delay_seconds)

    # Handle error scenarios
    if _mock_config.error_scenario == MockErrorScenario.RATE_LIMIT:
        # Import here to avoid circular imports
        from openai import RateLimitError

        raise RateLimitError(
            message="Rate limit exceeded",
            response=type("Response", (), {"status_code": 429, "headers": {}})(),
            body={
                "error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}
            },
        )

    if _mock_config.error_scenario == MockErrorScenario.TIMEOUT:
        from openai import APITimeoutError

        raise APITimeoutError(request=None)

    if _mock_config.error_scenario == MockErrorScenario.CONNECTION_ERROR:
        from openai import APIConnectionError

        raise APIConnectionError(message="Connection failed", request=None)

    if _mock_config.error_scenario == MockErrorScenario.QUOTA_EXCEEDED:
        from openai import RateLimitError

        raise RateLimitError(
            message="You exceeded your current quota",
            response=type("Response", (), {"status_code": 429, "headers": {}})(),
            body={
                "error": {
                    "message": "You exceeded your current quota",
                    "type": "insufficient_quota",
                }
            },
        )

    # Generate response content
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:8]

    # Check for custom response first
    if prompt_hash in _mock_config.custom_responses:
        xml_content = _mock_config.custom_responses[prompt_hash]
    else:
        xml_content = _generate_deterministic_xml(prompt)

    # Handle malformed response scenarios
    if _mock_config.error_scenario == MockErrorScenario.MALFORMED_JSON:
        # Return response with invalid JSON in content field
        content = "{{not valid json"
    elif _mock_config.error_scenario == MockErrorScenario.INVALID_XML:
        # Return response with malformed XML
        content = json.dumps(
            {
                "xml_content": "<invalid><unclosed>",
                "document_type": "MODULE",
                "completeness_check": True,
            }
        )
    elif _mock_config.error_scenario == MockErrorScenario.EMPTY_RESPONSE:
        content = json.dumps(
            {
                "xml_content": "",
                "document_type": "MODULE",
                "completeness_check": False,
            }
        )
    else:
        # Normal response
        content = json.dumps(
            {
                "xml_content": xml_content,
                "document_type": "MODULE",
                "completeness_check": True,
            }
        )

    # Calculate realistic token counts and costs
    prompt_tokens = len(prompt) // 4  # Rough: 4 chars per token
    completion_tokens = len(content) // 4

    model_pricing = pricing_info.get(model, pricing_info.get(_mock_config.model, {}))
    input_cost = prompt_tokens * model_pricing.get("input_cost_per_token", 0.00001)
    output_cost = completion_tokens * model_pricing.get(
        "output_cost_per_token", 0.00003
    )
    request_cost = input_cost + output_cost

    # Return exact structure matching make_openai_request()
    return {
        "id": f"chatcmpl-mock-{prompt_hash}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "request_cost": request_cost,
        "finish_reason": "stop",
    }
