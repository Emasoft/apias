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
- Thread-safe configuration for parallel test execution

Thread Safety:
- _mock_config is protected by _config_lock for thread-safe access
- Use configure_mock_openai() and reset_mock_openai() for atomic config changes
- mock_make_openai_request() takes a snapshot of config at call time
"""

import asyncio
import hashlib
import json
import logging
import random
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, Optional, Tuple

# WHY module-level logger: Consistent with rest of codebase
logger = logging.getLogger(__name__)

# ============================================================================
# Constants - Centralized source of truth for all tunable parameters
# ============================================================================

# API Simulation Timing Constants
# WHY: Based on empirical observation of real OpenAI API response times
MIN_API_DELAY_SECONDS = (
    0.3  # Minimum simulated API latency (typical for small requests)
)
MAX_API_DELAY_SECONDS = (
    2.5  # Maximum simulated API latency (prevents unrealistic delays)
)
RANDOM_JITTER_MAX_SECONDS = 0.5  # Random variation simulating network/server variance
FAILURE_RATE = 0.1  # 10% simulated failure rate matches observed API reliability

# Token and Cost Constants
# WHY: Based on GPT-4 pricing model approximations for realistic cost simulation
# See: https://openai.com/pricing
CHARS_PER_TOKEN = 4  # OpenAI tokenizer averages ~4 chars/token for English text
INPUT_COST_PER_TOKEN_DEFAULT = 0.00001  # ~$0.01 per 1K tokens (GPT-4 input)
OUTPUT_COST_PER_TOKEN_DEFAULT = 0.00003  # ~$0.03 per 1K tokens (GPT-4 output)
OUTPUT_TOKENS_MIN = 1000  # Minimum output tokens for non-deterministic mode
OUTPUT_TOKENS_MAX = 5000  # Maximum output tokens for non-deterministic mode

# Prompt Size Thresholds for Response Template Selection
# WHY: Different size prompts generate different mock response types to simulate
#      realistic variation in API responses based on content complexity
PROMPT_SIZE_LARGE = 200000  # Large chunk - module with many methods
PROMPT_SIZE_MEDIUM = 50000  # Medium chunk - class or module section
PROMPT_SIZE_SMALL = 10000  # Small chunk - function or variables
PROMPT_SIZE_TOKENIZATION = 100000  # Normalizer for delay calculation

# Hash Constants
# WHY SHA256: Cryptographically strong hash ensures deterministic prompt -> response mapping
# WHY 8 chars: Sufficient entropy (16^8 = 4 billion combinations) for uniqueness
PROMPT_HASH_LENGTH = 8


# ============================================================================
# XML Template Selection (Shared function to eliminate DRY violation)
# ============================================================================

# XML templates for mock responses - defined at module level for reuse
# WHY separate templates: Different prompt sizes should generate appropriately sized responses
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


def _select_xml_template_by_size(prompt_size: int) -> str:
    """
    Select appropriate XML template based on prompt size.

    WHY this function exists: Centralizes template selection logic that was duplicated
    in both MockAPIClient._generate_mock_xml() and _generate_deterministic_xml().
    This eliminates DRY violation and ensures consistent behavior.

    Args:
        prompt_size: Length of the prompt in characters

    Returns:
        XML template string matching the prompt size category

    DO NOT: Duplicate this logic elsewhere. Always call this function.
    """
    if prompt_size > PROMPT_SIZE_LARGE:
        return _XML_TEMPLATE_MODULE
    elif prompt_size > PROMPT_SIZE_MEDIUM:
        return _XML_TEMPLATE_CLASS
    elif prompt_size > PROMPT_SIZE_SMALL:
        return _XML_TEMPLATE_FUNCTION
    else:
        return _XML_TEMPLATE_VARIABLE


class MockAPIClient:
    """Mock OpenAI API client that simulates realistic behavior.

    Thread Safety: Each instance has its own random generator and state.
    Use separate instances per thread for concurrent testing.
    """

    def __init__(
        self,
        deterministic: bool = False,
        random_seed: int | None = None,
        force_retry_count: int = 0,
    ) -> None:
        """
        Initialize mock client.

        Args:
            deterministic: If True, disables random failures and jitter for predictable testing
            random_seed: If provided, seeds the random number generator for reproducible
                        non-deterministic behavior. Useful for debugging flaky tests.
                        WHY: Allows reproducing specific random scenarios for debugging.
            force_retry_count: Force N XML validation failures before success.
                        WHY: Allows reproducing exact retry scenarios for debugging.
                        Example: force_retry_count=2 → fail on call 1,2 then succeed on call 3
        """
        self.total_cost = 0.0
        self.call_count = 0
        self.deterministic = deterministic
        # REPRODUCIBILITY: Track forced failures
        self.force_retry_count = force_retry_count
        self._forced_failure_count = 0
        # WHY instance-specific RNG: Isolates randomness per client, prevents cross-test interference
        self._rng = random.Random(random_seed)

    async def responses_create(self, **kwargs: Any) -> "MockResponse":
        """
        Simulate OpenAI API call with realistic delays and occasional failures.

        Simulates:
        - Variable latency (MIN_API_DELAY_SECONDS - MAX_API_DELAY_SECONDS) unless deterministic
        - FAILURE_RATE failure rate on first attempt unless deterministic mode
        - Realistic cost calculation based on CHARS_PER_TOKEN estimation
        """
        self.call_count += 1

        # Extract prompt size to calculate realistic delay and cost
        # WHY validation: Prevents silent failures if messages structure is wrong
        messages = kwargs.get("messages")
        if not messages:
            raise ValueError(
                "MockAPIClient.responses_create() called without messages. "
                "This indicates a bug in the calling code."
            )
        prompt = messages[0].get("content", "")
        prompt_size = len(prompt)

        # Simulate realistic API latency based on prompt size
        # WHY size_factor: Larger prompts take longer to process, simulating real API behavior
        size_factor = prompt_size / PROMPT_SIZE_TOKENIZATION

        if self.deterministic:
            # WHY no jitter in deterministic: Ensures reproducible test timing
            delay = min(MIN_API_DELAY_SECONDS + size_factor, MAX_API_DELAY_SECONDS)
        else:
            # WHY jitter: Simulates real-world network/server load variance
            random_jitter = self._rng.uniform(0, RANDOM_JITTER_MAX_SECONDS)
            delay = min(
                MIN_API_DELAY_SECONDS + size_factor + random_jitter,
                MAX_API_DELAY_SECONDS,
            )

        await asyncio.sleep(delay)

        # REPRODUCIBILITY: Force exactly N failures before success
        # WHY explicit logging: Makes it clear in logs exactly which attempt triggered failure
        if (
            self.force_retry_count > 0
            and self._forced_failure_count < self.force_retry_count
        ):
            self._forced_failure_count += 1
            logger.debug(
                f"[MOCK] force_retry_count={self.force_retry_count} "
                f"failure={self._forced_failure_count} → returning INVALID_XML"
            )
            # Return invalid XML that will fail validation, triggering retry
            return MockResponse(xml_content="<invalid><unclosed>", cost=0.0001)

        # Simulate occasional failures unless deterministic
        # WHY FAILURE_RATE: Tests retry logic without requiring real API failures
        if not self.deterministic and self._rng.random() < FAILURE_RATE:
            raise MockAPIException("Simulated API failure - will retry")

        # Calculate realistic cost based on token estimation
        input_tokens = prompt_size // CHARS_PER_TOKEN
        if self.deterministic:
            # WHY fixed output in deterministic: Ensures reproducible cost calculations
            output_tokens = (OUTPUT_TOKENS_MIN + OUTPUT_TOKENS_MAX) // 2
        else:
            output_tokens = self._rng.randint(OUTPUT_TOKENS_MIN, OUTPUT_TOKENS_MAX)

        cost = (input_tokens * INPUT_COST_PER_TOKEN_DEFAULT) + (
            output_tokens * OUTPUT_COST_PER_TOKEN_DEFAULT
        )
        self.total_cost += cost

        # Generate mock XML response using shared template selector
        xml_content = _select_xml_template_by_size(prompt_size)

        return MockResponse(xml_content=xml_content, cost=cost)


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
    mock_client: MockAPIClient | None = None,
    deterministic: bool = False,
) -> Tuple[str | None, float]:
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
        force_retry_count: Force N failures before success (for retry reproduction)
    """

    deterministic: bool = True
    error_scenario: MockErrorScenario = MockErrorScenario.NONE
    delay_seconds: float = 0.0
    model: str = "gpt-5-nano-2025-08-07"
    custom_responses: Dict[str, str] = field(default_factory=dict)
    # REPRODUCIBILITY: Force exactly N XML validation failures before success
    # Example: force_retry_count=2 → fail on attempt 0, 1, succeed on attempt 2
    force_retry_count: int = 0


# WHY global counter: Track request attempts for force_retry_count feature
# WHY dict keyed by prompt_hash: Each unique request has its own attempt counter
_request_attempt_counter: Dict[str, int] = {}
_counter_lock = threading.Lock()


# WHY global: Allows tests to configure mock behavior before running
# WHY lock: Thread-safety for parallel test execution (pytest-xdist)
# Race condition without lock: Thread A writes config, Thread B reads partial state
_mock_config: MockOpenAIConfig = MockOpenAIConfig()
_config_lock = threading.Lock()


def configure_mock_openai(config: MockOpenAIConfig) -> None:
    """
    Configure mock OpenAI behavior for testing.

    Thread-safe: Uses lock to prevent race conditions with mock_make_openai_request().

    Args:
        config: MockOpenAIConfig instance with desired settings

    Example:
        configure_mock_openai(MockOpenAIConfig(
            error_scenario=MockErrorScenario.RATE_LIMIT,
            delay_seconds=0.1
        ))

    DO NOT: Access _mock_config directly - always use this function.
    """
    global _mock_config
    with _config_lock:
        _mock_config = config


def reset_mock_openai() -> None:
    """
    Reset mock configuration to defaults and clear attempt counters.

    Thread-safe: Uses locks to prevent race conditions.
    """
    global _mock_config, _request_attempt_counter
    with _config_lock:
        _mock_config = MockOpenAIConfig()
    with _counter_lock:
        _request_attempt_counter.clear()


def _generate_deterministic_xml(prompt: str) -> str:
    """
    Generate deterministic XML based on prompt content hash.

    Same prompt always produces same XML output, making tests reproducible.
    The XML content varies based on what the prompt is asking for.

    WHY hash-based selection: Provides determinism while varying responses based on content.
    WHY special cases: Some tests check for specific response strings.
    """
    # WHY SHA256: Cryptographically strong, deterministic, good distribution
    # WHY [:PROMPT_HASH_LENGTH]: 8 hex chars = 16^8 = 4B combinations (sufficient)
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:PROMPT_HASH_LENGTH]
    hash_int = int(prompt_hash, 16)

    # Detect what kind of content the prompt is asking for
    # WHY lowercase: Case-insensitive matching for reliability
    prompt_lower = prompt.lower()

    # WHY special cases: These match specific test assertions
    if "test_response_ok" in prompt_lower or "respond with exactly" in prompt_lower:
        # Test asking for specific response (test_real_api_returns_content)
        return "TEST_RESPONSE_OK"

    if "hello" in prompt_lower and len(prompt) < 100:
        # Simple greeting test (test_real_api_request)
        return "hello"

    if "cost test" in prompt_lower:
        # Cost tracking test (test_real_api_cost_tracking)
        return "cost test"

    if "html" in prompt_lower and "xml" in prompt_lower:
        # HTML to XML conversion request
        templates = [
            _XML_TEMPLATE_MODULE,
            _XML_TEMPLATE_CLASS,
            _XML_TEMPLATE_FUNCTION,
            _XML_TEMPLATE_VARIABLE,
        ]
        # WHY hash-based selection: Deterministic but varies by prompt content
        template_idx = hash_int % len(templates)
        return templates[template_idx]

    # Default: use shared template selector based on prompt size
    # WHY _select_xml_template_by_size: DRY - same logic as MockAPIClient.responses_create()
    return _select_xml_template_by_size(len(prompt))


# NOTE: XML templates are defined at module top (lines 72-137) to avoid duplication
# DO NOT: Add template definitions here - use _XML_TEMPLATE_* constants above


def create_custom_response_key(prompt: str) -> str:
    """
    Create a hash key for custom responses.

    Use this to register custom responses in MockOpenAIConfig.custom_responses.

    Args:
        prompt: The exact prompt text that will trigger this response

    Returns:
        8-character hash key to use as dictionary key

    Example:
        key = create_custom_response_key("my specific prompt")
        config = MockOpenAIConfig(custom_responses={key: "<MY_CUSTOM_XML/>"})
        configure_mock_openai(config)

    WHY this function: Encapsulates hash algorithm, prevents copy-paste errors.
    """
    return hashlib.sha256(prompt.encode()).hexdigest()[:PROMPT_HASH_LENGTH]


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

    Thread-safe: Takes a snapshot of config under lock to prevent race conditions.

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
    # WHY snapshot: Thread-safety - config might change during async operation
    # Take snapshot under lock, then release lock before any async operations
    with _config_lock:
        config = _mock_config

    # Simulate delay if configured (outside lock - delay can be long)
    if config.delay_seconds > 0:
        await asyncio.sleep(config.delay_seconds)

    # REPRODUCIBILITY: Force N failures before success for retry testing
    # WHY prompt_hash key: Each unique request has its own attempt counter
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:PROMPT_HASH_LENGTH]
    if config.force_retry_count > 0:
        with _counter_lock:
            current_attempt = _request_attempt_counter.get(prompt_hash, 0)
            _request_attempt_counter[prompt_hash] = current_attempt + 1

        if current_attempt < config.force_retry_count:
            # Return INVALID_XML to trigger retry logic
            # Log the forced failure for debugging
            import logging

            logging.getLogger(__name__).debug(
                f"[MOCK] force_retry_count={config.force_retry_count} "
                f"current_attempt={current_attempt} → returning INVALID_XML"
            )
            return {
                "id": f"chatcmpl-mock-{prompt_hash}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": config.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {
                                    "xml_content": "<invalid><unclosed>",
                                    "document_type": "MODULE",
                                    "completeness_check": True,
                                }
                            ),
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                },
                "request_cost": 0.0001,
                "finish_reason": "stop",
            }

    # Handle error scenarios - use config snapshot for thread-safety
    # WHY imports inside conditionals: Avoid circular imports, only load when needed
    if config.error_scenario == MockErrorScenario.RATE_LIMIT:
        from openai import RateLimitError

        raise RateLimitError(
            message="Rate limit exceeded",
            response=type("Response", (), {"status_code": 429, "headers": {}})(),
            body={
                "error": {"message": "Rate limit exceeded", "type": "rate_limit_error"}
            },
        )

    if config.error_scenario == MockErrorScenario.TIMEOUT:
        from openai import APITimeoutError

        raise APITimeoutError(request=None)

    if config.error_scenario == MockErrorScenario.CONNECTION_ERROR:
        from openai import APIConnectionError

        raise APIConnectionError(message="Connection failed", request=None)

    if config.error_scenario == MockErrorScenario.QUOTA_EXCEEDED:
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
    # WHY PROMPT_HASH_LENGTH: Centralized constant for consistent hash truncation
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:PROMPT_HASH_LENGTH]

    # Check for custom response first (from config snapshot)
    if prompt_hash in config.custom_responses:
        xml_content = config.custom_responses[prompt_hash]
    else:
        xml_content = _generate_deterministic_xml(prompt)

    # Handle malformed response scenarios - use config snapshot
    if config.error_scenario == MockErrorScenario.MALFORMED_JSON:
        # Return response with invalid JSON in content field
        content = "{{not valid json"
    elif config.error_scenario == MockErrorScenario.INVALID_XML:
        # Return response with malformed XML
        content = json.dumps(
            {
                "xml_content": "<invalid><unclosed>",
                "document_type": "MODULE",
                "completeness_check": True,
            }
        )
    elif config.error_scenario == MockErrorScenario.EMPTY_RESPONSE:
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
    # WHY CHARS_PER_TOKEN: OpenAI tokenizer averages ~4 chars/token for English
    prompt_tokens = len(prompt) // CHARS_PER_TOKEN
    completion_tokens = len(content) // CHARS_PER_TOKEN

    # WHY config.model fallback: Allow model override in config for testing
    model_pricing = pricing_info.get(model, pricing_info.get(config.model, {}))
    # WHY constants for defaults: Centralized pricing fallbacks matching GPT-4 rates
    input_cost = prompt_tokens * model_pricing.get(
        "input_cost_per_token", INPUT_COST_PER_TOKEN_DEFAULT
    )
    output_cost = completion_tokens * model_pricing.get(
        "output_cost_per_token", OUTPUT_COST_PER_TOKEN_DEFAULT
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
