"""
Mock API module for testing TUI without spending tokens.

Simulates realistic API behavior with:
- Variable response times (0.3s - 2.5s)
- Occasional failures (~10% rate)
- Realistic XML generation
- Cost simulation
"""

import asyncio
import json
import random
from typing import Tuple, Optional, Any


class MockAPIClient:
    """Mock OpenAI API client that simulates realistic behavior"""

    def __init__(self) -> None:
        self.total_cost = 0.0
        self.call_count = 0

    async def responses_create(self, **kwargs: Any) -> "MockResponse":
        """
        Simulate OpenAI API call with realistic delays and occasional failures.

        Simulates:
        - Variable latency (300ms - 2500ms)
        - ~10% failure rate on first attempt
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
        random_jitter = random.uniform(0, 0.5)
        delay = min(base_delay + size_factor + random_jitter, 2.5)

        await asyncio.sleep(delay)

        # Simulate occasional failures (~10% on first attempt)
        if random.random() < 0.1:
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
) -> Tuple[Optional[str], float]:
    """
    Mock version of call_openai_api that simulates realistic behavior.

    Args:
        prompt: The prompt to send (used for size calculations)
        pricing_info: Pricing information (ignored in mock)
        mock_client: Optional shared mock client for cost tracking

    Returns:
        Tuple of (xml_content, cost)
    """
    if mock_client is None:
        mock_client = MockAPIClient()

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
