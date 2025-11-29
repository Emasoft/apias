"""
Pytest configuration for APIAS tests.

This conftest.py provides:
1. Special configuration for testing logger_interceptor.py (monkey-patches logging)
2. Auto-mocking of OpenAI API when no API key is available
3. Command-line option --real-api to force real API usage
"""

import logging
import os
from collections.abc import Callable, Generator
from typing import Any
from unittest.mock import patch

import pytest
from _pytest.config import Config
from _pytest.config.argparsing import Parser
from _pytest.fixtures import FixtureRequest

# Save ORIGINAL logging.Logger.addHandler at conftest import time
_ORIGINAL_ADD_HANDLER = logging.Logger.addHandler


def pytest_configure(config: Config) -> None:
    """
    Disable pytest's logging plugin entirely for logger_interceptor tests.

    The logger_interceptor.py module monkey-patches logging.Logger.addHandler,
    which breaks pytest's logging plugin during teardown. We completely disable
    the logging plugin to prevent it from trying to use logging during cleanup.
    """
    # Disable ALL pytest logging to prevent it from using logging.Logger.addHandler
    # during session finish, which would crash if logging is monkey-patched
    config.option.log_level = None
    config.option.log_file = None
    config.option.log_cli = False
    config.option.log_cli_level = None
    config.option.log_file_level = None

    # Try to unregister the logging plugin entirely
    if config.pluginmanager.hasplugin("logging-plugin"):
        config.pluginmanager.unregister(name="logging-plugin")


@pytest.hookimpl(tryfirst=True, hookwrapper=True)  # type: ignore[misc]
def pytest_sessionfinish(
    session: pytest.Session,  # noqa: ARG001
    exitstatus: int,  # noqa: ARG001
) -> Generator[None, None, None]:
    """
    Force restore logging.Logger.addHandler BEFORE pytest's logging cleanup.

    This hook uses hookwrapper=True to run code both before and after
    other pytest_sessionfinish implementations. The tryfirst=True ensures
    we run before pytest's logging plugin.
    """
    # BEFORE other session finish hooks
    # Monkey-patch restoration - intentionally bypassing type system
    current_handler = logging.Logger.addHandler
    if current_handler != _ORIGINAL_ADD_HANDLER:
        logging.Logger.addHandler = _ORIGINAL_ADD_HANDLER  # type: ignore[method-assign]

    # Let other hooks run
    yield

    # AFTER other session finish hooks
    # Restore again just in case
    current_handler = logging.Logger.addHandler
    if current_handler != _ORIGINAL_ADD_HANDLER:
        logging.Logger.addHandler = _ORIGINAL_ADD_HANDLER  # type: ignore[method-assign]


@pytest.fixture  # type: ignore[misc]
def sample_api_doc() -> str:
    """Return sample API documentation for testing."""
    return """
    # Sample API Documentation
    ## Endpoint: /api/v1/users
    GET /api/v1/users
    Returns a list of users
    """


@pytest.fixture  # type: ignore[misc]
def sample_config() -> dict[str, str]:
    """Return sample configuration for testing."""
    return {"base_url": "http://example.com", "output_format": "markdown"}


# ============================================================================
# OpenAI API Mock Configuration
# ============================================================================


def pytest_addoption(parser: Parser) -> None:
    """Add --real-api command line option."""
    parser.addoption(
        "--real-api",
        action="store_true",
        default=False,
        help="Use real OpenAI API instead of mock (requires OPENAI_API_KEY)",
    )


def _has_openai_api_key() -> bool:
    """Check if a valid OpenAI API key is available."""
    key = os.environ.get("OPENAI_API_KEY", "")
    # Key must exist and not be a placeholder
    return bool(key) and not key.startswith("sk-test") and len(key) > 20


@pytest.fixture(autouse=True)  # type: ignore[misc]
def auto_mock_openai(request: FixtureRequest) -> Generator[None, None, None]:
    """
    Automatically mock make_openai_request when no API key is available.

    This fixture runs for every test. It will:
    1. If --real-api is passed and API key exists: use real API
    2. If --real-api is passed but no API key: skip real_api tests
    3. If no --real-api and test is marked real_api: use mock
    4. Otherwise: don't patch (test uses its own mocking)

    This allows "real_api" tests to run with the mock when no API key
    is available, achieving 100% test pass rate in CI environments.
    """
    # Import here to avoid circular imports at module load time
    from apias.mock_api import mock_make_openai_request, reset_mock_openai

    # Get test markers
    markers = [m.name for m in request.node.iter_markers()]
    is_real_api_test = "real_api" in markers
    use_real_api: bool = request.config.getoption("--real-api", default=False)
    has_key = _has_openai_api_key()

    # Decide whether to mock
    should_mock = False

    if is_real_api_test:
        if use_real_api:
            if not has_key:
                # User wants real API but no key - skip test
                pytest.skip("OPENAI_API_KEY not set (use mock or set key)")
            # User wants real API and has key - don't mock
            should_mock = False
        else:
            # Real API test but --real-api not passed - use mock
            should_mock = True
    # For non-real_api tests (including mock_api), don't auto-patch

    if should_mock:
        # Reset mock config to defaults for clean state
        reset_mock_openai()

        # Patch make_openai_request with mock version
        with patch(
            "apias.apias.make_openai_request",
            new=mock_make_openai_request,
        ):
            yield
    else:
        yield


@pytest.fixture  # type: ignore[misc]
def mock_openai_config() -> Generator[Callable[..., None], None, None]:
    """
    Fixture to configure mock OpenAI behavior for specific tests.

    Usage:
        def test_rate_limit_handling(mock_openai_config):
            from apias.mock_api import configure_mock_openai, MockOpenAIConfig, MockErrorScenario
            configure_mock_openai(MockOpenAIConfig(
                error_scenario=MockErrorScenario.RATE_LIMIT
            ))
            # Test code here...

    Returns:
        Function to configure mock with cleanup
    """
    from apias.mock_api import (
        MockOpenAIConfig,
        configure_mock_openai,
        reset_mock_openai,
    )

    def _configure(config: MockOpenAIConfig) -> None:
        configure_mock_openai(config)

    yield _configure

    # Cleanup: reset to defaults after test
    reset_mock_openai()
