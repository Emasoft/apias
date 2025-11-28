"""
Pytest configuration for APIAS tests.

This conftest.py provides special configuration for testing logger_interceptor.py,
which monkey-patches logging.Logger.addHandler and can interfere with pytest's
own logging mechanisms.
"""

import logging

import pytest

# Save ORIGINAL logging.Logger.addHandler at conftest import time
_ORIGINAL_ADD_HANDLER = logging.Logger.addHandler


def pytest_configure(config):
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


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_sessionfinish(session, exitstatus):
    """
    Force restore logging.Logger.addHandler BEFORE pytest's logging cleanup.

    This hook uses hookwrapper=True to run code both before and after
    other pytest_sessionfinish implementations. The tryfirst=True ensures
    we run before pytest's logging plugin.
    """
    # BEFORE other session finish hooks
    if logging.Logger.addHandler != _ORIGINAL_ADD_HANDLER:
        logging.Logger.addHandler = _ORIGINAL_ADD_HANDLER

    # Let other hooks run
    yield

    # AFTER other session finish hooks
    # Restore again just in case
    if logging.Logger.addHandler != _ORIGINAL_ADD_HANDLER:
        logging.Logger.addHandler = _ORIGINAL_ADD_HANDLER


@pytest.fixture
def sample_api_doc():
    return """
    # Sample API Documentation
    ## Endpoint: /api/v1/users
    GET /api/v1/users
    Returns a list of users
    """


@pytest.fixture
def sample_config():
    return {"base_url": "http://example.com", "output_format": "markdown"}
