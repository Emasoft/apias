"""
Comprehensive unit tests for apias/logger_interceptor.py

Tests cover LoggerInterceptor core functionality including install/uninstall,
StreamHandler blocking, blocked attempt logging, context manager usage,
and utility functions with realistic, non-mocked test scenarios.

All tests use real logging.Logger instances, real handlers, and real files.
No mocks for Logger or Handler classes - only real functional tests.
"""

import logging
import sys
import tempfile
from io import StringIO
from pathlib import Path
from typing import List

import pytest

from apias.logger_interceptor import (
    LoggerInterceptor,
    create_interceptor_with_session_log,
)


# ============================================================================
# NOTE: Pytest hooks for logging restoration are in conftest.py
# ============================================================================


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def cleanup_interceptor():
    """
    Fixture to ensure interceptor is always uninstalled after each test.

    This prevents test pollution where one test's monkey-patch affects
    subsequent tests. Yields a list that tests can append interceptor
    instances to for automatic cleanup.
    """
    interceptors: List[LoggerInterceptor] = []

    def register(interceptor: LoggerInterceptor) -> LoggerInterceptor:
        """Register interceptor for cleanup."""
        interceptors.append(interceptor)
        return interceptor

    yield register

    # Cleanup: uninstall all registered interceptors BEFORE pytest tries to modify logging
    for interceptor in interceptors:
        if interceptor._installed:
            interceptor.uninstall()


@pytest.fixture(scope="function", autouse=True)
def ensure_logging_restored_after_each_test():
    """
    Function-scoped autouse fixture to ensure logging is restored after EACH test.

    This runs after every test (even after cleanup_interceptor) to guarantee
    that logging.Logger.addHandler is always restored to its original state
    before the next test or pytest's own logging operations.
    """
    # Save original addHandler before test
    original_addHandler = logging.Logger.addHandler

    yield

    # Force restore after test (runs even if cleanup_interceptor already ran)
    # This is a safety net to prevent ANY test from leaving logging patched
    if logging.Logger.addHandler != original_addHandler:
        logging.Logger.addHandler = original_addHandler


@pytest.fixture
def temp_log_file(tmp_path):
    """Create a temporary log file for session logging."""
    log_file = tmp_path / "session.log"
    return log_file


# ============================================================================
# Install/Uninstall (3 tests)
# ============================================================================


def test_install_monkey_patches_logger_addhandler(cleanup_interceptor):
    """Test install() monkey-patches Logger.addHandler (verify original method saved, new method installed)."""
    # Save the original addHandler method before any modifications
    original_method = logging.Logger.addHandler

    # Create interceptor
    interceptor = cleanup_interceptor(LoggerInterceptor())

    # Verify not installed initially
    assert not interceptor._installed, "Interceptor should not be installed initially"
    assert interceptor._original_addHandler is None, "Original method should not be saved initially"

    # Install interceptor
    interceptor.install()

    # Verify installation
    assert interceptor._installed, "Interceptor should be marked as installed"
    assert interceptor._original_addHandler is not None, "Original method should be saved"
    assert interceptor._original_addHandler == original_method, "Saved method should match original"
    assert logging.Logger.addHandler != original_method, "Logger.addHandler should be replaced with wrapper function"
    # Note: Logger.addHandler is now an unbound wrapper function, not _intercepted_addHandler
    # The wrapper delegates to _intercepted_addHandler, but they are not the same object
    assert callable(logging.Logger.addHandler), "Logger.addHandler should be callable (wrapper function)"


def test_uninstall_restores_original_method(cleanup_interceptor):
    """Test uninstall() restores original method (verify original method restored, stats logged)."""
    # Save the original addHandler method
    original_method = logging.Logger.addHandler

    # Create and install interceptor
    interceptor = cleanup_interceptor(LoggerInterceptor())
    interceptor.install()

    # Verify installed
    assert interceptor._installed, "Interceptor should be installed"
    assert logging.Logger.addHandler != original_method, "Logger.addHandler should be monkey-patched"

    # Block a handler to test stats logging
    test_logger = logging.getLogger("test_uninstall")
    test_logger.addHandler(logging.StreamHandler(sys.stdout))

    # Uninstall interceptor
    interceptor.uninstall()

    # Verify restoration
    assert not interceptor._installed, "Interceptor should be marked as not installed"
    assert interceptor._original_addHandler is None, "Original method reference should be cleared"
    assert logging.Logger.addHandler == original_method, "Logger.addHandler should be restored to original"
    assert interceptor._blocked_attempts == 1, "Should have logged 1 blocked attempt"


def test_install_idempotency(cleanup_interceptor):
    """Test install() idempotency (calling install() twice has no effect)."""
    interceptor = cleanup_interceptor(LoggerInterceptor())

    # First install
    interceptor.install()
    assert interceptor._installed, "Should be installed after first call"
    saved_method = interceptor._original_addHandler
    first_wrapper = logging.Logger.addHandler

    # Second install (should be no-op)
    interceptor.install()
    assert interceptor._installed, "Should still be installed"
    assert interceptor._original_addHandler == saved_method, "Original method should not change"

    # Verify only one level of monkey-patching (not nested)
    # The wrapper function should be the same object (no new wrapper created)
    assert logging.Logger.addHandler is first_wrapper, "Should still point to the same wrapper function (no nested wrapping)"


# ============================================================================
# StreamHandler Blocking (4 tests)
# ============================================================================


def test_blocks_streamhandler_to_stdout(cleanup_interceptor):
    """Test blocks StreamHandler to stdout (create logger, add StreamHandler(sys.stdout), verify NOT added)."""
    interceptor = cleanup_interceptor(LoggerInterceptor())
    interceptor.install()

    # Create test logger
    test_logger = logging.getLogger("test_stdout_blocking")
    initial_handler_count = len(test_logger.handlers)

    # Try to add StreamHandler to stdout
    stdout_handler = logging.StreamHandler(sys.stdout)
    test_logger.addHandler(stdout_handler)

    # Verify handler was NOT added
    assert len(test_logger.handlers) == initial_handler_count, "StreamHandler to stdout should be blocked"
    assert stdout_handler not in test_logger.handlers, "stdout handler should not be in handlers list"
    assert interceptor._blocked_attempts == 1, "Should have blocked 1 attempt"


def test_blocks_streamhandler_to_stderr(cleanup_interceptor):
    """Test blocks StreamHandler to stderr (create logger, add StreamHandler(sys.stderr), verify NOT added)."""
    interceptor = cleanup_interceptor(LoggerInterceptor())
    interceptor.install()

    # Create test logger
    test_logger = logging.getLogger("test_stderr_blocking")
    initial_handler_count = len(test_logger.handlers)

    # Try to add StreamHandler to stderr
    stderr_handler = logging.StreamHandler(sys.stderr)
    test_logger.addHandler(stderr_handler)

    # Verify handler was NOT added
    assert len(test_logger.handlers) == initial_handler_count, "StreamHandler to stderr should be blocked"
    assert stderr_handler not in test_logger.handlers, "stderr handler should not be in handlers list"
    assert interceptor._blocked_attempts == 1, "Should have blocked 1 attempt"


def test_allows_filehandler(cleanup_interceptor, temp_log_file):
    """Test allows FileHandler (create logger, add FileHandler, verify added)."""
    interceptor = cleanup_interceptor(LoggerInterceptor())
    interceptor.install()

    # Create test logger
    test_logger = logging.getLogger("test_filehandler_allowed")
    initial_handler_count = len(test_logger.handlers)

    # Add FileHandler (should be allowed)
    file_handler = logging.FileHandler(temp_log_file)
    test_logger.addHandler(file_handler)

    # Verify handler WAS added
    assert len(test_logger.handlers) == initial_handler_count + 1, "FileHandler should be allowed"
    assert file_handler in test_logger.handlers, "FileHandler should be in handlers list"
    assert interceptor._blocked_attempts == 0, "Should have blocked 0 attempts"

    # Cleanup: remove handler
    test_logger.removeHandler(file_handler)
    file_handler.close()


def test_allows_streamhandler_to_custom_stream(cleanup_interceptor):
    """Test allows StreamHandler to custom stream (create StringIO, add StreamHandler(StringIO), verify added)."""
    interceptor = cleanup_interceptor(LoggerInterceptor())
    interceptor.install()

    # Create test logger
    test_logger = logging.getLogger("test_custom_stream_allowed")
    initial_handler_count = len(test_logger.handlers)

    # Create custom stream and handler
    custom_stream = StringIO()
    custom_handler = logging.StreamHandler(custom_stream)
    test_logger.addHandler(custom_handler)

    # Verify handler WAS added (custom streams are allowed)
    assert len(test_logger.handlers) == initial_handler_count + 1, "StreamHandler to custom stream should be allowed"
    assert custom_handler in test_logger.handlers, "Custom stream handler should be in handlers list"
    assert interceptor._blocked_attempts == 0, "Should have blocked 0 attempts"

    # Cleanup: remove handler
    test_logger.removeHandler(custom_handler)


# ============================================================================
# Blocked Attempt Logging (2 tests)
# ============================================================================


def test_logs_blocked_attempts_to_session_log(cleanup_interceptor, temp_log_file):
    """Test logs blocked attempts to session log (block 3 handlers, verify 3 log entries)."""
    # Create session log handler
    session_handler = logging.FileHandler(temp_log_file, mode="w", encoding="utf-8")
    session_handler.setLevel(logging.DEBUG)
    session_handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    # Create interceptor with session log
    interceptor = cleanup_interceptor(LoggerInterceptor(session_log_handler=session_handler))
    interceptor.install()

    # Block 3 handlers from different loggers
    logger1 = logging.getLogger("test_logger_1")
    logger2 = logging.getLogger("test_logger_2")
    logger3 = logging.getLogger("test_logger_3")

    logger1.addHandler(logging.StreamHandler(sys.stdout))
    logger2.addHandler(logging.StreamHandler(sys.stderr))
    logger3.addHandler(logging.StreamHandler(sys.stdout))

    # Flush session handler
    session_handler.flush()

    # Read session log
    log_content = temp_log_file.read_text()

    # Verify 3 blocked attempts logged
    blocked_lines = [line for line in log_content.split("\n") if "🚫 Blocked StreamHandler" in line]
    assert len(blocked_lines) == 3, f"Should have 3 blocked attempt log entries, got {len(blocked_lines)}"
    assert "test_logger_1" in log_content, "Should log logger name for logger1"
    assert "test_logger_2" in log_content, "Should log logger name for logger2"
    assert "test_logger_3" in log_content, "Should log logger name for logger3"
    assert interceptor._blocked_attempts == 3, "Should have blocked 3 attempts"

    # Cleanup
    session_handler.close()


def test_get_stats_returns_blocked_attempts_count(cleanup_interceptor):
    """Test get_stats() returns blocked_attempts count."""
    interceptor = cleanup_interceptor(LoggerInterceptor())

    # Initial stats (not installed)
    stats = interceptor.get_stats()
    assert stats["installed"] is False, "Should report not installed"
    assert stats["blocked_attempts"] == 0, "Should have 0 blocked attempts initially"

    # Install and block some handlers
    interceptor.install()
    test_logger = logging.getLogger("test_stats")
    test_logger.addHandler(logging.StreamHandler(sys.stdout))
    test_logger.addHandler(logging.StreamHandler(sys.stderr))

    # Check stats after blocking
    stats = interceptor.get_stats()
    assert stats["installed"] is True, "Should report installed"
    assert stats["blocked_attempts"] == 2, "Should have 2 blocked attempts"


# ============================================================================
# Context Manager (2 tests)
# ============================================================================


def test_context_manager_installs_and_uninstalls(cleanup_interceptor):
    """Test context manager installs and uninstalls (verify installed in __enter__, uninstalled in __exit__)."""
    # Save original method
    original_method = logging.Logger.addHandler

    # Create interceptor (not installed yet)
    interceptor = LoggerInterceptor()
    assert not interceptor._installed, "Should not be installed before context manager"

    # Use context manager
    with interceptor as ctx:
        # Verify installed inside context
        assert ctx is interceptor, "Context manager should return self"
        assert interceptor._installed, "Should be installed inside context"
        assert logging.Logger.addHandler != original_method, "Logger.addHandler should be monkey-patched"

        # Test blocking works
        test_logger = logging.getLogger("test_context_manager")
        test_logger.addHandler(logging.StreamHandler(sys.stdout))
        assert interceptor._blocked_attempts == 1, "Should block handlers inside context"

    # Verify uninstalled after context
    assert not interceptor._installed, "Should be uninstalled after context exit"
    assert logging.Logger.addHandler == original_method, "Logger.addHandler should be restored"


def test_context_manager_uninstalls_on_exception(cleanup_interceptor):
    """Test context manager uninstalls on exception (raise exception in with block, verify uninstall still called)."""
    # Save original method
    original_method = logging.Logger.addHandler

    # Create interceptor
    interceptor = LoggerInterceptor()

    # Use context manager with exception
    with pytest.raises(ValueError, match="Test exception"):
        with interceptor:
            assert interceptor._installed, "Should be installed inside context"
            # Raise exception
            raise ValueError("Test exception")

    # Verify uninstalled even after exception
    assert not interceptor._installed, "Should be uninstalled after exception"
    assert logging.Logger.addHandler == original_method, "Logger.addHandler should be restored after exception"


# ============================================================================
# Utility Functions (1 test)
# ============================================================================


def test_create_interceptor_with_session_log(cleanup_interceptor, temp_log_file):
    """Test create_interceptor_with_session_log() creates configured interceptor."""
    # Create interceptor using utility function
    interceptor = cleanup_interceptor(create_interceptor_with_session_log(str(temp_log_file)))

    # Verify interceptor created
    assert isinstance(interceptor, LoggerInterceptor), "Should return LoggerInterceptor instance"
    assert interceptor._session_log_handler is not None, "Should have session log handler configured"
    assert not interceptor._installed, "Should not be installed automatically"

    # Verify session log handler configured correctly
    session_handler = interceptor._session_log_handler
    assert isinstance(session_handler, logging.FileHandler), "Should be FileHandler"
    assert session_handler.level == logging.DEBUG, "Should be set to DEBUG level"
    assert session_handler.baseFilename == str(temp_log_file), "Should use correct log file path"

    # Install and test blocking with logging
    interceptor.install()
    test_logger = logging.getLogger("test_utility_function")
    test_logger.addHandler(logging.StreamHandler(sys.stdout))

    # Flush and verify log
    session_handler.flush()
    log_content = temp_log_file.read_text()
    assert "🚫 Blocked StreamHandler" in log_content, "Should log blocked attempt to session log"
    assert "test_utility_function" in log_content, "Should include logger name in log"

    # Cleanup
    session_handler.close()
