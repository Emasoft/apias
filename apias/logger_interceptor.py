"""
Bulletproof logger isolation via monkey-patching.

This module prevents third-party libraries from adding StreamHandlers to loggers,
ensuring that NO console output occurs during TUI operation. Libraries like urllib3,
requests, selenium, and others often add StreamHandlers despite our best efforts
to suppress them. This module uses monkey-patching to intercept ALL addHandler()
calls and block StreamHandlers globally.

Key Design Principles:
- Monkey-patch: Override logging.Logger.addHandler() at class level
- Selective blocking: Block only StreamHandlers to stdout/stderr
- Allow FileHandlers: Session log and other file logging still works
- Audit trail: Log all blocked attempts for debugging
- Clean uninstall: Restore original behavior on exit

Architecture:
    Library Code → logging.Logger.addHandler(StreamHandler) → BLOCKED
                                                                   ↓
                                                          Logged to session.log

Usage:
    # Set up session log
    session_log_handler = setup_session_log_file(session_log_path)

    # Install interceptor BEFORE library imports
    interceptor = LoggerInterceptor(session_log_handler)
    interceptor.install()

    # Suppress existing handlers
    suppress_console_logging()

    try:
        # ... TUI operation (guaranteed zero console output)
    finally:
        # Always restore original behavior
        interceptor.uninstall()
"""

import logging
import sys
import time
from typing import Optional

logger = logging.getLogger(__name__)


# ============================================================================
# Logger Interceptor
# ============================================================================


class LoggerInterceptor:
    """
    Monkey-patch logging.Logger.addHandler() to block console output.

    This class intercepts ALL attempts to add handlers to loggers and
    selectively blocks StreamHandlers that write to stdout/stderr.
    FileHandlers and other handler types are allowed through.

    Thread Safety: The monkey-patch operates at the class level, so it
    affects all Logger instances in all threads. The interceptor itself
    is not thread-safe (install/uninstall should be called from main thread).

    Design Note: We use monkey-patching because:
    - Libraries can add handlers at import time (before we can suppress)
    - Libraries can re-add handlers after we remove them
    - Libraries can create new loggers dynamically
    - Monkey-patching catches ALL addHandler calls regardless of when/where

    Example Blocked Scenarios:
    - urllib3.connectionpool adds StreamHandler on first request
    - selenium.webdriver adds StreamHandler during browser initialization
    - anthropic SDK adds StreamHandler for debug logging
    - Any library calling logger.addHandler(logging.StreamHandler())
    """

    def __init__(self, session_log_handler: Optional[logging.FileHandler] = None):
        """
        Initialize logger interceptor.

        Args:
            session_log_handler: Optional FileHandler for session.log.
                                If provided, blocked attempts will be logged here.
                                If None, blocked attempts are silently dropped.

        Design Note: We accept the session log handler as a parameter so we
        can log blocked attempts for debugging. This creates an audit trail
        of all libraries that tried to add console output.
        """
        self._session_log_handler = session_log_handler
        self._original_addHandler: Optional[callable] = None
        self._blocked_attempts = 0
        self._installed = False

        logger.debug("LoggerInterceptor initialized")

    def install(self) -> None:
        """
        Install monkey-patch to intercept Logger.addHandler().

        This replaces logging.Logger.addHandler with our interceptor method.
        The original method is saved and can be restored via uninstall().

        Thread Safety: Should only be called from main thread before
        starting worker threads.

        Idempotency: Safe to call multiple times (no-op if already installed).
        """
        if self._installed:
            logger.warning("LoggerInterceptor already installed, skipping")
            return

        # Save original method (class-level, affects all Logger instances)
        self._original_addHandler = logging.Logger.addHandler

        # Create unbound wrapper function that properly receives logger_instance
        # WHY: Bound methods don't work for monkey-patching because Python
        # doesn't know to pass the logger instance as the first argument.
        # We need an unbound function that receives logger_instance and delegates
        # to our interceptor method.
        def interceptor_wrapper(logger_instance: logging.Logger, handler: logging.Handler) -> None:
            """Unbound wrapper that delegates to _intercepted_addHandler"""
            self._intercepted_addHandler(logger_instance, handler)

        # Replace with our wrapper
        # WHY: This affects ALL Logger instances, including those created
        # by third-party libraries that we don't control
        logging.Logger.addHandler = interceptor_wrapper

        self._installed = True
        logger.info("LoggerInterceptor installed - StreamHandlers to stdout/stderr will be blocked")

    def uninstall(self) -> None:
        """
        Restore original Logger.addHandler() method.

        This removes the monkey-patch and restores normal logging behavior.
        Should be called in a finally block to ensure cleanup.

        Thread Safety: Should only be called from main thread after
        worker threads have stopped.

        Idempotency: Safe to call multiple times (no-op if not installed).
        """
        if not self._installed:
            logger.debug("LoggerInterceptor not installed, nothing to uninstall")
            return

        if not self._original_addHandler:
            logger.error("LoggerInterceptor in invalid state: installed but no original method saved")
            return

        # Restore original method
        logging.Logger.addHandler = self._original_addHandler
        self._original_addHandler = None
        self._installed = False

        logger.info(
            f"LoggerInterceptor uninstalled - blocked {self._blocked_attempts} "
            f"StreamHandler attempts during session"
        )

    def _intercepted_addHandler(self, logger_instance: logging.Logger, handler: logging.Handler) -> None:
        """
        Intercept addHandler() calls and block StreamHandlers.

        This method replaces logging.Logger.addHandler() and is called
        whenever ANY code tries to add a handler to ANY logger.

        Args:
            logger_instance: The Logger instance receiving the handler
            handler: The Handler being added

        Behavior:
        - StreamHandler to stdout/stderr → BLOCKED (logged to session.log)
        - StreamHandler to other streams → ALLOWED (custom streams OK)
        - FileHandler → ALLOWED (session.log, error.log, etc.)
        - Other handler types → ALLOWED (NullHandler, MemoryHandler, etc.)

        Design Note: We check both isinstance(StreamHandler) and stream
        identity to avoid blocking custom StreamHandlers that write to
        StringIO or other non-console streams.
        """
        # Check if this is a StreamHandler to stdout/stderr
        if isinstance(handler, logging.StreamHandler):
            # Get the stream (may be None for custom handlers)
            stream = getattr(handler, "stream", None)

            # Block only stdout/stderr (allow custom streams)
            if stream in (sys.stdout, sys.stderr):
                # Log the blocked attempt for debugging
                self._log_blocked_attempt(logger_instance.name, handler)
                self._blocked_attempts += 1
                return  # BLOCK IT - don't call original addHandler

        # Allow all other handler types (FileHandler, NullHandler, etc.)
        # and StreamHandlers to custom streams
        if self._original_addHandler:
            self._original_addHandler(logger_instance, handler)
        else:
            logger.error("LoggerInterceptor in invalid state: no original addHandler method")

    def _log_blocked_attempt(self, logger_name: str, handler: logging.Handler) -> None:
        """
        Log a blocked StreamHandler attempt to session.log.

        Args:
            logger_name: Name of the logger that tried to add the handler
            handler: The StreamHandler that was blocked

        Design Note: We use makeLogRecord() to create a LogRecord without
        going through the normal logging flow (which might trigger recursion).
        """
        if not self._session_log_handler:
            # No session log configured - silent blocking
            return

        # Get stream name (stdout or stderr)
        stream = getattr(handler, "stream", None)
        stream_name = "stdout" if stream is sys.stdout else "stderr" if stream is sys.stderr else "unknown"

        # Create log record manually to avoid recursion
        # WHY makeLogRecord: Bypasses normal logging flow which might
        # trigger more addHandler calls
        record = logging.makeLogRecord(
            {
                "name": __name__,
                "levelno": logging.DEBUG,
                "levelname": "DEBUG",
                "msg": f"🚫 Blocked StreamHandler from logger '{logger_name}' (stream={stream_name})",
                "created": time.time(),
                "filename": __file__,
                "lineno": 0,
                "funcName": "_log_blocked_attempt",
            }
        )

        # Emit directly to session log handler (bypasses logger hierarchy)
        try:
            self._session_log_handler.emit(record)
        except Exception as e:
            # If session log fails, we can't do much (can't log the failure!)
            # Just increment the error count
            pass

    def get_stats(self) -> dict:
        """
        Get interceptor statistics for monitoring.

        Returns:
            Dict with:
            - installed: Whether interceptor is currently installed
            - blocked_attempts: Number of StreamHandlers blocked
        """
        return {
            "installed": self._installed,
            "blocked_attempts": self._blocked_attempts,
        }

    def __enter__(self):
        """Context manager entry: install interceptor"""
        self.install()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit: uninstall interceptor"""
        self.uninstall()
        return False  # Don't suppress exceptions


# ============================================================================
# Utility Functions
# ============================================================================


def create_interceptor_with_session_log(session_log_path: str) -> LoggerInterceptor:
    """
    Create a LoggerInterceptor with a session log handler.

    This is a convenience function that creates both the session log
    FileHandler and the LoggerInterceptor in one call.

    Args:
        session_log_path: Path to session.log file

    Returns:
        Configured LoggerInterceptor instance

    Design Note: This function doesn't install the interceptor automatically.
    Caller must call .install() when ready.

    Example:
        interceptor = create_interceptor_with_session_log("session.log")
        interceptor.install()
        try:
            # ... TUI operation
        finally:
            interceptor.uninstall()
    """
    # Create session log handler
    # WHY FileHandler: All blocked attempts logged to file for debugging
    session_handler = logging.FileHandler(session_log_path, mode="a", encoding="utf-8")
    session_handler.setLevel(logging.DEBUG)
    session_handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    # Create interceptor
    interceptor = LoggerInterceptor(session_log_handler=session_handler)

    logger.debug(f"Created LoggerInterceptor with session log: {session_log_path}")
    return interceptor
