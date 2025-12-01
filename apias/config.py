"""
Configuration Module for APIAS.

Provides:
- YAML configuration file support for batch operations
- OpenAI model configuration with validation
- Default settings management
- Environment variable overrides
- Configuration validation
- Centralized constants for all timeouts, retries, and limits

DESIGN NOTES:
- All magic numbers should be defined here as constants
- Constants are organized by category (network, API, TUI, etc.)
- Each constant has a comment explaining its purpose and constraints

DO NOT:
- Scatter timeout/retry values throughout the codebase
- Add magic numbers to other files - add constants here first
- Remove constants without updating all references
"""

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Final, List, Optional, Union

logger = logging.getLogger(__name__)

# Try to import PyYAML, but make it optional
try:
    import yaml  # type: ignore[import-untyped]

    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    yaml = None


# =============================================================================
# CENTRALIZED CONSTANTS - Single source of truth for all configuration values
# =============================================================================
# DO NOT scatter these values throughout the codebase. Add new constants here.

# --- Network Timeouts (seconds) ---
# These affect how long we wait for external services to respond
HTTP_REQUEST_TIMEOUT: Final[int] = 10  # General HTTP requests (sitemap, pricing)
BROWSER_NAVIGATION_TIMEOUT: Final[int] = 30  # Playwright browser navigation
BROWSER_NETWORK_IDLE_TIMEOUT: Final[int] = 5000  # ms - wait for network idle
SUBPROCESS_TIMEOUT: Final[int] = 5  # Timeout for subprocess calls (ps, etc.)

# --- API Configuration ---
DEFAULT_MODEL: Final[str] = "gpt-4o-mini"  # Default OpenAI model
# CRITICAL: Set to 0 to disable OpenAI library's internal retry mechanism
# We handle retries ourselves with circuit breaker for proper user feedback
# Internal retries cause "retry storms" that flood the terminal with errors
OPENAI_MAX_RETRIES: Final[int] = 0  # NO internal retries - circuit breaker handles this
XML_VALIDATION_MAX_RETRIES: Final[int] = 1  # Retries for XML validation failures

# --- Retry Backoff Configuration ---
# WHY exponential backoff: Prevents API hammering with doubling delays
# WHY no jitter: APIAS is a single-client scraper, not a distributed system
# Jitter spreads retries for multiple competing clients - we don't have that
# Pure exponential backoff is 100% reproducible: attempt 0=1s, 1=2s, 2=4s, 3=8s...
# Base delay in seconds for first retry (doubles each attempt: 1s, 2s, 4s, 8s...)
RETRY_BASE_DELAY_SECONDS: Final[float] = 1.0
# Maximum delay cap to prevent excessively long waits
RETRY_MAX_DELAY_SECONDS: Final[float] = 30.0

# --- Event System Timeouts ---
# Event dispatch timeout for processing queued events in main loop
# WHY 50ms: Long enough to process multiple events, short enough for responsive TUI
EVENT_DISPATCH_TIMEOUT: Final[float] = (
    0.05  # seconds (50ms) for standard event processing
)
# Fast timeout for tight processing loops where responsiveness is critical
# WHY 10ms: Minimizes latency in rapid-fire processing loops
EVENT_DISPATCH_FAST_TIMEOUT: Final[float] = (
    0.01  # seconds (10ms) for fast loop processing
)

# --- HTML Content Chunking ---
# Maximum characters per HTML chunk for AI processing
# WHY 200000: Large chunks reduce API calls but must fit within model context window
# GPT-4 has 128K context, so 200K chars (~50K tokens) leaves room for response
HTML_MAX_CHUNK_SIZE: Final[int] = 200000  # Max chars for very large pages
# Default chunk size for normal processing - smaller for better accuracy
# WHY 80000: Balance between API cost efficiency and processing reliability
HTML_CHUNK_SIZE: Final[int] = 80000  # Default chunk size for chunking

# --- Batch Processing ---
DEFAULT_NUM_THREADS: Final[int] = 5  # Default concurrent threads for batch mode
MAX_SAFE_THREADS: Final[int] = 20  # Warn if more threads requested
BATCH_TUI_POLL_INTERVAL: Final[float] = 0.05  # seconds (50ms) for fluid batch TUI
# Single-page TUI polling is slower since less content to update
SINGLE_PAGE_TUI_POLL_INTERVAL: Final[float] = 0.1  # seconds (100ms) for single-page TUI
# Spinner animation interval for non-TUI progress indicators
SPINNER_ANIMATION_INTERVAL: Final[float] = 0.1  # seconds (100ms) for spinner frames
# Brief pause at end of processing so user can see final state
FINAL_STATE_PAUSE: Final[float] = 0.5  # seconds to display final state (single page)
# Batch mode has more information to review, so longer pause
BATCH_FINAL_STATE_PAUSE: Final[float] = 1.0  # seconds to display batch final state

# --- Memory Limits (to prevent OOM) ---
MAX_ERRORS_TO_TRACK: Final[int] = 1000  # Max errors in SessionErrorTracker
MAX_QUEUE_SIZE: Final[int] = 1000  # Max messages in TUIMessageQueue
MAX_DEFERRED_MESSAGES: Final[int] = 500  # Max deferred messages for summary

# --- File System ---
TEMP_FOLDER_PREFIX: Final[str] = "apias_"  # Prefix for temp folders
PROGRESS_FILE_NAME: Final[str] = "progress.json"
ERROR_LOG_FILE_NAME: Final[str] = "error_log.txt"

# --- TUI Display Settings ---
# Batch TUI (multiple URLs) - higher FPS for smooth scrolling and animations
TUI_REFRESH_FPS: Final[int] = 20  # Frames per second for batch Rich Live display
# Single-page TUI (chunk processing) - lower FPS since less dynamic
TUI_SINGLE_PAGE_FPS: Final[int] = 4  # FPS for single-page chunk processing display
# Waiting dashboard FPS - needs smooth spinner animation
TUI_WAITING_FPS: Final[int] = 10  # FPS for "press space to start" waiting screens
MAX_FAILED_URLS_TO_SHOW: Final[int] = 5  # Max failed URLs/errors to show in summary
FALLBACK_VERSION: Final[str] = "0.1.4"  # Fallback version if import fails

# --- Terminal Defaults ---
# Default terminal dimensions when size detection fails (e.g., headless mode)
DEFAULT_TERMINAL_WIDTH: Final[int] = 80  # Standard terminal width
DEFAULT_TERMINAL_HEIGHT: Final[int] = 24  # Standard terminal height
# Default max length for URL truncation in TUI displays
URL_TRUNCATE_MAX_LENGTH: Final[int] = 60  # Max chars before truncating URL
# Default width for progress bar in stats panel
STATS_PROGRESS_BAR_WIDTH: Final[int] = 40  # Character width of stats panel bar

# --- Keyboard/Thread Timing ---
# Keyboard listener polling interval and thread cleanup timeout
KEYBOARD_POLL_INTERVAL: Final[float] = 0.1  # seconds for keyboard input polling
KEYBOARD_THREAD_TIMEOUT: Final[float] = (
    1.0  # seconds to wait for keyboard thread cleanup
)
# Scroll debounce to prevent jittery scrolling from rapid key repeats
# WHY 50ms: Fast enough for responsive feel, slow enough to filter key repeat
# DO NOT: Set too low (<20ms) - causes jitter; too high (>100ms) - feels sluggish
SCROLL_DEBOUNCE_SECONDS: Final[float] = 0.05  # 50ms debounce between scroll events

# ThreadPoolExecutor shutdown timeout
# WHY: Gives running tasks time to complete gracefully before forcing exit
# Too short: Tasks may be interrupted mid-write, corrupting output
# Too long: Unresponsive exit experience for users
EXECUTOR_SHUTDOWN_TIMEOUT: Final[float] = 5.0  # seconds to wait for executor shutdown


# --- Progress Percentages (Single Source of Truth) ---
# WHY: Centralize progress values to avoid DRY violations and make adjustments easy
# USAGE: Import these constants instead of hardcoding percentages in multiple places
# Design: Progress represents percentage of overall processing for a single URL
class ProgressPercent:
    """Centralized progress percentages for TUI status updates.

    These values represent the overall progress through the processing pipeline.
    Adjust these values here to change progress display across entire application.

    Flow: SCRAPING -> CLEANING -> CHUNKING -> SENDING -> RECEIVING -> VALIDATING -> SAVING -> COMPLETE
    """

    SCRAPING: Final[float] = 10.0  # Started scraping HTML
    CLEANING: Final[float] = 20.0  # Cleaning HTML content
    CHUNKING: Final[float] = 30.0  # Preparing content for AI
    SENDING: Final[float] = 40.0  # Sending to AI model
    RECEIVING: Final[float] = 70.0  # Received AI response
    VALIDATING: Final[float] = 85.0  # Validating XML output
    SAVING: Final[float] = 95.0  # Saving XML files
    COMPLETE: Final[float] = 100.0  # Processing complete
    FAILED: Final[float] = 0.0  # Processing failed (reset to 0)


def get_system_temp_dir() -> Path:
    """
    Get the OS-specific temporary directory.

    Uses tempfile.gettempdir() which respects:
    - TMPDIR, TEMP, TMP environment variables
    - /tmp on Unix, %TEMP% on Windows

    Returns:
        Path to the system temporary directory
    """
    return Path(tempfile.gettempdir())


# Supported OpenAI models with their context windows
SUPPORTED_MODELS: Dict[str, Dict[str, Any]] = {
    "gpt-4o": {"context_window": 128000, "description": "Most capable GPT-4 model"},
    "gpt-4o-mini": {
        "context_window": 128000,
        "description": "Cost-effective GPT-4 model",
    },
    "gpt-4-turbo": {"context_window": 128000, "description": "GPT-4 Turbo model"},
    "gpt-4": {"context_window": 8192, "description": "Original GPT-4 model"},
    "gpt-3.5-turbo": {
        "context_window": 16385,
        "description": "Fast, cost-effective model",
    },
    "gpt-3.5-turbo-16k": {
        "context_window": 16385,
        "description": "GPT-3.5 with extended context",
    },
}


@dataclass
class APIASConfig:
    """
    Main configuration class for APIAS.

    Supports loading from YAML files, environment variables, or direct instantiation.
    """

    # OpenAI settings
    model: str = DEFAULT_MODEL
    api_key: str | None = None  # If None, reads from OPENAI_API_KEY env var
    max_tokens: int = 4096
    temperature: float = 0.1

    # Scraping settings
    num_threads: int = 5
    chunk_size: int = 50000  # Max characters per chunk
    max_retries: int = 3
    retry_delay: float = 1.0

    # Output settings
    output_folder: str | None = None  # Auto-generated if None
    output_format: str = "xml"

    # URL filtering
    whitelist_patterns: List[str] = field(default_factory=list)
    blacklist_patterns: List[str] = field(default_factory=list)
    limit: int | None = None  # Max URLs to process

    # TUI settings
    no_tui: bool = False  # Disable Rich TUI
    quiet: bool = False  # Minimal output, implies no_tui
    auto_resume: bool = False  # Automatically resume last session without prompting

    # Progress settings
    progress_file: str = "progress.json"
    atomic_saves: bool = True  # Use atomic file writes for progress

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self._validate()

    def _validate(self) -> None:
        """Validate configuration values."""
        # Validate model
        if self.model not in SUPPORTED_MODELS:
            logger.warning(
                f"Model '{self.model}' not in supported list. "
                f"Supported models: {list(SUPPORTED_MODELS.keys())}. "
                "Proceeding anyway as OpenAI may have new models."
            )

        # Validate numeric ranges
        if self.num_threads < 1:
            raise ValueError("num_threads must be at least 1")
        if self.num_threads > 20:
            logger.warning(
                f"num_threads={self.num_threads} is very high. "
                "This may cause rate limiting issues."
            )

        if self.chunk_size < 1000:
            raise ValueError("chunk_size must be at least 1000 characters")

        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")

        if self.temperature < 0 or self.temperature > 2:
            raise ValueError("temperature must be between 0 and 2")

        # quiet implies no_tui
        if self.quiet:
            self.no_tui = True

    def get_api_key(self) -> str:
        """Get the OpenAI API key from config or environment."""
        if self.api_key:
            return self.api_key
        env_key = os.environ.get("OPENAI_API_KEY")
        if not env_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable "
                "or provide api_key in configuration."
            )
        return env_key

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "num_threads": self.num_threads,
            "chunk_size": self.chunk_size,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "output_folder": self.output_folder,
            "output_format": self.output_format,
            "whitelist_patterns": self.whitelist_patterns,
            "blacklist_patterns": self.blacklist_patterns,
            "limit": self.limit,
            "no_tui": self.no_tui,
            "quiet": self.quiet,
            "auto_resume": self.auto_resume,
            "progress_file": self.progress_file,
            "atomic_saves": self.atomic_saves,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "APIASConfig":
        """Create config from dictionary."""
        # Filter to only known fields
        known_fields = {
            "model",
            "api_key",
            "max_tokens",
            "temperature",
            "num_threads",
            "chunk_size",
            "max_retries",
            "retry_delay",
            "output_folder",
            "output_format",
            "whitelist_patterns",
            "blacklist_patterns",
            "limit",
            "no_tui",
            "quiet",
            "auto_resume",
            "progress_file",
            "atomic_saves",
        }
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "APIASConfig":
        """
        Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file

        Returns:
            APIASConfig instance

        Raises:
            ImportError: If PyYAML is not installed
            FileNotFoundError: If config file doesn't exist
            ValueError: If YAML is invalid
        """
        if not HAS_YAML:
            raise ImportError(
                "PyYAML is required for YAML configuration. "
                "Install it with: pip install pyyaml"
            )

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in configuration file: {e}") from e

        if not isinstance(data, dict):
            raise ValueError(
                "Configuration file must contain a YAML mapping/dictionary"
            )

        return cls.from_dict(data)

    @classmethod
    def from_json(cls, path: str | Path) -> "APIASConfig":
        """
        Load configuration from a JSON file.

        Args:
            path: Path to the JSON configuration file

        Returns:
            APIASConfig instance
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return cls.from_dict(data)

    def save_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        if not HAS_YAML:
            raise ImportError(
                "PyYAML is required for YAML configuration. "
                "Install it with: pip install pyyaml"
            )

        path = Path(path)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def save_json(self, path: str | Path) -> None:
        """Save configuration to a JSON file."""
        path = Path(path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


def load_config(
    config_path: str | Path | None = None,
    cli_overrides: Dict[str, Any] | None = None,
) -> APIASConfig:
    """
    Load configuration with precedence: CLI args > config file > defaults.

    Args:
        config_path: Optional path to YAML or JSON config file
        cli_overrides: Optional dictionary of CLI argument overrides

    Returns:
        Merged APIASConfig instance
    """
    # Start with defaults
    config_dict: Dict[str, Any] = {}

    # Load from file if provided
    if config_path:
        path = Path(config_path)
        if path.suffix in (".yml", ".yaml"):
            file_config = APIASConfig.from_yaml(path)
        elif path.suffix == ".json":
            file_config = APIASConfig.from_json(path)
        else:
            raise ValueError(
                f"Unsupported config file format: {path.suffix}. "
                "Use .yaml, .yml, or .json"
            )
        config_dict = file_config.to_dict()

    # Apply CLI overrides (highest precedence)
    if cli_overrides:
        for key, value in cli_overrides.items():
            if value is not None:  # Only override if explicitly set
                config_dict[key] = value

    return APIASConfig.from_dict(config_dict)


def generate_example_config(path: str | Path = "apias_config.yaml") -> None:
    """
    Generate an example configuration file with comments.

    Args:
        path: Where to save the example config
    """
    example_yaml = """# APIAS Configuration File
# ========================
# This file configures APIAS behavior for batch scraping operations.
# All settings are optional - defaults will be used if not specified.

# OpenAI Model Settings
# ---------------------
# model: The OpenAI model to use for HTML-to-XML conversion
# Supported models: gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-4, gpt-3.5-turbo
model: gpt-4o-mini

# max_tokens: Maximum tokens in the response
max_tokens: 4096

# temperature: Creativity level (0.0 = deterministic, 2.0 = very creative)
temperature: 0.1

# Scraping Settings
# -----------------
# num_threads: Number of parallel threads for batch processing
num_threads: 5

# chunk_size: Maximum characters per HTML chunk sent to the API
chunk_size: 50000

# max_retries: Number of retry attempts for failed API calls
max_retries: 3

# retry_delay: Seconds to wait between retries
retry_delay: 1.0

# URL Filtering
# -------------
# whitelist_patterns: Only process URLs matching these patterns (regex)
whitelist_patterns: []
  # - ".*\\/api\\/.*"
  # - ".*\\/docs\\/.*"

# blacklist_patterns: Skip URLs matching these patterns (regex)
blacklist_patterns: []
  # - ".*\\/blog\\/.*"
  # - ".*\\.pdf$"

# limit: Maximum number of URLs to process (null = no limit)
limit: null

# Output Settings
# ---------------
# output_folder: Where to save results (null = auto-generated with timestamp)
output_folder: null

# output_format: Output format for extracted documentation
output_format: xml

# TUI Settings
# ------------
# no_tui: Disable the Rich terminal UI (for headless/script usage)
no_tui: false

# quiet: Minimal output mode (implies no_tui)
quiet: false

# auto_resume: Automatically resume the most recent incomplete session
auto_resume: false

# Progress Settings
# -----------------
# progress_file: Name of the progress tracking file
progress_file: progress.json

# atomic_saves: Use atomic file writes for progress (safer but slightly slower)
atomic_saves: true
"""

    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        f.write(example_yaml)
    logger.info(f"Example configuration saved to: {path}")


def validate_url(url: str) -> bool:
    """
    Validate that a URL is well-formed.

    Args:
        url: URL string to validate

    Returns:
        True if valid, False otherwise
    """
    from urllib.parse import urlparse

    try:
        result = urlparse(url)
        return all([result.scheme in ("http", "https"), result.netloc])
    except Exception:
        return False


def validate_urls(urls: List[str]) -> List[str]:
    """
    Validate a list of URLs, returning only valid ones.

    Args:
        urls: List of URL strings

    Returns:
        List of valid URLs (invalid ones are logged and skipped)
    """
    valid_urls = []
    for url in urls:
        if validate_url(url):
            valid_urls.append(url)
        else:
            logger.warning(f"Skipping invalid URL: {url}")
    return valid_urls
