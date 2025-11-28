#!/usr/bin/env python

"""
APIAS - API AUTO SCRAPER
by fmuaddib

**DESCRIPTION**
    When programming with an LLM as your copilot, you often find that the
    LLM does not know the latest version of the libraries or service APIs
    that you are using in your project. But converting such documentation
    into a format good enough for the LLM is not easy. An ideal format should
    be compact (to consume the minimum amount of tokens), well semantically
    annotated (to ensure that the scraped text meaning is not lost), and
    well structured (to be able to be machine readable by the LLM without
    errors).
    API AUTO SCRAPER is a program written in Python that does just that:
    scraping the documentations websites of the libraries to get their
    API specifications, remove all unneded html elements, and using an
    llm service from OpenAI (model: gpt-5-nano) it transform the html
    into a well structured and annotated XML file.
    It also produces a merged xml file concatenating all the scraped
    pages, so that with just adding such file to the LLM chat context,
    the AI can learn the updated API of the library.
    For example aider-chat has an option to add a read only document
    to the llm chat context (i.e. "/read-only api_doc.xml").

    Note that this is a scraper, NOT a crawler. To scrape all pages
    from an api website it uses the "sitemap.xml" file, usually
    found appended to the domain name (i.e. https://example.com/sitemap.xml).
    You can use whitelists and blacklists txt file to filter the urls of the
    sitemap.xml and scrape only the ones you are interested in.

    Enjoy!

**REQUIREMENTS**
    Your package installer (pip, pipx, conda, poetry..) may have troubles
    installing all dependencies. In that case, here is the guide to install
    the required libraries manually:

    *the Playwright library*
    You can install it manually with those two commands:

    python -m pip install --upgrade --upgrade-strategy only-if-needed playwright

    python -m playwright install --with-deps chromium

    *the BeautifulSoup library*
    You can install it manually with this command:

    python -m pip install --upgrade --upgrade-strategy only-if-needed beautifulsoup4

    *the Requests library*
    You can install it manually with this command:

    python -m pip install --upgrade --upgrade-strategy only-if-needed requests



**USAGE INSTRUCTIONS:**

**Set Up OpenAI API Key:**

   Export your OpenAI API key as an environment variable:

   export OPENAI_API_KEY='your-api-key-here'

   Replace `'your-api-key-here'` with your actual OpenAI API key.

   If you don't have an API key, you can get it here:

   https://platform.openai.com/api-keys

**Run the Script:**

   - **Single Page Processing:**

     python apias.py --url "https://example.com" --mode single

     This command processes the base URL `https://example.com`, extracting XML from the main page.

   - **Batch Mode Processing:**

     python apias.py --url "https://example.com" --mode batch --whitelist "whitelist.txt" --blacklist "blacklist.txt"

     This command processes all URLs extracted from the sitemap.xml of the base url, optionally filtering the urls using a whitelist text file (only urls matching at least one whitelist pattern are scraped) and a blacklist text file (the urls matching at least one blacklist pattern are not scraped). The resulting xml files are saved in a temp folder.

   - **Resume Batch Scrape Job:**

     python apias.py --resume "./temp_dir/progress.json"

     This command resumes a batch scraping job that was interrupted (or that ended with some urls failed to be scraped into xml). The --resume (or -r) parameter must be followed by the path to the "progress.json" file that is inside the temp folder of the scrape job to resume.


"""

from . import __version__

APP_NAME = "APIAS - API AUTO SCRAPER"
APP_FILENAME = "apias.py"
VERSION = __version__

import os
import sys
import time
import shutil
import logging
import argparse
import asyncio
import concurrent.futures
from typing import Dict, List, Optional, Tuple, Union, Any, Callable, Type, cast
from bs4 import BeautifulSoup, Comment
from types import TracebackType
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse
import requests
from openai import AsyncOpenAI
from openai import (
    OpenAIError,
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    APIStatusError,
)

# tenacity import removed - AsyncOpenAI handles retries internally
from requests.exceptions import RequestException
import fnmatch
import itertools
import platform
import re
import shlex
from types import FrameType
import subprocess
import threading
import defusedxml.ElementTree as DefusedET  # Use defusedxml to prevent XXE attacks
import xml.etree.ElementTree as ET  # For Element class and tree manipulation
from signal import SIGINT, SIGTERM, signal
import json

# Import TUI and mock API modules
from .tui import RichTUIManager, ChunkState, ChunkStatus, ProcessingStep
from .mock_api import MockAPIClient, mock_call_openai_api
from .batch_tui import BatchTUIManager, URLState
from .error_handler import (
    ErrorCategory,
    ErrorEvent,
    SessionErrorTracker,
    classify_openai_error,
    get_error_description,
    is_recoverable_error,  # Use centralized check instead of hardcoding categories
)
from .config import (
    # Network timeouts - DO NOT hardcode these values elsewhere
    HTTP_REQUEST_TIMEOUT,
    BROWSER_NAVIGATION_TIMEOUT,
    BROWSER_NETWORK_IDLE_TIMEOUT,
    SUBPROCESS_TIMEOUT,
    # API configuration
    OPENAI_MAX_RETRIES,
    XML_VALIDATION_MAX_RETRIES,
    # Batch processing - DO NOT hardcode polling intervals
    BATCH_TUI_POLL_INTERVAL,
    SINGLE_PAGE_TUI_POLL_INTERVAL,
    SPINNER_ANIMATION_INTERVAL,
    FINAL_STATE_PAUSE,
    BATCH_FINAL_STATE_PAUSE,
    # File system
    TEMP_FOLDER_PREFIX,
    ERROR_LOG_FILE_NAME,
    get_system_temp_dir,
)


def validate_xml(xml_string: str) -> bool:
    try:
        DefusedET.fromstring(xml_string)
        return True
    except (ET.ParseError, ValueError):
        return False


def count_valid_xml_files(folder: Path) -> tuple[int, int]:
    valid_count = 0
    total_count = 0
    for xml_file in folder.glob("processed_*.xml"):
        total_count += 1
        with open(xml_file, "r", encoding="utf-8") as f:
            content = f.read()
        if validate_xml(content):
            valid_count += 1
    return valid_count, total_count


def format_timestamp() -> str:
    """Format current timestamp for status messages (HH:MM:SS format)"""
    return datetime.now().strftime("%H:%M:%S")


def update_batch_status(
    batch_tui: Optional[BatchTUIManager],
    task_id: Optional[int],
    state: URLState,
    message: str,
    **kwargs: Any,
) -> None:
    """
    Helper function to update batch TUI with status messages.
    Handles Rich markup escaping, Unicode fallback, and logging.

    Args:
        batch_tui: Optional BatchTUIManager instance
        task_id: Optional task ID
        state: URLState for the update
        message: Status message (will be timestamped and escaped)
        **kwargs: Additional parameters to pass to update_task()
    """
    if not batch_tui or task_id is None:
        return

    try:
        # Add timestamp
        timestamp = format_timestamp()
        timestamped_msg = f"[{timestamp}] {message}"

        # Escape Rich markup special characters to prevent rendering issues
        # Replace [ and ] with escaped versions
        escaped_msg = timestamped_msg.replace("[", r"\[").replace("]", r"\]")
        # But allow our own timestamp brackets through
        escaped_msg = escaped_msg.replace(f"\\[{timestamp}\\]", f"[{timestamp}]")

        # Log the status message for debugging
        logger.info(f"Task #{task_id}: {message}")

        # Update the batch TUI
        batch_tui.update_task(task_id, state, status_message=escaped_msg, **kwargs)
    except Exception as e:
        # Fail gracefully if status update fails
        logger.warning(f"Failed to update batch TUI status: {e}")


# Global variables for cost tracking
total_cost = 0.0
cost_lock = threading.Lock()

# Global variable for progress tracking
progress_tracker: Dict[str, Dict[str, Union[str, float]]] = {}
progress_file = "progress.json"

# JSON Schema for structured XML output from GPT-5 Nano
# This ensures the model returns properly formatted XML with guaranteed structure
XML_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "xml_content": {
            "type": "string",
            "description": "The complete XML document as a properly escaped XML string",
        },
        "document_type": {
            "type": "string",
            "enum": ["CLASS", "MODULE", "API_USAGE"],
            "description": "The classification of the documentation page",
        },
        "completeness_check": {
            "type": "boolean",
            "description": "Confirmation that all content from the HTML was extracted without omissions",
        },
    },
    "required": ["xml_content", "document_type", "completeness_check"],
    "additionalProperties": False,
}

# Unicode block characters for separators and box drawing
SEPARATOR = "━" * 80
DOUBLE_SEPARATOR = "═" * 80
SUCCESS_SEPARATOR = "✨" + ("━" * 78) + "✨"
INFO_SEPARATOR = "ℹ️ " + ("─" * 76) + " ℹ️"
ERROR_SEPARATOR = "❌" + ("━" * 78) + "❌"
WARNING_SEPARATOR = "⚠️ " + ("─" * 76) + " ⚠️"
BOX_HORIZONTAL = "─"
BOX_VERTICAL = "│"
BOX_TOP_LEFT = "┌"
BOX_TOP_RIGHT = "┐"
BOX_BOTTOM_LEFT = "└"
BOX_BOTTOM_RIGHT = "┘"
BOX_T_DOWN = "┬"
BOX_T_UP = "┴"
BOX_T_RIGHT = "├"
BOX_T_LEFT = "┤"
BOX_CROSS = "┼"

# ============================
# Configuration and Setup
# ============================

# Configure Logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def configure_logging_for_tui(no_tui: bool) -> None:
    """
    Configure logging level based on TUI mode.

    When TUI is active (no_tui=False), suppress DEBUG logs to keep terminal clean.
    When TUI is disabled (no_tui=True), keep DEBUG level for detailed output.

    Args:
        no_tui: If True, keep DEBUG logging. If False, suppress to INFO level.
    """
    if not no_tui:
        # TUI mode - suppress DEBUG logs for clean display
        logging.getLogger().setLevel(logging.INFO)
        logger.setLevel(logging.INFO)
    else:
        # No TUI mode - keep DEBUG logs for detailed output
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)


def suppress_console_logging() -> tuple[list[logging.Handler], int]:
    """
    Suppress console logging by removing StreamHandlers AND raising log level.
    Returns the removed handlers and original level so they can be restored later.

    This prevents logger.info/warning/error from corrupting the Rich TUI.
    Rich's Live display uses its own Console which bypasses Python logging.

    NOTE: We do NOT redirect stdout/stderr because Rich needs them to render.
    The TUI uses screen=True which enables alternate screen buffer mode,
    keeping all Rich output separate from normal terminal output.
    """
    root_logger = logging.getLogger()
    removed_handlers: list[logging.Handler] = []
    original_level = root_logger.level

    # Remove all console logging handlers
    for handler in root_logger.handlers[
        :
    ]:  # Copy list to avoid modification during iteration
        if isinstance(handler, logging.StreamHandler) and handler.stream in (
            sys.stderr,
            sys.stdout,
        ):
            root_logger.removeHandler(handler)
            removed_handlers.append(handler)

    # Also raise logging level to CRITICAL to suppress all normal logging
    # This ensures that even if a handler wasn't removed, it won't log
    root_logger.setLevel(logging.CRITICAL + 1)  # Higher than CRITICAL = nothing logs

    return removed_handlers, original_level


def restore_console_logging(
    handlers_and_level: tuple[list[logging.Handler], int],
) -> None:
    """
    Restore console logging handlers and original log level.

    Args:
        handlers_and_level: Tuple of (handlers, level) returned by suppress_console_logging
    """
    handlers, original_level = handlers_and_level
    root_logger = logging.getLogger()

    # Restore original log level
    root_logger.setLevel(original_level)

    # Restore handlers
    if handlers:
        for handler in handlers:
            root_logger.addHandler(handler)


# Create a temp folder in OS-specific temp directory with datetime suffix
# Using get_system_temp_dir() from config ensures proper temp location on all OSes
# The folder name uses a prefix for easy identification and cleanup
_temp_folder_name = f"{TEMP_FOLDER_PREFIX}{datetime.now().strftime('%Y%m%d_%H%M%S')}"
temp_folder = get_system_temp_dir() / _temp_folder_name
temp_folder.mkdir(exist_ok=True)

# Create an error log file using centralized filename constant
# This ensures consistent naming across the codebase
error_log_file = temp_folder / ERROR_LOG_FILE_NAME
# Create an empty error log file (truncate if exists)
with open(error_log_file, "w", encoding="utf-8") as f:
    pass


def get_openai_api_key() -> str:
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    if not openai_api_key.startswith("sk-"):
        raise ValueError("Invalid OpenAI API key format - should start with 'sk-'")
    return openai_api_key


# Global flag for graceful shutdown
# This flag is checked by processing loops to stop cleanly
shutdown_flag = False

# Track if we've already handled a shutdown signal
# Prevents double-handling if user presses Ctrl+C multiple times
_shutdown_handled = False


def signal_handler(signum: int, frame: Optional[FrameType]) -> None:
    """
    Handle interrupt signals (SIGINT/Ctrl+C and SIGTERM) with graceful shutdown.

    Sets shutdown_flag to True so processing loops can exit cleanly.
    Saves progress to allow resuming later and displays resume instructions.

    DESIGN NOTES:
    - Does NOT call sys.exit() in the handler - this is an anti-pattern
    - Instead, sets shutdown_flag and lets the main loop handle exit
    - On second signal, exits immediately (for users who really want to quit)

    DO NOT:
    - Call sys.exit() here - it can cause issues with cleanup
    - Do heavy processing - signal handlers should be fast
    - Use non-reentrant functions like print() for critical operations
    """
    global shutdown_flag, _shutdown_handled

    # On second signal, exit immediately (user really wants to quit)
    if _shutdown_handled:
        logger.warning("Second interrupt received. Forcing immediate exit...")
        # Use os._exit to bypass cleanup - user wants out NOW
        os._exit(1)

    _shutdown_handled = True
    signal_name = "SIGINT" if signum == SIGINT else "SIGTERM"
    logger.info(f"Received {signal_name}. Initiating graceful shutdown...")
    shutdown_flag = True

    # Save progress before exiting so user can resume later
    # We do this in the handler because the main loop may not reach cleanup
    try:
        update_progress_file()
        if progress_file and os.path.exists(progress_file):
            # Use sys.stderr for signal handler output (more reliable than print)
            sys.stderr.write("\n")
            sys.stderr.write("=" * 60 + "\n")
            sys.stderr.write("  SESSION SAVED - You can resume later!\n")
            sys.stderr.write("=" * 60 + "\n")
            sys.stderr.write("\n  To resume this session, run:\n")
            sys.stderr.write(f'    apias --resume "{progress_file}"\n')
            sys.stderr.write("\n  Progress has been saved to:\n")
            sys.stderr.write(f"    {progress_file}\n")
            sys.stderr.write("=" * 60 + "\n")
            sys.stderr.write("\n")
            sys.stderr.flush()
    except Exception as e:
        logger.warning(f"Could not save progress on shutdown: {e}")

    # Do NOT call sys.exit() here - let the main loop detect shutdown_flag
    # and exit cleanly. This allows proper cleanup of resources.


# Register signal handlers for graceful shutdown
# SIGINT: Ctrl+C from terminal
# SIGTERM: kill command, Docker stop, systemd stop, etc.
signal(SIGINT, signal_handler)
signal(SIGTERM, signal_handler)

# ============================
# Helper Functions from DSL
# ============================


def escape_xml(xml_doc: str) -> str:
    """
    Escapes XML special characters in a string.
    """
    xml_doc = xml_doc.replace('"', "&quot;")
    xml_doc = xml_doc.replace("'", "&apos;")
    xml_doc = xml_doc.replace("<", "&lt;")
    xml_doc = xml_doc.replace(">", "&gt;")
    xml_doc = xml_doc.replace("&", "&amp;")
    return xml_doc


def unescape_xml(xml_doc: str) -> str:
    """
    To unescape the text strings and the attributes values.
    DO NOT UNESCAPE CDATA, Comments and Processing Instructions
    """
    xml_doc = xml_doc.replace("&quot;", '"')
    xml_doc = xml_doc.replace("&apos;", "'")
    xml_doc = xml_doc.replace("&lt;", "<")
    xml_doc = xml_doc.replace("&gt;", ">")
    xml_doc = xml_doc.replace("&amp;", "&")
    return xml_doc


def extract_xml_from_input(input_data: str) -> str:
    """
    Extracts and validates XML content from the input string.
    """
    input_data = input_data.strip()
    if input_data.startswith("```xml") and input_data.endswith("```"):
        input_data = input_data[len("```xml") : -len("```")]
    elif input_data.startswith("```XML") and input_data.endswith("```"):
        input_data = input_data[len("```XML") : -len("```")]
    elif input_data.startswith("```") and input_data.endswith("```"):
        input_data = input_data[len("```") : -len("```")]

    input_data = input_data.replace("\\\\n", "\n").replace("\\n", "\n")
    input_data = input_data.replace('\\\\\\"', '\\\\"').replace('\\"', '"')

    xml_content = input_data.strip()
    if not (
        xml_content.lower().startswith("<?xml")
        or xml_content.lower().startswith("<xml")
    ):
        xml_content = (
            '<?xml version="1.0" encoding="UTF-8"?>\n<XML>\n' + xml_content + "\n</XML>"
        )

    # Handle code tags
    xml_content = re.sub(
        r"<code>(.*?)</code>",
        lambda m: f"<CODE>{escape_xml(m.group(1))}</CODE>",
        xml_content,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # Validate the XML content
    logger.debug(f"Extracted XML Content:\n{xml_content}")
    try:
        DefusedET.fromstring(xml_content)  # Validates XML
    except ET.ParseError as e:
        logger.error(f"Extracted XML is not valid: {e}")
        raise ValueError("Invalid XML content.") from e

    return xml_content


def extract_xml_from_input_iter(input_data: str) -> str:
    """
    Extracts and validates XML content from the input string within iterations.
    """
    input_data = input_data.strip()
    if input_data.startswith("```xml") and input_data.endswith("```"):
        input_data = input_data[len("```xml") : -len("```")]
    elif input_data.startswith("```XML") and input_data.endswith("```"):
        input_data = input_data[len("```XML") : -len("```")]
    elif input_data.startswith("```") and input_data.endswith("```"):
        input_data = input_data[len("```") : -len("```")]

    input_data = input_data.replace("\\\\n", "\n").replace("\\n", "\n")
    input_data = input_data.replace('\\\\\\"', '\\\\"').replace('\\"', '"')

    xml_content = input_data
    if not (
        xml_content.lower().startswith("<xml") and xml_content.lower().endswith("xml>")
    ):
        xml_content = (
            '<XML version="1.0" encoding="UTF-8" standalone="yes" >\n'
            + xml_content
            + "\n</XML>"
        )

    # Validate the XML content
    logger.debug(f"Extracted Iteration XML Content:\n{input_data}")
    try:
        DefusedET.fromstring(xml_content)  # Validates XML
    except ET.ParseError:
        logger.error("ERROR - Extracted XML is not valid")
    finally:
        return xml_content


def merge_xmls(
    temp_folder: Path,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> str:
    """
    Merges multiple XML documents from the temp folder into a single XML API document.
    Includes the source URL as the first child node for each document.

    Args:
        temp_folder: Path to folder containing XML files
        progress_callback: Optional callback function(current, total, message) for progress updates
    """
    root = ET.Element("TEXTUAL_API")
    error_log = []

    # Get list of XML files and sort by task ID for deterministic order
    # Files are named: processed_1.xml, processed_2.xml, etc.
    xml_files = sorted(
        temp_folder.glob("processed_*.xml"),
        key=lambda f: int(f.stem.split("_")[1]),  # Extract task ID from filename
    )
    total_files = len(xml_files)

    for idx, xml_file in enumerate(xml_files, start=1):
        if progress_callback:
            progress_callback(idx, total_files, f"Merging {xml_file.name}")

        try:
            try:
                with open(xml_file, "r", encoding="utf-8") as f:
                    xml_content = f.read()
            except UnicodeDecodeError:
                logger.warning(
                    f"Unicode decode error in file {xml_file}. Trying with 'latin-1' encoding."
                )
                with open(xml_file, "r", encoding="latin-1") as f:
                    xml_content = f.read()

            # Parse the XML content
            doc = DefusedET.fromstring(xml_content)

            # Create a new element for this document
            doc_element = ET.SubElement(root, "DOCUMENT")

            # Add the source URL as the first child
            source_url = doc.find(".//SOURCE_URL")
            if source_url is not None:
                doc_element.append(source_url)

            # Append all other elements
            for child in doc:
                if child.tag != "SOURCE_URL":
                    doc_element.append(child)

        except ET.ParseError as e:
            error_message = f"Invalid XML in file {xml_file}: {e}"
            logger.warning(error_message)
            error_log.append(error_message)
        except FileNotFoundError:
            error_message = f"XML file not found: {xml_file}"
            logger.warning(error_message)
            error_log.append(error_message)
        except Exception as e:
            error_message = f"Error processing file {xml_file}: {str(e)}"
            logger.warning(error_message)
            error_log.append(error_message)

    # Notify validation step
    if progress_callback:
        progress_callback(total_files, total_files, "Validating merged XML")

    # Write error log
    with open(temp_folder / "errors.log", "w", encoding="utf-8") as f:
        f.write("\n".join(error_log))

    merged_xml = ET.tostring(root, encoding="unicode", method="xml")

    # Notify completion
    if progress_callback:
        progress_callback(total_files, total_files, "Merge complete")

    return merged_xml


def remove_links(element: ET.Element) -> None:
    """
    Recursively removes all link elements from the given XML element.
    """
    for child in list(element):
        if child.tag.lower() in ["a", "link"]:
            element.remove(child)
        else:
            remove_links(child)


# Navigation keywords limited to English
NAVIGATION_KEYWORDS = [
    "nav",
    "menu",
    "sidebar",
    "footer",
    "breadcrumb",
    "pager",
    "pagination",
    "header",
    "navigation",
    "submenu",
    "tabs",
    "navbar",
    "main-navigation",
    "site-navigation",
    "skip-navigation",
    "top-nav",
    "bottom-nav",
    "side-nav",
    "advert",
    "ads",
    "sponsor",
    "related",
    "cookies",
    "banner",
]

# Hidden styles, classes, and attributes
HIDDEN_STYLES = [
    "display:none",
    "visibility:hidden",
    "opacity:0",
    "height:0",
    "width:0",
    "position:absolute",
]
HIDDEN_CLASSES = [
    "hidden",
    "collapsed",
    "d-none",
    "invisible",
    "sr-only",
    "visually-hidden",
    "offscreen",
]
HIDDEN_ATTRIBUTES = ["hidden", "aria-hidden", "data-hidden"]


def is_main_content(element: BeautifulSoup) -> bool:
    """Check if the element is the main content."""
    if element.name == "main" or element.get("role") == "main":
        return True
    for attr in ["id", "class"]:
        attr_values = element.get(attr, [])
        if isinstance(attr_values, list):
            attr_values = " ".join(attr_values)
        if attr_values and any(
            keyword in attr_values.lower() for keyword in ["content", "main"]
        ):
            return True
    return False


def has_significant_text(element: BeautifulSoup, threshold: int = 100) -> bool:
    """Determine if an element contains significant text content."""
    text_length = len(element.get_text(strip=True))
    return text_length > threshold


def contains_navigation_keywords(attr_values: Union[str, List[str]]) -> bool:
    """Check if attribute values contain any navigation keywords."""
    if isinstance(attr_values, list):
        attr_values = " ".join(attr_values)
    attr_values_lower = attr_values.lower()
    return any(keyword in attr_values_lower for keyword in NAVIGATION_KEYWORDS)


def is_navigation(element: BeautifulSoup) -> bool:
    """Determine if an element is likely a navigation or non-content element."""
    if is_main_content(element):
        return False
    if element.name in ["nav", "header", "footer", "aside"]:
        return True
    for attr in ["class", "id", "role", "aria-label"]:
        attr_values = element.get(attr, [])
        if attr_values and contains_navigation_keywords(attr_values):
            return True
    links = element.find_all("a")
    text_length = len(element.get_text(strip=True))
    if len(links) > 5 and text_length < 100:
        return True
    if element.find_all("ul") and not has_significant_text(element):
        return True
    if element.name == "div":
        class_attr = element.get("class", [])
        if isinstance(class_attr, list) and any(
            any(keyword in cls.lower() for keyword in ["advert", "ads", "sponsor"])
            for cls in class_attr
        ):
            return True
    return False


def clean_styles(styles: str, hidden_styles: List[str]) -> str:
    """Remove hidden styles from the style attribute."""
    style_list = [s.strip() for s in styles.split(";") if s.strip()]
    visible_styles = [
        s
        for s in style_list
        if not any(hidden_style in s.replace(" ", "") for hidden_style in hidden_styles)
    ]
    return "; ".join(visible_styles)


def expand_hidden_content(soup: BeautifulSoup) -> None:
    """Expand hidden content by removing styles, classes, and attributes that hide elements."""
    for element in soup.find_all(True):
        # Remove hidden styles
        if element.has_attr("style"):
            cleaned_style = clean_styles(element["style"], HIDDEN_STYLES)
            if cleaned_style:
                element["style"] = cleaned_style
            else:
                del element["style"]
        # Remove hidden classes
        if element.has_attr("class"):
            visible_classes = [
                cls for cls in element["class"] if cls.lower() not in HIDDEN_CLASSES
            ]
            if visible_classes:
                element["class"] = visible_classes
            else:
                del element["class"]
        # Remove hidden attributes
        for attr in HIDDEN_ATTRIBUTES:
            element.attrs.pop(attr, None)


def remove_elements(soup: BeautifulSoup, condition_func: Callable[[Any], bool]) -> None:
    """Remove elements from the soup based on a condition function."""
    elements_to_remove = soup.find_all(condition_func)
    for element in elements_to_remove:
        element.decompose()


def remove_comments(soup: BeautifulSoup) -> None:
    """Remove comments from the soup."""
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()


def clean_html(html_content: str) -> str:
    """
    Clean the HTML content by removing navigation elements and expanding hidden content.

    Args:
        html_content (str): The input HTML content as a string.

    Returns:
        str: The cleaned and compacted HTML content.
    """
    # Use lxml parser for better handling of malformed HTML (optional)
    soup = BeautifulSoup(html_content, "lxml")

    # Remove comments
    remove_comments(soup)

    # Expand hidden content
    expand_hidden_content(soup)

    # Remove navigation and non-content elements
    remove_elements(soup, is_navigation)

    # Remove script and style tags
    for element in soup(["script", "style"]):
        element.decompose()

    # DO NOT prettify pre/code tags - this causes double-escaping and bloats HTML size!
    # Just leave them as-is - BeautifulSoup will handle them correctly

    # Get the compact HTML string
    compact_html = soup.decode(formatter="minimal")

    # Remove excessive whitespace, except within pre and code tags
    compact_html = re.sub(r"\s+(?![^<>]*</(?:pre|code)>)", " ", compact_html)

    return compact_html


# ============================
# Playwright Scraper Module
# ============================

# Copied from user-provided scraper module


def get_best_invocation_for_this_python() -> str:
    """Try to figure out the best way to invoke the current Python."""
    exe = sys.executable
    exe_name = os.path.basename(exe)

    # Try to use the basename, if it's the first executable.
    found_executable = shutil.which(exe_name)
    if found_executable and os.path.samefile(found_executable, exe):
        return exe_name

    # Use the full executable name, because we couldn't find something simpler.
    return exe


def safe_abs_path(res: Union[str, Path]) -> str:
    """Gives an abs path, which safely returns a full (not 8.3) windows path"""
    res = Path(res).resolve()
    return str(res)


def touch_file(fname: Union[str, Path]) -> bool:
    fname = Path(fname)
    try:
        fname.parent.mkdir(parents=True, exist_ok=True)
        fname.touch()
        return True
    except OSError:
        return False


def printable_shell_command(cmd_list: List[str]) -> str:
    """
    Convert a list of command arguments to a properly shell-escaped string.

    Args:
        cmd_list (list): List of command arguments.

    Returns:
        str: Shell-escaped command string.
    """
    if platform.system() == "Windows":
        return subprocess.list2cmdline(cmd_list)
    else:
        return shlex.join(cmd_list)


def get_pip_install(args: List[str]) -> List[str]:
    cmd = [
        get_best_invocation_for_this_python(),
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--upgrade-strategy",
        "only-if-needed",
    ]
    cmd += args
    return cmd


def run_install(cmd: List[str]) -> Tuple[bool, str]:
    print()
    print("Installing:", printable_shell_command(cmd))

    try:
        output: List[str] = []
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            encoding=sys.stdout.encoding,
            errors="replace",
        )
        # Create and start the spinner using context manager
        with Spinner("Installing...") as spinner:
            assert process.stdout is not None
            while True:
                char = process.stdout.read(1)
                if not char:
                    break

                output.append(char)
                spinner.step()

        return_code = process.wait()
        output_str = "".join(output)

        if return_code == 0:
            print("Installation complete.")
            print()
            return True, output_str

    except subprocess.CalledProcessError as e:
        print(f"\nError running pip install: {e}")

    print("\nInstallation failed.\n")

    return False, "".join(output)


class Spinner:
    spinner_chars = itertools.cycle(["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"])

    def __init__(self, text: str, quiet: bool = False) -> None:
        """
        Initialize spinner.

        Args:
            text: The text to display next to the spinner
            quiet: If True, suppress all output (for use when TUI is active)
        """
        self.text = text
        self.quiet = quiet
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.spinner_thread = threading.Thread(target=self._spin)
        self.spinner_thread.daemon = True

    def __enter__(self) -> "Spinner":
        self.start()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        self.stop()

    def start(self) -> None:
        if not self.quiet:
            print("Press CTRL-C to stop the processing.")
        self.spinner_thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.pause_event.set()
        # Only join if thread was started
        if self.spinner_thread.is_alive():
            self.spinner_thread.join()
        self._clear_line()

    def pause(self) -> None:
        self.pause_event.set()

    def resume(self) -> None:
        self.pause_event.clear()

    def _spin(self) -> None:
        while not self.stop_event.is_set():
            if not self.pause_event.is_set() and not self.quiet:
                self._clear_line()
                print(f"\r{self.text} {next(self.spinner_chars)}", end="", flush=True)
            # Use centralized spinner animation interval - DO NOT hardcode timing values
            time.sleep(SPINNER_ANIMATION_INTERVAL)

    def update_text(self, new_text: str) -> None:
        self.pause()
        if not self.quiet:
            self._clear_line()
            print(f"\r{new_text}")
        self.text = new_text
        self.resume()

    def step(self) -> None:
        self.pause()
        if not self.quiet:
            self._clear_line()
            print(f"\r{self.text} {next(self.spinner_chars)}", end="", flush=True)
        self.resume()

    def end(self) -> None:
        self.stop()

    def _clear_line(self) -> None:
        if not self.quiet:
            print(
                "\r" + " " * (shutil.get_terminal_size().columns - 1),
                end="",
                flush=True,
            )


def spinner_context(text: str) -> Spinner:
    spinner = Spinner(text)
    return spinner


def find_common_root(abs_fnames: List[str]) -> str:
    if len(abs_fnames) == 1:
        return safe_abs_path(os.path.dirname(list(abs_fnames)[0]))
    if abs_fnames:
        return safe_abs_path(os.path.commonpath(list(abs_fnames)))
    return safe_abs_path(os.getcwd())


def install_playwright() -> bool:
    try:
        from playwright.sync_api import sync_playwright

        has_pip = True
    except ImportError:
        has_pip = False

    try:
        with sync_playwright() as p:
            p.chromium.launch()
            has_chromium = True
    except Exception:
        has_chromium = False

    if has_pip and has_chromium:
        return True

    pip_cmd = get_pip_install(["playwright"])
    chromium_cmd = "-m playwright install --with-deps chromium"
    chromium_cmd_list: List[str] = [sys.executable] + chromium_cmd.split()

    cmds = ""
    if not has_pip:
        cmds += " ".join(pip_cmd) + "\n"
    if not has_chromium:
        cmds += " ".join(chromium_cmd_list) + "\n"

    text = f"""For the best web scraping, install Playwright:

{cmds}
"""

    print(text)

    if not has_pip:
        success, output = run_install(pip_cmd)
        if not success:
            print(output)
            return False

    success, output = run_install(chromium_cmd_list)
    if not success:
        print(output)
        return False

    return True


class Scraper:
    playwright_available: Optional[bool] = None
    playwright_instructions_shown: bool = False

    def __init__(
        self,
        playwright_available: Optional[bool] = None,
        verify_ssl: bool = True,
        timeout: int = 30,
        quiet: bool = False,
    ):
        """
        `print_error` - a function to call to print error/debug info.
        `verify_ssl` - if False, disable SSL certificate verification when scraping.
        `timeout` - timeout in seconds for the scraping operation.
        `quiet` - if True, suppress all spinner and print output (for TUI mode).
        """

        self.print_error: Callable[[str], None] = print
        self.quiet = quiet

        self.playwright_available = (
            playwright_available
            if playwright_available is not None
            else install_playwright()
        )
        self.verify_ssl = verify_ssl
        self.timeout = timeout

    def scrape(self, url: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Scrape a url. If HTML scrape it with playwright.
        If it's plain text or non-HTML, return it as-is.

        Args:
            url (str): The URL to scrape.

        Returns:
            tuple: A tuple containing:
                - str: The scraped content, or None if scraping failed.
                - str: The MIME type of the content, or None if not available.
        """

        if not self.playwright_available:
            install_playwright()

        content, mime_type = self.scrape_with_playwright(url)

        if not content:
            self.print_error(f"Failed to retrieve content from {url}")
            return None, None

        # Check if the content is HTML based on MIME type or content
        if (mime_type and mime_type.startswith("text/html")) or (
            mime_type is None and self.looks_like_html(content)
        ):
            slimdown_result = slimdown_html(content)
            content, page_title = slimdown_result[0], slimdown_result[1]

        filename = str(Path(temp_folder, f"{page_title}_scraped.html"))
        self.write_text(filename, content)

        return content, mime_type

    def looks_like_html(self, content: str) -> bool:
        """
        Check if the content looks like HTML.
        """
        if isinstance(content, str):
            # Check for common HTML patterns
            html_patterns = [
                r"<!DOCTYPE\s+html",
                r"<html\b",
                r"<head\b",
                r"<body\b",
                r"<div\b",
                r"<p\b",
                r"<a\s+href=",
                r"<img\b",
                r"<script\b",
                r"<link\b",
                r"<meta\b",
                r"<table\b",
                r"<form\b",
                r"<input\b",
                r"<style\b",
                r"<span\b",
                r"<ul\b",
                r"<ol\b",
                r"<li\b",
                r"<h[1-6]\b",
            ]
            # Check if any of the patterns match
            if any(
                re.search(pattern, content, re.IGNORECASE) for pattern in html_patterns
            ):
                return True

            # Additional check for HTML entity references
            if re.search(r"&[a-z]+;|&#\d+;", content, re.IGNORECASE):
                return True

            # Check for a high ratio of HTML-like content
            total_length = len(content)
            html_like_content = re.findall(r"<[^>]+>", content)
            html_ratio = sum(len(tag) for tag in html_like_content) / total_length
            if html_ratio > 0.1:  # If more than 10% of content looks like HTML tags
                return True

        return False

    def scrape_with_playwright(self, url: str) -> Tuple[Optional[str], Optional[str]]:
        import playwright  # noqa: F401
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
        from playwright.sync_api import sync_playwright

        with Spinner(f"Scraping {url}", quiet=self.quiet) as spinner:
            try:
                with sync_playwright() as p:
                    try:
                        browser = p.chromium.launch()
                    except Exception as e:
                        self.playwright_available = False
                        self.print_error(str(e))
                        return None, None

                    try:
                        context = browser.new_context(
                            ignore_https_errors=not self.verify_ssl
                        )
                        page = context.new_page()

                        user_agent = page.evaluate("navigator.userAgent")
                        user_agent = user_agent.replace("Headless", "")
                        user_agent = user_agent.replace("headless", "")
                        # user_agent += " " + custom_user_agent

                        page.set_extra_http_headers({"User-Agent": user_agent})

                        response = None
                        try:
                            spinner.update_text(f"Loading {url}")
                            # Use centralized timeout constant from config
                            response = page.goto(
                                url,
                                wait_until="networkidle",
                                timeout=BROWSER_NETWORK_IDLE_TIMEOUT,
                            )
                        except PlaywrightTimeoutError:
                            self.print_error(f"Timeout while loading {url}")
                        except PlaywrightError as e:
                            self.print_error(f"Error navigating to {url}: {str(e)}")
                            return None, None

                        try:
                            spinner.update_text("Retrieving content")
                            content = page.content()
                            mime_type = None
                            if response:
                                content_type = response.header_value("content-type")
                                if content_type:
                                    mime_type = content_type.split(";")[0]
                        except PlaywrightError as e:
                            self.print_error(f"Error retrieving page content: {str(e)}")
                            content = None
                            mime_type = None
                    finally:
                        browser.close()

                return content, mime_type
            except Exception as e:
                self.print_error(f"Unexpected error during scraping: {str(e)}")
                return None, None

    def write_text(self, filename: str, content: str) -> None:
        try:
            with open(str(filename), "w", encoding="utf-8") as f:
                f.write(content)
        except OSError as err:
            print(f"Unable to write file {filename}: {err}")


def slimdown_html(
    page_source: str,
) -> Tuple[
    str,
    Optional[str],
    List[str],
    List[str],
    List[str],
    List[str],
    List[Tuple[str, str]],
]:
    logger.debug("=== slimdown_html invoked ===")
    logger.debug(f"Input HTML size: {len(page_source)} characters")

    # Clean the HTML content from navigation elements and ads
    logger.debug("Cleaning HTML (removing navigation and ads)...")
    cleaned_html = clean_html(page_source)
    logger.debug(f"After clean_html: {len(cleaned_html)} characters")

    soup = BeautifulSoup(cleaned_html, "html.parser")

    # Remove SVG elements
    svg_count = len(soup.find_all("svg"))
    logger.debug(f"Removing {svg_count} SVG elements...")
    for svg in soup.find_all("svg"):
        svg.decompose()

    # Extract and remove images
    images = []
    for img in soup.find_all("img"):
        if "src" in img.attrs:
            images.append(img["src"])
        img.decompose()
    logger.debug(f"Extracted and removed {len(images)} images")

    # Remove data URIs
    data_href_count = len(soup.find_all(href=lambda x: x and x.startswith("data:")))
    data_src_count = len(soup.find_all(src=lambda x: x and x.startswith("data:")))
    logger.debug(f"Removing {data_href_count + data_src_count} data URI elements...")
    for tag in soup.find_all(href=lambda x: x and x.startswith("data:")):
        tag.decompose()
    for tag in soup.find_all(src=lambda x: x and x.startswith("data:")):
        tag.decompose()

    # Remove script and style tags
    script_style_count = len(soup.find_all(["script", "style"]))
    logger.debug(f"Removing {script_style_count} script/style tags...")
    for tag in soup.find_all(["script", "style"]):
        tag.decompose()

    # Extract code examples
    code_examples = []
    for code in soup.find_all("pre"):
        code_examples.append(code.get_text())
        code["class"] = code.get("class", []) + ["extracted-code"]
    logger.debug(f"Extracted {len(code_examples)} code examples")

    # Extract method signatures
    method_signatures = []
    for method in soup.find_all("div", class_="method-signature"):
        method_signatures.append(method.get_text())
        method["class"] = method.get("class", []) + ["extracted-method"]
    logger.debug(f"Extracted {len(method_signatures)} method signatures")

    # Extract class definitions
    class_definitions = []
    for class_def in soup.find_all("div", class_="class-definition"):
        class_definitions.append(class_def.get_text())
        class_def["class"] = class_def.get("class", []) + ["extracted-class"]
    logger.debug(f"Extracted {len(class_definitions)} class definitions")

    # Extract links
    links = []
    for a in soup.find_all("a", href=True):
        links.append((a["href"], a.get_text()))
    logger.debug(f"Extracted {len(links)} links")

    # Remove all attributes except href and class
    logger.debug("Removing unnecessary attributes (keeping only href and class)...")
    for tag in soup.find_all(True):
        for attr in list(tag.attrs):
            if attr not in ["href", "class"]:
                tag.attrs.pop(attr, None)

    # Remove empty tags
    initial_tag_count = len(soup.find_all())
    empty_tags_removed = 0
    for tag in soup.find_all():
        if len(tag.get_text(strip=True)) == 0 and tag.name not in ["br", "hr"]:
            tag.decompose()
            empty_tags_removed += 1
    logger.debug(
        f"Removed {empty_tags_removed} empty tags (kept {initial_tag_count - empty_tags_removed})"
    )

    final_html = str(soup)
    page_title = soup.title.string if soup.title else "scraped_page"

    logger.debug("=== slimdown_html completed ===")
    logger.debug(f"Final HTML size: {len(final_html)} characters")
    logger.debug(f"Page title: {page_title}")
    logger.debug(
        f"Extraction summary - Code: {len(code_examples)}, Methods: {len(method_signatures)}, Classes: {len(class_definitions)}, Images: {len(images)}, Links: {len(links)}"
    )

    return (
        final_html,
        page_title,
        code_examples,
        method_signatures,
        class_definitions,
        images,
        links,
    )


def start_scraping(
    url: str, no_tui: bool = False, quiet: bool = False
) -> Optional[str]:
    """
    Start scraping a URL.

    Args:
        url: The URL to scrape
        no_tui: If True, don't use TUI (legacy parameter)
        quiet: If True, suppress ALL output including spinners (for batch TUI mode)
    """
    url = url.strip()
    if not url:
        if no_tui and not quiet:
            print("Please provide a URL to scrape.")
        return None

    if no_tui and not quiet:
        print(f"Scraping {url}...")

    res = install_playwright()
    if not res:
        if no_tui and not quiet:
            print("Unable to initialize playwright.")
        return None

    scraper = Scraper(playwright_available=res, verify_ssl=False, quiet=quiet)

    try:
        content, mime_type = scraper.scrape(url)
        if content:
            content = f"{url}:\n\n" + content
            # Only print completion message if not in quiet mode (for TUI)
            if no_tui and not quiet:
                print("... done.")
            return content
        else:
            if no_tui and not quiet:
                print("No content retrieved.")
            return None
    except Exception as e:
        if no_tui and not quiet:
            print(f"Error during scraping: {str(e)}")
        return None


# ============================
# Sitemap Processing from DSL
# ============================


def extract_urls_from_sitemap(
    sitemap_file: Optional[str] = None,
    sitemap_content: Optional[str] = None,
    whitelist_str: Optional[str] = None,
    blacklist_str: Optional[str] = None,
) -> List[str]:
    """
    Extracts URLs from a sitemap and filters them based on whitelist and blacklist patterns.
    """
    logger.info("Starting sitemap extraction and filtering.")

    def process_pattern_list(pattern_str: Optional[str]) -> Optional[List[str]]:
        if not pattern_str:
            return None
        return [
            pattern.strip() for pattern in pattern_str.split(",") if pattern.strip()
        ]

    whitelist_patterns = process_pattern_list(whitelist_str)
    blacklist_patterns = process_pattern_list(blacklist_str)

    # Fetch and parse sitemap
    if sitemap_content:
        logger.info("Parsing sitemap content from provided string.")
        try:
            root = DefusedET.fromstring(sitemap_content)
        except ET.ParseError as e:
            logger.error(f"Error parsing sitemap content: {e}")
            return []
    elif sitemap_file:
        logger.info(f"Parsing sitemap from file: {sitemap_file}")
        if not os.path.isfile(sitemap_file):
            logger.error(f"Sitemap file '{sitemap_file}' does not exist.")
            return []
        try:
            tree = DefusedET.parse(sitemap_file)
            root = tree.getroot()
        except ET.ParseError as e:
            logger.error(f"Error parsing sitemap file '{sitemap_file}': {e}")
            return []
    else:
        logger.error("No sitemap content or sitemap file path provided.")
        return []

    # Handle XML namespaces
    namespace = ""
    if root.tag.startswith("{"):
        namespace = root.tag.split("}")[0] + "}"

    urls = []
    for url_elem in root.findall(f".//{namespace}url"):
        loc = url_elem.find(f"{namespace}loc")
        if loc is not None and loc.text:
            url = loc.text.strip()
            include_url = True

            # Apply whitelist
            if whitelist_patterns:
                include_url = any(
                    fnmatch.fnmatch(url, pattern) for pattern in whitelist_patterns
                )

            # Apply blacklist
            if include_url and blacklist_patterns:
                include_url = not any(
                    fnmatch.fnmatch(url, pattern) for pattern in blacklist_patterns
                )

            if include_url:
                urls.append(url)

    logger.info(f"Extracted {len(urls)} URLs from sitemap.")
    return urls


# ============================
# LLM Processing using OpenAI
# ============================


# Load model pricing info
def load_model_pricing() -> Optional[Dict[str, Any]]:
    """Attempt to extract JSON from a string."""
    pricing_urls = [
        "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json",
        "https://cdn.jsdelivr.net/gh/BerriAI/litellm@main/model_prices_and_context_window.json",
        "https://raw.fastgit.org/BerriAI/litellm/main/model_prices_and_context_window.json",
    ]
    for pricing_url in pricing_urls:
        try:
            # Use centralized timeout from config - DO NOT hardcode
            response = requests.get(pricing_url, timeout=HTTP_REQUEST_TIMEOUT)
            response.raise_for_status()
            logger.info(
                f"Successfully fetched model pricing information from {pricing_url}"
            )
            return cast(Dict[str, Any], response.json())
        except requests.RequestException as e:
            logger.warning(
                f"Error fetching model pricing information from {pricing_url}: {e}"
            )
    raise RuntimeError(
        "Failed to fetch model pricing information from all available mirrors."
    )


async def make_openai_request(
    api_key: str,
    prompt: str,
    pricing_info: Dict[str, Dict[str, float]],
    model: str = "gpt-5-nano-2025-08-07",
) -> Dict[str, Any]:
    global total_cost

    # Debug logging: Request details
    logger.debug("=== OpenAI API Request ===")
    logger.debug(f"Model: {model}")
    logger.debug(f"Prompt length: {len(prompt)} characters")
    logger.debug(f"Using structured output with schema: {XML_OUTPUT_SCHEMA}")

    # Calculate proportional timeout based on payload size
    # The timeout must account for: base API response time + retry attempts with exponential backoff
    # Base formula: 120s base + 0.5s per 1K chars (much more realistic for GPT-5)
    # With OPENAI_MAX_RETRIES, OpenAI will make up to (retries+1) attempts with exponential backoff
    # Total timeout = base_time * (1 + 2 + 4) = base_time * 7 (for 3 attempts with 2x backoff)
    # Max: 30 minutes (1800s) to handle very large payloads
    payload_chars = len(prompt)
    base_timeout = min(
        120 + (payload_chars / 1000 * 0.5), 600
    )  # Cap single attempt at 600s (10 min)
    timeout_seconds = min(
        base_timeout * 4, 1800
    )  # 4x for retries with backoff, max 30 min

    logger.debug(
        f"Payload size: {payload_chars} chars, timeout: {timeout_seconds:.1f}s"
    )

    try:
        # Create AsyncOpenAI client with timeout and max_retries
        # Use centralized retry constant from config - DO NOT hardcode
        client = AsyncOpenAI(
            api_key=api_key,
            timeout=timeout_seconds,
            max_retries=OPENAI_MAX_RETRIES,
        )

        logger.debug("Sending request to OpenAI API (library will handle retries)...")

        # Make the async API call with structured output
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert assistant that converts HTML API documentation to structured XML format. You must extract ALL content comprehensively without omissions, following the exact XML structure specifications provided.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "xml_documentation_output",
                    "strict": True,
                    "schema": XML_OUTPUT_SCHEMA,
                },
            },
        )

        logger.debug("=== OpenAI API Response ===")
        logger.debug(f"Response ID: {response.id}")
        logger.debug(f"Model: {response.model}")
        logger.debug(f"Finish reason: {response.choices[0].finish_reason}")

        # Compute cost
        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0

        logger.debug(
            f"Token usage - Prompt: {prompt_tokens}, Completion: {completion_tokens}"
        )

        model_pricing = pricing_info.get(model, {})
        logger.debug(f"Model pricing config: {model_pricing}")
        input_cost = prompt_tokens * model_pricing.get("input_cost_per_token", 0)
        output_cost = completion_tokens * model_pricing.get("output_cost_per_token", 0)
        request_cost = input_cost + output_cost

        logger.debug(
            f"Cost breakdown - Input: ${input_cost:.6f}, Output: ${output_cost:.6f}, Total: ${request_cost:.6f}"
        )

        with cost_lock:
            total_cost += request_cost

        logger.info(f"OpenAI API request successful. Cost: ${request_cost:.6f}")
        logger.info(f"Total cost so far: ${total_cost:.6f}")

        # Convert response to dictionary format for backward compatibility
        response_dict = {
            "id": response.id,
            "object": response.object,
            "created": response.created,
            "model": response.model,
            "choices": [
                {
                    "index": choice.index,
                    "message": {
                        "role": choice.message.role,
                        "content": choice.message.content,
                    },
                    "finish_reason": choice.finish_reason,
                }
                for choice in response.choices
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": usage.total_tokens if usage else 0,
            },
            "request_cost": request_cost,
            "finish_reason": response.choices[0].finish_reason,
        }

        logger.debug(
            f"Response content length: {len(response.choices[0].message.content or '')} characters"
        )

        return response_dict

    except APITimeoutError as e:
        logger.error(
            f"OpenAI API request timed out after {timeout_seconds:.1f}s: {str(e)}"
        )
        raise

    except RateLimitError as e:
        logger.error(f"Rate limit exceeded (429): {str(e)}")
        raise

    except APIConnectionError as e:
        logger.error(f"API connection failed: {str(e)}")
        logger.error(f"Underlying cause: {e.__cause__}")
        raise

    except APIStatusError as e:
        logger.error(f"API returned non-200 status code: {e.status_code}")
        logger.error(f"Response: {e.response}")
        raise

    except OpenAIError as e:
        logger.error(f"OpenAI API error: {str(e)}")
        raise


async def call_openai_api(
    prompt: str, pricing_info: Dict[str, Dict[str, float]]
) -> Tuple[str, float]:
    """
    Makes a request to the OpenAI API using the async OpenAI client.
    Parses the structured JSON response to extract the XML content.
    """
    logger.debug("=== call_openai_api invoked ===")
    logger.debug("Getting OpenAI API key...")
    api_key = get_openai_api_key()
    logger.debug(f"API key retrieved (length: {len(api_key)})")

    logger.debug("Making OpenAI request...")
    response_json = await make_openai_request(
        api_key, prompt, pricing_info=pricing_info
    )
    logger.debug("OpenAI request completed")

    # Parse the structured JSON response
    content_str = str(response_json["choices"][0]["message"]["content"]).strip()
    logger.debug("=== Parsing Structured Response ===")
    logger.debug(f"Raw content string length: {len(content_str)}")
    logger.debug(f"Raw content (first 500 chars): {content_str[:500]}...")

    try:
        structured_response = json.loads(content_str)
        logger.debug("Successfully parsed JSON response")
        logger.debug(f"Structured response keys: {list(structured_response.keys())}")

        # Extract the XML content from the structured response
        xml_content = structured_response.get("xml_content", "")
        document_type = structured_response.get("document_type", "UNKNOWN")

        logger.debug(f"Document type: {document_type}")
        logger.debug(f"XML content length: {len(xml_content)} characters")

        content = xml_content
        logger.debug("Successfully extracted XML content from structured response")
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse structured JSON response: {e}")
        logger.debug(f"Attempted to parse: {content_str[:1000]}...")
        # Fallback to raw content if JSON parsing fails
        content = content_str
        logger.debug("Using raw content as fallback")

    request_cost = response_json.get("request_cost", 0)
    logger.debug(f"Request cost: ${request_cost:.6f}")
    logger.debug("=== call_openai_api completed ===")
    return content, request_cost


def chunk_html_by_size(html_content: str, max_chars: int = 200000) -> List[str]:
    """
    Split HTML content into chunks that won't exceed token limits.
    Tries to split on logical boundaries (doc objects, sections).
    Max chars ~200K = ~85K tokens with safety margin for GPT-5 Nano.
    """
    if len(html_content) <= max_chars:
        return [html_content]

    chunks = []
    soup = BeautifulSoup(html_content, "html.parser")

    # Try to find doc objects (class members, methods, etc.)
    doc_objects = soup.find_all(class_=re.compile(r"doc-object"))

    if doc_objects and len(doc_objects) > 1:
        # Split by doc objects
        chunk_header = str(soup.find("h1")) if soup.find("h1") else ""
        header_size = len(chunk_header)

        # Reserve space for header in each chunk
        effective_max = max_chars - header_size

        current_chunk = ""

        for obj in doc_objects:
            obj_html = str(obj)
            obj_size = len(obj_html)

            # If this single object exceeds the limit, split it further
            if obj_size > effective_max:
                # Save current chunk if not empty
                if current_chunk:
                    chunks.append(chunk_header + current_chunk)
                    current_chunk = ""

                # Try to split the large object by its internal doc-objects or sections
                obj_soup = BeautifulSoup(obj_html, "html.parser")
                sub_objects = obj_soup.find_all(
                    class_=re.compile(r"doc-object|doc-section")
                )

                if sub_objects and len(sub_objects) > 1:
                    # Can split by sub-objects
                    sub_chunk = ""
                    for sub_obj in sub_objects:
                        sub_html = str(sub_obj)
                        if sub_chunk and len(sub_chunk) + len(sub_html) > effective_max:
                            chunks.append(chunk_header + sub_chunk)
                            sub_chunk = ""
                        sub_chunk += sub_html
                    if sub_chunk:
                        chunks.append(chunk_header + sub_chunk)
                else:
                    # Cannot split further, just put it in its own chunk (will exceed limit)
                    logger.warning(
                        f"Single doc-object is {obj_size} chars, exceeding {effective_max} char limit. Placing in its own chunk anyway."
                    )
                    chunks.append(chunk_header + obj_html)
                continue

            # If adding this object would exceed limit, save current chunk
            if current_chunk and len(current_chunk) + obj_size > effective_max:
                chunks.append(chunk_header + current_chunk)
                current_chunk = ""

            current_chunk += obj_html

        # Add final chunk
        if current_chunk:
            chunks.append(chunk_header + current_chunk)
    else:
        # Fallback: split by character count at tag boundaries
        content_parts = re.split(r"(</[^>]+>)", html_content)
        current_chunk = ""

        for part in content_parts:
            if len(current_chunk) + len(part) > max_chars:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
            current_chunk += part

        if current_chunk:
            chunks.append(current_chunk)

    logger.info(f"Split HTML into {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        logger.debug(
            f"Chunk {i + 1} size: {len(chunk)} chars (~{len(chunk) // 3} tokens estimated)"
        )

    return chunks


def merge_duplicate_classes(xml_content: str) -> str:
    """
    Intelligently merge duplicate CLASS elements with the same name.

    When processing large HTML in chunks, a single class may be split across
    multiple chunks, resulting in multiple <CLASS> elements with the same name.
    This function merges those duplicates by combining all their child elements.

    Args:
        xml_content: XML string potentially containing duplicate CLASS elements

    Returns:
        XML string with duplicate classes merged into single elements
    """
    try:
        # Parse the XML content using BeautifulSoup with xml parser
        soup = BeautifulSoup(xml_content, "xml")

        # Find all CLASS elements
        class_elements = soup.find_all("CLASS")

        if len(class_elements) <= 1:
            # No duplicates possible with 0 or 1 class
            return xml_content

        # Group classes by their NAME (the first NAME tag is the class name)
        classes_by_name: Dict[str, List[Any]] = {}
        for class_elem in class_elements:
            # Get the NAME element (first direct child NAME is the class name)
            class_name_elem = class_elem.find("NAME", recursive=False)
            if class_name_elem and class_name_elem.string:
                class_name = class_name_elem.string.strip()
                if class_name not in classes_by_name:
                    classes_by_name[class_name] = []
                classes_by_name[class_name].append(class_elem)

        # Merge duplicate classes
        for class_name, class_list in classes_by_name.items():
            if len(class_list) > 1:
                logger.info(
                    f"Merging {len(class_list)} duplicate CLASS elements for '{class_name}'"
                )

                # Keep the first class as the base
                base_class = class_list[0]

                # Find the CLASS_API section (where methods/properties/fields are stored)
                base_api = base_class.find("CLASS_API")
                if not base_api:
                    # Create CLASS_API section if it doesn't exist
                    base_api = soup.new_tag("CLASS_API")
                    base_class.append(base_api)

                # Merge content from all other duplicate classes
                for duplicate_class in class_list[1:]:
                    # Get the CLASS_API section from duplicate
                    dup_api = duplicate_class.find("CLASS_API")
                    if dup_api:
                        # Move all children from duplicate API to base API
                        for child in list(dup_api.children):
                            if child.name:  # Skip text nodes
                                base_api.append(child)

                    # Also check for any other sections (CLASS_DESCRIPTION, etc.)
                    # and merge them if they exist in duplicate but not in base
                    for child in list(duplicate_class.children):
                        if child.name and child.name not in [
                            "CLASS_NAME",
                            "CLASS_API",
                        ]:
                            # Check if base already has this section
                            if not base_class.find(child.name):
                                base_class.append(child)

                    # Remove the duplicate class element from the tree
                    duplicate_class.decompose()

        # Convert back to string
        merged_xml = str(soup)

        logger.info(
            f"Class merging complete: {len(xml_content)} -> {len(merged_xml)} chars"
        )
        return merged_xml

    except Exception as e:
        logger.error(f"Error merging duplicate classes: {e}")
        # On error, return original content unchanged
        return xml_content


async def call_llm_to_convert_html_to_xml(
    html_content: str,
    additional_content: Dict[str, List[str]],
    pricing_info: Dict[str, Dict[str, float]],
    no_tui: bool = False,
    mock: bool = False,
    tui_manager: Optional[RichTUIManager] = None,
    batch_tui: Optional[BatchTUIManager] = None,
    task_id: Optional[int] = None,
    error_tracker: Optional[SessionErrorTracker] = None,
    url: Optional[str] = None,
) -> Tuple[Optional[str], float, Optional[RichTUIManager]]:
    """
    Uses OpenAI's API to convert HTML content to structured XML asynchronously.
    Implements chunking for large content that exceeds safe token limits.

    Args:
        html_content: HTML content to convert
        additional_content: Additional metadata (code examples, links, etc.)
        pricing_info: Model pricing information
        no_tui: If True, disable Rich TUI output
        mock: If True, use mock API instead of real OpenAI
        batch_tui: Optional BatchTUIManager for status message display
        task_id: Optional task ID for batch TUI updates
    """
    # Reduced chunk size to ~80K chars (~27K tokens worst-case with 1:1 ratio)
    # This ensures each chunk stays well within GPT-5 Nano's safe input limits
    chunks = chunk_html_by_size(html_content, max_chars=80000)

    # Use provided TUI manager or create new one
    if tui_manager is None and not no_tui:
        tui_manager = RichTUIManager(total_chunks=len(chunks), no_tui=no_tui)
    elif tui_manager is not None:
        # CRITICAL: Do NOT reset chunks if they already exist!
        # Resetting would lose all progress tracking (steps go back to QUEUED)
        # Only add new chunks if the chunk count increased
        current_chunk_count = len(tui_manager.chunks)
        new_chunk_count = len(chunks)

        if new_chunk_count > current_chunk_count:
            # Add new chunks for increased count (preserve existing chunks)
            logger.info(
                f"[TUI] Adding {new_chunk_count - current_chunk_count} new chunks to TUI manager"
            )
            for i in range(current_chunk_count + 1, new_chunk_count + 1):
                tui_manager.chunks[i] = ChunkStatus(chunk_id=i)
            tui_manager.stats.total_chunks = new_chunk_count
        # If count is same or decreased, keep existing chunks (preserve progress)

    # Create mock client if mock mode is enabled
    mock_client: Optional[MockAPIClient] = None
    if mock:
        mock_client = MockAPIClient()
        logger.info("Using mock API client for testing")

    # Note: wait_for_start() is called earlier in process_single_page() before scraping
    # So TUI is already displayed and user has already pressed SPACE by the time we get here

    if len(chunks) == 1:
        # Single chunk - process with TUI
        if tui_manager:
            # TUI already started by wait_for_start(), just update status
            # Update chunk as processing
            tui_manager.update_chunk_status(
                chunk_id=1,
                state=ChunkState.PROCESSING,
                size_in=len(html_content),
                attempt=1,
            )

        result = await _process_single_chunk(
            html_content,
            pricing_info,
            chunk_num=1,
            mock=mock,
            mock_client=mock_client,
            batch_tui=batch_tui,
            task_id=task_id,
            error_tracker=error_tracker,
            url=url,
        )

        if tui_manager:
            # Update final status
            if result[0]:  # XML content
                tui_manager.update_chunk_status(
                    chunk_id=1,
                    state=ChunkState.COMPLETE,
                    size_in=len(html_content),
                    size_out=len(result[0]),
                    cost=result[1],
                    attempt=1,
                )
            else:
                tui_manager.update_chunk_status(
                    chunk_id=1,
                    state=ChunkState.FAILED,
                    size_in=len(html_content),
                    error="Processing failed",
                    attempt=1,
                )
            tui_manager.stop_live_display()

        return result[0], result[1], tui_manager

    # Multiple chunks - process in parallel using asyncio
    logger.info(
        f"Processing {len(chunks)} chunks in parallel with true async concurrency..."
    )

    # TUI already started by wait_for_start(), no need to start again

    async def process_chunk_wrapper(
        args: Tuple[int, str],
    ) -> Tuple[int, Optional[str], float]:
        """Async wrapper function for parallel processing"""
        i, chunk = args
        logger.info(f"Processing chunk {i}/{len(chunks)}...")

        # Update TUI: mark as processing
        if tui_manager:
            tui_manager.update_chunk_status(
                chunk_id=i, state=ChunkState.PROCESSING, size_in=len(chunk), attempt=1
            )

        xml_part, cost = await _process_single_chunk(
            chunk,
            pricing_info,
            chunk_num=i,
            mock=mock,
            mock_client=mock_client,
            batch_tui=batch_tui,
            task_id=task_id,
            error_tracker=error_tracker,
            url=url,
        )

        if xml_part:
            # Extract just the content, strip wrapper tags
            xml_part = xml_part.strip()
            if xml_part.startswith("<?xml"):
                # Remove XML declaration
                xml_part = re.sub(r"<\?xml[^>]+\?>\s*", "", xml_part)
            if xml_part.startswith("<XML>"):
                xml_part = re.sub(r"^<XML>\s*", "", xml_part)
                xml_part = re.sub(r"\s*</XML>$", "", xml_part)

            # Update TUI: mark as complete
            if tui_manager:
                tui_manager.update_chunk_status(
                    chunk_id=i,
                    state=ChunkState.COMPLETE,
                    size_in=len(chunk),
                    size_out=len(xml_part),
                    cost=cost,
                    attempt=1,
                )

            return i, xml_part, cost
        else:
            logger.error(f"Failed to process chunk {i}/{len(chunks)}")

            # Update TUI: mark as failed
            if tui_manager:
                tui_manager.update_chunk_status(
                    chunk_id=i,
                    state=ChunkState.FAILED,
                    size_in=len(chunk),
                    error="Processing failed",
                    attempt=1,
                )

            return i, None, cost

    # Prepare chunks with their indices (no delays - truly parallel)
    chunk_args = [(i, chunk) for i, chunk in enumerate(chunks, 1)]

    # Process chunks in parallel using asyncio.gather for true async concurrency
    tasks = [process_chunk_wrapper(args) for args in chunk_args]
    try:
        results = await asyncio.gather(*tasks, return_exceptions=False)
    except Exception as e:
        logger.error(f"Error processing chunks: {e}")
        results = []

    # Sort results by chunk index to maintain order
    results.sort(key=lambda x: x[0])

    # Extract XML parts and calculate total cost
    all_xml_parts = []
    total_cost = 0.0
    for _i, xml_part, cost in results:
        if xml_part:
            all_xml_parts.append(xml_part)
        total_cost += cost

    if not all_xml_parts:
        return None, total_cost, tui_manager

    # Merge all XML parts into a valid XML document
    # Wrap all chunks in a root element to create valid XML structure
    merged_xml = f"<XML>\n{''.join(all_xml_parts)}\n</XML>"
    logger.info(
        f"Merged {len(all_xml_parts)} chunks into final XML ({len(merged_xml)} chars)"
    )

    # Intelligently merge duplicate CLASS elements that were split across chunks
    merged_xml = merge_duplicate_classes(merged_xml)

    # Stop TUI live display if it was started
    if tui_manager:
        tui_manager.stop_live_display()

    return merged_xml, total_cost, tui_manager


async def _process_single_chunk(
    html_content: str,
    pricing_info: Dict[str, Dict[str, float]],
    chunk_num: Optional[int] = None,
    mock: bool = False,
    mock_client: Optional[MockAPIClient] = None,
    batch_tui: Optional[BatchTUIManager] = None,
    task_id: Optional[int] = None,
    error_tracker: Optional[SessionErrorTracker] = None,
    url: Optional[str] = None,
) -> Tuple[Optional[str], float]:
    """Process a single chunk of HTML content asynchronously with automatic XML validation retry."""

    chunk_label = f" (chunk {chunk_num})" if chunk_num else ""

    # Simplified prompt optimized for chunked processing
    # Focuses on extracting ALL content from THIS chunk without confusing "completeness" requirements
    base_prompt = f"""You are extracting API documentation from HTML content into structured XML format.

The HTML content describes the Python Textual library API with detailed information about classes, modules, functions, or usage examples.

YOUR TASK:
Extract ALL information from the provided HTML content completely and accurately.

EXCLUDE:
- Navigation menus, side menus, indexes
- Images, base64-encoded data
- External links (remove href attributes)

PRESERVE AND EXTRACT:
- All API documentation content
- Method/function signatures with parameter names and types
- Return types and type annotations
- Class descriptions and member documentation
- Code examples and usage patterns
- Tables (convert to XML structures)
- All descriptive text and documentation

CONTENT CLASSIFICATION:
Classify the content and wrap it in the appropriate root tag:

1. <CLASS>: Class documentation
   - Include <CLASS_DESCRIPTION> and <CLASS_API> sections
   - Each method/property/field as separate elements with full signatures

2. <MODULE>: Module documentation
   - Include <FUNCTION>, <VARIABLE>, <CONSTANT> elements
   - Each with name, signature, and modifiers

3. <API_USAGE>: Tutorials, guides, examples
   - Organize with <SUB_SECTION> elements
   - Each with <TITLE> and content

SEMANTIC CONTENT ANALYSIS (Website-Agnostic Classification):
CRITICAL: Different documentation sites use different HTML structures and CSS naming.
DO NOT rely on specific CSS class names (they vary by site: "doc-class", "py-class", "h876", etc.).
Instead, analyze the SEMANTIC CONTENT to determine the element type.

**UNIVERSAL INDICATORS FOR PYTHON CLASSES:**

1. **Inheritance Keywords** (Primary Signal):
   - Text contains: "Bases:", "Inherits:", "Extends:", "Parent:", "Superclass:"
   - Example: "Bases: Exception", "Inherits: BaseClass"
   - This is THE STRONGEST indicator of a Python class

2. **Class Signature Patterns**:
   - Contains: `class ClassName:`, `class ClassName(Parent):`, `class ClassName(`
   - Python keyword "class" at start of signature
   - CamelCase or CapitalizedName (PEP 8 convention)

3. **Contains Methods** (Structural Signal):
   - Has multiple child elements with `def method(self, ...)`
   - Methods have `self` as first parameter
   - Contains `__init__`, `__new__`, or other dunder methods

4. **Capitalization Convention**:
   - Name starts with uppercase letter: App, Screen, ActionError
   - Follows PEP 8 class naming (not snake_case)

**DECISION LOGIC:**
```
IF (has inheritance keyword) OR
   (signature contains "class ClassName") OR
   (has methods with self parameter AND capitalized name)
THEN → <CLASS>

ELSE IF (has "def name(self, ...)" AND inside a class)
THEN → <METHOD>

ELSE IF (has "@property" OR getter/setter pattern)
THEN → <PROPERTY>

ELSE IF (has "def name(...)" WITHOUT self)
THEN → <FUNCTION>

ELSE IF (has "var = value" OR "var: Type = value")
THEN → <VARIABLE>

ELSE → Analyze context and infer from surrounding elements
```

**EXAMPLES OF SEMANTIC ANALYSIS:**

✅ **CLASS** (recognized by inheritance):
```
Name: ActionError
Text: "Bases: Exception. Base class for exceptions..."
→ Contains "Bases:" → This is a CLASS
```

✅ **CLASS** (recognized by signature):
```
Signature: class App(Generic[ReturnType], DOMNode)
→ Contains "class" keyword → This is a CLASS
```

✅ **METHOD** (recognized by self parameter):
```
Signature: def render(self) -> RenderResult
→ Has "self" parameter → This is a METHOD
```

✅ **PROPERTY** (recognized by decorator):
```
Signature: @property
def active_bindings(self) -> dict
→ Has @property decorator → This is a PROPERTY
```

✅ **FUNCTION** (recognized by no self):
```
Signature: def run_app(app_class) -> int
→ No "self" parameter → This is a FUNCTION
```

✅ **VARIABLE** (recognized by assignment):
```
Signature: ScreenType = TypeVar('ScreenType', bound=Screen)
→ Has "=" assignment → This is a VARIABLE
```

**CRITICAL RULES:**
1. DO NOT rely on CSS class names - they vary by site
2. DO analyze text content for semantic keywords
3. DO look for Python syntax patterns (class, def, self, @property)
4. DO use inheritance/bases as PRIMARY class indicator
5. DO NOT create <UNKNOWN_ELEMENT> - infer from context instead

XML TAG SPECIFICATIONS:
Use the correct XML tags based on the type of API element:

FOR CLASS MEMBERS:
- <METHOD>: Use for class/instance methods (functions that belong to a class)
  * MUST include <RETURN_TYPE> tag with the return type
  * Example: async def render(self) -> RenderResult

- <PROPERTY>: Use ONLY for @property decorators (getter/setter accessors)
  * MUST include <RETURN_TYPE> tag with the property's type
  * Example: @property def active_bindings(self) -> ActiveBindings
  * NOT for regular methods or fields

- <FIELD>: Use for class member variables (class attributes or instance variables)
  * Include type annotation if available
  * Example: title: str = "My App"
  * NOT for methods or properties

FOR MODULE-LEVEL ELEMENTS:
- <FUNCTION>: Use for standalone functions (not class methods)
  * MUST include <RETURN_TYPE> tag with the return type
  * Example: def main() -> int

- <VARIABLE>: Use for module-level variables and constants
  * Include type in <SIGNATURE> tag only
  * DO NOT create separate <TYPES> or <TYPE> tags
  * Example: <SIGNATURE>AutopilotCallbackType = Callable[[Pilot[object]], Coroutine[Any, Any, None]]</SIGNATURE>

IMPORTANT DISTINCTIONS:
1. Methods vs Properties:
   - Methods are callable functions: obj.method()
   - Properties are accessed like attributes: obj.property (no parentheses)

2. Fields vs Properties:
   - Fields are simple data attributes
   - Properties have @property decorator and may have logic

3. Functions vs Methods:
   - Functions are standalone (module-level)
   - Methods belong to classes (have self/cls parameter)

4. Return Types:
   - Always extract and include return type information
   - Place in <RETURN_TYPE> tag, not in signature repetition
   - If return type is missing from docs, infer from context when possible

EXAMPLES OF CORRECT TAGGING:

Correct CLASS METHOD with PARAMETERS:
<METHOD>
  <NAME>action_add_class</NAME>
  <SIGNATURE>async action_add_class(selector, class_name)</SIGNATURE>
  <RETURN_TYPE>None</RETURN_TYPE>
  <DESCRIPTION>An action to add a CSS class to the selected widget.</DESCRIPTION>
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

CRITICAL: Use <PARAMETER> (singular) for each parameter inside <PARAMETERS> (plural).
DO NOT use <PARAM> - always use the full tag name <PARAMETER>.

Correct PROPERTY (NOT "attribute"):
<PROPERTY>
  <NAME>active_bindings</NAME>
  <SIGNATURE>@property active_bindings</SIGNATURE>
  <RETURN_TYPE>ActiveBindings</RETURN_TYPE>
  <DESCRIPTION>Get currently active bindings.</DESCRIPTION>
</PROPERTY>

Correct FIELD:
<FIELD>
  <NAME>title</NAME>
  <SIGNATURE>title: str</SIGNATURE>
  <DESCRIPTION>The title of the application.</DESCRIPTION>
</FIELD>

Correct MODULE FUNCTION:
<FUNCTION>
  <NAME>run_app</NAME>
  <SIGNATURE>def run_app(app_class)</SIGNATURE>
  <RETURN_TYPE>int</RETURN_TYPE>
  <DESCRIPTION>Run the application.</DESCRIPTION>
</FUNCTION>

Correct MODULE VARIABLE (NO separate TYPES tag):
<VARIABLE name='AutopilotCallbackType' modifiers='module-attribute'>
  <SIGNATURE>AutopilotCallbackType = Callable[[Pilot[object]], Coroutine[Any, Any, None]]</SIGNATURE>
  <DESCRIPTION>Signature for valid callbacks that can be used to control apps.</DESCRIPTION>
</VARIABLE>

INCORRECT EXAMPLES (DO NOT DO THIS):

Wrong - Method labeled as ATTRIBUTE:
<ATTRIBUTE>
  <NAME>render</NAME>
  <SIGNATURE>render()</SIGNATURE>
</ATTRIBUTE>

Wrong - Missing RETURN_TYPE:
<METHOD>
  <NAME>render</NAME>
  <SIGNATURE>render()</SIGNATURE>
  <TYPES>RenderResult</TYPES>  <!-- Should be <RETURN_TYPE> -->
</METHOD>

Wrong - Variable with redundant TYPES tag:
<VARIABLE>
  <SIGNATURE>my_var = 5</SIGNATURE>
  <TYPES>int</TYPES>  <!-- Don't add separate TYPES tag -->
</VARIABLE>

REQUIREMENTS:
- Explicitly indicate types, return types, and modifiers when available
- If type information is implicit or can be inferred from context, make it explicit in the XML
- Use descriptive element names that reflect the content
- Properly escape XML special characters in text content and attribute values:
  - & → &amp;
  - < → &lt;
  - > → &gt;
  - " → &quot;
  - ' → &apos;

EXAMPLE ESCAPING:
Wrong: <DESCRIPTION>Grow space by (<top>, <right>, <bottom>, <left>).</DESCRIPTION>
Right: <DESCRIPTION>Grow space by (&lt;top&gt;, &lt;right&gt;, &lt;bottom&gt;, &lt;left&gt;).</DESCRIPTION>

OUTPUT FORMAT:
Your response must be ONLY a valid JSON object with this structure:
```json
{{
  "xml_content": "<complete XML document as string>",
  "document_type": "CLASS|MODULE|API_USAGE"
}}
```

The xml_content field must contain the complete XML document with all extracted content.
No additional commentary or explanations - ONLY the JSON object.

<TEXT>
{html_content}
</TEXT>"""

    total_cost = 0.0
    # Use centralized retry constant from config - DO NOT hardcode
    max_retries = XML_VALIDATION_MAX_RETRIES

    for attempt in range(max_retries + 1):
        try:
            # Add XML validation instructions on retry
            if attempt > 0:
                prompt = f"""{base_prompt}

CRITICAL XML VALIDATION ERROR ON PREVIOUS ATTEMPT:
The XML generated in the previous attempt had invalid structure.

COMMON XML ERRORS TO AVOID:
1. Tag Mismatch: Closing tags MUST match opening tags exactly
   - Wrong: <PARAMETER>...</PARAMETERS> (singular open, plural close)
   - Right: <PARAMETER>...</PARAMETER> (both singular)

2. Nested Tags: Ensure proper nesting without orphaned closing tags
   - Wrong: </PARAMETERS></PARAMETERS></PARAMETERS> (too many closing tags)
   - Right: </PARAMETER></PARAMETERS> (correct nesting)

3. Singular vs Plural:
   - <PARAMETER> (singular) - for individual parameter
   - <PARAMETERS> (plural) - for parameter container
   - <METHOD> (singular) - for individual method
   - <METHODS> (plural) - for methods container

PLEASE REGENERATE THE XML WITH CAREFUL ATTENTION TO TAG MATCHING."""
            else:
                prompt = base_prompt

            logger.debug(
                f"Processing{chunk_label} (attempt {attempt + 1}/{max_retries + 1}): {len(html_content)} chars"
            )

            # Use mock or real API based on flag
            if mock and mock_client:
                xml_output, request_cost = await mock_call_openai_api(
                    prompt, pricing_info, mock_client
                )
            else:
                xml_output, request_cost = await call_openai_api(prompt, pricing_info)

            total_cost += request_cost

            # Check if API returned None
            if xml_output is None:
                raise ValueError("LLM returned no content")

            # Validate XML
            xml_output = extract_xml_from_input(xml_output)

            # Success! XML is valid
            # Record success in error tracker to reset consecutive error count
            # This prevents false circuit breaker trips from intermittent errors
            if error_tracker:
                error_tracker.record_success()

            logger.info(
                f"Completed{chunk_label}: {len(xml_output) if xml_output else 0} chars XML, cost ${total_cost:.6f}"
                + (" (succeeded on retry)" if attempt > 0 else "")
            )
            return xml_output, total_cost

        except ValueError as e:
            # XML validation error
            if attempt < max_retries:
                logger.warning(
                    f"XML validation failed{chunk_label} (attempt {attempt + 1}): {e}. Retrying with corrective instructions..."
                )
                # Show retry message in batch TUI
                update_batch_status(
                    batch_tui,
                    task_id,
                    URLState.PROCESSING,
                    "🔄 LLM returned invalid XML. Retrying...",
                )
                continue  # Try again with validation instructions
            else:
                logger.error(
                    f"XML validation failed{chunk_label} after {max_retries + 1} attempts: {e}"
                )
                # Show final failure message in batch TUI
                update_batch_status(
                    batch_tui,
                    task_id,
                    URLState.PROCESSING,
                    "❌ XML validation failed after retries.",
                )
                return None, total_cost

        except Exception as e:
            # Classify the error for tracking and summary reporting
            # This helps users understand WHY their job failed (quota, rate limit, etc.)
            error_category = classify_openai_error(e)
            error_desc = get_error_description(error_category)

            # Record in error tracker if available
            if error_tracker:
                # Use centralized is_recoverable_error() - DO NOT hardcode category checks
                # Recoverable errors can be retried; non-recoverable require user action
                recoverable = is_recoverable_error(error_category)
                error_event = ErrorEvent(
                    category=error_category,
                    message=str(e),
                    task_id=task_id,
                    url=url,
                    recoverable=recoverable,
                    raw_exception=type(e).__name__,
                )
                # Record error - circuit breaker may trigger to stop wasting API credits
                should_stop = error_tracker.record(error_event)
                if should_stop:
                    # Circuit breaker tripped - log prominently so user knows processing stopped
                    logger.warning(
                        f"Circuit breaker triggered: {error_tracker.circuit_breaker.trigger_reason}"
                    )

            # Log with classified error type for debugging
            logger.error(
                f"LLM processing failed{chunk_label}: [{error_category.name}] {e}"
            )

            # Show API error message in batch TUI with classified error
            msg = f"❌ {error_desc}"
            update_batch_status(batch_tui, task_id, URLState.PROCESSING, msg)
            return None, total_cost

    # Should never reach here, but just in case
    return None, total_cost


# ============================
# Web Scraper Replacement
# ============================


def web_scraper(url: str, no_tui: bool = False, quiet: bool = False) -> Optional[str]:
    """
    Web scraper with optional output suppression.

    Args:
        url: URL to scrape
        no_tui: If True, disable TUI (legacy)
        quiet: If True, suppress all output including spinners (for batch TUI)
    """
    logger.debug("=== web_scraper invoked ===")
    logger.debug(f"Target URL: {url}")
    try:
        logger.debug("Starting scraping process...")
        content = start_scraping(url, no_tui=no_tui, quiet=quiet)
        if content:
            content_size = len(content) / 1024  # Convert to KB
            logger.info(
                f"Scraping successful for {url}. {content_size:.2f} KB scraped."
            )
            logger.debug(f"Content preview (first 200 chars): {content[:200]}...")
            logger.debug("=== web_scraper completed successfully ===")
            return content
        else:
            logger.error(f"No content retrieved from {url}")
            logger.debug("=== web_scraper failed (no content) ===")
            return None
    except Exception as e:
        logger.error(f"Error scraping {url}: {str(e)}", exc_info=True)
        logger.debug("=== web_scraper failed (exception) ===")
        return None


# ============================
# Main Workflow Functions
# ============================

# ============================
# Sitemap Processing Functions
# ============================


def fetch_sitemap(urls: List[str], quiet: bool = False) -> Optional[str]:
    """
    Fetches the sitemap.xml from the base domain of the given URLs using HTTP requests.

    Args:
        urls: List of URLs to fetch sitemap from
        quiet: If True, suppress Spinner output (for batch TUI mode)
    """
    for url in urls:
        parsed_url = urlparse(url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        sitemap_url = base_url.rstrip("/") + "/sitemap.xml"
        logger.info(f"Fetching sitemap from {sitemap_url}")
        try:
            with Spinner("Fetching sitemap...", quiet=quiet) as spinner:
                # Use centralized timeout from config - DO NOT hardcode
                response = requests.get(sitemap_url, timeout=HTTP_REQUEST_TIMEOUT)
                response.raise_for_status()
            logger.info("Fetched sitemap successfully.")
            sitemap_content: str = response.text
            return sitemap_content
        except requests.RequestException as e:
            logger.error(f"Failed to fetch sitemap.xml from {sitemap_url}: {e}")
    logger.error("Failed to fetch sitemap.xml from all provided URLs")
    return None


# ============================
# Main Workflow Functions
# ============================


def process_single_page(
    url: str,
    pricing_info: Dict[str, Dict[str, float]],
    scrape_only: bool = False,
    no_tui: bool = False,
    mock: bool = False,
) -> Optional[str]:
    """
    Processes a single page: scrapes, converts to XML via LLM, and saves the result.
    Note: slimdown_html is already called inside web_scraper/Scraper.scrape(),
    so we don't call it again here.

    Args:
        url: URL to process
        pricing_info: Model pricing information
        scrape_only: If True, only scrape without AI processing
        no_tui: If True, disable Rich TUI output
        mock: If True, use mock API instead of real OpenAI
    """
    logger.info("Processing single page.")
    if scrape_only:
        logger.info("Scrape-only mode: AI processing will be skipped")
    logger.debug("=== process_single_page invoked ===")
    logger.debug(f"URL: {url}")

    class ProcessingResult:
        def __init__(self) -> None:
            self.html_content: Optional[str] = None
            self.xml_content: Optional[str] = None
            self.error: Optional[str] = None
            self.tui_manager: Optional[RichTUIManager] = None

    result = ProcessingResult()

    # Create TUI manager FIRST (before any processing)
    # We'll use 1 chunk as placeholder since we don't know yet how many chunks there will be
    if not no_tui and not scrape_only:
        result.tui_manager = RichTUIManager(total_chunks=1, no_tui=no_tui)

        # Display waiting screen and wait for SPACE press
        result.tui_manager.wait_for_start()

        # Check if user cancelled before starting
        if result.tui_manager.should_stop:
            logger.info("Processing cancelled by user before start")
            result.tui_manager.stop_live_display()
            return None

    async def process_in_background_async() -> None:
        try:
            # Update step: Scraping
            if result.tui_manager:
                result.tui_manager.update_chunk_status(
                    chunk_id=1,
                    state=ChunkState.PROCESSING,
                    step=ProcessingStep.SCRAPING,
                )

            logger.debug("Step 1: Scraping HTML content (includes slimdown)...")
            # Run the synchronous web_scraper in a separate thread to avoid Playwright sync/async conflicts
            result.html_content = await asyncio.to_thread(web_scraper, url, no_tui)
            if not result.html_content:
                result.error = (
                    "Failed to retrieve HTML content for single page processing."
                )
                logger.debug("Step 1 FAILED: No HTML content retrieved")
                return
            logger.debug(
                f"Step 1 SUCCESS: Retrieved {len(result.html_content)} characters (already slimmed)"
            )

            # Update step: Cleaning (already done by scraper, but mark as complete)
            if result.tui_manager:
                result.tui_manager.update_chunk_status(
                    chunk_id=1,
                    state=ChunkState.PROCESSING,
                    step=ProcessingStep.CLEANING,
                )

            # Note: slimdown_html was already called in Scraper.scrape()
            # The content is already cleaned, so we use it directly
            slimmed_html = result.html_content

            # Save the cleaned HTML in scrape-only mode
            if scrape_only:
                # Create a URL slug for the filename using urlparse (already imported at module level)
                parsed_url = urlparse(url)
                url_slug = parsed_url.path.strip("/").replace("/", "_") or "index"
                if not url_slug or url_slug == "":
                    url_slug = "index"

                html_file = temp_folder / f"{url_slug}_cleaned.html"
                with open(html_file, "w", encoding="utf-8") as f:
                    f.write(slimmed_html)
                logger.info(f"Scrape-only mode: Saved cleaned HTML to {html_file}")
                logger.debug("=== process_single_page completed (scrape-only) ===")
                return

            # Extract additional metadata from the HTML for context
            # We still need to parse for code examples, etc.
            soup = BeautifulSoup(slimmed_html, "html.parser")
            page_title = (
                soup.find("title").get_text(strip=True)
                if soup.find("title")
                else "Unknown"
            )

            # Extract code examples
            code_examples = []
            for code_tag in soup.find_all("code"):
                code_text = code_tag.get_text(strip=True)
                if code_text:
                    code_examples.append(code_text)

            # Extract links
            links = []
            for link in soup.find_all("a", href=True):
                href = link["href"]
                text = link.get_text(strip=True)
                if text:
                    links.append((href, text))

            # Prepare additional content for LLM
            additional_content: Dict[str, List[str]] = {
                "code_examples": code_examples,
                "method_signatures": [],  # Already extracted in slimdown_html
                "class_definitions": [],  # Already extracted in slimdown_html
                "images": [],  # Already extracted in slimdown_html
                "links": [f"{href}: {text}" for href, text in links],
            }
            logger.debug(
                f"Step 2: Extracted metadata - Title: {page_title}, Code: {len(code_examples)}, Links: {len(links)}"
            )

            # Update step: Chunking (for single page, this is trivial but marks progress)
            if result.tui_manager:
                result.tui_manager.update_chunk_status(
                    chunk_id=1,
                    state=ChunkState.PROCESSING,
                    step=ProcessingStep.CHUNKING,
                )

            # Update step: Sending to AI
            if result.tui_manager:
                result.tui_manager.update_chunk_status(
                    chunk_id=1, state=ChunkState.PROCESSING, step=ProcessingStep.SENDING
                )

            logger.debug("Step 3: Converting HTML to XML via LLM (GPT-5 Nano)...")
            llm_result = await call_llm_to_convert_html_to_xml(
                slimmed_html,
                additional_content,
                pricing_info,
                no_tui=no_tui,
                mock=mock,
                tui_manager=result.tui_manager,
            )
            if llm_result is None:
                result.error = "Failed to convert HTML to XML."  # type: ignore[unreachable]
                logger.debug("Step 3 FAILED: LLM conversion returned None")
                return

            # Don't replace tui_manager - keep using the same one the main thread references
            xml_content, _, _ = llm_result
            logger.debug(
                f"Step 3 SUCCESS: Generated XML content ({len(xml_content) if xml_content else 0} characters)"
            )

            # Update step: Receiving from AI
            if result.tui_manager:
                result.tui_manager.update_chunk_status(
                    chunk_id=1,
                    state=ChunkState.PROCESSING,
                    step=ProcessingStep.RECEIVING,
                )

            if xml_content:
                # Update step: Validating XML
                if result.tui_manager:
                    result.tui_manager.update_chunk_status(
                        chunk_id=1,
                        state=ChunkState.PROCESSING,
                        step=ProcessingStep.VALIDATING,
                    )

                # Strip any XML declaration and root <XML> tag from LLM output (we'll add our own)
                import re

                xml_content = re.sub(
                    r'<\?xml\s+version="[^"]+"\s+encoding="[^"]+"\?>\s*',
                    "",
                    xml_content,
                    count=1,
                )
                xml_content = re.sub(
                    r"^\s*<XML>\s*", "", xml_content, count=1
                )  # Remove opening <XML>
                xml_content = re.sub(
                    r"\s*</XML>\s*$", "", xml_content, count=1
                )  # Remove closing </XML>
                # Wrap XML content with proper declaration and source URL
                result.xml_content = f'<?xml version="1.0" encoding="UTF-8"?>\n<XML>\n<SOURCE_URL>{url}</SOURCE_URL>\n{xml_content}\n</XML>'

                # Update step: Saving XML
                if result.tui_manager:
                    result.tui_manager.update_chunk_status(
                        chunk_id=1,
                        state=ChunkState.PROCESSING,
                        step=ProcessingStep.SAVING,
                    )

                logger.debug("Step 4: Saving XML to file...")

                # Save individual XML file
                xml_file = temp_folder / "processed_single_page.xml"
                with open(xml_file, "w", encoding="utf-8") as f:
                    f.write(result.xml_content)
                logger.debug(f"Step 4 SUCCESS: Saved XML to {xml_file}")
        except Exception as e:
            result.error = f"Error processing page: {str(e)}"
            logger.debug(f"Background processing FAILED with exception: {str(e)}")

    def process_in_background() -> None:
        """Wrapper to run async function in sync context"""
        asyncio.run(process_in_background_async())

    logger.debug("Starting background processing thread...")
    # Use Spinner only when TUI is disabled (otherwise TUI handles progress display)
    if no_tui or scrape_only:
        with Spinner("Processing page...") as spinner:
            thread = threading.Thread(target=process_in_background)
            thread.start()
            while thread.is_alive():
                spinner.step()
                # Use centralized spinner animation interval - DO NOT hardcode timing values
                time.sleep(SPINNER_ANIMATION_INTERVAL)
            thread.join()
    else:
        # TUI mode - run thread and keep updating TUI while it's alive
        thread = threading.Thread(target=process_in_background)
        thread.start()

        logger.debug(
            f"TUI update loop starting. TUI manager exists: {result.tui_manager is not None}, Live widget exists: {result.tui_manager.live is not None if result.tui_manager else False}"
        )

        update_count = 0
        while thread.is_alive():
            # Update TUI display while background thread is working
            if result.tui_manager and result.tui_manager.live:
                result.tui_manager.live.update(result.tui_manager._create_dashboard())
                update_count += 1
                if update_count % 50 == 0:  # Log every 5 seconds
                    logger.debug(f"TUI updated {update_count} times")
            # Use centralized TUI polling interval - DO NOT hardcode timing values
            time.sleep(SINGLE_PAGE_TUI_POLL_INTERVAL)
        thread.join()

        logger.debug(f"Thread finished. Total TUI updates: {update_count}")

        # Final update to show completed state
        if result.tui_manager and result.tui_manager.live:
            logger.debug("Performing final TUI update")
            result.tui_manager.live.update(result.tui_manager._create_dashboard())
            # Use centralized pause duration - DO NOT hardcode timing values
            time.sleep(FINAL_STATE_PAUSE)
        else:
            logger.debug(
                f"No final TUI update - TUI manager exists: {result.tui_manager is not None}, Live exists: {result.tui_manager.live is not None if result.tui_manager else False}"
            )
    logger.debug("Background processing thread completed")

    if result.error:
        logger.error(result.error)
        logger.debug("=== process_single_page FAILED ===")
        return None

    # In scrape-only mode, return success without XML
    if scrape_only:
        logger.info("Scrape-only mode: Successfully completed without XML generation")
        return "scrape_only_completed"

    if result.xml_content is not None:
        logger.debug("Step 5: Merging XML files...")
        merged_xml = merge_xmls(temp_folder)
        merged_xml_file = temp_folder / "merged_output.xml"
        with open(merged_xml_file, "w", encoding="utf-8") as f:
            f.write(merged_xml)
        logger.info(f"Merged XML saved to {merged_xml_file}")
        logger.debug(
            f"Step 5 SUCCESS: Merged XML file created ({len(merged_xml)} characters)"
        )

        # Show final TUI summary if TUI manager exists
        if result.tui_manager:
            xml_file = temp_folder / "processed_single_page.xml"
            result.tui_manager.show_final_summary(
                xml_files=[str(xml_file), str(merged_xml_file)],
                output_dir=str(temp_folder),
            )

        logger.debug("=== process_single_page completed successfully ===")
        return result.xml_content

    logger.error("Failed to convert HTML to XML. No XML content generated.")
    logger.debug("=== process_single_page FAILED (no XML content) ===")
    return None


def process_url(
    url: str,
    idx: int,
    total: int,
    pricing_info: Dict[str, Dict[str, float]],
    scrape_only: bool = False,
    batch_tui: Optional[BatchTUIManager] = None,
    error_tracker: Optional[SessionErrorTracker] = None,
) -> Optional[str]:
    global progress_tracker
    """
    Process a single URL: scrape, convert to XML, and save temp files.
    """
    global shutdown_flag, progress_tracker
    logger.info(f"Processing URL {idx}/{total}: {url}")
    if scrape_only:
        logger.info(f"Scrape-only mode: Skipping AI processing for URL {idx}/{total}")
    try:
        if shutdown_flag:
            logger.info(f"Shutdown requested. Skipping URL {url}")
            return None

        if url not in progress_tracker:
            progress_tracker[url] = {"status": "pending", "cost": 0.0}
        elif progress_tracker[url].get("status") == "successful":
            logger.info(f"URL {url} already processed successfully. Skipping.")
            return None

        update_progress_file()

        # Update TUI: Starting scraping
        if batch_tui:
            batch_tui.update_task(idx, URLState.SCRAPING, progress_pct=10.0)

        logger.info(f"Scraping HTML content for URL: {url}")
        # Suppress spinner output when batch TUI is active
        html_content = web_scraper(url, no_tui=True, quiet=bool(batch_tui))
        if not html_content:
            logger.warning(f"Failed to retrieve HTML content for {url}")
            progress_tracker[url] = {
                "status": "failed",
                "cost": progress_tracker[url].get("cost", 0.0),
            }
            with cost_lock:
                update_progress_file()
            update_batch_status(
                batch_tui,
                idx,
                URLState.FAILED,
                "❌ Source page not found. Aborting task.",
                progress_pct=0.0,
                error="Failed to retrieve HTML",
            )
            return None

        logger.info(f"Saving HTML content for URL: {url}")
        html_file = temp_folder / f"scraped_{idx}.html"
        with open(html_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        # In scrape-only mode, skip LLM processing
        if scrape_only:
            logger.info(
                f"Scrape-only mode: HTML saved for URL {url}, skipping XML generation"
            )
            progress_tracker[url] = {
                "status": "successful",
                "cost": 0.0,
            }
            update_progress_file()
            if batch_tui:
                batch_tui.update_task(
                    idx,
                    URLState.COMPLETE,
                    progress_pct=100.0,
                    size_in=len(html_content),
                )
            return "scrape_only_completed"

        if shutdown_flag:
            logger.info(f"Shutdown requested. Skipping LLM processing for URL {url}")  # type: ignore[unreachable]
            progress_tracker[url]["status"] = "pending"
            update_progress_file()
            return None

        # Update TUI: Starting LLM processing
        if batch_tui:
            batch_tui.update_task(
                idx, URLState.PROCESSING, progress_pct=30.0, size_in=len(html_content)
            )

        logger.info(f"Converting HTML to XML for URL: {url}")
        additional_content: Dict[str, List[str]] = {}
        result = asyncio.run(
            call_llm_to_convert_html_to_xml(
                html_content,
                additional_content,
                pricing_info,
                batch_tui=batch_tui,
                task_id=idx,
                error_tracker=error_tracker,
                url=url,
            )
        )
        if result is None:
            logger.warning(f"Failed to convert HTML to XML for {url}")  # type: ignore[unreachable]
            progress_tracker[url] = {
                "status": "failed",
                "cost": progress_tracker[url].get("cost", 0.0),
            }
            update_progress_file()
            update_batch_status(
                batch_tui,
                idx,
                URLState.FAILED,
                "❌ LLM processing failed. Aborting task.",
                progress_pct=0.0,
                error="Failed to convert to XML",
            )
            return None

        xml_content, request_cost, _ = (
            result  # Unpack all 3 values (tui_manager not used in batch mode)
        )

        with cost_lock:
            progress_tracker[url] = {
                "status": progress_tracker[url].get("status", "pending"),
                "cost": float(progress_tracker[url].get("cost", 0.0)) + request_cost,
            }

        # Include source URL in XML content
        if xml_content is not None:
            xml_content = xml_content.replace(
                '<?xml version="1.0" encoding="UTF-8"?>\n<XML>',
                f'<?xml version="1.0" encoding="UTF-8"?>\n<XML>\n<SOURCE_URL>{url}</SOURCE_URL>',
            )
            logger.info(f"Saving XML content for URL: {url}")
            xml_file = temp_folder / f"processed_{idx}.xml"
            with open(xml_file, "w", encoding="utf-8") as f:
                f.write(xml_content)
        else:
            logger.warning(f"No XML content generated for URL: {url}")
            progress_tracker[url] = {
                "status": "failed",
                "cost": progress_tracker[url].get("cost", 0.0),
            }
            update_progress_file()
            update_batch_status(
                batch_tui,
                idx,
                URLState.FAILED,
                "❌ LLM returned empty response. Aborting task.",
                progress_pct=0.0,
                error="No XML content generated",
            )
            return None

        if xml_content:
            logger.info(f"Saving XML content for URL: {url}")
            xml_file = temp_folder / f"processed_{idx}.xml"
            with open(xml_file, "w", encoding="utf-8") as f:
                f.write(xml_content)
        else:
            logger.warning(f"No XML content to save for URL: {url}")

        logger.info(f"Successfully processed URL {idx}/{total}: {url}")
        is_valid_xml = validate_xml(xml_content)
        progress_tracker[url] = {
            "status": "successful",
            "cost": progress_tracker[url].get("cost", 0.0),
            "valid_xml": is_valid_xml,
        }
        update_progress_file()

        # Update TUI: Processing complete (final state update in process_multiple_pages)
        if batch_tui:
            batch_tui.update_task(
                idx,
                URLState.COMPLETE,
                progress_pct=100.0,
                size_in=len(html_content),
                size_out=len(xml_content),
                cost=float(progress_tracker[url].get("cost", 0.0)),
            )

        return xml_content
    except RequestException as e:
        error_message = f"Request error processing URL {url}: {str(e)}"
        logger.error(error_message)
        with open(error_log_file, "a", encoding="utf-8") as f:
            f.write(f"{error_message}\n")
        progress_tracker[url] = {
            "status": "failed",
            "cost": progress_tracker[url].get("cost", 0.0),
        }
        update_progress_file()
        update_batch_status(
            batch_tui,
            idx,
            URLState.FAILED,
            "❌ Connection error. Aborting task.",
            progress_pct=0.0,
            error=f"Request error: {str(e)}",
        )
        return None
    except Exception as e:
        error_message = f"Error processing URL {url}: {str(e)}"
        logger.error(error_message)
        with open(error_log_file, "a", encoding="utf-8") as f:
            f.write(f"{error_message}\n")
        progress_tracker[url] = {
            "status": "failed",
            "cost": progress_tracker[url].get("cost", 0.0),
        }
        update_progress_file()
        update_batch_status(
            batch_tui,
            idx,
            URLState.FAILED,
            "❌ Processing error. Aborting task.",
            progress_pct=0.0,
            error=str(e),
        )
        return None


def update_progress_file(atomic: bool = True) -> None:
    """
    Update the progress file with current scraping state.

    Args:
        atomic: If True, use atomic write (temp file + rename) to prevent corruption.
                If False, write directly to the file (faster but less safe).
    """
    global progress_file, temp_folder
    if not progress_file:
        logger.warning("Progress file path not set. Unable to update progress.")
        return

    with cost_lock:
        data = {
            "output_folder": str(temp_folder),
            "urls": [
                {
                    "index": i,
                    "url": url,
                    "status": url_data["status"],
                    "costs": url_data["cost"],
                    "valid_xml": url_data.get("valid_xml"),
                }
                for i, (url, url_data) in enumerate(progress_tracker.items())
            ],
        }

        if atomic:
            # Atomic write: write to temp file, then rename
            # This ensures the progress file is never in a corrupted state
            import tempfile

            progress_path = Path(progress_file)
            temp_fd, temp_path = tempfile.mkstemp(
                suffix=".tmp",
                prefix="progress_",
                dir=progress_path.parent,
            )
            try:
                with os.fdopen(temp_fd, "w") as f:
                    json.dump(data, f, indent=2)
                # Atomic rename (on POSIX systems)
                os.replace(temp_path, progress_file)
            except Exception:
                # Clean up temp file if rename fails
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                raise
        else:
            # Direct write (faster but not crash-safe)
            with open(progress_file, "w") as f:
                json.dump(data, f, indent=2)


def process_multiple_pages(
    urls: List[str],
    pricing_info: Dict[str, Dict[str, float]],
    num_threads: int = 5,
    scrape_only: bool = False,
    no_tui: bool = False,
    handlers_and_level: tuple[list[logging.Handler], int] = ([], logging.INFO),
) -> Tuple[Optional[str], Optional[SessionErrorTracker]]:
    """
    Processes multiple pages: scrapes each, converts to XML via LLM, and merges.

    Args:
        urls: List of URLs to process
        pricing_info: Model pricing information
        num_threads: Number of concurrent threads
        scrape_only: If True, skip AI processing
        no_tui: If True, disable Rich TUI (use simple spinner)
        handlers_and_level: Tuple of (handlers, level) from suppress_console_logging() for restoration
    """
    from rich.console import Console
    from rich.text import Text
    from rich.panel import Panel
    from rich import box

    global shutdown_flag
    logger.info(
        f"Processing multiple pages: {len(urls)} URLs found using {num_threads} threads."
    )
    if scrape_only:
        logger.info("Scrape-only mode: AI processing and XML merging will be skipped")
    xml_list = []

    # Use Rich TUI for batch mode or fallback to simple spinner
    batch_tui = BatchTUIManager(urls, no_tui=no_tui) if not no_tui else None

    # Create error tracker for session-wide error aggregation and circuit breaker
    error_tracker = SessionErrorTracker(
        consecutive_threshold=3, quota_immediate_stop=True
    )

    # NOTE: Logging is already suppressed in main_workflow before this function is called
    # Do NOT suppress again here or it will create conflicts with handler restoration

    if batch_tui:
        # Show waiting screen and wait for SPACE
        batch_tui.wait_for_start()

        # Check if user cancelled during waiting
        if batch_tui.should_stop:
            # Restore logging before returning
            restore_console_logging(handlers_and_level)
            logger.info("Processing cancelled by user before start")
            return None, error_tracker

        # Start live display
        batch_tui.start_live_display()
    else:
        # Fallback to simple spinner
        spinner = Spinner("Processing URLs")
        spinner.start()

    # Map URLs to task IDs
    url_to_task_id = {url: idx + 1 for idx, url in enumerate(urls)}

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            future_to_url = {
                executor.submit(
                    process_url,
                    url,
                    idx + 1,
                    len(urls),
                    pricing_info,
                    scrape_only,
                    batch_tui,
                    error_tracker,
                ): url
                for idx, url in enumerate(urls)
            }

            # Update TUI while processing
            while True:
                # Use centralized poll interval from config for consistent TUI responsiveness
                done, pending = concurrent.futures.wait(
                    future_to_url.keys(),
                    timeout=BATCH_TUI_POLL_INTERVAL,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )

                # Update TUI display
                if batch_tui:
                    batch_tui.update_display()

                # Check for shutdown or circuit breaker
                if shutdown_flag or (batch_tui and batch_tui.should_stop):
                    logger.info("Shutdown requested. Cancelling remaining tasks.")
                    executor.shutdown(wait=False)
                    break

                # Check circuit breaker for fatal errors (quota exceeded, rate limit, etc.)
                if error_tracker.circuit_breaker.is_triggered:
                    trigger_reason = (
                        error_tracker.circuit_breaker.trigger_reason or "Unknown error"
                    )
                    logger.warning(
                        f"Circuit breaker triggered: {trigger_reason}. "
                        f"Stopping batch processing to avoid wasting resources."
                    )
                    # Cancel remaining futures gracefully
                    executor.shutdown(wait=False, cancel_futures=True)

                    # Stop live display FIRST (while logging still suppressed)
                    if batch_tui:
                        batch_tui.stop_live_display()

                        # Show graceful error dialog to user (while logging STILL suppressed)
                        # This prevents worker threads from logging during dialog display
                        batch_tui.show_circuit_breaker_dialog(
                            trigger_reason, str(temp_folder)
                        )

                        # Only restore logging AFTER dialog is shown
                        # (worker threads have had time to finish)
                        restore_console_logging(handlers_and_level)

                    # Return early - do NOT continue to merge phase
                    return None, error_tracker

                # Process completed futures
                for future in done:
                    url = future_to_url[future]
                    task_id = url_to_task_id[url]

                    try:
                        xml_content = future.result()
                        if xml_content:
                            xml_list.append(xml_content)
                            if batch_tui:
                                batch_tui.update_task(
                                    task_id, URLState.COMPLETE, progress_pct=100.0
                                )
                            else:
                                spinner.update_text(
                                    f"Processed {len(xml_list)}/{len(urls)} URLs"
                                )
                        else:
                            if batch_tui:
                                batch_tui.update_task(
                                    task_id,
                                    URLState.FAILED,
                                    progress_pct=0.0,
                                    error="Processing failed",
                                )
                    except Exception as e:
                        logger.error(f"Unhandled exception for URL {url}: {str(e)}")
                        if batch_tui:
                            batch_tui.update_task(
                                task_id, URLState.FAILED, progress_pct=0.0, error=str(e)
                            )

                    # Remove from map
                    del future_to_url[future]

                # If all done, break
                if not future_to_url:
                    break

    finally:
        if batch_tui:
            # Final update before stopping
            batch_tui.update_display()
            # Use centralized pause duration - DO NOT hardcode timing values
            time.sleep(BATCH_FINAL_STATE_PAUSE)
            batch_tui.stop_live_display()
            # DON'T restore logging yet - keep it suppressed during merge
        else:
            spinner.end()

    if shutdown_flag:
        logger.info("Shutdown initiated. Saving partial results.")
        # Restore logging before returning
        if batch_tui:
            restore_console_logging(handlers_and_level)

    # In scrape-only mode, skip XML merging
    if scrape_only:
        logger.info("Scrape-only mode: Completed successfully without XML merging")
        # Restore logging before returning
        if batch_tui:
            restore_console_logging(handlers_and_level)
        return "scrape_only_completed", error_tracker

    # ========== Clean Merge Progress Display ==========
    # Logging is still suppressed here - show clean merge progress
    console = Console(force_terminal=True)

    # Clear screen and show merge header
    console.clear()
    console.print()
    console.print(
        Panel(
            Text.assemble(
                ("🔄 ", "bold cyan"),
                ("Merging XML Files", "bold cyan"),
            ),
            box=box.DOUBLE,
            border_style="cyan",
            padding=(1, 2),
        )
    )
    console.print()

    # Progress variables
    merge_progress = {"current": 0, "total": 0, "message": ""}

    def merge_progress_callback(current: int, total: int, message: str) -> None:
        """Callback to update merge progress"""
        merge_progress["current"] = current
        merge_progress["total"] = total
        merge_progress["message"] = message

        if total > 0:
            percentage = int((current / total) * 85)  # 0-85% for merging files
            console.print(
                f"  [{current}/{total}] {message}... {percentage}%", style="cyan"
            )

    logger.info("Merging XML content from all processed pages")
    merged_xml = merge_xmls(temp_folder, progress_callback=merge_progress_callback)

    if not merged_xml or merged_xml == "<TEXTUAL_API />":
        console.print(
            "  ❌ No valid XML content extracted from pages.", style="bold red"
        )
        console.print()
        # Restore logging before returning
        if batch_tui:
            restore_console_logging(handlers_and_level)
        logger.error("No valid XML content extracted from pages.")
        return None, error_tracker

    # Show validation progress
    console.print("  Validating merged XML... 90%", style="cyan")

    # Save merged XML
    logger.info("Saving merged XML output")
    merged_xml_file = temp_folder / "merged_output.xml"
    try:
        temp_folder.mkdir(parents=True, exist_ok=True)
        with open(merged_xml_file, "w", encoding="utf-8") as f:
            f.write(merged_xml)
        console.print("  Saving merged XML... 100%", style="bold green")
        console.print()
        console.print(
            Panel(
                Text.assemble(
                    ("✅ ", "bold green"),
                    (f"Merge Complete! Saved to: {merged_xml_file}", "green"),
                ),
                box=box.ROUNDED,
                border_style="green",
            )
        )
        console.print()
        logger.info(f"Merged XML saved successfully to {merged_xml_file}")
    except OSError as e:
        console.print(f"  ❌ Error saving merged XML: {e}", style="bold red")
        console.print()
        # Restore logging before returning
        if batch_tui:
            restore_console_logging(handlers_and_level)
        logger.error(f"Error saving merged XML file: {e}")
        return None, error_tracker

    # NOW restore console logging after merge is complete
    if batch_tui:
        restore_console_logging(handlers_and_level)

    return merged_xml, error_tracker


def read_patterns_from_file(file_path: str) -> Optional[str]:
    """
    Read patterns from a file and return them as a comma-separated string.
    """
    if not file_path:
        return None
    try:
        with open(file_path, "r") as f:
            patterns = [line.strip() for line in f if line.strip()]
        if not patterns:
            logger.warning(f"No valid patterns found in {file_path}")
            return None
        return ",".join(patterns)
    except OSError as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return None


def display_scraping_summary(
    result: Dict[str, Any],
    urls: List[str],
    temp_folder: Path,
    error_log_file: Path,
    error_tracker: Optional[SessionErrorTracker] = None,
) -> None:
    summary_data = {
        "Base URL": urls[0] if urls else "N/A",
        "Sitemap.xml URL": f"{urls[0].rstrip('/')}/sitemap.xml" if urls else "N/A",
        "Number of URLs before filtering": str(result.get("total_urls", "N/A")),
        "Number of URLs after filtering": str(result.get("filtered_urls", "N/A")),
        "Number of Scraped URLs": str(result.get("scraped_urls", "N/A")),
        "Number of Successfully Processed URLs": str(
            result.get("successful_urls", "N/A")
        ),
        "Number of URLs failed to scrape": str(result.get("failed_urls", "N/A")),
        "Total number of XML files generated": str(
            result.get("total_xml_files", "N/A")
        ),
        "Valid XML files generated": str(result.get("valid_xml_files", "N/A")),
        "Invalid XML files generated": str(result.get("invalid_xml_files", "N/A")),
        "Total costs of the scraping job": f"${result['total_cost']:.6f}",
        "Temporary folder path": str(temp_folder),
        "Error log file path": str(error_log_file),
        "Merged XML file path": str(temp_folder / "merged_output.xml"),
    }

    summary_box = create_summary_box(summary_data)

    # Build root cause analysis section if there were errors
    root_cause_section = ""
    if error_tracker and error_tracker.total_errors > 0:
        root_cause_section = "\n"
        root_cause_section += "=" * 60 + "\n"
        root_cause_section += "🔍 ROOT CAUSE ANALYSIS\n"
        root_cause_section += "=" * 60 + "\n"

        # Primary failure reason
        primary_reason = error_tracker.get_primary_failure_reason()
        if primary_reason:
            root_cause_section += f"\n⚠️  Primary failure: {primary_reason}\n"

        # Error breakdown by category
        error_summary = error_tracker.get_error_summary()
        if error_summary:
            root_cause_section += "\n📊 Error breakdown:\n"
            for category, count in sorted(
                error_summary.items(), key=lambda x: x[1], reverse=True
            ):
                desc = get_error_description(category)
                root_cause_section += f"   • {desc}: {count} occurrence(s)\n"

        # Circuit breaker status
        if error_tracker.circuit_breaker.is_triggered:
            root_cause_section += f"\n🛑 Processing stopped: {error_tracker.circuit_breaker.trigger_reason}\n"

        root_cause_section += "=" * 60 + "\n"

    # Build data safety reassurance section
    data_safety_section = "\n"
    data_safety_section += "💾 DATA SAFETY\n"
    data_safety_section += "-" * 40 + "\n"
    data_safety_section += "✓ All scraped data has been saved to:\n"
    data_safety_section += f"  {temp_folder}\n"
    data_safety_section += "✓ Progress saved - job can be resumed\n"
    data_safety_section += "\n"
    data_safety_section += "📌 To resume this job later, run:\n"
    progress_file_path = temp_folder / "progress.json"
    data_safety_section += f"  apias --resume {progress_file_path}\n"
    data_safety_section += "-" * 40 + "\n"

    if result["result"] == "already_completed":
        print(SUCCESS_SEPARATOR)
        print("✅ Scraping process already completed successfully.")
        print(summary_box)
        print(SUCCESS_SEPARATOR)
    elif result["result"]:
        print(SUCCESS_SEPARATOR)
        print("✅ XML Extraction Successful.")
        print(summary_box)
        if error_tracker and error_tracker.total_errors > 0:
            # Show any warnings/errors that occurred during successful run
            print(root_cause_section)
        print(data_safety_section)
        print(SUCCESS_SEPARATOR)
    else:
        print(ERROR_SEPARATOR)
        print("❌ XML Extraction Failed or No Content Extracted.")
        print(summary_box)
        if root_cause_section:
            print(root_cause_section)
        print(data_safety_section)
        print(ERROR_SEPARATOR)


def main_workflow(
    urls: List[str],
    mode: str = "single",
    whitelist: Optional[str] = None,
    blacklist: Optional[str] = None,
    num_threads: int = 5,
    resume_file: Optional[str] = None,
    scrape_only: bool = False,
    no_tui: bool = False,
    mock: bool = False,
    limit: Optional[int] = None,
) -> Dict[str, Union[Optional[str], float, int, Optional[SessionErrorTracker]]]:
    global progress_tracker, total_cost
    """
    Executes the Web API Retrieval and XML Extraction workflow.

    Args:
        urls: List of URLs to process
        mode: Processing mode ("single" or "batch")
        whitelist: Path to whitelist file (optional)
        blacklist: Path to blacklist file (optional)
        num_threads: Number of concurrent threads for batch mode
        resume_file: Path to resume file for interrupted jobs (optional)
        scrape_only: If True, only scrape HTML without AI processing
        no_tui: If True, disable Rich TUI (use plain text output)
        mock: If True, use mock API for testing (no token costs)
        limit: Maximum number of pages to scrape (only applies in batch mode)
    """
    global shutdown_flag, total_cost, progress_tracker, progress_file, temp_folder

    # Configure logging level based on TUI mode
    configure_logging_for_tui(no_tui)

    # Suppress console logging IMMEDIATELY in batch mode with TUI
    # This prevents ANY logger output before TUI starts
    handlers_and_level: tuple[list[logging.Handler], int] = (
        [],
        logging.INFO,
    )  # Default if not batch mode with TUI
    if mode == "batch" and not no_tui:
        handlers_and_level = suppress_console_logging()
    else:
        # Only show initial messages if NOT in batch TUI mode
        print(INFO_SEPARATOR)
        logger.info("Starting Web API Retrieval workflow.")
        if scrape_only:
            logger.info("Scrape-only mode enabled: AI processing will be skipped")
        if mock:
            logger.info(
                "MOCK MODE enabled: Using simulated API responses (no token costs)"
            )
        if no_tui:
            logger.info("TUI disabled: Using plain text output")
        print(INFO_SEPARATOR)

    result: Dict[
        str, Union[Optional[str], float, int, Optional[SessionErrorTracker]]
    ] = {
        "result": None,
        "total_cost": 0.0,
        "total_urls": 0,
        "filtered_urls": 0,
        "scraped_urls": 0,
        "successful_urls": 0,
        "failed_urls": 0,
        "total_xml_files": 0,
        "valid_xml_files": 0,
        "invalid_xml_files": 0,
    }

    try:
        # Load pricing information
        pricing_info = load_model_pricing()
        if pricing_info is None:
            logger.error("Failed to load pricing information. Exiting workflow.")
            return result
        if not isinstance(pricing_info, dict) or not pricing_info:
            logger.error("Invalid pricing information. Exiting workflow.")
            return result
        logger.info("Loaded model pricing information.")

        # Handle resume functionality
        if resume_file:
            resume_file = os.path.abspath(resume_file)
            if os.path.exists(resume_file):
                with open(resume_file, "r") as f:
                    resume_data = json.load(f)
                temp_folder = Path(resume_data.get("output_folder", temp_folder))
                progress_tracker = {
                    url_data["url"]: {
                        "status": url_data["status"],
                        "cost": float(url_data["costs"]),
                        "valid_xml": url_data.get("valid_xml"),
                    }
                    for url_data in resume_data["urls"]
                }
                progress_file = resume_file
                urls = [url_data["url"] for url_data in resume_data["urls"]]

                # Check if all URLs are already processed successfully
                successful_urls = sum(
                    1
                    for url_data in progress_tracker.values()
                    if url_data.get("status") == "successful"
                )
                total_urls = len(urls)
                total_cost = sum(
                    float(url_data.get("cost", 0.0))
                    for url_data in progress_tracker.values()
                )

                if successful_urls == total_urls:
                    logger.info(
                        f"Scraping process already completed successfully ({successful_urls}/{total_urls}). Total costs: ${total_cost:.6f}"
                    )
                    result.update(
                        {
                            "result": "already_completed",
                            "total_cost": total_cost,
                            "total_urls": total_urls,
                            "filtered_urls": total_urls,
                            "scraped_urls": successful_urls,
                            "successful_urls": successful_urls,
                            "failed_urls": 0,
                            "total_xml_files": successful_urls,
                            "valid_xml_files": successful_urls,
                            "invalid_xml_files": 0,
                        }
                    )
                    result["result"] = cast(Optional[str], result["result"])
                    return result
                elif successful_urls > 0:
                    logger.info(
                        f"Scraping process partially completed. ({successful_urls} successful, {total_urls - successful_urls} failed). Total costs so far: ${total_cost:.6f}"
                    )
                    user_input = input(
                        "Do you want to attempt the failed URLs again? (y/n): "
                    ).lower()
                    if user_input != "y":
                        logger.info("Exiting without processing failed URLs.")
                        result.update(
                            {
                                "total_cost": total_cost,
                                "total_urls": total_urls,
                                "filtered_urls": total_urls,
                                "scraped_urls": successful_urls,
                                "successful_urls": successful_urls,
                                "failed_urls": total_urls - successful_urls,
                                "total_xml_files": successful_urls,
                                "valid_xml_files": successful_urls,
                                "invalid_xml_files": 0,
                            }
                        )
                        return result
                    urls = [
                        url
                        for url, data in progress_tracker.items()
                        if data.get("status") != "successful"
                    ]
                    logger.info(f"Resuming scraping for {len(urls)} failed URLs")
                else:
                    logger.info(f"Resuming scraping for {len(urls)} URLs")
            else:
                logger.info(
                    f"Resume file {resume_file} does not exist. Creating a new one."
                )
                progress_tracker = {}
                progress_file = resume_file
        else:
            progress_file = os.path.join(temp_folder, "progress.json")
            progress_tracker = {}

        # Ensure temp_folder exists
        try:
            temp_folder.mkdir(parents=True, exist_ok=True)
            logger.info(f"Temporary folder created/verified: {temp_folder}")
        except OSError as e:
            logger.error(f"Error creating temporary folder: {e}")
            return result

        # Create an empty error log file at the start of the workflow
        error_log_file = temp_folder / "error_log.txt"
        with open(error_log_file, "w", encoding="utf-8") as f:
            pass

        # Process whitelist and blacklist inputs
        whitelist_str = read_patterns_from_file(whitelist) if whitelist else None
        blacklist_str = read_patterns_from_file(blacklist) if blacklist else None

        # Extract and filter URLs if in batch mode
        if mode == "batch":
            # Only print separator and messages if NOT using TUI
            if no_tui:
                print(SEPARATOR)
                logger.info("Fetching sitemap")
            if not urls:
                logger.error("No URLs provided for batch mode.")
                # Restore logging if we suppressed it
                if not no_tui:
                    restore_console_logging(handlers_and_level)
                return result
            base_url = urls[0]
            # Pass quiet=True to suppress Spinner output when TUI will be used
            sitemap_content = fetch_sitemap([base_url], quiet=not no_tui)
            if not sitemap_content:
                logger.error("Sitemap retrieval failed. Exiting workflow.")
                return result
            logger.info("Extracting URLs from sitemap")
            extracted_urls = extract_urls_from_sitemap(
                sitemap_content=sitemap_content,
                sitemap_file=None,
                whitelist_str=whitelist_str,
                blacklist_str=blacklist_str,
            )
            if extracted_urls:
                result["total_urls"] = len(extracted_urls)
                urls = extracted_urls

                # Apply limit if specified
                if limit is not None and limit > 0:
                    original_count = len(urls)
                    urls = urls[:limit]
                    logger.info(
                        f"Applied limit: {original_count} URLs → {len(urls)} URLs (limit={limit})"
                    )

                result["filtered_urls"] = len(urls)
                logger.info(f"Extracted {len(urls)} URLs from sitemap")
            else:
                logger.error("No URLs extracted from sitemap. Exiting workflow.")
                return result
            print(SEPARATOR)
        elif mode == "single":
            if not urls:
                logger.error("No URL provided for single mode. Exiting workflow.")
                return result
            urls = [urls[0]]  # Ensure we only process the first URL in single mode
            result["total_urls"] = 1
            result["filtered_urls"] = 1

        # Initialize progress tracker with extracted URLs
        progress_tracker = {url: {"status": "pending", "cost": 0.0} for url in urls}

        # Save initial progress
        update_progress_file()

        # Determine processing type
        if mode == "single":
            logger.info("Workflow Type: Single Page Processing")
            xml_result = process_single_page(
                urls[0], pricing_info, scrape_only, no_tui, mock
            )
            if xml_result:
                is_valid = validate_xml(xml_result)
                print(
                    f"The XML file was successfully generated and it is {'valid' if is_valid else 'non valid'} XML"
                )
                result.update(
                    {
                        "result": xml_result,
                        "scraped_urls": 1,
                        "successful_urls": 1,
                        "failed_urls": 0,
                        "total_xml_files": 1,
                        "valid_xml_files": 1 if is_valid else 0,
                        "invalid_xml_files": 0 if is_valid else 1,
                    }
                )
                result["result"] = cast(Optional[str], result["result"])
            elif scrape_only:
                # In scrape-only mode, success means HTML was saved
                logger.info(
                    "Scrape-only mode: HTML saved successfully without XML generation"
                )
                result.update(
                    {
                        "result": "scrape_only_completed",
                        "scraped_urls": 1,
                        "successful_urls": 1,
                        "failed_urls": 0,
                        "total_xml_files": 0,
                        "valid_xml_files": 0,
                        "invalid_xml_files": 0,
                    }
                )
                result["result"] = cast(Optional[str], result["result"])
            else:
                result.update(
                    {
                        "scraped_urls": 1,
                        "successful_urls": 0,
                        "failed_urls": 1,
                        "total_xml_files": 0,
                        "valid_xml_files": 0,
                        "invalid_xml_files": 0,
                    }
                )
        elif mode == "batch":
            logger.info("Workflow Type: Batch Processing")
            xml_result, batch_error_tracker = process_multiple_pages(
                urls, pricing_info, num_threads, scrape_only, no_tui, handlers_and_level
            )
            # Store error tracker in result for summary display
            result["error_tracker"] = batch_error_tracker
            if xml_result and temp_folder.exists():
                valid_count, total_count = count_valid_xml_files(temp_folder)
                # Clean output - counts will show in summary table
                logger.info(
                    f"Generated {total_count} XML files ({valid_count} valid, {total_count - valid_count} invalid)"
                )
                result.update(
                    {
                        "result": xml_result,
                        "scraped_urls": len(urls),
                        "successful_urls": total_count,
                        "failed_urls": len(urls) - total_count,
                        "total_xml_files": total_count,
                        "valid_xml_files": valid_count,
                        "invalid_xml_files": total_count - valid_count,
                    }
                )
                result["result"] = cast(Optional[str], result["result"])
            elif scrape_only and xml_result:
                # In scrape-only mode, success means HTML files were saved
                logger.info(
                    "Scrape-only mode: HTML files saved successfully without XML generation"
                )
                result.update(
                    {
                        "result": "scrape_only_completed",
                        "scraped_urls": len(urls),
                        "successful_urls": len(urls),
                        "failed_urls": 0,
                        "total_xml_files": 0,
                        "valid_xml_files": 0,
                        "invalid_xml_files": 0,
                    }
                )
                result["result"] = cast(Optional[str], result["result"])
            elif xml_result:
                logger.warning(
                    "Temporary folder not found. Unable to count valid XML files."
                )
                result["result"] = cast(Optional[str], xml_result)
            else:
                result.update(
                    {
                        "scraped_urls": len(urls),
                        "successful_urls": 0,
                        "failed_urls": len(urls),
                        "total_xml_files": 0,
                        "valid_xml_files": 0,
                        "invalid_xml_files": 0,
                    }
                )
        else:
            logger.error(f"Invalid mode specified: {mode}. Choose 'single' or 'batch'.")

        if shutdown_flag:
            logger.info("Workflow completed early due to shutdown request.")
        else:
            print(SUCCESS_SEPARATOR)
            logger.info("Workflow completed successfully.")
            logger.info(f"Total cost for all API calls: ${total_cost:.6f}")
            print(SUCCESS_SEPARATOR)

        result["total_cost"] = total_cost
        return result
    except KeyboardInterrupt:
        print(WARNING_SEPARATOR)
        logger.info("Keyboard interrupt received. Shutting down gracefully.")
        shutdown_flag = True
        result["total_cost"] = total_cost
        return result
    except Exception as e:
        print(ERROR_SEPARATOR)
        logger.error(f"An unexpected error occurred in the main workflow: {str(e)}")
        result["total_cost"] = total_cost
        return result


def start_resume_mode(json_file_path: str) -> None:
    # Check if the resume file exists
    if not os.path.exists(json_file_path):
        print(f"Error: Resume file {json_file_path} does not exist.")
        sys.exit(1)

    # Load progress from the resume file
    try:
        with open(json_file_path, "r") as f:
            progress_data = json.load(f)
        urls = [url_data["url"] for url_data in progress_data.get("urls", [])]
        if not urls:
            print(f"Error: No valid URLs found in the resume file {json_file_path}")
            sys.exit(1)

        # Update global variables with saved data
        global temp_folder, total_cost, progress_tracker
        temp_folder = Path(progress_data.get("output_folder", temp_folder))
        total_cost = sum(
            float(url_data.get("costs", 0.0)) for url_data in progress_data["urls"]
        )
        progress_tracker = {
            url_data["url"]: {
                "status": url_data["status"],
                "cost": float(url_data["costs"]),
                "valid_xml": url_data.get("valid_xml", False),
            }
            for url_data in progress_data["urls"]
        }

    except json.JSONDecodeError:
        print(f"Error: The resume file {json_file_path} is not a valid JSON file.")
        sys.exit(1)
    except KeyError:
        print(
            f"Error: The resume file {json_file_path} does not have the expected structure."
        )
        sys.exit(1)
    except Exception as e:
        print(
            f"Error: An unexpected error occurred while reading the resume file: {str(e)}"
        )
        sys.exit(1)

    result = main_workflow(
        urls=urls,
        mode="batch",
        whitelist=None,
        blacklist=None,
        num_threads=5,
        resume_file=json_file_path,
    )

    display_scraping_summary(
        result,
        urls,
        temp_folder,
        error_log_file,
        cast(Optional[SessionErrorTracker], result.get("error_tracker")),
    )


def _find_auto_resume_session() -> Optional[str]:
    """
    Find the most recent incomplete session for auto-resume.

    Non-interactive version of check_for_resumable_sessions().
    Used when --auto-resume flag is set for headless/CI operation.

    Returns:
        Path to most recent progress.json with incomplete work, or None
    """
    import glob

    progress_files = glob.glob("temp_*/progress.json")

    resumable_sessions: list[dict[str, Any]] = []
    for pf in progress_files:
        try:
            with open(pf, "r") as f:
                data = json.load(f)

            urls = data.get("urls", [])
            if not urls:
                continue

            completed = sum(1 for u in urls if u.get("status") == "completed")
            failed = sum(1 for u in urls if u.get("status") == "failed")
            total = len(urls)
            incomplete = total - completed - failed

            if incomplete > 0:
                resumable_sessions.append(
                    {
                        "path": pf,
                        "incomplete": incomplete,
                        "mtime": os.path.getmtime(pf),
                    }
                )
        except (json.JSONDecodeError, OSError, KeyError):
            continue

    if not resumable_sessions:
        return None

    # Return the most recent session (sort by mtime descending)
    resumable_sessions.sort(key=lambda x: float(x["mtime"]), reverse=True)
    return str(resumable_sessions[0]["path"])


def check_for_resumable_sessions() -> Optional[str]:
    """
    Check for existing incomplete sessions that can be resumed.

    Looks for temp_* directories with progress.json files that have
    incomplete URLs (status != 'completed' and status != 'failed').

    Returns:
        Path to progress.json if user wants to resume, None otherwise
    """
    import glob

    # Find all temp directories with progress.json files
    progress_files = glob.glob("temp_*/progress.json")

    resumable_sessions = []
    for pf in progress_files:
        try:
            with open(pf, "r") as f:
                data = json.load(f)

            urls = data.get("urls", [])
            if not urls:
                continue

            # Count incomplete URLs
            completed = sum(1 for u in urls if u.get("status") == "completed")
            failed = sum(1 for u in urls if u.get("status") == "failed")
            total = len(urls)
            incomplete = total - completed - failed

            if incomplete > 0:
                # This session has incomplete work
                folder = data.get("output_folder", os.path.dirname(pf))
                resumable_sessions.append(
                    {
                        "path": pf,
                        "folder": folder,
                        "total": total,
                        "completed": completed,
                        "failed": failed,
                        "incomplete": incomplete,
                        "mtime": os.path.getmtime(pf),
                    }
                )
        except (json.JSONDecodeError, OSError, KeyError):
            continue

    if not resumable_sessions:
        return None

    # Sort by most recent first
    resumable_sessions.sort(key=lambda x: x["mtime"], reverse=True)

    # Show the user the available sessions
    print("\n" + "=" * 60)
    print("  RESUMABLE SESSIONS FOUND")
    print("=" * 60)

    for i, session in enumerate(resumable_sessions[:5], 1):  # Show max 5 sessions
        mtime_str = datetime.fromtimestamp(session["mtime"]).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        print(f"\n  [{i}] {session['folder']}")
        print(f"      Last modified: {mtime_str}")
        print(
            f"      Progress: {session['completed']}/{session['total']} completed, "
            f"{session['incomplete']} pending, {session['failed']} failed"
        )

    print("\n  [0] Start a new session (don't resume)")
    print("=" * 60)

    # Ask user which session to resume
    while True:
        try:
            choice = input("\n  Enter number to resume (0 for new session): ").strip()
            if choice == "0" or choice == "":
                return None

            idx = int(choice) - 1
            if 0 <= idx < len(resumable_sessions):
                selected = resumable_sessions[idx]
                print(f"\n  Resuming session from: {selected['path']}")
                return str(selected["path"])
            else:
                print("  Invalid choice. Please enter a valid number.")
        except ValueError:
            print("  Invalid input. Please enter a number.")
        except KeyboardInterrupt:
            print("\n  Cancelled.")
            return None


def start_single_scrape(
    url: str, scrape_only: bool = False, no_tui: bool = False, mock: bool = False
) -> None:
    """Start scraping a single URL."""
    from apias.config import validate_url

    if not validate_url(url):
        print(f"Error: Invalid URL format: {url}")
        print("URL must start with http:// or https:// and have a valid domain")
        sys.exit(1)

    result = main_workflow(
        urls=[url],
        mode="single",
        num_threads=1,
        scrape_only=scrape_only,
        no_tui=no_tui,
        mock=mock,
    )
    display_scraping_summary(
        result,
        [url],
        temp_folder,
        error_log_file,
        cast(Optional[SessionErrorTracker], result.get("error_tracker")),
    )


def create_summary_box(summary_data: Dict[str, str]) -> str:
    terminal_width = shutil.get_terminal_size().columns
    max_label_length = max(len(label) for label in summary_data.keys())
    max_value_length = max(len(str(value)) for value in summary_data.values())

    box_width = min(
        terminal_width - 3, max(max_label_length + max_value_length + 5, 50)
    )
    content_width = box_width - 2

    def create_line(left: str, middle: str, right: str, fill: str = "─") -> str:
        return f"{left}{fill * (box_width - 2)}{right}"

    def format_item(label: str, value: str) -> str:
        if len(label) + len(value) + 3 <= content_width:
            return f"│ {label}:{value.ljust(content_width - len(label) - 2)}│"
        else:
            return f"│ {label}:\n│ {value.ljust(content_width)}│"

    box = [
        create_line("┌", "┬", "┐"),
        f"│{'Scraping Job Summary'.center(box_width - 2)}│",
        create_line("├", "┼", "┤"),
    ]

    for label, value in summary_data.items():
        box.extend(format_item(label, str(value)).split("\n"))

    box.append(create_line("└", "┴", "┘"))
    return "\n".join(box)


def start_batch_scrape(
    url: str,
    whitelist: str,
    blacklist: str,
    scrape_only: bool = False,
    no_tui: bool = False,
    mock: bool = False,
    limit: Optional[int] = None,
) -> None:
    """Start batch scraping from a sitemap URL."""
    from apias.config import validate_url

    if not validate_url(url):
        print(f"Error: Invalid URL format: {url}")
        print("URL must start with http:// or https:// and have a valid domain")
        sys.exit(1)

    result = main_workflow(
        urls=[url],
        mode="batch",
        whitelist=whitelist,
        blacklist=blacklist,
        num_threads=5,
        scrape_only=scrape_only,
        no_tui=no_tui,
        mock=mock,
        limit=limit,
    )
    display_scraping_summary(
        result,
        [url],
        temp_folder,
        error_log_file,
        cast(Optional[SessionErrorTracker], result.get("error_tracker")),
    )


# ============================
# Command-Line Interface
# ============================


class APIDocument:
    """Class representing a parsed API document."""

    def __init__(self, content: str) -> None:
        """Initialize APIDocument with raw content."""
        self.content: str = content
        self.endpoints: list[str] = []
        self.methods: list[str] = []
        self.descriptions: list[str] = []
        self._parse()

    def _parse(self) -> None:
        """Parse the raw content into structured data."""
        lines = self.content.split("\n")
        for line in lines:
            line = line.strip()
            if "/api/" in line:
                self.endpoints.append(line)
            if any(
                method in line.upper() for method in ["GET", "POST", "PUT", "DELETE"]
            ):
                self.methods.append(line)
            if line and not line.startswith("#"):
                self.descriptions.append(line)

    def to_markdown(self) -> str:
        """Convert the document to markdown format."""
        return "\n".join(self.descriptions)

    def to_json(self) -> Dict[str, List[str]]:
        """Convert the document to a dictionary."""
        return {
            "endpoints": self.endpoints,
            "methods": self.methods,
            "descriptions": self.descriptions,
        }

    def save(self, path: Union[str, Path]) -> None:
        """Save the document to a file."""
        path = Path(path)
        with open(path, "w") as f:
            f.write(self.to_markdown())

    def __str__(self) -> str:
        """Return a string representation of the document."""
        return "\n".join(self.endpoints + self.methods + self.descriptions)


def parse_documentation(doc_content: str) -> APIDocument:
    """Parse API documentation content and extract structured information.

    Args:
        doc_content: Raw API documentation text

    Returns:
        APIDocument: Structured API documentation
    """
    if not isinstance(doc_content, str):
        raise TypeError("Documentation content must be a string")

    return APIDocument(doc_content)


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate the configuration dictionary.

    Args:
        config: Configuration dictionary to validate

    Returns:
        bool: True if configuration is valid, False otherwise
    """
    required_fields = {"base_url", "output_format"}

    # Check if all required fields are present
    if not all(field in config for field in required_fields):
        return False

    # Validate base_url format
    if not isinstance(config["base_url"], str) or not config["base_url"].startswith(
        "http"
    ):
        return False

    # Validate output_format
    if not isinstance(config["output_format"], str) or config["output_format"] not in [
        "markdown",
        "html",
        "xml",
    ]:
        return False

    return True


def check_for_running_apias_instances() -> None:
    """
    Check if any other APIAS process is already running.
    If found, print error message and exit to prevent multiple instances
    from making simultaneous API calls and wasting money.
    """
    try:
        # Get current process ID and parent process info
        current_pid = os.getpid()
        parent_pid = os.getppid()

        # Use ps command with full command line output
        # Use centralized timeout from config - DO NOT hardcode
        result = subprocess.run(
            ["ps", "-eo", "pid,command"],
            capture_output=True,
            text=True,
            timeout=SUBPROCESS_TIMEOUT,
        )

        # Find APIAS processes by matching Python processes running apias.py or apias module
        apias_processes = []
        for line in result.stdout.split("\n"):
            # Skip header line and current process
            if not line.strip() or "PID" in line:
                continue

            # Parse PID and command
            parts = line.strip().split(None, 1)
            if len(parts) < 2:
                continue

            try:
                pid = int(parts[0])
                command = parts[1]
            except ValueError:
                continue

            # Skip current process and parent process (shell/uv launcher)
            if pid == current_pid or pid == parent_pid:
                continue

            # Match actual APIAS execution patterns:
            # ONLY match Python processes running apias, NOT shell wrappers or uv launchers
            # This prevents false positives from parent shell processes
            is_apias = False

            # Check for Python executing apias.py or apias module
            if re.search(r"\bpython[0-9.]*\s+.*apias\.py\b", command):
                is_apias = True
            elif re.search(r"\bpython[0-9.]*\s+-m\s+apias\b", command):
                is_apias = True
            # Check for direct apias script execution (installed via pip/uv)
            elif re.search(r"\b/[^\s]*/bin/apias\s+", command):
                is_apias = True

            if is_apias:
                apias_processes.append((pid, command))

        if apias_processes:
            print("=" * 80)
            print("ERROR: Another APIAS process is already running!")
            print("=" * 80)
            print(
                "\nRunning multiple APIAS instances simultaneously wastes money on API calls."
            )
            print("\nFound the following APIAS process(es):")
            for pid, cmdline in apias_processes:
                print(f"  PID {pid}: {cmdline[:100]}...")
            print(
                "\nPlease wait for the existing process to complete, or kill it with:"
            )
            print(f"  kill {apias_processes[0][0]}")
            print("\nExiting to prevent duplicate API calls.")
            print("=" * 80)
            sys.exit(1)

    except subprocess.TimeoutExpired:
        # If ps command times out, just continue (better than blocking startup)
        pass
    except Exception as e:
        # If check fails for any reason, just continue
        # (better to possibly have duplicates than to block legitimate runs)
        logger.debug(f"Singleton check failed: {e}")


def main() -> None:
    """Main entry point for APIAS CLI."""
    # Check for running APIAS instances to prevent multiple simultaneous API calls
    check_for_running_apias_instances()

    parser = argparse.ArgumentParser(
        description="APIAS - API Auto Scraper: Extract API documentation from web pages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  Single page mode (default):
    apias --url "https://api.example.com/docs"
    
  Batch mode (scrape multiple pages from sitemap):
    apias --url "https://example.com" --mode batch
    apias --url "https://example.com" --mode batch --whitelist patterns.txt
    apias --url "https://example.com" --mode batch --limit 50
    
  Resume interrupted session:
    apias --resume "./temp_20240101_120000/progress.json"
    
  Headless/CI mode (no TUI, quiet output):
    apias --url "https://example.com" --mode batch --quiet --auto-resume
    
  Using configuration file:
    apias --url "https://example.com" --config apias_config.yaml

For more information, visit: https://github.com/Emasoft/apias
""",
    )

    # URL and mode options
    parser.add_argument(
        "-u",
        "--url",
        type=str,
        default=None,
        help="Base URL to scrape",
        metavar="URL",
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        default="single",
        choices=["single", "batch"],
        help="Scraping mode: 'single' for one page, 'batch' for sitemap crawl (default: single)",
    )

    # Resume and config options
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        default=None,
        help="Path to progress.json file to resume an interrupted session",
        metavar="FILE",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help="Path to YAML/JSON configuration file (see --generate-config)",
        metavar="FILE",
    )
    parser.add_argument(
        "--generate-config",
        action="store_true",
        default=False,
        help="Generate example configuration file (apias_config.yaml) and exit",
    )

    # URL filtering (batch mode)
    parser.add_argument(
        "-w",
        "--whitelist",
        type=str,
        default=None,
        help="File with URL patterns to include (one regex per line, batch mode only)",
        metavar="FILE",
    )
    parser.add_argument(
        "-b",
        "--blacklist",
        type=str,
        default=None,
        help="File with URL patterns to exclude (one regex per line, batch mode only)",
        metavar="FILE",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of pages to scrape (batch mode only)",
        metavar="N",
    )

    # Output mode options
    parser.add_argument(
        "--no-tui",
        action="store_true",
        default=False,
        help="Disable Rich TUI (plain text output for scripts/CI)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        default=False,
        help="Minimal output (implies --no-tui, ideal for headless/CI)",
    )
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        default=False,
        help="Automatically resume most recent incomplete session without prompting",
    )

    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="OpenAI model to use (default: gpt-4o-mini). Options: gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo",
        metavar="MODEL",
    )

    # Development/testing options
    parser.add_argument(
        "--scrape-only",
        action="store_true",
        default=False,
        help="Scrape and clean HTML only (no AI processing)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        default=False,
        help="Use mock API for TUI testing (development only)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"APIAS - API AUTO SCRAPER version {VERSION}",
    )

    args = parser.parse_args()

    # Handle --generate-config first
    if args.generate_config:
        from apias.config import generate_example_config

        generate_example_config("apias_config.yaml")
        print("Generated example configuration: apias_config.yaml")
        print(
            "Edit the file and use with: apias --config apias_config.yaml --url <URL>"
        )
        sys.exit(0)

    # Load configuration file if provided, with CLI overrides
    from apias.config import load_config

    cli_overrides = {
        "no_tui": args.no_tui if args.no_tui else None,
        "quiet": args.quiet if args.quiet else None,
        "auto_resume": args.auto_resume if args.auto_resume else None,
        "model": args.model,
        "limit": args.limit,
    }
    # Remove None values to let config file values take precedence
    cli_overrides = {k: v for k, v in cli_overrides.items() if v is not None}

    config = load_config(config_path=args.config, cli_overrides=cli_overrides)

    # Apply config values to args (CLI takes precedence if set)
    if not args.no_tui:
        args.no_tui = config.no_tui
    if not args.quiet:
        args.quiet = config.quiet
    if not args.auto_resume:
        args.auto_resume = config.auto_resume
    if args.limit is None:
        args.limit = config.limit

    # Quiet mode implies no_tui
    if args.quiet:
        args.no_tui = True

    # Validate required arguments
    if args.url is None and args.resume is None:
        parser.print_help()
        print("\nError: You must specify --url or --resume")
        sys.exit(1)

    if args.mode == "single" and (args.whitelist or args.blacklist):
        print("Error: --whitelist and --blacklist are only valid with --mode batch")
        sys.exit(1)

    # Handle resume mode
    if args.resume:
        if any([args.url, args.blacklist, args.whitelist]):
            print("Warning: When using --resume, other parameters are ignored.")
        start_resume_mode(args.resume)
    elif args.url and args.mode == "single":
        start_single_scrape(args.url, args.scrape_only, args.no_tui, args.mock)
    elif args.url and args.mode == "batch":
        # Auto-resume: check for existing sessions without prompting
        if args.auto_resume:
            resume_path = _find_auto_resume_session()
            if resume_path:
                if not args.quiet:
                    print(f"Auto-resuming session: {resume_path}")
                start_resume_mode(resume_path)
                print("Job Finished.\n")
                return
        elif not args.quiet and not args.no_tui:
            # Interactive: ask user if they want to resume
            resume_path = check_for_resumable_sessions()
            if resume_path:
                start_resume_mode(resume_path)
                print("Job Finished.\n")
                return

        # Start new batch scrape
        start_batch_scrape(
            args.url,
            args.whitelist,
            args.blacklist,
            args.scrape_only,
            args.no_tui,
            args.mock,
            args.limit,
        )
    else:
        parser.print_help()
        sys.exit(1)

    print("Job Finished.\n")


if __name__ == "__main__":
    main()
