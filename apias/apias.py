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
    llm service from OpenAI (model: gpt-4o-mini) it transform the html
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

    *the Tenacity library*
    You can install it manually with this command:

    python -m pip install --upgrade --upgrade-strategy only-if-needed tenacity

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
APP_NAME = "APIAS - API AUTO SCRAPER"
APP_FILENAME = "apias.py"
VERSION = "0.1.0"

import os
import sys
import time
import shutil
import logging
import argparse
import concurrent.futures
from typing import Dict, List, Optional, Tuple, Union, Any, Callable, Type, cast
from bs4 import BeautifulSoup, Comment
from types import TracebackType
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from requests.exceptions import RequestException
import fnmatch
import itertools
import platform
import re
import shlex
from types import FrameType
import subprocess
import threading
import xml.etree.ElementTree as ET
from signal import SIGINT, signal
import json


def validate_xml(xml_string):
    try:
        ET.fromstring(xml_string)
        return True
    except (ET.ParseError, ValueError):
        return False

def count_valid_xml_files(folder):
    valid_count = 0
    total_count = 0
    for xml_file in folder.glob('processed_*.xml'):
        total_count += 1
        with open(xml_file, 'r', encoding='utf-8') as f:
            content = f.read()
        if validate_xml(content):
            valid_count += 1
    return valid_count, total_count

# Global variables for cost tracking
total_cost = 0.0
cost_lock = threading.Lock()

# Global variable for progress tracking
progress_tracker: Dict[str, Dict[str, Union[str, float]]] = {}
progress_file = "progress.json"

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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create a temp folder with datetime suffix
temp_folder = Path(f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
temp_folder.mkdir(exist_ok=True)

# Create an error log file
error_log_file = temp_folder / "error_log.txt"
# Create an empty error log file
with open(error_log_file, 'w', encoding='utf-8') as f:
    pass

def get_openai_api_key() -> str:
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        logger.error("OPENAI_API_KEY environment variable not set.")
        sys.exit(1)
    return openai_api_key

# Global flag for graceful shutdown
shutdown_flag = False

def signal_handler(signum: int, frame: Optional[FrameType]) -> None:
    global shutdown_flag
    logger.info("Received interrupt signal. Initiating graceful shutdown...")
    shutdown_flag = True
    sys.exit(0)

signal(SIGINT, signal_handler)

# ============================
# Helper Functions from DSL
# ============================

def escape_xml(xml_doc: str) -> str:
    """
    Escapes XML special characters in a string.
    """
    xml_doc = xml_doc.replace('"', '&quot;')
    xml_doc = xml_doc.replace("'", '&apos;')
    xml_doc = xml_doc.replace('<', '&lt;')
    xml_doc = xml_doc.replace('>', '&gt;')
    xml_doc = xml_doc.replace('&', '&amp;')
    return xml_doc

def unescape_xml(xml_doc: str) -> str:
    """
    To unescape the text strings and the attributes values.
    DO NOT UNESCAPE CDATA, Comments and Processing Instructions
    """
    xml_doc = xml_doc.replace('&quot;', '"')
    xml_doc = xml_doc.replace('&apos;', "'")
    xml_doc = xml_doc.replace('&lt;', '<')
    xml_doc = xml_doc.replace('&gt;', '>')
    xml_doc = xml_doc.replace('&amp;', '&')
    return xml_doc

def extract_xml_from_input(input_data: str) -> str:
    """
    Extracts and validates XML content from the input string.
    """
    input_data = input_data.strip()
    if input_data.startswith("```xml") and input_data.endswith("```"):
        input_data = input_data[len('```xml'):-len('```')]
    elif input_data.startswith("```XML") and input_data.endswith("```"):
        input_data = input_data[len('```XML'):-len('```')]
    elif input_data.startswith("```") and input_data.endswith("```"):
        input_data = input_data[len('```'):-len('```')]

    input_data = input_data.replace('\\\\n', '\n').replace('\\n', '\n')
    input_data = input_data.replace('\\\\\\"', '\\\\"').replace('\\\"', '"')

    xml_content = input_data.strip()
    if not (xml_content.lower().startswith('<?xml') or xml_content.lower().startswith('<xml')):
        xml_content = '<?xml version="1.0" encoding="UTF-8"?>\n<XML>\n' + xml_content + '\n</XML>'

    # Handle code tags
    xml_content = re.sub(r'<code>(.*?)</code>', lambda m: f'<CODE>{escape_xml(m.group(1))}</CODE>', xml_content, flags=re.DOTALL | re.IGNORECASE)

    # Validate the XML content
    logger.debug(f"Extracted XML Content:\n{xml_content}")
    try:
        ET.fromstring(xml_content)  # Validates XML
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
        input_data = input_data[len('```xml'):-len('```')]
    elif input_data.startswith("```XML") and input_data.endswith("```"):
        input_data = input_data[len('```XML'):-len('```')]
    elif input_data.startswith("```") and input_data.endswith("```"):
        input_data = input_data[len('```'):-len('```')]

    input_data = input_data.replace('\\\\n', '\n').replace('\\n', '\n')
    input_data = input_data.replace('\\\\\\"', '\\\\"').replace('\\\"', '"')

    xml_content = input_data
    if not (xml_content.lower().startswith('<xml') and xml_content.lower().endswith('xml>')):
        xml_content = '<XML version="1.0" encoding="UTF-8" standalone="yes" >\n' + xml_content + '\n</XML>'

    # Validate the XML content
    logger.debug(f"Extracted Iteration XML Content:\n{input_data}")
    try:
        ET.fromstring(xml_content)  # Validates XML
    except ET.ParseError:
        logger.error("ERROR - Extracted XML is not valid")
    finally:
        return xml_content

def merge_xmls(temp_folder: Path) -> str:
    """
    Merges multiple XML documents from the temp folder into a single XML API document.
    Includes the source URL as the first child node for each document.
    """
    root = ET.Element("TEXTUAL_API")
    error_log = []

    for xml_file in temp_folder.glob('processed_*.xml'):
        try:
            try:
                with open(xml_file, 'r', encoding='utf-8') as f:
                    xml_content = f.read()
            except UnicodeDecodeError:
                logger.warning(f"Unicode decode error in file {xml_file}. Trying with 'latin-1' encoding.")
                with open(xml_file, 'r', encoding='latin-1') as f:
                    xml_content = f.read()
            
            # Parse the XML content
            doc = ET.fromstring(xml_content)
            
            # Create a new element for this document
            doc_element = ET.SubElement(root, "DOCUMENT")
            
            # Add the source URL as the first child
            source_url = doc.find('.//SOURCE_URL')
            if source_url is not None:
                doc_element.append(source_url)
            
            # Append all other elements
            for child in doc:
                if child.tag != 'SOURCE_URL':
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

    # Write error log
    with open(temp_folder / 'errors.log', 'w', encoding='utf-8') as f:
        f.write('\n'.join(error_log))

    merged_xml = ET.tostring(root, encoding='unicode', method='xml')
    return merged_xml

def remove_links(element: ET.Element) -> None:
    """
    Recursively removes all link elements from the given XML element.
    """
    for child in list(element):
        if child.tag.lower() in ['a', 'link']:
            element.remove(child)
        else:
            remove_links(child)


# Navigation keywords limited to English
NAVIGATION_KEYWORDS = [
    'nav', 'menu', 'sidebar', 'footer', 'breadcrumb', 'pager', 'pagination',
    'header', 'navigation', 'submenu', 'tabs', 'navbar', 'main-navigation',
    'site-navigation', 'skip-navigation', 'top-nav', 'bottom-nav', 'side-nav',
    'advert', 'ads', 'sponsor', 'related', 'cookies', 'banner'
]

# Hidden styles, classes, and attributes
HIDDEN_STYLES = [
    'display:none', 'visibility:hidden', 'opacity:0',
    'height:0', 'width:0', 'position:absolute'
]
HIDDEN_CLASSES = [
    'hidden', 'collapsed', 'd-none', 'invisible',
    'sr-only', 'visually-hidden', 'offscreen'
]
HIDDEN_ATTRIBUTES = ['hidden', 'aria-hidden', 'data-hidden']

def is_main_content(element: BeautifulSoup) -> bool:
    """Check if the element is the main content."""
    if element.name == 'main' or element.get('role') == 'main':
        return True
    for attr in ['id', 'class']:
        attr_values = element.get(attr, [])
        if isinstance(attr_values, list):
            attr_values = ' '.join(attr_values)
        if attr_values and any(keyword in attr_values.lower() for keyword in ['content', 'main']):
            return True
    return False

def has_significant_text(element: BeautifulSoup, threshold: int = 100) -> bool:
    """Determine if an element contains significant text content."""
    text_length = len(element.get_text(strip=True))
    return text_length > threshold

def contains_navigation_keywords(attr_values: Union[str, List[str]]) -> bool:
    """Check if attribute values contain any navigation keywords."""
    if isinstance(attr_values, list):
        attr_values = ' '.join(attr_values)
    attr_values_lower = attr_values.lower()
    return any(keyword in attr_values_lower for keyword in NAVIGATION_KEYWORDS)

def is_navigation(element: BeautifulSoup) -> bool:
    """Determine if an element is likely a navigation or non-content element."""
    if is_main_content(element):
        return False
    if element.name in ['nav', 'header', 'footer', 'aside']:
        return True
    for attr in ['class', 'id', 'role', 'aria-label']:
        attr_values = element.get(attr, [])
        if attr_values and contains_navigation_keywords(attr_values):
            return True
    links = element.find_all('a')
    text_length = len(element.get_text(strip=True))
    if len(links) > 5 and text_length < 100:
        return True
    if element.find_all('ul') and not has_significant_text(element):
        return True
    if element.name == 'div':
        class_attr = element.get('class', [])
        if isinstance(class_attr, list) and any(
            any(keyword in cls.lower() for keyword in ['advert', 'ads', 'sponsor'])
            for cls in class_attr
        ):
            return True
    return False

def clean_styles(styles: str, hidden_styles: List[str]) -> str:
    """Remove hidden styles from the style attribute."""
    style_list = [s.strip() for s in styles.split(';') if s.strip()]
    visible_styles = [
        s for s in style_list if not any(hidden_style in s.replace(' ', '') for hidden_style in hidden_styles)
    ]
    return '; '.join(visible_styles)

def expand_hidden_content(soup: BeautifulSoup):
    """Expand hidden content by removing styles, classes, and attributes that hide elements."""
    for element in soup.find_all(True):
        # Remove hidden styles
        if element.has_attr('style'):
            cleaned_style = clean_styles(element['style'], HIDDEN_STYLES)
            if cleaned_style:
                element['style'] = cleaned_style
            else:
                del element['style']
        # Remove hidden classes
        if element.has_attr('class'):
            visible_classes = [cls for cls in element['class'] if cls.lower() not in HIDDEN_CLASSES]
            if visible_classes:
                element['class'] = visible_classes
            else:
                del element['class']
        # Remove hidden attributes
        for attr in HIDDEN_ATTRIBUTES:
            element.attrs.pop(attr, None)

def remove_elements(soup: BeautifulSoup, condition_func):
    """Remove elements from the soup based on a condition function."""
    elements_to_remove = soup.find_all(condition_func)
    for element in elements_to_remove:
        element.decompose()

def remove_comments(soup: BeautifulSoup):
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
    soup = BeautifulSoup(html_content, 'lxml')

    # Remove comments
    remove_comments(soup)

    # Expand hidden content
    expand_hidden_content(soup)

    # Remove navigation and non-content elements
    remove_elements(soup, is_navigation)

    # Remove script and style tags
    for element in soup(['script', 'style']):
        element.decompose()

    # Preserve formatting in pre and code tags
    for tag in soup.find_all(['pre', 'code']):
        tag.string = tag.prettify(formatter='html')

    # Get the compact HTML string
    compact_html = soup.decode(formatter='minimal')

    # Remove excessive whitespace, except within pre and code tags
    compact_html = re.sub(r'\s+(?![^<>]*</(?:pre|code)>)', ' ', compact_html)

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
        spinner = Spinner("Installing...")

        assert process.stdout is not None
        while True:
            char = process.stdout.read(1)
            if not char:
                break

            output.append(char)
            spinner.step()

        spinner.end()
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

    def __init__(self, text: str) -> None:
        self.text = text
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.spinner_thread = threading.Thread(target=self._spin)
        self.spinner_thread.daemon = True

    def __enter__(self) -> 'Spinner':
        self.start()
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_value: Optional[BaseException], traceback: Optional[TracebackType]) -> None:
        self.stop()

    def start(self) -> None:
        print("Press CTRL-C to stop the processing.")
        self.spinner_thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.pause_event.set()
        self.spinner_thread.join()
        self._clear_line()

    def pause(self) -> None:
        self.pause_event.set()

    def resume(self) -> None:
        self.pause_event.clear()

    def _spin(self) -> None:
        while not self.stop_event.is_set():
            if not self.pause_event.is_set():
                self._clear_line()
                print(f"\r{self.text} {next(self.spinner_chars)}", end="", flush=True)
            time.sleep(0.1)

    def update_text(self, new_text: str) -> None:
        self.pause()
        self._clear_line()
        print(f"\r{new_text}")
        self.text = new_text
        self.resume()

    def step(self) -> None:
        self.pause()
        self._clear_line()
        print(f"\r{self.text} {next(self.spinner_chars)}", end="", flush=True)
        self.resume()

    def end(self) -> None:
        self.stop()

    def _clear_line(self) -> None:
        print("\r" + " " * (shutil.get_terminal_size().columns - 1), end="", flush=True)

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

    def __init__(self, playwright_available: Optional[bool] = None, verify_ssl: bool = True, timeout: int = 30):
        """
        `print_error` - a function to call to print error/debug info.
        `verify_ssl` - if False, disable SSL certificate verification when scraping.
        `timeout` - timeout in seconds for the scraping operation.
        """

        self.print_error: Callable[[str], None] = print

        self.playwright_available = playwright_available if playwright_available is not None else install_playwright()
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
            if any(re.search(pattern, content, re.IGNORECASE) for pattern in html_patterns):
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

        with Spinner(f"Scraping {url}") as spinner:
            try:
                with sync_playwright() as p:
                    try:
                        browser = p.chromium.launch()
                    except Exception as e:
                        self.playwright_available = False
                        self.print_error(str(e))
                        return None, None

                    try:
                        context = browser.new_context(ignore_https_errors=not self.verify_ssl)
                        page = context.new_page()

                        user_agent = page.evaluate("navigator.userAgent")
                        user_agent = user_agent.replace("Headless", "")
                        user_agent = user_agent.replace("headless", "")
                        #user_agent += " " + custom_user_agent

                        page.set_extra_http_headers({"User-Agent": user_agent})

                        response = None
                        try:
                            spinner.update_text(f"Loading {url}")
                            response = page.goto(url, wait_until="networkidle", timeout=5000)
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

def slimdown_html(page_source: str) -> Tuple[str, Optional[str], List[str], List[str], List[str], List[str], List[Tuple[str, str]]]:

    # Clean the HTML content from navigation elements and ads
    cleaned_html = clean_html(page_source)

    soup = BeautifulSoup(cleaned_html, "html.parser")

    # Remove SVG elements
    for svg in soup.find_all("svg"):
        svg.decompose()

    # Extract and remove images
    images = []
    for img in soup.find_all("img"):
        if 'src' in img.attrs:
            images.append(img['src'])
        img.decompose()

    # Remove data URIs
    for tag in soup.find_all(href=lambda x: x and x.startswith("data:")):
        tag.decompose()
    for tag in soup.find_all(src=lambda x: x and x.startswith("data:")):
        tag.decompose()

    # Remove script and style tags
    for tag in soup.find_all(['script', 'style']):
        tag.decompose()

    # Extract code examples
    code_examples = []
    for code in soup.find_all('pre'):
        code_examples.append(code.get_text())
        code['class'] = code.get('class', []) + ['extracted-code']

    # Extract method signatures
    method_signatures = []
    for method in soup.find_all('div', class_='method-signature'):
        method_signatures.append(method.get_text())
        method['class'] = method.get('class', []) + ['extracted-method']

    # Extract class definitions
    class_definitions = []
    for class_def in soup.find_all('div', class_='class-definition'):
        class_definitions.append(class_def.get_text())
        class_def['class'] = class_def.get('class', []) + ['extracted-class']

    # Extract links
    links = []
    for a in soup.find_all('a', href=True):
        links.append((a['href'], a.get_text()))

    # Remove all attributes except href and class
    for tag in soup.find_all(True):
        for attr in list(tag.attrs):
            if attr not in ['href', 'class']:
                tag.attrs.pop(attr, None)
    
    # Remove empty tags
    for tag in soup.find_all():
        if len(tag.get_text(strip=True)) == 0 and tag.name not in ['br', 'hr']:
            tag.decompose()

    return (str(soup), 
            soup.title.string if soup.title else "scraped_page",
            code_examples,
            method_signatures,
            class_definitions,
            images,
            links)

def start_scraping(url: str) -> Optional[str]:
    url = url.strip()
    if not url:
        print("Please provide a URL to scrape.")
        return None

    print(f"Scraping {url}...")
    
    res = install_playwright()
    if not res:
        print("Unable to initialize playwright.")
        return None

    scraper = Scraper(
        playwright_available=res, verify_ssl=False
    )

    try:
        content, mime_type = scraper.scrape(url)
        if content:
            content = f"{url}:\n\n" + content
            print("... done.")
            return content
        else:
            print("No content retrieved.")
            return None
    except Exception as e:
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
        return [pattern.strip() for pattern in pattern_str.split(",") if pattern.strip()]

    whitelist_patterns = process_pattern_list(whitelist_str)
    blacklist_patterns = process_pattern_list(blacklist_str)

    # Fetch and parse sitemap
    if sitemap_content:
        logger.info("Parsing sitemap content from provided string.")
        try:
            root = ET.fromstring(sitemap_content)
        except ET.ParseError as e:
            logger.error(f"Error parsing sitemap content: {e}")
            return []
    elif sitemap_file:
        logger.info(f"Parsing sitemap from file: {sitemap_file}")
        if not os.path.isfile(sitemap_file):
            logger.error(f"Sitemap file '{sitemap_file}' does not exist.")
            return []
        try:
            tree = ET.parse(sitemap_file)
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
                include_url = any(fnmatch.fnmatch(url, pattern) for pattern in whitelist_patterns)

            # Apply blacklist
            if include_url and blacklist_patterns:
                include_url = not any(fnmatch.fnmatch(url, pattern) for pattern in blacklist_patterns)

            if include_url:
                urls.append(url)

    logger.info(f"Extracted {len(urls)} URLs from sitemap.")
    return urls

# ============================
# LLM Processing using OpenAI
# ============================

# Load model pricing info with retry
@retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=1, min=4, max=30))
def load_model_pricing() -> Optional[Dict[str, Any]]:
    """Attempt to extract JSON from a string."""
    pricing_urls = [
        'https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json',
        'https://cdn.jsdelivr.net/gh/BerriAI/litellm@main/model_prices_and_context_window.json',
        'https://raw.fastgit.org/BerriAI/litellm/main/model_prices_and_context_window.json'
    ]
    for pricing_url in pricing_urls:
        try:
            response = requests.get(pricing_url, timeout=10)
            response.raise_for_status()
            logger.info(f"Successfully fetched model pricing information from {pricing_url}")
            return cast(Dict[str, Any], response.json())
        except requests.RequestException as e:
            logger.warning(f"Error fetching model pricing information from {pricing_url}: {e}")
    raise RuntimeError("Failed to fetch model pricing information from all available mirrors.")



@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RequestException, TimeoutError))
)
def make_openai_request(api_key: str, prompt: str, pricing_info: Dict[str, Dict[str, float]], model: str="gpt-4o-mini") -> Dict[str, Any]:
    global total_cost
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    url = "https://api.openai.com/v1/chat/completions"
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an assistant that converts HTML to XML."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=600)  # Increased timeout to 600 seconds
        response.raise_for_status()
        response_json = response.json()
        
        # Compute cost
        usage = response_json.get('usage', {})
        prompt_tokens = usage.get('prompt_tokens', 0)
        completion_tokens = usage.get('completion_tokens', 0)
        
        model_pricing = pricing_info.get(model, {})
        input_cost = prompt_tokens * model_pricing.get('input_cost_per_token', 0)
        output_cost = completion_tokens * model_pricing.get('output_cost_per_token', 0)
        request_cost = input_cost + output_cost
        
        with cost_lock:
            total_cost += request_cost
        
        logger.info(f"OpenAI API request successful. Cost: ${request_cost:.6f}")
        logger.info(f"Total cost so far: ${total_cost:.6f}")
        
        response_json['request_cost'] = request_cost
        response_json['finish_reason'] = response_json['choices'][0]['finish_reason']
        
        return cast(Dict[str, Any], response_json)
    except RequestException as e:
        logger.error(f"OpenAI API request failed: {e}")
        raise
    except TimeoutError as e:
        logger.error(f"OpenAI API request timed out: {e}")
        raise

def call_openai_api(prompt: str, pricing_info: Dict[str, Dict[str, float]]) -> Tuple[str, float]:
    """
    Makes a request to the OpenAI API using the requests library.
    """
    api_key = get_openai_api_key()
    response_json = make_openai_request(api_key, prompt, pricing_info=pricing_info)
    content = str(response_json['choices'][0]['message']['content']).strip()
    request_cost = response_json.get('request_cost', 0)
    return content, request_cost

def call_llm_to_convert_html_to_xml(html_content: str, additional_content: Dict[str, List[str]], pricing_info: Dict[str, Dict[str, float]]) -> Tuple[Optional[str], float]:
    """
    Uses OpenAI's API to convert HTML content to structured XML.
    """
    prompt = f"""
You must examine the following html file and extract its content into a XML API document. The page is about the API of the python Textual library. There are detailed informations about the python classes and their usage. The XML output should provide a structured version of the documentation contained in the page. It should contains all the informations found in the html document, excluding side menus, indexes, and or other navigation elements of the page, without missing anything of the content. Ignore images or data encoded in base64. Remove links. The XML output should be machine readable. There can be only two html page classifications: <CLASS> or <API_USAGE>. If the html page is about a introduction, tutorial, guide, example, how-to, quickstart, etc. then you must wrap the content inside the <API_USAGE> tags. If the html page is about the class description and its members, then you must wrap the content inside the <CLASS> tags, and inside it you should divide it in two sub branches: <CLASS_DESCRIPTION> and <CLASS_API> tags. Inside the <API_USAGE> you must categorize simply with a division in <SUB_SECTION> tags, each one with inside the <TITLE> of the subject followed by its content. Inside the usage examples sections you can identify sub section titles by the h1, h2, h3, etc. tags. In the CLASS API sections instead, the h1, h2, h3, etc headers are likely to be class members or member descriptions, but there can be exceptions. Sometimes the html page describes a module instead of a class. In this case wrap the section with the <MODULE> tags, and the members with <FUNCTION>, <VARIABLE>, <CONSTANT>, etc. indicating name, signature, return types and modifiers when available. You must explicitly indicate the type of the member and its return type when available, and possibly the modifiers. Do not miss any entry. Completeness is mandatory: even if some information about a type, member, a method or a return type is missing from the html page because it is implicit, you must infer it from the context and explicitly declare it in the XML document. Tables must be converted to XML structures. Remember to escape XML characters of all text inside the XML text strings and attributes values. You must escape the text strings and the attribute values exactly as this python function:

```python
def escape_xml(xml_doc: str) -> str:
    # to escape the text strings and the attributes values.
    # DO NOT ESCAPE CDATA, Comments and Processing Instructions
    xml_doc = xml_doc.replace('"', '&quot;')
    xml_doc = xml_doc.replace("'", '&apos;')
    xml_doc = xml_doc.replace('<', '&lt;')
    xml_doc = xml_doc.replace('>', '&gt;')
    xml_doc = xml_doc.replace('&', '&amp;')
    return xml_doc
```
For example writing an XML string like this one is wrong:
```xml
<DESCRIPTION>Grow space by (<top>, <right>, <bottom>, <left>).</DESCRIPTION>
```
Instead you should write:
```xml
<DESCRIPTION>Grow space by (&lt;top&gt;, &lt;right&gt;, &lt;bottom&gt;, &lt;left&gt;).</DESCRIPTION>
```
Be exhaustive and accurate. Do not allucinate or misinterpret the content. Be sure to produce a solid reliable XML document. DO NOT ADD, ANNOTATE OR COMMENT ANYTHING. Your output must be only the document in XML format, nothing else.

<TEXT>
{html_content}
"""

    max_retries = 3
    for attempt in range(max_retries):
        try:
            xml_output, request_cost = call_openai_api(prompt, pricing_info)
            xml_output = extract_xml_from_input(xml_output)
            return xml_output, request_cost
        except Exception as e:
            logger.error(f"LLM processing failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                logger.error("All attempts to process with LLM failed. Returning None.")
                return None, 0.0
            time.sleep(5)  # Wait for 5 seconds before retrying
    return None, 0.0


# ============================
# Web Scraper Replacement
# ============================

def web_scraper(url: str) -> Optional[str]:
    try:
        content = start_scraping(url)
        if content:
            content_size = len(content) / 1024  # Convert to KB
            logger.info(f"Scraping successful for {url}. {content_size:.2f} KB scraped.")
            return content
        else:
            logger.error(f"No content retrieved from {url}")
            return None
    except Exception as e:
        logger.error(f"Error scraping {url}: {str(e)}", exc_info=True)
        return None

# ============================
# Main Workflow Functions
# ============================

# ============================
# Sitemap Processing Functions
# ============================

def fetch_sitemap(urls: List[str]) -> Optional[str]:
    """
    Fetches the sitemap.xml from the base domain of the given URLs using HTTP requests.
    """
    for url in urls:
        parsed_url = urlparse(url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        sitemap_url = base_url.rstrip('/') + '/sitemap.xml'
        logger.info(f"Fetching sitemap from {sitemap_url}")
        try:
            with Spinner("Fetching sitemap...") as spinner:
                response = requests.get(sitemap_url, timeout=10)
                response.raise_for_status()
            logger.info("Fetched sitemap successfully.")
            return response.text
        except requests.RequestException as e:
            logger.error(f"Failed to fetch sitemap.xml from {sitemap_url}: {e}")
    logger.error("Failed to fetch sitemap.xml from all provided URLs")
    return None

# ============================
# Main Workflow Functions
# ============================

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def process_single_page(url: str, pricing_info: Dict[str, Dict[str, float]]) -> Optional[str]:
    """
    Processes a single page: scrapes, converts to XML via LLM, and saves the result.
    """
    logger.info("Processing single page.")
    
    html_content: Optional[str] = None
    xml_content: Optional[str] = None
    
    def process_in_background() -> None:
        nonlocal html_content, xml_content
        html_content = web_scraper(url)
        if not html_content:
            logger.error("Failed to retrieve HTML content for single page processing.")
            return
        
        slimmed_html, page_title, code_examples, method_signatures, class_definitions, images, links = slimdown_html(html_content)
        
        # Prepare additional content for LLM
        additional_content: Dict[str, List[str]] = {
            "code_examples": code_examples,
            "method_signatures": method_signatures,
            "class_definitions": class_definitions,
            "images": images,
            "links": [f"{href}: {text}" for href, text in links]
        }
        
        result = call_llm_to_convert_html_to_xml(slimmed_html, additional_content, pricing_info)
        if result is None:
            logger.error("Failed to convert HTML to XML.")
            return None
        xml_content, _ = result
        
        if xml_content:
            # Include source URL in XML content
            xml_content = f'<SOURCE_URL>{url}</SOURCE_URL>\n' + xml_content
            
            # Save individual XML file
            xml_file = temp_folder / "processed_single_page.xml"
            with open(xml_file, 'w', encoding='utf-8') as f:
                f.write(xml_content)
    
    with Spinner("Processing page...") as spinner:
        thread = threading.Thread(target=process_in_background)
        thread.start()
        while thread.is_alive():
            spinner.step()
            time.sleep(0.1)
        thread.join()

    if xml_content is not None:
        merged_xml = merge_xmls(temp_folder)
        merged_xml_file = temp_folder / "merged_output.xml"
        with open(merged_xml_file, 'w', encoding='utf-8') as f:
            f.write(merged_xml)
        logger.info(f"Merged XML saved to {merged_xml_file}")
    else:
        logger.error("Failed to convert HTML to XML. No XML content generated.")
    return xml_content

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10), retry=retry_if_exception_type(RequestException))
def process_url(url: str, idx: int, total: int, pricing_info: Dict[str, Dict[str, float]]) -> Optional[str]:
    global progress_tracker
    """
    Process a single URL: scrape, convert to XML, and save temp files.
    """
    global shutdown_flag, progress_tracker
    logger.info(f"Processing URL {idx}/{total}: {url}")
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

        logger.info(f"Scraping HTML content for URL: {url}")
        html_content = web_scraper(url)
        if not html_content:
            logger.warning(f"Failed to retrieve HTML content for {url}")
            progress_tracker[url] = {"status": "failed", "cost": progress_tracker[url].get("cost", 0.0)}
            with cost_lock:
                update_progress_file()
            return None
        
        logger.info(f"Saving HTML content for URL: {url}")
        html_file = temp_folder / f"scraped_{idx}.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        if shutdown_flag:
            logger.info(f"Shutdown requested. Skipping LLM processing for URL {url}")
            progress_tracker[url]["status"] = "pending"
            update_progress_file()
            return None

        logger.info(f"Converting HTML to XML for URL: {url}")
        additional_content: Dict[str, List[str]] = {}
        result = call_llm_to_convert_html_to_xml(html_content, additional_content, pricing_info)
        if result is None:
            logger.warning(f"Failed to convert HTML to XML for {url}")
            progress_tracker[url] = {"status": "failed", "cost": progress_tracker[url].get("cost", 0.0)}
            update_progress_file()
            return None
        
        xml_content, request_cost = result
        
        with cost_lock:
            progress_tracker[url] = {"status": progress_tracker[url].get("status", "pending"), "cost": float(progress_tracker[url].get("cost", 0.0)) + request_cost}
        
        # Include source URL in XML content
        if xml_content is not None:
            xml_content = xml_content.replace('<?xml version="1.0" encoding="UTF-8"?>\n<XML>', f'<?xml version="1.0" encoding="UTF-8"?>\n<XML>\n<SOURCE_URL>{url}</SOURCE_URL>')
            logger.info(f"Saving XML content for URL: {url}")
            xml_file = temp_folder / f"processed_{idx}.xml"
            with open(xml_file, 'w', encoding='utf-8') as f:
                f.write(xml_content)
        else:
            logger.warning(f"No XML content generated for URL: {url}")
            progress_tracker[url] = {"status": "failed", "cost": progress_tracker[url].get("cost", 0.0)}
            update_progress_file()
            return None
        
        if xml_content:
            logger.info(f"Saving XML content for URL: {url}")
            xml_file = temp_folder / f"processed_{idx}.xml"
            with open(xml_file, 'w', encoding='utf-8') as f:
                f.write(xml_content)
        else:
            logger.warning(f"No XML content to save for URL: {url}")
        
        logger.info(f"Successfully processed URL {idx}/{total}: {url}")
        is_valid_xml = validate_xml(xml_content)
        progress_tracker[url] = {"status": "successful", "cost": progress_tracker[url].get("cost", 0.0), "valid_xml": is_valid_xml}
        update_progress_file()
        return xml_content
    except RequestException as e:
        error_message = f"Request error processing URL {url}: {str(e)}"
        logger.error(error_message)
        with open(error_log_file, 'a', encoding='utf-8') as f:
            f.write(f"{error_message}\n")
        progress_tracker[url] = {"status": "failed", "cost": progress_tracker[url].get("cost", 0.0)}
        update_progress_file()
        return None
    except Exception as e:
        error_message = f"Error processing URL {url}: {str(e)}"
        logger.error(error_message)
        with open(error_log_file, 'a', encoding='utf-8') as f:
            f.write(f"{error_message}\n")
        progress_tracker[url] = {"status": "failed", "cost": progress_tracker[url].get("cost", 0.0)}
        update_progress_file()
        return None

def update_progress_file():
    global progress_file, temp_folder
    if progress_file:
        with cost_lock:
            with open(progress_file, 'w') as f:
                json.dump({
                    "output_folder": str(temp_folder),
                    "urls": [{"index": i, "url": url, "status": data["status"], "costs": data["cost"], "valid_xml": data.get("valid_xml")} 
                             for i, (url, data) in enumerate(progress_tracker.items())]
                }, f, indent=2)
    else:
        logger.warning("Progress file path not set. Unable to update progress.")

def process_multiple_pages(urls: List[str], pricing_info: Dict[str, Dict[str, float]], num_threads: int = 5) -> Optional[str]:
    """
    Processes multiple pages: scrapes each, converts to XML via LLM, and merges.
    """
    global shutdown_flag
    logger.info(f"Processing multiple pages: {len(urls)} URLs found using {num_threads} threads.")
    xml_list = []
    
    spinner = Spinner("Processing URLs")
    spinner.start()
    
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            future_to_url = {executor.submit(process_url, url, idx+1, len(urls), pricing_info): url for idx, url in enumerate(urls)}
            for future in concurrent.futures.as_completed(future_to_url):
                if shutdown_flag:
                    logger.info("Shutdown requested. Cancelling remaining tasks.")
                    executor.shutdown(wait=False)
                    break
                
                url = future_to_url[future]
                try:
                    xml_content = future.result()
                    if xml_content:
                        xml_list.append(xml_content)
                    spinner.update_text(f"Processed {len(xml_list)}/{len(urls)} URLs")
                except Exception as e:
                    logger.error(f"Unhandled exception for URL {url}: {str(e)}")
    finally:
        spinner.end()
    
    if shutdown_flag:
        logger.info("Shutdown initiated. Saving partial results.")
    
    logger.info("Merging XML content from all processed pages")
    merged_xml = merge_xmls(temp_folder)
    
    if not merged_xml or merged_xml == "<TEXTUAL_API />":
        logger.error("No valid XML content extracted from pages.")
        return None
    
    logger.info("Saving merged XML output")
    merged_xml_file = temp_folder / "merged_output.xml"
    try:
        temp_folder.mkdir(parents=True, exist_ok=True)
        with open(merged_xml_file, 'w', encoding='utf-8') as f:
            f.write(merged_xml)
        logger.info(f"Merged XML saved successfully to {merged_xml_file}")
    except IOError as e:
        logger.error(f"Error saving merged XML file: {e}")
        return None
    
    return merged_xml


def read_patterns_from_file(file_path: str) -> Optional[str]:
    """
    Read patterns from a file and return them as a comma-separated string.
    """
    if not file_path:
        return None
    try:
        with open(file_path, 'r') as f:
            patterns = [line.strip() for line in f if line.strip()]
        if not patterns:
            logger.warning(f"No valid patterns found in {file_path}")
            return None
        return ','.join(patterns)
    except IOError as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return None


def display_scraping_summary(result: Dict[str, Any], urls: List[str], temp_folder: Path, error_log_file: Path) -> None:
    summary_data = {
        "Base URL": urls[0] if urls else "N/A",
        "Sitemap.xml URL": f"{urls[0].rstrip('/')}/sitemap.xml" if urls else "N/A",
        "Number of URLs before filtering": str(result.get("total_urls", "N/A")),
        "Number of URLs after filtering": str(result.get("filtered_urls", "N/A")),
        "Number of Scraped URLs": str(result.get("scraped_urls", "N/A")),
        "Number of Successfully Processed URLs": str(result.get("successful_urls", "N/A")),
        "Number of URLs failed to scrape": str(result.get("failed_urls", "N/A")),
        "Total number of XML files generated": str(result.get("total_xml_files", "N/A")),
        "Valid XML files generated": str(result.get("valid_xml_files", "N/A")),
        "Invalid XML files generated": str(result.get("invalid_xml_files", "N/A")),
        "Total costs of the scraping job": f"${result['total_cost']:.6f}",
        "Temporary folder path": str(temp_folder),
        "Error log file path": str(error_log_file),
        "Merged XML file path": str(temp_folder / 'merged_output.xml')
    }
    
    summary_box = create_summary_box(summary_data)

    if result["result"] == "already_completed":
        print(SUCCESS_SEPARATOR)
        print("✅ Scraping process already completed successfully.")
        print(summary_box)
        print(SUCCESS_SEPARATOR)
    elif result["result"]:
        print(SUCCESS_SEPARATOR)
        print("✅ XML Extraction Successful.")
        print(summary_box)
        print(SUCCESS_SEPARATOR)
    else:
        print(ERROR_SEPARATOR)
        print("❌ XML Extraction Failed or No Content Extracted.")
        print(summary_box)
        print(ERROR_SEPARATOR)



def main_workflow(
    urls: List[str],
    mode: str = "single",
    whitelist: Optional[str] = None,
    blacklist: Optional[str] = None,
    num_threads: int = 5,
    resume_file: Optional[str] = None
) -> Dict[str, Union[Optional[str], float, int]]:
    global progress_tracker, total_cost
    """
    Executes the Web API Retrieval and XML Extraction workflow.
    """
    global shutdown_flag, total_cost, progress_tracker, progress_file, temp_folder
    print(INFO_SEPARATOR)
    logger.info("Starting Web API Retrieval workflow.")
    print(INFO_SEPARATOR)

    result: Dict[str, Union[Optional[str], float, int]] = {
        "result": None,
        "total_cost": 0.0,
        "total_urls": 0,
        "filtered_urls": 0,
        "scraped_urls": 0,
        "successful_urls": 0,
        "failed_urls": 0,
        "total_xml_files": 0,
        "valid_xml_files": 0,
        "invalid_xml_files": 0
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
                with open(resume_file, 'r') as f:
                    resume_data = json.load(f)
                temp_folder = Path(resume_data.get("output_folder", temp_folder))
                progress_tracker = {url_data["url"]: {"status": url_data["status"], "cost": float(url_data["costs"]), "valid_xml": url_data.get("valid_xml")} 
                                    for url_data in resume_data["urls"]}
                progress_file = resume_file
                urls = [url_data["url"] for url_data in resume_data["urls"]]
                
                # Check if all URLs are already processed successfully
                successful_urls = sum(1 for url_data in progress_tracker.values() if url_data.get("status") == "successful")
                total_urls = len(urls)
                total_cost = sum(float(url_data.get("cost", 0.0)) for url_data in progress_tracker.values())
                
                if successful_urls == total_urls:
                    logger.info(f"Scraping process already completed successfully ({successful_urls}/{total_urls}). Total costs: ${total_cost:.6f}")
                    result.update({
                        "result": "already_completed",
                        "total_cost": total_cost,
                        "total_urls": total_urls,
                        "filtered_urls": total_urls,
                        "scraped_urls": successful_urls,
                        "successful_urls": successful_urls,
                        "failed_urls": 0,
                        "total_xml_files": successful_urls,
                        "valid_xml_files": successful_urls,
                        "invalid_xml_files": 0
                    })
                    result["result"] = cast(Optional[str], result["result"])
                    return result
                elif successful_urls > 0:
                    logger.info(f"Scraping process partially completed. ({successful_urls} successful, {total_urls - successful_urls} failed). Total costs so far: ${total_cost:.6f}")
                    user_input = input("Do you want to attempt the failed URLs again? (y/n): ").lower()
                    if user_input != 'y':
                        logger.info("Exiting without processing failed URLs.")
                        result.update({
                            "total_cost": total_cost,
                            "total_urls": total_urls,
                            "filtered_urls": total_urls,
                            "scraped_urls": successful_urls,
                            "successful_urls": successful_urls,
                            "failed_urls": total_urls - successful_urls,
                            "total_xml_files": successful_urls,
                            "valid_xml_files": successful_urls,
                            "invalid_xml_files": 0
                        })
                        return result
                    urls = [url for url, data in progress_tracker.items() if data.get("status") != "successful"]
                    logger.info(f"Resuming scraping for {len(urls)} failed URLs")
                else:
                    logger.info(f"Resuming scraping for {len(urls)} URLs")
            else:
                logger.info(f"Resume file {resume_file} does not exist. Creating a new one.")
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
        with open(error_log_file, 'w', encoding='utf-8') as f:
            pass

        # Process whitelist and blacklist inputs
        whitelist_str = read_patterns_from_file(whitelist) if whitelist else None
        blacklist_str = read_patterns_from_file(blacklist) if blacklist else None

        # Extract and filter URLs if in batch mode
        if mode == "batch":
            print(SEPARATOR)
            logger.info("Fetching sitemap")
            if not urls:
                logger.error("No URLs provided for batch mode.")
                return result
            base_url = urls[0]
            sitemap_content = fetch_sitemap([base_url])
            if not sitemap_content:
                logger.error("Sitemap retrieval failed. Exiting workflow.")
                return result
            logger.info("Extracting URLs from sitemap")
            extracted_urls = extract_urls_from_sitemap(
                sitemap_content=sitemap_content,
                sitemap_file=None,
                whitelist_str=whitelist_str,
                blacklist_str=blacklist_str
            )
            if extracted_urls:
                result["total_urls"] = len(extracted_urls)
                urls = extracted_urls
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
            xml_result = process_single_page(urls[0], pricing_info)
            if xml_result:
                is_valid = validate_xml(xml_result)
                print(f"The XML file was successfully generated and it is {'valid' if is_valid else 'non valid'} XML")
                result.update({
                    "result": xml_result,
                    "scraped_urls": 1,
                    "successful_urls": 1,
                    "failed_urls": 0,
                    "total_xml_files": 1,
                    "valid_xml_files": 1 if is_valid else 0,
                    "invalid_xml_files": 0 if is_valid else 1
                })
                result["result"] = cast(Optional[str], result["result"])
            else:
                result.update({
                    "scraped_urls": 1,
                    "successful_urls": 0,
                    "failed_urls": 1,
                    "total_xml_files": 0,
                    "valid_xml_files": 0,
                    "invalid_xml_files": 0
                })
        elif mode == "batch":
            logger.info("Workflow Type: Batch Processing")
            xml_result = process_multiple_pages(urls, pricing_info, num_threads)
            if xml_result and temp_folder.exists():
                valid_count, total_count = count_valid_xml_files(temp_folder)
                print(f"Successfully generated {total_count} XML files ({valid_count} are valid XML, {total_count - valid_count} are non valid XML).")
                result.update({
                    "result": xml_result,
                    "scraped_urls": len(urls),
                    "successful_urls": total_count,
                    "failed_urls": len(urls) - total_count,
                    "total_xml_files": total_count,
                    "valid_xml_files": valid_count,
                    "invalid_xml_files": total_count - valid_count
                })
                result["result"] = cast(Optional[str], result["result"])
            elif xml_result:
                logger.warning("Temporary folder not found. Unable to count valid XML files.")
                result["result"] = cast(Optional[str], xml_result)
            else:
                result.update({
                    "scraped_urls": len(urls),
                    "successful_urls": 0,
                    "failed_urls": len(urls),
                    "total_xml_files": 0,
                    "valid_xml_files": 0,
                    "invalid_xml_files": 0
                })
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
        with open(json_file_path, 'r') as f:
            progress_data = json.load(f)
        urls = [url_data["url"] for url_data in progress_data.get("urls", [])]
        if not urls:
            print(f"Error: No valid URLs found in the resume file {json_file_path}")
            sys.exit(1)
        
        # Update global variables with saved data
        global temp_folder, total_cost, progress_tracker
        temp_folder = Path(progress_data.get("output_folder", temp_folder))
        total_cost = sum(float(url_data.get("costs", 0.0)) for url_data in progress_data["urls"])
        progress_tracker = {
            url_data["url"]: {
                "status": url_data["status"],
                "cost": float(url_data["costs"]),
                "valid_xml": url_data.get("valid_xml", False)
            } for url_data in progress_data["urls"]
        }
        
    except json.JSONDecodeError:
        print(f"Error: The resume file {json_file_path} is not a valid JSON file.")
        sys.exit(1)
    except KeyError:
        print(f"Error: The resume file {json_file_path} does not have the expected structure.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: An unexpected error occurred while reading the resume file: {str(e)}")
        sys.exit(1)

    result = main_workflow(
        urls=urls,
        mode="batch",
        whitelist=None,
        blacklist=None,
        num_threads=5,
        resume_file=json_file_path
    )

    display_scraping_summary(result, urls, temp_folder, error_log_file)


def start_single_scrape(url: str) -> None:
    result = main_workflow(
        urls=[url],
        mode="single",
        num_threads=1
    )
    display_scraping_summary(result, [url], temp_folder, error_log_file)

def create_summary_box(summary_data: Dict[str, str]) -> str:
    terminal_width = shutil.get_terminal_size().columns
    max_label_length = max(len(label) for label in summary_data.keys())
    max_value_length = max(len(str(value)) for value in summary_data.values())
    
    box_width = min(terminal_width - 3, max(max_label_length + max_value_length + 5, 50))
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
        create_line("├", "┼", "┤")
    ]

    for label, value in summary_data.items():
        box.extend(format_item(label, str(value)).split('\n'))

    box.append(create_line("└", "┴", "┘"))
    return '\n'.join(box)

def start_batch_scrape(url: str, whitelist: str, blacklist: str) -> None:
    result = main_workflow(
        urls=[url],
        mode="batch",
        whitelist=whitelist,
        blacklist=blacklist,
        num_threads=5
    )
    display_scraping_summary(result, [url], temp_folder, error_log_file)


# ============================
# Command-Line Interface
# ============================

def main():
    print("APIAS - API AUTO SCRAPER")
    print(f"Version {VERSION}\n")
    parser = argparse.ArgumentParser(description="Web API Retrieval and XML Extraction")
    parser.add_argument("-r", "--resume", type=str, default=None, help="Path to the resume file (JSON)", required=False)
    parser.add_argument("-u", "--url", type=str, default=None, help="Base url to scrape", required=False)
    parser.add_argument("-w", "--whitelist", type=str, default=None, help="Path to the txt file with urls patterns to whitelist (one for each line)", required=False)
    parser.add_argument("-b", "--blacklist", type=str, default=None, help="Path to the txt file with urls patterns to blacklist (one for each line)", required=False)
    parser.add_argument("-m", "--mode", type=str, default='single', help="Scraping mode: single or batch (if -m=batch the script will parse the sitemap.xml file of the domain from the base url)", required=False)
    

    args = parser.parse_args()

    if args.url is None and args.resume is None:
        print("Error: you need to specify an url or a json file.")
        sys.exit()

    if args.mode == 'single' and (args.whitelist or args.blacklist):
        print("Error: when using the single mode (default) you cannot use whitelist or blacklist.")
    if args.resume:
        # When using --resume, ignore other parameters
        if any([args.url, args.blacklist, args.whitelist, args.mode]):
            print("Warning: When using --resume, other parameters will be ignored.")
        start_resume_mode(args.resume)
    elif args.url and args.mode == 'single':
        start_single_scrape(args.url)
    elif args.url and args.mode == 'batch':
        start_batch_scrape(args.url, args.whitelist, args.blacklist)
    else:
        print("Error: Invalid combination of arguments. Please provide either --resume or --url argument.")
        print()
        print("EXAMPLE USAGE for a single url:")
        print('    python apias.py --url "https://example.com" --mode single')
        print()
        print("EXAMPLE USAGE for multiple urls from the same domain:")
        print('    python apias.py --url "https://example.com" --mode batch --whitelist "whitelist.txt" --blacklist "blacklist.txt"')
        print()
        print("EXAMPLE USAGE for resuming a batch job terminated prematurely:")
        print('    python apias.py --resume "./temp_output_folder/progress.json"')
        print()
        sys.exit(1)

    print("Job Finished.\n")

if __name__ == "__main__":
    main()







