# APIAS Python API Reference

This document provides comprehensive documentation for using APIAS as a Python module in your own applications.

## Installation

```bash
pip install apias
# or
uv add apias
```

## Quick Start

```python
from apias.config import APIASConfig
from apias.apias import Scraper, clean_html, validate_xml

# Create a scraper and fetch a page
scraper = Scraper(quiet=True)
content, mime_type = scraper.scrape("https://api.example.com/docs")

if content:
    # Clean the HTML
    cleaned = clean_html(content)
    print(f"Cleaned HTML: {len(cleaned)} characters")
```

---

## Package-Level Imports

APIAS exports key functions at the package level for convenience:

```python
import apias

# Access version
print(apias.__version__)  # e.g., "0.1.26"

# Parse documentation content
doc = apias.parse_documentation("GET /api/users - List users")

# Validate configuration dictionary
is_valid = apias.validate_config({
    "base_url": "https://example.com",
    "output_format": "xml"
})
```

### Exported Names

| Name | Type | Description |
|------|------|-------------|
| `__version__` | `str` | Package version string |
| `parse_documentation` | `function` | Parse raw documentation into APIDocument |
| `validate_config` | `function` | Validate configuration dictionary |

---

## Module: `apias.config`

### Constants

```python
from apias.config import (
    DEFAULT_MODEL,           # Default OpenAI model: "gpt-5-nano"
    SUPPORTED_MODELS,        # Dict of supported models with context windows
    MODEL_PRICING,           # Dict of model pricing per 1M tokens
    COST_RATIO_CONSERVATIVE, # P50 output/input ratio: 0.58
    COST_RATIO_AVERAGE,      # Mean output/input ratio: 1.84
    COST_RATIO_WORST_CASE,   # P95 output/input ratio: 11.82
    CHARS_PER_TOKEN,         # Characters per token estimate: 4
)
```

**Example: List supported models**
```python
from apias.config import SUPPORTED_MODELS, DEFAULT_MODEL

print(f"Default model: {DEFAULT_MODEL}")
print("\nSupported models:")
for model, info in SUPPORTED_MODELS.items():
    print(f"  {model}: {info['context_window']:,} tokens - {info['description']}")
```

---

### `class APIASConfig`

Main configuration class for APIAS. Supports loading from YAML/JSON files or direct instantiation.

#### Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | `"gpt-5-nano"` | OpenAI model to use |
| `api_key` | `str \| None` | `None` | API key (uses `OPENAI_API_KEY` env var if None) |
| `max_tokens` | `int` | `4096` | Maximum tokens in response |
| `temperature` | `float` | `0.1` | Model temperature (0-2) |
| `num_threads` | `int` | `5` | Number of concurrent workers |
| `chunk_size` | `int` | `50000` | Max characters per HTML chunk |
| `max_retries` | `int` | `3` | Retry attempts on failure |
| `retry_delay` | `float` | `1.0` | Base delay between retries (seconds) |
| `output_folder` | `str \| None` | `None` | Output directory (auto-generated if None) |
| `output_format` | `str` | `"xml"` | Output format |
| `whitelist_patterns` | `List[str]` | `[]` | URL patterns to include |
| `blacklist_patterns` | `List[str]` | `[]` | URL patterns to exclude |
| `limit` | `int \| None` | `None` | Max URLs to process |
| `no_tui` | `bool` | `False` | Disable Rich TUI |
| `quiet` | `bool` | `False` | Minimal output (implies `no_tui`) |
| `auto_resume` | `bool` | `False` | Auto-resume last session |
| `progress_file` | `str` | `"progress.json"` | Progress tracking file name |
| `atomic_saves` | `bool` | `True` | Use atomic file writes for progress |

#### Methods

##### `__init__(**kwargs)`

Create a new configuration with custom values.

```python
from apias.config import APIASConfig

# Basic configuration
config = APIASConfig()

# Custom configuration
config = APIASConfig(
    model="gpt-5-mini",
    num_threads=10,
    max_retries=5,
    quiet=True
)
```

##### `get_api_key() -> str`

Get the OpenAI API key from config or environment variable.

```python
from apias.config import APIASConfig

config = APIASConfig()
try:
    api_key = config.get_api_key()
    print(f"API key found: {api_key[:8]}...")
except ValueError as e:
    print(f"Error: {e}")
```

##### `to_dict() -> Dict[str, Any]`

Convert configuration to a dictionary.

```python
from apias.config import APIASConfig

config = APIASConfig(model="gpt-5-mini", num_threads=8)
config_dict = config.to_dict()
print(config_dict)
# {'model': 'gpt-5-mini', 'num_threads': 8, ...}
```

##### `from_dict(data: Dict[str, Any]) -> APIASConfig` (classmethod)

Create configuration from a dictionary.

```python
from apias.config import APIASConfig

data = {
    "model": "gpt-5-nano",
    "num_threads": 5,
    "quiet": True
}
config = APIASConfig.from_dict(data)
print(config.model)  # gpt-5-nano
```

##### `from_yaml(path: str | Path) -> APIASConfig` (classmethod)

Load configuration from a YAML file.

```python
from apias.config import APIASConfig

# config.yaml:
# model: gpt-5-mini
# num_threads: 8
# quiet: true

config = APIASConfig.from_yaml("config.yaml")
print(config.model)  # gpt-5-mini
```

##### `from_json(path: str | Path) -> APIASConfig` (classmethod)

Load configuration from a JSON file.

```python
from apias.config import APIASConfig

# config.json:
# {"model": "gpt-5-mini", "num_threads": 8}

config = APIASConfig.from_json("config.json")
print(config.num_threads)  # 8
```

##### `save_yaml(path: str | Path) -> None`

Save configuration to a YAML file.

```python
from apias.config import APIASConfig

config = APIASConfig(model="gpt-5-nano", num_threads=5)
config.save_yaml("my_config.yaml")
```

##### `save_json(path: str | Path) -> None`

Save configuration to a JSON file.

```python
from apias.config import APIASConfig

config = APIASConfig(model="gpt-5-nano", num_threads=5)
config.save_json("my_config.json")
```

---

### `load_config(config_path, cli_overrides) -> APIASConfig`

Load configuration with precedence: CLI args > config file > defaults.

```python
from apias.config import load_config

# Load defaults only
config = load_config()

# Load from file
config = load_config(config_path="apias_config.yaml")

# Load from file with CLI overrides
config = load_config(
    config_path="apias_config.yaml",
    cli_overrides={"model": "gpt-5-mini", "quiet": True}
)
```

---

### `generate_example_config(path) -> None`

Generate an example YAML configuration file with explanatory comments.

```python
from apias.config import generate_example_config

# Generate with default name
generate_example_config()  # Creates "apias_config.yaml"

# Generate with custom path
generate_example_config("my_config.yaml")
```

The generated file includes all available options with documentation:

```yaml
# APIAS Configuration File
# ========================
# model: The OpenAI model to use
model: gpt-5-nano

# num_threads: Number of parallel threads
num_threads: 5

# ... (all other options documented)
```

---

### `validate_url(url: str) -> bool`

Validate that a URL is well-formed with HTTP/HTTPS scheme.

```python
from apias.config import validate_url

print(validate_url("https://example.com"))      # True
print(validate_url("http://localhost:3000"))    # True
print(validate_url("example.com"))              # False (no scheme)
print(validate_url("ftp://example.com"))        # False (wrong scheme)
```

---

### `validate_urls(urls: List[str]) -> List[str]`

Filter a list of URLs, returning only valid ones.

```python
from apias.config import validate_urls

urls = [
    "https://example.com",
    "not-a-url",
    "http://api.example.com/docs",
    "ftp://invalid.com"
]
valid = validate_urls(urls)
print(valid)  # ['https://example.com', 'http://api.example.com/docs']
```

---

### `estimate_tokens(text: str) -> int`

Estimate token count from text (approximately 4 characters per token).

```python
from apias.config import estimate_tokens

text = "Hello, this is a sample text for token estimation."
tokens = estimate_tokens(text)
print(f"Estimated tokens: {tokens}")  # ~12-13 tokens
```

---

### `estimate_cost(input_tokens, model, ratio) -> Tuple[float, float, float]`

Calculate estimated API costs for given input tokens and output ratio.

**Parameters:**
- `input_tokens` (int): Number of input tokens
- `model` (str): Model name (default: `DEFAULT_MODEL`)
- `ratio` (float): Output/input ratio (default: `COST_RATIO_CONSERVATIVE`)

**Returns:** Tuple of (input_cost, output_cost, total_cost) in USD

```python
from apias.config import estimate_cost, COST_RATIO_AVERAGE

input_tokens = 100_000
input_cost, output_cost, total_cost = estimate_cost(
    input_tokens,
    model="gpt-5-nano",
    ratio=COST_RATIO_AVERAGE  # 1.84x output ratio
)
print(f"Input: ${input_cost:.4f}, Output: ${output_cost:.4f}, Total: ${total_cost:.4f}")
```

---

### `get_cost_estimates(input_tokens, model) -> Dict[str, Dict[str, float]]`

Get cost estimates for all three scenarios (conservative, average, worst case).

```python
from apias.config import get_cost_estimates, estimate_tokens

# Estimate costs for 1MB of text
text = "a" * 1_000_000
input_tokens = estimate_tokens(text)

estimates = get_cost_estimates(input_tokens, model="gpt-5-nano")

for scenario, data in estimates.items():
    print(f"{scenario.upper()}:")
    print(f"  Output tokens: {data['output_tokens']:,}")
    print(f"  Input cost:  ${data['input_cost']:.4f}")
    print(f"  Output cost: ${data['output_cost']:.4f}")
    print(f"  Total cost:  ${data['total_cost']:.4f}")
    print()
```

---

## Module: `apias.apias`

### `class Scraper`

Web scraper using Playwright for JavaScript-rendered pages.

#### Constructor

```python
Scraper(
    playwright_available: bool | None = None,
    verify_ssl: bool = True,
    timeout: int = 30,
    quiet: bool = False
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `playwright_available` | `bool \| None` | `None` | Override Playwright availability check |
| `verify_ssl` | `bool` | `True` | Verify SSL certificates |
| `timeout` | `int` | `30` | Scraping timeout in seconds |
| `quiet` | `bool` | `False` | Suppress spinner output |

#### Methods

##### `scrape(url: str) -> Tuple[str | None, str | None]`

Scrape a URL and return the content and MIME type.

```python
from apias.apias import Scraper

scraper = Scraper(quiet=True)
content, mime_type = scraper.scrape("https://docs.python.org/3/")

if content:
    print(f"Content length: {len(content)} characters")
    print(f"MIME type: {mime_type}")
else:
    print("Scraping failed")
```

##### `looks_like_html(content: str) -> bool`

Check if content appears to be HTML.

```python
from apias.apias import Scraper

scraper = Scraper()
print(scraper.looks_like_html("<html><body>Hello</body></html>"))  # True
print(scraper.looks_like_html("Just plain text"))  # False
```

---

### `start_scraping(url, no_tui, quiet) -> str | None`

High-level function to scrape a URL with automatic Playwright setup.

**Parameters:**
- `url` (str): The URL to scrape
- `no_tui` (bool): If True, disable TUI output (default: False)
- `quiet` (bool): If True, suppress all output (default: False)

**Returns:** Scraped content prefixed with URL, or None on failure

```python
from apias.apias import start_scraping

# Basic usage
content = start_scraping("https://docs.python.org/3/", quiet=True)
if content:
    print(f"Scraped {len(content)} characters")

# For batch processing (suppress output)
content = start_scraping("https://example.com/api", no_tui=True, quiet=True)
```

---

### `class APIDocument`

Represents a parsed API document with extracted endpoints, methods, and descriptions.

#### Constructor

```python
APIDocument(content: str)
```

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `content` | `str` | Raw document content |
| `endpoints` | `List[str]` | Extracted API endpoints |
| `methods` | `List[str]` | Extracted HTTP methods |
| `descriptions` | `List[str]` | Extracted descriptions |

#### Methods

##### `to_markdown() -> str`

Convert document to Markdown format.

```python
from apias.apias import APIDocument

content = """
GET /api/users - Get all users
POST /api/users - Create a new user
DELETE /api/users/{id} - Delete a user
"""

doc = APIDocument(content)
print(doc.to_markdown())
```

##### `to_json() -> Dict[str, List[str]]`

Convert document to a dictionary.

```python
from apias.apias import APIDocument

doc = APIDocument("GET /api/users - List users")
data = doc.to_json()
print(data)
# {'endpoints': [...], 'methods': [...], 'descriptions': [...]}
```

##### `save(path: str | Path) -> None`

Save document to a file in Markdown format.

```python
from apias.apias import APIDocument

doc = APIDocument("GET /api/users - List users")
doc.save("api_docs.md")
```

---

### `parse_documentation(doc_content: str) -> APIDocument`

Parse raw API documentation content into a structured APIDocument.

```python
from apias.apias import parse_documentation

content = """
# User API

GET /api/users
Returns a list of all users.

POST /api/users
Creates a new user.
"""

doc = parse_documentation(content)
print(f"Found {len(doc.endpoints)} endpoints")
print(f"Found {len(doc.methods)} methods")
```

---

### `validate_config(config: Dict[str, Any]) -> bool`

Validate a configuration dictionary for required fields and correct formats.

**Required fields:**
- `base_url`: Must be a string starting with "http"
- `output_format`: Must be one of "markdown", "html", or "xml"

```python
from apias.apias import validate_config

# Valid configuration
config = {
    "base_url": "https://api.example.com",
    "output_format": "xml"
}
print(validate_config(config))  # True

# Invalid - missing required field
config = {"output_format": "xml"}
print(validate_config(config))  # False

# Invalid - wrong output format
config = {
    "base_url": "https://api.example.com",
    "output_format": "pdf"  # Not supported
}
print(validate_config(config))  # False
```

---

### `clean_html(html_content: str) -> str`

Clean HTML by removing navigation, scripts, styles, and comments.

```python
from apias.apias import clean_html

raw_html = """
<html>
<head><style>.nav { color: red; }</style></head>
<body>
<nav>Navigation here</nav>
<main>
<h1>API Documentation</h1>
<p>This is the main content.</p>
</main>
<script>alert('hello');</script>
</body>
</html>
"""

cleaned = clean_html(raw_html)
print(cleaned)
# Returns HTML with nav, script, style removed
```

---

### `slimdown_html(page_source: str) -> Tuple[...]`

Process HTML and extract structured data (code examples, methods, classes, images, links).

Returns a tuple of:
1. Cleaned HTML string
2. Page title (or "scraped_page" if none)
3. List of code examples (from `<pre>` tags)
4. List of method signatures
5. List of class definitions
6. List of image URLs
7. List of (href, text) link tuples

```python
from apias.apias import slimdown_html

html = """
<html>
<head><title>My API</title></head>
<body>
<pre>def hello(): pass</pre>
<a href="/docs">Documentation</a>
<img src="logo.png">
</body>
</html>
"""

result = slimdown_html(html)
cleaned_html, title, code_examples, methods, classes, images, links = result

print(f"Title: {title}")                    # "My API"
print(f"Code examples: {len(code_examples)}")  # 1
print(f"Links: {len(links)}")               # 1
print(f"Images: {len(images)}")             # 1
```

---

### `chunk_html_by_size(html_content, max_chars) -> List[str]`

Split large HTML content into smaller chunks for processing within token limits.

**Parameters:**
- `html_content` (str): The HTML content to split
- `max_chars` (int): Maximum characters per chunk (default: ~200K)

**Returns:** List of HTML chunk strings

```python
from apias.apias import chunk_html_by_size

# Large HTML content
large_html = "<html><body>" + "<p>content</p>" * 10000 + "</body></html>"

# Split into manageable chunks
chunks = chunk_html_by_size(large_html, max_chars=50000)
print(f"Split into {len(chunks)} chunks")

for i, chunk in enumerate(chunks):
    print(f"  Chunk {i+1}: {len(chunk)} characters")
```

---

### `escape_xml(xml_doc: str) -> str`

Escape XML special characters in a string.

**Escapes:** `"`, `'`, `<`, `>`, `&`

```python
from apias.apias import escape_xml

text = 'Hello <world> & "friends"'
escaped = escape_xml(text)
print(escaped)  # Hello &lt;world&gt; &amp; &quot;friends&quot;
```

---

### `validate_xml(xml_string: str) -> bool`

Validate that a string is well-formed XML.

```python
from apias.apias import validate_xml

# Valid XML
print(validate_xml("<root><item>test</item></root>"))  # True

# Invalid XML
print(validate_xml("<root><item>test</root>"))  # False (unclosed tag)
print(validate_xml("not xml at all"))  # False
```

---

### `merge_xmls(temp_folder: Path, progress_callback) -> str`

Merge multiple XML files from a folder into a single XML document.

**Parameters:**
- `temp_folder` (Path): Path to folder containing XML files
- `progress_callback` (Callable): Optional callback(current, total, message)

**Returns:** Merged XML string

```python
from pathlib import Path
from apias.apias import merge_xmls

def on_progress(current, total, message):
    print(f"[{current}/{total}] {message}")

temp_folder = Path("/tmp/apias_output")
merged_xml = merge_xmls(temp_folder, progress_callback=on_progress)
print(f"Merged XML: {len(merged_xml)} characters")
```

---

### `extract_urls_from_sitemap(...) -> List[str]`

Extract and filter URLs from a sitemap XML file or content.

**Parameters:**
- `sitemap_file` (str | None): Path to sitemap file
- `sitemap_content` (str | None): Sitemap XML content string
- `whitelist_str` (str | None): Comma-separated patterns to include
- `blacklist_str` (str | None): Comma-separated patterns to exclude

```python
from apias.apias import extract_urls_from_sitemap

# From file
urls = extract_urls_from_sitemap(sitemap_file="sitemap.xml")

# From string content
sitemap_content = """
<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://example.com/page1</loc></url>
  <url><loc>https://example.com/page2</loc></url>
  <url><loc>https://example.com/admin/secret</loc></url>
</urlset>
"""

# With filtering
urls = extract_urls_from_sitemap(
    sitemap_content=sitemap_content,
    whitelist_str="*example.com*",
    blacklist_str="*admin*,*secret*"
)
print(urls)  # ['https://example.com/page1', 'https://example.com/page2']
```

---

## Complete Example: Programmatic API Documentation Extraction

```python
from pathlib import Path
from apias.config import APIASConfig, estimate_tokens, get_cost_estimates
from apias.apias import Scraper, clean_html, slimdown_html, validate_xml

def extract_api_docs(url: str, output_path: str) -> str | None:
    """Extract API documentation from a URL.

    Args:
        url: The URL to scrape
        output_path: Where to save the processed HTML

    Returns:
        Processed HTML content, or None on failure
    """
    # 1. Configure
    config = APIASConfig(
        model="gpt-5-nano",
        num_threads=5,
        quiet=True
    )

    # 2. Scrape the page
    scraper = Scraper(quiet=True)
    content, mime_type = scraper.scrape(url)

    if not content:
        print(f"Failed to scrape {url}")
        return None

    print(f"Scraped {len(content)} characters ({mime_type})")

    # 3. Clean and process HTML
    cleaned = clean_html(content)
    result = slimdown_html(cleaned)
    final_html, title, code_examples, methods, classes, images, links = result

    print(f"Title: {title}")
    print(f"Found {len(code_examples)} code examples")
    print(f"Found {len(links)} links")

    # 4. Estimate costs before processing
    tokens = estimate_tokens(final_html)
    estimates = get_cost_estimates(tokens, config.model)

    print(f"\nCost estimates for {tokens:,} input tokens:")
    print(f"  Conservative: ${estimates['conservative']['total_cost']:.4f}")
    print(f"  Average:      ${estimates['average']['total_cost']:.4f}")
    print(f"  Worst case:   ${estimates['worst_case']['total_cost']:.4f}")

    # 5. Validate any XML output
    test_xml = f"<doc><title>{title}</title></doc>"
    if validate_xml(test_xml):
        print("\nXML validation: passed")

    # 6. Save processed HTML
    output = Path(output_path)
    output.write_text(final_html)
    print(f"\nSaved to {output_path}")

    return final_html


# Run the extraction
if __name__ == "__main__":
    extract_api_docs(
        "https://docs.python.org/3/library/asyncio.html",
        "asyncio_docs.html"
    )
```

---

## Batch Processing Example

```python
from pathlib import Path
from apias.config import APIASConfig, validate_urls
from apias.apias import start_scraping, clean_html, chunk_html_by_size

def batch_scrape(urls: list[str], output_dir: str) -> dict[str, str]:
    """Scrape multiple URLs and save results.

    Args:
        urls: List of URLs to scrape
        output_dir: Directory to save results

    Returns:
        Dict mapping URLs to output file paths
    """
    # Validate URLs first
    valid_urls = validate_urls(urls)
    print(f"Valid URLs: {len(valid_urls)}/{len(urls)}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {}
    for i, url in enumerate(valid_urls, 1):
        print(f"\n[{i}/{len(valid_urls)}] Scraping: {url}")

        # Use start_scraping for automatic setup
        content = start_scraping(url, quiet=True)
        if not content:
            print(f"  Failed to scrape")
            continue

        # Clean and chunk if necessary
        cleaned = clean_html(content)
        chunks = chunk_html_by_size(cleaned, max_chars=100000)

        # Save each chunk
        for j, chunk in enumerate(chunks):
            filename = f"page_{i}_chunk_{j}.html"
            filepath = output_path / filename
            filepath.write_text(chunk)
            print(f"  Saved: {filename} ({len(chunk)} chars)")

        results[url] = str(output_path / f"page_{i}_chunk_0.html")

    return results


if __name__ == "__main__":
    urls = [
        "https://docs.python.org/3/library/asyncio.html",
        "https://docs.python.org/3/library/typing.html",
        "https://docs.python.org/3/library/pathlib.html",
    ]
    batch_scrape(urls, "/tmp/batch_output")
```

---

## Error Handling

All functions raise appropriate exceptions:

```python
from apias.config import APIASConfig, validate_url

# Configuration validation
try:
    config = APIASConfig(num_threads=-1)  # Invalid
except ValueError as e:
    print(f"Config error: {e}")  # "num_threads must be at least 1"

try:
    config = APIASConfig(temperature=3.0)  # Invalid
except ValueError as e:
    print(f"Config error: {e}")  # "temperature must be between 0 and 2"

# API key validation
try:
    config = APIASConfig()
    key = config.get_api_key()  # Raises if no key found
except ValueError as e:
    print(f"API key error: {e}")

# File loading errors
try:
    config = APIASConfig.from_yaml("nonexistent.yaml")
except FileNotFoundError as e:
    print(f"File error: {e}")

# URL validation (returns bool, doesn't raise)
if not validate_url("invalid-url"):
    print("Invalid URL format")
```

---

## Type Hints

APIAS uses comprehensive type hints throughout. Example:

```python
from typing import Dict, List
from pathlib import Path

from apias.config import APIASConfig, load_config
from apias.apias import Scraper, APIDocument

def process_docs(urls: List[str]) -> Dict[str, APIDocument]:
    """Process multiple URLs and return documents."""
    config: APIASConfig = load_config()
    scraper: Scraper = Scraper(quiet=config.quiet)
    results: Dict[str, APIDocument] = {}

    for url in urls:
        content, _ = scraper.scrape(url)
        if content:
            results[url] = APIDocument(content)

    return results
```

---

## See Also

- [README.md](README.md) - General documentation and CLI usage
- [CHANGELOG.md](CHANGELOG.md) - Version history
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines
