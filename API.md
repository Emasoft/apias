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
from apias.config import APIASConfig, load_config
from apias.apias import Scraper, clean_html, validate_xml

# Create a scraper and fetch a page
scraper = Scraper(quiet=True)
content, mime_type = scraper.scrape("https://api.example.com/docs")

# Clean the HTML
cleaned = clean_html(content)
print(f"Cleaned HTML: {len(cleaned)} characters")
```

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
2. Page title
3. List of code examples
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

print(f"Title: {title}")
print(f"Code examples: {len(code_examples)}")
print(f"Links: {len(links)}")
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
import asyncio
from pathlib import Path
from apias.config import APIASConfig, estimate_tokens, get_cost_estimates
from apias.apias import Scraper, clean_html, slimdown_html, validate_xml

async def extract_api_docs(url: str, output_path: str):
    """Extract API documentation from a URL."""

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

    # 5. Save processed HTML
    output = Path(output_path)
    output.write_text(final_html)
    print(f"\nSaved to {output_path}")

    return final_html

# Run the extraction
if __name__ == "__main__":
    asyncio.run(extract_api_docs(
        "https://docs.python.org/3/library/asyncio.html",
        "asyncio_docs.html"
    ))
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

# API key validation
try:
    config = APIASConfig()
    key = config.get_api_key()  # Raises if no key found
except ValueError as e:
    print(f"API key error: {e}")

# URL validation (returns bool, doesn't raise)
if not validate_url("invalid-url"):
    print("Invalid URL format")
```

---

## Type Hints

APIAS uses comprehensive type hints throughout. Example:

```python
from typing import Dict, List, Tuple
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
