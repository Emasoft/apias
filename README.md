# APIAS - AI Powered API Documentation Scraper

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/apias.svg)](https://badge.fury.io/py/apias)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

APIAS (AI Powered API Documentation Scraper) is a powerful tool that helps you extract and convert API documentation from various sources into structured formats.

## Features

- Scrape API documentation from web pages
- Support for multiple documentation formats
- AI-powered content extraction and structuring
- Command-line interface for easy use
- Multiple output formats (Markdown, JSON, YAML)
- Batch processing mode with interactive TUI

![APIAS Batch Processing TUI](assets/apias_tui_screenshot.gif)

## Requirements

- **Python 3.10 or higher** (Python 3.9 is not supported)
- OpenAI API key (for AI-powered extraction)

## Installation

### Using uv (Recommended)

The fastest way to install APIAS is using [uv](https://docs.astral.sh/uv/):

```bash
# Install as a tool (recommended for CLI usage)
uv tool install apias --python=3.10

# Or install in a project
uv add apias
```

### Using pip

```bash
pip install apias
```

### Verify Python Version

```bash
python --version  # Should be 3.10 or higher
```

## Quick Start

```python
from apias import apias

# Basic usage
doc = apias.scrape_url("https://api.example.com/docs")
print(doc.to_markdown())

# With custom configuration
config = {
    "format": "markdown",
    "output": "api_docs.md"
}
apias.scrape_and_save("https://api.example.com/docs", config)
```

## Command Line Usage

```bash
# Scrape a single page
apias --url https://api.example.com/docs

# Scrape multiple pages from a website (batch mode)
apias --url https://example.com --mode batch

# Limit how many pages to scrape
apias --url https://example.com --mode batch --limit 50

# Estimate costs before processing (no API calls made)
apias --url https://example.com --mode batch --estimate-cost

# Use a configuration file
apias --url https://example.com --config apias_config.yaml
```

---

## Configuration Guide

> **Think of APIAS like a team of workers in a factory!**

APIAS can be configured using a YAML file. Generate an example with:

```bash
apias --generate-config
```

This creates `apias_config.yaml` that you can edit.

### Understanding the Settings (Explained Simply)

#### `num_threads` - How Many Workers?

```yaml
num_threads: 5   # Default: 5 workers
```

Imagine you have a big pile of web pages to process. `num_threads` is like choosing how many workers to hire:

```
                    +---> Worker 1 ---> processes page A
                    |
Your Pages -------->+---> Worker 2 ---> processes page B
(waiting)           |
                    +---> Worker 3 ---> processes page C
                    |
                    +---> Worker 4 ---> processes page D
                    |
                    +---> Worker 5 ---> processes page E
```

- **num_threads: 1** = One worker, processes pages one by one (slow but gentle on the website)
- **num_threads: 5** = Five workers processing 5 pages at the same time (faster!)
- **num_threads: 10** = Ten workers (even faster, but uses more computer power)

> **Warning**: Don't use more than 10-15 threads! Too many workers might:
> - Overwhelm the website you're scraping (they might block you!)
> - Hit OpenAI rate limits (the AI can only handle so many requests)
> - Use too much memory on your computer

**Recommendation**: Start with 5. Increase to 10 if everything works smoothly.

---

#### `max_retries` - How Many Times to Try Again?

```yaml
max_retries: 3   # Default: 3 attempts
```

Sometimes things fail (network hiccups, server busy, etc.). `max_retries` is how many times APIAS will try again before giving up:

```
Attempt 1: "Hey server, give me this page!"
           Server: "Sorry, I'm busy!" (FAIL)

Attempt 2: *waits 1 second* "Okay, how about now?"
           Server: "Still busy!" (FAIL)

Attempt 3: *waits 2 seconds* "Please?"
           Server: "Here you go!" (SUCCESS!)
```

- **max_retries: 0** = Never retry (give up immediately on any error)
- **max_retries: 3** = Try up to 3 times before giving up
- **max_retries: 5** = Very persistent, keeps trying longer

---

#### `chunk_size` - How Big Are the Pieces?

```yaml
chunk_size: 50000   # Default: 50,000 characters
```

Web pages can be HUGE. We can't send a giant page to the AI all at once (it would choke!). So we cut it into smaller pieces called "chunks":

```
   Giant Web Page (200,000 characters)
   ====================================

   Gets cut into pieces:

   [  Chunk 1  ]  [  Chunk 2  ]  [  Chunk 3  ]  [  Chunk 4  ]
    (50,000)       (50,000)       (50,000)       (50,000)
       |              |              |              |
       v              v              v              v
      AI            AI             AI             AI
       |              |              |              |
       v              v              v              v
   [Result 1]    [Result 2]     [Result 3]    [Result 4]

   Then all results get merged back together!
```

- **chunk_size: 30000** = Smaller pieces (more API calls, but safer for complex pages)
- **chunk_size: 50000** = Default balance
- **chunk_size: 100000** = Bigger pieces (fewer API calls, but might hit token limits)

---

#### `model` - Which AI Brain to Use?

```yaml
model: gpt-5-nano   # Default: fast, affordable, and highly capable
```

OpenAI GPT-5 models offer excellent quality at different price points. Prices shown below are approximate and may change - check [OpenAI Pricing](https://openai.com/api/pricing/) for current rates:

| Model | Context | Input | Output | Best For |
|-------|---------|-------|--------|----------|
| `gpt-5-nano` | 272K | Very Low | Very Low | Most scraping tasks (recommended default) |
| `gpt-5-mini` | 272K | Low | Low | Complex documentation |
| `gpt-5` | 272K | Medium | Medium | Premium quality extraction |
| `gpt-5.1` | 272K | Medium | Medium | Agentic tasks, coding (newest) |
| `gpt-5-pro` | 400K | High | High | Extended context, highest quality |

> **Note**: All GPT-5 models support up to 128K output tokens. The `gpt-5-nano` model offers the best cost-performance ratio for API documentation scraping.

---

#### `limit` - Maximum Pages to Scrape

```yaml
limit: 50   # Only scrape up to 50 pages (null = no limit)
```

In batch mode, a website might have thousands of pages. Use `limit` to control how many:

```bash
# Command line:
apias --url https://example.com --mode batch --limit 100

# Or in config file:
limit: 100
```

---

#### `--estimate-cost` - Preview API Costs Before Processing

Before committing to a full extraction, you can estimate costs without making any OpenAI API calls:

```bash
apias --url https://example.com --mode batch --estimate-cost
```

This will:
1. Scrape all pages (respecting `--limit` if set)
2. Calculate total input tokens from page content
3. Display three cost scenarios based on real-world usage data:

```
┌─────────────────────────────────────────────────────────────┐
│                    Cost Estimation                          │
├─────────────────────────────────────────────────────────────┤
│ Input Tokens: 1,234,567                                     │
├─────────────────────────────────────────────────────────────┤
│ Scenario        │ Output Tokens │ Input Cost │ Total Cost   │
├─────────────────┼───────────────┼────────────┼──────────────┤
│ Conservative    │     716,249   │    $0.06   │    $0.35     │
│ Average         │   2,271,603   │    $0.06   │    $0.97     │
│ Worst Case      │  14,592,582   │    $0.06   │    $5.90     │
└─────────────────────────────────────────────────────────────┘
```

**Cost Scenarios Explained:**

| Scenario | Output Ratio | Description |
|----------|-------------|-------------|
| **Conservative** | 0.58x input | P50 median - half of jobs cost this or less |
| **Average** | 1.84x input | Mean across all extractions |
| **Worst Case** | 11.82x input | P95 - only 5% of jobs exceed this |

> **Tip**: The Conservative estimate is typically accurate for well-structured API documentation. Use the Worst Case estimate for budget planning with complex or messy HTML.

---

### Quick Reference: Common Configurations

#### For Small Websites (< 50 pages)

```yaml
num_threads: 3
max_retries: 3
chunk_size: 50000
model: gpt-5-nano
limit: null
```

#### For Large Websites (100+ pages)

```yaml
num_threads: 8
max_retries: 5
chunk_size: 40000
model: gpt-5-nano
limit: 500
```

#### For Slow/Unstable Connections

```yaml
num_threads: 2
max_retries: 5
retry_delay: 2.0
chunk_size: 30000
model: gpt-5-nano
```

#### For CI/CD (Headless, No User Interaction)

```yaml
num_threads: 5
no_tui: true
quiet: true
auto_resume: true
```

---

### Environment Variables

You can also use environment variables:

```bash
# Required: Your OpenAI API key
export OPENAI_API_KEY="sk-your-key-here"

# Then run APIAS
apias --url https://example.com
```

---

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Security

For security issues, please see our [Security Policy](SECURITY.md).

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for a list of changes.

## Support

- Documentation: [https://github.com/Emasoft/apias/docs](https://github.com/Emasoft/apias/docs)
- Issues: [https://github.com/Emasoft/apias/issues](https://github.com/Emasoft/apias/issues)
