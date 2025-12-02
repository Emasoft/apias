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
model: gpt-4o-mini   # Default: the smart but affordable one
```

Different AI models have different abilities and costs:

| Model | Speed | Quality | Cost | Best For |
|-------|-------|---------|------|----------|
| `gpt-4o-mini` | Fast | Good | Low | Most scraping tasks (recommended) |
| `gpt-4o` | Fast | Excellent | Medium | Complex documentation |
| `gpt-4-turbo` | Medium | Excellent | High | When quality matters most |
| `gpt-3.5-turbo` | Very Fast | Okay | Very Low | Simple pages, budget mode |

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

### Quick Reference: Common Configurations

#### For Small Websites (< 50 pages)

```yaml
num_threads: 3
max_retries: 3
chunk_size: 50000
model: gpt-4o-mini
limit: null
```

#### For Large Websites (100+ pages)

```yaml
num_threads: 8
max_retries: 5
chunk_size: 40000
model: gpt-4o-mini
limit: 500
```

#### For Slow/Unstable Connections

```yaml
num_threads: 2
max_retries: 5
retry_delay: 2.0
chunk_size: 30000
model: gpt-4o-mini
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
