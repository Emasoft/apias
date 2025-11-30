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

## Requirements

- **Python 3.10 or higher** (Python 3.9 is not supported)
- OpenAI API key (for AI-powered extraction)

## Installation

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
# Scrape documentation from a URL
apias scrape https://api.example.com/docs

# Convert to specific format
apias convert input.html --format markdown --output api_docs.md
```

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
