# Contributing to APIAS

First off, thank you for considering contributing to APIAS! Its people like you that make APIAS such a great tool.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
   ```bash
   git clone git@github.com:your-username/apias.git
   ```
3. Create a branch for your changes
   ```bash
   git checkout -b feature/amazing-feature
   ```

## Development Setup

**Requirements:** Python 3.10 or higher

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

2. Verify Python version:
   ```bash
   python --version  # Must be 3.10 or higher
   ```

3. Install development dependencies:
   ```bash
   pip install -e ".[dev,test]"
   ```

4. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Making Changes

1. Make your changes
2. Write or adapt tests as needed
3. Run the test suite
   ```bash
   pytest
   ```
4. Update documentation if needed
5. Commit your changes:
   ```bash
   git add .
   git commit -m "Add some amazing feature"
   ```

## Pull Request Process

1. Update the README.md with details of changes if needed
2. Update the documentation
3. The PR should work for Python 3.10 and above
4. Make sure all tests pass
5. Update the CHANGELOG.md

## Code Style

We use:
- ruff for code formatting and linting
- isort for import sorting
- mypy for type checking (non-fatal, known issues exist)

## Questions?

Feel free to open an issue for any questions you might have.

Thank you for your contribution! 🎉
