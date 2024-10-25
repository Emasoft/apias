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

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

2. Install development dependencies:
   ```bash
   pip install -e ".[dev,test]"
   ```

3. Install pre-commit hooks:
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
3. The PR should work for Python 3.9 and above
4. Make sure all tests pass
5. Update the CHANGELOG.md

## Code Style

We use:
- black for code formatting
- isort for import sorting
- mypy for type checking
- ruff for linting

## Questions?

Feel free to open an issue for any questions you might have.

Thank you for your contribution! 🎉
