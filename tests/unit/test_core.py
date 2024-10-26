import pytest
from apias import apias


def test_basic_scraping(sample_api_doc):
    """Test basic API documentation scraping"""
    result = apias.parse_documentation(sample_api_doc)
    assert result is not None
    assert "users" in str(result).lower()


def test_config_validation(sample_config):
    """Test configuration validation"""
    assert apias.validate_config(sample_config)
