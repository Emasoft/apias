import pytest
from pathlib import Path
from apias.apias import APIDocument, parse_documentation, validate_config

def test_api_document_creation():
    """Test APIDocument initialization and parsing"""
    doc = APIDocument("""
    # API Documentation
    ## GET /api/v1/users
    Returns a list of users
    """)
    assert len(doc.endpoints) > 0
    assert "/api/v1/users" in doc.endpoints[0]
    assert len(doc.methods) > 0
    assert "GET" in doc.methods[0]

def test_api_document_to_markdown():
    """Test conversion to markdown"""
    doc = APIDocument("# Test\nGET /api/test\nDescription")
    markdown = doc.to_markdown()
    assert isinstance(markdown, str)
    assert "Description" in markdown

def test_api_document_to_json():
    """Test conversion to JSON/dict"""
    doc = APIDocument("# Test\nGET /api/test\nDescription")
    json_data = doc.to_json()
    assert isinstance(json_data, dict)
    assert "endpoints" in json_data
    assert "methods" in json_data
    assert "descriptions" in json_data

def test_api_document_save(tmp_path):
    """Test saving document to file"""
    doc = APIDocument("# Test\nGET /api/test\nDescription")
    output_file = tmp_path / "test_doc.md"
    doc.save(output_file)
    assert output_file.exists()
    content = output_file.read_text()
    assert "Description" in content

def test_parse_documentation_invalid_input():
    """Test parse_documentation with invalid input"""
    with pytest.raises(TypeError):
        parse_documentation(None)  # type: ignore

def test_validate_config_invalid_cases():
    """Test validate_config with various invalid configurations"""
    # Missing required field
    assert not validate_config({})
    assert not validate_config({"base_url": "http://example.com"})
    
    # Invalid base_url
    assert not validate_config({
        "base_url": "not-a-url",
        "output_format": "markdown"
    })
    
    # Invalid output format
    assert not validate_config({
        "base_url": "http://example.com",
        "output_format": "invalid"
    })

def test_validate_config_valid_formats():
    """Test validate_config with different valid output formats"""
    base_config = {"base_url": "http://example.com"}
    
    for format in ["markdown", "html", "xml"]:
        config = base_config.copy()
        config["output_format"] = format
        assert validate_config(config)

def test_api_document_str_representation():
    """Test string representation of APIDocument"""
    doc = APIDocument("""
    # API Documentation
    ## GET /api/v1/users
    Returns a list of users
    """)
    str_rep = str(doc)
    assert "users" in str_rep
    assert "/api/v1/users" in str_rep
    assert "GET" in str_rep
