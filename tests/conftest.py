import pytest


@pytest.fixture
def sample_api_doc():
    return """
    # Sample API Documentation
    ## Endpoint: /api/v1/users
    GET /api/v1/users
    Returns a list of users
    """


@pytest.fixture
def sample_config():
    return {"base_url": "http://example.com", "output_format": "markdown"}
