"""
Tests for the APIAS configuration module.

Tests configuration loading, validation, and serialization.
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

from apias.config import (
    DEFAULT_MODEL,
    SUPPORTED_MODELS,
    APIASConfig,
    generate_example_config,
    load_config,
    validate_url,
    validate_urls,
)


class TestAPIASConfig:
    """Tests for APIASConfig dataclass."""

    def test_default_values(self) -> None:
        """Config uses correct default values when created without arguments."""
        config = APIASConfig()
        assert config.model == DEFAULT_MODEL
        assert config.max_tokens == 4096
        assert config.temperature == 0.1
        assert config.num_threads == 5
        assert config.chunk_size == 50000
        assert config.no_tui is False
        assert config.quiet is False
        assert config.auto_resume is False

    def test_custom_values(self) -> None:
        """Config accepts and stores custom values correctly."""
        config = APIASConfig(
            model="gpt-5-nano",
            max_tokens=8192,
            temperature=0.5,
            num_threads=10,
            quiet=True,
        )
        assert config.model == "gpt-5-nano"
        assert config.max_tokens == 8192
        assert config.temperature == 0.5
        assert config.num_threads == 10
        assert config.quiet is True
        assert config.no_tui is True  # quiet implies no_tui

    def test_quiet_implies_no_tui(self) -> None:
        """Setting quiet=True automatically sets no_tui=True."""
        config = APIASConfig(quiet=True, no_tui=False)
        assert config.quiet is True
        assert config.no_tui is True

    def test_invalid_num_threads_zero(self) -> None:
        """Config rejects num_threads less than 1."""
        with pytest.raises(ValueError, match="num_threads must be at least 1"):
            APIASConfig(num_threads=0)

    def test_invalid_num_threads_negative(self) -> None:
        """Config rejects negative num_threads."""
        with pytest.raises(ValueError, match="num_threads must be at least 1"):
            APIASConfig(num_threads=-1)

    def test_invalid_chunk_size(self) -> None:
        """Config rejects chunk_size below 1000."""
        with pytest.raises(ValueError, match="chunk_size must be at least 1000"):
            APIASConfig(chunk_size=500)

    def test_invalid_max_retries(self) -> None:
        """Config rejects negative max_retries."""
        with pytest.raises(ValueError, match="max_retries must be non-negative"):
            APIASConfig(max_retries=-1)

    def test_invalid_temperature_low(self) -> None:
        """Config rejects temperature below 0."""
        with pytest.raises(ValueError, match="temperature must be between 0 and 2"):
            APIASConfig(temperature=-0.1)

    def test_invalid_temperature_high(self) -> None:
        """Config rejects temperature above 2."""
        with pytest.raises(ValueError, match="temperature must be between 0 and 2"):
            APIASConfig(temperature=2.5)

    def test_to_dict(self) -> None:
        """Config correctly serializes to dictionary."""
        config = APIASConfig(model="gpt-5-nano", num_threads=3)
        data = config.to_dict()
        assert data["model"] == "gpt-5-nano"
        assert data["num_threads"] == 3
        assert "api_key" not in data  # api_key should not be in to_dict output

    def test_from_dict(self) -> None:
        """Config correctly deserializes from dictionary."""
        data = {"model": "gpt-5-mini", "num_threads": 8, "quiet": True}
        config = APIASConfig.from_dict(data)
        assert config.model == "gpt-5-mini"
        assert config.num_threads == 8
        assert config.quiet is True

    def test_from_dict_ignores_unknown_fields(self) -> None:
        """Config ignores unknown fields in dictionary."""
        data = {"model": "gpt-5-nano", "unknown_field": "should_be_ignored"}
        config = APIASConfig.from_dict(data)
        assert config.model == "gpt-5-nano"
        assert not hasattr(config, "unknown_field")


class TestConfigFileIO:
    """Tests for configuration file loading and saving."""

    def test_save_and_load_json(self, tmp_path: Path) -> None:
        """Config can be saved and loaded from JSON file."""
        config = APIASConfig(model="gpt-5-nano", num_threads=7)
        json_path = tmp_path / "config.json"

        config.save_json(json_path)
        loaded = APIASConfig.from_json(json_path)

        assert loaded.model == "gpt-5-nano"
        assert loaded.num_threads == 7

    def test_load_json_not_found(self, tmp_path: Path) -> None:
        """Loading non-existent JSON file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            APIASConfig.from_json(tmp_path / "nonexistent.json")

    def test_generate_example_config(self, tmp_path: Path) -> None:
        """Example config file is generated with correct content."""
        config_path = tmp_path / "example_config.yaml"
        generate_example_config(config_path)

        assert config_path.exists()
        content = config_path.read_text()
        assert "model:" in content
        assert "num_threads:" in content
        assert "quiet:" in content


class TestLoadConfig:
    """Tests for load_config function with precedence handling."""

    def test_load_defaults_no_file(self) -> None:
        """load_config returns defaults when no file is provided."""
        config = load_config()
        assert config.model == DEFAULT_MODEL
        assert config.num_threads == 5

    def test_load_with_cli_overrides(self) -> None:
        """CLI overrides take precedence over defaults."""
        config = load_config(cli_overrides={"model": "gpt-5-nano", "quiet": True})
        assert config.model == "gpt-5-nano"
        assert config.quiet is True

    def test_load_from_json_file(self, tmp_path: Path) -> None:
        """Config loads values from JSON file."""
        json_path = tmp_path / "config.json"
        json_path.write_text(json.dumps({"model": "gpt-5-mini", "num_threads": 3}))

        config = load_config(config_path=json_path)
        assert config.model == "gpt-5-mini"
        assert config.num_threads == 3

    def test_cli_overrides_file_values(self, tmp_path: Path) -> None:
        """CLI overrides take precedence over config file values."""
        json_path = tmp_path / "config.json"
        json_path.write_text(json.dumps({"model": "gpt-5-mini", "num_threads": 3}))

        config = load_config(
            config_path=json_path, cli_overrides={"model": "gpt-5-nano"}
        )
        assert config.model == "gpt-5-nano"  # CLI override
        assert config.num_threads == 3  # From file

    def test_unsupported_file_format(self, tmp_path: Path) -> None:
        """Loading unsupported file format raises ValueError."""
        txt_path = tmp_path / "config.txt"
        txt_path.write_text("model: gpt-5-nano")

        with pytest.raises(ValueError, match="Unsupported config file format"):
            load_config(config_path=txt_path)


class TestURLValidation:
    """Tests for URL validation functions."""

    def test_validate_url_valid_https(self) -> None:
        """Valid HTTPS URL passes validation."""
        assert validate_url("https://example.com") is True
        assert validate_url("https://api.example.com/docs") is True
        assert validate_url("https://example.com:8080/path") is True

    def test_validate_url_valid_http(self) -> None:
        """Valid HTTP URL passes validation."""
        assert validate_url("http://example.com") is True
        assert validate_url("http://localhost:3000") is True

    def test_validate_url_invalid_no_scheme(self) -> None:
        """URL without scheme fails validation."""
        assert validate_url("example.com") is False
        assert validate_url("www.example.com") is False

    def test_validate_url_invalid_scheme(self) -> None:
        """URL with invalid scheme fails validation."""
        assert validate_url("ftp://example.com") is False
        assert validate_url("file:///path/to/file") is False

    def test_validate_url_invalid_format(self) -> None:
        """Malformed URLs fail validation."""
        assert validate_url("not a url") is False
        assert validate_url("") is False
        assert validate_url("http://") is False

    def test_validate_urls_filters_invalid(self) -> None:
        """validate_urls returns only valid URLs from list."""
        urls = [
            "https://example.com",
            "not-a-url",
            "http://api.example.com",
            "ftp://invalid.com",
        ]
        valid = validate_urls(urls)
        assert valid == ["https://example.com", "http://api.example.com"]


class TestSupportedModels:
    """Tests for supported models configuration."""

    def test_default_model_is_supported(self) -> None:
        """Default model is in the supported models list."""
        assert DEFAULT_MODEL in SUPPORTED_MODELS

    def test_supported_models_have_context_window(self) -> None:
        """All supported models have context_window defined."""
        for model, info in SUPPORTED_MODELS.items():
            assert "context_window" in info, f"{model} missing context_window"
            assert isinstance(info["context_window"], int)

    def test_supported_models_have_description(self) -> None:
        """All supported models have description defined."""
        for model, info in SUPPORTED_MODELS.items():
            assert "description" in info, f"{model} missing description"
            assert isinstance(info["description"], str)
