#!/usr/bin/env python3
"""
Release Pipeline Configuration Parser (v2.0)

Parses release-config.yaml for the release script.
Handles branch pipelines, hotfixes, build configs, and channel settings.

Usage:
    python release_config.py get <path>              # Get a single value
    python release_config.py get-channel <channel>   # Get channel config
    python release_config.py get-pipeline-stage <stage>  # Get pipeline stage
    python release_config.py resolve-branch <branch> # Find channel for branch
    python release_config.py validate                # Validate config
    python release_config.py export-env              # Export as shell vars
    python release_config.py build-command <channel> # Get build command
"""

import json
import re
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


def find_config_file() -> Path:
    """Find the release-config.yaml file."""
    current = Path.cwd()
    script_dir = Path(__file__).parent.parent

    candidates = [
        current / "release-config.yaml",
        current / "release-config.yml",
        script_dir / "release-config.yaml",
        script_dir / "release-config.yml",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    while current != current.parent:
        for name in ["release-config.yaml", "release-config.yml"]:
            candidate = current / name
            if candidate.exists():
                return candidate
        current = current.parent

    raise FileNotFoundError("Could not find release-config.yaml")


def load_config(config_path: Path | None = None) -> dict:
    """Load and parse the YAML configuration file."""
    if config_path is None:
        config_path = find_config_file()

    if not HAS_YAML:
        print("Error: PyYAML required. Install: uv add pyyaml", file=sys.stderr)
        sys.exit(1)

    with open(config_path) as f:
        return yaml.safe_load(f)


def get_nested_value(config: dict, path: str) -> Any:
    """Get nested value using dot notation."""
    keys = path.split(".")
    value = config

    for key in keys:
        if isinstance(value, dict):
            if key not in value:
                return None
            value = value[key]
        elif isinstance(value, list):
            try:
                value = value[int(key)]
            except (ValueError, IndexError):
                return None
        else:
            return None

    return value


def format_value(value: Any) -> str:
    """Format value for shell consumption."""
    if value is None:
        return ""
    elif isinstance(value, bool):
        return "true" if value else "false"
    elif isinstance(value, (list, dict)):
        return json.dumps(value)
    return str(value)


def get_channel_config(config: dict, channel: str) -> dict:
    """Get complete configuration for a channel with defaults."""
    channels = config.get("channels", {})
    channel_config = channels.get(channel, {})

    if not channel_config:
        return {}

    defaults = {
        "suffix": "",
        "tag_format": "v{version}",
        "prerelease": False,
        "build": {"wheel": True, "sdist": True, "dev": False},
        "publish": {"pypi": False, "github": True, "npm": False},
        "github_release": {"prerelease": False, "draft": False, "generate_notes": True},
    }

    result = {**defaults, **channel_config}

    for key in ["build", "publish", "github_release"]:
        if key in channel_config:
            result[key] = {**defaults.get(key, {}), **channel_config[key]}

    return result


def get_pipeline_stage(config: dict, stage: str) -> dict:
    """Get configuration for a pipeline stage."""
    pipeline = config.get("pipeline", {})
    return pipeline.get(stage, {})


def resolve_branch_to_channel(config: dict, branch: str) -> str | None:
    """Find which channel a branch belongs to."""
    pipeline = config.get("pipeline", {})

    for stage_name, stage_config in pipeline.items():
        stage_branch = stage_config.get("branch", "")

        if stage_branch == branch:
            return stage_config.get("channel")

        if "*" in stage_branch:
            pattern = stage_branch.replace("*", ".*")
            if re.match(f"^{pattern}$", branch):
                return stage_config.get("channel")

    hotfix = config.get("hotfix", {})
    if hotfix.get("enabled"):
        hotfix_pattern = hotfix.get("branch_pattern", "hotfix/*")
        pattern = hotfix_pattern.replace("*", ".*")
        if re.match(f"^{pattern}$", branch):
            return hotfix.get("channel", "stable")

    return None


def is_hotfix_branch(config: dict, branch: str) -> bool:
    """Check if branch is a hotfix branch."""
    hotfix = config.get("hotfix", {})
    if not hotfix.get("enabled"):
        return False

    pattern = hotfix.get("branch_pattern", "hotfix/*").replace("*", ".*")
    return bool(re.match(f"^{pattern}$", branch))


def get_build_command(config: dict, channel: str) -> str:
    """Get the build command for a channel."""
    build_config = config.get("build", {})
    base_command = build_config.get("command", "uv build")

    channel_config = get_channel_config(config, channel)
    channel_build = channel_config.get("build", {})

    parts = [base_command]

    if not channel_build.get("wheel", True) and channel_build.get("sdist", True):
        parts.append("--sdist")
    elif channel_build.get("wheel", True) and not channel_build.get("sdist", True):
        parts.append("--wheel")

    if channel_build.get("dev", False):
        return build_config.get("artifacts", {}).get("dev", {}).get("command", "uv build --dev")

    return " ".join(parts)


def get_version_files(config: dict) -> list[dict]:
    """Get version files to sync."""
    version_sync = config.get("version_sync", {})
    return version_sync.get("files", [])


def get_tag_format(config: dict, channel: str, is_hotfix: bool = False) -> str:
    """Get tag format for a channel."""
    if is_hotfix:
        hotfix = config.get("hotfix", {})
        prefix = hotfix.get("tag_prefix", "hotfix-v")
        return f"{prefix}{{version}}"

    channel_config = get_channel_config(config, channel)
    return channel_config.get("tag_format", "v{version}")


def is_channel_enabled(config: dict, channel: str) -> bool:
    """Check if a channel is enabled in the configuration."""
    channels = config.get("channels", {})
    channel_config = channels.get(channel, {})

    if not channel_config:
        return False

    # Default to True for backwards compatibility, but explicit 'enabled: false' disables
    return channel_config.get("enabled", True)


def get_enabled_channels(config: dict) -> list[str]:
    """Get list of all enabled channels."""
    channels = config.get("channels", {})
    enabled = []

    for name, channel_config in channels.items():
        if channel_config.get("enabled", True):
            enabled.append(name)

    return enabled


def get_workflow_mode(config: dict) -> str:
    """Get the workflow mode: 'single-branch' or 'multi-branch'."""
    workflow = config.get("workflow", {})
    return workflow.get("mode", "single-branch")


def get_primary_branch(config: dict) -> str:
    """Get the primary branch for releases."""
    workflow = config.get("workflow", {})
    return workflow.get("primary_branch", "main")


def validate_config(config: dict) -> list[str]:
    """Validate the configuration."""
    errors = []

    version = config.get("schema_version", "")
    if not version:
        errors.append("Missing 'schema_version'")

    pipeline = config.get("pipeline", {})
    if not pipeline:
        errors.append("No pipeline stages defined")

    has_release = False
    for stage_name, stage_config in pipeline.items():
        if not stage_config.get("branch"):
            errors.append(f"Pipeline stage '{stage_name}' has no branch")
        if not stage_config.get("channel"):
            errors.append(f"Pipeline stage '{stage_name}' has no channel")
        if stage_name == "release":
            has_release = True

    if not has_release:
        errors.append("No 'release' stage in pipeline")

    # Check that at least one channel is enabled
    channels = config.get("channels", {})
    enabled_channels = get_enabled_channels(config)
    if not enabled_channels:
        errors.append("No channels are enabled - at least one must be enabled")

    # Verify that pipeline channels reference enabled channels
    for stage_name, stage_config in pipeline.items():
        channel = stage_config.get("channel")
        if channel and channel not in enabled_channels:
            # This is a warning, not an error - stage may be disabled
            pass

    return errors


def export_shell_env(config: dict) -> str:
    """Export configuration as shell variables."""
    lines = []

    pipeline = config.get("pipeline", {})
    release_stage = pipeline.get("release", {})
    lines.append(f'export RELEASE_BRANCH="{release_stage.get("branch", "main")}"')

    build = config.get("build", {})
    lines.append(f'export BUILD_COMMAND="{build.get("command", "uv build")}"')
    lines.append(f'export BUILD_CLEAN="{format_value(build.get("clean_before", True))}"')

    safety = config.get("safety", {})
    lines.append(f'export REQUIRE_CLEAN="{format_value(safety.get("require_clean", True))}"')
    lines.append(f'export REQUIRE_SYNCED="{format_value(safety.get("require_synced", True))}"')
    lines.append(f'export BLOCK_PRERELEASE_PYPI="{format_value(safety.get("block_prerelease_pypi", True))}"')

    hotfix = config.get("hotfix", {})
    lines.append(f'export HOTFIX_ENABLED="{format_value(hotfix.get("enabled", True))}"')
    lines.append(f'export HOTFIX_PATTERN="{hotfix.get("branch_pattern", "hotfix/*")}"')
    lines.append(f'export HOTFIX_TARGET="{hotfix.get("target", "main")}"')

    publishing = config.get("publishing", {})
    pypi = publishing.get("pypi", {})
    lines.append(f'export PYPI_TOKEN_ENV="{pypi.get("token_env", "UV_PUBLISH_TOKEN")}"')

    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]

    try:
        config = load_config()
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if command == "get":
        if len(sys.argv) < 3:
            print("Error: Missing path", file=sys.stderr)
            sys.exit(1)
        value = get_nested_value(config, sys.argv[2])
        print(format_value(value))

    elif command == "get-channel":
        if len(sys.argv) < 3:
            print("Error: Missing channel", file=sys.stderr)
            sys.exit(1)
        channel_config = get_channel_config(config, sys.argv[2])
        print(json.dumps(channel_config, indent=2))

    elif command == "get-pipeline-stage":
        if len(sys.argv) < 3:
            print("Error: Missing stage", file=sys.stderr)
            sys.exit(1)
        stage_config = get_pipeline_stage(config, sys.argv[2])
        print(json.dumps(stage_config, indent=2))

    elif command == "resolve-branch":
        if len(sys.argv) < 3:
            print("Error: Missing branch", file=sys.stderr)
            sys.exit(1)
        channel = resolve_branch_to_channel(config, sys.argv[2])
        print(channel or "")

    elif command == "is-hotfix":
        if len(sys.argv) < 3:
            print("Error: Missing branch", file=sys.stderr)
            sys.exit(1)
        print("true" if is_hotfix_branch(config, sys.argv[2]) else "false")

    elif command == "build-command":
        if len(sys.argv) < 3:
            print("Error: Missing channel", file=sys.stderr)
            sys.exit(1)
        print(get_build_command(config, sys.argv[2]))

    elif command == "tag-format":
        if len(sys.argv) < 3:
            print("Error: Missing channel", file=sys.stderr)
            sys.exit(1)
        is_hotfix = len(sys.argv) > 3 and sys.argv[3] == "--hotfix"
        print(get_tag_format(config, sys.argv[2], is_hotfix))

    elif command == "get-version-files":
        files = get_version_files(config)
        print(json.dumps(files, indent=2))

    elif command == "validate":
        errors = validate_config(config)
        if errors:
            print("Configuration errors:", file=sys.stderr)
            for error in errors:
                print(f"  - {error}", file=sys.stderr)
            sys.exit(1)
        print("Configuration is valid")

    elif command == "export-env":
        print(export_shell_env(config))

    elif command == "is-channel-enabled":
        if len(sys.argv) < 3:
            print("Error: Missing channel", file=sys.stderr)
            sys.exit(1)
        print("true" if is_channel_enabled(config, sys.argv[2]) else "false")

    elif command == "list-enabled-channels":
        enabled = get_enabled_channels(config)
        print(" ".join(enabled))

    elif command == "get-workflow-mode":
        print(get_workflow_mode(config))

    elif command == "get-primary-branch":
        print(get_primary_branch(config))

    else:
        print(f"Unknown command: {command}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
