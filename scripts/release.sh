#!/usr/bin/env bash
# =============================================================================
# Universal Release Script v2.1 - Pipeline-Aware YAML Configuration
# =============================================================================
# A flexible, pipeline-driven release system with hotfix support.
#
# Features:
#   - Branch promotion pipeline (dev→test→staging→release)
#   - Hotfix workflow that bypasses normal pipeline
#   - Auto-detection: current branch → channel mapping
#   - Build artifact configuration (wheel, sdist, dev)
#   - Per-channel tag formats with hotfix prefix support
#   - Multi-channel releases (alpha, beta, rc, stable)
#   - Git-cliff changelog generation
#   - GitHub releases with release notes
#   - PyPI publishing (configurable per channel)
#   - Automatic backport-to branches after hotfix
#
# Usage:
#   ./release.sh [OPTIONS] [CHANNEL]
#
# Examples:
#   ./release.sh                      # Auto-detect channel from current branch
#   ./release.sh stable               # Release stable from pipeline config
#   ./release.sh --hotfix             # Release hotfix from hotfix/* branch
#   ./release.sh --dry-run stable     # Preview stable release
#   ./release.sh --config my.yaml stable  # Use custom config
#
# Configuration:
#   See release-config.yaml for all available options.
# =============================================================================

set -euo pipefail

# =============================================================================
# Script Setup
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
readonly PROJECT_ROOT

# Version of this release script
RELEASE_SCRIPT_VERSION="2.1.0"

# =============================================================================
# Colors and Output Functions
# =============================================================================

# Color codes (will be disabled if colors are turned off in config)
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

# Output functions
print_header() {
    echo -e "\n${BOLD}${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}${CYAN}  $1${NC}"
    echo -e "${BOLD}${CYAN}═══════════════════════════════════════════════════════════════${NC}"
}

print_section() {
    echo -e "\n${BOLD}${BLUE}▶ $1${NC}"
}

print_step() {
    echo -e "  ${BLUE}→${NC} $1"
}

print_info() {
    echo -e "  ${DIM}ℹ${NC} $1"
}

print_success() {
    echo -e "  ${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "  ${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "  ${RED}✗${NC} $1" >&2
}

print_debug() {
    if [[ "${VERBOSE:-false}" == "true" ]]; then
        echo -e "  ${DIM}[DEBUG]${NC} $1" >&2
    fi
}

# =============================================================================
# Configuration Loading
# =============================================================================

# Default configuration file
CONFIG_FILE=""
CONFIG_LOADED=false

# Python interpreter for config parsing
PYTHON_CMD=""

find_python() {
    # Find a working Python interpreter
    for cmd in python3 python; do
        if command -v "$cmd" &>/dev/null; then
            PYTHON_CMD="$cmd"
            return 0
        fi
    done

    # Try uv's Python
    if command -v uv &>/dev/null; then
        PYTHON_CMD="uv run python"
        return 0
    fi

    return 1
}

find_config_file() {
    local candidates=(
        "$PROJECT_ROOT/release-config.yaml"
        "$PROJECT_ROOT/release-config.yml"
        "$PROJECT_ROOT/.release-config.yaml"
        "$PROJECT_ROOT/.release-config.yml"
    )

    for candidate in "${candidates[@]}"; do
        if [[ -f "$candidate" ]]; then
            CONFIG_FILE="$candidate"
            return 0
        fi
    done

    return 1
}

# Configuration getter - uses Python helper
config_get() {
    local path="$1"
    local default="${2:-}"

    if [[ "$CONFIG_LOADED" != "true" ]]; then
        echo "$default"
        return
    fi

    local value
    value=$($PYTHON_CMD "$SCRIPT_DIR/release_config.py" get "$path" 2>/dev/null) || value=""

    if [[ -z "$value" || "$value" == "null" ]]; then
        echo "$default"
    else
        echo "$value"
    fi
}

config_get_list() {
    local path="$1"

    if [[ "$CONFIG_LOADED" != "true" ]]; then
        return
    fi

    $PYTHON_CMD "$SCRIPT_DIR/release_config.py" get-list "$path" 2>/dev/null || true
}

config_get_channel() {
    local channel="$1"

    if [[ "$CONFIG_LOADED" != "true" ]]; then
        echo "{}"
        return
    fi

    $PYTHON_CMD "$SCRIPT_DIR/release_config.py" get-channel "$channel" 2>/dev/null || echo "{}"
}

validate_config() {
    if [[ "$CONFIG_LOADED" != "true" ]]; then
        print_warning "No configuration file loaded, using defaults"
        return 0
    fi

    print_step "Validating configuration..."

    if ! $PYTHON_CMD "$SCRIPT_DIR/release_config.py" validate 2>/dev/null; then
        print_error "Configuration validation failed"
        return 1
    fi

    print_success "Configuration is valid"
}

load_config() {
    print_section "Loading Configuration"

    # Find Python
    if ! find_python; then
        print_error "Python is required for YAML parsing"
        exit 1
    fi
    print_debug "Using Python: $PYTHON_CMD"

    # Find config file
    if [[ -n "$CONFIG_FILE" ]]; then
        if [[ ! -f "$CONFIG_FILE" ]]; then
            print_error "Config file not found: $CONFIG_FILE"
            exit 1
        fi
    elif ! find_config_file; then
        print_warning "No release-config.yaml found, using defaults"
        return 0
    fi

    print_info "Config file: $CONFIG_FILE"
    CONFIG_LOADED=true

    # Validate
    validate_config || exit 1

    # Load global settings
    PROJECT_NAME=$(config_get "project.name" "unknown")
    PACKAGE_MANAGER=$(config_get "project.package_manager" "uv")
    BUILD_CMD=$(config_get "project.build_command" "uv build")
    PUBLISH_CMD=$(config_get "project.publish_command" "uv publish")

    PRIMARY_BRANCH=$(config_get "git.primary_branch" "main")
    TAG_FORMAT=$(config_get "git.tag_format" "v{version}")

    CHANGELOG_GENERATOR=$(config_get "changelog.generator" "git-cliff")
    CHANGELOG_FILE=$(config_get "changelog.file" "CHANGELOG.md")
    CHANGELOG_CONFIG=$(config_get "changelog.config" "cliff.toml")

    REQUIRE_CLEAN=$(config_get "safety.require_clean" "true")
    REQUIRE_SYNCED=$(config_get "safety.require_synced" "true")
    BLOCK_PRERELEASE_PYPI=$(config_get "safety.block_prerelease_pypi" "true")

    print_success "Configuration loaded for project: $PROJECT_NAME"
}

# =============================================================================
# Pipeline & Hotfix Helpers (v2.1)
# =============================================================================

# Resolve current branch to a release channel using pipeline config
resolve_branch_to_channel() {
    local branch="$1"

    if [[ "$CONFIG_LOADED" != "true" ]]; then
        # Fallback: map common branches to channels
        case "$branch" in
            develop|dev)  echo "alpha" ;;
            test)         echo "beta" ;;
            staging)      echo "rc" ;;
            main|master)  echo "stable" ;;
            *)            echo "" ;;
        esac
        return
    fi

    $PYTHON_CMD "$SCRIPT_DIR/release_config.py" resolve-branch "$branch" 2>/dev/null || echo ""
}

# Check if current branch is a hotfix branch
is_hotfix_branch() {
    local branch="$1"

    if [[ "$CONFIG_LOADED" != "true" ]]; then
        # Default: branches starting with hotfix/
        [[ "$branch" == hotfix/* ]] && echo "true" || echo "false"
        return
    fi

    $PYTHON_CMD "$SCRIPT_DIR/release_config.py" is-hotfix "$branch" 2>/dev/null || echo "false"
}

# Get the build command for a channel (respects wheel/sdist/dev config)
get_build_command() {
    local channel="$1"

    if [[ "$CONFIG_LOADED" != "true" ]]; then
        echo "uv build"
        return
    fi

    $PYTHON_CMD "$SCRIPT_DIR/release_config.py" build-command "$channel" 2>/dev/null || echo "uv build"
}

# Get the tag format for a channel (with hotfix prefix support)
get_tag_format() {
    local channel="$1"
    local is_hotfix="${2:-false}"

    if [[ "$CONFIG_LOADED" != "true" ]]; then
        if [[ "$is_hotfix" == "true" ]]; then
            echo "hotfix-v{version}"
        else
            echo "v{version}"
        fi
        return
    fi

    local hotfix_flag=""
    [[ "$is_hotfix" == "true" ]] && hotfix_flag="--hotfix"

    $PYTHON_CMD "$SCRIPT_DIR/release_config.py" tag-format "$channel" $hotfix_flag 2>/dev/null || echo "v{version}"
}

# Get pipeline stage info for a stage name
get_pipeline_stage() {
    local stage="$1"

    if [[ "$CONFIG_LOADED" != "true" ]]; then
        echo "{}"
        return
    fi

    $PYTHON_CMD "$SCRIPT_DIR/release_config.py" get-pipeline-stage "$stage" 2>/dev/null || echo "{}"
}

# Get hotfix backport targets
get_hotfix_backport_targets() {
    if [[ "$CONFIG_LOADED" != "true" ]]; then
        echo ""
        return
    fi

    local backport_to
    backport_to=$(config_get "hotfix.backport_to" "[]")

    # Parse JSON array to space-separated list
    echo "$backport_to" | $PYTHON_CMD -c "
import json
import sys
try:
    data = json.load(sys.stdin)
    if isinstance(data, list):
        print(' '.join(data))
except:
    pass
" 2>/dev/null || echo ""
}

# Get the target branch for hotfixes (usually main)
get_hotfix_target() {
    config_get "hotfix.target" "main"
}

# Check if a channel is enabled in the configuration
is_channel_enabled() {
    local channel="$1"

    if [[ "$CONFIG_LOADED" != "true" ]]; then
        # Without config, only allow stable
        [[ "$channel" == "stable" ]] && echo "true" || echo "false"
        return
    fi

    $PYTHON_CMD "$SCRIPT_DIR/release_config.py" is-channel-enabled "$channel" 2>/dev/null || echo "false"
}

# Get list of enabled channels
get_enabled_channels() {
    if [[ "$CONFIG_LOADED" != "true" ]]; then
        echo "stable"
        return
    fi

    $PYTHON_CMD "$SCRIPT_DIR/release_config.py" list-enabled-channels 2>/dev/null || echo "stable"
}

# Get workflow mode (single-branch or multi-branch)
get_workflow_mode() {
    if [[ "$CONFIG_LOADED" != "true" ]]; then
        echo "single-branch"
        return
    fi

    $PYTHON_CMD "$SCRIPT_DIR/release_config.py" get-workflow-mode 2>/dev/null || echo "single-branch"
}

# Validate that a channel is enabled, exit with error if not
validate_channel_enabled() {
    local channel="$1"

    if [[ "$(is_channel_enabled "$channel")" != "true" ]]; then
        local enabled_channels
        enabled_channels=$(get_enabled_channels)

        print_error "Channel '$channel' is NOT enabled in this project's configuration"
        echo ""
        print_info "Available channels: $enabled_channels"
        print_info "To enable more channels, edit release-config.yaml"
        print_info "and set 'enabled: true' under the desired channel."
        exit 1
    fi
}

# =============================================================================
# Command Line Parsing
# =============================================================================

print_help() {
    cat << 'EOF'
Universal Release Script v2.1 - Pipeline-Aware YAML Configuration

USAGE:
    ./release.sh [OPTIONS] [CHANNEL] [BRANCH]

CHANNELS:
    (auto)     Auto-detect from current branch using pipeline config
    stable     Create a stable release (GitHub + PyPI if configured)
    alpha      Create an alpha pre-release (GitHub only)
    beta       Create a beta pre-release (GitHub only)
    rc         Create a release candidate (GitHub only)

OPTIONS:
    --config <file>     Use a specific configuration file
    --dry-run           Preview the release without making changes
    --hotfix            Treat as hotfix (bypass pipeline, use hotfix tag prefix)
    --no-pypi           Skip PyPI publishing (even for stable)
    --no-github         Skip GitHub release creation
    --no-backport       Skip automatic backport after hotfix
    --force             Skip safety checks
    --verbose           Show detailed debug output
    -h, --help          Show this help message
    -v, --version       Show script version

ARGUMENTS:
    CHANNEL             The release channel (optional if on configured branch)
    BRANCH              Override the default branch for this channel

PIPELINE WORKFLOW:
    The script uses pipeline configuration to determine which channel
    corresponds to which branch:
        develop  → alpha channel
        test     → beta channel
        staging  → rc channel
        main     → stable channel

HOTFIX WORKFLOW:
    Hotfixes bypass the normal pipeline for critical fixes:
    1. Create branch: git checkout -b hotfix/critical-fix main
    2. Make fixes and commit
    3. Release: ./release.sh --hotfix (or auto-detected from branch name)
    4. Backport is automatic to configured branches (develop, staging)

EXAMPLES:
    # Auto-detect channel from current branch
    ./release.sh

    # Release stable from main branch
    ./release.sh stable

    # Release alpha from develop branch
    ./release.sh alpha develop

    # Preview what a stable release would do
    ./release.sh --dry-run stable

    # Release hotfix (auto-detected from hotfix/* branch)
    ./release.sh --hotfix

    # Release without publishing to PyPI
    ./release.sh --no-pypi stable

CONFIGURATION:
    The script reads from release-config.yaml in the project root.
    See that file for all available configuration options including:
    - Pipeline stages (branch → channel mapping)
    - Hotfix workflow settings
    - Build artifact configuration (wheel, sdist, dev)
    - Per-channel tag formats
    - Hooks for custom commands
    - Safety checks

EOF
}

# Script options with defaults
DRY_RUN=false
NO_PYPI=false
NO_GITHUB=false
NO_BACKPORT=false
FORCE=false
VERBOSE=false
HOTFIX_MODE=false
AUTO_DETECT_CHANNEL=false
CHANNEL=""
OVERRIDE_BRANCH=""

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --config)
                if [[ $# -lt 2 ]]; then
                    print_error "--config requires a file path"
                    exit 1
                fi
                CONFIG_FILE="$2"
                shift 2
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --hotfix)
                HOTFIX_MODE=true
                shift
                ;;
            --no-pypi)
                NO_PYPI=true
                shift
                ;;
            --no-github)
                NO_GITHUB=true
                shift
                ;;
            --no-backport)
                NO_BACKPORT=true
                shift
                ;;
            --force)
                FORCE=true
                shift
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            -h|--help)
                print_help
                exit 0
                ;;
            -v|--version)
                echo "Universal Release Script v$RELEASE_SCRIPT_VERSION"
                exit 0
                ;;
            stable|alpha|beta|rc)
                CHANNEL="$1"
                shift
                # Check if next arg is a branch name (doesn't start with --)
                if [[ $# -gt 0 && ! "$1" =~ ^-- ]]; then
                    OVERRIDE_BRANCH="$1"
                    shift
                fi
                ;;
            *)
                print_error "Unknown option: $1"
                print_info "Use --help for usage information"
                exit 1
                ;;
        esac
    done

    # If no channel specified, mark for auto-detection (will happen after config load)
    if [[ -z "$CHANNEL" ]]; then
        AUTO_DETECT_CHANNEL=true
    fi
}

# =============================================================================
# Prerequisites Check
# =============================================================================

check_prerequisites() {
    print_section "Checking Prerequisites"

    local missing=()

    # Check uv or other package manager
    if [[ "$PACKAGE_MANAGER" == "uv" ]]; then
        if ! command -v uv &>/dev/null; then
            missing+=("uv")
        fi
    elif [[ "$PACKAGE_MANAGER" == "poetry" ]]; then
        if ! command -v poetry &>/dev/null; then
            missing+=("poetry")
        fi
    fi

    # Check git
    if ! command -v git &>/dev/null; then
        missing+=("git")
    fi

    # Check changelog generator
    if [[ "$CHANGELOG_GENERATOR" == "git-cliff" ]]; then
        if ! git cliff -h &>/dev/null 2>&1; then
            missing+=("git-cliff")
        fi
    fi

    # Check GitHub CLI (if GitHub releases enabled)
    if [[ "$NO_GITHUB" != "true" ]]; then
        if ! command -v gh &>/dev/null; then
            missing+=("gh (GitHub CLI)")
        fi
    fi

    if [[ ${#missing[@]} -gt 0 ]]; then
        print_error "Missing required tools: ${missing[*]}"
        exit 1
    fi

    print_success "All prerequisites installed"

    # Check for pyproject.toml
    if [[ ! -f "pyproject.toml" ]]; then
        print_error "pyproject.toml not found in project root"
        exit 1
    fi
    print_success "Project configuration found"
}

# =============================================================================
# Git Helpers
# =============================================================================

start_branch=""
declare -a RELEASE_NOTES_FILES=()

switch_to_branch() {
    local target_branch="$1"
    local current_branch
    current_branch="$(git rev-parse --abbrev-ref HEAD)"

    if [[ "$current_branch" != "$target_branch" ]]; then
        print_step "Switching from '$current_branch' to '$target_branch'..."
        if [[ "$DRY_RUN" == "true" ]]; then
            print_info "[DRY-RUN] Would checkout branch: $target_branch"
        else
            git checkout "$target_branch"
        fi
    fi
}

restore_original_branch() {
    local current_branch
    current_branch="$(git rev-parse --abbrev-ref HEAD)"

    if [[ "$current_branch" != "$start_branch" ]]; then
        print_step "Restoring original branch '$start_branch'..."
        git checkout "$start_branch" 2>/dev/null || {
            print_warning "Could not restore original branch '$start_branch'"
            return 1
        }
    fi
}

cleanup_temporary_files() {
    if [[ ${#RELEASE_NOTES_FILES[@]} -gt 0 ]]; then
        print_step "Cleaning up ${#RELEASE_NOTES_FILES[@]} temporary file(s)..."
        for notes_file in "${RELEASE_NOTES_FILES[@]}"; do
            if [[ -f "$PROJECT_ROOT/$notes_file" ]]; then
                rm -f "$PROJECT_ROOT/$notes_file" 2>/dev/null || true
                print_debug "Deleted: $notes_file"
            fi
        done
    fi
}

ensure_clean() {
    if [[ "$FORCE" == "true" ]]; then
        print_warning "Skipping clean check (--force)"
        return 0
    fi

    if [[ "$REQUIRE_CLEAN" != "true" ]]; then
        return 0
    fi

    if ! git diff --quiet || ! git diff --cached --quiet; then
        print_error "Uncommitted changes detected"
        print_info "Commit or stash changes before releasing"
        exit 1
    fi

    print_success "Working directory is clean"
}

ensure_synced() {
    local branch="$1"

    if [[ "$FORCE" == "true" ]]; then
        print_warning "Skipping sync check (--force)"
        return 0
    fi

    if [[ "$REQUIRE_SYNCED" != "true" ]]; then
        return 0
    fi

    git fetch origin "$branch" 2>/dev/null || {
        print_warning "Could not fetch from origin/$branch"
        return 0
    }

    if [[ "$(git rev-parse HEAD)" != "$(git rev-parse "origin/$branch" 2>/dev/null)" ]]; then
        print_error "Branch '$branch' is not in sync with origin"
        print_info "Pull or rebase to sync before releasing"
        exit 1
    fi

    print_success "Branch is in sync with remote"
}

# =============================================================================
# Version Management
# =============================================================================

compute_bump_args() {
    local channel="$1"
    local current_version="$2"
    local bump=()

    case "$channel" in
        alpha)
            if [[ "$current_version" == *a[0-9]* ]]; then
                bump+=(--bump alpha)
            else
                bump+=(--bump patch --bump alpha)
            fi
            ;;
        beta)
            if [[ "$current_version" == *b[0-9]* ]]; then
                bump+=(--bump beta)
            else
                bump+=(--bump patch --bump beta)
            fi
            ;;
        rc)
            if [[ "$current_version" == *rc[0-9]* ]]; then
                bump+=(--bump rc)
            else
                bump+=(--bump patch --bump rc)
            fi
            ;;
        stable)
            if [[ "$current_version" == *a[0-9]* || "$current_version" == *b[0-9]* || "$current_version" == *rc[0-9]* ]]; then
                bump+=(--bump stable)
            else
                bump+=(--bump patch)
            fi
            ;;
    esac

    echo "${bump[@]}"
}

sync_version_files() {
    local new_version="$1"

    print_step "Syncing version files..."

    if [[ "$CONFIG_LOADED" != "true" ]]; then
        # Default: sync apias/__init__.py
        if [[ -f "apias/__init__.py" ]]; then
            if [[ "$DRY_RUN" == "true" ]]; then
                print_info "[DRY-RUN] Would update apias/__init__.py"
            else
                sed -i '' "s/^__version__ = \".*\"/__version__ = \"$new_version\"/" "apias/__init__.py"
                print_success "Updated apias/__init__.py"
            fi
        fi
        return
    fi

    # Read version files from config
    local version_files
    version_files=$($PYTHON_CMD "$SCRIPT_DIR/release_config.py" get-version-files 2>/dev/null) || version_files="[]"

    if [[ "$version_files" == "[]" ]]; then
        print_debug "No version files configured"
        return
    fi

    # Parse JSON and update each file
    echo "$version_files" | $PYTHON_CMD -c "
import json
import sys
import re

data = json.load(sys.stdin)
version = '$new_version'
dry_run = '$DRY_RUN' == 'true'

for entry in data:
    path = entry.get('path', '')
    search = entry.get('search', '')
    replace = entry.get('replace', '').replace('{version}', version)

    if not path or not search:
        continue

    try:
        with open(path, 'r') as f:
            content = f.read()

        new_content = re.sub(search, replace, content, flags=re.MULTILINE)

        if dry_run:
            print(f'[DRY-RUN] Would update {path}')
        else:
            with open(path, 'w') as f:
                f.write(new_content)
            print(f'Updated {path}')
    except Exception as e:
        print(f'Warning: Could not update {path}: {e}', file=sys.stderr)
" 2>&1 | while read -r line; do
        if [[ "$line" == *"[DRY-RUN]"* ]]; then
            print_info "$line"
        elif [[ "$line" == "Updated"* ]]; then
            print_success "$line"
        else
            print_warning "$line"
        fi
    done
}

# =============================================================================
# Changelog Generation
# =============================================================================

generate_changelog_and_notes() {
    local new_version="$1"
    local notes_file=".release-notes-${new_version}.md"

    print_step "Generating changelog..."

    if [[ "$DRY_RUN" == "true" ]]; then
        print_info "[DRY-RUN] Would regenerate $CHANGELOG_FILE"
        print_info "[DRY-RUN] Would create release notes: $notes_file"
        echo "$notes_file"
        return
    fi

    case "$CHANGELOG_GENERATOR" in
        git-cliff)
            git cliff --tag "$new_version" -o "$CHANGELOG_FILE"
            print_success "Updated $CHANGELOG_FILE"

            git cliff --tag "$new_version" --latest -o "$notes_file"
            print_success "Created release notes: $notes_file"
            ;;
        *)
            print_warning "Unknown changelog generator: $CHANGELOG_GENERATOR"
            echo ""
            return
            ;;
    esac

    RELEASE_NOTES_FILES+=("$notes_file")
    echo "$notes_file"
}

# =============================================================================
# Hooks Execution
# =============================================================================

run_hooks() {
    local hook_name="$1"

    if [[ "$CONFIG_LOADED" != "true" ]]; then
        return 0
    fi

    local hooks
    hooks=$($PYTHON_CMD "$SCRIPT_DIR/release_config.py" get-hooks "$hook_name" 2>/dev/null) || return 0

    if [[ -z "$hooks" ]]; then
        return 0
    fi

    print_step "Running $hook_name hooks..."

    while IFS= read -r cmd; do
        if [[ -n "$cmd" ]]; then
            print_debug "Hook command: $cmd"
            if [[ "$DRY_RUN" == "true" ]]; then
                print_info "[DRY-RUN] Would run: $cmd"
            else
                if ! eval "$cmd"; then
                    print_error "Hook command failed: $cmd"
                    return 1
                fi
            fi
        fi
    done <<< "$hooks"
}

# =============================================================================
# Release Channel
# =============================================================================

release_channel() {
    local channel="$1"
    local branch="$2"

    print_header "Releasing $channel from $branch"

    if [[ "$DRY_RUN" == "true" ]]; then
        print_warning "DRY-RUN MODE - No changes will be made"
    fi

    # Run pre-release hooks
    run_hooks "pre_release" || exit 1

    # Switch to branch
    switch_to_branch "$branch"

    # Safety checks
    ensure_clean
    ensure_synced "$branch"

    # Get current version
    local version_line
    version_line="$(uv version)"
    local current_version="${version_line##* }"

    print_info "Current version: $current_version"

    # Compute bump arguments
    local -a bump_args=()
    read -r -a bump_args <<< "$(compute_bump_args "$channel" "$current_version")"

    print_debug "Bump args: ${bump_args[*]}"

    # Bump version
    print_step "Bumping version..."
    if [[ "$DRY_RUN" == "true" ]]; then
        print_info "[DRY-RUN] Would run: uv version --no-sync ${bump_args[*]}"
        # Calculate what the new version would be
        local new_version
        new_version=$(uv version --no-sync "${bump_args[@]}" 2>/dev/null || echo "$current_version")
        new_version="${new_version##* }"
    else
        uv version --no-sync "${bump_args[@]}"
        local new_line
        new_line="$(uv version)"
        local new_version="${new_line##* }"
    fi

    print_success "New version: $new_version"

    # Sync version files
    sync_version_files "$new_version"

    # Run post-bump hooks
    run_hooks "post_bump" || exit 1

    # Generate changelog
    local notes_file
    notes_file="$(generate_changelog_and_notes "$new_version")"

    # Create tag name using channel-specific format
    local tag_format
    tag_format=$(get_tag_format "$channel" "false")
    local tag="${tag_format//\{version\}/$new_version}"
    print_info "Tag: $tag"

    # Stage files
    print_step "Staging files..."
    if [[ "$DRY_RUN" == "true" ]]; then
        print_info "[DRY-RUN] Would stage: pyproject.toml, $CHANGELOG_FILE, version files"
    else
        git add pyproject.toml
        [[ -f "$CHANGELOG_FILE" ]] && git add "$CHANGELOG_FILE"

        # Stage version files from config
        if [[ "$CONFIG_LOADED" == "true" ]]; then
            local version_files
            version_files=$($PYTHON_CMD "$SCRIPT_DIR/release_config.py" get-version-files 2>/dev/null) || version_files="[]"
            echo "$version_files" | $PYTHON_CMD -c "
import json
import sys
data = json.load(sys.stdin)
for entry in data:
    print(entry.get('path', ''))
" 2>/dev/null | while read -r path; do
                [[ -f "$path" ]] && git add "$path"
            done
        else
            [[ -f "apias/__init__.py" ]] && git add "apias/__init__.py"
        fi
    fi

    # Commit
    local commit_msg
    commit_msg=$(config_get "git.commit_messages.release" "chore(release): release {channel} {version}")
    commit_msg="${commit_msg//\{channel\}/$channel}"
    commit_msg="${commit_msg//\{version\}/$new_version}"

    print_step "Committing changes..."
    if [[ "$DRY_RUN" == "true" ]]; then
        print_info "[DRY-RUN] Would commit: $commit_msg"
    else
        git commit --no-verify -m "$commit_msg"
        print_success "Created release commit"
    fi

    # Build - use channel-specific build command
    local build_cmd
    build_cmd=$(get_build_command "$channel")
    print_step "Building distributions..."
    print_debug "Build command: $build_cmd"
    if [[ "$DRY_RUN" == "true" ]]; then
        print_info "[DRY-RUN] Would run: rm -rf dist/ && uv sync && $build_cmd"
    else
        rm -rf dist/
        uv sync
        eval "$build_cmd"
        print_success "Built distributions"
    fi

    # Run post-build hooks
    run_hooks "post_build" || exit 1

    # Handle lockfile changes
    if [[ "$DRY_RUN" != "true" ]] && [[ -n $(git status --porcelain uv.lock 2>/dev/null) ]]; then
        git add uv.lock
        git commit --no-verify -m "chore: update lockfile for $new_version" || true
    fi

    # Tag
    print_step "Creating tag..."
    if [[ "$DRY_RUN" == "true" ]]; then
        print_info "[DRY-RUN] Would create tag: $tag"
    else
        git tag "$tag"
        print_success "Created tag: $tag"
    fi

    # Push
    print_step "Pushing to remote..."
    if [[ "$DRY_RUN" == "true" ]]; then
        print_info "[DRY-RUN] Would push branch and tag"
    else
        git push origin "$branch" "$tag"
        print_success "Pushed to origin"
    fi

    # GitHub Release
    if [[ "$NO_GITHUB" != "true" ]]; then
        create_github_release "$channel" "$tag" "$new_version" "$notes_file"
    fi

    # PyPI Publish
    publish_to_pypi "$channel" "$new_version"

    # Run post-release hooks
    run_hooks "post_release" || true

    # Cleanup notes file
    if [[ -n "$notes_file" && -f "$PROJECT_ROOT/$notes_file" && "$DRY_RUN" != "true" ]]; then
        rm -f "$PROJECT_ROOT/$notes_file"
        print_debug "Cleaned up: $notes_file"
    fi

    print_success "Finished release: $channel $new_version"
}

# =============================================================================
# Hotfix Release (v2.1)
# =============================================================================
# Hotfixes bypass the normal pipeline for critical fixes.
# They use a special tag prefix and auto-backport to configured branches.

release_hotfix() {
    local channel="$1"
    local branch="$2"  # This is the hotfix/* branch

    print_header "HOTFIX Release: $channel from $branch"

    if [[ "$DRY_RUN" == "true" ]]; then
        print_warning "DRY-RUN MODE - No changes will be made"
    fi

    # Run hotfix pre-release hooks
    run_hooks "hotfix.pre_release" || exit 1

    # Stay on the hotfix branch (don't switch)
    local current_branch
    current_branch="$(git rev-parse --abbrev-ref HEAD)"
    if [[ "$current_branch" != "$branch" ]]; then
        switch_to_branch "$branch"
    fi

    # Safety checks
    ensure_clean
    ensure_synced "$branch"

    # Get current version
    local version_line
    version_line="$(uv version)"
    local current_version="${version_line##* }"

    print_info "Current version: $current_version"

    # For hotfixes, always bump patch and add stable suffix
    local -a bump_args=(--bump patch)
    print_debug "Hotfix bump args: ${bump_args[*]}"

    # Bump version
    print_step "Bumping version (hotfix)..."
    local new_version
    if [[ "$DRY_RUN" == "true" ]]; then
        print_info "[DRY-RUN] Would run: uv version --no-sync ${bump_args[*]}"
        new_version=$(uv version --no-sync "${bump_args[@]}" 2>/dev/null || echo "$current_version")
        new_version="${new_version##* }"
    else
        uv version --no-sync "${bump_args[@]}"
        local new_line
        new_line="$(uv version)"
        new_version="${new_line##* }"
    fi

    print_success "Hotfix version: $new_version"

    # Sync version files
    sync_version_files "$new_version"

    # Generate changelog
    local notes_file
    notes_file="$(generate_changelog_and_notes "$new_version")"

    # Create tag name with HOTFIX prefix
    local tag_format
    tag_format=$(get_tag_format "$channel" "true")  # true = is_hotfix
    local tag="${tag_format//\{version\}/$new_version}"
    print_info "Hotfix Tag: $tag"

    # Stage files
    print_step "Staging files..."
    if [[ "$DRY_RUN" == "true" ]]; then
        print_info "[DRY-RUN] Would stage: pyproject.toml, $CHANGELOG_FILE, version files"
    else
        git add pyproject.toml
        [[ -f "$CHANGELOG_FILE" ]] && git add "$CHANGELOG_FILE"

        # Stage version files from config
        if [[ "$CONFIG_LOADED" == "true" ]]; then
            local version_files
            version_files=$($PYTHON_CMD "$SCRIPT_DIR/release_config.py" get-version-files 2>/dev/null) || version_files="[]"
            echo "$version_files" | $PYTHON_CMD -c "
import json
import sys
data = json.load(sys.stdin)
for entry in data:
    print(entry.get('path', ''))
" 2>/dev/null | while read -r path; do
                [[ -f "$path" ]] && git add "$path"
            done
        else
            [[ -f "apias/__init__.py" ]] && git add "apias/__init__.py"
        fi
    fi

    # Commit
    local commit_msg="hotfix(release): $new_version"
    print_step "Committing changes..."
    if [[ "$DRY_RUN" == "true" ]]; then
        print_info "[DRY-RUN] Would commit: $commit_msg"
    else
        git commit --no-verify -m "$commit_msg"
        print_success "Created hotfix commit"
    fi

    # Build
    local build_cmd
    build_cmd=$(get_build_command "$channel")
    print_step "Building distributions..."
    if [[ "$DRY_RUN" == "true" ]]; then
        print_info "[DRY-RUN] Would run: rm -rf dist/ && uv sync && $build_cmd"
    else
        rm -rf dist/
        uv sync
        eval "$build_cmd"
        print_success "Built distributions"
    fi

    # Handle lockfile changes
    if [[ "$DRY_RUN" != "true" ]] && [[ -n $(git status --porcelain uv.lock 2>/dev/null) ]]; then
        git add uv.lock
        git commit --no-verify -m "chore: update lockfile for hotfix $new_version" || true
    fi

    # Tag
    print_step "Creating hotfix tag..."
    if [[ "$DRY_RUN" == "true" ]]; then
        print_info "[DRY-RUN] Would create tag: $tag"
    else
        git tag "$tag"
        print_success "Created tag: $tag"
    fi

    # Push hotfix branch and tag
    print_step "Pushing hotfix to remote..."
    if [[ "$DRY_RUN" == "true" ]]; then
        print_info "[DRY-RUN] Would push branch and tag"
    else
        git push origin "$branch" "$tag"
        print_success "Pushed hotfix to origin"
    fi

    # Merge hotfix into target branch (usually main)
    local hotfix_target
    hotfix_target=$(get_hotfix_target)
    print_step "Merging hotfix into $hotfix_target..."
    if [[ "$DRY_RUN" == "true" ]]; then
        print_info "[DRY-RUN] Would merge $branch into $hotfix_target"
    else
        git checkout "$hotfix_target"
        git merge --no-ff "$branch" -m "Merge hotfix $new_version into $hotfix_target"
        git push origin "$hotfix_target"
        print_success "Merged hotfix into $hotfix_target"
    fi

    # GitHub Release
    if [[ "$NO_GITHUB" != "true" ]]; then
        create_github_release "$channel" "$tag" "$new_version" "$notes_file"
    fi

    # PyPI Publish
    publish_to_pypi "$channel" "$new_version"

    # Backport to configured branches
    if [[ "$NO_BACKPORT" != "true" ]]; then
        backport_hotfix "$branch" "$new_version"
    fi

    # Run hotfix post-release hooks
    run_hooks "hotfix.post_release" || true

    # Cleanup notes file
    if [[ -n "$notes_file" && -f "$PROJECT_ROOT/$notes_file" && "$DRY_RUN" != "true" ]]; then
        rm -f "$PROJECT_ROOT/$notes_file"
        print_debug "Cleaned up: $notes_file"
    fi

    print_success "Finished hotfix release: $new_version"
}

# Backport hotfix to configured branches (e.g., develop, staging)
backport_hotfix() {
    local hotfix_branch="$1"
    local version="$2"

    local backport_targets
    backport_targets=$(get_hotfix_backport_targets)

    if [[ -z "$backport_targets" ]]; then
        print_info "No backport targets configured"
        return
    fi

    print_section "Backporting Hotfix"
    print_info "Targets: $backport_targets"

    for target in $backport_targets; do
        print_step "Backporting to $target..."

        if [[ "$DRY_RUN" == "true" ]]; then
            print_info "[DRY-RUN] Would cherry-pick hotfix commits to $target"
            continue
        fi

        # Check if target branch exists
        if ! git show-ref --verify --quiet "refs/heads/$target" 2>/dev/null; then
            if ! git show-ref --verify --quiet "refs/remotes/origin/$target" 2>/dev/null; then
                print_warning "Branch '$target' does not exist, skipping"
                continue
            fi
            # Create local tracking branch
            git checkout -b "$target" "origin/$target" 2>/dev/null || {
                print_warning "Could not checkout $target, skipping"
                continue
            }
        else
            git checkout "$target"
        fi

        # Try to merge the hotfix branch
        if git merge --no-ff "$hotfix_branch" -m "Backport hotfix $version to $target"; then
            git push origin "$target"
            print_success "Backported to $target"
        else
            print_warning "Merge conflict in $target - manual resolution needed"
            git merge --abort 2>/dev/null || true
        fi
    done
}

create_github_release() {
    local channel="$1"
    local tag="$2"
    local version="$3"
    local notes_file="$4"

    print_step "Creating GitHub release..."

    # Determine if prerelease
    local prerelease_flag=""
    if [[ "$channel" != "stable" ]]; then
        prerelease_flag="--prerelease"
    fi

    if [[ "$DRY_RUN" == "true" ]]; then
        print_info "[DRY-RUN] Would create GitHub release: $tag $prerelease_flag"
        return
    fi

    local title="$tag"
    local fallback_notes="Release $version"

    if [[ -n "$notes_file" && -s "$notes_file" ]]; then
        gh release create "$tag" dist/* --title "$title" --notes-file "$notes_file" $prerelease_flag
    else
        gh release create "$tag" dist/* --title "$title" --notes "$fallback_notes" $prerelease_flag
    fi

    print_success "Created GitHub release: $tag"
}

publish_to_pypi() {
    local channel="$1"
    local version="$2"

    # Check if publishing should be skipped
    if [[ "$NO_PYPI" == "true" ]]; then
        print_info "Skipping PyPI publish (--no-pypi flag)"
        return
    fi

    # Get channel config
    local publish_pypi
    publish_pypi=$(config_get "channels.$channel.publish_pypi" "false")

    if [[ "$publish_pypi" != "true" ]]; then
        print_info "Skipping PyPI publish for $channel channel (not configured)"
        return
    fi

    # Safety check: block pre-release versions
    if [[ "$BLOCK_PRERELEASE_PYPI" == "true" ]]; then
        if [[ "$version" == *a[0-9]* || "$version" == *b[0-9]* || "$version" == *rc[0-9]* ]]; then
            print_error "SAFETY: Pre-release version $version cannot be published to PyPI"
            exit 1
        fi
    fi

    # Check for token
    local token_env
    token_env=$(config_get "publishing.pypi.token_env" "UV_PUBLISH_TOKEN")

    if [[ -z "${!token_env:-}" ]]; then
        print_error "$token_env is not set"
        print_info "Set your PyPI token or use --no-pypi"
        exit 1
    fi

    print_step "Publishing to PyPI..."

    if [[ "$DRY_RUN" == "true" ]]; then
        print_info "[DRY-RUN] Would run: $PUBLISH_CMD"
        return
    fi

    echo ""
    print_warning "PUBLISHING TO PyPI"
    print_info "Version: $version"
    print_info "Channel: $channel"
    echo ""

    eval "$PUBLISH_CMD"
    print_success "Published to PyPI: $version"
}

# =============================================================================
# Main Entry Point
# =============================================================================

main() {
    # Save starting branch
    start_branch="$(git rev-parse --abbrev-ref HEAD)"

    # Parse command line arguments
    parse_args "$@"

    # Load configuration
    load_config

    # Check prerequisites
    check_prerequisites

    # =========================================================================
    # Auto-detection and Hotfix Resolution (v2.1)
    # =========================================================================

    local branch="$OVERRIDE_BRANCH"
    local is_hotfix="false"

    # Check if this is a hotfix (explicit flag or auto-detected from branch)
    if [[ "$HOTFIX_MODE" == "true" ]]; then
        is_hotfix="true"
        print_info "Hotfix mode enabled via --hotfix flag"
    elif [[ "$(is_hotfix_branch "$start_branch")" == "true" ]]; then
        is_hotfix="true"
        HOTFIX_MODE=true
        print_info "Auto-detected hotfix branch: $start_branch"
    fi

    # Auto-detect channel from current branch if not specified
    if [[ "$AUTO_DETECT_CHANNEL" == "true" ]]; then
        print_section "Auto-detecting Release Channel"

        if [[ "$is_hotfix" == "true" ]]; then
            # Hotfixes use stable channel by default
            CHANNEL=$(config_get "hotfix.channel" "stable")
            print_info "Hotfix channel: $CHANNEL"
        else
            # Resolve branch to channel using pipeline config
            CHANNEL=$(resolve_branch_to_channel "$start_branch")

            if [[ -z "$CHANNEL" ]]; then
                local enabled_channels
                enabled_channels=$(get_enabled_channels)
                print_error "Could not determine channel for branch '$start_branch'"
                echo ""
                print_info "Available channels: $enabled_channels"
                print_info "Either:"
                print_info "  1. Specify a channel: ./release.sh <$enabled_channels>"
                print_info "  2. Configure this branch in release-config.yaml pipeline"
                exit 1
            fi
            print_info "Branch '$start_branch' → Channel '$CHANNEL'"
        fi
    fi

    # Determine the target branch
    if [[ -z "$branch" ]]; then
        if [[ "$is_hotfix" == "true" ]]; then
            # Hotfix: stay on current hotfix branch, target main for merge
            branch="$start_branch"
        else
            # Normal release: use pipeline configuration
            local channel_config
            channel_config=$(config_get_channel "$CHANNEL")
            branch=$(echo "$channel_config" | $PYTHON_CMD -c "import json,sys; print(json.load(sys.stdin).get('default_branch', '$PRIMARY_BRANCH'))" 2>/dev/null) || branch="$PRIMARY_BRANCH"
        fi
    fi

    # Validate channel exists
    if [[ -z "$CHANNEL" ]]; then
        local enabled_channels
        enabled_channels=$(get_enabled_channels)
        print_error "No channel specified or detected"
        echo ""
        print_info "Available channels for this project: $enabled_channels"
        print_info "Usage: ./release.sh [OPTIONS] <CHANNEL> [branch]"
        exit 1
    fi

    # Validate channel is enabled in this project's configuration
    validate_channel_enabled "$CHANNEL"

    # Display release summary
    print_section "Release Configuration"
    print_info "Channel: $CHANNEL"
    print_info "Branch: $branch"
    print_info "Hotfix: $is_hotfix"
    if [[ "$DRY_RUN" == "true" ]]; then
        print_warning "Mode: DRY-RUN"
    fi

    # Execute release
    if [[ "$is_hotfix" == "true" ]]; then
        release_hotfix "$CHANNEL" "$branch"
    else
        release_channel "$CHANNEL" "$branch"
    fi

    # Cleanup
    cleanup_temporary_files
    restore_original_branch

    print_header "Release Complete"
    print_success "Successfully released $CHANNEL from $branch"
    if [[ "$is_hotfix" == "true" ]]; then
        print_info "Hotfix tag prefix was used"
    fi
}

# Run main
main "$@"
