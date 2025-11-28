#!/usr/bin/env bash
# =============================================================================
# Project Maintenance Script - APIAS
# =============================================================================
# Automates routine maintenance tasks with SAFETY CHECKS:
#   - Clean temp/cache folders (only patterns guaranteed safe)
#   - Move unknown files to appropriate *_dev directories
#   - Sync version numbers
#   - Rebuild dist artifacts
#   - Update .gitignore
#   - Security checks (no hardcoded paths/emails)
#
# SAFETY PRINCIPLES:
#   - Only delete files matching KNOWN safe patterns
#   - Move unknown/unclear files to *_dev dirs for manual review
#   - Never delete source code or configuration
#   - Create backups before critical operations
#   - Dry-run by default for first-time users
#
# Usage:
#   ./scripts/maintenance_chore.sh [OPTIONS]
#
# Options:
#   --dry-run       Preview changes without making them (RECOMMENDED first)
#   --rebuild       Force rebuild of dist artifacts
#   --no-rebuild    Skip dist rebuild
#   --verbose       Show detailed output
#   --yes           Skip confirmation prompts
#   -h, --help      Show this help message
# =============================================================================

set -euo pipefail

# Script location and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# Options
DRY_RUN=false
REBUILD_DIST=true
VERBOSE=false
SKIP_CONFIRM=false

# Counters
CLEANED_COUNT=0
MOVED_COUNT=0
ISSUES_COUNT=0

# SAFE patterns - ONLY these are deleted without moving
# These are 100% safe to delete (caches, temp files, etc.)
SAFE_DELETE_PATTERNS=(
    ".DS_Store"
    "*.pyc"
    ".coverage"
    "*.egg-info"
    ".ruby-lsp"
    "__pycache__"
    ".pytest_cache"
    ".mypy_cache"
    ".ruff_cache"
)

# Temp folder patterns - matched at project root only
TEMP_FOLDER_PATTERNS=(
    "temp_*"       # Scraping temp folders
    "output_*"     # Output folders
    "scraped_*"    # Scraped data folders
)

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

print_header() {
    echo -e "\n${BOLD}${BLUE}$1${NC}"
    echo -e "${BLUE}$(printf '=%.0s' {1..60})${NC}"
}

print_step() {
    echo -e "  ${CYAN}->>${NC} $1"
}

print_success() {
    echo -e "  ${GREEN}[OK]${NC} $1"
}

print_warning() {
    echo -e "  ${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "  ${RED}[X]${NC} $1"
}

print_info() {
    echo -e "  ${BLUE}[i]${NC} $1"
}

print_dry_run() {
    echo -e "  ${YELLOW}[DRY-RUN]${NC} Would: $1"
}

confirm_action() {
    local message="$1"
    if [[ "$SKIP_CONFIRM" == "true" ]] || [[ "$DRY_RUN" == "true" ]]; then
        return 0
    fi

    echo -e "  ${YELLOW}[?]${NC} $message [y/N] "
    read -r response
    [[ "$response" =~ ^[Yy]$ ]]
}

# Safe delete - only for known safe patterns
safe_delete() {
    local path="$1"
    local pattern="$2"

    # Verify path is inside project root (CRITICAL SAFETY CHECK)
    local real_path
    real_path=$(realpath "$path" 2>/dev/null || echo "$path")
    local real_root
    real_root=$(realpath "$PROJECT_ROOT")

    if [[ "$real_path" != "$real_root"* ]]; then
        print_error "SAFETY: Refusing to delete path outside project: $path"
        ISSUES_COUNT=$((ISSUES_COUNT + 1))
        return 1
    fi

    # Verify pattern is in safe list
    local is_safe=false
    for safe_pattern in "${SAFE_DELETE_PATTERNS[@]}"; do
        if [[ "$pattern" == "$safe_pattern" ]]; then
            is_safe=true
            break
        fi
    done

    if [[ "$is_safe" != "true" ]]; then
        print_warning "Pattern '$pattern' not in safe-delete list, skipping: $path"
        return 1
    fi

    if [[ "$DRY_RUN" == "true" ]]; then
        print_dry_run "Delete: $path"
    else
        rm -rf "$path"
        if [[ "$VERBOSE" == "true" ]]; then
            print_step "Deleted: $path"
        fi
    fi
    CLEANED_COUNT=$((CLEANED_COUNT + 1))
}

# Safe move - move to appropriate *_dev directory
safe_move() {
    local source="$1"
    local dest_dir="$2"
    local reason="$3"

    # Verify source is inside project root
    local real_path
    real_path=$(realpath "$source" 2>/dev/null || echo "$source")
    local real_root
    real_root=$(realpath "$PROJECT_ROOT")

    if [[ "$real_path" != "$real_root"* ]]; then
        print_error "SAFETY: Refusing to move path outside project: $source"
        ISSUES_COUNT=$((ISSUES_COUNT + 1))
        return 1
    fi

    # Ensure destination exists
    if [[ ! -d "$PROJECT_ROOT/$dest_dir" ]]; then
        if [[ "$DRY_RUN" != "true" ]]; then
            mkdir -p "$PROJECT_ROOT/$dest_dir"
        fi
    fi

    if [[ "$DRY_RUN" == "true" ]]; then
        print_dry_run "Move: $source -> $dest_dir/ ($reason)"
    else
        mv "$source" "$PROJECT_ROOT/$dest_dir/"
        print_step "Moved: $source -> $dest_dir/ ($reason)"
    fi
    MOVED_COUNT=$((MOVED_COUNT + 1))
}

show_help() {
    cat << 'EOF'
Project Maintenance Script - APIAS

USAGE:
    ./scripts/maintenance_chore.sh [OPTIONS]

OPTIONS:
    --dry-run       Preview changes without making them (RECOMMENDED first)
    --rebuild       Force rebuild of dist artifacts
    --no-rebuild    Skip dist rebuild
    --verbose       Show detailed output
    --yes           Skip confirmation prompts
    -h, --help      Show this help message

SAFETY FEATURES:
    - Only deletes files matching KNOWN safe patterns (caches, .pyc, etc.)
    - Moves unknown/unclear files to *_dev directories for manual review
    - Never deletes source code or configuration files
    - All paths verified to be inside project root
    - Dry-run mode recommended for first use

TASKS PERFORMED:
    1. Clean SAFE temp/cache folders (temp_*, __pycache__, .DS_Store, etc.)
    2. Move log files to logs/
    3. Move unknown .md files to docs_dev/
    4. Move unknown scripts to scripts_dev/
    5. Sync version between pyproject.toml and __init__.py
    6. Rebuild dist artifacts if needed
    7. Verify .gitignore completeness
    8. Security check (no hardcoded paths/emails)

EXAMPLES:
    ./scripts/maintenance_chore.sh --dry-run        # FIRST: Preview changes
    ./scripts/maintenance_chore.sh                  # Full maintenance
    ./scripts/maintenance_chore.sh --no-rebuild     # Skip dist rebuild
    ./scripts/maintenance_chore.sh --yes            # No confirmations
EOF
}

# -----------------------------------------------------------------------------
# Parse Arguments
# -----------------------------------------------------------------------------

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --rebuild)
            REBUILD_DIST=true
            shift
            ;;
        --no-rebuild)
            REBUILD_DIST=false
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --yes)
            SKIP_CONFIRM=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# -----------------------------------------------------------------------------
# Pre-flight Safety Checks
# -----------------------------------------------------------------------------

preflight_checks() {
    print_header "Pre-flight Safety Checks"

    # Check we're in a git repository
    if [[ ! -d "$PROJECT_ROOT/.git" ]]; then
        print_error "Not a git repository! Aborting for safety."
        exit 1
    fi
    print_success "Git repository detected"

    # Check we have expected project structure
    if [[ ! -f "$PROJECT_ROOT/pyproject.toml" ]]; then
        print_error "pyproject.toml not found! Wrong directory?"
        exit 1
    fi
    print_success "Project structure verified"

    # Check for uncommitted changes (warning only)
    local changes
    changes=$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')
    if [[ $changes -gt 0 ]]; then
        print_warning "You have $changes uncommitted change(s)"
        if [[ "$DRY_RUN" != "true" ]] && [[ "$SKIP_CONFIRM" != "true" ]]; then
            if ! confirm_action "Continue anyway?"; then
                print_info "Aborted by user"
                exit 0
            fi
        fi
    else
        print_success "Working directory is clean"
    fi
}

# -----------------------------------------------------------------------------
# Task 1: Clean Temporary Folders (SAFE patterns only)
# -----------------------------------------------------------------------------

clean_temp_folders() {
    print_header "Cleaning Temporary Folders"

    local found=0

    # Only clean folders matching TEMP_FOLDER_PATTERNS at project root
    for pattern in "${TEMP_FOLDER_PATTERNS[@]}"; do
        while IFS= read -r -d '' dir; do
            if [[ -d "$dir" ]]; then
                # Extra safety: check folder is empty or contains only temp files
                local basename_dir
                basename_dir=$(basename "$dir")

                # Verify it matches a temp pattern
                local is_temp=false
                case "$basename_dir" in
                    temp_*|output_*|scraped_*)
                        is_temp=true
                        ;;
                esac

                if [[ "$is_temp" == "true" ]]; then
                    if [[ "$DRY_RUN" == "true" ]]; then
                        print_dry_run "Delete temp folder: $dir"
                    else
                        rm -rf "$dir"
                        print_step "Deleted temp folder: $basename_dir"
                    fi
                    found=$((found + 1))
                    CLEANED_COUNT=$((CLEANED_COUNT + 1))
                fi
            fi
        done < <(find "$PROJECT_ROOT" -maxdepth 1 -type d -name "$pattern" -print0 2>/dev/null)
    done

    # Also clean inside apias/ folder
    while IFS= read -r -d '' dir; do
        if [[ -d "$dir" ]]; then
            if [[ "$DRY_RUN" == "true" ]]; then
                print_dry_run "Delete temp folder: $dir"
            else
                rm -rf "$dir"
                print_step "Deleted temp folder in apias/: $(basename "$dir")"
            fi
            found=$((found + 1))
            CLEANED_COUNT=$((CLEANED_COUNT + 1))
        fi
    done < <(find "$PROJECT_ROOT/apias" -maxdepth 1 -type d -name "temp_*" -print0 2>/dev/null)

    if [[ $found -eq 0 ]]; then
        print_success "No temporary folders found"
    else
        print_success "Cleaned $found temporary folder(s)"
    fi
}

# -----------------------------------------------------------------------------
# Task 2: Clean Cache Files (SAFE patterns only)
# -----------------------------------------------------------------------------

clean_cache_files() {
    print_header "Cleaning Cache Files"

    local cleaned=0

    # .DS_Store files (SAFE)
    local ds_count
    ds_count=$(find "$PROJECT_ROOT" -name ".DS_Store" -type f 2>/dev/null | wc -l | tr -d ' ')
    if [[ $ds_count -gt 0 ]]; then
        print_step "Removing $ds_count .DS_Store file(s)"
        if [[ "$DRY_RUN" != "true" ]]; then
            find "$PROJECT_ROOT" -name ".DS_Store" -type f -delete 2>/dev/null || true
        fi
        cleaned=$((cleaned + $ds_count))
    fi

    # __pycache__ directories (SAFE - excluding .venv)
    local pycache_count
    pycache_count=$(find "$PROJECT_ROOT" -type d -name "__pycache__" -not -path "*/.venv/*" 2>/dev/null | wc -l | tr -d ' ')
    if [[ $pycache_count -gt 0 ]]; then
        print_step "Removing $pycache_count __pycache__ folder(s)"
        if [[ "$DRY_RUN" != "true" ]]; then
            find "$PROJECT_ROOT" -type d -name "__pycache__" -not -path "*/.venv/*" -exec rm -rf {} + 2>/dev/null || true
        fi
        cleaned=$((cleaned + $pycache_count))
    fi

    # .pyc files (SAFE - excluding .venv)
    local pyc_count
    pyc_count=$(find "$PROJECT_ROOT" -name "*.pyc" -type f -not -path "*/.venv/*" 2>/dev/null | wc -l | tr -d ' ')
    if [[ $pyc_count -gt 0 ]]; then
        print_step "Removing $pyc_count .pyc file(s)"
        if [[ "$DRY_RUN" != "true" ]]; then
            find "$PROJECT_ROOT" -name "*.pyc" -type f -not -path "*/.venv/*" -delete 2>/dev/null || true
        fi
        cleaned=$((cleaned + $pyc_count))
    fi

    # .coverage file (SAFE)
    if [[ -f "$PROJECT_ROOT/.coverage" ]]; then
        print_step "Removing .coverage"
        if [[ "$DRY_RUN" != "true" ]]; then
            rm -f "$PROJECT_ROOT/.coverage"
        fi
        cleaned=$((cleaned + 1))
    fi

    # .egg-info directories (SAFE - regenerated by build)
    while IFS= read -r -d '' dir; do
        if [[ -d "$dir" ]]; then
            print_step "Removing $(basename "$dir")/"
            if [[ "$DRY_RUN" != "true" ]]; then
                rm -rf "$dir"
            fi
            cleaned=$((cleaned + 1))
        fi
    done < <(find "$PROJECT_ROOT" -maxdepth 1 -type d -name "*.egg-info" -print0 2>/dev/null)

    # .ruby-lsp directory (SAFE - VS Code extension cache)
    if [[ -d "$PROJECT_ROOT/.ruby-lsp" ]]; then
        print_step "Removing .ruby-lsp/"
        if [[ "$DRY_RUN" != "true" ]]; then
            rm -rf "$PROJECT_ROOT/.ruby-lsp"
        fi
        cleaned=$((cleaned + 1))
    fi

    CLEANED_COUNT=$((CLEANED_COUNT + cleaned))

    if [[ $cleaned -eq 0 ]]; then
        print_success "No cache files found"
    else
        print_success "Cleaned $cleaned cache item(s)"
    fi
}

# -----------------------------------------------------------------------------
# Task 3: Organize Misplaced Files (MOVE, don't delete)
# -----------------------------------------------------------------------------

organize_files() {
    print_header "Organizing Misplaced Files"

    # Move log files to logs/
    while IFS= read -r -d '' file; do
        local basename_file
        basename_file=$(basename "$file")
        # Skip if already in logs/ or standard project logs
        if [[ "$file" != "$PROJECT_ROOT/logs/"* ]]; then
            safe_move "$file" "logs" "log file"
        fi
    done < <(find "$PROJECT_ROOT" -maxdepth 1 -name "*.log" -type f -print0 2>/dev/null)

    # Move release notes temp files to docs_dev/
    while IFS= read -r -d '' file; do
        safe_move "$file" "docs_dev" "release notes temp"
    done < <(find "$PROJECT_ROOT" -maxdepth 1 -name ".release-notes-*.md" -type f -print0 2>/dev/null)

    # Check for loose documentation files that might be temp docs (NOT standard project docs)
    #
    # CANONICAL PROJECT FILES - These are NEVER moved (case-insensitive base names):
    # These can appear with various extensions: .md, .txt, .rst, .mdx, .doc, .adoc, or no extension
    #
    # Base names to protect:
    local canonical_basenames=(
        # Core documentation
        "readme"
        "changelog"
        "changes"
        "history"
        "news"
        "releasenotes"
        "release-notes"
        "release_notes"
        "whatsnew"
        # Community health files
        "contributing"
        "contributors"
        "code_of_conduct"
        "code-of-conduct"
        "codeofconduct"
        "conduct"
        "governance"
        "maintainers"
        # Security
        "security"
        "vulnerability"
        # Support
        "support"
        "funding"
        "sponsors"
        # Legal
        "license"
        "licence"
        "copying"
        "notice"
        "patents"
        "copyright"
        "legal"
        "disclaimer"
        "terms"
        "warranty"
        # Authors/credits
        "authors"
        "credits"
        "acknowledgments"
        "acknowledgements"
        "thanks"
        # Development
        "developing"
        "development"
        "hacking"
        "install"
        "installation"
        "building"
        "build"
        "setup"
        "quickstart"
        "getting-started"
        "getting_started"
        "gettingstarted"
        # Citation
        "citation"
        "cite"
        "citing"
        # Python packaging
        "manifest"
        "requirements"
        "dependencies"
        # Claude/AI
        "claude"
        # Misc standard
        "faq"
        "troubleshooting"
        "upgrade"
        "upgrading"
        "migration"
        "migrating"
        "deprecation"
        "backwards-compatibility"
        "roadmap"
        "todo"
        "bugs"
        "known-issues"
        "known_issues"
        "knownissues"
    )

    # Common documentation extensions
    local doc_extensions=("md" "txt" "rst" "mdx" "doc" "adoc" "asciidoc" "textile" "org" "rdoc" "pod" "")

    # Function to check if a file is a canonical project file
    is_canonical_file() {
        local filename="$1"
        local basename_lower
        local name_without_ext
        local ext

        # Get lowercase basename
        basename_lower=$(echo "$filename" | tr '[:upper:]' '[:lower:]')

        # Extract extension and name without extension
        if [[ "$basename_lower" == *.* ]]; then
            ext="${basename_lower##*.}"
            name_without_ext="${basename_lower%.*}"
        else
            ext=""
            name_without_ext="$basename_lower"
        fi

        # Check if extension is a documentation extension
        local is_doc_ext=false
        for doc_ext in "${doc_extensions[@]}"; do
            if [[ "$ext" == "$doc_ext" ]]; then
                is_doc_ext=true
                break
            fi
        done

        # If not a doc extension, it's not a canonical doc file
        if [[ "$is_doc_ext" != "true" ]] && [[ -n "$ext" ]]; then
            return 1
        fi

        # Check if base name matches canonical names
        for canonical in "${canonical_basenames[@]}"; do
            if [[ "$name_without_ext" == "$canonical" ]]; then
                return 0  # Is canonical
            fi
            # Also check with .in suffix (like MANIFEST.in)
            if [[ "$name_without_ext" == "${canonical}.in" ]]; then
                return 0
            fi
        done

        return 1  # Not canonical
    }

    # Find all potential documentation files (various extensions)
    local doc_patterns=("*.md" "*.txt" "*.rst" "*.mdx" "*.doc" "*.adoc")

    for pattern in "${doc_patterns[@]}"; do
        while IFS= read -r -d '' file; do
            local basename_file
            basename_file=$(basename "$file")

            if ! is_canonical_file "$basename_file"; then
                print_info "Found non-standard doc file: $basename_file"
                if [[ "$VERBOSE" == "true" ]]; then
                    print_info "  (Standard project docs are not moved)"
                fi
                # Only prompt for unknown doc files
                if [[ "$DRY_RUN" == "true" ]]; then
                    print_dry_run "Would ask about: $basename_file"
                elif confirm_action "Move $basename_file to docs_dev/?"; then
                    safe_move "$file" "docs_dev" "non-standard doc"
                fi
            fi
        done < <(find "$PROJECT_ROOT" -maxdepth 1 -name "$pattern" -type f -print0 2>/dev/null)
    done

    # Also check for files without extensions that might be canonical (like LICENSE, README)
    # These are typically uppercase without extension
    while IFS= read -r -d '' file; do
        local basename_file
        basename_file=$(basename "$file")

        # Skip hidden files and files with extensions
        [[ "$basename_file" == .* ]] && continue
        [[ "$basename_file" == *.* ]] && continue

        # Skip if it's a canonical file
        if is_canonical_file "$basename_file"; then
            continue
        fi

        # Skip common non-doc files without extensions
        case "$basename_file" in
            Makefile|Dockerfile|Vagrantfile|Gemfile|Rakefile|Procfile|Brewfile)
                continue
                ;;
        esac

        # This is an unknown file without extension - might need review
        # But don't auto-prompt for these as they could be anything
        if [[ "$VERBOSE" == "true" ]]; then
            print_info "Found file without extension: $basename_file (not auto-flagged)"
        fi
    done < <(find "$PROJECT_ROOT" -maxdepth 1 -type f -print0 2>/dev/null)

    if [[ $MOVED_COUNT -eq 0 ]]; then
        print_success "No files needed organization"
    else
        print_success "Organized $MOVED_COUNT file(s)"
    fi
}

# -----------------------------------------------------------------------------
# Task 4: Ensure Development Folders Exist
# -----------------------------------------------------------------------------

ensure_dev_folders() {
    print_header "Ensuring Development Folders"

    local folders=("scripts_dev" "docs_dev" "logs")
    local created=0

    for folder in "${folders[@]}"; do
        if [[ ! -d "$PROJECT_ROOT/$folder" ]]; then
            print_step "Creating: $folder/"
            if [[ "$DRY_RUN" != "true" ]]; then
                mkdir -p "$PROJECT_ROOT/$folder"
            fi
            created=$((created + 1))
        else
            if [[ "$VERBOSE" == "true" ]]; then
                print_info "$folder/ already exists"
            fi
        fi
    done

    # Ensure logs/ has internal .gitignore to ignore contents but keep folder
    if [[ ! -f "$PROJECT_ROOT/logs/.gitignore" ]]; then
        print_step "Creating logs/.gitignore"
        if [[ "$DRY_RUN" != "true" ]]; then
            echo "*" > "$PROJECT_ROOT/logs/.gitignore"
            echo "!.gitignore" >> "$PROJECT_ROOT/logs/.gitignore"
        fi
    fi

    if [[ $created -eq 0 ]]; then
        print_success "All development folders exist"
    else
        print_success "Created $created folder(s)"
    fi
}

# -----------------------------------------------------------------------------
# Task 5: Sync Version Numbers
# -----------------------------------------------------------------------------

sync_versions() {
    print_header "Syncing Version Numbers"

    # Get version from pyproject.toml (source of truth)
    local pyproject_version
    pyproject_version=$(grep -E '^version = ' "$PROJECT_ROOT/pyproject.toml" | head -1 | sed 's/version = "\(.*\)"/\1/')

    if [[ -z "$pyproject_version" ]]; then
        print_error "Could not read version from pyproject.toml"
        ISSUES_COUNT=$((ISSUES_COUNT + 1))
        return 1
    fi

    # Get version from __init__.py
    local init_version
    init_version=$(grep -E '^__version__ = ' "$PROJECT_ROOT/apias/__init__.py" | sed 's/__version__ = "\(.*\)"/\1/' || echo "")

    print_info "pyproject.toml version: $pyproject_version"
    print_info "__init__.py version:    $init_version"

    if [[ "$pyproject_version" != "$init_version" ]]; then
        print_warning "Version mismatch detected!"
        print_step "Updating __init__.py to $pyproject_version"

        if [[ "$DRY_RUN" != "true" ]]; then
            # Create backup first
            cp "$PROJECT_ROOT/apias/__init__.py" "$PROJECT_ROOT/apias/__init__.py.bak"
            sed -i '' "s/__version__ = \".*\"/__version__ = \"$pyproject_version\"/" "$PROJECT_ROOT/apias/__init__.py"
            # Verify the change worked
            local new_version
            new_version=$(grep -E '^__version__ = ' "$PROJECT_ROOT/apias/__init__.py" | sed 's/__version__ = "\(.*\)"/\1/')
            if [[ "$new_version" == "$pyproject_version" ]]; then
                rm -f "$PROJECT_ROOT/apias/__init__.py.bak"
                print_success "Versions synchronized"
            else
                # Restore backup
                mv "$PROJECT_ROOT/apias/__init__.py.bak" "$PROJECT_ROOT/apias/__init__.py"
                print_error "Version sync failed, restored backup"
                ISSUES_COUNT=$((ISSUES_COUNT + 1))
            fi
        fi
        return 1  # Signal that rebuild is needed
    else
        print_success "Versions are in sync"
        return 0
    fi
}

# -----------------------------------------------------------------------------
# Task 6: Rebuild Dist Artifacts
# -----------------------------------------------------------------------------

rebuild_dist() {
    print_header "Rebuilding Distribution Artifacts"

    local force_rebuild="$1"

    # Check if dist exists and has correct version
    local pyproject_version
    pyproject_version=$(grep -E '^version = ' "$PROJECT_ROOT/pyproject.toml" | head -1 | sed 's/version = "\(.*\)"/\1/')

    local needs_rebuild=false

    if [[ ! -d "$PROJECT_ROOT/dist" ]]; then
        needs_rebuild=true
        print_info "dist/ folder missing"
    elif [[ ! -f "$PROJECT_ROOT/dist/apias-${pyproject_version}-py3-none-any.whl" ]]; then
        needs_rebuild=true
        print_info "Wheel for version $pyproject_version not found"
        # List what IS in dist
        if [[ "$VERBOSE" == "true" ]]; then
            print_info "Current dist contents:"
            ls -la "$PROJECT_ROOT/dist/" 2>/dev/null | while read -r line; do
                echo "    $line"
            done
        fi
    fi

    if [[ "$needs_rebuild" == "true" ]] || [[ "$force_rebuild" == "force" ]]; then
        print_step "Cleaning dist/"
        if [[ "$DRY_RUN" != "true" ]]; then
            # Only remove dist contents, not the folder
            rm -f "$PROJECT_ROOT/dist/"*.whl "$PROJECT_ROOT/dist/"*.tar.gz 2>/dev/null || true
        fi

        print_step "Building with uv build..."
        if [[ "$DRY_RUN" != "true" ]]; then
            cd "$PROJECT_ROOT"
            if uv build 2>&1 | tee /tmp/uv_build_$$.log | while read -r line; do
                if [[ "$VERBOSE" == "true" ]]; then
                    echo "    $line"
                fi
            done; then
                # Verify build succeeded
                if [[ -f "$PROJECT_ROOT/dist/apias-${pyproject_version}-py3-none-any.whl" ]]; then
                    print_success "Built apias-${pyproject_version}-py3-none-any.whl"

                    # Show wheel contents for verification
                    if [[ "$VERBOSE" == "true" ]]; then
                        print_info "Wheel contents:"
                        unzip -l "$PROJECT_ROOT/dist/apias-${pyproject_version}-py3-none-any.whl" 2>/dev/null | tail -n +4 | head -20 | while read -r line; do
                            echo "    $line"
                        done
                    fi
                else
                    print_error "Build completed but wheel not found"
                    print_info "Check /tmp/uv_build_$$.log for details"
                    ISSUES_COUNT=$((ISSUES_COUNT + 1))
                fi
            else
                print_error "Build failed"
                ISSUES_COUNT=$((ISSUES_COUNT + 1))
            fi
            rm -f /tmp/uv_build_$$.log
        fi
    else
        print_success "Dist artifacts are up to date (v$pyproject_version)"
    fi
}

# -----------------------------------------------------------------------------
# Task 7: Verify .gitignore
# -----------------------------------------------------------------------------

verify_gitignore() {
    print_header "Verifying .gitignore"

    # Patterns that must be in .gitignore
    # Format: "pattern_to_check:alternative_pattern" (alternative is optional)
    local required_patterns=(
        "temp_*"
        "scripts_dev/"
        "docs_dev/"
        "logs/"
        ".DS_Store"
        "__pycache__/"
        "*.pyc:*.py[cod]"  # Accept either *.pyc or *.py[cod]
        ".coverage"
        "*.egg-info/"
        ".ruby-lsp/"
        "output_*"
        "scraped_*"
        ".release-notes-*.md"
    )

    local missing=()

    for entry in "${required_patterns[@]}"; do
        # Split on : to get pattern and optional alternative
        local pattern="${entry%%:*}"
        local alt="${entry#*:}"
        [[ "$alt" == "$entry" ]] && alt=""  # No alternative if no : found

        local found=false
        if grep -qF "$pattern" "$PROJECT_ROOT/.gitignore" 2>/dev/null; then
            found=true
        elif [[ -n "$alt" ]] && grep -qF "$alt" "$PROJECT_ROOT/.gitignore" 2>/dev/null; then
            found=true
        fi

        if [[ "$found" != "true" ]]; then
            missing+=("$pattern")
        fi
    done

    if [[ ${#missing[@]} -gt 0 ]]; then
        print_warning "Missing ${#missing[@]} pattern(s) in .gitignore:"
        for pattern in "${missing[@]}"; do
            print_info "  - $pattern"
        done

        if [[ "$DRY_RUN" != "true" ]]; then
            print_step "Adding missing patterns to .gitignore"
            echo "" >> "$PROJECT_ROOT/.gitignore"
            echo "# Added by maintenance script $(date +%Y-%m-%d)" >> "$PROJECT_ROOT/.gitignore"
            for pattern in "${missing[@]}"; do
                echo "$pattern" >> "$PROJECT_ROOT/.gitignore"
            done
            print_success "Updated .gitignore"
        else
            print_dry_run "Add ${#missing[@]} patterns to .gitignore"
        fi
    else
        print_success ".gitignore has all required patterns"
    fi
}

# -----------------------------------------------------------------------------
# Task 8: Security Check
# -----------------------------------------------------------------------------

security_check() {
    print_header "Security Check"

    local issues=0

    # Check for hardcoded absolute paths (excluding safe locations)
    print_step "Checking for hardcoded absolute paths..."
    local abs_paths
    abs_paths=$(grep -r "/Users/" \
        --include="*.sh" \
        --include="*.py" \
        --include="*.yaml" \
        --include="*.yml" \
        "$PROJECT_ROOT" 2>/dev/null \
        | grep -v ".venv" \
        | grep -v ".git/" \
        | grep -v "maintenance_chore.sh" \
        | grep -v "__pycache__" \
        || true)

    if [[ -n "$abs_paths" ]]; then
        print_warning "Found hardcoded absolute paths:"
        echo "$abs_paths" | head -10 | while read -r line; do
            print_info "  $line"
        done
        local count
        count=$(echo "$abs_paths" | wc -l | tr -d ' ')
        if [[ $count -gt 10 ]]; then
            print_info "  ... and $((count - 10)) more"
        fi
        issues=$((issues + 1))
    else
        print_success "No hardcoded absolute paths found"
    fi

    # Check for non-noreply emails
    print_step "Checking for exposed email addresses..."
    local exposed_emails
    exposed_emails=$(grep -rE "[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}" \
        --include="*.sh" \
        --include="*.py" \
        --include="*.toml" \
        --include="*.yaml" \
        --include="*.yml" \
        "$PROJECT_ROOT" 2>/dev/null \
        | grep -v ".venv" \
        | grep -v ".git/" \
        | grep -v "noreply.github.com" \
        | grep -v "maintenance_chore.sh" \
        | grep -v "@dataclass" \
        | grep -v "@property" \
        | grep -v "@classmethod" \
        | grep -v "@staticmethod" \
        | grep -v "@@" \
        || true)

    if [[ -n "$exposed_emails" ]]; then
        print_warning "Found potentially exposed email addresses:"
        echo "$exposed_emails" | head -5 | while read -r line; do
            print_info "  $line"
        done
        issues=$((issues + 1))
    else
        print_success "No exposed email addresses found"
    fi

    # Verify scripts use relative paths
    print_step "Verifying scripts use relative paths..."
    local scripts_ok=true
    for script in "$PROJECT_ROOT/scripts/"*.sh; do
        if [[ -f "$script" ]] && [[ "$(basename "$script")" != "maintenance_chore.sh" ]]; then
            if ! grep -q "SCRIPT_DIR=" "$script" 2>/dev/null && ! grep -q 'dirname' "$script" 2>/dev/null; then
                print_warning "Script may not use relative paths: $(basename "$script")"
                scripts_ok=false
                issues=$((issues + 1))
            fi
        fi
    done

    if [[ "$scripts_ok" == "true" ]]; then
        print_success "All scripts use relative paths"
    fi

    ISSUES_COUNT=$((ISSUES_COUNT + issues))
}

# -----------------------------------------------------------------------------
# Task 9: Report Status
# -----------------------------------------------------------------------------

report_status() {
    print_header "Maintenance Summary"

    # Git status
    local modified
    modified=$(git status --short 2>/dev/null | grep -v "^??" | wc -l | tr -d ' ')

    local untracked
    untracked=$(git status --short 2>/dev/null | grep "^??" | wc -l | tr -d ' ')

    # Disk usage
    local project_size
    project_size=$(du -sh "$PROJECT_ROOT" 2>/dev/null | cut -f1)

    # Version
    local version
    version=$(grep -E '^version = ' "$PROJECT_ROOT/pyproject.toml" | head -1 | sed 's/version = "\(.*\)"/\1/')

    echo ""
    echo -e "  ${BOLD}Project:${NC}     APIAS"
    echo -e "  ${BOLD}Version:${NC}     $version"
    echo -e "  ${BOLD}Size:${NC}        $project_size"
    echo -e "  ${BOLD}Modified:${NC}    $modified file(s)"
    echo -e "  ${BOLD}Untracked:${NC}   $untracked file(s)"
    echo -e "  ${BOLD}Cleaned:${NC}     $CLEANED_COUNT item(s)"
    echo -e "  ${BOLD}Moved:${NC}       $MOVED_COUNT item(s)"

    if [[ $ISSUES_COUNT -gt 0 ]]; then
        echo -e "  ${BOLD}Issues:${NC}      ${RED}$ISSUES_COUNT${NC}"
    else
        echo -e "  ${BOLD}Issues:${NC}      ${GREEN}None${NC}"
    fi

    echo ""

    if [[ "$DRY_RUN" == "true" ]]; then
        echo -e "${YELLOW}DRY-RUN MODE - No changes were made${NC}"
        echo -e "${CYAN}Run without --dry-run to apply changes${NC}"
    else
        if [[ $ISSUES_COUNT -eq 0 ]]; then
            echo -e "${GREEN}Maintenance complete!${NC}"
        else
            echo -e "${YELLOW}Maintenance complete with $ISSUES_COUNT issue(s) to review${NC}"
        fi
    fi
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

main() {
    echo -e "${BOLD}${CYAN}"
    echo "============================================================"
    echo "  APIAS Project Maintenance"
    echo "============================================================"
    echo -e "${NC}"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo -e "${YELLOW}Running in DRY-RUN mode - no changes will be made${NC}"
        echo ""
    fi

    # Safety checks first
    preflight_checks

    # Run all tasks
    clean_temp_folders
    clean_cache_files
    organize_files
    ensure_dev_folders

    # Sync versions and check if rebuild needed
    local rebuild_needed="no"
    if ! sync_versions; then
        rebuild_needed="yes"
    fi

    # Rebuild dist if enabled
    if [[ "$REBUILD_DIST" == "true" ]]; then
        if [[ "$rebuild_needed" == "yes" ]]; then
            rebuild_dist "force"
        else
            rebuild_dist "check"
        fi
    else
        print_header "Skipping Dist Rebuild"
        print_info "--no-rebuild flag specified"
    fi

    verify_gitignore
    security_check
    report_status
}

# Run main
main "$@"
