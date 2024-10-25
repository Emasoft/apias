#!/bin/bash

# Backup the current branch name
current_branch=$(git rev-parse --abbrev-ref HEAD)

# Create a temporary branch
git checkout --orphan temp_branch

# Add all files
git add -A

# Create a new commit with the correct email
GIT_COMMITTER_EMAIL="713559+Emasoft@users.noreply.github.com" \
GIT_AUTHOR_EMAIL="713559+Emasoft@users.noreply.github.com" \
GIT_COMMITTER_NAME="Emanuele Sabetta" \
GIT_AUTHOR_NAME="Emanuele Sabetta" \
git commit -m "Initial commit with clean history"

# Delete the old branch
git branch -D $current_branch

# Rename the temp branch to the original branch name
git branch -m $current_branch

# Create pre-commit hook to enforce email
cat > .git/hooks/pre-commit << 'HOOK'
#!/bin/bash

# Check if commit email is correct
if git config user.email | grep -q -v "713559+Emasoft@users.noreply.github.com"; then
    echo "Error: Incorrect email configuration"
    echo "Please set your email to 713559+Emasoft@users.noreply.github.com"
    exit 1
fi

# Check if any file contains the private email
if git diff --cached | grep -q "713559+Emasoft@users.noreply.github.com"; then
    echo "Error: Private email found in changes"
    exit 1
fi
HOOK

# Make the hook executable
chmod +x .git/hooks/pre-commit

# Set local git config
git config --local user.email "713559+Emasoft@users.noreply.github.com"
git config --local user.name "Emanuele Sabetta"

echo "Git history has been cleaned and pre-commit hook installed"
