#!/bin/bash

# Search for email in all files
find . -type f -not -path "./.git/*" -exec grep -l "713559+Emasoft@users.noreply.github.com" {} \;

# Replace email in found files
find . -type f -not -path "./.git/*" -exec sed -i 's/713559+Emasoft@users.noreply.github.com/713559+Emasoft@users.noreply.github.com/g' {} \;

echo "Files cleaned of private email references"
