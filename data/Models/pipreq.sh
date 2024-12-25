#!/bin/bash

# File to log failed installations
FAILED_FILE="failed_installs.txt"

# Clear previous failed log file if it exists
> "$FAILED_FILE"

# Read the requirements.txt line by line
while read -r package; do
    if pip install "$package"; then
        echo "Successfully installed $package"
    else
        echo "Failed to install $package, skipping..."
        echo "$package" >> "$FAILED_FILE"
    fi
done < requirements.txt

echo "Failed installs logged to $FAILED_FILE"
