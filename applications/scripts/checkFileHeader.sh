#!/bin/bash

# Define the expected header (multiple lines)
expected_header=("\"\"\""
" _____     _____     _____ _____ _____ _____    |  High-order Python FOAM"
"|  |  |___|  _  |_ _|   __|     |  _  |     |   |  Python Version: 3.10"
"|     | . |   __| | |   __|  |  |     | | | |   |  Code Version: 0.0"
"|__|__|___|__|  |_  |__|  |_____|__|__|_|_|_|   |  License: GPLv3"
"                |___|")

# Function to check if the file's header matches the expected header
check_header() {
    local file="$1"
    local line_number=0

    # Check if the file exists
    if [[ ! -f "$file" ]]; then
        echo "File not found: $file"
        return
    fi

    # Read the same number of lines from the file as in the expected header
    while IFS= read -r file_line; do
        expected_line="${expected_header[$line_number]}"
        if [[ "$file_line" != "$expected_line" ]]; then
            return 1
        fi
        line_number=$((line_number + 1))
    done < <(head -n ${#expected_header[@]} "$file")

    return 0
}

# Check if the correct number of arguments are provided
if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <file>"
    exit 1
fi

# Call the check_header function with the provided file argument
check_header "$1"

# Check the return value and set the exit value
if [[ $? -eq 0 ]]; then
    echo "File header matches the expected header."
    exit 0
else
    echo "File header does not match the expected header."
    exit 1
fi
