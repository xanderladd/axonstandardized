#!/bin/bash

directory="./"

# Function to remove x86_64 folders recursively
remove_x86_64_folders() {
    local dir="$1"

    # Iterate through the directory
    for folder in "$dir"/*; do
        if [ -d "$folder" ]; then  # Check if it's a directory
            if [ "$(basename "$folder")" = "x86_64" ]; then
                # Remove the folder
                echo "Removing folder: $folder"
                rm -rf "$folder"
            else
                # Recursively call the function for subdirectories
                remove_x86_64_folders "$folder"
            fi
        fi
    done
}

# Call the function to remove x86_64 folders
remove_x86_64_folders "$directory"