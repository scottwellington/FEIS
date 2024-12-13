#!/bin/bash

# Define the base directory
BASE_DIR="data_eeg"

# Loop through each folder in the base directory
for folder in "$BASE_DIR"/*; do
    # Check if it is a directory
    if [ -d "$folder" ]; then
        echo "Processing folder: $folder"
        # Loop through each .zip file in the folder
        for zip_file in "$folder"/*.zip; do
            # Check if the .zip file exists (to avoid errors when no .zip files are present)
            if [ -f "$zip_file" ]; then
                echo "Unzipping $zip_file"
                # Unzip the file into the same folder
                unzip -o "$zip_file" -d "$folder"
                # Check if the unzip was successful before deleting
                if [ $? -eq 0 ]; then
                    echo "Deleting $zip_file"
                    rm "$zip_file"
                else
                    echo "Failed to unzip $zip_file. Skipping deletion."
                fi
            fi
        done
    fi

done

echo "Processing complete."