#!/bin/bash


URL="http://bionlp.nlm.nih.gov/bioace-data-models.zip"
ZIP_FILE="bioace-data-models.zip"
TARGET_DIR="resources"

echo "Downloading file..."
wget -O "$ZIP_FILE" "$URL"

TEMP_DIR="temp_extraction_dir"
mkdir -p "$TEMP_DIR"

echo "Unzipping ZIP file..."
unzip -q "$ZIP_FILE" -d "$TEMP_DIR"

EXTRACTED_SUBDIR=$(find "$TEMP_DIR" -mindepth 1 -maxdepth 1 -type d | head -n 1)

if [ -d "$EXTRACTED_SUBDIR" ]; then
    echo "Renaming/moving extracted folder to $TARGET_DIR..."
    mv "$EXTRACTED_SUBDIR" "$TARGET_DIR"
else
    echo "No folder found in ZIP; copying files directly to $TARGET_DIR..."
    mkdir -p "$TARGET_DIR"
    mv "$TEMP_DIR"/* "$TARGET_DIR"/
fi

rm -rf $ZIP_FILE
rm -rf $TEMP_DIR


echo "finished!"