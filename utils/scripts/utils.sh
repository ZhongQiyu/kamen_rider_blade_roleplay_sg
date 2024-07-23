#!/bin/bash

# Function to set environment variables
function setup_environment {
    export sensor_msgs_DIR="/opt/sensor_msgs/share/sensor_msgs/cmake"
    echo "Environment variables set."
}

# Function to create folders
function create_folders {
    local video_path="/Users/qaz1214/Downloads/kamen-rider-blade-roleplay-sv/data/image/frames"

    # loop from 01 to 49
    for i in $(seq -w 1 49); do
        # if the folder does not exist, then create the folder
        if [ ! -d "${video_path}/ep${i}" ]; then
            mkdir "${video_path}/ep${i}"
            echo "This is a README for episode ${i}." > "${video_path}/ep${i}/README.txt"
            echo "Created folder and README for episode ${i}."
        else
            echo "Episode ${i} already exists, skipping..."
        fi
    done
}

# Function to upgrade all Python packages
function upgrade_python_packages {
    echo "Upgrading all outdated Python packages..."
    
    # Getting all outdated packages, extracting the package names, and upgrading them
    pip list --outdated | grep -v "Package" | awk '{print $1}' | xargs -n 1 pip install --upgrade
    echo "All packages have been upgraded."

    # 获取所有过时的包并提取包名
    outdated_packages=$(pip list --outdated | awk 'NR>2 {print $1}')

    # 更新每个包
    for package in $outdated_packages
    do
        echo "Upgrading $package"
        pip install --upgrade $package
    done
}

# Function to process files
function FileProcessor {
    echo "Processing file: $1"
    # Logic to copy the file goes here
    cp "$1" "$2"
    echo "File has been copied to $2"
}

# Function for data analysis
function DataAnalyzer {
    echo "Analyzing data: $1"
    # Logic to count the number of lines in the file goes here
    local line_count=$(wc -l < "$1")
    echo "The file $1 has $line_count lines."
}

# "Main method", which calls other "methods"
function main {
    # Set environment variables
    setup_environment

    # Create folders
    create_folders

    # Upgrade all Python packages
    upgrade_python_packages

    # Define file paths
    local file_path="/path/to/your/input_file.txt"
    local cleaned_file_path="/path/to/your/output_file.txt"

    # Use sort and uniq to remove duplicate lines and save to a new file
    sort "$file_path" | uniq > "$cleaned_file_path"
    echo "去重后的文件已经保存到 $cleaned_file_path"

    # Call the function to process the file, passing file paths as parameters
    FileProcessor "$file_path" "$cleaned_file_path"

    # Call the function to perform data analysis, passing the processed file path as a parameter
    DataAnalyzer "$cleaned_file_path"
}

# Script entry point
main
