# packager.py

import os
import shutil
import subprocess

def setup_environment():
    os.environ['sensor_msgs_DIR'] = "/opt/sensor_msgs/share/sensor_msgs/cmake"
    print("Environment variables set.")

def create_folders():
    video_path = "/Users/qaz1214/Downloads/kamen-rider-blade-roleplay-sv/data/image/frames"

    for i in range(1, 50):
        folder_name = f"ep{i:02}"
        folder_path = os.path.join(video_path, folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            readme_path = os.path.join(folder_path, "README.txt")
            with open(readme_path, "w") as readme_file:
                readme_file.write(f"This is a README for episode {i:02}.")
            print(f"Created folder and README for episode {i:02}.")
        else:
            print(f"Episode {i:02} already exists, skipping...")

def upgrade_python_packages():
    print("Upgrading all outdated Python packages...")
    
    # 获取所有过时的包
    result = subprocess.run(['pip', 'list', '--outdated'], stdout=subprocess.PIPE, text=True)
    lines = result.stdout.splitlines()
    
    outdated_packages = [line.split()[0] for line in lines[2:]]
    
    # 更新每个包
    for package in outdated_packages:
        print(f"Upgrading {package}")
        subprocess.run(['pip', 'install', '--upgrade', package])
    
    print("All packages have been upgraded.")

def file_processor(input_file, output_file):
    print(f"Processing file: {input_file}")
    shutil.copy(input_file, output_file)
    print(f"File has been copied to {output_file}")

def data_analyzer(file_path):
    print(f"Analyzing data: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as file:
        line_count = sum(1 for line in file)
    print(f"The file {file_path} has {line_count} lines.")

def main():
    # Set environment variables
    setup_environment()

    # Create folders
    create_folders()

    # Upgrade all Python packages
    upgrade_python_packages()

    # Define file paths
    file_path = "/path/to/your/input_file.txt"
    cleaned_file_path = "/path/to/your/output_file.txt"

    # Use sort and uniq to remove duplicate lines and save to a new file
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = sorted(set(file.readlines()))
    with open(cleaned_file_path, 'w', encoding='utf-8') as cleaned_file:
        cleaned_file.writelines(lines)
    print(f"去重后的文件已经保存到 {cleaned_file_path}")

    # Call the function to process the file, passing file paths as parameters
    file_processor(file_path, cleaned_file_path)

    # Call the function to perform data analysis, passing the processed file path as a parameter
    data_analyzer(cleaned_file_path)

if __name__ == "__main__":
    main()
