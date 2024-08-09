# saved_utils.py

import os
import shutil
import subprocess
import time
from pathlib import Path

class CommandExecutor:
    def __init__(self, command=None):
        self.command = command

    def set_command(self, command):
        self.command = command

    def run_command(self):
        if not self.command:
            raise ValueError("No command has been set.")
        try:
            subprocess.run(self.command, check=True, shell=True)
            print(f"Command '{self.command}' executed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to execute command '{self.command}'. Error: {e}")

    def run_subprocess(self, command):
        try:
            subprocess.run(command, check=True)
            print(f"Subprocess command '{command}' executed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to execute subprocess command '{command}'. Error: {e}")

class RayClusterManager:
    def __init__(self, head_node_ip, port, nodes_array):
        self.head_node_ip = head_node_ip
        self.port = port
        self.nodes_array = nodes_array

    def start_cluster(self):
        print(f"Starting Ray head node on {self.head_node_ip}...")
        subprocess.Popen([
            'ray', 'start', '--head', 
            f'--node-ip-address={self.head_node_ip}', 
            f'--port={self.port}', '--block'
        ])

        for node in self.nodes_array[1:]:
            print(f"Starting Ray worker on {node}...")
            subprocess.Popen([
                'ray', 'start', f'--address={self.head_node_ip}:{self.port}', '--block'
            ])
        
        time.sleep(30)  # Wait for nodes to start
        print("Ray cluster started.")

    def run_training_script(self, script_path):
        print(f"Running training script: {script_path}")
        subprocess.run(['python', script_path], check=True)

    def stop_cluster(self):
        print("Stopping Ray cluster...")
        subprocess.run(['ray', 'stop'], check=True)

class PDFManager:
    def __init__(self, zotero_storage_path, output_folder_path):
        self.zotero_storage_path = zotero_storage_path
        self.output_folder_path = output_folder_path

    def copy_pdfs(self):
        print(f"Copying PDFs from {self.zotero_storage_path} to {self.output_folder_path}...")
        os.makedirs(self.output_folder_path, exist_ok=True)

        for root, _, files in os.walk(self.zotero_storage_path):
            for file in files:
                if file.endswith('.pdf'):
                    shutil.copy(os.path.join(root, file), self.output_folder_path)
        
        print("All PDFs have been copied.")

class VideoManager:
    def __init__(self, video_dir, split_size_mb):
        self.video_dir = video_dir
        self.split_size = split_size_mb * 1024 * 1024  # Convert MB to Bytes

    def split_large_videos(self):
        print(f"Splitting large videos in {self.video_dir}...")
        for file_path in Path(self.video_dir).glob("*.mov"):
            file_size = file_path.stat().st_size
            if file_size > self.split_size:
                print(f"Splitting file: {file_path}")
                subprocess.run([
                    'split', '-b', f"{self.split_size}m", str(file_path), 
                    f"{file_path.stem}_part_"
                ], check=True)
            else:
                print(f"File {file_path} is smaller than the split size threshold, skipping.")
        
        print("All file splits completed.")

def main():
    # Ray Cluster Setup
    nodes_array = ["node1", "node2"]  # Example nodes array
    head_node_ip = "192.168.0.1"  # Example head node IP
    port = 6379

    ray_manager = RayClusterManager(head_node_ip, port, nodes_array)
    ray_manager.start_cluster()
    ray_manager.run_training_script("your_training_script.py")
    ray_manager.stop_cluster()

    # PDF Copying
    pdf_manager = PDFManager(
        "/Users/qaz1214/Downloads/ref/ref-to-extract",
        "/Users/qaz1214/Downloads/ref/ref-to-extract/misc"
    )
    pdf_manager.copy_pdfs()

    # Video Splitting
    video_manager = VideoManager("data/video", 50)
    video_manager.split_large_videos()

if __name__ == "__main__":
    main()

# 

import os
import streamlit as st
from temp.OneDC_Updater.update import perform_update

def list_files(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        st.text('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            st.text('{}{}'.format(subindent, f))

# Streamlit app
st.title("项目文件结构")
project_dir = "C:\\Users\\xiaoy\\Downloads\\alumni-network"
list_files(project_dir)

# Update section
st.header("更新日志")
if st.button("执行更新"):
    perform_update()
    st.success("更新已执行")

log_file = os.path.join(project_dir, 'temp', 'OneDC_Updater', 'update.log')
with open(log_file, 'r') as file:
    log_contents = file.read()
    st.text(log_contents)


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
