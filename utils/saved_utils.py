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
