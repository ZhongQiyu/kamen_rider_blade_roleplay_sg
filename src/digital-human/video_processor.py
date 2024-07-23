# video_processor.py

import os
import json
import subprocess
from fugashi import Tagger

class VideoProcessor:
    def __init__(self, frames_folder, audio_output):
        self.frames_folder = frames_folder
        self.audio_output = audio_output

    def run_command(self, command):
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def extract_frames(self, video_path, fps=1):
        if not os.path.exists(self.frames_folder):
            os.makedirs(self.frames_folder)
        command = ['ffmpeg', '-i', video_path, '-vf', f'fps={fps}', f'{self.frames_folder}/frame_%04d.png']
        self.run_command(command)

    def extract_audio(self, video_path):
        command = ['ffmpeg', '-i', video_path, '-q:a', '0', '-map', 'a', self.audio_output]
        self.run_command(command)

class TextProcessor:
    def __init__(self):
        self.tagger = Tagger('-Owakati')

    def tokenize_japanese(self, text):
        return self.tagger.parse(text).strip().split()

    def load_additional_dataset(self, dataset_path):
        with open(dataset_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def create_dataset(self, frames_folder, audio_file, text_data, additional_data, dataset_output):
        tokenized_text_data = [self.tokenize_japanese(sentence) for sentence in text_data]
        combined_data = {
            'frames': [os.path.join(frames_folder, frame) for frame in os.listdir(frames_folder) if frame.endswith('.png')],
            'audio': audio_file,
            'text': tokenized_text_data,
            'additional_data': additional_data
        }
        with open(dataset_output, 'w', encoding='utf-8') as f:
            json.dump(combined_data, f, ensure_ascii=False, indent=4)
