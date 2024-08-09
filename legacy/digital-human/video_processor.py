# video_clipper.py

from moviepy.editor import VideoFileClip
from PIL import Image  # 导入Image类
import os

# 设置视频文件路径
video_path = 'clip.mp4'

# 加载视频文件
clip = VideoFileClip(video_path)

# 提取音频
audio = clip.audio
audio.write_audiofile('output.mp3')

# 创建存储帧的目录
if not os.path.exists('frames'):
    os.makedirs('frames')

# 提取帧图像
frame_rate = clip.fps  # 提取每一帧
for i, frame in enumerate(clip.iter_frames(fps=frame_rate)):
    frame_image = Image.fromarray(frame)
    frame_image.save(f"frames/frame_{i+1:03d}.png")

# 释放资源
clip.close()
<<<<<<< HEAD

from utils import run_subprocess

def extract_frames(video_path, frames_folder, fps=1):
    command = ['ffmpeg', '-i', video_path, '-vf', f'fps={fps}', f'{frames_folder}/frame_%04d.png']
    run_subprocess(command)

def extract_audio(video_path, audio_output):
    command = ['ffmpeg', '-i', video_path, '-q:a', '0', '-map', 'a', audio_output]
    run_subprocess(command)

from utils import run_subprocess

def extract_frames(video_path, frames_folder, fps=1):
    """
    提取视频帧。
    
    Args:
        video_path (str): 视频文件的路径。
        frames_folder (str): 保存提取帧的文件夹路径。
        fps (int): 每秒提取的帧数。
    """
    command = ['ffmpeg', '-i', video_path, '-vf', f'fps={fps}', f'{frames_folder}/frame_%04d.png']
    run_subprocess(command)

def extract_audio(video_path, audio_output):
    """
    提取视频的音轨。
    
    Args:
        video_path (str): 视频文件的路径。
        audio_output (str): 音频输出文件的路径。
    """
    command = ['ffmpeg', '-i', video_path, '-q:a', '0', '-map', 'a', audio_output]
    run_subprocess(command)

def convert_video_format(video_path, output_path, output_format='mp4'):
    """
    转换视频格式。
    
    Args:
        video_path (str): 原视频文件的路径。
        output_path (str): 转换后的视频保存路径。
        output_format (str): 输出视频的格式。
    """
    command = ['ffmpeg', '-i', video_path, f'{output_path}.{output_format}']
    run_subprocess(command)

=======
>>>>>>> origin/main

from moviepy.editor import VideoFileClip
from PIL import Image
import os
import subprocess
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
import joblib

class VideoTextClassifier:
    def __init__(self, video_path=None, vectorizer=None, model=None):
        self.video_path = video_path
        self.clip = VideoFileClip(video_path) if video_path else None
        self.vectorizer = vectorizer if vectorizer else CountVectorizer(tokenizer=TextProcessor().tokenize_japanese)
        self.model = model if model else SGDClassifier()
        self.initial_accuracy = None

    def extract_audio(self, output_audio_path):
        audio = self.clip.audio
        audio.write_audiofile(output_audio_path)

    def extract_frames(self, frames_folder, fps=1):
        if not os.path.exists(frames_folder):
            os.makedirs(frames_folder)
        frame_rate = self.clip.fps
        for i, frame in enumerate(self.clip.iter_frames(fps=frame_rate)):
            frame_image = Image.fromarray(frame)
            frame_image.save(f"{frames_folder}/frame_{i+1:03d}.png")

    def convert_video_format(self, output_path, output_format='mp4'):
        command = ['ffmpeg', '-i', self.video_path, f'{output_path}.{output_format}']
        subprocess.run(command, check=True)

    def process_video(self, frames_folder_path, audio_file_path):
        # Extract frames and audio from video
        self.extract_frames(frames_folder_path)
        self.extract_audio(audio_file_path)

        # Create dataset (假设有对应的函数)
        dataset = create_dataset(frames_folder_path, audio_file_path)

        # Process dataset text data
        processed_data = process_text_data(dataset['text_data'])

        return processed_data

    def online_learning(self, stream, threshold=0.1):
        for i, (documents, labels) in enumerate(stream):
            X_new = self.vectorizer.transform(documents)
            if i == 0:
                self.model.fit(X_new, labels)
                self.initial_accuracy = self.model.score(X_new, labels)
                print(f"Initial Batch - Accuracy: {self.initial_accuracy}")
                continue
            predictions = self.model.predict(X_new)
            new_accuracy = self.model.score(X_new, labels)
            print(f"Batch {i+1} - New Data Accuracy: {new_accuracy}")
            print(classification_report(labels, predictions))

            if new_accuracy < self.initial_accuracy - threshold:
                print("Performance decreased, updating the model.")
                self.model.partial_fit(X_new, labels)
                self.initial_accuracy = self.model.score(X_new, labels)

    def save_model(self, model_path='model.joblib', vectorizer_path='vectorizer.joblib'):
        joblib.dump(self.model, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)

    def load_model(self, model_path='model.joblib', vectorizer_path='vectorizer.joblib'):
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)

    def close(self):
        if self.clip:
            self.clip.close()

# 使用示例
if __name__ == "__main__":
    video_classifier = VideoTextClassifier(video_path='path_to_video.mp4')
    
    frames_folder = 'path_to_frames'
    audio_path = 'path_to_audio.wav'
    
    processed_data = video_classifier.process_video(frames_folder, audio_path)
    stream = (text_processor.generate_text_data(n_samples=10, random_state=i) for i in range(100))
    video_classifier.online_learning(stream)
    video_classifier.save_model()

    video_classifier.close()


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

# video_scraper.py

import requests
from bs4 import BeautifulSoup

def get_video_info(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # 获取视频标题
        title = soup.find('h1').text.strip()
        
        # 获取视频描述
        description = soup.find('div', class_='video-description').text.strip()
        
        # 获取视频时长
        duration = soup.find('span', class_='video-duration').text.strip()
        
        # 获取视频播放链接
        video_url = soup.find('video')['src']
        
        return {
            'title': title,
            'description': description,
            'duration': duration,
            'video_url': video_url
        }
    except Exception as e:
        print(f"无法从 {url} 获取视频信息: {e}")

# 示例视频播放器链接
blade_url = "https://www.yhdmhy.com/_player_x_/592724"
# video_player_url = "https://example.com/video-player"

# 获取视频信息
video_info = get_video_info(video_player_url)
if video_info:
    print("视频标题:", video_info['title'])
    print("视频描述:", video_info['description'])
    print("视频时长:", video_info['duration'])
    print("视频播放链接:", video_info['video_url'])
