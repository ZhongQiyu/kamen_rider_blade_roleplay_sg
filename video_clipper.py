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
