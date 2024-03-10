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
