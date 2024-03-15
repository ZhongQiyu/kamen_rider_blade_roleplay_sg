import json

json_path = "transcripts_output_transcript_68c650f0-0000-26af-b8d3-2405887b1c1c.json"

# 加载JSON数据
with open(json_path, 'r') as f:
    data = json.load(f)

# 从JSON数据中提取信息
for result in data['results']:
    for alternative in result['alternatives']:
        transcript = alternative['transcript']
        confidence = alternative['confidence']
        print(f"Transcript: {transcript}")
        print(f"Confidence: {confidence}")

        # 如果有词时间偏移（word timing offsets）
        if 'words' in alternative:
            for word_info in alternative['words']:
                word = word_info['word']
                start_time = word_info['startTime']
                end_time = word_info['endTime']
                print(f"Word: {word}, start time: {start_time}, end time: {end_time}")
<<<<<<< HEAD

# 伪代码，需要实际的图像和音频处理代码
def create_dataset(frames_folder, audio_file):
    # process frames and audio
    # extract features
    # save to dataset
    pass

import subprocess
import os
import json
from fugashi import Tagger

# 功能：运行外部命令
def run_command(command):
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# 功能：提取帧
def extract_frames(video_path, frames_folder, fps=1):
    if not os.path.exists(frames_folder):
        os.makedirs(frames_folder)
    command = ['ffmpeg', '-i', video_path, '-vf', f'fps={fps}', f'{frames_folder}/frame_%04d.png']
    run_command(command)

# 功能：提取音频
def extract_audio(video_path, audio_output):
    command = ['ffmpeg', '-i', video_path, '-q:a', '0', '-map', 'a', audio_output]
    run_command(command)

# 功能：日文分词
def tokenize_japanese(text):
    tagger = Tagger('-Owakati')
    return tagger.parse(text).strip().split()

# 功能：加载内部或外部数据集
def load_additional_dataset(dataset_path):
    with open(dataset_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# 功能：创建并保存数据集
def create_dataset(frames_folder, audio_file, text_data, additional_data, dataset_output):
    tokenized_text_data = [tokenize_japanese(sentence) for sentence in text_data]
    combined_data = {
        'frames': [os.path.join(frames_folder, frame) for frame in os.listdir(frames_folder) if frame.endswith('.png')],
        'audio': audio_file,
        'text': tokenized_text_data,
        'additional_data': additional_data  # 加载的额外数据集
    }
    
    with open(dataset_output, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=4)

# 主逻辑
if __name__ == "__main__":
    video_file = 'path_to_your_video.mp4'
    frames_folder = 'path_to_frames_directory'
    audio_output = 'path_to_audio_output.wav'
    additional_dataset_path = 'internal_lm2.json'
    dataset_output = 'combined_dataset.json'
    
    # 这里我们提取现有文本数据
    text_data = ["こんにちは", "元気ですか", "さようなら"]
    additional_data = load_additional_dataset(additional_dataset_path)
    
    # 提取帧和音频
    extract_frames(video_file, frames_folder)
    extract_audio(video_file, audio_output)
    
    # 创建并保存数据集
    create_dataset(frames_folder, audio_output, text_data, additional_data, dataset_output)

=======
>>>>>>> origin/main
