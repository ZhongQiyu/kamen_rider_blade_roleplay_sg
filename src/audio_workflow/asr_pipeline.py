# asr_pipeline.py

import os
import json
import re
from pydub import AudioSegment

class ASRPipeline:
    def __init__(self, config_file="model_config.json"):
        self.config_file = config_file

    # 音频转换功能
    def convert_m4a_to_wav(self, input_folder, output_folder):
        # 检查输出文件夹是否存在，如果不存在则创建
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # 遍历输入文件夹中的所有文件
        for filename in os.listdir(input_folder):
            if filename.endswith(".m4a"):
                # 构建完整的文件路径
                input_file_path = os.path.join(input_folder, filename)
                output_file_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.wav")
                
                # 加载音频文件并转换为wav
                audio = AudioSegment.from_file(input_file_path, format="m4a")
                audio.export(output_file_path, format="wav")
                print(f"Converted {filename} to {output_file_path}")

    # 加载模型配置
    def load_model_config(self, model_name):
        with open(self.config_file, "r") as f:
            config = json.load(f)
        return config["models"].get(model_name)

    # ASR 处理功能
    def start_asr_workflow(self, choice):
        if choice == '1':
            self.start_aws_workflow()
            model_config = self.load_model_config("BERT")
            print("Loaded BERT model config for AWS workflow:", model_config)
        elif choice == '2':
            self.start_iflytek_workflow()
            model_config = self.load_model_config("BERT-CHINESE")
            print("Loaded BERT-Chinese model config for iFlytek workflow:", model_config)
        else:
            print("无效选择")

    def start_aws_workflow(self):
        # 这是AWS处理的占位符逻辑
        print("AWS workflow started.")

    def start_iflytek_workflow(self):
        # 这是讯飞处理的占位符逻辑
        print("iFlytek workflow started.")

    # 处理对话文件
    def handle_dialog_from_file(self, file_path):
        data = []
        current_speaker = None
        current_time = None
        dialog = []

        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                speaker_match = re.match(r'^说话人(\d+) (\d{2}:\d{2})', line)
                if speaker_match:
                    if current_speaker is not None:
                        # 当遇到新的说话人时，保存之前的对话
                        data.append({
                            'speaker': current_speaker,
                            'time': current_time,
                            'text': ' '.join(dialog).strip()
                        })
                        dialog = []  # 重置对话列表
                    current_speaker, current_time = speaker_match.groups()
                else:
                    # 收集对话行
                    dialog.append(line.strip())

        # 确保最后一个对话被添加
        if current_speaker and dialog:
            data.append({
                'speaker': current_speaker,
                'time': current_time,
                'text': ' '.join(dialog).strip()
            })

        return data

    # 加载数据
    def load_data(self, data_file):
        # 加载数据的逻辑，返回加载后的数据
        with open(data_file, 'r', encoding='utf-8') as file:
            return json.load(file)

    # 解析训练数据
    def parse_train_data(self, data):
        # 解析训练数据的逻辑
        pass

    # 解析 ASR 数据
    def parse_asr_data(self, data):
        # 解析ASR数据的逻辑
        pass

if __name__ == "__main__":
    # 实例化ProcessingPipeline类
    pipeline = ProcessingPipeline()

    # 1. 转换音频文件
    input_folder = '/path/to/your/m4a/files'  # 替换为你的m4a文件夹路径
    output_folder = '/path/to/your/output/wav/files'  # 替换为你想保存wav文件的文件夹路径
    pipeline.convert_m4a_to_wav(input_folder, output_folder)

    # 2. 选择并执行ASR处理流程
    choice = input("选择处理流程 (1: AWS, 2: 讯飞): ")
    pipeline.start_asr_workflow(choice)

    # 3. 处理对话文件
    file_path = '/mnt/data/1.txt'
    all_dialogs = pipeline.handle_dialog_from_file(file_path)
    print(all_dialogs[:5])  # 打印前5条对话以检查输出

    # 4. 加载和解析数据（根据具体需求）
    # 示例: data = pipeline.load_data("data_file.json")
    # 示例: parsed_data = pipeline.parse_train_data(data)
