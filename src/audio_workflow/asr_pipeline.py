# asr_pipeline.py

import os
import json
import re
import boto3
from pydub import AudioSegment

class ASRPipeline:
    def __init__(self, config_file="model_config.json", s3_bucket=None, aws_access_key_id=None, aws_secret_access_key=None, region_name=None):
        self.config_file = config_file
        self.s3_bucket = s3_bucket
        if s3_bucket:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                region_name=region_name
            )

    # 从S3下载文件夹中的所有文件
    def download_folder_from_s3(self, s3_folder, local_folder):
        # 列出S3文件夹中的所有文件
        paginator = self.s3_client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=self.s3_bucket, Prefix=s3_folder):
            for obj in page.get('Contents', []):
                s3_key = obj['Key']
                file_name = os.path.basename(s3_key)
                local_path = os.path.join(local_folder, file_name)

                # 创建本地文件夹（如果不存在）
                os.makedirs(os.path.dirname(local_path), exist_ok=True)

                # 下载文件
                self.s3_client.download_file(self.s3_bucket, s3_key, local_path)
                print(f"Downloaded {s3_key} from S3 to {local_path}")

    # 音频转换功能
    def convert_m4a_to_wav(self, input_folder, output_folder):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for filename in os.listdir(input_folder):
            if filename.endswith(".m4a"):
                input_file_path = os.path.join(input_folder, filename)
                output_file_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.wav")
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
        print("AWS workflow started.")

    def start_iflytek_workflow(self):
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
                        data.append({
                            'speaker': current_speaker,
                            'time': current_time,
                            'text': ' '.join(dialog).strip()
                        })
                        dialog = []
                    current_speaker, current_time = speaker_match.groups()
                else:
                    dialog.append(line.strip())

        if current_speaker and dialog:
            data.append({
                'speaker': current_speaker,
                'time': current_time,
                'text': ' '.join(dialog).strip()
            })

        return data

    # 加载数据
    def load_data(self, data_file):
        with open(data_file, 'r', encoding='utf-8') as file:
            return json.load(file)

    # 解析训练数据
    def parse_train_data(self, data):
        pass

    # 解析 ASR 数据
    def parse_asr_data(self, data):
        pass

if __name__ == "__main__":
    pipeline = ASRPipeline(
        s3_bucket='your-s3-bucket-name',
        aws_access_key_id='your-access-key-id',
        aws_secret_access_key='your-secret-access-key',
        region_name='your-region'
    )

    # 从S3文件夹下载所有文件
    s3_folder = 'path/in/s3/to/your/folder/'  # 注意这里是文件夹路径
    local_folder = '/path/to/local/folder'
    pipeline.download_folder_from_s3(s3_folder, local_folder)

    # 转换音频文件
    input_folder = local_folder
    output_folder = '/path/to/local/output/folder'
    pipeline.convert_m4a_to_wav(input_folder, output_folder)

    # 选择并执行ASR处理流程
    choice = input("选择处理流程 (1: AWS, 2: 讯飞): ")
    pipeline.start_asr_workflow(choice)

    # 处理对话文件
    file_path = '/mnt/data/1.txt'
    all_dialogs = pipeline.handle_dialog_from_file(file_path)
    print(all_dialogs[:5])  # 打印前5条对话以检查输出
