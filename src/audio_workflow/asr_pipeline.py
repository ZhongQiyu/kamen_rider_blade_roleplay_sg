# asr_pipeline.py

import os
import re
import json
import boto3
import shutil
import logging
from pydub import AudioSegment

# 手动指定FFmpeg路径
AudioSegment.ffmpeg = "C:/ffmpeg/bin/ffmpeg.exe"

# 配置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        paginator = self.s3_client.get_paginator('list_objects_v2')
        try:
            for page in paginator.paginate(Bucket=self.s3_bucket, Prefix=s3_folder):
                for obj in page.get('Contents', []):
                    s3_key = obj['Key']
                    file_name = os.path.basename(s3_key)
                    local_path = os.path.join(local_folder, file_name)

                    # 创建本地文件夹（如果不存在）
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)

                    # 下载文件
                    self.s3_client.download_file(self.s3_bucket, s3_key, local_path)
                    logger.info(f"Downloaded {s3_key} from S3 to {local_path}")
        except boto3.exceptions.S3UploadFailedError as e:
            logger.error(f"Failed to download from S3: {str(e)}")
            raise e
    
    # 音频转换功能
    def convert_m4a_to_wav(self, input_folder, output_folder):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for filename in os.listdir(input_folder):
            if filename.endswith(".m4a"):
                input_file_path = os.path.join(input_folder, filename)
                output_file_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.wav")
                
                # 检查目标文件是否已经存在
                if os.path.exists(output_file_path):
                    logger.info(f"File {output_file_path} already exists. Skipping conversion.")
                    continue

                try:
                    audio = AudioSegment.from_file(input_file_path, format="m4a")
                    audio.export(output_file_path, format="wav")
                    logger.info(f"Converted {filename} to {output_file_path}")
                except Exception as e:
                    logger.error(f"Failed to convert {filename}: {str(e)}")
                    raise e
    
    # 加载模型配置
    def load_model_config(self, model_name):
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"Config file {self.config_file} not found.")
        with open(self.config_file, "r") as f:
            try:
                config = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Error decoding JSON from {self.config_file}: {str(e)}")
        return config["models"].get(model_name)

    # ASR 处理功能
    def start_asr_workflow(self, choice):
        if choice == '1':
            self.start_aws_workflow()
            model_config = self.load_model_config("BERT")
            logger.info("Loaded BERT model config for AWS workflow:", model_config)
        elif choice == '2':
            self.start_iflytek_workflow()
            model_config = self.load_model_config("BERT-CHINESE")
            logger.info("Loaded BERT-Chinese model config for iFlytek workflow:", model_config)
        else:
            logger.error("无效选择")

    def start_aws_workflow(self):
        logger.info("AWS workflow started.")

    def start_iflytek_workflow(self):
        logger.info("iFlytek workflow started.")

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
        pass  # Placeholder for future implementation

    # 解析 ASR 数据
    def parse_asr_data(self, data):
        pass  # Placeholder for future implementation

if __name__ == "__main__":
    # 从环境变量中获取AWS访问密钥
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')

    if not aws_access_key_id or not aws_secret_access_key:
        raise ValueError("AWS access key ID and secret access key must be set as environment variables.")

    # 初始化ASR Pipeline
    pipeline = ASRPipeline(
        s3_bucket='kamen-rider-blade-roleplay-sv',
        aws_access_key_id=aws_access_key_id,  # 从环境变量中获取
        aws_secret_access_key=aws_secret_access_key,  # 从环境变量中获取
        region_name='us-east-2'  # 替换为你选择的AWS区域
    )

    # 检查FFmpeg是否安装并可用
    if not shutil.which("ffmpeg"):
        raise EnvironmentError("FFmpeg is not installed or not found in the system path.")

    # 从S3文件夹下载所有文件
    s3_folder = 'clips/'  # 注意这里是文件夹路径
    # local_folder = '../data/clips/m4a'
    local_folder = 'C:\\Users\\xiaoy\\Downloads\\legacy\\audio\\krb'
    # output_folder = '../data/clips/wav'
    # pipeline.download_folder_from_s3(s3_folder, local_folder)

    # 转换音频文件
    input_folder = local_folder
    output_folder = 'C:\\Users\\xiaoy\\Documents\\kamen-rider-blade\\data\\audio\\wav'
    pipeline.convert_m4a_to_wav(input_folder, output_folder)
    
    # 选择并执行ASR处理流程
    choice = input("选择处理流程 (1: AWS, 2: 讯飞): ")
    pipeline.start_asr_workflow(choice)

    """
    # 处理对话文件
    file_path = '/mnt/data/1.txt'
    all_dialogs = pipeline.handle_dialog_from_file(file_path)
    logger.info(all_dialogs[:5])  # 打印前5条对话以检查输出
    """
