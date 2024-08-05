# amzn_transcribe.py

import boto3
import ffmpeg
import os
import logging
import time

class AWSTranscribe:
    def __init__(self, bucket_name):
        self.s3 = boto3.client('s3')
        self.transcribe = boto3.client('transcribe')
        self.bucket_name = bucket_name
        self.logger = self._setup_logger()

    def _setup_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        return logger

    def upload_folder_to_s3(self, local_folder_path, s3_folder_key):
        for root, dirs, files in os.walk(local_folder_path):
            for file_name in files:
                local_file_path = os.path.join(root, file_name)
                relative_path = os.path.relpath(local_file_path, local_folder_path)
                s3_file_key = os.path.join(s3_folder_key, relative_path)
                self.s3.upload_file(local_file_path, self.bucket_name, s3_file_key)
                self.logger.info(f"File {local_file_path} uploaded to S3 at {s3_file_key}")

    def download_and_concatenate_audio(self, file_keys):
        input_files = []
        for file_key in file_keys:
            local_file_path = '/tmp/' + os.path.basename(file_key)
            self.s3.download_file(self.bucket_name, file_key, local_file_path)
            input_files.append(local_file_path)
        
        output_file = '/tmp/output_audio.mp3'
        ffmpeg.input('concat:' + '|'.join(input_files)).output(output_file).run()
        return output_file

    def concatenate_and_upload_audio(self, files_to_concatenate):
        output_file = self.download_and_concatenate_audio(files_to_concatenate)
        s3_output_key = 'concatenated/concatenated_audio.mp3'
        self.s3.upload_file(output_file, self.bucket_name, s3_output_key)
        self.logger.info(f"Concatenated audio uploaded to S3 at {s3_output_key}")
        return f's3://{self.bucket_name}/{s3_output_key}'

    def transcribe_audio(self, s3_uri):
        job_name = f'transcription-job-{int(time.time())}'
        self.transcribe.start_transcription_job(
            TranscriptionJobName=job_name,
            Media={'MediaFileUri': s3_uri},
            MediaFormat='mp3',
            LanguageCode='ja-JP'
        )
        
        self.logger.info(f"Transcription job {job_name} started for {s3_uri}")
        
        while True:
            status = self.transcribe.get_transcription_job(TranscriptionJobName=job_name)
            job_status = status['TranscriptionJob']['TranscriptionJobStatus']
            if job_status in ['COMPLETED', 'FAILED']:
                self.logger.info(f"Transcription job {job_name} completed with status {job_status}")
                break
            self.logger.info(f"Transcription job {job_name} status: {job_status}")
            time.sleep(15)
        
        return status

    def process_event(self, event):
        try:
            # 从事件中提取文件信息
            bucket_name = event['Records'][0]['s3']['bucket']['name']
            file_key = event['Records'][0]['s3']['object']['key']
            
            self.logger.info(f"Processing file: {file_key} from bucket: {bucket_name}")
            
            # 如果有多个文件需要拼接，可以在这里定义 file_keys 列表
            file_keys = [file_key]  # 示例中只处理单个文件，您可以扩展为多个文件

            # 拼接并上传音频
            s3_uri = self.concatenate_and_upload_audio(file_keys)

            # 进行音频转录
            transcription_status = self.transcribe_audio(s3_uri)
            
            self.logger.info("Audio processing and transcription completed successfully.")
            
        except Exception as e:
            self.logger.error(f"Error processing file {file_key}: {e}")
            raise e

        return {
            'statusCode': 200,
            'body': 'Audio processed and transcribed.'
        }

    def start_aws_workflow(self, local_folder, s3_folder, event):
        # 上传本地文件夹到S3
        self.upload_folder_to_s3(local_folder, s3_folder)
        
        # 处理S3事件进行音频转录
        return self.process_event(event)


if __name__ == "__main__":
    bucket_name = 'kamen-rider-blade-roleplay-sv'
    processor = AWSTranscribe(bucket_name)
    
    # 测试上传整个文件夹并处理事件
    local_folder = 'C:\\Users\\xiaoy\\Documents\\backup\\audio\\krb'
    s3_folder = 'audio_files'
    
    # 模拟事件
    mock_event = {
        'Records': [
            {
                's3': {
                    'bucket': {'name': bucket_name},
                    'object': {'key': 'example_audio_file.mp3'}
                }
            }
        ]
    }
    
    # 调用AWS工作流
    processor.start_aws_workflow(local_folder, s3_folder, mock_event)
