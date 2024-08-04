# amzn_transcribe.py

import boto3
import ffmpeg
import os
import logging
import time

# 日志配置
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# S3和Transcribe客户端
s3 = boto3.client('s3')
transcribe = boto3.client('transcribe')
bucket_name = 'kamen-rider-blade-roleplay-sv'  # S3存储桶的名称

# 上传文件夹到S3
def upload_folder_to_s3(local_folder_path, bucket_name, s3_folder_key):
    for root, dirs, files in os.walk(local_folder_path):
        for file_name in files:
            local_file_path = os.path.join(root, file_name)
            relative_path = os.path.relpath(local_file_path, local_folder_path)
            s3_file_key = os.path.join(s3_folder_key, relative_path)
            s3.upload_file(local_file_path, bucket_name, s3_file_key)
            logger.info(f"File {local_file_path} uploaded to S3 at {s3_file_key}")

# 下载并拼接音频文件
def download_and_concatenate_audio(file_keys):
    input_files = []
    for file_key in file_keys:
        local_file_path = '/tmp/' + os.path.basename(file_key)
        s3.download_file(bucket_name, file_key, local_file_path)
        input_files.append(local_file_path)
    
    output_file = '/tmp/output_audio.mp3'
    ffmpeg.input('concat:' + '|'.join(input_files)).output(output_file).run()
    return output_file

# 拼接音频并上传到S3
def concatenate_and_upload_audio(files_to_concatenate):
    output_file = download_and_concatenate_audio(files_to_concatenate)
    s3_output_key = 'concatenated/concatenated_audio.mp3'
    s3.upload_file(output_file, bucket_name, s3_output_key)
    logger.info(f"Concatenated audio uploaded to S3 at {s3_output_key}")
    return f's3://{bucket_name}/{s3_output_key}'

# 使用AWS Transcribe进行音频转录
def transcribe_audio(s3_uri):
    job_name = f'transcription-job-{int(time.time())}'
    transcribe.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={'MediaFileUri': s3_uri},
        MediaFormat='mp3',
        LanguageCode='ja-JP'
    )
    
    logger.info(f"Transcription job {job_name} started for {s3_uri}")
    
    while True:
        status = transcribe.get_transcription_job(TranscriptionJobName=job_name)
        job_status = status['TranscriptionJob']['TranscriptionJobStatus']
        if job_status in ['COMPLETED', 'FAILED']:
            logger.info(f"Transcription job {job_name} completed with status {job_status}")
            break
        logger.info(f"Transcription job {job_name} status: {job_status}")
        time.sleep(15)
    
    return status

# Lambda函数处理流程
def lambda_handler(event=None, context=None):
    try:
        # 从事件中提取文件信息
        bucket_name = event['Records'][0]['s3']['bucket']['name']
        file_key = event['Records'][0]['s3']['object']['key']
        
        logger.info(f"Processing file: {file_key} from bucket: {bucket_name}")
        
        # 如果有多个文件需要拼接，可以在这里定义 file_keys 列表
        file_keys = [file_key]  # 示例中只处理单个文件，您可以扩展为多个文件

        # 拼接并上传音频
        s3_uri = concatenate_and_upload_audio(file_keys)

        # 进行音频转录
        transcription_status = transcribe_audio(s3_uri)
        
        logger.info("Audio processing and transcription completed successfully.")
        
    except Exception as e:
        logger.error(f"Error processing file {file_key}: {e}")
        raise e

    return {
        'statusCode': 200,
        'body': 'Audio processed and transcribed.'
    }

# 本地测试运行
if __name__ == "__main__":
    # 测试上传整个文件夹
    local_folder = 'C:\\Users\\xiaoy\\Documents\\backup\\audio\\krb'
    s3_folder = 'audio_files'
    upload_folder_to_s3(local_folder, bucket_name, s3_folder)
    
    # 测试Lambda处理流程
    lambda_handler()
