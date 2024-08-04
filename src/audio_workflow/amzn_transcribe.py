# amzn_transcribe.py

import boto3
import ffmpeg
import os

# S3配置
s3 = boto3.client('s3')
transcribe = boto3.client('transcribe')
bucket_name = 'your-s3-bucket'  # S3存储桶的名称

def lambda_handler(event=None, context=None):
    try:
        # 获取S3 Bucket和文件名
        bucket_name = event['Records'][0]['s3']['bucket']['name']
        file_key = event['Records'][0]['s3']['object']['key']
        
        logger.info(f"Processing file: {file_key} from bucket: {bucket_name}")
        
        # 下载文件
        local_file_path = '/tmp/' + file_key
        s3.download_file(bucket_name, file_key, local_file_path)
        
        # 拼接音频文件（假设只有一个文件，示例中无实际拼接步骤）
        concatenated_file = '/tmp/concatenated.mp3'
        ffmpeg.input(local_file_path).output(concatenated_file).run()
        
        # 上传拼接后的文件到S3
        s3.upload_file(concatenated_file, bucket_name, 'concatenated/' + 'concatenated_audio.mp3')
        
        # 调用AWS Transcribe
        transcribe.start_transcription_job(
            TranscriptionJobName='transcription-job',
            Media={'MediaFileUri': f's3://{bucket_name}/concatenated/concatenated_audio.mp3'},
            MediaFormat='mp3',
            LanguageCode='ja-JP'
        )
        
        logger.info("Audio processing and transcription initiated successfully.")
        
    except Exception as e:
        # 捕获异常并记录日志
        logger.error(f"Error processing file {file_key}: {e}")
        # 重新抛出异常以触发Step Functions的重试机制
        raise e

    return {
        'statusCode': 200,
        'body': 'Audio processed and transcribed.'
    }

# 下载并拼接音频文件
def download_and_concatenate_audio(file_keys):
    input_files = []
    for file_key in file_keys:
        s3.download_file(bucket_name, file_key, file_key)
        input_files.append(file_key)
    
    output_file = 'output_audio.mp3'
    ffmpeg.input('concat:' + '|'.join(input_files)).output(output_file).run()
    return output_file

# 拼接音频并上传到S3
def concatenate_and_upload_audio(files_to_concatenate):
    output_file = 'concatenated_audio.mp3'
    ffmpeg.input('concat:' + '|'.join(files_to_concatenate)).output('/tmp/' + output_file).run()
    s3.upload_file('/tmp/' + output_file, bucket_name, 'concatenated/' + output_file)
    return output_file

# 使用AWS Transcribe进行音频转录
def transcribe_audio(s3_uri):
    job_name = 'transcription-job'
    transcribe.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={'MediaFileUri': s3_uri},
        MediaFormat='mp3',
        LanguageCode='ja-JP'
    )
    
    # 检查转录任务的状态
    while True:
        status = transcribe.get_transcription_job(TranscriptionJobName=job_name)
        if status['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
            break
    return status

# 本地测试运行
if __name__ == "__main__":
    lambda_handler()  # 运行Lambda逻辑
