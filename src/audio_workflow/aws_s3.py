import boto3
import json

def lambda_handler(event, context):
    s3 = boto3.client('s3')

    # 假设每个上传的文件都会触发此Lambda函数
    for record in event['Records']:
        bucket_name = record['s3']['bucket']['name']
        object_key = record['s3']['object']['key']
        print(f"A new file {object_key} was uploaded in bucket {bucket_name}")

        # 在这里添加代码，例如更新GitHub仓库或其他处理逻辑
        # 这可能需要额外的 API 调用或集成其他服务

    return {
        'statusCode': 200,
        'body': json.dumps('Process completed successfully!')
    }
