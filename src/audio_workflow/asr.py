# asr.py

from aws_workflow.transcribe_workflow import start_aws_workflow
from iflytek_workflow.process_workflow import start_iflytek_workflow

def main():
    # 根据需求选择调用AWS或讯飞的流程
    choice = input("选择处理流程 (1: AWS, 2: 讯飞): ")
    if choice == '1':
        start_aws_workflow()
    elif choice == '2':
        start_iflytek_workflow()
    else:
        print("无效选择")

if __name__ == "__main__":
    main()

import boto3
import ffmpeg
import os

s3 = boto3.client('s3')
transcribe = boto3.client('transcribe')

def lambda_handler(event, context):
    bucket_name = event['Records'][0]['s3']['bucket']['name']
    file_key = event['Records'][0]['s3']['object']['key']
    
    # 下载文件
    s3.download_file(bucket_name, file_key, '/tmp/' + file_key)
    
    # 拼接逻辑 (假设有多文件)
    files_to_concatenate = ['/tmp/' + file_key]  # 可以根据需求获取更多文件
    concatenated_file = '/tmp/concatenated.mp3'
    ffmpeg.input('concat:' + '|'.join(files_to_concatenate)).output(concatenated_file).run()
    
    # 上传拼接后的文件
    s3.upload_file(concatenated_file, bucket_name, 'concatenated/' + 'concatenated_audio.mp3')
    
    # 调用Transcribe
    transcribe.start_transcription_job(
        TranscriptionJobName='transcription-job',
        Media={'MediaFileUri': f's3://{bucket_name}/concatenated/concatenated_audio.mp3'},
        MediaFormat='mp3',
        LanguageCode='ja-JP'
    )
    
    return {
        'statusCode': 200,
        'body': 'Audio processed and transcribed.'
    }


from xfyun_sdk import Inference

class RequestApi(object):
    def __init__(self, appid, secret_key, upload_file_path):
        self.appid = appid
        self.secret_key = secret_key
        self.upload_file_path = upload_file_path
        self.inference = Inference(appid=self.appid, secret_key=self.secret_key)  # 初始化推理对象

    def run_inference(self):
        # 这里是调用推理的方法
        try:
            result = self.inference.transcribe(self.upload_file_path)
            print("推理结果:", result)
            return result
        except Exception as e:
            print("推理失败:", str(e))
            return None

    def all_api_request(self):
        # 1. 预处理
        pre_result = self.prepare_request()
        taskid = pre_result["data"]

        # 2 . 分片上传
        self.upload_request(taskid=taskid, upload_file_path=self.upload_file_path)

        # 3 . 文件合并
        self.merge_request(taskid=taskid)

        # 4 . 获取任务进度
        while True:
            # 每隔20秒获取一次任务进度
            progress = self.get_progress_request(taskid)
            progress_dic = progress
            if progress_dic['err_no'] != 0 and progress_dic['err_no'] != 26605:
                print('task error: ' + progress_dic['failed'])
                return
            else:
                data = progress_dic['data']
                task_status = json.loads(data)
                if task_status['status'] == 9:
                    print('task ' + taskid + ' finished')
                    break
                print('The task ' + taskid + ' is in processing, task status: ' + str(data))

            # 每次获取进度间隔20S
            time.sleep(20)
        
        # 5 . 获取结果
        result = self.get_result_request(taskid=taskid)
        
        # 6 . 通过推理套件进行推理
        inference_result = self.run_inference()
        return inference_result



# https://www.xfyun.cn/doc/asr/lfasr/API.html#%E6%8E%A5%E5%8F%A3%E8%AF%B4%E6%98%8E

import os
import time
import hmac
import json
import boto3
import base64
import hashlib
import logging
import requests
import configparser

# 确保您的AWS访问密钥和私钥作为环境变量存储，或者您已经配置了AWS CLI
# os.environ['AWS_ACCESS_KEY_ID'] = 'your_access_key'
# os.environ['AWS_SECRET_ACCESS_KEY'] = 'your_secret_key'

lfasr_host = 'http://raasr.xfyun.cn/api'

# 请求的接口名
api_prepare = '/prepare'
api_upload = '/upload'
api_merge = '/merge'
api_get_progress = '/getProgress'
api_get_result = '/getResult'
# 文件分片大小10M
file_piece_sice = 10485760

config = configparser.ConfigParser()
config.read('config.ini')
lfasr_host = config['DEFAULT']['LfasrHost']

# ——————————————————转写可配置参数————————————————
# 参数可在官网界面（https://doc.xfyun.cn/rest_api/%E8%AF%AD%E9%9F%B3%E8%BD%AC%E5%86%99.html）查看，根据需求可自行在gene_params方法里添加修改
# 转写类型
lfasr_type = 0
# 是否开启分词
has_participle = 'false'
has_seperate = 'true'
# 多候选词个数
max_alternatives = 0
# 子用户标识
suid = ''

class RequestError(Exception):
    """Custom exception for request errors."""
    pass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 使用：
logging.info('upload slice ' + str(index) + ' success')

class SliceIdGenerator:
    """slice id生成器"""

    def __init__(self):
        self.__ch = 'aaaaaaaaa`'

    def getNextSliceId(self):
        ch = self.__ch
        j = len(ch) - 1
        while j >= 0:
            cj = ch[j]
            if cj != 'z':
                ch = ch[:j] + chr(ord(cj) + 1) + ch[j + 1:]
                break
            else:
                ch = ch[:j] + 'a' + ch[j + 1:]
                j = j - 1
        self.__ch = ch
        return self.__ch

class RequestApi(object):
    def __init__(self, appid, secret_key, upload_file_path):
        self.appid = appid
        self.secret_key = secret_key
        self.upload_file_path = upload_file_path

    @staticmethod
    def download_file_from_s3(bucket_name, s3_file_key, local_file_path):
        """从S3下载文件至本地"""
        s3 = boto3.client('s3')
        s3.download_file(bucket_name, s3_file_key, local_file_path)

    def validate_params(params):
    if not all(key in params for key in ['app_id', 'signa', 'ts']):
        raise ValueError("Missing required parameters in the request.")

    # 根据不同的apiname生成不同的参数,本示例中未使用全部参数您可在官网(https://doc.xfyun.cn/rest_api/%E8%AF%AD%E9%9F%B3%E8%BD%AC%E5%86%99.html)查看后选择适合业务场景的进行更换
    def gene_params(self, apiname, taskid=None, slice_id=None):
        appid = self.appid
        secret_key = self.secret_key
        upload_file_path = self.upload_file_path
        ts = str(int(time.time()))
        m2 = hashlib.md5()
        m2.update((appid + ts).encode('utf-8'))
        md5 = m2.hexdigest()
        md5 = bytes(md5, encoding='utf-8')
        # 以secret_key为key, 上面的md5为msg， 使用hashlib.sha1加密结果为signa
        signa = hmac.new(secret_key.encode('utf-8'), md5, hashlib.sha1).digest()
        signa = base64.b64encode(signa)
        signa = str(signa, 'utf-8')
        file_len = os.path.getsize(upload_file_path)
        file_name = os.path.basename(upload_file_path)
        param_dict = {}

        if apiname == api_prepare:
            # slice_num是指分片数量，如果您使用的音频都是较短音频也可以不分片，直接将slice_num指定为1即可
            slice_num = int(file_len / file_piece_sice) + (0 if (file_len % file_piece_sice == 0) else 1)
            param_dict['app_id'] = appid
            param_dict['signa'] = signa
            param_dict['ts'] = ts
            param_dict['file_len'] = str(file_len)
            param_dict['file_name'] = file_name
            param_dict['slice_num'] = str(slice_num)
        elif apiname == api_upload:
            param_dict['app_id'] = appid
            param_dict['signa'] = signa
            param_dict['ts'] = ts
            param_dict['task_id'] = taskid
            param_dict['slice_id'] = slice_id
        elif apiname == api_merge:
            param_dict['app_id'] = appid
            param_dict['signa'] = signa
            param_dict['ts'] = ts
            param_dict['task_id'] = taskid
            param_dict['file_name'] = file_name
        elif apiname == api_get_progress or apiname == api_get_result:
            param_dict['app_id'] = appid
            param_dict['signa'] = signa
            param_dict['ts'] = ts
            param_dict['task_id'] = taskid
        return param_dict

    # 请求和结果解析，结果中各个字段的含义可参考：https://doc.xfyun.cn/rest_api/%E8%AF%AD%E9%9F%B3%E8%BD%AC%E5%86%99.html
    def gene_request(self, apiname, data, files=None, headers=None):
        response = requests.post(lfasr_host + apiname, data=data, files=files, headers=headers)
        result = json.loads(response.text)
        if result["ok"] == 0:
            print("{} success:".format(apiname) + str(result))
            return result
        else:
            print("{} error:".format(apiname) + str(result))
            exit(0)
            return result

    # 预处理
    def prepare_request(self):
        return self.gene_request(apiname=api_prepare,
                                 data=self.gene_params(api_prepare))

    # 上传
    def upload_request(self, taskid, upload_file_path):
        file_object = open(upload_file_path, 'rb')
        try:
            index = 1
            sig = SliceIdGenerator()
            while True:
                content = file_object.read(file_piece_sice)
                if not content or len(content) == 0:
                    break
                files = {
                    "filename": self.gene_params(api_upload).get("slice_id"),
                    "content": content
                }
                response = self.gene_request(api_upload,
                                             data=self.gene_params(api_upload, taskid=taskid,
                                                                   slice_id=sig.getNextSliceId()),
                                             files=files)
                if response.get('ok') != 0:
                    # 上传分片失败
                    print('upload slice fail, response: ' + str(response))
                    return False
                print('upload slice ' + str(index) + ' success')
                index += 1
        finally:
            'file index:' + str(file_object.tell())
            file_object.close()
        return True

    # 合并
    def merge_request(self, taskid):
        return self.gene_request(api_merge, data=self.gene_params(api_merge, taskid=taskid))

    # 获取进度
    def get_progress_request(self, taskid):
        return self.gene_request(api_get_progress, data=self.gene_params(api_get_progress, taskid=taskid))

    # 获取结果
    def get_result_request(self, taskid):
        return self.gene_request(api_get_result, data=self.gene_params(api_get_result, taskid=taskid))

    def all_api_request(self):
        # 1. 预处理
        pre_result = self.prepare_request()
        taskid = pre_result["data"]
        # 2 . 分片上传
        self.upload_request(taskid=taskid, upload_file_path=self.upload_file_path)
        # 3 . 文件合并
        self.merge_request(taskid=taskid)
        # 4 . 获取任务进度
        while True:
            # 每隔20秒获取一次任务进度
            progress = self.get_progress_request(taskid)
            progress_dic = progress
            if progress_dic['err_no'] != 0 and progress_dic['err_no'] != 26605:
                print('task error: ' + progress_dic['failed'])
                return
            else:
                data = progress_dic['data']
                task_status = json.loads(data)
                if task_status['status'] == 9:
                    print('task ' + taskid + ' finished')
                    break
                print('The task ' + taskid + ' is in processing, task status: ' + str(data))

            # 每次获取进度间隔20S
            time.sleep(20)
        # 5 . 获取结果
        self.get_result_request(taskid=taskid)

# 注意：如果出现requests模块报错："NoneType" object has no attribute 'read', 请尝试将requests模块更新到2.20.0或以上版本(本demo测试版本为2.20.0)
# 输入讯飞开放平台的appid，secret_key和待转写的文件路径
if __name__ == '__main__':
    # S3桶名和文件键值
    bucket_name = 'your_bucket_name_here'
    s3_file_key = 'your_file_key_here'
    local_file_path = r'path_to_local_file'

    # 从S3下载音频文件
    RequestApi.download_file_from_s3(bucket_name, s3_file_key, local_file_path)

    # 科大讯飞开放平台的appid和secret_key，以及要上传文件的本地路径
    appid = "your_appid_here"
    secret_key = "your_secret_key_here"
    upload_file_path = local_file_path

    # 创建RequestApi对象并执行所有API请求
    api = RequestApi(appid=appid, secret_key=secret_key, upload_file_path=upload_file_path)
    api.all_api_request()



import os
import time
import hmac
import json
import boto3
import base64
import hashlib
import logging
import requests
import configparser

# 确保您的AWS访问密钥和私钥作为环境变量存储，或者您已经配置了AWS CLI
# os.environ['AWS_ACCESS_KEY_ID'] = 'your_access_key'
# os.environ['AWS_SECRET_ACCESS_KEY'] = 'your_secret_key'

lfasr_host = 'http://raasr.xfyun.cn/api'

# 请求的接口名
api_prepare = '/prepare'
api_upload = '/upload'
api_merge = '/merge'
api_get_progress = '/getProgress'
api_get_result = '/getResult'
# 文件分片大小10M
file_piece_sice = 10485760

config = configparser.ConfigParser()
config.read('config.ini')
lfasr_host = config['DEFAULT']['LfasrHost']

# ——————————————————转写可配置参数————————————————
# 转写类型
lfasr_type = 0
# 是否开启分词
has_participle = 'false'
has_seperate = 'true'
# 多候选词个数
max_alternatives = 0
# 子用户标识
suid = ''

class RequestError(Exception):
    """Custom exception for request errors."""
    pass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 使用：
logging.info('upload slice ' + str(index) + ' success')

class SliceIdGenerator:
    """slice id生成器"""

    def __init__(self):
        self.__ch = 'aaaaaaaaa`'

    def getNextSliceId(self):
        ch = self.__ch
        j = len(ch) - 1
        while j >= 0:
            cj = ch[j]
            if cj != 'z':
                ch = ch[:j] + chr(ord(cj) + 1) + ch[j + 1:]
                break
            else:
                ch = ch[:j] + 'a' + ch[j + 1:]
                j = j - 1
        self.__ch = ch
        return self.__ch

class RequestApi(object):
    def __init__(self, appid, secret_key, upload_file_path):
        self.appid = appid
        self.secret_key = secret_key
        self.upload_file_path = upload_file_path

    @staticmethod
    def download_file_from_s3(bucket_name, s3_file_key, local_file_path):
        """从S3下载文件至本地"""
        s3 = boto3.client('s3')
        s3.download_file(bucket_name, s3_file_key, local_file_path)

    @staticmethod
    def validate_params(params):
        if not all(key in params for key in ['app_id', 'signa', 'ts']):
            raise ValueError("Missing required parameters in the request.")

    def gene_params(self, apiname, taskid=None, slice_id=None):
        appid = self.appid
        secret_key = self.secret_key
        upload_file_path = self.upload_file_path
        ts = str(int(time.time()))
        m2 = hashlib.md5()
        m2.update((appid + ts).encode('utf-8'))
        md5 = m2.hexdigest()
        md5 = bytes(md5, encoding='utf-8')
        # 以secret_key为key, 上面的md5为msg， 使用hashlib.sha1加密结果为signa
        signa = hmac.new(secret_key.encode('utf-8'), md5, hashlib.sha1).digest()
        signa = base64.b64encode(signa)
        signa = str(signa, 'utf-8')
        file_len = os.path.getsize(upload_file_path)
        file_name = os.path.basename(upload_file_path)
        param_dict = {}

        if apiname == api_prepare:
            # slice_num是指分片数量，如果您使用的音频都是较短音频也可以不分片，直接将slice_num指定为1即可
            slice_num = int(file_len / file_piece_sice) + (0 if (file_len % file_piece_sice == 0) else 1)
            param_dict['app_id'] = appid
            param_dict['signa'] = signa
            param_dict['ts'] = ts
            param_dict['file_len'] = str(file_len)
            param_dict['file_name'] = file_name
            param_dict['slice_num'] = str(slice_num)
        elif apiname == api_upload:
            param_dict['app_id'] = appid
            param_dict['signa'] = signa
            param_dict['ts'] = ts
            param_dict['task_id'] = taskid
            param_dict['slice_id'] = slice_id
        elif apiname == api_merge:
            param_dict['app_id'] = appid
            param_dict['signa'] = signa
            param_dict['ts'] = ts
            param_dict['task_id'] = taskid
            param_dict['file_name'] = file_name
        elif apiname == api_get_progress or apiname == api_get_result:
            param_dict['app_id'] = appid
            param_dict['signa'] = signa
            param_dict['ts'] = ts
            param_dict['task_id'] = taskid
        return param_dict

    def gene_request(self, apiname, data, files=None, headers=None):
        response = requests.post(lfasr_host + apiname, data=data, files=files, headers=headers)
        result = json.loads(response.text)
        if result["ok"] == 0:
            print("{} success:".format(apiname) + str(result))
            return result
        else:
            print("{} error:".format(apiname) + str(result))
            exit(0)
            return result

    def prepare_request(self):
        return self.gene_request(apiname=api_prepare,
                                 data=self.gene_params(api_prepare))

    def upload_request(self, taskid, upload_file_path):
        file_object = open(upload_file_path, 'rb')
        try:
            index = 1
            sig = SliceIdGenerator()
            while True:
                content = file_object.read(file_piece_sice)
                if not content or len(content) == 0:
                    break
                files = {
                    "filename": self.gene_params(api_upload).get("slice_id"),
                    "content": content
                }
                response = self.gene_request(api_upload,
                                             data=self.gene_params(api_upload, taskid=taskid,
                                                                   slice_id=sig.getNextSliceId()),
                                             files=files)
                if response.get('ok') != 0:
                    # 上传分片失败
                    print('upload slice fail, response: ' + str(response))
                    return False
                print('upload slice ' + str(index) + ' success')
                index += 1
        finally:
            'file index:' + str(file_object.tell())
            file_object.close()
        return True

    def merge_request(self, taskid):
        return self.gene_request(api_merge, data=self.gene_params(api_merge, taskid=taskid))

    def get_progress_request(self, taskid):
        return self.gene_request(api_get_progress, data=self.gene_params(api_get_progress, taskid=taskid))

    def get_result_request(self, taskid):
        return self.gene_request(api_get_result, data=self.gene_params(api_get_result, taskid=taskid))

    def all_api_request(self):
        # 1. 预处理
        pre_result = self.prepare_request()
        taskid = pre_result["data"]
        # 2 . 分片上传
        self.upload_request(taskid=taskid, upload_file_path=self.upload_file_path)
        # 3 . 文件合并
        self.merge_request(taskid=taskid)
        # 4 . 获取任务进度
        while True:
            # 每隔20秒获取一次任务进度
            progress = self.get_progress_request(taskid)
            progress_dic = progress
            if progress_dic['err_no'] != 0 and progress_dic['err_no'] != 26605:
                print('task error: ' + progress_dic['failed'])
                return
            else:
                data = progress_dic['data']
                task_status = json.loads(data)
                if task_status['status'] == 9:
                    print('task ' + taskid + ' finished')
                    break
                print('The task ' + taskid + ' is in processing, task status: ' + str(data))

            # 每次获取进度间隔20S
            time.sleep(20)
        # 5 . 获取结果
        self.get_result_request(taskid=taskid)

        # 6 . 调用科大讯飞推理套件进行推理
        self.run_inference()

    def run_inference(self):
        """调用科大讯飞推理套件进行推理"""
        try:
            # 假设你已经配置并导入了科大讯飞的推理SDK
            result = self.inference.transcribe(self.upload_file_path)
            print("推理结果:", result)
        except Exception as e:
            print("推理失败:", str(e))

if __name__ == '__main__':
    # S3桶名和文件键值
    bucket_name = 'your_bucket_name_here'
    s3_file_key = 'your_file_key_here'
    local_file_path = r'path_to_local_file'

    # 从S3下载音频文件
    RequestApi.download_file_from_s3(bucket_name, s3_file_key, local_file_path)

    # 科大讯飞开放平台的appid和secret_key，以及要上传文件的本地路径
    appid = "your_appid_here"
    secret_key = "your_secret_key_here"
    upload_file_path = local_file_path

    # 创建RequestApi对象并执行所有API请求
    api = RequestApi(appid=appid, secret_key=secret_key, upload_file_path=upload_file_path)
    api.all_api_request()
