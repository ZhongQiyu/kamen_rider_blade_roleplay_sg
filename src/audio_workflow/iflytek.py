# iflytek.py

from xfyun_sdk import Inference

def start_iflytek_workflow():
    appid = "your_appid_here"
    secret_key = "your_secret_key_here"
    local_file_path = r'path_to_local_file'
    
    # 创建RequestApi对象并执行所有API请求
    api = RequestApi(appid=appid, secret_key=secret_key, upload_file_path=local_file_path)
    api.all_api_request()

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
        # 这里继续之前的API调用流程
        pre_result = self.prepare_request()
        taskid = pre_result["data"]

        # 上传并处理音频文件
        self.upload_request(taskid=taskid, upload_file_path=self.upload_file_path)
        self.merge_request(taskid=taskid)

        # 获取并处理任务进度
        while True:
            progress = self.get_progress_request(taskid)
            if progress['status'] == 9:
                break
            time.sleep(20)
        
        # 获取结果并调用推理
        result = self.get_result_request(taskid=taskid)
        inference_result = self.run_inference()
        return inference_result
