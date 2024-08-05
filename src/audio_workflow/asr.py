# asr.py

import json
from aws_transcribe import start_aws_workflow
from iflytek import start_iflytek_workflow

def load_model_config(model_name, config_file="model_config.json"):
    with open(config_file, "r") as f:
        config = json.load(f)
    return config["models"].get(model_name)

def main():
    # 根据需求选择调用AWS或讯飞的流程
    choice = input("选择处理流程 (1: AWS, 2: 讯飞): ")
    
    if choice == '1':
        start_aws_workflow()
        model_config = load_model_config("BERT")
        print("Loaded BERT model config for AWS workflow:", model_config)
    elif choice == '2':
        start_iflytek_workflow()
        model_config = load_model_config("BERT-CHINESE")
        print("Loaded BERT-Chinese model config for iFlytek workflow:", model_config)
    else:
        print("无效选择")

if __name__ == "__main__":
    main()
