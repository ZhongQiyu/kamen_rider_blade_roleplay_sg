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
