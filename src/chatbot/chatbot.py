# chatbot.py

import time
import hmac
import hashlib
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class JapaneseChatbot:
    def __init__(self, appid, secret_key):
        self.appid = appid
        self.secret_key = secret_key

    def generate_signature(self):
        """生成API请求签名"""
        ts = str(int(time.time()))
        m2 = hashlib.md5()
        m2.update((self.appid + ts).encode('utf-8'))
        md5 = m2.hexdigest()
        md5 = bytes(md5, encoding='utf-8')
        signa = hmac.new(self.secret_key.encode('utf-8'), md5, hashlib.sha1).digest()
        signa = base64.b64encode(signa).decode('utf-8')
        return signa, ts

    def send_message(self, message):
        """处理用户消息并生成日语响应"""
        # 模拟生成一个API请求签名（用于集成第三方服务）
        signa, ts = self.generate_signature()

        # 模拟处理用户输入，生成响应
        logging.info(f"受信メッセージ: {message}")
        
        # 简单的规则来模拟对话
        if "こんにちは" in message:
            response = "こんにちは！何かお手伝いできることはありますか？"
        elif "時間" in message:
            response = f"現在の時刻は: {time.strftime('%Y年%m月%d日 %H:%M:%S', time.localtime())} です。"
        elif "お名前" in message:
            response = "私はあなたのチャットボットです。"
        else:
            response = "すみません、よくわかりませんでした。もう一度言っていただけますか？"

        logging.info(f"生成された応答: {response}")
        return response

if __name__ == '__main__':
    appid = "your_appid_here"
    secret_key = "your_secret_key_here"

    # JapaneseChatbotオブジェクトの作成
    bot = JapaneseChatbot(appid=appid, secret_key=secret_key)
    
    # ユーザーとの対話をテスト
    user_input = input("ユーザー: ")
    while user_input.lower() not in ['終了', 'やめる']:
        response = bot.send_message(user_input)
        print("チャットボット:", response)
        user_input = input("ユーザー: ")
    
    print("チャットボット: さようなら！")
