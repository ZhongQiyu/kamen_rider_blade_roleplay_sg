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



# run_gradio.py

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, LongformerTokenizer, LongformerModel
import gradio as gr

class ChatBot:
    def __init__(self):
        # 初始化GPT-2和Longformer模型和分词器
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.longformer_tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
        self.longformer_model = LongformerModel.from_pretrained('allenai/longformer-base-4096')

    def generate_text(self, prompt, max_length=50):
        # 使用GPT-2生成文本
        input_ids = self.gpt2_tokenizer.encode(prompt, return_tensors='pt')
        output = self.gpt2_model.generate(input_ids, max_length=max_length)
        return self.gpt2_tokenizer.decode(output[0], skip_special_tokens=True)

    def generate_long_text(self, prompt, max_length=4096):
        # 使用Longformer生成长文本
        input_ids = self.longformer_tokenizer.encode(prompt, return_tensors='pt')
        outputs = self.longformer_model(input_ids)
        return outputs.last_hidden_state

    def generate_conversation(self, formatted_prompt, max_new_tokens=1024, stop_sequences=["\nUser:", ""]):
        # 使用GPT-2生成对话
        input_ids = self.gpt2_tokenizer.encode(formatted_prompt, return_tensors='pt').to('cuda')
        output = self.gpt2_model.generate(input_ids, max_new_tokens=max_new_tokens, stop_sequences=stop_sequences)
        return self.gpt2_tokenizer.decode(output[0], skip_special_tokens=True)

    def format_chat_prompt(self, message, chat_history):
        # 格式化对话历史为Prompt
        prompt = ""
        for turn in chat_history:
            user_message, bot_message = turn
            prompt = f"{prompt}\nUser: {user_message}\nAssistant: {bot_message}"
        prompt = f"{prompt}\nUser: {message}\nAssistant:"
        return prompt

    def respond(self, message, chat_history):
        # 处理用户输入并生成响应
        formatted_prompt = self.format_chat_prompt(message, chat_history)
        bot_message = self.generate_conversation(formatted_prompt)
        chat_history.append((message, bot_message))
        return "", chat_history

    def launch_gradio(self):
        # 使用Gradio启动Web界面
        with gr.Blocks() as demo:
            chatbot = gr.Chatbot(height=240) 
            msg = gr.Textbox(label="Prompt")
            btn = gr.Button("Submit")
            clear = gr.ClearButton(components=[msg, chatbot], value="Clear console")

            btn.click(self.respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
            msg.submit(self.respond, inputs=[msg, chatbot], outputs=[msg, chatbot])

            gr.Markdown("""<h1><center>Chat Robot</center></h1>
            <center>Local Knowledge Base Q&A with LLM</center>
            """)

        demo.launch(share=True)

# 使用示例
if __name__ == '__main__':
    chat_bot = ChatBot()
    chat_bot.launch_gradio()
