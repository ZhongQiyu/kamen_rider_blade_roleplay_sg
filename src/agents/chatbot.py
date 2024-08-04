# chatbot.py

import os
import gradio as gr
from transformers import pipeline, set_seed

character_image_path = "image/character1.png"

def chatbot_response(user_id, message):
    responses = {
        "你好": "你好！",
        "再见": "再见！",
        "你是谁？": "我是一个聊天机器人。"
    }
    default_response = "我不确定如何回答这个问题。"
    response = responses.get(message, default_response)
    return {
        "message": response,
        "name": user_id,
        "image": character_image_path,
        "is_user": False
    }

generator = pipeline('text-generation', model='gpt2')

def generate_text(prompt, temperature=0.7):
    set_seed(42)
    outputs = generator(prompt, max_length=100, num_return_sequences=1, temperature=temperature)
    return outputs[0]['generated_text']

with gr.Blocks() as demo:
    gr.Markdown("### 聊天对话系统")
    with gr.Row():
        user_id_input = gr.Textbox(label="用户 ID", placeholder="请输入您的用户 ID", value="user123")
        message_input = gr.Textbox(label="你的消息", placeholder="输入你想说的话...")
        submit_button = gr.Button("发送")
    chatbox = gr.Chatbot(label="聊天历史")

    submit_button.click(
        fn=chatbot_response,
        inputs=[user_id_input, message_input],
        outputs=chatbox
    )

    gr.Markdown("### 文本生成")
    with gr.Row():
        input_text = gr.Textbox(label="Prompt")
        temperature_slider = gr.Slider(label="Temperature", value=0.7, minimum=0, maximum=1)
    output_text = gr.Textbox(label="Generated Text")

    input_text.submit(
        fn=generate_text,
        inputs=[input_text, temperature_slider],
        outputs=output_text
    )
    temperature_slider.submit(
        fn=generate_text,
        inputs=[input_text, temperature_slider],
        outputs=output_text
    )

demo.launch(share=True)

import json
import logging
import gradio as gr
import torch
from transformers import pipeline, set_seed, BertTokenizerFast, BertForSequenceClassification

# 设置随机种子，以确保文本生成的可重复性
set_seed(42)

class ChatBot:
    def __init__(self, config, tokenizer, model, device='cpu'):
        self.history = []
        self.config = config
        self.device = device
        self.initialize_model()
        self.tokenizer = tokenizer
        self.model = model.to(device)
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)

    def initialize_model(self):
        try:
            self.tokenizer = BertTokenizerFast.from_pretrained(self.config.get('tokenizer_model', 'bert-base-japanese'))
            self.model = BertForSequenceClassification.from_pretrained(self.config.get('model_base', 'bert-base-japanese'), num_labels=self.config.get('num_labels', 2))
        except Exception as e:
            self.logger.error(f"Model loading failed: {str(e)}")
            raise

    def create_gradio_interface(self, chatbot):
        def chat_interface(prompt, temperature=0.7):
            response = chatbot.chat(prompt, temperature)
            return [(prompt, True), (response, False)]

        demo = gr.Interface(
            fn=chat_interface,
            inputs=[gr.Textbox(label="Prompt"), gr.Slider(label="Temperature", minimum=0, maximum=1, value=0.7)],
            outputs=gr.Textbox(label="Response"),
            title="Chat Robot",
            description="Enter your question or statement and get a response."
        )
        return demo

    def chat(self, prompt, temperature=0.7):
        inputs = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        outputs = self.model.generate(inputs, max_length=100, num_return_sequences=1, temperature=temperature)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        self.history.append(prompt)
        self.history.append(response)
        return response

    def load_data(self, dialogues_file, roles_file):
        with open(dialogues_file, 'r', encoding='utf-8') as f:
            dialogues = json.load(f)
        
        with open(roles_file, 'r', encoding='utf-8') as f:
            roles = json.load(f)
        
        # 合并对话数据和角色信息
        for scene in dialogues:
            for dialogue in scene['dialogues']:  # Changed 'dialogue' to 'dialogues'
                role_id = dialogue.get('role_id')
                dialogue['role_info'] = roles.get(role_id, {})
        
        return dialogues

def main():
    config = {
        'tokenizer_model': 'bert-base-multilingual-cased',
        'model_base': 'bert-base-multilingual-cased',
        'num_labels': 2,  # 根据你的任务调整，例如二分类
        'task': 'classification',  # 'ner' for Named Entity Recognition
    }

    tokenizer = BertTokenizerFast.from_pretrained(config['tokenizer_model'])
    model = BertForSequenceClassification.from_pretrained(config['model_base'], num_labels=config['num_labels'])

    chatbot = ChatBot(config=config, tokenizer=tokenizer, model=model)
    
    # 加载对话和角色数据
    data_dir = "/Users/qaz1214/Downloads/kamen-rider-blade-roleplay/data/processed/text/scripts/"
    dialogues = chatbot.load_data(data_dir + 'dialogues.json', data_dir + 'roles.json')
    print(dialogues)

    demo = chatbot.create_gradio_interface(chatbot)
    demo.launch(share=True)

if __name__ == "__main__":
    main()
