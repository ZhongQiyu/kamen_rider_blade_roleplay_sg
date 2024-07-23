# run_gradio.py

import os
import re
import json
import torch
import redis
import logging
import gradio as gr
from PIL import Image
from typing import List
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from langdetect import detect
from transformers import (
    pipeline, set_seed, Trainer, AutoTokenizer, BertTokenizerFast, TrainingArguments,
    BertForTokenClassification, BertForSequenceClassification, GPT2Tokenizer, GPT2LMHeadModel,
    BertJapaneseTokenizer, BertForMaskedLM, LongformerTokenizer, LongformerModel, CLIPProcessor, CLIPModel
)
from pydantic import BaseModel, Field, field_validator
from janome.tokenizer import Tokenizer
from fugashi import Tagger

# 设置随机种子，以确保文本生成的可重复性
set_seed(42)

app = Flask(__name__)

# ChatBot class
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
            outputs=gr.Chatbot(label="Response"),
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
        
        for scene in dialogues:
            for dialogue in scene['dialogue']:
                role_id = dialogue.get('role_id')
                dialogue['role_info'] = roles.get(role_id, {})
        
        return dialogues

# FileManager class
class FileManager:
    def __init__(self, base_path='.'):
        self.base_path = base_path

    def read_file(self, filename):
        with open(os.path.join(self.base_path, filename), 'r') as file:
            return file.read()

    def write_file(self, filename, content):
        with open(os.path.join(self.base_path, filename), 'w') as file:
            file.write(content)

# FileParser class
class FileParser:
    def parse_json(self, content):
        return json.loads(content)

    def to_json(self, data):
        return json.dumps(data, indent=4)

# BiasHandler class
class BiasHandler:
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.fugashi_tagger = Tagger()
        self.bert_tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese')
        self.bert_model = BertForMaskedLM.from_pretrained('cl-tohoku/bert-base-japanese')
        self.bias_words = {
            '悪い': '良くない',
            'ダメ': '良くない',
            '嫌い': '好きではない'
        }
    
    def preprocess_text(self, text):
        text = re.sub(r'[^\w\s]', '', text)
        tokens = self.tokenizer.tokenize(text, wakati=True)
        return tokens

    def detect_bias(self, tokens):
        return [word for word in tokens if word in self.bias_words]

    def correct_bias(self, tokens):
        return [self.bias_words.get(word, word) for word in tokens]

    def contextual_correction(self, text):
        input_ids = self.bert_tokenizer.encode(text, return_tensors='pt')
        with torch.no_grad():
            output = self.bert_model(input_ids)
        logits = output.logits
        mask_token_index = torch.where(input_ids == self.bert_tokenizer.mask_token_id)[1]
        top_k_words = torch.topk(logits[0, mask_token_index, :], 1, dim=1).indices[0].tolist()
        return [self.bert_tokenizer.decode(word_id) for word_id in top_k_words]

    def process_text(self, text):
        tokens = self.preprocess_text(text)
        if self.detect_bias(tokens):
            tokens = self.correct_bias(tokens)
            corrected_text = ''.join(tokens)
            return ''.join(self.contextual_correction(corrected_text))
        return text

# DialogueGenerator class
class DialogueGenerator:
    def __init__(self):
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.longformer_tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
        self.longformer_model = LongformerModel.from_pretrained('allenai/longformer-base-4096')
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

    def generate_conversation(self, prompt, max_turns=500, window_size=5):
        input_ids = self.gpt2_tokenizer.encode(prompt, return_tensors='pt')
        chat_history_ids = input_ids
        conversation = prompt

        for _ in range(max_turns):
            current_input_ids = chat_history_ids[:, -window_size*self.gpt2_tokenizer.model_max_length:]
            outputs = self.gpt2_model.generate(current_input_ids, max_length=len(current_input_ids[0]) + 50, pad_token_id=self.gpt2_tokenizer.eos_token_id)
            new_tokens = outputs[:, current_input_ids.shape[-1]:]
            chat_history_ids = torch.cat([chat_history_ids, new_tokens], dim=-1)
            new_text = self.gpt2_tokenizer.decode(new_tokens[0], skip_special_tokens=True)
            conversation += new_text
            if new_text.strip() == "":
                break
        
        return conversation

    def generate_long_text(self, prompt, max_length=4096):
        input_ids = self.longformer_tokenizer.encode(prompt, return_tensors='pt')
        outputs = self.longformer_model(input_ids)
        last_hidden_states = outputs.last_hidden_state
        return last_hidden_states

    def process_multimodal_input(self, text, image_path):
        text_inputs = self.gpt2_tokenizer(text, return_tensors='pt')
        text_outputs = self.gpt2_model.generate(text_inputs['input_ids'], max_length=100)
        text_response = self.gpt2_tokenizer.decode(text_outputs[0], skip_special_tokens=True)

        image = Image.open(image_path)
        image_inputs = self.clip_processor(images=image, return_tensors='pt')
        image_outputs = self.clip_model.get_image_features(**image_inputs)
        
        return text_response, image_outputs.tolist()

# ConversationStorage class
class ConversationStorage:
    def __init__(self, host='localhost', port=6379, db=0):
        self.r = redis.Redis(host=host, port=port, db=db)

    def save_conversation(self, conversation_id, conversation):
        self.r.set(conversation_id, json.dumps(conversation))

    def load_conversation(self, conversation_id):
        conversation = self.r.get(conversation_id)
        if conversation:
            return json.loads(conversation)
        return []

# TextUtilities class
class TextUtilities:
    @staticmethod
    def split_japanese_text(text: str) -> List[str]:
        sentences = text.split("。")
        return [sentence.strip() + "。" for sentence in sentences if sentence.strip()]

    @staticmethod
    def process_asr_output(raw_asr_text: str) -> str:
        return raw_asr_text.replace("\n", " ").strip()

    @staticmethod
    def tokenize_text(text: str) -> List[str]:
        tagger = Tagger('-Owakati')
        return tagger.parse(text).strip().split()

    @staticmethod
    def tokenize_japanese_janome(text: str) -> List[str]:
        tokenizer = Tokenizer()
        return list(tokenizer.tokenize(text, wakati=True))

class Dialogue(BaseModel):
    text: str
    tags: List[str]
    dialogue_id: str
    utterances: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    participants: List[str] = []

    @field_validator('tags', mode='before')
    def check_tags(cls, v):
        if not isinstance(v, list) or not all(isinstance(item, str) for item in v):
            raise ValueError('Each tag must be a list of strings')
        return v

    @property
    def number_of_utterances(self):
        return len(self.utterances)

class KnowledgeBase:
    def __init__(self, file_path='dialogues.json'):
        self.file_path = file_path
        self.dialogues = self.load_from_file()

    def load_from_file(self):
        if not os.path.exists(self.file_path):
            return {}
        with open(self.file_path, 'r', encoding='utf-8') as file:
            return json.load(file)

    def save_to_file(self):
        with open(self.file_path, 'w', encoding='utf-8') as file:
            json.dump(self.dialogues, file, ensure_ascii=False, indent=4)

    def add_dialogue(self, dialogue):
        self.dialogues[dialogue.dialogue_id] = dialogue.dict()
        self.save_to_file()

# Initialize components
kb = KnowledgeBase()
bias_handler = BiasHandler()
generator = DialogueGenerator()
conversation_storage = ConversationStorage()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/add_dialogue', methods=['POST'])
def add_dialogue():
    data = request.json
    dialogue = Dialogue(**data)
    kb.add_dialogue(dialogue)
    return jsonify({"message": "Dialogue added successfully"})

@app.route('/generate_dialogue', methods=['POST'])
def generate_dialogue():
    data = request.json
    prompt = data.get('prompt', '')
    processed_prompt = bias_handler.process_text(prompt)
    generated_text = generator.generate_conversation(processed_prompt)
    return jsonify({"generated_text": generated_text})

@app.route('/generate_long_text', methods=['POST'])
def generate_long_text():
    data = request.json
    prompt = data.get('prompt', '')
    long_text = generator.generate_long_text(prompt)
    return jsonify({"long_text": long_text.tolist()})

@app.route('/process_multimodal', methods=['POST'])
def process_multimodal():
    data = request.json
    text = data.get('text', '')
    image_path = data.get('image_path', '')
    text_response, image_outputs = generator.process_multimodal_input(text, image_path)
    return jsonify({"text_response": text_response, "image_outputs": image_outputs})

@app.route('/process_asr', methods=['POST'])
def process_asr():
    data = request.json
    raw_text = data.get('raw_text', '')
    processed_text = TextUtilities.process_asr_output(raw_text)
    return jsonify({"processed_text": processed_text})

@app.route('/save_conversation', methods=['POST'])
def save_conversation():
    data = request.json
    conversation_id = data.get('conversation_id')
    conversation = data.get('conversation')
    conversation_storage.save_conversation(conversation_id, conversation)
    return jsonify({"message": "Conversation saved successfully"})

@app.route('/load_conversation', methods=['POST'])
def load_conversation():
    data = request.json
    conversation_id = data.get('conversation_id')
    conversation = conversation_storage.load_conversation(conversation_id)
    return jsonify({"conversation": conversation})

# Gradio interface functions
def generate(input, temperature):
    output = generator.gpt2_model.generate(generator.gpt2_tokenizer.encode(input, return_tensors='pt').to('cuda'), temperature=temperature)
    return generator.gpt2_tokenizer.decode(output[0], skip_special_tokens=True)

def format_chat_prompt(message, chat_history):
    prompt = ""
    for turn in chat_history:
        user_message, bot_message = turn
        prompt = f"{prompt}\nUser: {user_message}\nAssistant: {bot_message}"
    prompt = f"{prompt}\nUser: {message}\nAssistant:"
    return prompt

def respond(message, chat_history):
    formatted_prompt = format_chat_prompt(message, chat_history)
    bot_message = generator.gpt2_model.generate(generator.gpt2_tokenizer.encode(formatted_prompt, return_tensors='pt').to('cuda'), max_new_tokens=1024, stop_sequences=["\nUser:", ""])
    bot_message_text = generator.gpt2_tokenizer.decode(bot_message[0], skip_special_tokens=True)
    chat_history.append((message, bot_message_text))
    return "", chat_history

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(height=240) 
    msg = gr.Textbox(label="Prompt")
    btn = gr.Button("Submit")
    clear = gr.ClearButton(components=[msg, chatbot], value="Clear console")

    btn.click(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])
    msg.submit(respond, inputs=[msg, chatbot], outputs=[msg, chatbot])

    db_with_his_btn = gr.Button("Chat db with history")
    db_wo_his_btn = gr.Button("Chat db without history")

    gr.File(label='请选择知识库目录', file_count='directory', file_types=['.txt', '.md', '.docx', '.pdf'])

    temperature = gr.Slider(0, 1, value=0.00, step=0.01, label="llm temperature", interactive=True)

    llm = gr.Dropdown([], label="large language model", value=None, interactive=True)  # Placeholder for the LLM models

    model_select = gr.Accordion("模型选择")
    with model_select:
        llm = gr.Dropdown([], label="large language model", value=None, interactive=True)
        embedding = gr.Dropdown([], label="embedding model", value=None, interactive=True)

    gr.Markdown("""<h1><center>Chat Robot</center></h1>
    <center>Local Knowledge Base Q&A with llm</center>
    """)

    with gr.Row():
        db_with_his_btn = gr.Button("Chat db with history")
        db_wo_his_btn = gr.Button("Chat db without history")
        llm_btn = gr.Button("Chat with llm")

    with gr.Column(scale=4):
        chatbot = gr.Chatbot(height=480) 
    with gr.Column(scale=1):
        model_argument = gr.Accordion("参数配置", open=False)
        with model_argument:
            model_select = gr.Accordion("模型选择")
            with model_select:
                pass

gr.close_all()
demo.launch(share=True)  # 设置共享链接

if __name__ == "__main__":
    app.run(debug=True, port=5001)

