# LLM.py

import os
import json
import torch
import logging
from PIL import Image
from typing import List
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from transformers import (
    set_seed, GPT2Tokenizer, GPT2LMHeadModel, LongformerTokenizer, LongformerModel, CLIPProcessor, CLIPModel,
    BertForQuestionAnswering, BertTokenizer, pipeline
)
from pydantic import BaseModel, Field, validator

# 设置随机种子，以确保文本生成的可重复性
set_seed(42)

app = Flask(__name__)

# ChatBot class
class ChatBot:
    def __init__(self, tokenizer, model, device='cpu'):
        self.history = []
        self.device = device
        self.tokenizer = tokenizer
        self.model = model.to(device)
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)

    def chat(self, prompt, temperature=0.7):
        inputs = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        outputs = self.model.generate(inputs, max_length=100, num_return_sequences=1, temperature=temperature)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        self.history.append(prompt)
        self.history.append(response)
        return response

# DialogueGenerator class
class DialogueGenerator:
    def __init__(self):
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.longformer_tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
        self.longformer_model = LongformerModel.from_pretrained('allenai/longformer-base-4096')
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

    def generate_conversation(self, prompt, max_turns=5):
        input_ids = self.gpt2_tokenizer.encode(prompt, return_tensors='pt')
        chat_history_ids = input_ids
        conversation = prompt

        for _ in range(max_turns):
            current_input_ids = chat_history_ids[:, -self.gpt2_tokenizer.model_max_length:]
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

class Dialogue(BaseModel):
    text: str
    tags: List[str]
    dialogue_id: str
    utterances: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    participants: List[str] = []

    @validator('tags', pre=True)
    def check_tags(cls, v):
        if not isinstance(v, list) or not all(isinstance(item, str) for item in v):
            raise ValueError('Each tag must be a list of strings')
        return v

    @property
    def number_of_utterances(self):
        return len(self.utterances)

# Additional model loading for text classification and QA
class AdditionalModels:
    def __init__(self):
        self.text_classification_pipeline = None
        self.qa_pipeline = None

    def load_text_classification_model(self, model_name):
        model = BertForSequenceClassification.from_pretrained(model_name)
        tokenizer = BertTokenizer.from_pretrained(model_name)
        self.text_classification_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)

    def classify_text(self, text):
        if not self.text_classification_pipeline:
            raise ValueError("Text classification model not loaded.")
        return self.text_classification_pipeline(text)

    def load_qa_model(self, model_name):
        model = BertForQuestionAnswering.from_pretrained(model_name)
        tokenizer = BertTokenizer.from_pretrained(model_name)
        self.qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

    def answer_question(self, question, context):
        if not self.qa_pipeline:
            raise ValueError("QA model not loaded.")
        return self.qa_pipeline(question=question, context=context)

# Initialize components
generator = DialogueGenerator()
additional_models = AdditionalModels()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_dialogue', methods=['POST'])
def generate_dialogue():
    data = request.json
    prompt = data.get('prompt', '')
    generated_text = generator.generate_conversation(prompt)
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

@app.route('/classify_text', methods=['POST'])
def classify_text():
    data = request.json
    text = data.get('text', '')
    result = additional_models.classify_text(text)
    return jsonify(result)

@app.route('/answer_question', methods=['POST'])
def answer_question():
    data = request.json
    question = data.get('question', '')
    context = data.get('context', '')
    result = additional_models.answer_question(question, context)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, port=5001)
