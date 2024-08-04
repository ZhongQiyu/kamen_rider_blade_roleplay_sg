# rag.py

import json
import random
from datetime import datetime
from collections import Counter
from typing import List, Optional
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from pydantic import BaseModel, Field
import nltk
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer, GPT2LMHeadModel, pipeline, AutoModelForSequenceClassification, DPRContextEncoder, DPRContextEncoderTokenizer, RagTokenizer, RagTokenForGeneration, RagRetriever
import torch
import faiss

# 定义对话类
class Dialogue(BaseModel):
    dialogue_id: str
    utterances: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    participants: List[str] = []

# 知识库类
class KnowledgeBase:
    def __init__(self):
        self.dialogues = []
        self.load_dialogues()

    def add_dialogue(self, dialogue: Dialogue):
        self.dialogues.append(dialogue)
        self.save_dialogues()

    def save_dialogues(self):
        with open('dialogues.json', 'w') as f:
            json.dump([d.dict() for d in self.dialogues], f, indent=4, default=str)

    def load_dialogues(self):
        try:
            with open('dialogues.json', 'r') as f:
                self.dialogues = [Dialogue(**d) for d in json.load(f)]
        except (FileNotFoundError, json.JSONDecodeError):
            self.dialogues = []

# 文本工具类
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
        return word_tokenize(text)

# 对话生成类
class DialogueGenerator:
    def __init__(self, model_name: str, tokenizer_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)

    def generate_dialogue_japanese(self, prompt: str, max_length: int = 50) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            repetition_penalty=1.2,
            temperature=0.7
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# 可视化类
class Visualization:
    @staticmethod
    def visualize_data(frequencies):
        words, counts = zip(*frequencies)
        plt.bar(words, counts)
        plt.show()

def main():
    nltk.download('punkt')

    kb = KnowledgeBase()
    text_utils = TextUtilities()
    generator = DialogueGenerator("gpt2", "gpt2")
    vis = Visualization()

    # 生成对话
    prompt = "今日は天気がいいから、"
    generated_dialogue = generator.generate_dialogue_japanese(prompt)
    print("Generated Dialogue:", generated_dialogue)

    # 处理文本
    raw_asr_output = "今日の天気は素晴らしいですね。公園に散歩に行きましょう。"
    processed_text = text_utils.process_asr_output(raw_asr_output)
    utterances = text_utils.split_japanese_text(processed_text)
    dialogue_id = f"dlg_{random.randint(1000, 9999)}"
    new_dialogue = Dialogue(dialogue_id=dialogue_id, utterances=utterances)
    kb.add_dialogue(new_dialogue)

    # 设置 RAG 系统
    dpr_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-multiset-base")
    dpr_model = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-multiset-base")

    # 向量化文本
    encoded_texts = dpr_tokenizer([generated_dialogue], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        embeddings = dpr_model(**encoded_texts).pooler_output

    # 创建 FAISS 索引
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.numpy())

    # 初始化 RAG 模型和检索器
    rag_tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
    rag_model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq")
    retriever = RagRetriever.from_pretrained("facebook/rag-token-nq", index_name="custom", index_path="my_custom_index.faiss", passages_path="my_knowledge_dataset")

    # 将检索器与模型集成
    rag_model.set_retriever(retriever)

    # 问答函数
    question = "映画のストーリーは何ですか？"
    inputs = rag_tokenizer(question, return_tensors="pt")
    with torch.no_grad():
        outputs = rag_model.generate(**inputs, num_return_sequences=1)
    answer = rag_tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Answer:", answer)

import os

# 设置你的文件路径
passages_path = "/home/user/project/data/my_knowledge_dataset"
index_path = "/home/user/project/data/my_custom_index.faiss"

# 检查文件是否存在
assert os.path.exists(passages_path), f"Passages file not found at {passages_path}"
assert os.path.exists(index_path), f"Index file not found at {index_path}"

print("Both files are correctly located.")


if __name__ == "__main__":
    main()

# generator.py

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, LongformerTokenizer, LongformerModel, CLIPProcessor, CLIPModel, BertJapaneseTokenizer, BertForMaskedLM
from PIL import Image
import re
from janome.tokenizer import Tokenizer as JanomeTokenizer

class DialogueGenerator:
    def __init__(self):
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.longformer_tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
        self.longformer_model = LongformerModel.from_pretrained('allenai/longformer-base-4096')
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.bert_tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese')
        self.bert_model = BertForMaskedLM.from_pretrained('cl-tohoku/bert-base-japanese')
        self.tokenizer = JanomeTokenizer()

    def preprocess_text(self, text):
        text = re.sub(r'[^\w\s]', '', text)
        tokens = self.tokenizer.tokenize(text, wakati=True)
        return tokens

    def generate_conversation(self, prompt, max_turns=5):
        input_ids = self.gpt2_tokenizer.encode(prompt, return_tensors='pt')
        outputs = self.gpt2_model.generate(input_ids, max_length=100, num_return_sequences=1)
        return self.gpt2_tokenizer.decode(outputs[0], skip_special_tokens=True)

    def process_multimodal_input(self, text, image_path):
        # Process text
        text_inputs = self.gpt2_tokenizer(text, return_tensors='pt')
        text_outputs = self.gpt2_model.generate(text_inputs['input_ids'], max_length=100)
        text_response = self.gpt2_tokenizer.decode(text_outputs[0], skip_special_tokens=True)

        # Process image
        image = Image.open(image_path)
        image_inputs = self.clip_processor(images=image, return_tensors='pt')
        image_outputs = self.clip_model.get_image_features(**image_inputs)

        return text_response, image_outputs.tolist()

    def generate_long_text(self, prompt, max_length=4096):
        input_ids = self.longformer_tokenizer.encode(prompt, return_tensors='pt')
        outputs = self.longformer_model(input_ids)
        last_hidden_states = outputs.last_hidden_state
        return last_hidden_states

    def process_asr_output(self, raw_asr_text):
        processed_text = raw_asr_text.replace("\n", " ").strip()
        tokens = self.preprocess_text(processed_text)
        return ' '.join(tokens)

# 使用示例
if __name__ == "__main__":
    generator = DialogueGenerator()
    
    # 示例对话生成
    prompt = "こんにちは、お元気ですか？"
    response = generator.generate_conversation(prompt)
    print(f"Generated conversation: {response}")
    
    # 示例多模态处理
    text = "これはテキストです。"
    image_path = "path_to_image.jpg"
    text_response, image_features = generator.process_multimodal_input(text, image_path)
    print(f"Text response: {text_response}")
    print(f"Image features: {image_features}")

    # 示例ASR处理
    raw_asr_text = "こんにちは\nお元気ですか？"
    processed_text = generator.process_asr_output(raw_asr_text)
    print(f"Processed ASR text: {processed_text}")
