from janome.tokenizer import Tokenizer

t = Tokenizer()
tokens = list(t.tokenize(text, wakati=True))
print(tokens)

# dialogue_processor.py

import ssl
import nltk
import json
import random
from datetime import datetime
import matplotlib.pyplot as plt
from collections import Counter
from typing import List, Optional
from nltk.tokenize import word_tokenize
import matplotlib.font_manager as font_manager
from pydantic import Field, field_validator, BaseModel
from transformers import pipeline, AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer, AutoModelForSequenceClassification

# 定义一个基本的对话数据模型
class Dialogue(BaseModel):
    dialogue_id: str
    utterances: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    participants: List[str] = []

    @property
    def number_of_utterances(self):
        return len(self.utterances)

    @field_validator('tags')
    def check_tags(cls, v):
        if not isinstance(v, str):
            raise ValueError('每个标签必须是字符串')
        return v

class LanguageSettings(BaseModel):
    language_code: str
    welcome_message: str

class KnowledgeBase:
    def __init__(self):
        self.dialogues = []
        self.load_dialogues()

    def add_dialogue(self, dialogue: Dialogue):
        self.dialogues.append(dialogue)
        self.notify_observers(dialogue)
        self.save_dialogues()

    def notify_observers(self, dialogue: Dialogue):
        print(f"New dialogue added with ID: {dialogue.dialogue_id}")

    def dialogue_count(self) -> int:
        return len(self.dialogues)

    def search_dialogues(self, keyword: str) -> List[Dialogue]:
        return [d for d in self.dialogues if keyword in ' '.join(d.utterances)]

    def get_dialogues_by_participant(self, participant: str) -> List[Dialogue]:
        return [d for d in self.dialogues if participant in d.participants]

    def save_dialogues(self):
        with open('dialogues.json', 'w') as f:
            json.dump([d.dict() for d in self.dialogues], f, indent=4, default=str)

    def load_dialogues(self):
        try:
            with open('dialogues.json', 'r') as f:
                self.dialogues = [Dialogue(**d) for d in json.load(f)]
        except (FileNotFoundError, json.JSONDecodeError):
            self.dialogues = []  # 文件不存在或为空时初始化为空列表

    def get_dialogue(self, dialogue_id: str) -> Optional[Dialogue]:
        return next((d for d in self.dialogues if d.dialogue_id == dialogue_id), None)

    def average_utterances_per_dialogue(self) -> float:
        return sum(len(d.utterances) for d in self.dialogues) / len(self.dialogues) if self.dialogues else 0

def tokenize_text(text):
    tokens = word_tokenize(text)
    return tokens

# 日语的句子分割函数
def split_japanese_text(text: str) -> list:
    sentences = text.split("。")
    sentences = [sentence.strip() + "。" for sentence in sentences if sentence.strip()]
    if sentences:
        sentences[-1] = sentences[-1].strip("。")
    return sentences

# 使用GPT-2生成对话
def generate_dialogue_japanese(prompt: str, max_length: int = 50, tokenizer=None, model=None) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=1,
        repetition_penalty=1.2,
        temperature=0.7
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 模拟ASR处理函数
def process_asr_output(raw_asr_text: str) -> str:
    return raw_asr_text.replace("\n", " ").strip()

# 生成对话数据
def generate_dialogue(asr_text: str) -> Dialogue:
    utterances = split_japanese_text(asr_text)
    dialogue_id = f"dlg_{random.randint(1000, 9999)}"
    return Dialogue(dialogue_id=dialogue_id, utterances=utterances)

# 日语词频分析函数
def analyze_frequency(dialogues):
    word_counts = Counter()
    for dialogue in dialogues:
        words = word_tokenize(dialogue)
        word_counts.update(words)
    return word_counts.most_common(10)

# 日语对话可视化函数
def visualize_data(frequencies):
    # 设定字符及个数统计器
    words, counts = zip(*frequencies)
    plt.bar(words, counts)

    # 设置日语字体路径
    jp_font = FontProperties(fname='/path/to/your/japanese/font.ttf')

    # 使用日语字体绘图
    plt.figure()
    plt.title('日本語のタイトル', fontproperties=jp_font)
    plt.xlabel('X軸', fontproperties=jp_font)
    plt.ylabel('Y軸', fontproperties=jp_font)
    plt.show()

# 主函数
def main():
    # 测试知识库的代码
    kb = KnowledgeBase()

    # 禁用 SSL 证书验证
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    # 确保 NLTK 资源已下载
    nltk.download('punkt')

    # 示例文本
    sample_text = "Hello there, how are you doing today?"
    tokens = tokenize_text(sample_text)
    print("Tokenized Text:", tokens)

    # 设置预训练模型及其tokenizer
    model_name = "gpt2"  # 示例模型，可以替换为支持日语的模型
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()

    # 示例单句数据
    prompt = "今日は天気がいいから、"
    generated_dialogue = generate_dialogue_japanese(prompt, tokenizer=tokenizer, model=model)
    print("Generated Dialogue:", generated_dialogue)
    print("Split Sentences:", split_japanese_text(generated_dialogue))

    # 示例对话数据
    dialogues = [
        "今日は天気がいいから、公園に行きましょう。",
        "明日は雨が降るそうです。",
        "週末に映画を見に行く予定です。",
        "最近、仕事が忙しいですね。"
    ]
    frequencies = analyze_frequency(dialogues)
    visualize_data(frequencies)

    # 假设这是ASR的输出
    raw_asr_output = "今日の天気は素晴らしいですね。公園に散歩に行きましょう。あの花を見ましたか？とても綺麗です。"
    processed_text = process_asr_output(raw_asr_output)
    dialogue = generate_dialogue(processed_text)
    kb = KnowledgeBase()
    kb.add_dialogue(dialogue)
    retrieved_dialogue = kb.get_dialogue(dialogue.dialogue_id)
    if retrieved_dialogue:
        print("Retrieved Dialogue:", retrieved_dialogue)

    # 加载模型和tokenizer
    model_name = "jarvisx17/japanese-sentiment-analysis"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 创建情感分析pipeline
    sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    # 测试文本
    text = "この製品は本当に素晴らしいです！"
    result = sentiment_pipeline(text)
    print(result)

if __name__ == "__main__":
    main()
