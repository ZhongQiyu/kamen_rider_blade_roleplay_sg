# tokenizer.py

import ssl
import nltk
import json
import random
from datetime import datetime
from collections import Counter
from typing import List, Optional
from nltk.tokenize import word_tokenize
from janome.tokenizer import Tokenizer
from pydantic import Field, BaseModel

# 定义一个基本的对话数据模型
class Dialogue(BaseModel):
    dialogue_id: str
    utterances: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    participants: List[str] = []

    @property
    def number_of_utterances(self):
        return len(self.utterances)

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

# 日语的句子分割函数
def split_japanese_text(text: str) -> list:
    sentences = text.split("。")
    sentences = [sentence.strip() + "。" for sentence in sentences if sentence.strip()]
    if sentences:
        sentences[-1] = sentences[-1].strip("。")
    return sentences

# 使用Janome进行日语文本分词
def tokenize_japanese_text(text: str) -> list:
    t = Tokenizer()
    tokens = list(t.tokenize(text, wakati=True))
    return tokens

# 模拟ASR处理函数
def process_asr_output(raw_asr_text: str) -> str:
    return raw_asr_text.replace("\n", " ").strip()

# 生成对话数据
def generate_dialogue(asr_text: str) -> Dialogue:
    utterances = split_japanese_text(asr_text)
    dialogue_id = f"dlg_{random.randint(1000, 9999)}"
    return Dialogue(dialogue_id=dialogue_id, utterances=utterances)

# 日语词频分析函数
def analyze_frequency(dialogues: List[str]):
    word_counts = Counter()
    for dialogue in dialogues:
        words = tokenize_japanese_text(dialogue)
        word_counts.update(words)
    return word_counts.most_common(10)

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
    sample_text = "こんにちは、今日はいい天気ですね。公園に行きましょう。"
    tokens = tokenize_japanese_text(sample_text)
    print("Tokenized Text:", tokens)

    # 假设这是ASR的输出
    raw_asr_output = "今日の天気は素晴らしいですね。公園に散歩に行きましょう。あの花を見ましたか？とても綺麗です。"
    processed_text = process_asr_output(raw_asr_output)
    dialogue = generate_dialogue(processed_text)
    kb.add_dialogue(dialogue)
    retrieved_dialogue = kb.get_dialogue(dialogue.dialogue_id)
    if retrieved_dialogue:
        print("Retrieved Dialogue:", retrieved_dialogue.dict())

    # 示例对话数据
    dialogues = [
        "今日は天気がいいから、公園に行きましょう。",
        "明日は雨が降るそうです。",
        "週末に映画を見に行く予定です。",
        "最近、仕事が忙しいですね。"
    ]
    frequencies = analyze_frequency(dialogues)
    print("Top 10 word frequencies:", frequencies)

if __name__ == "__main__":
    main()
