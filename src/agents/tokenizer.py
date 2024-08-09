# tokenizer.py

import ssl
import nltk
import json
import random
from datetime import datetime
from collections import Counter
from typing import List, Optional
from nltk.tokenize import word_tokenize
from janome.tokenizer import Tokenizer as JanomeTokenizer
from fugashi import Tagger as FugashiTagger
from pydantic import Field, BaseModel
from google.cloud import speech
from google.cloud.speech import enums, types
import io
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import torch

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

# 日语文本处理工具类
class TextUtilities:
    @staticmethod
    def split_japanese_text(text: str) -> list:
        sentences = text.split("。")
        sentences = [sentence.strip() + "。" for sentence in sentences if sentence.strip()]
        if sentences:
            sentences[-1] = sentences[-1].strip("。")
        return sentences

    @staticmethod
    def process_asr_output(raw_asr_text: str) -> str:
        return raw_asr_text.replace("\n", " ").strip()

    @staticmethod
    def tokenize_text(text: str) -> list:
        tagger = FugashiTagger('-Owakati')
        return tagger.parse(text).strip().split()

    @staticmethod
    def tokenize_japanese_janome(text: str) -> list:
        tokenizer = JanomeTokenizer()
        return list(tokenizer.tokenize(text, wakati=True))

# 生成对话数据
def generate_dialogue(asr_text: str) -> Dialogue:
    utterances = TextUtilities.split_japanese_text(asr_text)
    dialogue_id = f"dlg_{random.randint(1000, 9999)}"
    return Dialogue(dialogue_id=dialogue_id, utterances=utterances)

# 日语词频分析函数
def analyze_frequency(dialogues: List[str]):
    word_counts = Counter()
    for dialogue in dialogues:
        words = TextUtilities.tokenize_text(dialogue)
        word_counts.update(words)
    return word_counts.most_common(10)

# Google Speech-to-Text处理函数
def process_speech_to_text(audio_file_path: str) -> str:
    client = speech.SpeechClient()

    with io.open(audio_file_path, 'rb') as audio_file:
        content = audio_file.read()
        audio = types.RecognitionAudio(content=content)

    config = types.RecognitionConfig(
        encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code='en-US'
    )

    response = client.recognize(config=config, audio=audio)

    return ' '.join([result.alternatives[0].transcript for result in response.results])

# GPT-2角色对话数据微调
def fine_tune_gpt2(dialogs: dict, model_name='gpt2'):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    for role, lines in dialogs.items():
        with open(f"{role}_lines.txt", "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + tokenizer.eos_token)

        dataset = TextDataset(
            tokenizer=tokenizer,
            file_path=f"{role}_lines.txt",
            block_size=128,
        )

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False,
        )

        model = GPT2LMHeadModel.from_pretrained(model_name)

        training_args = TrainingArguments(
            output_dir=f'./{role}_finetuned_gpt2',
            overwrite_output_dir=True,
            num_train_epochs=5,
            per_device_train_batch_size=4,
            save_steps=10_000,
            save_total_limit=2,
            prediction_loss_only=True,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset,
        )

        trainer.train()
        model.save_pretrained(f'./{role}_finetuned_gpt2')

# 主函数
def main():
    # 禁用 SSL 证书验证
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    # 确保 NLTK 资源已下载
    nltk.download('punkt')

    # 测试知识库的代码
    kb = KnowledgeBase()

    # 示例文本
    sample_text = "こんにちは、今日はいい天気ですね。公園に行きましょう。"
    tokens = TextUtilities.tokenize_japanese_janome(sample_text)
    print("Tokenized Text:", tokens)

    # 假设这是ASR的输出
    raw_asr_output = "今日の天気は素晴らしいですね。公園に散歩に行きましょう。あの花を見ましたか？とても綺麗です。"
    processed_text = TextUtilities.process_asr_output(raw_asr_output)
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
