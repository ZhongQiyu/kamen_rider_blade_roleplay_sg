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
from transformers import AutoTokenizer, GPT2LMHeadModel, pipeline, AutoModelForSequenceClassification

class Dialogue(BaseModel):
    dialogue_id: str
    utterances: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    participants: List[str] = []

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

    prompt = "今日は天気がいいから、"
    generated_dialogue = generator.generate_dialogue_japanese(prompt)
    print("Generated Dialogue:", generated_dialogue)

    raw_asr_output = "今日の天気は素晴らしいですね。公園に散歩に行きましょう。"
    processed_text = text_utils.process_asr_output(raw_asr_output)
    utterances = text_utils.split_japanese_text(processed_text)
    dialogue_id = f"dlg_{random.randint(1000, 9999)}"
    new_dialogue = Dialogue(dialogue_id=dialogue_id, utterances=utterances)
    kb.add_dialogue(new_dialogue)

if __name__ == "__main__":
    main()
