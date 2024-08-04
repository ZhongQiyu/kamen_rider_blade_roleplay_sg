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
