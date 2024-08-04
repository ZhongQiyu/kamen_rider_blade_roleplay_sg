# aligner.py

# coding: utf-8
from __future__ import unicode_literals
from pyknp import Juman

jumanpp = Juman()
result = jumanpp.analysis("下鴨神社の参道は暗かった。")

for mrph in result.mrph_list(): # 各形態素にアクセス
    print("見出し:%s, 読み:%s, 原形:%s, 品詞:%s, 品詞細分類:%s, 活用型:%s, 活用形:%s, 意味情報:%s, 代表表記:%s" \
            % (mrph.midasi, mrph.yomi, mrph.genkei, mrph.hinsi, mrph.bunrui, mrph.katuyou1, mrph.katuyou2, mrph.imis, mrph.repname))# coding: utf-8

import torch
from transformers import BertTokenizer, BertForMaskedLM, AutoTokenizer

def load_model(model_name="cl-tohoku/bert-base-japanese"):
    # 使用AutoTokenizer自动选择正确的分词器类
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = BertForMaskedLM.from_pretrained(model_name)
    return tokenizer, model

def correct_text(text, tokenizer, model, device='cpu'):  # 默认使用CPU
    model.eval()
    model.to(device)

    # 示例：简单的纠错方法（仅示例，需要根据实际情况调整）
    masked_text = text.replace("間違った", "[MASK]")
    encoded_input = tokenizer(masked_text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**encoded_input)
        predictions = outputs.logits

    masked_index = torch.where(encoded_input["input_ids"] == tokenizer.mask_token_id)[1]
    predicted_id = predictions[0, masked_index].argmax(dim=-1)
    predicted_token = tokenizer.decode(predicted_id).strip()

    corrected_text = masked_text.replace("[MASK]", predicted_token)
    return corrected_text

def main():
    asr_text = "ここで間違ったテキストが入力されます"  # 示例文本
    tokenizer, model = load_model()

    # 确定运行设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    corrected_text = correct_text(asr_text, tokenizer, model, device)
    print("Original Text:", asr_text)
    print("Corrected Text:", corrected_text)

if __name__ == "__main__":
    main()

import json

json_path = "transcripts_output_transcript_68c650f0-0000-26af-b8d3-2405887b1c1c.json"

# 加载JSON数据
with open(json_path, 'r') as f:
    data = json.load(f)

# 从JSON数据中提取信息
for result in data['results']:
    for alternative in result['alternatives']:
        transcript = alternative['transcript']
        confidence = alternative['confidence']
        print(f"Transcript: {transcript}")
        print(f"Confidence: {confidence}")

        # 如果有词时间偏移（word timing offsets）
        if 'words' in alternative:
            for word_info in alternative['words']:
                word = word_info['word']
                start_time = word_info['startTime']
                end_time = word_info['endTime']
                print(f"Word: {word}, start time: {start_time}, end time: {end_time}")
<<<<<<< HEAD

# 伪代码，需要实际的图像和音频处理代码
def create_dataset(frames_folder, audio_file):
    # process frames and audio
    # extract features
    # save to dataset
    pass

import subprocess
import os
import json
from fugashi import Tagger

# 功能：运行外部命令
def run_command(command):
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# 功能：提取帧
def extract_frames(video_path, frames_folder, fps=1):
    if not os.path.exists(frames_folder):
        os.makedirs(frames_folder)
    command = ['ffmpeg', '-i', video_path, '-vf', f'fps={fps}', f'{frames_folder}/frame_%04d.png']
    run_command(command)

# 功能：提取音频
def extract_audio(video_path, audio_output):
    command = ['ffmpeg', '-i', video_path, '-q:a', '0', '-map', 'a', audio_output]
    run_command(command)

# 功能：日文分词
def tokenize_japanese(text):
    tagger = Tagger('-Owakati')
    return tagger.parse(text).strip().split()

# 功能：加载内部或外部数据集
def load_additional_dataset(dataset_path):
    with open(dataset_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# 功能：创建并保存数据集
def create_dataset(frames_folder, audio_file, text_data, additional_data, dataset_output):
    tokenized_text_data = [tokenize_japanese(sentence) for sentence in text_data]
    combined_data = {
        'frames': [os.path.join(frames_folder, frame) for frame in os.listdir(frames_folder) if frame.endswith('.png')],
        'audio': audio_file,
        'text': tokenized_text_data,
        'additional_data': additional_data  # 加载的额外数据集
    }
    
    with open(dataset_output, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=4)

# 主逻辑
if __name__ == "__main__":
    video_file = 'path_to_your_video.mp4'
    frames_folder = 'path_to_frames_directory'
    audio_output = 'path_to_audio_output.wav'
    additional_dataset_path = 'internal_lm2.json'
    dataset_output = 'combined_dataset.json'
    
    # 这里我们提取现有文本数据
    text_data = ["こんにちは", "元気ですか", "さようなら"]
    additional_data = load_additional_dataset(additional_dataset_path)
    
    # 提取帧和音频
    extract_frames(video_file, frames_folder)
    extract_audio(video_file, audio_output)
    
    # 创建并保存数据集
    create_dataset(frames_folder, audio_output, text_data, additional_data, dataset_output)

=======
>>>>>>> origin/main

# aligner.py

import re
import json
import torch
from transformers import BertJapaneseTokenizer, BertForTokenClassification, RagTokenizer, RagRetriever, RagSequenceForGeneration, T5Tokenizer, T5Model
from sentence_transformers import SentenceTransformer
from fugashi import Tagger as FugashiTagger
from janome.tokenizer import Tokenizer as JanomeTokenizer
from typing import List

class JapaneseGrammarAligner:
    def __init__(self):
        self.bert_tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese')
        self.bert_model = BertForTokenClassification.from_pretrained('cl-tohoku/bert-base-japanese')
        self.fugashi_tagger = FugashiTagger()
        self.janome_tokenizer = JanomeTokenizer()

    def preprocess_text(self, text):
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        return text

    def bert_tokenize(self, text):
        inputs = self.bert_tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            outputs = self.bert_model(**inputs).logits
        predictions = torch.argmax(outputs, dim=2)
        tokens = self.bert_tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        return tokens, predictions[0].tolist()

    def fugashi_tokenize(self, text):
        return [word.surface for word in self.fugashi_tagger(text)]

    def janome_tokenize(self, text):
        return [token.surface for token in self.janome_tokenizer.tokenize(text)]

    def align_grammar(self, text):
        processed_text = self.preprocess_text(text)
        bert_tokens, bert_predictions = self.bert_tokenize(processed_text)
        fugashi_tokens = self.fugashi_tokenize(processed_text)
        janome_tokens = self.janome_tokenize(processed_text)
        
        alignment = {
            "original_text": text,
            "processed_text": processed_text,
            "bert_tokens": bert_tokens,
            "bert_predictions": bert_predictions,
            "fugashi_tokens": fugashi_tokens,
            "janome_tokens": janome_tokens
        }
        return alignment

class RAGDialogueGenerator:
    def __init__(self, retriever_model_name="facebook/dpr-ctx_encoder-multiset-base", rag_model_name="facebook/rag-sequence-nq"):
        self.retriever = SentenceTransformer(retriever_model_name)
        self.rag_tokenizer = RagTokenizer.from_pretrained(rag_model_name)
        self.rag_retriever = RagRetriever.from_pretrained(rag_model_name, index_name="exact", use_dummy_dataset=True)
        self.rag_model = RagSequenceForGeneration.from_pretrained(rag_model_name)

    def generate_response(self, question, context_documents):
        inputs = self.rag_tokenizer(question, return_tensors="pt")
        question_embeddings = self.retriever.encode([question], convert_to_tensor=True)
        
        # Perform retrieval using the context documents
        docs = self.rag_retriever(question_inputs=inputs['input_ids'], prefix_allowed_tokens_fn=None)
        
        # Generate response using RAG
        outputs = self.rag_model.generate(input_ids=inputs['input_ids'], context_input_ids=docs['context_input_ids'])
        response = self.rag_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        return response

    def retrieve_and_generate(self, query, context):
        context_embeddings = self.retriever.encode(context, convert_to_tensor=True)
        response = self.generate_response(query, context_embeddings)
        return response

class T5JapaneseEmbedder:
    def __init__(self, model_name="rinna/japanese-t5-small"):
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5Model.from_pretrained(model_name)

    def generate_embeddings(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.squeeze(0)
        return embeddings

    def align_embeddings(self, text, aligner):
        aligned_grammar = aligner.align_grammar(text)
        embeddings = self.generate_embeddings(text)
        aligned_grammar["embeddings"] = embeddings.tolist()  # Convert to list for JSON serialization
        return aligned_grammar

if __name__ == "__main__":
    # Example conversation from "Kamen Rider Blade"
    conversation = [
        "剣崎、一緒に戦おう！",
        "俺の運命は俺が決める！",
        "ああ、分かった。共に行こう！"
    ]

    context_documents = [
        "剣崎は決意を新たにした。",
        "彼は運命を自ら切り開くと誓った。",
        "仲間たちとの絆が深まった。"
    ]

    # Initialize aligner and generators
    aligner = JapaneseGrammarAligner()
    rag_generator = RAGDialogueGenerator()
    t5_embedder = T5JapaneseEmbedder()

    # Process conversation with RAG
    rag_aligned_conversation = []
    for line in conversation:
        alignment = aligner.align_grammar(line)
        response = rag_generator.retrieve_and_generate(line, context_documents)
        alignment["response"] = response
        rag_aligned_conversation.append(alignment)

    # Save RAG results to a JSON file
    with open('rag_aligned_conversation.json', 'w', encoding='utf-8') as f:
        json.dump(rag_aligned_conversation, f, ensure_ascii=False, indent=4)

    # Print the RAG aligned conversation
    for alignment in rag_aligned_conversation:
        print(json.dumps(alignment, ensure_ascii=False, indent=4))

    # Process conversation with T5 embeddings
    t5_aligned_conversation = []
    for line in conversation:
        alignment = t5_embedder.align_embeddings(line, aligner)
        t5_aligned_conversation.append(alignment)

    # Save T5 results to a JSON file
    with open('t5_aligned_conversation.json', 'w', encoding='utf-8') as f:
        json.dump(t5_aligned_conversation, f, ensure_ascii=False, indent=4)

    # Print the T5 aligned conversation
    for alignment in t5_aligned_conversation:
        print(json.dumps(alignment, ensure_ascii=False, indent=4))
