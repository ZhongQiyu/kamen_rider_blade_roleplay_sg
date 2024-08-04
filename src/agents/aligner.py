# aligner.py

import re
import os
import json
import torch
import subprocess
from transformers import BertJapaneseTokenizer, BertForTokenClassification, BertForMaskedLM, AutoTokenizer, RagTokenizer, RagRetriever, RagSequenceForGeneration, T5Tokenizer, T5Model
from sentence_transformers import SentenceTransformer
from fugashi import Tagger as FugashiTagger
from janome.tokenizer import Tokenizer as JanomeTokenizer
from typing import List
from pyknp import Juman

class JapaneseGrammarAligner:
    def __init__(self):
        self.bert_tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese')
        self.bert_model = BertForTokenClassification.from_pretrained('cl-tohoku/bert-base-japanese')
        self.fugashi_tagger = FugashiTagger()
        self.janome_tokenizer = JanomeTokenizer()
        self.jumanpp = Juman()

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

    def juman_tokenize(self, text):
        result = self.jumanpp.analysis(text)
        return [(mrph.midasi, mrph.yomi, mrph.genkei) for mrph in result.mrph_list()]

    def align_grammar(self, text):
        processed_text = self.preprocess_text(text)
        bert_tokens, bert_predictions = self.bert_tokenize(processed_text)
        fugashi_tokens = self.fugashi_tokenize(processed_text)
        janome_tokens = self.janome_tokenize(processed_text)
        juman_tokens = self.juman_tokenize(processed_text)
        
        alignment = {
            "original_text": text,
            "processed_text": processed_text,
            "bert_tokens": bert_tokens,
            "bert_predictions": bert_predictions,
            "fugashi_tokens": fugashi_tokens,
            "janome_tokens": janome_tokens,
            "juman_tokens": juman_tokens
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

def load_model(model_name="cl-tohoku/bert-base-japanese"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = BertForMaskedLM.from_pretrained(model_name)
    return tokenizer, model

def correct_text(text, tokenizer, model, device='cpu'):
    model.eval()
    model.to(device)

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

def run_command(command):
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def extract_frames(video_path, frames_folder, fps=1):
    if not os.path.exists(frames_folder):
        os.makedirs(frames_folder)
    command = ['ffmpeg', '-i', video_path, '-vf', f'fps={fps}', f'{frames_folder}/frame_%04d.png']
    run_command(command)

def extract_audio(video_path, audio_output):
    command = ['ffmpeg', '-i', video_path, '-q:a', '0', '-map', 'a', audio_output]
    run_command(command)

def load_additional_dataset(dataset_path):
    with open(dataset_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_dataset(frames_folder, audio_file, text_data, additional_data, dataset_output):
    tokenized_text_data = [tokenize_japanese(sentence) for sentence in text_data]
    combined_data = {
        'frames': [os.path.join(frames_folder, frame) for frame in os.listdir(frames_folder) if frame.endswith('.png')],
        'audio': audio_file,
        'text': tokenized_text_data,
        'additional_data': additional_data
    }
    
    with open(dataset_output, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, ensure_ascii=False, indent=4)

def main():
    asr_text = "ここで間違ったテキストが入力されます"  # 示例文本
    tokenizer, model = load_model()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    corrected_text = correct_text(asr_text, tokenizer, model, device)
    print("Original Text:", asr_text)
    print("Corrected Text:", corrected_text)

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

if __name__ == "__main__":
    main()
