import json
import language_tool_python

# 初始化 LanguageTool，用于中文语法检查
tool = language_tool_python.LanguageTool('zh-CN')

# 从 data.json 文件中加载数据
with open('data.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 修正语法错误
for item in data:
    # 使用 LanguageTool 进行语法检查和修正
    matches = tool.check(item['text'])
    corrected_text = language_tool_python.utils.correct(item['text'], matches)
    item['text'] = corrected_text

# 将修正后的数据保存到新的 JSON 文件中
with open('corrected_data.json', 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)

print("语法修正完成，结果已保存到 corrected_data.json 文件中。")

import json
import language_tool_python

# 初始化 LanguageTool，用于中文语法检查
tool = language_tool_python.LanguageTool('zh-CN')

# 从 data.json 文件中加载数据
with open('data.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 修正语法错误
for item in data:
    # 使用 LanguageTool 进行语法检查和修正
    matches = tool.check(item['text'])
    corrected_text = language_tool_python.utils.correct(item['text'], matches)
    item['text'] = corrected_text

# 将修正后的数据保存到新的 JSON 文件中
with open('corrected_data.json', 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)

print("语法修正完成，结果已保存到 corrected_data.json 文件中。")


# aligner.py

import re
import torch
from transformers import BertJapaneseTokenizer, BertForTokenClassification
from fugashi import Tagger as FugashiTagger
from janome.tokenizer import Tokenizer as JanomeTokenizer
from pyknp import Juman

class JapaneseGrammarAligner:
    def __init__(self):
        self.bert_tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese')
        self.bert_model = BertForTokenClassification.from_pretrained('cl-tohoku/bert-base-japanese')
        self.fugashi_tagger = FugashiTagger()
        self.janome_tokenizer = JanomeTokenizer()
        self.jumanpp = Juman()

    def preprocess_text(self, text):
        # 移除标点符号
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def bert_tokenize(self, text):
        # 使用BERT模型进行分词
        inputs = self.bert_tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            outputs = self.bert_model(**inputs).logits
        predictions = torch.argmax(outputs, dim=2)
        tokens = self.bert_tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        return tokens, predictions[0].tolist()

    def fugashi_tokenize(self, text):
        # 使用fugashi进行分词
        return [word.surface for word in self.fugashi_tagger(text)]

    def janome_tokenize(self, text):
        # 使用janome进行分词
        return [token.surface for token in self.janome_tokenizer.tokenize(text)]

    def juman_tokenize(self, text):
        # 使用Juman++进行分词
        result = self.jumanpp.analysis(text)
        return [(mrph.midasi, mrph.yomi, mrph.genkei) for mrph in result.mrph_list()]

    def align_grammar(self, text):
        # 对齐语法结构
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

# 使用示例
if __name__ == "__main__":
    aligner = JapaneseGrammarAligner()
    text = "これはテスト文です。"
    alignment = aligner.align_grammar(text)
    print(alignment)

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
