# rag.py

import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification, BertModel, RagTokenizer, RagRetriever, RagSequenceForGeneration, Trainer, TrainingArguments
from datasets import Dataset, load_metric
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class TextProcessor:
    def __init__(self, data):
        self.data = data
        self.dataset = Dataset.from_dict(data)
        self.tokenizer = BertTokenizer.from_pretrained('cl-tohoku/bert-base-japanese')
    
    def preprocess(self):
        return self.dataset.map(self._preprocess_function, batched=True)

    def _preprocess_function(self, examples):
        return self.tokenizer(examples['texts'], truncation=True, padding=True)

class BERTTrainer:
    def __init__(self, encoded_dataset, num_labels=2, num_epochs=3):
        self.tokenizer = BertTokenizer.from_pretrained('cl-tohoku/bert-base-japanese')
        self.model = BertForSequenceClassification.from_pretrained('cl-tohoku/bert-base-japanese', num_labels=num_labels)
        self.encoded_dataset = encoded_dataset
        self.num_epochs = num_epochs

    def train(self):
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=4,
            warmup_steps=10,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
        )

        metric = load_metric("accuracy")
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.encoded_dataset,
            eval_dataset=self.encoded_dataset,
            compute_metrics=self.compute_metrics,
        )
        trainer.train()

    def compute_metrics(self, p):
        return load_metric("accuracy").compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)

class Retriever:
    def __init__(self, model, tokenizer, dataset, window_size=1):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.window_size = window_size
        self.dataset_embeddings = self.create_embeddings(dataset['texts'])

    def create_embeddings(self, texts):
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model.bert(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings

    def retrieve_with_window(self, query):
        query_embedding = self.create_embeddings([query])[0]
        similarities = torch.matmul(self.dataset_embeddings, query_embedding)
        top_indices = torch.topk(similarities, k=1).indices
        
        results = []
        for idx in top_indices:
            idx = idx.item()
            start_idx = max(0, idx - self.window_size)
            end_idx = min(len(self.dataset) - 1, idx + self.window_size)
            results.extend(self.dataset['texts'][start_idx:end_idx + 1])
        return results

class AnswerGenerator:
    def __init__(self):
        self.rag_tokenizer = RagTokenizer.from_pretrained('facebook/rag-sequence-nq')
        self.rag_model = RagSequenceForGeneration.from_pretrained('facebook/rag-sequence-nq')
        self.retriever = RagRetriever.from_pretrained('facebook/rag-sequence-nq', index_name='exact', passages_path=None)

    def generate_answer(self, query, retrieved_texts):
        context = " ".join(retrieved_texts)
        inputs = self.rag_tokenizer.prepare_seq2seq_batch([query], context=[context], return_tensors='pt')
        outputs = self.rag_model.generate(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'], context_input_ids=inputs['context_input_ids'])
        answer = self.rag_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return answer

def main():
    # 示例日语对话文本数据集
    data = {
        "texts": [
            "これは情報検索タスクのための例文です。",
            "次の文を探しています。",
            "前の文と次の文の両方が必要です。",
            "これが私のリクエストです。",
            "情報検索は面白い分野です。"
        ],
        "labels": [0, 1, 1, 0, 0]
    }

    # 数据预处理
    text_processor = TextProcessor(data)
    encoded_dataset = text_processor.preprocess()

    # 训练BERT模型
    bert_trainer = BERTTrainer(encoded_dataset)
    bert_trainer.train()

    # 检索功能
    retriever = Retriever(bert_trainer.model, bert_trainer.tokenizer, text_processor.dataset)
    query = "情報検索タスクに関する情報が必要です。"
    retrieved_texts = retriever.retrieve_with_window(query)

    print("Retrieved Texts with Context Window:")
    for text in retrieved_texts:
        print(text)

    # 生成答案
    answer_generator = AnswerGenerator()
    answer = answer_generator.generate_answer(query, retrieved_texts)
    print("Generated Answer:", answer)

if __name__ == "__main__":
    main()

