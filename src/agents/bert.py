# bert_chatbot.py

import torch
from transformers import BertTokenizer, BertForSequenceClassification
import MeCab
from utils import mecab_tokenize, calculate_metrics

# 初始化BERT模型和tokenizer
tokenizer = BertTokenizer.from_pretrained('cl-tohoku/bert-base-japanese')
model = BertForSequenceClassification.from_pretrained('cl-tohoku/bert-base-japanese', num_labels=2)

# 对话生成函数
def generate_response(question):
    inputs = tokenizer(mecab_tokenize(question), return_tensors='pt', truncation=True, padding=True)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits).item()
    return "Positive" if predicted_class_id == 1 else "Negative"

# 示例对话
if __name__ == "__main__":
    question = "今日は元気ですか？"
    response = generate_response(question)
    print(f"User: {question}")
    print(f"Bot: {response}")

    # 评估指标示例
    reference = "今日は元気ですか？"
    candidate = response
    metrics = calculate_metrics(reference, candidate)
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
