# sentiment_analysis.py

import sqlite3
from transformers import BertTokenizer, BertForSequenceClassification
import torch

class SentimentAnalysis:
    def __init__(self, language='zh', db_path='sentiment_analysis.db'):
        self.language = language
        self.db_path = db_path
        self._load_model()

    def _load_model(self):
        if self.language == 'zh':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
            self.model = BertForSequenceClassification.from_pretrained('bert-base-chinese')
        elif self.language == 'en':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        elif self.language == 'ja':
            self.tokenizer = BertTokenizer.from_pretrained('cl-tohoku/bert-base-japanese')
            self.model = BertForSequenceClassification.from_pretrained('cl-tohoku/bert-base-japanese')
        else:
            raise ValueError("不支持的语言。请选择 'zh', 'en', 或 'ja'。")

    def analyze_sentiment(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            sentiment_score = predictions[0][1].item()  # 假设1表示积极情感
        return sentiment_score

    def save_to_db(self, text, sentiment_score):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS SentimentAnalysis
                          (id INTEGER PRIMARY KEY AUTOINCREMENT,
                           text TEXT,
                           sentiment_score REAL)''')
        cursor.execute("INSERT INTO SentimentAnalysis (text, sentiment_score) VALUES (?, ?)",
                       (text, sentiment_score))
        conn.commit()
        conn.close()

    def get_sentiments_above_threshold(self, threshold=0.5):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM SentimentAnalysis WHERE sentiment_score > ?", (threshold,))
        rows = cursor.fetchall()
        conn.close()
        return rows

# 使用示例：
if __name__ == '__main__':
    # 创建一个中文情感分析对象
    sentiment_analyzer = SentimentAnalysis(language='zh')

    # 进行情感分析
    text = "我今天很高兴"
    score = sentiment_analyzer.analyze_sentiment(text)
    print(f"情感得分: {score}")

    # 将结果保存到数据库
    sentiment_analyzer.save_to_db(text, score)

    # 查询情感得分高于0.5的记录
    results = sentiment_analyzer.get_sentiments_above_threshold(0.5)
    for result in results:
        print(result)
