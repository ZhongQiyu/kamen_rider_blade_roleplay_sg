# sentiment_analysis.py

# coding: utf-8
from __future__ import unicode_literals
import sqlite3
from datetime import datetime
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from pyknp import Juman

class SentimentAnalysis:
    def __init__(self, language='zh', db_path='sentiment_analysis.db'):
        self.language = language
        self.db_path = db_path
        self._load_model()
        self._init_db()
        self.jumanpp = None
        if language == 'ja':
            self.jumanpp = Juman()  # 初始化Juman++分词器

    def _load_model(self):
        # 加载不同语言的模型和分词器
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

    def _init_db(self):
        # 初始化数据库并创建表格
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS SentimentAnalysis
                          (id INTEGER PRIMARY KEY AUTOINCREMENT,
                           text TEXT NOT NULL,
                           language TEXT NOT NULL,
                           sentiment_score REAL NOT NULL,
                           timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
        conn.commit()
        conn.close()

    def analyze_sentiment(self, text):
        # 如果是日语文本，使用Juman++进行形态素分析
        if self.language == 'ja' and self.jumanpp:
            result = self.jumanpp.analysis(text)
            print("Juman++ 分析结果:")
            for mrph in result.mrph_list():
                print("见出词:%s, 读音:%s, 原形:%s, 词性:%s, 词性细分类:%s, 活用型:%s, 活用形:%s, 意味信息:%s, 代表表记:%s" \
                    % (mrph.midasi, mrph.yomi, mrph.genkei, mrph.hinsi, mrph.bunrui, mrph.katuyou1, mrph.katuyou2, mrph.imis, mrph.repname))

        # 分析情感并返回得分
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            sentiment_score = predictions[0][1].item()  # 假设1表示积极情感
        return sentiment_score

    def save_to_db(self, text, sentiment_score):
        # 将分析结果保存到数据库
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO SentimentAnalysis (text, language, sentiment_score) VALUES (?, ?, ?)",
                       (text, self.language, sentiment_score))
        conn.commit()
        conn.close()

    def get_sentiments_above_threshold(self, threshold=0.5):
        # 查询情感得分高于阈值的记录
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM SentimentAnalysis WHERE sentiment_score > ? AND language = ?",
                       (threshold, self.language))
        rows = cursor.fetchall()
        conn.close()
        return rows

# 使用示例
if __name__ == '__main__':
    # 中文情感分析
    sentiment_analyzer_zh = SentimentAnalysis(language='zh')
    text_zh = "我今天很高兴"
    score_zh = sentiment_analyzer_zh.analyze_sentiment(text_zh)
    sentiment_analyzer_zh.save_to_db(text_zh, score_zh)
    print(f"中文情感得分: {score_zh}")

    # 英文情感分析
    sentiment_analyzer_en = SentimentAnalysis(language='en')
    text_en = "I am very happy today"
    score_en = sentiment_analyzer_en.analyze_sentiment(text_en)
    sentiment_analyzer_en.save_to_db(text_en, score_en)
    print(f"英文情感得分: {score_en}")

    # 日文情感分析
    sentiment_analyzer_ja = SentimentAnalysis(language='ja')
    text_ja = "今日はとても嬉しいです"
    score_ja = sentiment_analyzer_ja.analyze_sentiment(text_ja)
    sentiment_analyzer_ja.save_to_db(text_ja, score_ja)
    print(f"日文情感得分: {score_ja}")

    # 查询中文情感得分高于0.5的记录
    results_zh = sentiment_analyzer_zh.get_sentiments_above_threshold(0.5)
    for result in results_zh:
        print(result)
