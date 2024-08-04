# amzn_comprehend.py

import boto3

def analyze_text_with_comprehend(text):
    comprehend = boto3.client('comprehend')
    response = comprehend.detect_sentiment(
        Text=text,
        LanguageCode='ja'
    )
    print("文本情感分析结果:", response)
    return response
