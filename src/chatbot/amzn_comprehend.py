# amzn_comprehend.py

import boto3

class AWSAnalyzer:
    def __init__(self):
        self.comprehend = boto3.client('comprehend')
        self.rekognition = boto3.client('rekognition')
        self.sagemaker_runtime = boto3.client('sagemaker-runtime')

    # 文本情感分析
    def analyze_text_with_comprehend(self, text):
        response = self.comprehend.detect_sentiment(
            Text=text,
            LanguageCode='ja'
        )
        print("文本情感分析结果:", response)
        return response

    # 图像标签分析
    def analyze_image_with_rekognition(self, image_bytes):
        response = self.rekognition.detect_labels(
            Image={'Bytes': image_bytes},
            MaxLabels=10,
            MinConfidence=80
        )
        print("图像标签分析结果:", response)
        return response

    # SageMaker模型推理
    def invoke_sagemaker_endpoint(self, endpoint_name, payload):
        response = self.sagemaker_runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='application/json',
            Body=payload
        )
        result = response['Body'].read().decode('utf-8')
        print("SageMaker模型推理结果:", result)
        return result

    # 关键短语提取
    def detect_key_phrases(self, text):
        response = self.comprehend.detect_key_phrases(
            Text=text,
            LanguageCode='ja'
        )
        print("关键短语提取结果:", response)
        return response

    # 实体识别
    def detect_entities(self, text):
        response = self.comprehend.detect_entities(
            Text=text,
            LanguageCode='ja'
        )
        print("实体识别结果:", response)
        return response

    # 语言检测
    def detect_language(self, text):
        response = self.comprehend.detect_dominant_language(
            Text=text
        )
        print("语言检测结果:", response)
        return response

    # 文本分类
    def classify_text(self, text, endpoint_arn):
        response = self.comprehend.classify_document(
            Text=text,
            EndpointArn=endpoint_arn
        )
        print("文本分类结果:", response)
        return response

# Example usage:
if __name__ == "__main__":
    analyzer = AWSAnalyzer()

    # Comprehend - 情感分析
    text = "これは素晴らしい映画です"
    comprehend_result = analyzer.analyze_text_with_comprehend(text)

    # Comprehend - 关键短语提取
    key_phrases_result = analyzer.detect_key_phrases(text)

    # Comprehend - 实体识别
    entities_result = analyzer.detect_entities(text)

    # Comprehend - 语言检测
    language_result = analyzer.detect_language(text)

    # Comprehend - 文本分类（使用自定义模型）
    endpoint_arn = 'arn:aws:comprehend:us-west-2:123456789012:document-classifier-endpoint/your-endpoint'
    classification_result = analyzer.classify_text(text, endpoint_arn)

    # Rekognition - 图像分析
    with open('image.jpg', 'rb') as image_file:
        image_bytes = image_file.read()
    rekognition_result = analyzer.analyze_image_with_rekognition(image_bytes)

    # SageMaker - 模型推理
    sagemaker_result = analyzer.invoke_sagemaker_endpoint(endpoint_name='your-sagemaker-endpoint', payload='{"instances": [{"data": "example data"}]}')
