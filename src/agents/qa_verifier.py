# qa_verifier.py

import random
import io
from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types
from fugashi import Tagger as FugashiTagger
from janome.tokenizer import Tokenizer as JanomeTokenizer

class QAProcessing:
    def __init__(self, audio_file_path=None):
        self.audio_file_path = audio_file_path
        self.client = speech.SpeechClient() if audio_file_path else None

    def verify_fact(self, question, answer):
        # 这里我们随机返回 True 或 False 来模拟验证过程
        # 实际应用中应该替换为基于事实核实的代码
        return random.choice([True, False])

    def remove_duplicates_and_verify(self, questions_and_answers):
        # 去除重复的问答对
        unique_questions_and_answers = list(set(questions_and_answers))

        # 验证问答对
        verified_questions_and_answers = [(q, a) for q, a in unique_questions_and_answers if self.verify_fact(q, a)]

        # 如果删除了不正确的问答对后数量减少了，我们需要从剩余的问答对中随机选择补充
        while len(verified_questions_and_answers) < len(questions_and_answers):
            q, a = random.choice(questions_and_answers)
            if (q, a) not in verified_questions_and_answers:
                verified_questions_and_answers.append((q, a))

        # 确保最终数量和原始数量一致
        assert len(verified_questions_and_answers) == len(questions_and_answers)

        # 打乱顺序以去除可能的偏差
        random.shuffle(verified_questions_and_answers)

        return verified_questions_and_answers

    def transcribe_audio(self):
        if not self.audio_file_path:
            raise ValueError("No audio file path provided.")

        # 从本地文件加载音频
        with io.open(self.audio_file_path, 'rb') as audio_file:
            content = audio_file.read()
            audio = types.RecognitionAudio(content=content)

        # 设置识别配置
        config = types.RecognitionConfig(
            encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code='en-US'
        )

        # 调用Google Speech-to-Text API进行语音识别
        response = self.client.recognize(config=config, audio=audio)

        transcripts = [result.alternatives[0].transcript for result in response.results]
        return transcripts

    @staticmethod
    def split_japanese_text(text):
        sentences = text.split("。")
        return [sentence.strip() + "。" for sentence in sentences if sentence.strip()]

    @staticmethod
    def process_asr_output(raw_asr_text):
        return raw_asr_text.replace("\n", " ").strip()

    @staticmethod
    def tokenize_text(text):
        tagger = FugashiTagger('-Owakati')
        return tagger.parse(text).strip().split()

    @staticmethod
    def tokenize_japanese_janome(text):
        tokenizer = JanomeTokenizer()
        return list(tokenizer.tokenize(text, wakati=True))

# 示例用法
if __name__ == "__main__":
    # 初始化QA处理类
    qa_processor = QAProcessing(audio_file_path='audio_file.wav')

    # 示例问答对列表
    base_questions_and_answers = [
        # 示例问答对 (q, a)
        ("質問1", "答え1"),
        ("質問2", "答え2"),
        ("質問1", "答え1"),  # 重复的问答对
    ]

    # 处理问答对去重与验证
    verified_questions_and_answers = qa_processor.remove_duplicates_and_verify(base_questions_and_answers)
    print("Verified Questions and Answers:", verified_questions_and_answers)

    # 语音转录
    transcripts = qa_processor.transcribe_audio()
    print("Transcripts:", transcripts)

    # 示例文本处理
    japanese_text = "これは素晴らしい経験でした。サービスがとても良かったです。"
    sentences = QAProcessing.split_japanese_text(japanese_text)
    print("Sentences:", sentences)

    # Tokenization
    tokens_fugashi = QAProcessing.tokenize_text(japanese_text)
    print("Fugashi Tokenization:", tokens_fugashi)

    tokens_janome = QAProcessing.tokenize_japanese_janome(japanese_text)
    print("Janome Tokenization:", tokens_janome)
