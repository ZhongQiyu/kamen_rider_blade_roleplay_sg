# text_utilities.py

from fugashi import Tagger as FugashiTagger
from janome.tokenizer import Tokenizer as JanomeTokenizer

class TextUtilities:
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
