# metrics.py

import MeCab
from jiwer import cer
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from sentence_transformers import SentenceTransformer, util

class TextMetricsEvaluator:
    def __init__(self):
        self.mecab = MeCab.Tagger("-Owakati")
        self.sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    def mecab_tokenize(self, text):
        return self.mecab.parse(text).strip()

    def calculate_metrics(self, reference, candidate):
        cer_value = cer(reference, candidate)
        bleu_score = sentence_bleu([reference.split()], candidate.split())
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_scores = scorer.score(reference, candidate)
        meteor = meteor_score([reference], candidate)
        embeddings1 = self.sentence_model.encode(reference, convert_to_tensor=True)
        embeddings2 = self.sentence_model.encode(candidate, convert_to_tensor=True)
        semantic_similarity = util.pytorch_cos_sim(embeddings1, embeddings2).item()

        return {
            "CER": cer_value,
            "BLEU": bleu_score,
            "ROUGE": rouge_scores,
            "METEOR": meteor,
            "Semantic Similarity": semantic_similarity
        }

import cv2
import numpy as np

class ImageMetricsEvaluator:
    def __init__(self):
        pass

    def calculate_ssim(self, image_path1, image_path2):
        image1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
        image2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)
        ssim_value = cv2.quality.QualitySSIM_compute(image1, image2)[0]
        return ssim_value

    def calculate_image_metrics(self, image_path1, image_path2):
        ssim_value = self.calculate_ssim(image_path1, image_path2)
        return {
            "SSIM": ssim_value
        }

class MetricsEvaluatorFactory:
    @staticmethod
    def get_evaluator(modality):
        if modality == 'text':
            return TextMetricsEvaluator()
        elif modality == 'audio':
            return AudioMetricsEvaluator()
        elif modality == 'image':
            return ImageMetricsEvaluator()
        else:
            raise ValueError("Unsupported modality")
