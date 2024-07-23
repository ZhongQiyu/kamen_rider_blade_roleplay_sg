# utils.py

import MeCab
from jiwer import cer
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from sentence_transformers import SentenceTransformer, util

def mecab_tokenize(text):
    mecab = MeCab.Tagger("-Owakati")
    return mecab.parse(text).strip()

def calculate_mer(reference, candidate):
    reference_tokens = mecab_tokenize(reference).split()
    candidate_tokens = mecab_tokenize(candidate).split()
    errors = sum(1 for ref, cand in zip(reference_tokens, candidate_tokens) if ref != cand)
    return errors / len(reference_tokens)

def calculate_metrics(reference, candidate):
    # CER
    cer_value = cer(reference, candidate)
    
    # BLEU
    bleu_score = sentence_bleu([reference.split()], candidate.split())
    
    # ROUGE
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(reference, candidate)
    
    # METEOR
    meteor = meteor_score([reference], candidate)
    
    # Semantic Similarity
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    embeddings1 = model.encode(reference, convert_to_tensor=True)
    embeddings2 = model.encode(candidate, convert_to_tensor=True)
    semantic_similarity = util.pytorch_cos_sim(embeddings1, embeddings2).item()

    # Morpheme Error Rate (MER)
    mer = calculate_mer(reference, candidate)
    
    return {
        "CER": cer_value,
        "BLEU": bleu_score,
        "ROUGE": rouge_scores,
        "METEOR": meteor,
        "Semantic Similarity": semantic_similarity,
        "MER": mer
    }
