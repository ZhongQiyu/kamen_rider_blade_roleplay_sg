# model_hub.py

import logging
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mutual_info_score
from transformers import (BertTokenizerFast, BertForSequenceClassification,
                          GPT2LMHeadModel, GPT2Tokenizer, pipeline, set_seed,
                          AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM,
                          AdamW, Trainer, TrainingArguments)
from datasets import Dataset, load_metric

# 设置随机种子，以确保文本生成的可重复性
set_seed(42)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelHub:
    def __init__(self, config, device='cpu'):
        self.config = config
        self.device = device
        self.initialize_models()

    def initialize_models(self):
        try:
            self.tokenizer = BertTokenizerFast.from_pretrained(self.config.get('tokenizer_model', 'bert-base-multilingual-cased'))
            self.classification_model = BertForSequenceClassification.from_pretrained(
                self.config.get('classification_model', 'bert-base-multilingual-cased'), 
                num_labels=self.config.get('num_labels', 2)
            ).to(self.device)
            
            self.generator = GPT2LMHeadModel.from_pretrained('gpt2').to(self.device)
            self.generator_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            
            self.multi_model_manager = MultiModelManager()
        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            raise

    def classify(self, text):
        inputs = self.tokenizer(text, return_tensors='pt').to(self.device)
        outputs = self.classification_model(**inputs)
        return torch.argmax(outputs.logits, dim=1).item()

    def generate_text(self, prompt, temperature=0.7):
        inputs = self.generator_tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        outputs = self.generator.generate(inputs, max_length=100, num_return_sequences=1, temperature=temperature)
        return self.generator_tokenizer.decode(outputs[0], skip_special_tokens=True)

    def generate_pipeline_text(self, prompt, temperature=0.7):
        generator = pipeline('text-generation', model=self.generator, tokenizer=self.generator_tokenizer)
        outputs = generator(prompt, max_length=100, num_return_sequences=1, temperature=temperature)
        return outputs[0]['generated_text']

class MultiModelManager:
    def __init__(self):
        self.gpt_model, self.gpt_tokenizer = self.load_model('gpt2', GPT2LMHeadModel)
        self.bert_model, self.bert_tokenizer = self.load_model('bert-base-multilingual-cased', BertForSequenceClassification)

    def load_model(self, model_name, model_class=AutoModelForSeq2SeqLM):
        model = model_class.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model, tokenizer

    def generate_text(self, text, model_type='gpt'):
        if model_type == 'gpt':
            inputs = self.gpt_tokenizer(text, return_tensors='pt')
            outputs = self.gpt_model.generate(inputs['input_ids'], max_length=100)
            return self.gpt_tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            pass

class TaskEvaluator:
    def __init__(self, relatedness_matrix=None, task_complexities=None):
        self.relatedness_matrix = relatedness_matrix
        self.task_complexities = task_complexities

    def calculate_average_relatedness(self):
        if self.relatedness_matrix is not None:
            return self.relatedness_matrix.mean(axis=1)
        return None

    @staticmethod
    def calculate_mutual_information(task1_data, task2_data):
        return mutual_info_score(task1_data, task2_data)

    def identify_complex_tasks(self, complexity_threshold):
        if self.task_complexities is not None:
            return [i for i, complexity in enumerate(self.task_complexities) if complexity > complexity_threshold]
        return []

    @staticmethod
    def measure_execution_time(task_function):
        start_time = time.time()
        task_function()
        end_time = time.time()
        return end_time - start_time

    @staticmethod
    def calculate_comprehensive_scores(relatedness_scores, information_gain_scores, complexity_scores, execution_time_scores, weights):
        return [
            weights['relatedness'] * relatedness_scores[i] +
            weights['information_gain'
