# proc_demo.py

import re
import os
import csv
import json
import logging
import boto3
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, AdamW,
                          BertTokenizer, BertForSequenceClassification, GPT2LMHeadModel, GPT2Tokenizer,
                          pipeline, set_seed)
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from datasets import Dataset, load_metric
import random
import numpy as np

class DataProcessor:
    def __init__(self, directory_path, config_file):
        self.directory_path = directory_path
        self.data = []
        self.dialog = []
        self.current_time = None
        self.current_episode = {'episode': 'Unknown', 'dialogs': []}
        self.current_speaker = None
        self.config = self.load_config(config_file)

    def get_directory_path(self):
        return self.directory_path

    def get_data(self):
        return self.data

    def get_current_episode(self):
        return self.current_episode

    def load_config(self, config_file):
        with open(config_file, 'r', encoding='utf-8') as file:
            config = json.load(file)
        return config

    def load_data(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data

    def parse_train_data(self, data):
        input_key = self.config['train_data']['input_key']
        output_key = self.config['train_data']['output_key']
        parsed_data = [{'input': entry[input_key], 'ideal_response': entry[output_key]} for entry in data]
        return parsed_data

    def parse_asr_data(self, data):
        text_key = self.config['asr_data']['text_key']
        parsed_data = [{'text': entry[text_key]} for entry in data]
        return parsed_data

    def lambda_handler(self, event, context):
        s3 = boto3.client('s3')

        for record in event['Records']:
            bucket_name = record['s3']['bucket']['name']
            object_key = record['s3']['object']['key']
            print(f"A new file {object_key} was uploaded in bucket {bucket_name}")

        return {
            'statusCode': 200,
            'body': json.dumps('Process completed successfully!')
        }

    @staticmethod
    def sort_files(filename):
        part = filename.split('.')[0]
        try:
            return int(part)
        except ValueError:
            return float('inf')

    def read_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        return lines

    def finalize_episode(self):
        if self.current_episode:
            if self.dialog:
                self.current_episode['dialogs'].append({
                    'speaker': self.current_speaker,
                    'time': self.current_time,
                    'text': ' '.join(self.dialog).strip()
                })
                self.dialog = []
            self.data.append(self.current_episode)
            print(f"Finalized episode: {self.current_episode}")
            self.current_episode = {'episode': 'Unknown', 'dialogs': []}

    def process_line(self, line):
        speaker_match = re.match(r'^話者(\d+)\s+(\d{2}:\d{2})\s+(.*)$', line)  # 日语版本的正则表达式
        if speaker_match:
            if self.dialog:  # 如果有未完成的对话，先完成它
                self.current_episode['dialogs'].append({
                    'speaker': self.current_speaker,
                    'time': self.current_time,
                    'text': ' '.join(self.dialog).strip()
                })
                self.dialog = []
            self.current_speaker, self.current_time, text = speaker_match.groups()
            self.dialog = [text]
        else:
            self.dialog.append(line)

    def process_file(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line:
                    self.process_line(line)
        self.finalize_episode()
        print(f"Processed file: {file_path} with data: {self.data[-1] if self.data else 'No Data'}")

    def process_all_files(self):
        files = [f for f in os.listdir(self.directory_path) if f.endswith('.txt')]
        files = sorted(files, key=self.sort_files)
        for filename in files:
            file_path = os.path.join(self.directory_path, filename)
            self.process_file(file_path)

    def export_to_txt(self, output_file):
        with open(output_file, 'w', encoding='utf-8') as file:
            for content in self.data:
                file.write(json.dumps(content, ensure_ascii=False) + '\n')

    def handle_dialog(self, lines):
        for line in lines:
            speaker_match = re.match(r'^話者(\d+) (\d{2}:\d{2})', line)  # 日语版本的正则表达式
            if speaker_match:
                if self.current_speaker is not None:
                    self.data.append({
                        'speaker': self.current_speaker,
                        'time': self.current_time,
                        'text': ' '.join(self.dialog).strip()
                    })
                    self.dialog = []
                self.current_speaker, self.current_time = speaker_match.groups()
            else:
                self.dialog.append(line.strip())

        if self.current_speaker and self.dialog:
            self.data.append({
                'speaker': self.current_speaker,
                'time': self.current_time,
                'text': ' '.join(self.dialog).strip()
            })

        return self.data

    def save_as_json(self, output_file):
        with open(output_file, 'w', encoding='utf-8') as file:
            json.dump(self.data, file, ensure_ascii=False, indent=4)

    def save_as_csv(self, output_file):
        with open(output_file, 'w', newline='', encoding='utf-8') as file:
            fieldnames = ['episode', 'time', 'speaker', 'text']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for episode in self.data:
                if 'dialogs' not in episode:
                    continue
                for dialog in episode['dialogs']:
                    writer.writerow({
                        'episode': episode['episode'],
                        'time': dialog['time'],
                        'speaker': dialog['speaker'],
                        'text': dialog['text']
                    })

    def generate_new_entry(self, last_dialog):
        new_prompt = last_dialog['text']
        new_response = "新しい回答"  # 日语版本
        return {
            'prompt': new_prompt,
            'response': new_response,
            'chosen': new_response,
            'rejected': "他の選択肢"
        }

    def decide_chosen_and_rejected(self, responses):
        if responses:
            chosen = responses[0]
            rejected = responses[1:]
        else:
            chosen = None
            rejected = []
        return chosen, rejected

class MultiModalProcessor:
    def __init__(self, config_file, device='cpu'):
        self.config = self.load_config(config_file)
        self.device = device
        self.initialize_models()
        set_seed(42)  # 设置随机种子，以确保文本生成的可重复性
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)

    def load_config(self, config_file):
        with open(config_file, 'r', encoding='utf-8') as file:
            config = json.load(file)
        return config

    def import_module(self, module_name):
        module = __import__(module_name)
        return module

    def initialize_models(self):
        try:
            self.tokenizer = BertTokenizer.from_pretrained(self.config.get('tokenizer_model', 'bert-base-multilingual-cased'))
            self.classification_model = BertForSequenceClassification.from_pretrained(
                self.config.get('classification_model', 'bert-base-multilingual-cased'),
                num_labels=self.config.get('num_labels', 2)
            ).to(self.device)
            
            self.generator = GPT2LMHeadModel.from_pretrained('gpt2').to(self.device)
            self.generator_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            
            # 加载其他需要的模型
            self.gemma_model = self.import_gemma_model()
            self.multi_model_manager = self.import_multi_model_manager()
        except Exception as e:
            self.logger.error(f"Model loading failed: {str(e)}")
            raise

    def import_gemma_model(self):
        class GemmaModel(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super(GemmaModel, self).__init__()
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(hidden_size, output_size)

            def forward(self, x):
                x = self.fc1(x)
                x = self.relu(x)
                x = self.fc2(x)
                return x

        return GemmaModel

    def import_multi_model_manager(self):
        class MultiModelManager:
            def __init__(self):
                # 加载所有需要的模型
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
                    # 处理其他类型模型的生成
                    pass

        return MultiModelManager()

    def load_data(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data

    def parse_train_data(self, data):
        input_key = self.config['train_data']['input_key']
        output_key = self.config['train_data']['output_key']
        parsed_data = [{'input': entry[input_key], 'ideal_response': entry[output_key]} for entry in data]
        return parsed_data

    def parse_asr_data(self, data):
        text_key = self.config['asr_data']['text_key']
        parsed_data = [{'text': entry[text_key]} for entry in data]
        return parsed_data

    def process_audio(self, audio_file):
        return self.audio_processor.process(audio_file)

    def process_image(self, image_file):
        return self.image_processor.process(image_file)

    def process_text(self, text_file):
        return self.text_processor.process(text_file)

    def process_video(self, video_file):
        return self.video_processor.process(video_file)

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

    def reward_function(self, generated_response, ideal_response):
        similarity = F.cosine_similarity(generated_response.unsqueeze(0), ideal_response.unsqueeze(0), dim=1)
        return similarity.item()

    def train_step(self, model, tokenizer, optimizer, input_text, ideal_response):
        model.train()

        inputs = tokenizer(input_text, return_tensors='pt').input_ids.cuda()
        ideal_outputs = tokenizer(ideal_response, return_tensors='pt').input_ids.cuda()

        outputs = model.generate(inputs, max_length=50)
        generated_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        generated_response_embedding = model(input_ids=tokenizer(generated_response, return_tensors='pt').input_ids.cuda()).last_hidden_state.mean(dim=1)
        ideal_response_embedding = model(input_ids=ideal_outputs).last_hidden_state.mean(dim=1)
        reward = self.reward_function(generated_response_embedding, ideal_response_embedding)

        model_output = model(inputs, labels=ideal_outputs)
        loss = model_output.loss
        reward_loss = loss * reward
        reward_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        return loss.item(), reward

    def train_seq2seq_model(self, model_name, train_data, lr=5e-5, num_epochs=3):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        if torch.cuda.is_available():
            model.cuda()
        optimizer = AdamW(model.parameters(), lr=lr)

        for epoch in range(num_epochs):
            random.shuffle(train_data)
            total_loss = 0
            total_reward = 0
            for data in train_data:
                input_text = data['input']
                ideal_response = data['ideal_response']
                loss, reward = self.train_step(model, tokenizer, optimizer, input_text, ideal_response)
                total_loss += loss
                total_reward += reward
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_data)}, Reward: {total_reward/len(train_data)}")

        return model, tokenizer

    def train_classification_model(self, model_name, train_data, num_labels=2, num_epochs=3):
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        if torch.cuda.is_available():
            model.cuda()

        dataset = Dataset.from_dict(train_data)
        encoded_dataset = dataset.map(lambda examples: tokenizer(examples['texts'], truncation=True, padding=True), batched=True)

        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=num_epochs,
            per_device_train_batch_size=4,
            warmup_steps=10,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
        )

        metric = load_metric("accuracy")

        def compute_metrics(p):
            preds = p.predictions.argmax(-1)
            return {'accuracy': (preds == p.label_ids).astype(float).mean().item()}

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=encoded_dataset,
            eval_dataset=encoded_dataset,
            compute_metrics=compute_metrics,
        )

        trainer.train()

        return model, tokenizer

    def train_gemma_model(self, X, y, input_size, hidden_size, output_size, num_epochs, learning_rate=0.01):
        model = self.gemma_model(input_size, hidden_size, output_size)
        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        accuracies = []

        kfold = KFold(n_splits=5, shuffle=True)

        for fold, (train_indices, val_indices) in enumerate(kfold.split(X)):
            X_train, y_train = X[train_indices], y[train_indices]
            X_val, y_val = X[val_indices], y[val_indices]

            # 重置模型
            model = self.gemma_model(input_size, hidden_size, output_size)
            model.train()

            # 使用新的模型配置重新训练模型
            for epoch in range(num_epochs):
                # 前向传播
                outputs = model(X_train)
                loss = criterion(outputs, y_train.float())
                
                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # 在验证集上评估模型
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_predictions = torch.round(val_outputs)  # 对于二分类问题，使用round进行预测
                val_accuracy = accuracy_score(y_val, val_predictions)
                accuracies.append(val_accuracy)
                print(f"Fold {fold+1}, Validation Accuracy: {val_accuracy}")

        avg_accuracy = sum(accuracies) / len(accuracies)
        print(f"Average Validation Accuracy: {avg_accuracy}")

        return model

    def save_model(self, model, tokenizer, save_path):
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

    def generate_response(self, model, tokenizer, text):
        inputs = tokenizer(text, return_tensors='pt')
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        outputs = model.generate(inputs['input_ids'], max_length=50)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    def train_model(self, args):
        train_data = self.load_data(args.train_data_file)
        parsed_train_data = self.parse_train_data(train_data)

        if args.model_type == 'seq2seq':
            model, tokenizer = self.train_seq2seq_model(model_name=args.model_name, train_data=parsed_train_data)
        elif args.model_type == 'classification':
            model, tokenizer = self.train_classification_model(model_name=args.model_name, train_data={
                'texts': [entry['input'] for entry in parsed_train_data],
                'labels': [0] * len(parsed_train_data)
            })
        elif args.model_type == 'gemma':
            X = torch.randn(100, int(self.config['gemma_input_size']))  # 示例数据
            y = torch.randint(0, 2, (100,))  # 示例标签（0或1的二分类）
            model = self.train_gemma_model(X, y, int(self.config['gemma_input_size']),
                                           int(self.config['gemma_hidden_size']),
                                           int(self.config['gemma_output_size']),
                                           int(self.config['num_epochs']))
        self.save_model(model, tokenizer, args.output_dir)

    def process_data(self, args):
        asr_data = self.load_data(args.asr_data_file)
        parsed_asr_data = self.parse_asr_data(asr_data)

        print("数据处理完成。")

        model, tokenizer = self.train_classification_model(model_name='bert-base-uncased', train_data={
            'texts': [entry['text'] for entry in parsed_asr_data],
            'labels': [0] * len(parsed_asr_data)
        })
        self.save_model(model, tokenizer, './peft_model')

    def main(self, args):
        if args.command == 'train':
            self.train_model(args)
        elif args.command == 'process':
            self.process_data(args)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multi-Modal Processing Script")

    subparsers = parser.add_subparsers(dest='command')

    train_parser = subparsers.add_parser('train')
    train_parser.add_argument('--config_file', type=str, required=True, help="Path to the config file")
    train_parser.add_argument('--train_data_file', type=str, required=True, help="Path to the training data file")
    train_parser.add_argument('--validation_data_file', type=str, help="Path to the validation data file")
    train_parser.add_argument('--model_name', type=str, required=True, help="Name of the model")
    train_parser.add_argument('--model_type', type=str, required=True, choices=['seq2seq', 'causal', 'classification', 'rag', 'gemma'], help="Type of the model")
    train_parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the trained model")

    process_parser = subparsers.add_parser('process')
    process_parser.add_argument('--config_file', type=str, required=True, help="Path to the config file")
    process_parser.add_argument('--asr_data_file', type=str, required=True, help="Path to the ASR data file")

    args = parser.parse_args()

    processor = MultiModalProcessor(args.config_file)
    processor.main(args)

    # Directory path and config can be passed or modified in main() as required
    directory_path = '/path/to/your/data/episodes_txt'  # 更新为你的实际路径
    output_json_path = os.path.join(directory_path, 'data.json')
    output_csv_path = os.path.join(directory_path, 'data.csv')
    output_txt_path = os.path.join(directory_path, 'combined.txt')

    data_processor = DataProcessor(directory_path, args.config_file)

    data_processor.process_all_files()
    print("Data ready for export:", data_processor.get_data()[:1])
    data_processor.export_to_txt(output_txt_path)

    all_lines = data_processor.read_file(os.path.join(directory_path, '1.txt'))
    dialog_data = data_processor.handle_dialog(all_lines)

    data_processor.data = dialog_data
    data_processor.save_as_json(output_json_path)
    data_processor.save_as_csv(output_csv_path)
