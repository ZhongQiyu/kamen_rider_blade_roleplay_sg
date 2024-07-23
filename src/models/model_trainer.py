# model_trainer.py

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, AdamW, BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, load_metric
import random
import argparse
import os
from data_loader import DataLoader

class ModelTrainer:
    def __init__(self, model_name, train_data=None, model_type='seq2seq', lr=5e-5, num_epochs=3):
        self.model_name = model_name
        self.train_data = train_data
        self.model_type = model_type
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if model_type == 'seq2seq':
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        elif model_type == 'causal':
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
        elif model_type == 'classification':
            self.model = BertForSequenceClassification.from_pretrained(model_name)
        else:
            raise ValueError("Unsupported model type")
        
        if torch.cuda.is_available():
            self.model.cuda()
        self.optimizer = AdamW(self.model.parameters(), lr=lr)
        self.num_epochs = num_epochs

    def reward_function(self, generated_response, ideal_response):
        similarity = F.cosine_similarity(generated_response.unsqueeze(0), ideal_response.unsqueeze(0), dim=1)
        return similarity.item()

    def train_step(self, input_text, ideal_response):
        self.model.train()

        inputs = self.tokenizer(input_text, return_tensors='pt').input_ids.cuda()
        ideal_outputs = self.tokenizer(ideal_response, return_tensors='pt').input_ids.cuda()

        outputs = self.model.generate(inputs, max_length=50)
        generated_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        generated_response_embedding = self.model(input_ids=self.tokenizer(generated_response, return_tensors='pt').input_ids.cuda()).last_hidden_state.mean(dim=1)
        ideal_response_embedding = self.model(input_ids=ideal_outputs).last_hidden_state.mean(dim=1)
        reward = self.reward_function(generated_response_embedding, ideal_response_embedding)

        model_output = self.model(inputs, labels=ideal_outputs)
        loss = model_output.loss
        reward_loss = loss * reward
        reward_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.item(), reward

    def train_seq2seq_model(self):
        for epoch in range(self.num_epochs):
            random.shuffle(self.train_data)
            total_loss = 0
            total_reward = 0
            for data in self.train_data:
                input_text = data['input']
                ideal_response = data['ideal_response']
                loss, reward = self.train_step(input_text, ideal_response)
                total_loss += loss
                total_reward += reward
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(self.train_data)}, Reward: {total_reward/len(self.train_data)}")

    def train_classification_model(self, data):
        self.train_data = data
        dataset = Dataset.from_dict(self.train_data)
        encoded_dataset = dataset.map(self.preprocess_function, batched=True)

        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=self.num_epochs,
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
            model=self.model,
            args=training_args,
            train_dataset=encoded_dataset,
            eval_dataset=encoded_dataset,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        self.save_model('./finetuned_model')

    def save_model(self, save_path):
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    def generate_response(self, text):
        inputs = self.tokenizer(text, return_tensors='pt')
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        outputs = self.model.generate(inputs['input_ids'], max_length=50)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    def stream_infer(self, text):
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).cuda()
        inputs = self.tokenizer(text, return_tensors='pt').cuda()
        outputs = self.model.generate(inputs['input_ids'], max_length=50, stream=True)
        for output in outputs:
            yield self.tokenizer.decode(output, skip_special_tokens=True)

    def preprocess_function(self, examples):
        return self.tokenizer(examples['texts'], truncation=True, padding=True)

# 训练模型的函数
def train_model(args):
    # 初始化DataLoader
    data_loader = DataLoader(args.config_file)

    # 加载并解析训练数据
    train_data = data_loader.load_data(args.train_data_file)
    parsed_train_data = data_loader.parse_train_data(train_data)

    # 初始化模型训练器
    model_trainer = ModelTrainer(model_name=args.model_name, train_data=parsed_train_data, model_type=args.model_type)
    
    if args.model_type == 'seq2seq':
        model_trainer.train_seq2seq_model()
    elif args.model_type == 'classification':
        model_trainer.train_classification_model({
            'texts': [entry['input'] for entry in parsed_train_data],
            'labels': [0] * len(parsed_train_data)  # 假设所有标签为0，根据需要调整
        })
    
    model_trainer.save_model(args.output_dir)

# 处理数据的函数
def process_data(args):
    # 初始化DataLoader
    data_loader = DataLoader(args.config_file)

    # 加载并解析ASR数据
    asr_data = data_loader.load_data(args.asr_data_file)
    parsed_asr_data = data_loader.parse_asr_data(asr_data)

    # 处理数据的逻辑
    print("数据处理完成。")

    # 训练PEFT模型示例
    peft_trainer = ModelTrainer(model_name='bert-base-uncased', train_data={
        'texts': [entry['text'] for entry in parsed_asr_data],
        'labels': [0] * len(parsed_asr_data)  # 假设所有标签为0，根据需要调整
    }, model_type='classification')
    peft_trainer.train_classification_model({
        'texts': [entry['text'] for entry in parsed_asr_data],
        'labels': [0] * len(parsed_asr_data)
    })
    peft_trainer.save_model('./peft_model')

# 主函数
def main():
    parser = argparse.ArgumentParser(description="Model Training and Data Processing Script")

    # 公共参数
    parser.add_argument('--config_file', type=str, required=True, help="Path to the config file")

    # 模型训练参数
    parser.add_argument('--train_data_file', type=str, help="Path to the training data file")
    parser.add_argument('--model_name', type=str, help
