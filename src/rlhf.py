# rlhf.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, BertTokenizer, BertForSequenceClassification, AdamW, Trainer, TrainingArguments
from datasets import Dataset as HFDataset, load_metric
from nltk.translate.bleu_score import sentence_bleu
import random
import argparse
import os
from utils.data_loader import load_data, load_config, parse_train_data, parse_asr_data

class FeedbackDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class RewardModel(nn.Module):
    def __init__(self):
        super(RewardModel, self).__init__()
        self.fc = nn.Linear(768, 1)  # 假设嵌入维度为768
    
    def forward(self, x):
        return self.fc(x)

def embed_text(texts):
    embeddings = torch.randn(len(texts), 768)
    return embeddings

class RLHFTrainer:
    def __init__(self, data, reward_model, epochs=10, lr=0.001):
        self.data = data
        self.reward_model = reward_model
        self.epochs = epochs
        self.lr = lr

    def train_reward_model(self):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.reward_model.parameters(), lr=self.lr)
        
        dataset = FeedbackDataset(self.data)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        for epoch in range(self.epochs):
            for batch in dataloader:
                queries = [item['query'] for item in batch]
                responses = [item['response'] for item in batch]
                feedbacks = torch.tensor([item['feedback'] for item in batch], dtype=torch.float32).unsqueeze(1)
                
                query_embeddings = embed_text(queries)
                response_embeddings = embed_text(responses)
                
                rewards = self.reward_model(response_embeddings)
                
                loss = criterion(rewards, feedbacks)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {loss.item()}")

class StreamingInference:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).cuda()

    def stream_infer(self, text):
        inputs = self.tokenizer(text, return_tensors='pt').cuda()
        outputs = self.model.generate(inputs['input_ids'], max_length=50, stream=True)
        for output in outputs:
            yield self.tokenizer.decode(output, skip_special_tokens=True)

class DialogueModelTrainer:
    def __init__(self, model_name, train_data, lr=5e-5, num_epochs=3):
        self.model_name = model_name
        self.train_data = train_data
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        if torch.cuda.is_available():
            self.model.cuda()
        self.optimizer = AdamW(self.model.parameters(), lr=lr)
        self.num_epochs = num_epochs

    def reward_function(self, generated_response, ideal_response):
        reference = [ideal_response.split()]
        candidate = generated_response.split()
        reward = sentence_bleu(reference, candidate)
        return reward

    def train_step(self, input_text, ideal_response):
        self.model.train()

        inputs = self.tokenizer(input_text, return_tensors='pt').input_ids.cuda()
        ideal_outputs = self.tokenizer(ideal_response, return_tensors='pt').input_ids.cuda()

        outputs = self.model.generate(inputs, max_length=50)
        generated_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        reward = self.reward_function(generated_response, ideal_response)

        model_output = self.model(inputs, labels=ideal_outputs)
        loss = model_output.loss
        reward_loss = loss * reward
        reward_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.item(), reward

    def train(self):
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

    def save_model(self, save_path):
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

class BERTFineTuner:
    def __init__(self, model_name, data, num_labels=2, num_epochs=3):
        self.model_name = model_name
        self.data = data
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        if torch.cuda.is_available():
            self.model.cuda()
        self.num_epochs = num_epochs

    def preprocess_function(self, examples):
        return self.tokenizer(examples['texts'], truncation=True, padding=True)

    def train(self):
        dataset = HFDataset.from_dict(self.data)
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

class ASRInference:
    def __init__(self, model_names):
        self.models = {name: AutoModelForSeq2SeqLM.from_pretrained(name) for name in model_names}
        self.tokenizers = {name: AutoTokenizer.from_pretrained(name) for name in model_names}
        if torch.cuda.is_available():
            for model in self.models.values():
                model.cuda()

    def generate_response(self, model, tokenizer, text):
        inputs = tokenizer(text, return_tensors='pt')
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        outputs = model.generate(inputs['input_ids'], max_length=50)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    def run_inference(self, text):
        responses = {}
        for name, model in self.models.items():
            tokenizer = self.tokenizers[name]
            responses[name] = self.generate_response(model, tokenizer, text)
        return responses

class MPISetup:
    @staticmethod
    def setup():
        dist.init_process_group(backend='mpi')

    @staticmethod
    def cleanup():
        dist.destroy_process_group()

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer = nn.Linear(10, 10)

    def forward(self, x):
        return self.layer(x)

class MPIRunner:
    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size

    def train(self):
        MPISetup.setup()
        
        torch.cuda.set_device(self.rank)
        model = MyModel().to(self.rank)
        model = DDP(model, device_ids=[self.rank])
        
        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        
        # Dummy training loop
        for epoch in range(10):
            inputs = torch.randn(20, 10).to(self.rank)
            targets = torch.randn(20, 10).to(self.rank)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if self.rank == 0:
                print(f"Epoch [{epoch+1}/10], Loss: {loss.item()}")
        
        MPISetup.cleanup()

def main():
    # Example usage
    reward_model = RewardModel()
    rlhf_trainer = RLHFTrainer(data=[{'query': 'example', 'response': 'example', 'feedback': 1}], reward_model=reward_model)
    rlhf_trainer.train_reward_model()

    dialogue_trainer = DialogueModelTrainer(model_name='t5-small', train_data=[{'input': 'Hello', 'ideal_response': 'Hi'}])
    dialogue_trainer.train()
    dialogue_trainer.save_model('dialogue_model')

    bert_fine_tuner = BERTFineTuner(model_name='bert-base-uncased', data={'texts': ['Example text'], 'labels': [1]})
    bert_fine_tuner.train()

    asr_inference = ASRInference(model_names=['facebook/wav2vec2-base-960h'])
    print(asr_inference.run_inference('Test speech'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    args = parser.parse_args()

    if args.world_size > 1:
        mpirunner = MPIRunner(rank=args.rank, world_size=args.world_size)
        mpirunner.train()
    else:
        main()
