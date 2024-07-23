# peft.py

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
import argparse

class PEFTTrainer:
    def __init__(self, model_name, train_data, peft_method, num_labels=2, lr=5e-5, num_epochs=3):
        self.model_name = model_name
        self.train_data = train_data
        self.peft_method = peft_method
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        if torch.cuda.is_available():
            self.model.cuda()
        self.lr = lr
        self.num_epochs = num_epochs

        if peft_method == "freeze_layers":
            self.freeze_layers()
        elif peft_method == "adapter":
            self.add_adapter()

    def freeze_layers(self):
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.classifier.parameters():
            param.requires_grad = True

    def add_adapter(self):
        self.model.add_adapter("classification_adapter")
        self.model.train_adapter("classification_adapter")

    def preprocess_function(self, examples):
        return self.tokenizer(examples['texts'], truncation=True, padding=True)

    def train(self):
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
            learning_rate=self.lr,
        )

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
        self.save_model('./peft_model')

    def save_model(self, save_path):
        if self.peft_method == "adapter":
            self.model.save_adapter(save_path, "classification_adapter")
        else:
            self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

# 示例用法
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PEFT Trainer")
    parser.add_argument('--model_name', type=str, required=True, help="Name of the pretrained model")
    parser.add_argument('--train_data_file', type=str, required=True, help="Path to the training data file")
    parser.add_argument('--peft_method', type=str, required=True, choices=["freeze_layers", "adapter"], help="PEFT method to use")
    parser.add_argument('--num_labels', type=int, default=2, help="Number of labels for classification")
    parser.add_argument('--lr', type=float, default=5e-5, help="Learning rate")
    parser.add_argument('--num_epochs', type=int, default=3, help="Number of training epochs")
    args = parser.parse_args()

    with open(args.train_data_file, 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    peft_trainer = PEFTTrainer(
        model_name=args.model_name,
        train_data=train_data,
        peft_method=args.peft_method,
        num_labels=args.num_labels,
        lr=args.lr,
        num_epochs=args.num_epochs
    )
    peft_trainer.train()
