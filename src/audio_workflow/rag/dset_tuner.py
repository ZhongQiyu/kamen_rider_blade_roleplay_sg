# dset_tuner.py

from transformers import RagTokenizer, RagTokenForGeneration, RagConfig, RagRetriever, Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import joblib
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from utils import tokenize_japanese, generate_text_data
from extract_frames_and_audio import extract_frames, extract_audio
from create_dataset import create_dataset
from process_dataset import process_text_data

# 1. 加载和预处理数据集
dataset = load_dataset('csv', data_files={'train': 'path_to_train_data.csv', 'validation': 'path_to_validation_data.csv'})
model_name = "facebook/rag-token-nq"
tokenizer = RagTokenizer.from_pretrained(model_name)
config = RagConfig.from_pretrained(model_name)
retriever = RagRetriever.from_pretrained(model_name, config=config, indexed_dataset=None)  # 请根据您的情况修改
model = RagTokenForGeneration.from_pretrained(model_name, config=config, retriever=retriever)

# 格式化数据集
def preprocess_function(examples):
    inputs = [f"Q: {q}" for q in examples['question']]
    targets = [f"A: {a}" for a in examples['answer']]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=512, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True)

# 2. 设置训练参数并初始化Trainer
training_args = TrainingArguments(
    output_dir="./rag_finetuned",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    save_steps=10_000,
    save_total_limit=3,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

# 3. 微调模型
trainer.train()

# 4. 超参数调优 (假设你有另外的模型和数据需要调优)
def tune_model(X_train, y_train):
    model = RandomForestClassifier()
    param_distributions = {
        'n_estimators': np.arange(100, 500, 100),
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': np.arange(2, 11),
        'min_samples_leaf': np.arange(1, 11),
        'bootstrap': [True, False]
    }
    random_search = RandomizedSearchCV(
        estimator=model, 
        param_distributions=param_distributions,
        n_iter=10, 
        scoring='accuracy', 
        cv=5, 
        verbose=1, 
        random_state=42,
        n_jobs=-1
    )
    random_search.fit(X_train, y_train)
    print("Best parameters found: ", random_search.best_params_)
    best_model = random_search.best_estimator_
    return best_model

# 5. 在线学习流程
def online_learning(stream, model, vectorizer, threshold=0.1):
    initial_accuracy = None
    for i, (documents, labels) in enumerate(stream):
        X_new = vectorizer.transform(documents)
        if i == 0:
            model.fit(X_new, labels)
            initial_accuracy = model.score(X_new, labels)
            print(f"Initial Batch - Accuracy: {initial_accuracy}")
            continue
        predictions = model.predict(X_new)
        new_accuracy = model.score(X_new, labels)
        print(f"Batch {i+1} - New Data Accuracy: {new_accuracy}")
        print(classification_report(labels, predictions))

        if new_accuracy < initial_accuracy - threshold:
            print("Performance decreased, updating the model.")
            model.partial_fit(X_new, labels)
            initial_accuracy = model.score(X_new, labels)

# 6. 从视频中提取帧和音频并创建数据集
video_file_path = 'path_to_video.mp4'
frames_folder_path = 'path_to_frames'
audio_file_path = 'path_to_audio.wav'

extract_frames(video_file_path, frames_folder_path)
extract_audio(video_file_path, audio_file_path)

dataset = create_dataset(frames_folder_path, audio_file_path)

# 7. 处理数据集文本数据
processed_data = process_text_data(dataset['text_data'])

# 8. 保存模型和向量化器到磁盘
vectorizer = CountVectorizer(tokenizer=tokenize_japanese)
model = SGDClassifier()
stream = (generate_text_data(n_samples=10, random_state=i) for i in range(100))
online_learning(stream, model, vectorizer)

joblib.dump(model, 'model.joblib')
joblib.dump(vectorizer, 'vectorizer.joblib')
