import joblib
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from utils import tokenize_japanese, generate_text_data

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

# 初始化向量化器和在线学习模型
vectorizer = CountVectorizer(tokenizer=tokenize_japanese)
model = SGDClassifier()

# 创建数据流并开始在线学习
stream = (generate_text_data(n_samples=10, random_state=i) for i in range(100))
online_learning(stream, model, vectorizer)

# 模型持久化：保存模型和向量化器到磁盘
joblib.dump(model, 'model.joblib')
joblib.dump(vectorizer, 'vectorizer.joblib')

# 若要加载模型和向量化器，可以使用以下代码：
# model = joblib.load('model.joblib')
# vectorizer = joblib.load('vectorizer.joblib')



from extract_frames_and_audio import extract_frames, extract_audio
from create_dataset import create_dataset
from process_dataset import process_text_data
from tune_model import tune_model

# 示例路径，你需要替换为实际的文件路径
video_file_path = 'path_to_video.mp4'
frames_folder_path = 'path_to_frames'
audio_file_path = 'path_to_audio.wav'

# Step 1: Extract frames and audio from video
extract_frames(video_file_path, frames_folder_path)
extract_audio(video_file_path, audio_file_path)

# Step 2: Create dataset
dataset = create_dataset(frames_folder_path, audio_file_path)

# Step 3: Process dataset text data
# 此处假设dataset中有文本数据需要处理
processed_data = process_text_data(dataset['text_data'])

# Step 4: Tune model
# 此处假设你有一个初始化的模型和需要调整的数据集
model = None  # Your initialized model goes here
best_model = tune_model(model, processed_data)

# 接下来可以保存模型，或者使用模型进行进一步的操作
