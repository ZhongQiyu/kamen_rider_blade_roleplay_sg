# utils.py

from moviepy.editor import VideoFileClip
from PIL import Image
import os
import subprocess
import MeCab
from jiwer import cer
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from nltk.translate.meteor_score import meteor_score
from sentence_transformers import SentenceTransformer, util
import joblib
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from extract_frames_and_audio import extract_frames, extract_audio
from create_dataset import create_dataset
from process_dataset import process_text_data
from tune_model import tune_model
from utils import tokenize_japanese, generate_text_data
import librosa

class VideoProcessor:
    def __init__(self, video_path):
        self.video_path = video_path
        self.clip = VideoFileClip(video_path)

    def extract_audio(self, output_audio_path):
        audio = self.clip.audio
        audio.write_audiofile(output_audio_path)

    def extract_frames(self, frames_folder, fps=1):
        if not os.path.exists(frames_folder):
            os.makedirs(frames_folder)
        frame_rate = self.clip.fps
        for i, frame in enumerate(self.clip.iter_frames(fps=frame_rate)):
            frame_image = Image.fromarray(frame)
            frame_image.save(f"{frames_folder}/frame_{i+1:03d}.png")

    def convert_video_format(self, output_path, output_format='mp4'):
        command = ['ffmpeg', '-i', self.video_path, f'{output_path}.{output_format}']
        self.run_subprocess(command)

    def run_subprocess(self, command):
        subprocess.run(command, check=True)

    def close(self):
        self.clip.close()

class TextProcessor:
    def __init__(self):
        self.tokenizer = Tokenizer()

    def tokenize_japanese(self, text):
        return list(self.tokenizer.tokenize(text, wakati=True))

    def generate_text_data(self, n_samples, n_features=20, n_classes=2, random_state=None):
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=n_samples, n_features=n_features,
                                   n_informative=2, n_redundant=2,
                                   n_classes=n_classes, random_state=random_state)
        documents = [" ".join(["feature"+str(i) if value > 0 else "" for i, value in enumerate(sample)]) for sample in X]
        return documents, y

class AudioProcessor:
    def __init__(self, model_path=None):
        self.model_path = model_path

    def extract_audio_features(self, audio_path):
        data, sample_rate = librosa.load(audio_path)
        mfccs = librosa.feature.mfcc(y=data, sr=sample_rate)
        return mfccs

    def classify_sound_features(self, features, model):
        # Assuming 'model' is a pre-trained audio classification model
        pass

class MetricsEvaluator:
    def __init__(self):
        self.mecab = MeCab.Tagger("-Owakati")
        self.sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    def mecab_tokenize(self, text):
        return self.mecab.parse(text).strip()

    def calculate_mer(self, reference, candidate):
        reference_tokens = self.mecab_tokenize(reference).split()
        candidate_tokens = self.mecab_tokenize(candidate).split()
        errors = sum(1 for ref, cand in zip(reference_tokens, candidate_tokens) if ref != cand)
        return errors / len(reference_tokens)

    def calculate_metrics(self, reference, candidate):
        cer_value = cer(reference, candidate)
        bleu_score = sentence_bleu([reference.split()], candidate.split())
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        rouge_scores = scorer.score(reference, candidate)
        meteor = meteor_score([reference], candidate)
        embeddings1 = self.sentence_model.encode(reference, convert_to_tensor=True)
        embeddings2 = self.sentence_model.encode(candidate, convert_to_tensor=True)
        semantic_similarity = util.pytorch_cos_sim(embeddings1, embeddings2).item()
        mer = self.calculate_mer(reference, candidate)

        return {
            "CER": cer_value,
            "BLEU": bleu_score,
            "ROUGE": rouge_scores,
            "METEOR": meteor,
            "Semantic Similarity": semantic_similarity,
            "MER": mer
        }

class VideoTextClassifier:
    def __init__(self, vectorizer=None, model=None):
        self.vectorizer = vectorizer if vectorizer else CountVectorizer(tokenizer=tokenize_japanese)
        self.model = model if model else SGDClassifier()
        self.initial_accuracy = None

    def online_learning(self, stream, threshold=0.1):
        for i, (documents, labels) in enumerate(stream):
            X_new = self.vectorizer.transform(documents)
            if i == 0:
                self.model.fit(X_new, labels)
                self.initial_accuracy = self.model.score(X_new, labels)
                print(f"Initial Batch - Accuracy: {self.initial_accuracy}")
                continue
            predictions = self.model.predict(X_new)
            new_accuracy = self.model.score(X_new, labels)
            print(f"Batch {i+1} - New Data Accuracy: {new_accuracy}")
            print(classification_report(labels, predictions))

            if new_accuracy < self.initial_accuracy - threshold:
                print("Performance decreased, updating the model.")
                self.model.partial_fit(X_new, labels)
                self.initial_accuracy = self.model.score(X_new, labels)

    def save_model(self, model_path='model.joblib', vectorizer_path='vectorizer.joblib'):
        joblib.dump(self.model, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)

    def load_model(self, model_path='model.joblib', vectorizer_path='vectorizer.joblib'):
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)

    def process_video(self, video_file_path, frames_folder_path, audio_file_path):
        # Extract frames and audio from video
        extract_frames(video_file_path, frames_folder_path)
        extract_audio(video_file_path, audio_file_path)

        # Create dataset
        dataset = create_dataset(frames_folder_path, audio_file_path)

        # Process dataset text data
        processed_data = process_text_data(dataset['text_data'])

        return processed_data

    def tune_and_save_model(self, processed_data, model_path='tuned_model.joblib'):
        # Tune model
        best_model = tune_model(self.model, processed_data)
        
        # Save the tuned model
        joblib.dump(best_model, model_path)

# Usage example
if __name__ == "__main__":
    video_path = 'path_to_video.mp4'
    frames_folder = 'path_to_frames'
    audio_path = 'path_to_audio.wav'

    classifier = VideoTextClassifier()
    
    # Process video and get processed data
    processed_data = classifier.process_video(video_path, frames_folder, audio_path)
    
    # Create data stream
    stream = (generate_text_data(n_samples=10, random_state=i) for i in range(100))
    
    # Perform online learning
    classifier.online_learning(stream)
    
    # Save the model and vectorizer
    classifier.save_model()
    
    # Optionally, you can load the model later using:
    # classifier.load_model()
    
    # Tune and save the model
    classifier.tune_and_save_model(processed_data)
