# download.py

import requests
from transformers import AutoTokenizer, AutoModel

class DownloadManager:
    def __init__(self, model_name: str, file_url: str, dest_path: str):
        self.model_name = model_name
        self.file_url = file_url
        self.dest_path = dest_path

    def download_file(self):
        response = requests.get(self.file_url, stream=True)
        with open(self.dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"Downloaded file from {self.file_url} to {self.dest_path}")

    def download_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModel.from_pretrained(self.model_name)
        print(f"Downloaded {self.model_name} model and tokenizer.")
        return tokenizer, model

    def download_all(self):
        self.download_file()
        tokenizer, model = self.download_model()
        return tokenizer, model

if __name__ == "__main__":
    model_name = "bert-base-uncased"
    file_url = "https://example.com/somefile.zip"
    dest_path = "path/to/save/somefile.zip"

    manager = DownloadManager(model_name, file_url, dest_path)
    tokenizer, model = manager.download_all()
