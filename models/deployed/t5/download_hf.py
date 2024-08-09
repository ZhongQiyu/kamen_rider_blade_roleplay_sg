# download_hf.py

# download_hf.py
from transformers import AutoTokenizer, AutoModel

def download_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model

if __name__ == "__main__":
    model_name = "bert-base-uncased"
    tokenizer, model = download_model(model_name)
    print(f"Downloaded {model_name} model and tokenizer.")
