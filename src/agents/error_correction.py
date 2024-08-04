import torch
from transformers import BertTokenizer, BertForMaskedLM, AutoTokenizer

def load_model(model_name="cl-tohoku/bert-base-japanese"):
    # 使用AutoTokenizer自动选择正确的分词器类
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = BertForMaskedLM.from_pretrained(model_name)
    return tokenizer, model

def correct_text(text, tokenizer, model, device='cpu'):  # 默认使用CPU
    model.eval()
    model.to(device)

    # 示例：简单的纠错方法（仅示例，需要根据实际情况调整）
    masked_text = text.replace("間違った", "[MASK]")
    encoded_input = tokenizer(masked_text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**encoded_input)
        predictions = outputs.logits

    masked_index = torch.where(encoded_input["input_ids"] == tokenizer.mask_token_id)[1]
    predicted_id = predictions[0, masked_index].argmax(dim=-1)
    predicted_token = tokenizer.decode(predicted_id).strip()

    corrected_text = masked_text.replace("[MASK]", predicted_token)
    return corrected_text

def main():
    asr_text = "ここで間違ったテキストが入力されます"  # 示例文本
    tokenizer, model = load_model()

    # 确定运行设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    corrected_text = correct_text(asr_text, tokenizer, model, device)
    print("Original Text:", asr_text)
    print("Corrected Text:", corrected_text)

if __name__ == "__main__":
    main()

