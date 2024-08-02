import os
import torch
from collections import Counter
from transformers import AutoTokenizer, BertForMaskedLM

def load_model(model_name="cl-tohoku/bert-base-japanese"):
    """加载tokenizer和模型。"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = BertForMaskedLM.from_pretrained(model_name)
    return tokenizer, model

def prepare_data(text, tokenizer):
    """准备模型输入数据。"""
    tokens = tokenizer.tokenize(text)
    tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_tensor = torch.tensor([input_ids])
    return input_tensor, tokens

def predict(input_tensor, model, device):
    """使用模型进行预测，并返回预测结果。"""
    model.eval()
    model.to(device)
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
    return outputs.logits

def correct_text_advanced(text, tokenizer, model, device='cpu'):
    """对文本进行纠错，能够处理特定和一般错误。"""
    # 将文本输入tokenizer进行编码
    encoded_input = tokenizer(text, return_tensors="pt").to(device)
    # 在不改变梯度的情况下，使用模型进行预测
    model.eval()
    model.to(device)
    with torch.no_grad():
        outputs = model(**encoded_input)
    predictions = outputs.logits

    # 纠正文本中的错误
    corrected_tokens = []
    for i, token_id in enumerate(encoded_input["input_ids"][0]):
        token = tokenizer.decode([token_id], clean_up_tokenization_spaces=True)
        if token in [tokenizer.cls_token, tokenizer.sep_token]:
            continue

        # 获取该位置的最可能的token ID
        predicted_id = predictions[0, i].argmax(dim=-1).item()
        predicted_token = tokenizer.decode([predicted_id], clean_up_tokenization_spaces=True)
        
        # 检查是否需要替换
        if predicted_token != token:
            corrected_tokens.append(predicted_token)
        else:
            corrected_tokens.append(token)
    
    # 将token转回字符串
    corrected_text = tokenizer.convert_tokens_to_string(corrected_tokens)
    return corrected_text

def extract_text_and_meta(line):
    # 假设每行格式为 "时间戳 发言者: 文本"
    parts = line.split(':', 1)
    if len(parts) == 2:
        return parts[0], parts[1].strip()
    else:
        return "", line  # 没有元数据的情况

def process_dialogues(filename, tokenizer, model, device='cpu'):
    corrected_lines = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():
                time_speaker, text = extract_text_and_meta(line)
                input_tensor = prepare_data(text, tokenizer)
                predictions = predict(input_tensor, model, device)
                corrected_text = correct_text_advanced(text, tokenizer, model, device)
                corrected_line = f"{time_speaker}: {corrected_text}"
                corrected_lines.append(corrected_line)
    return corrected_lines

def build_vocab(texts, tokenizer, top_n=1000):
    counter = Counter()
    for text in texts:
        tokens = tokenizer.tokenize(text)
        counter.update(tokens)
    most_common_tokens = [token for token, _ in counter.most_common(top_n)]
    vocab = {token: idx for idx, token in enumerate(most_common_tokens)}
    vocab['[UNK]'] = len(vocab)  # 添加未知词标记
    return vocab

def tokenize_with_vocab(text, tokenizer, vocab):
    tokens = tokenizer.tokenize(text)
    token_ids = [vocab.get(token, vocab['[UNK]']) for token in tokens]
    return token_ids

def main():
    # 确定运行设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # 加载模型和tokenizer
    tokenizer, model = load_model()

    # 示例文本#1, 2, 3（单句、多句）
    texts = [
            "今日は天気が良いです。しかし、間違った情報があります。",
            """
            昨日、私は友人と東京で美術展を見に行きました。
            展示されている作品は非常に多彩で、現代美術の多様な表現を感じることができました。
            その中でも、特に印象的だったのは、色彩豊かな絵画と、繊細な彫刻作品でした。
            美術館の外には広い庭園があり、季節の花々が咲いていて、訪れる人々を迎えていました。
            """,
            """
            昨日、私わ友达と東京で美術展を見に行きましが。
            展示されている作品わ非常に多かっで、現代美術の多様な表現お感じることができました。
            その中でも、特别印象的だったのわ、色彩豐かな絵画と、繊維な彫刻作品でした。
            美術館の外にわ広い庭園があり、季節の花々が咲いていて、訪れる人々を迎えていました。
            """
           ]

    for text in texts:
        # 打印原文
        print("Original text:", text)
        # 纠错
        corrected_text = correct_text_advanced(text, tokenizer, model, device)
        # 将token转回字符串，移除不需要的空格
        corrected_text = tokenizer.convert_tokens_to_string(corrected_text).replace(" ", "")
        # 打印纠错结果
        print("Corrected text:", corrected_text)

    # 示例文本4（对话）
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'dialogues') # 设置相对路径
    filename = os.path.join(data_dir, 'dialogue_file.txt')

    # 纠错
    corrected_dialogues = process_dialogues(filename, tokenizer, model, device)
    
    # 打印纠错结果
    for line in corrected_dialogues:
        print(line)

if __name__ == "__main__":
    main()
