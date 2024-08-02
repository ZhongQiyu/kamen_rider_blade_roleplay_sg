# run_agents.py

import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

# 定义加载和运行智能体的函数
def load_agent(config):
    model = AutoModelForCausalLM.from_pretrained(config['model_path']).to(config['device'])
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_path'])
    return model, tokenizer

def run_agent(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors='pt').to(model.device)
    outputs = model.generate(inputs['input_ids'], max_length=50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    config_file = 'agents_config.json'
    
    with open(config_file, 'r', encoding='utf-8') as f:
        agents_config = json.load(f)['characters']
    
    test_text = "这是一个测试对话。"
    
    for character, config in agents_config.items():
        model, tokenizer = load_agent(config)
        response = run_agent(model, tokenizer, test_text)
        print(f"Response from {character}: {response}")

if __name__ == "__main__":
    main()
