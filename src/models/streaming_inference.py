# streaming_inference.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class StreamingInference:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).cuda()

    def stream_infer(self, text):
        inputs = self.tokenizer(text, return_tensors='pt').cuda()
        outputs = self.model.generate(inputs['input_ids'], max_length=50, stream=True)
        for output in outputs:
            yield self.tokenizer.decode(output, skip_special_tokens=True)
