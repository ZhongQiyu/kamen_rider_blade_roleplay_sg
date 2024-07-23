# test_pipeline.py

from transformers import pipeline

# 创建 pipeline
nlp = pipeline("text-classification", model="model_name", tokenizer="tokenizer_name")

# 测试文本
text = "这个手机真好用，我非常喜欢！"
result = nlp(text)

# 输出结果
print(result)

# 加载模型和分词器
from transformers import BertForQuestionAnswering, BertTokenizer

model_name = "hfl/chinese-roberta-wwm-ext"
model = BertForQuestionAnswering.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

# 创建 pipeline
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)



from transformers import pipeline

# 创建 pipeline
nlp = pipeline("text-classification", model="model_name", tokenizer="tokenizer_name")

# 测试文本
text = "这个手机真好用，我非常喜欢！"
result = nlp(text)

# 输出结果
print(result)

# 加载模型和分词器
from transformers import BertForQuestionAnswering, BertTokenizer

model_name = "hfl/chinese-roberta-wwm-ext"
model = BertForQuestionAnswering.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

# 创建 pipeline
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)
