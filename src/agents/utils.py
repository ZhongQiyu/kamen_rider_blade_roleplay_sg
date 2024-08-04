from janome.tokenizer import Tokenizer
from sklearn.datasets import make_classification

# 日语分词器初始化
t = Tokenizer()

# 分词函数：使用janome分词器将文本进行分词
def tokenize_japanese(text):
    tokens = list(t.tokenize(text, wakati=True))
    return tokens

# 模拟生成文本数据
def generate_text_data(n_samples, n_features=20, n_classes=2, random_state=None):
    # 生成分类数据集
    X, y = make_classification(n_samples=n_samples, n_features=n_features,
                               n_informative=2, n_redundant=2,
                               n_classes=n_classes, random_state=random_state)
    # 将特征转换为文本格式
    documents = [" ".join(["feature"+str(i) if value > 0 else "" for i, value in enumerate(sample)]) for sample in X]
    return documents, y

