from janome.tokenizer import Tokenizer

t = Tokenizer()
tokens = list(t.tokenize(text, wakati=True))
print(tokens)
