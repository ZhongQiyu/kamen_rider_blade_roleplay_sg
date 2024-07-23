# test_knowledge_base.py

import os
import ssl
import json
import nltk
import numpy as np
from faiss_indexer import FaissIndexer
from knowledge_base import KnowledgeBase

# 解决nltk下载证书验证失败问题
ssl._create_default_https_context = ssl._create_unverified_context

class KnowledgeBaseLoader:
    def __init__(self, file_path='dialogues.json'):
        self.file_path = file_path
        self.dialogues = self.load_from_file()

    @staticmethod
    def setup_nlp_resources():
        nltk.download('punkt')

    def load_from_file(self):
        if not os.path.exists(self.file_path):
            return {}
        with open(self.file_path, 'r', encoding='utf-8') as file:
            return json.load(file)

def main():
    try:
        # Setup
        print("Setting up NLP resources...")
        KnowledgeBaseLoader.setup_nlp_resources()

        print("Initializing knowledge base and Faiss indexer...")
        kb_loader = KnowledgeBaseLoader()
        indexer = FaissIndexer(dimension=768)
        print("Knowledge base and Faiss indexer initialized successfully.")
    
        # 存储和加载示例向量
        vectors = np.random.rand(10, 768).astype('float32')
        ids = [f'vector_{i}' for i in range(10)]
        indexer.add_vectors(vectors, ids)

        loaded_vectors = indexer.load_vectors_from_redis(ids)
        print("Vectors loaded from Redis.")

        # 创建和查询 Faiss 索引
        query_vector = np.random.rand(1, 768).astype('float32')
        D, I = indexer.search(query_vector)
        print(f"Query vector: {query_vector}")
        print(f"Distances: {D}")
        print(f"Indices: {I}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
