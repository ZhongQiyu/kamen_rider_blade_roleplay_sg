# knowledge_base.py

import os
import json
from tqdm import tqdm
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader, UnstructuredMarkdownLoader
from faiss_indexer import FaissIndexer

class KnowledgeBase:
    def __init__(self, model_path, persist_directory):
        self.model_path = model_path
        self.persist_directory = persist_directory
        self.docs = []
        self.split_docs = []
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Path {self.model_path} not found.")

    @staticmethod
    def get_files(dir_path):
        file_list = []
        for filepath, dirnames, filenames in os.walk(dir_path):
            for filename in filenames:
                if filename.endswith(".md") or filename.endswith(".txt"):
                    file_list.append(os.path.join(filepath, filename))
        return file_list

    def get_text(self, dir_path):
        file_lst = self.get_files(dir_path)
        docs = []
        for one_file in tqdm(file_lst):
            file_type = one_file.split('.')[-1]
            if file_type == 'md':
                loader = UnstructuredMarkdownLoader(one_file)
            elif file_type == 'txt':
                loader = UnstructuredFileLoader(one_file)
            else:
                continue
            docs.extend(loader.load())
        return docs

    def load_documents(self, dir_paths):
        for dir_path in dir_paths:
            self.docs.extend(self.get_text(dir_path))
        if not self.docs:
            raise ValueError("No documents were loaded. Please check the file path and format.")

    def split_texts(self, chunk_size=500, chunk_overlap=150):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.split_docs = text_splitter.split_documents(self.docs)
        if not self.split_docs:
            raise ValueError("The input text list is empty after splitting.")

    def build_vector_db(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        vectordb = Chroma.from_documents(
            documents=self.split_docs,
            embedding=embeddings,
            persist_directory=self.persist_directory
        )
        vectordb.persist()

# 示例用法
if __name__ == "__main__":
    tar_dirs = [
        "/path/to/data/InternLM",
        "/path/to/data/InternLM-XComposer",
        "/path/to/data/lagent",
        "/path/to/data/lmdeploy",
        "/path/to/data/opencompass",
        "/path/to/data/xtuner"
    ]

    kb = KnowledgeBase(model_path="/path/to/data/model/sentence-transformer", persist_directory='/path/to/data_base/vector_db/chroma')
    kb.load_documents(tar_dirs)
    kb.split_texts()
    kb.build_vector_db()



import redis
import numpy as np
import faiss
import pysrt

# 连接到 Redis 服务器
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)

# 存储向量到 Redis
def store_vectors_in_redis(redis_client, vectors, ids):
    for vector, id in zip(vectors, ids):
        redis_client.set(id, vector.tobytes())

# 从 Redis 加载向量
def load_vectors_from_redis(redis_client, ids):
    vectors = []
    for id in ids:
        vector = np.frombuffer(redis_client.get(id), dtype='float32')
        vectors.append(vector)
    return np.array(vectors)

# 加载数据到 Redis
def load_data_to_redis(asr_file, subtitle_file):
    asr_data = load_asr_text(asr_file)
    subtitle_data = load_subtitles(subtitle_file)
    merged_data = merge_data(asr_data, subtitle_data)
    vectors = np.array([entry['asr_text'] for entry in merged_data])
    ids = [f'vector_{i}' for i in range(len(merged_data))]
    store_vectors_in_redis(redis_client, vectors, ids)

# 加载字幕文件
def load_subtitles(file_path):
    subs = pysrt.open(file_path)
    dialogues = [(sub.start, sub.end, sub.text) for sub in subs]
    return dialogues

# 加载ASR文本文件
def load_asr_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        asr_text = file.read().splitlines()
    return asr_text

# 合并ASR数据和字幕数据
def merge_data(asr_data, subtitle_data):
    merged_data = []
    for asr_line, (start, end, subtitle) in zip(asr_data, subtitle_data):
        merged_data.append({
            'asr_text': asr_line,
            'subtitle': subtitle,
            'timestamp': (start, end)
        })
    return merged_data

# 搜索相似向量
def search_similar_vectors(query_vector):
    d = 128  # 向量的维度
    index = faiss.IndexFlatL2(d)  # L2 距离
    all_ids = [key.decode() for key in redis_client.keys() if key.startswith(b'vector_')]
    vectors = load_vectors_from_redis(redis_client, all_ids)
    index.add(vectors)
    D, I = index.search(np.array([query_vector]), k=5)  # k 是返回的相似向量的数量
    similar_vectors = [all_ids[i] for i in I[0]]
    return similar_vectors

# data_base/knowledge_base.py

import redis
import json

def load_data_to_redis(asr_file, subtitle_file):
    r = redis.Redis(host='localhost', port=6379, db=0)
    
    with open(asr_file, 'r', encoding='utf-8') as file:
        asr_data = file.read().splitlines()
        
    with open(subtitle_file, 'r', encoding='utf-8') as file:
        subs = pysrt.open(file)
        subtitle_data = [(sub.start, sub.end, sub.text) for sub in subs]

    merged_data = []
    for asr_line, (start, end, subtitle) in zip(asr_data, subtitle_data):
        merged_data.append({
            'asr_text': asr_line,
            'subtitle': subtitle,
            'timestamp': (start, end)
        })
    
    for i, entry in enumerate(merged_data):
        r.set(f'dialogue:{i}', json.dumps(entry))

def search_similar_vectors(query_vector):
    # 这个函数需要根据你的向量搜索库进行实现，例如FAISS或者其他库
    # 这里仅提供一个伪代码实现
    similar_vectors = ["result1", "result2", "result3"]
    return similar_vectors
