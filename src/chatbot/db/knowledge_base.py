# knowledge_base.py

import os
import json
from tqdm import tqdm
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader, UnstructuredMarkdownLoader
from faiss_indexer import FaissIndexer
import redis
import pysrt
import numpy as np

class KnowledgeBase:
    def __init__(self, model_path, persist_directory, redis_host='localhost', redis_port=6379, redis_db=0, dimension=128):
        self.model_path = model_path
        self.persist_directory = persist_directory
        self.docs = []
        self.split_docs = []
        self.indexer = FaissIndexer(dimension, redis_host, redis_port, redis_db)
        
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

    def load_data_to_redis(self, asr_file, subtitle_file):
        with open(asr_file, 'r', encoding='utf-8') as file:
            asr_data = file.read().splitlines()
            
        subs = pysrt.open(subtitle_file)
        subtitle_data = [(sub.start, sub.end, sub.text) for sub in subs]

        merged_data = []
        for asr_line, (start, end, subtitle) in zip(asr_data, subtitle_data):
            merged_data.append({
                'asr_text': asr_line,
                'subtitle': subtitle,
                'timestamp': (start, end)
            })
        
        for i, entry in enumerate(merged_data):
            self.indexer.redis_client.set(f'dialogue:{i}', json.dumps(entry))

    def search_similar_vectors(self, query_vector, k=5):
        D, I = self.indexer.search(np.array([query_vector]), k)
        similar_vectors = [self.indexer.get_vector_by_id(i) for i in I[0]]
        return similar_vectors

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

    kb = KnowledgeBase(
        model_path="/path/to/data/model/sentence-transformer",
        persist_directory='/path/to/data_base/vector_db/chroma',
        redis_host='localhost',
        redis_port=6379,
        redis_db=0,
        dimension=128
    )
    
    kb.load_documents(tar_dirs)
    kb.split_texts()
    kb.build_vector_db()

    # 加载ASR数据和字幕数据到Redis
    kb.load_data_to_redis("/path/to/asr_file.txt", "/path/to/subtitle_file.srt")

    # 查询相似向量
    query_vector = np.random.rand(128).astype('float32')
    similar_vectors = kb.search_similar_vectors(query_vector)
    print(similar_vectors)
