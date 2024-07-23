# faiss_indexer.py

import faiss
import numpy as np
import redis

class FaissIndexer:
    def __init__(self, dimension, redis_host='localhost', redis_port=6379, redis_db=0):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)  # L2 距离
        self.redis_client = redis.StrictRedis(host=redis_host, port=redis_port, db=redis_db)

    def add_vectors(self, vectors, ids):
        self.index.add(vectors)
        self.store_vectors_in_redis(vectors, ids)

    def store_vectors_in_redis(self, vectors, ids):
        for vector, id in zip(vectors, ids):
            self.redis_client.set(id, vector.tobytes())

    def load_vectors_from_redis(self, ids):
        vectors = []
        for id in ids:
            vector = np.frombuffer(self.redis_client.get(id), dtype='float32')
            vectors.append(vector)
        return np.array(vectors)

    def search(self, query_vector, k=5):
        D, I = self.index.search(query_vector, k)  # k 是返回的相似向量的数量
        return D, I

    def get_vector_by_id(self, id):
        return np.frombuffer(self.redis_client.get(id), dtype='float32')

# 示例使用
if __name__ == "__main__":
    dimension = 128
    indexer = FaissIndexer(dimension)

    # 添加示例向量
    vectors = np.random.rand(10, dimension).astype('float32')
    ids = [f'vector_{i}' for i in range(10)]
    indexer.add_vectors(vectors, ids)

    # 查询示例
    query_vector = np.random.rand(1, dimension).astype('float32')
    D, I = indexer.search(query_vector)
    print(f"Distances: {D}")
    print(f"Indices: {I}")
