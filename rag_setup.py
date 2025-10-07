import os
from sentence_transformers import SentenceTransformer
import faiss
import pickle

class Retriever:
    def __init__(self, index_path='faiss_index.pkl', embedding_model='all-MiniLM-L6-v2'):
        self.index_path = index_path
        self.model = SentenceTransformer(embedding_model)
        if os.path.exists(index_path):
            with open(index_path,'rb') as f:
                saved = pickle.load(f)
            self.index = saved['index']
            self.docs = saved['docs']
        else:
            self.index = None
            self.docs = []

    def add_documents(self, texts: list):
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        if self.index is None:
            d = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(d)
            self.index.add(embeddings)
            self.docs = list(texts)
        else:
            self.index.add(embeddings)
            self.docs.extend(texts)
        with open(self.index_path,'wb') as f:
            pickle.dump({'index': self.index, 'docs': self.docs}, f)

    def get_relevant(self, query: str, k: int = 4):
        if self.index is None or len(self.docs) == 0:
            return []
        q_emb = self.model.encode([query], convert_to_numpy=True)
        _, I = self.index.search(q_emb, k)
        results = []
        for idx in I[0]:
            if idx < len(self.docs):
                results.append(self.docs[idx])
        return results
