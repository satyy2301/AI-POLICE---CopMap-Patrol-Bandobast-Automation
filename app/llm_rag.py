import os
import json
from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from transformers import pipeline


class RAGStore:
    def __init__(self, storage_dir: str = 'data/rag_store', embed_model: str = 'all-MiniLM-L6-v2'):
        os.makedirs(storage_dir, exist_ok=True)
        self.storage_dir = storage_dir
        self.embed_model_name = embed_model
        self.embedder = SentenceTransformer(self.embed_model_name)
        self.docs_path = os.path.join(storage_dir, 'docs.json')
        self.index_path = os.path.join(storage_dir, 'index.faiss')

        self.docs = []
        self.index = None
        self.dim = self.embedder.get_sentence_embedding_dimension()

        if os.path.exists(self.docs_path) and os.path.exists(self.index_path):
            try:
                with open(self.docs_path, 'r', encoding='utf-8') as f:
                    self.docs = json.load(f)
                self.index = faiss.read_index(self.index_path)
            except Exception:
                self.docs = []
                self.index = None

        if self.index is None:
            # inner product index; we'll normalize vectors for cosine similarity
            self.index = faiss.IndexFlatIP(self.dim)

    def _save(self):
        with open(self.docs_path, 'w', encoding='utf-8') as f:
            json.dump(self.docs, f)
        faiss.write_index(self.index, self.index_path)

    def ingest(self, text: str, metadata: Optional[dict] = None):
        emb = self.embedder.encode([text], normalize_embeddings=True)
        self.index.add(np.array(emb, dtype='float32'))
        self.docs.append({'text': text, 'metadata': metadata or {}})
        self._save()

    def search(self, query: str, top_k: int = 5) -> List[dict]:
        q_emb = self.embedder.encode([query], normalize_embeddings=True)
        if self.index.ntotal == 0:
            return []
        D, I = self.index.search(np.array(q_emb, dtype='float32'), top_k)
        results = []
        for idx in I[0]:
            if idx < len(self.docs):
                results.append(self.docs[idx])
        return results


class Summarizer:
    def __init__(self, rag_store: RAGStore):
        self.rag = rag_store
        # cost-aware generator: small open model
        self.generator = pipeline('text2text-generation', model='google/flan-t5-small')

    def summarize(self, query: str = 'patrol events', top_k: int = 5) -> str:
        items = self.rag.search(query, top_k=top_k)
        if not items:
            return 'No events available to summarize.'

        # build concise context (cost-aware): only include top-3 short entries
        contexts = []
        for it in items:
            txt = it.get('text', '')
            # keep it short
            contexts.append(txt if len(txt) < 800 else txt[:800] + '...')

        prompt = 'Summarize the following patrol events, highlight patterns, risks, and recommendations:\n\n' + '\n\n'.join(contexts)
        out = self.generator(prompt, max_length=256, do_sample=False)
        return out[0]['generated_text']
