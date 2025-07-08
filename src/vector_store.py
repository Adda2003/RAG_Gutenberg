"""
Vector store implementation using FAISS
"""
import faiss
import numpy as np
import pickle
import os
from typing import List, Tuple, Optional
from langchain.schema import Document
from src.embeddings import HuggingFaceEmbeddings

class FAISSVectorStore:
    """FAISS-based vector store"""
    
    def __init__(self, embeddings: HuggingFaceEmbeddings):
        self.embeddings = embeddings
        self.index = None
        self.documents = []
        self.dimension = embeddings.dimension
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store"""
        texts = [doc.page_content for doc in documents]
        embeddings_list = self.embeddings.embed_documents(texts)
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings_list).astype('float32')
        
        # Create or update FAISS index
        if self.index is None:
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings_array)
        
        # Add to index
        self.index.add(embeddings_array)
        self.documents.extend(documents)
    
    def similarity_search(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """Search for similar documents"""
        if self.index is None or len(self.documents) == 0:
            return []
        
        # Get query embedding
        query_embedding = np.array([self.embeddings.embed_query(query)]).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, k)
        
        # Return documents with scores
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.documents):
                results.append((self.documents[idx], float(score)))
        
        return results
    
    def save(self, path: str) -> None:
        """Save the vector store"""
        os.makedirs(path, exist_ok=True)
        
        if self.index is not None:
            faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        
        with open(os.path.join(path, "documents.pkl"), "wb") as f:
            pickle.dump(self.documents, f)
    
    def load(self, path: str) -> None:
        """Load the vector store"""
        index_path = os.path.join(path, "index.faiss")
        docs_path = os.path.join(path, "documents.pkl")
        
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
        
        if os.path.exists(docs_path):
            with open(docs_path, "rb") as f:
                self.documents = pickle.load(f)