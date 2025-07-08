"""
Embedding generation using HuggingFace models
"""
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List
import torch

class HuggingFaceEmbeddings:
    """Custom HuggingFace embeddings wrapper"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = SentenceTransformer(model_name)
        if torch.cuda.is_available() and device == "cuda":
            self.model = self.model.to(device)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents"""
        embeddings = self.model.encode(texts, convert_to_tensor=False)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query"""
        embedding = self.model.encode([text], convert_to_tensor=False)
        return embedding[0].tolist()
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension"""
        return self.model.get_sentence_embedding_dimension()