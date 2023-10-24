from transformers import AutoTokenizer, AutoModel
import torch
from sentence_transformers import SentenceTransformer

class ADAEmbedding:
    def __init__(self):

        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def embed(self, text):
        return self.model.encode(text)