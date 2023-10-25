from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
import numpy as np
import os

class Embedding:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        if model_name == "sentence-transformers/all-MiniLM-L6-v2":
            self.embedding_model = SentenceTransformer(model_name)
        else:
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": "cuda"},
                encode_kwargs={"device": "cuda", "batch_size": 100})

    def encode(self, batch):
        embeddings = self.embedding_model.embed_documents(batch["text"])
        return {"text": batch["text"], "source": batch["source"], "embeddings": embeddings}

    def get_llm(self):
        return self.embedding_model