from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import numpy as np
import os

class Embedding:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        if model_name == "text-embedding-ada-002":
            self.embedding_model = OpenAIEmbeddings(
                model=model_name,
                openai_api_base=os.environ["OPENAI_API_BASE"],
                openai_api_key=os.environ["OPENAI_API_KEY"])
        else:
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": "cuda"},
                encode_kwargs={"device": "cuda", "batch_size": 100})

    def encode(self, batch):
        embeddings = self.embedding_model.embed_documents(batch["text"])
        return {"text": batch["text"], "source": batch["source"], "embeddings": embeddings}
