import chromadb
from typing import List
from sentence_transformers import SentenceTransformer

class ChromaHandler:
    
    def __init__(self, persistent_path=None, embed_model=None) -> None:
        
        if embed_model:
            self.embedding_model = embed_model
        else:
            self.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        if persistent_path:
            self.client = chromadb.PersistentClient(path=persistent_path)
        else:
            self.client = chromadb.Client()
    
    def create_collection(self, name=str):
        collection = self.client.get_or_create_collection(name=name, embedding_function=self.embedding_model.encode) # Get a collection object from an existing collection, by name. If it doesn't exist, create it.
        
        return collection
    
    def add(self, collection, documents, metadata, ids):

        embeddings = []
        for x in documents:
            embeddings.append(self.embedding_model.encode(x).tolist())
            print(len(embeddings[-1]))
            
        collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadata,
            ids=ids,
        )

        return 200
    
    def add_embeddings(self, collection, embeddings, documents, metadata, ids):
        
        collection.upsert(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadata,
            ids=ids
        )

        return 200
    
    def query(self,collection, query:List[str], top_k=5):
        
        results = collection.query(
            query_texts=query,
            n_results=top_k
        )   

        return results
    
    def query_embeddings(self,collection, query_embeddings:List[List[float]], top_k=5):
        results = collection.query(
            query_embeddings=query_embeddings,
            n_results=top_k,
        )

        return results
    
