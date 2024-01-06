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
    
    def create_collection(self, name:str):
        collection = self.client.get_or_create_collection(name=name, embedding_function=self.embedding_model.encode) # Get a collection object from an existing collection, by name. If it doesn't exist, create it.
        
        return collection
    
    def get_collection(self, name:str):
        return self.client.get_collection(name)
    
    def list_collections(self):
        return self.client.list_collections()
    
    def add(self, collection, documents, metadata, ids):

        embeddings = []
        for x in documents:
            embeddings.append(self.embedding_model.encode(x).tolist())
            
        if not metadata:
            metadata = [None for i in embeddings]
        
        for e,d,m,i in zip(embeddings,documents, metadata,ids):
            
            sim = collection.query(
                query_embeddings=e,
                n_results=1
            )['distances'][0]

            if len(sim) > 0:
                sim = sim[0]
            else:
                sim = None
                
            if sim == None or sim < 0.9:
                collection.add(
                    embeddings=[e],
                    documents=[d],
                    metadatas=[m],
                    ids=[i],
                )

        return 200

    def add_embedding_to_database(self, collection, embeddings, documents, metadatas, similarity_limit=0.9):
        
        for embedding,document,metadata in zip(embeddings, documents, metadatas):

            max_similarity = self.query_embeddings(collection, embedding, top_k=1)["distances"][0]
            print(max_similarity)          
            if not max_similarity or max_similarity[0] == 0.0:
                # If the database is empty, add the new embedding directly
                import uuid
                ids = [str(uuid.uuid1())]
                collection.add(embeddings=[embedding], documents=[document], metadatas=[metadata], ids=ids)
            
            if isinstance(max_similarity, list):
                if len(max_similarity) > 0:
                    max_similarity = max_similarity[0]
                else:
                    max_similarity = 5
            # else:
            if max_similarity < similarity_limit:
                # If similarity exceeds the limit, skip adding the new embedding
                print("Similar embedding found. Skipping addition.")
            else:
                # Add the new embedding to the database
                import uuid
                new_id = str(uuid.uuid1())
                collection.add(embeddings=[embedding], documents=[document], metadatas=[metadata], ids=[new_id])


    def add_embeddings(self, collection, embeddings, documents, metadata, ids=None):
        

        if ids is None:
            import uuid
            ids = [str(uuid.uuid1()) for _ in embeddings]
            
            
            collection.add(
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
    
