from sentence_transformers import SentenceTransformer

class Embedding:
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        # self.model = SentenceTransformer('gsarti/scibert-nli')
        
    def embed(self, sentences):
        if not isinstance(sentences, list):
            sentences = [sentences]
        
        return self.model.encode(sentences)
    
    