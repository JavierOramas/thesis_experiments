from pipeline.embeddings.basic_embeddings import Embedding

def test_embedding():
    model = Embedding(model_name="thenlper/gte-small")

    vector = model.encode({'text':"test embedding", 'source': "test"})
    
    for i in vector:
        assert len(i) == 384
