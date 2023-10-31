from pipeline.embeddings.basic_embeddings import Embedding

def test_embedding():

    model = Embedding()
    vector = model.encode("test")
    assert len(vector) == 384
