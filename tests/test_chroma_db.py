import unittest

# Import your database client module here
from pipeline.database.chroma_db import ChromaHandler

class TestDatabaseClient(unittest.TestCase):

    def setUp(self):
        # Create a new database client for each test
        self.db_client = ChromaHandler()
        self.collection = self.db_client.create_collection(name="test")

    def test_create_db(self):
        self.assertIsNotNone(self.db_client.client.database)

    def test_add(self):
        result = self.db_client.add(collection=self.collection, documents=["document1"], metadata=[{"a":"a"}], ids=["1"])
        self.assertEqual(result, 200)
        
    def test_add_embeddings(self):
        # Test if add_embedding method adds an embedding to a record
        result = self.db_client.add_embeddings(self.collection, documents=["document2"], metadata=None, ids=["2"], embeddings=[[1.0]*384])
        self.assertEqual(result, 200)

    def test_query(self):
        # Test if query method returns the correct data from the database
        result = self.db_client.add(collection=self.collection, documents=["document3"], metadata=None, ids=["3"])
        result = self.db_client.query(collection=self.collection, query=["document3"], top_k=1)
        self.assertEqual(result['ids'][0][0], '3' )

    def test_query_embedding(self):
        # Test if query_embedding method returns the correct embedding
        embedding = [1.0]*384
        result = self.db_client.query_embeddings(collection=self.collection, query_embeddings=embedding, top_k=1)
        self.assertEqual(result['ids'][0][0],'2')

    def test_query_nonexistent_record(self):
        # Test if querying a non-existent record returns None
        result = self.db_client.query(self.collection, query="999", top_k=1)
        self.assertGreaterEqual(result["distances"][0][0], 1)

if __name__ == '__main__':
    unittest.main()
