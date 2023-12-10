from pipeline.doc_load.document_loader import multi_xscience_get_text
import unittest

class TestDocumentLoad(unittest.TestCase):
    
    def setUp(self) -> None:
        ...    
    
    def test_load_multi_xscience(self):
        train, test,val = multi_xscience_get_text()

        self.assertEqual(len(train[0]),len(train[1]), "Train size mismatch")
        self.assertEqual(len(test[0]), len(test[1]), "test size mismatch")
        self.assertEqual(len(val[0]), len(val[1]), "val size mismatch")

    # def test_load_pdf_data