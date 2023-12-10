from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from os import walk, path
from langchain.vectorstores import Chroma
from pipeline.embeddings import Embedding
 
def load_pdf_data(docs_path:str="."):

    if not path.exists(docs_path):
        return None
    
    print("Loading data from PDF document")

    for root,dir,files in walk(docs_path):
        for file in files:
            if file.endswith("pdf"):
                pdf_loader = PyPDFLoader(path.join(dir,file))

                # Load the data and return it
                data = pdf_loader.load()
                yield data

def get_chunks_splitted(data):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000, chunk_overlap=50)
    
    chunks = splitter.split_documents(data)

    return chunks
