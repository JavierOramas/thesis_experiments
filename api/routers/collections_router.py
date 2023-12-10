from fastapi import APIRouter, Depends
from pipeline.database.chroma_db import ChromaHandler
from typing import List

db = ChromaHandler()

router = APIRouter(
    prefix="/collections",
    tags=["db collections handler"],
    responses = {404: {"description": "Not Found"}}
)

@router.get("")
def get_collections():
    return { "collections": db.list_collections() }

@router.get("/new/<name>")
def create_new_collection(name:str):
    
    collection = db.create_collection(name)
    
    return {
        "status": 200,
        "collection": name
    }

@router.post("/store")
def store_sentences(collection_name:str, documents: List[str], embeddings: List[List[float]]):
    
    collection_item = db.get_collection(collection_name)
    db.add_embeddings(collection=collection_item, documents=documents, embeddings=embeddings)
    
    return {
        "status": 200,
        "message": "Data stored successfully"
    }
