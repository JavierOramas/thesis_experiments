from fastapi import FastAPI, Depends
from api.routers import collections_router
from pipeline.database.chroma_db import ChromaHandler

app = FastAPI()

app.include_router(collections_router.router)


@app.get("/")
def read_root():
    return {"Hello": "W"}




