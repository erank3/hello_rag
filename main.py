from typing import Union
from fastapi import FastAPI

import os.path
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)

app = FastAPI()

# check if storage already exists
PERSIST_DIR = "./storage"
if not os.path.exists(PERSIST_DIR):
    # load the documents and create the index
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    # store it for later
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)



@app.get("/")
def answer_quetsion(question: str = None):
    if question is None:  
        return {"result": "Missing argument 'question'"}  
    # query the index
    query_engine = index.as_query_engine()
    query_result = query_engine.query(question)
    #print(response)
    return {"answer": query_result.response}
