import os

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.operations import SearchIndexModel


load_dotenv()

MONGO_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("MONGO_DB_NAME", "foxconn_iphone_kb")
COLLECTION_NAME = os.getenv(
    "MONGO_COLLECTION_NAME",
    "document_chunks"
)


client = MongoClient(MONGO_URI)

collection = client[DB_NAME][COLLECTION_NAME]


vector_index = SearchIndexModel(
    definition={
        "fields": [
            {
                "type": "vector",
                "path": "embedding",
                "numDimensions": 384,
                "similarity": "cosine"
            }
        ]
    },
    name="vector_index",
    type="vectorSearch"
)


result = collection.create_search_index(model=vector_index)

print("Vector Search index creation requested.")
print("Index name:", result)

client.close()