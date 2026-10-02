import os

from dotenv import load_dotenv
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer


load_dotenv()

MONGO_URI = os.getenv("MONGODB_URI")
DB_NAME = os.getenv("MONGO_DB_NAME", "foxconn_iphone_kb")
COLLECTION_NAME = os.getenv(
    "MONGO_COLLECTION_NAME",
    "document_chunks"
)


model = SentenceTransformer("all-MiniLM-L6-v2")

client = MongoClient(MONGO_URI)

collection = client[DB_NAME][COLLECTION_NAME]


query = "How is the phone manufactured?"

query_embedding = model.encode(query).tolist()


pipeline = [
    {
        "$vectorSearch": {
            "index": "vector_index",
            "path": "embedding",
            "queryVector": query_embedding,
            "numCandidates": 20,
            "limit": 5
        }
    },
    {
        "$project": {
            "_id": 0,
            "document_id": 1,
            "chunk_id": 1,
            "text": 1,
            "domain": 1,
            "department": 1,
            "document_type": 1,
            "score": {
                "$meta": "vectorSearchScore"
            }
        }
    }
]


results = collection.aggregate(pipeline)


print("\nSemantic Search Results")
print("=" * 60)

for result in results:
    print("\nDocument:", result["document_id"])
    print("Chunk:", result["chunk_id"])
    print("Text:", result["text"])
    print("Score:", result["score"])


client.close()