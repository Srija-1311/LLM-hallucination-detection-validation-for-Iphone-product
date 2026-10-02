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


# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Connect to MongoDB Atlas
client = MongoClient(MONGO_URI)

client.admin.command("ping")

print("Connected to MongoDB Atlas.")


db = client[DB_NAME]
collection = db[COLLECTION_NAME]


# Test document
text = """
The iPhone 17 production process includes assembly,
inspection, testing and quality verification.
"""


# Generate embedding
embedding = model.encode(text).tolist()


document = {
    "document_id": "TEST-DOC-002",
    "chunk_id": "TEST-DOC-002-CHUNK-001",

    "text": text,

    "domain": "manufacturing",
    "department": "manufacturing",
    "document_type": "test",
    "source_type": "synthetic",

    "embedding": embedding
}


result = collection.insert_one(document)


print("Vector document inserted successfully.")
print("Inserted ID:", result.inserted_id)
print("Embedding dimensions:", len(embedding))


client.close()