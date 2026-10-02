import os

from dotenv import load_dotenv
from pymongo import MongoClient


load_dotenv()

MONGO_URI = os.getenv("MONGODB_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "foxconn_iphone_kb")
MONGO_COLLECTION_NAME = os.getenv(
    "MONGO_COLLECTION_NAME",
    "document_chunks"
)


def main():

    print("Connecting to MongoDB Atlas...")

    client = MongoClient(MONGO_URI)

    # Verify connection
    client.admin.command("ping")

    print("Connected to MongoDB Atlas successfully.")

    # Show the actual cluster/host being used
    print("\nConnected host:")
    print(client.address)

    # Show databases visible to this user
    print("\nDatabases:")
    print(client.list_database_names())

    db = client[MONGO_DB_NAME]
    collection = db[MONGO_COLLECTION_NAME]

    print("\nDatabase:", MONGO_DB_NAME)
    print("Collection:", MONGO_COLLECTION_NAME)

    test_document = {
        "document_id": "TEST-DOC-001",
        "chunk_id": "TEST-DOC-001-CHUNK-001",
        "text": "This is a test document for the Foxconn iPhone knowledge base.",
        "domain": "product",
        "document_type": "test",
        "source_type": "synthetic"
    }

    result = collection.insert_one(test_document)

    print("\nTest document inserted successfully.")
    print("Inserted ID:", result.inserted_id)

    print("\nDocuments currently in collection:")

    for document in collection.find():
        print(document)

    print("\nTotal documents:")
    print(collection.count_documents({}))

    client.close()

    print("\nMongoDB connection closed.")


if __name__ == "__main__":
    main()