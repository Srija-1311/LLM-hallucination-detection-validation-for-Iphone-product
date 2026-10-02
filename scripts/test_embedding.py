from sentence_transformers import SentenceTransformer


model = SentenceTransformer("all-MiniLM-L6-v2")

text = """
The iPhone 17 production process includes assembly,
inspection, testing and quality verification.
"""

embedding = model.encode(text)

print("Embedding generated successfully.")
print("Embedding type:", type(embedding))
print("Embedding dimensions:", len(embedding))
print("First 10 values:")
print(embedding[:10])