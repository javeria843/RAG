from sentence_transformers import SentenceTransformer, util
from sentence_transformers import SentenceTransformer, util

# Load a pre-trained Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

query = "What are the advancements in NLP?"
documents = [
    "Machine learning enables advancements in NLP.",
    "Climate change is a pressing issue globally.",
    "Natural language processing allows machines to understand text."
]

# Encode query and documents
query_embedding = model.encode(query)
doc_embeddings = model.encode(documents)

# Find the most similar document to the query
results = util.semantic_search(query_embedding, doc_embeddings, top_k=1)
most_similar_doc = documents[results[0][0]['corpus_id']]
print(f"Most similar document to the query: \"{most_similar_doc}\"")