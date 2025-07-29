from sentence_transformers import SentenceTransformer, util
import faiss
import numpy as np

# 1. Load Sentence Transformer Model
model = SentenceTransformer('all-MiniLM-L6-v2')  # small, fast, good quality

# 2. Corpus (Documents to Search From)
corpus = [
    "RAG stands for Retrieval-Augmented Generation.",
    "It combines a retriever and a generator model.",
    "FAISS is used for fast vector similarity search.",
    "BERT is a transformer model developed by Google.",
    "Python is a programming language.",
    "ChatGPT is built using large language models."
]

# 3. Encode documents into embeddings (vectors)
corpus_embeddings = model.encode(corpus, convert_to_tensor=False)

# 4. Convert to NumPy array for FAISS
corpus_np = np.array(corpus_embeddings).astype('float32')

# 5. Initialize FAISS index
dimension = corpus_np.shape[1]  # length of vector
index = faiss.IndexFlatL2(dimension)  # L2 distance (Euclidean)
index.add(corpus_np)  # Add document vectors to index

# 6. User Query
query = "What is Retrieval-Augmented Generation?"
query_embedding = model.encode(query).astype('float32')

# 7. Search top 2 similar documents
top_k = 2
distances, indices = index.search(np.array([query_embedding]), top_k)

# 8. Show results
print("Query:", query)
print("\nTop Matching Documents:")
for idx in indices[0]:
    print("→", corpus[idx])

# 9. (Optional) Generate Final Answer from Retrieved Docs
context = " ".join([corpus[i] for i in indices[0]])
final_answer = f"Based on the documents: {context} → Answer: RAG is a system that uses retrieval and generation together."

print("\n🧠 Final Answer:\n", final_answer)
