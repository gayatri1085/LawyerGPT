from sentence_transformers import SentenceTransformer
import faiss, pickle, numpy as np
from transformers import pipeline

with open("constitution_faiss.pkl", "rb") as f:
    db = pickle.load(f)

docs = [d.page_content for d in db.docstore._dict.values()]
embedder = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = embedder.encode(docs)

index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(np.array(doc_embeddings))

generator = pipeline("text-generation", model="gpt2")

def ask_lawyer(query):
    q_emb = embedder.encode([query])
    _, I = index.search(np.array(q_emb), 3)
    context = " ".join([docs[i] for i in I[0]])
    prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
    result = generator(prompt, max_length=200)
    return result[0]["generated_text"]

while True:
    q = input("Ask LawyerGPT: ")
    print(ask_lawyer(q))
