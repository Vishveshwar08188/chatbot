from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import faiss
import torch

llm_id = "sshleifer/tiny-gpt2"
tokenizer = AutoTokenizer.from_pretrained(llm_id)
model = AutoModelForCausalLM.from_pretrained(llm_id)

# Load sentence embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def create_vector_store(text_chunks):
    embeddings = embedder.encode(text_chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, text_chunks

def retrieve_relevant_chunks(query, index, chunks, k=3):
    query_embedding = embedder.encode([query])
    _, indices = index.search(query_embedding, k)
    return [chunks[i] for i in indices[0]]

def generate_answer(query, context_chunks):
    context = "\n".join(context_chunks)
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output_ids = model.generate(input_ids, max_new_tokens=200)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)
