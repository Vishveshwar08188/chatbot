import streamlit as st
from pdf_reader import load_pdf_text
from rag_chatbot import create_vector_store, retrieve_relevant_chunks, generate_answer

st.title("📄🔍 PDF RAG Chatbot with TinyLLaMA")

uploaded_pdf = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_pdf:
    with open("uploaded.pdf", "wb") as f:
        f.write(uploaded_pdf.read())

    text = load_pdf_text("uploaded.pdf")
    text_chunks = [text[i:i+500] for i in range(0, len(text), 500)]

    st.success("✅ PDF loaded and split into chunks.")

    index, chunks, _ = create_vector_store(text_chunks)

    query = st.text_input("Ask a question:")

    if query:
        top_chunks = retrieve_relevant_chunks(query, index, chunks)
        answer = generate_answer(query, top_chunks)
        st.markdown("### 🧠 Answer:")
        st.write(answer)
