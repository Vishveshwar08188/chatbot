import streamlit as st
from pdf_reader import load_pdf_text
from rag_chatbot import create_vector_store, retrieve_relevant_chunks, generate_answer

st.set_page_config(page_title="RAG PDF Chatbot")

st.title("📄🤖 RAG Chatbot with TinyLLaMA")
st.markdown("Upload a PDF and ask questions. Uses RAG and LLM to answer.")

uploaded_file = st.file_uploader("📤 Upload your PDF", type=["pdf"])

if uploaded_file:
    text = load_pdf_text(uploaded_file)
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]

    st.success("✅ PDF loaded and processed.")

    index, chunks = create_vector_store(chunks)

    query = st.text_input("❓ Ask a question from the PDF:")

    if query:
        top_chunks = retrieve_relevant_chunks(query, index, chunks)
        answer = generate_answer(query, top_chunks)
        st.markdown("### 🧠 Answer:")
        st.write(answer)
