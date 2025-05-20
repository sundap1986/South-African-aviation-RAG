import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from pinecone import Pinecone
import cohere
from PyPDF2 import PdfReader
import re
from uuid import uuid4

# --- Load environment variables ---
load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "aviation-docs")

co = cohere.Client(COHERE_API_KEY)

# --- Helper: Clean & Normalize Text ---
def clean_text(text):
    lines = text.splitlines()
    line_counts = {}
    for line in lines:
        line = line.strip()
        if line:
            line_counts[line] = line_counts.get(line, 0) + 1
    threshold = max(2, len(lines) // 20)
    cleaned = [l for l in lines if line_counts.get(l.strip(), 0) < threshold]
    text = "\n".join(cleaned)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# --- Helper: Extract text from PDF (OCR-ready) ---
def extract_pdf_text(file_path):
    reader = PdfReader(file_path)
    all_text = []
    for page in reader.pages:
        text = page.extract_text() or ""
        all_text.append(text)
    return "\n".join(all_text)

# --- Helper: Chunking with overlap and tagging ---
def chunk_text(text, chunk_size=1000, overlap=200):
    words = text.split()
    chunks = []
    i = 0
    chunk_id = 1
    while i < len(words):
        chunk_words = words[i:i+chunk_size]
        chunk = " ".join(chunk_words)
        tag = f"chunk-{chunk_id}"
        chunks.append({"id": tag, "text": chunk})
        i += chunk_size - overlap
        chunk_id += 1
    return chunks

# --- Helper: Summarize chunk via Cohere ---
def summarize_chunk(chunk_text):
    response = co.summarize(
        text=chunk_text,
        model="command",  # or "command-light" for faster/cheaper
        length="medium",
        format="paragraph",
        extractiveness="auto"
    )
    return response.summary.strip()

# --- Helper: Embed chunk via Cohere ---
def embed_chunk(chunk_text):
    response = co.embed(
        texts=[chunk_text],
        model="embed-english-v3.0",
        input_type="search_document"  # <-- This is correct!
    )
    return response.embeddings[0]

# --- Pinecone: Initialize and upsert ---
def upsert_chunks_to_pinecone(chunks):
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX)
    vectors = []
    for chunk in chunks:
        embedding = embed_chunk(chunk["summary"])
        meta = {
            "id": chunk["id"],
            "summary": chunk["summary"],
            "text": chunk["text"]
        }
        vectors.append({
            "id": str(uuid4()),
            "values": embedding,
            "metadata": meta
        })
    index.upsert(vectors=vectors)

# --- Pinecone: Query ---
def query_pinecone(query, top_k=3):
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX)
    query_embedding = embed_chunk(query)
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    return [match['metadata']['summary'] for match in results['matches']]

# --- Streamlit UI ---
st.title("Aviation Document Q&A (Cohere + Pinecone)")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
        temp.write(uploaded_file.getvalue())
        temp_file_path = temp.name

    st.write("Extracting and cleaning text...")
    raw_text = extract_pdf_text(temp_file_path)
    cleaned_text = clean_text(raw_text)

    st.write("Chunking and summarizing...")
    chunks = chunk_text(cleaned_text, chunk_size=1000, overlap=200)
    for chunk in chunks:
        chunk["summary"] = summarize_chunk(chunk["text"])

    st.write("Storing in Pinecone...")
    upsert_chunks_to_pinecone(chunks)
    st.success("Document processed and indexed!")

    st.session_state["doc_uploaded"] = True

if st.session_state.get("doc_uploaded"):
    question = st.text_input("Ask a question about the document:")
    if question:
        st.write("Searching for relevant summaries...")
        summaries = query_pinecone(question)
        for i, summary in enumerate(summaries, 1):
            st.markdown(f"**Relevant Summary {i}:** {summary}")

# --- End of file ---