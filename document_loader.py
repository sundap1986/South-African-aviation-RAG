from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone
import cohere

COHERE_API_KEY = "oM70GeVxxAZV1QQTqyj9bRUuosSjMM2TaH0pzk6S"  # Or load from env
co = cohere.Client(COHERE_API_KEY)
# Pinecone DB config (matching pinecone_db.py)
PINECONE_API_KEY = "pcsk_5bniYy_ANb3USQRhm5jpSEjgWwpozEPbJXfHwepotrQR64293PjxWvT2gVXwscFLkkKz5B"
PINECONE_INDEX_NAME = "aviation-docs"

def summarize_chunk(chunk_text):
    response = co.summarize(
        text=chunk_text,
        model="command",  # or "command-light"
        length="medium",
        format="paragraph",
        extractiveness="auto"
    )
    return response.summary.strip()

def embed_chunk(chunk_text):
    response = co.embed(
        texts=[chunk_text],
        model="embed-english-v3.0",
        input_type="search_document"  # <-- Add this line
    )
    return response.embeddings[0]

def load_document(pdf):
    """
    Load a PDF, split into chunks, embed, summarize, and store in Pinecone.
    """
    # Load and split
    loader = PyPDFLoader(pdf)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)

    # Initialize Pinecone (connect to existing index)
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)

    for i, chunk in enumerate(chunks):
        chunk_text = chunk.page_content if hasattr(chunk, "page_content") else chunk
        # 1. Embed using Cohere
        vector = embed_chunk(chunk_text)
        # 2. Summarize using Cohere
        summary = summarize_chunk(chunk_text)
        # 3. Store in Pinecone
        metadata = {
            "summary": summary,
            "source": getattr(chunk, "metadata", {}),
            "chunk_id": i
        }
        index.upsert(vectors=[{
            "id": f"{pdf}_chunk_{i}",
            "values": vector,
            "metadata": metadata
        }])

    return chunks