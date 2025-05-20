from pinecone import Pinecone, ServerlessSpec

# Use the same Pinecone API key and index name as in the searae Pinecone DB file
PINECONE_API_KEY = "pcsk_5bniYy_ANb3USQRhm5jpSEjgWwpozEPbJXfHwepotrQR64293PjxWvT2gVXwscFLkkKz5B"
INDEX_NAME = "aviation-docs"

pc = Pinecone(api_key=PINECONE_API_KEY)

# Check if the index already exists before creating
if INDEX_NAME not in [idx['name'] for idx in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=1024, 
        metric="cosine", 
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
else:
    print(f"Index '{INDEX_NAME}' already exists.")