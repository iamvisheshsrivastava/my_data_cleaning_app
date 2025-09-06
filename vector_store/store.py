import chromadb
from chromadb.utils import embedding_functions
import uuid

# Initialize Chroma persistent client
client = chromadb.PersistentClient(path="./vector_store")

# Default embedding function (can switch to OpenAI/HF later)
embedding_func = embedding_functions.DefaultEmbeddingFunction()

# Create or load collection
collection = client.get_or_create_collection(
    name="cleaning_suggestions",
    embedding_function=embedding_func
)

def add_suggestion(suggestion: str, metadata: dict):
    """
    Add a new cleaning suggestion to the vector DB.
    Each suggestion must have a unique metadata['id'].
    """
    # Ensure ID exists
    if "id" not in metadata:
        metadata["id"] = str(uuid.uuid4())

    collection.add(
        documents=[suggestion],
        metadatas=[metadata],
        ids=[metadata["id"]]
    )

def query_suggestions(query: str, n_results=3):
    """
    Query similar past cleaning suggestions.
    Returns a dictionary with 'documents' and 'metadatas'.
    """
    return collection.query(query_texts=[query], n_results=n_results)

# --- Seed dummy suggestions (only if DB is empty) ---
if collection.count() == 0:
    dummy_data = [
        ("Convert 'DOB' column to datetime", {"id": "seed1", "session_id": "seed", "instruction": "datetime"}),
        ("Drop columns with more than 50% missing values", {"id": "seed2", "session_id": "seed", "instruction": "drop_missing"}),
        ("Normalize numerical columns to 0-1 range", {"id": "seed3", "session_id": "seed", "instruction": "normalize"}),
        ("Trim whitespace in string columns", {"id": "seed4", "session_id": "seed", "instruction": "trim_whitespace"}),
    ]
    for text, meta in dummy_data:
        collection.add(documents=[text], metadatas=[meta], ids=[meta["id"]])
    print("✅ Seeded dummy suggestions into vector DB")
