import os
import json
import re
from tqdm import tqdm
import chromadb
from chromadb.config import Settings
from transformers import AutoTokenizer, AutoModel
import torch
from chromadb import PersistentClient
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load the BERT model and tokenizer
MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

ROOT_DIR = "Dataset"  # Path to your dataset folder
CHROMA_DB_DIR = "./chroma_db"  # Directory to store ChromaDB
COLLECTION_NAME = "legal_documents"  # Name of the ChromaDB collection
EMBEDDINGS_FILE = "embeddings.json"  # File to save/load embeddings

def ensure_chroma_db_dir(db_dir):
    """Ensure the ChromaDB directory exists."""
    absolute_path = os.path.abspath(db_dir)
    if not os.path.exists(db_dir):
        print(f"Creating ChromaDB directory at: {absolute_path}")
        os.makedirs(db_dir, exist_ok=True)
    else:
        print(f"ChromaDB directory exists at: {absolute_path}")

def clean_text(text):
    """Clean text to remove unwanted characters."""
    pattern = r"[\n\t\r\\]"
    if text:
        cleaned_text = re.sub(pattern, " ", text)
        cleaned_text = cleaned_text.replace("\\", "")
        return re.sub(r"\s+", " ", cleaned_text).strip()
    return text

def load_and_clean_data(input_dir):
    """Load and clean dataset from JSON files."""
    documents = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".json"):
                input_file_path = os.path.join(root, file)
                try:
                    with open(input_file_path, "r", encoding="utf-8") as infile:
                        data = json.load(infile)
                        if isinstance(data, dict) and "unofficial_text" in data:
                            cleaned_text = clean_text(data["unofficial_text"])
                            documents.append({"id": file, "content": cleaned_text})
                        elif isinstance(data, list):
                            for item in data:
                                if "unofficial_text" in item:
                                    cleaned_text = clean_text(item["unofficial_text"])
                                    documents.append({"id": file, "content": cleaned_text})
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON in file {input_file_path}: {e}")
    return documents

def generate_bert_embeddings(documents):
    """Generate embeddings using Legal-BERT."""
    successful_documents = []
    for doc in tqdm(documents, desc="Generating embeddings", unit="document"):
        try:
            inputs = tokenizer(doc["content"], return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
            doc["embedding"] = embedding
            successful_documents.append(doc)
        except Exception as e:
            print(f"Error generating embedding for document {doc['id']}: {e}")
    return successful_documents

def save_embeddings_to_file(documents, file_path):
    """Save embeddings to a JSON file."""
    with open(file_path, "w", encoding="utf-8") as outfile:
        json.dump(documents, outfile, ensure_ascii=False, indent=4)

def load_embeddings_from_file(file_path):
    """Load embeddings from a JSON file."""
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as infile:
            return json.load(infile)
    return None


# def save_to_chromadb(documents, db_dir, collection_name):
#     """Save embeddings to ChromaDB."""
#     ensure_chroma_db_dir(db_dir)
#     client = PersistentClient(path=db_dir)
#     collection = client.get_or_create_collection(collection_name)

#     for doc in documents:
#         if "embedding" in doc:
#             try:
#                 collection.add(
#                     ids=[doc["id"]],
#                     documents=[doc["content"]],
#                     metadatas=[{"source": "legal_dataset"}],
#                     embeddings=[doc["embedding"]]
#                 )
#                 print(f"Added document ID: {doc['id']} to ChromaDB")
#             except Exception as e:
#                 print(f"Error adding document ID {doc['id']}: {e}")
    
#     print(f"Embeddings saved to ChromaDB at {db_dir}")

#     # Debugging: Check collection existence
#     collections = client.list_collections()
#     print("Available collections:", [col.name for col in collections])

#     # Ensure proper shutdown
#     del client
#     print("ChromaDB client closed. Data should now be persisted to disk.")


def save_to_chromadb(documents, db_dir, collection_name):
    """Save embeddings to ChromaDB."""
    ensure_chroma_db_dir(db_dir)
    client = PersistentClient(path=db_dir)  # Initialize PersistentClient

    # Ensure collection exists or create it
    collection = client.get_or_create_collection(collection_name)

    for doc in documents:
        if "embedding" in doc:
            try:
                collection.add(
                    ids=[doc["id"]],
                    documents=[doc["content"]],
                    metadatas=[{"source": "legal_dataset"}],
                    embeddings=[doc["embedding"]]
                )
                print(f"Added document ID: {doc['id']} to ChromaDB")
            except Exception as e:
                print(f"Error adding document ID {doc['id']}: {e}")

    print(f"Embeddings saved to ChromaDB at {db_dir}")

    # Debugging: Check collection availability
    collections = client.list_collections()
    print("Available collections:", [col.name for col in collections])

    # client.close()  # Properly close the client
    # print("ChromaDB client closed. Data should now be persisted to disk.")

# def query_chromadb_with_context(query, tokenizer, model, db_dir, collection_name, top_k=5, filters=None):
#     """Query ChromaDB for relevant documents with optional filters."""
#     if not query:
#         raise ValueError("Query must be provided for retrieval!")

#     client = PersistentClient(
#         settings=Settings(
#             persist_directory=db_dir
#         )
#     )
#     collection = client.get_or_create_collection(collection_name)

#     # Generate query embedding
#     inputs = tokenizer(query, return_tensors="pt", truncation=True, max_length=512)
#     with torch.no_grad():
#         outputs = model(**inputs)
#         query_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

#     # Ensure filters is passed correctly or omitted
#     if filters:
#         print(f"Ignoring filters: {filters}. ChromaDB does not currently support dictionary filters.")
#         filters = None  # Remove filters for compatibility

#     # Search the collection
#     results = collection.query(
#         query_embeddings=[query_embedding],
#         n_results=top_k,
#         where=filters  # Ensure filters is None
#     )
#     documents = results["documents"]
#     embeddings = results["embeddings"]

#     # Combine embeddings with query-context pairs
#     query_context_pairs = [
#         {"query": query, "context": doc, "embedding": emb}
#         for doc, emb in zip(documents, embeddings)
#     ]
#     return query_context_pairs


def query_chromadb_with_context(query, tokenizer, model, db_dir, collection_name, top_k, filters=None):
    try:
        # Initialize the ChromaDB client
        client = PersistentClient(path=db_dir)
        collection = client.get_collection(collection_name)

        # Generate query embedding
        inputs = tokenizer(query, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            query_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()

        # Perform query
        results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

        # Flatten and process results
        documents = [doc for sublist in results.get("documents", []) for doc in sublist]
        metadatas = [meta for sublist in results.get("metadatas", []) for meta in sublist]
        distances = [dist for sublist in results.get("distances", []) for dist in sublist]

        return documents, metadatas, distances

    except Exception as e:
        raise RuntimeError(f"An error occurred while querying ChromaDB: {e}")



def query_and_rerank(query, tokenizer, model, db_dir, collection_name, top_k=5, filters=None):
    """Query ChromaDB and re-rank results."""
    if not query:
        raise ValueError("Query must be provided for re-ranking!")

    # Fetch query-context pairs from ChromaDB
    query_context_pairs = query_chromadb_with_context(query, tokenizer, model, db_dir, collection_name, top_k, filters)
    query_embedding = query_context_pairs[0]["embedding"]  # Use query embedding for re-ranking
    documents = [pair["context"] for pair in query_context_pairs]
    embeddings = [pair["embedding"] for pair in query_context_pairs]

    # Re-rank results using cosine similarity
    reranked_results = rerank_documents(query_embedding, documents, embeddings)
    return reranked_results

def rerank_documents(query_embedding, documents, embeddings):
    """
    Re-rank documents based on cosine similarity between query and document embeddings.

    Parameters:
        query_embedding: The embedding vector for the query.
        documents: List of document texts.
        embeddings: List of embedding vectors corresponding to the documents.

    Returns:
        A list of dictionaries with re-ranked documents and their scores.
    """
    if not query_embedding or not documents or not embeddings:
        raise ValueError("Query embedding, documents, and embeddings must be provided.")

    # Compute cosine similarity between the query embedding and document embeddings
    query_embedding = np.array(query_embedding).reshape(1, -1)
    document_embeddings = np.array(embeddings)
    scores = cosine_similarity(query_embedding, document_embeddings).flatten()

    # Combine scores with documents
    reranked_results = [
        {"document": doc, "score": score}
        for doc, score in zip(documents, scores)
    ]

    # Sort results by score in descending order
    reranked_results = sorted(reranked_results, key=lambda x: x["score"], reverse=True)
    return reranked_results





if __name__ == "__main__":
    try:
        # Ensure ChromaDB directory exists
        ensure_chroma_db_dir(CHROMA_DB_DIR)

        # Step 1: Load and Clean Dataset
        print("Cleaning and loading dataset...")
        cleaned_data = load_and_clean_data(ROOT_DIR)
        print(f"Number of documents loaded: {len(cleaned_data)}")

        # Check if embeddings already exist
        print("Checking for existing embeddings...")
        embeddings = load_embeddings_from_file(EMBEDDINGS_FILE)
        if embeddings:
            print(f"Loaded {len(embeddings)} embeddings from file.")
        else:
            print("Generating embeddings with Legal-BERT model...")
            embeddings = generate_bert_embeddings(cleaned_data)
            save_embeddings_to_file(embeddings, EMBEDDINGS_FILE)
            print(f"Generated and saved {len(embeddings)} embeddings.")

        # Step 3: Save Embeddings to ChromaDB
        print("Saving embeddings to ChromaDB...")
        save_to_chromadb(embeddings, CHROMA_DB_DIR, COLLECTION_NAME)

    except Exception as e:
        print(f"An error occurred: {e}")

    if os.path.exists(CHROMA_DB_DIR):
        print("ChromaDB directory exists. Contents:")
        print(os.listdir(CHROMA_DB_DIR))
    else:
        print("ChromaDB directory does not exist.")