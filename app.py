import os
from dotenv import load_dotenv
import openai
import streamlit as st
from chromadb import PersistentClient
from transformers import AutoTokenizer, AutoModel
from main_script import query_and_rerank
from evaluation import evaluate_bleu
from main_script import query_chromadb_with_context

# Load environment variables
load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Verify OpenAI API key
if not openai.api_key:
    st.error("OpenAI API key is missing! Add it to the .env file.")

# Constants
CHROMA_DB_DIR = "./chroma_db"  # ChromaDB directory
COLLECTION_NAME = "legal_documents"  # ChromaDB collection name
GPT_MODEL_NAME = "gpt-4"  # GPT-4 for generation
TOP_K = 5  # Number of top documents to retrieve

# Initialize ChromaDB client
@st.cache_resource
def initialize_chromadb_client():
    client = PersistentClient(path=CHROMA_DB_DIR)
    # Ensure collection exists
    collections = client.list_collections()
    if COLLECTION_NAME not in [col.name for col in collections]:
        st.warning(f"Collection '{COLLECTION_NAME}' not found. Creating it now.")
        client.get_or_create_collection(COLLECTION_NAME)
    return client

client = initialize_chromadb_client()

# Load Legal-BERT tokenizer and model
@st.cache_resource
def load_legal_bert():
    MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    return tokenizer, model

tokenizer, model = load_legal_bert()

# Function: Query GPT-4 with context
def query_gpt_4(prompt):
    response = openai.ChatCompletion.create(
        model=GPT_MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a helpful legal assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,  # Adjust as needed
        temperature=0.7
    )
    return response['choices'][0]['message']['content'].strip()

# Function: Build and run RAG pipeline
def run_rag_pipeline(query):
    # Retrieve documents, metadata, and distances
    documents, metadatas, distances = query_chromadb_with_context(
        query=query,
        tokenizer=tokenizer,
        model=model,
        db_dir=CHROMA_DB_DIR,
        collection_name=COLLECTION_NAME,
        top_k=TOP_K,
        filters=None
    )

    if not documents:
        raise RuntimeError("No documents retrieved from ChromaDB.")

    # Limit context size
    max_context_tokens = 4000
    context = " ".join(documents[:TOP_K])[:max_context_tokens]

    prompt = f"Query: {query}\nContext: {context}\nAnswer:"

    # Generate response with GPT-4
    gpt_response = query_gpt_4(prompt)

    # Return all necessary values
    return gpt_response, documents, metadatas, distances

# Function: Display results
def display_results(gpt_response, documents, metadatas, distances):
    st.subheader("GPT-4 Generated Response")
    st.write(gpt_response)

    st.subheader("Top Retrieved Documents")
    for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
        st.write(f"**Rank {i + 1}:**")
        st.write(f"Document: {doc}")
        st.write(f"Metadata: {meta}")
        st.write(f"Distance: {dist:.4f}")

# Streamlit UI
st.title("Legal Document Query System with GPT-4 and BLEU Evaluation")
st.sidebar.header("Options")

# User input
query_input = st.text_area("Enter your legal query:", "")
run_query = st.button("Run Query")
evaluate = st.button("Evaluate Performance")

# Handle query execution
if run_query and query_input.strip():
    st.info("Running query...")
    try:
        # Call the pipeline and display results
        gpt_response, documents, metadatas, distances = run_rag_pipeline(query_input)
        display_results(gpt_response, documents, metadatas, distances)
    except Exception as e:
        # Handle and display errors
        st.error(f"An error occurred: {e}")

# Performance Evaluation
if evaluate:
    st.info("Evaluating performance with BLEU metrics...")
    
    # Example reference and generated answers
    reference_answers = [
        "Legal due process ensures fair treatment through the judicial system.",
        "Tort law deals with civil wrongs and provides remedies to those harmed."
    ]
    generated_answers = [
        "Legal due process guarantees fair treatment in the judicial system.",
        "Tort law addresses civil wrongs and offers remedies to harmed individuals."
    ]
    
    # Evaluate BLEU scores
    results = evaluate_bleu(reference_answers, generated_answers)
    st.subheader("Performance Evaluation")
    st.write("BLEU Scores for Generated Answers:")
    for i, score in enumerate(results["individual_scores"]):
        st.write(f"Query {i + 1}: BLEU Score = {score:.4f}")
    st.write(f"Average BLEU Score: {results['average_score']:.4f}")
