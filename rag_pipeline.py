import os
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings

# Set environment variables for OpenAI API
os.environ["OPENAI_API_KEY"] = "your_openai_api_key"  # Replace with your OpenAI API key

# Directory for ChromaDB
CHROMA_DB_DIR = "./chroma_db"
COLLECTION_NAME = "legal_documents"

def initialize_retriever():
    """
    Initialize the ChromaDB retriever for LangChain.
    """
    # Load ChromaDB as the retriever
    retriever = Chroma(
        persist_directory=CHROMA_DB_DIR,
        collection_name=COLLECTION_NAME,
        embedding_function=HuggingFaceEmbeddings(model_name="nlpaueb/legal-bert-base-uncased"),
    )
    return retriever.as_retriever(search_kwargs={"k": 5})  # Retrieve top-5 relevant documents


def build_rag_pipeline():
    """
    Build a Retrieval-Augmented Generation (RAG) pipeline using LangChain.
    """
    # Initialize the retriever
    retriever = initialize_retriever()

    # Load GPT-4 as the generator
    llm = OpenAI(model="gpt-4", temperature=0.0)  # Set `temperature=0.0` for consistent outputs

    # Define a prompt template for generation
    prompt_template = PromptTemplate(
        input_variables=["context", "query"],
        template=(
            "You are an AI legal assistant. Use the context below to answer the question accurately.\n\n"
            "Context: {context}\n\n"
            "Question: {query}\n\n"
            "Answer:"
        ),
    )

    # Combine retriever and generator into a single chain
    rag_pipeline = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",  # This combines the context into a single prompt
        chain_type_kwargs={"prompt": prompt_template},
    )

    return rag_pipeline


def query_rag_pipeline(pipeline, query):
    """
    Perform a RAG pipeline query using the user input.

    Args:
        pipeline: The initialized RAG pipeline.
        query: The user's input query.

    Returns:
        Generated answer from GPT-4.
    """
    if not query.strip():
        raise ValueError("Query cannot be empty.")

    # Pass the query to the pipeline for processing
    return pipeline.run({"query": query})


if __name__ == "__main__":
    # Build the RAG pipeline
    print("Initializing RAG pipeline...")
    rag_pipeline = build_rag_pipeline()

    # Example query
    user_query = "What is legal due process?"
    print(f"User query: {user_query}")

    # Get the result
    print("Fetching answer...")
    answer = query_rag_pipeline(rag_pipeline, user_query)
    print(f"Answer: {answer}")
