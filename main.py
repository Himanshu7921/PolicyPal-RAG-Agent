# -----------------------------
# Imports for RAG Integration
# -----------------------------
from RetrievalMind.embeddings_manager import EmbeddingManager
from RetrievalMind.vector_store_manager import VectorStore
from RetrievalMind.rag_retriver import Retrieval
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader

# -----------------------------
# Imports for AI Agent
# -----------------------------
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from langchain.memory.summary import ConversationSummaryMemory
from langchain.chains import ConversationChain
from langchain_core.runnables import Runnable
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableLambda
import os

# -----------------------------
# Document Loading and Embedding
# -----------------------------
def load_pdf_documents(pdf_folder_path: str, file_pattern: str = "**/*.pdf") -> list:
    """
    Load all PDF documents from a folder and return them as chunks.

    Args:
        pdf_folder_path (str): Path to the folder containing PDFs.
        file_pattern (str): Glob pattern to match PDF files (default '**/*.pdf').

    Returns:
        list: List of document chunks with page content and metadata.
    """
    loader = DirectoryLoader(
        path=pdf_folder_path,
        loader_cls=PyMuPDFLoader,
        glob=file_pattern
    )
    document_chunks = loader.load()
    print(f"[INFO] Loaded {len(document_chunks)} document chunks from {pdf_folder_path}")

    # Debug: preview first 5 document chunks
    for idx, chunk in enumerate(document_chunks[:5]):
        print(f"Document Chunk {idx} preview:", repr(chunk.page_content[:100]))
    
    return document_chunks


def generate_document_embeddings(document_chunks: list, embedding_model: str = "all-miniLM-L6-v2") -> tuple:
    """
    Generate embeddings for a list of document chunks using a SentenceTransformer model.

    Args:
        document_chunks (list): List of document chunks.
        embedding_model (str): Name of the embedding model (default "all-miniLM-L6-v2").

    Returns:
        tuple: embeddings list, EmbeddingManager instance
    """
    texts = [chunk.page_content for chunk in document_chunks]
    embedding_manager = EmbeddingManager(model_name=embedding_model)
    embeddings = embedding_manager.generate_embeddings(texts)
    print(f"[INFO] Generated embeddings for {len(texts)} chunks using model '{embedding_model}'")
    return embeddings, embedding_manager


def store_documents_in_vector_store(document_chunks: list, embeddings: list, collection_name: str, persist_dir: str, doc_type: str = "PDF") -> VectorStore:
    """
    Store document chunks and embeddings in a ChromaDB vector store.

    Args:
        document_chunks (list): List of document chunks.
        embeddings (list): Corresponding embeddings for each chunk.
        collection_name (str): Name of the vector store collection.
        persist_dir (str): Directory where the vector store is persisted.
        doc_type (str): Type of documents being stored (default "PDF").

    Returns:
        VectorStore: Initialized and populated vector store instance.
    """
    vector_store = VectorStore(
        collection_name=collection_name,
        persist_directory=persist_dir,
        document_type=doc_type
    )

    vector_store.add_document(documents=document_chunks, embeddings=embeddings)
    print(f"[INFO] Added {len(document_chunks)} documents to vector store '{collection_name}'")
    return vector_store


def query_vector_store(vector_store: VectorStore, embedding_manager: EmbeddingManager, query_text: str, top_k: int = 3, min_score: float = 0.3) -> list:
    """
    Perform semantic search on the vector store.

    Args:
        vector_store (VectorStore): Initialized vector store instance.
        embedding_manager (EmbeddingManager): Embedding generator.
        query_text (str): Query string for retrieval.
        top_k (int): Number of top results to return.
        min_score (float): Minimum similarity score threshold.

    Returns:
        list: List of retrieved documents with metadata and similarity scores.
    """
    retriever = Retrieval(vector_store=vector_store, embedding_manager=embedding_manager)
    results = retriever.retrieve(query=query_text, top_k=top_k, score_threshold=min_score)
    print(f"[INFO] Retrieved {len(results)} results for query: '{query_text}'")
    return results

# -----------------------------
# RAG + LLM Integration
# -----------------------------
def initialize_llm(api_key_env_var: str = "Gemini_APIKEY") -> GoogleGenerativeAI:
    """
    Initialize the Google Gemini LLM using the API key from environment variables.
    """
    api_key = os.getenv(api_key_env_var)
    if not api_key:
        raise ValueError(f"API key not found in environment variable '{api_key_env_var}'")
    return GoogleGenerativeAI(model="gemini-2.5-flash", api_key=api_key)


def get_prompt() -> ChatPromptTemplate:
    """
    Returns a ChatPromptTemplate for the RAG AI assistant specialized in company policies.
    """
    return ChatPromptTemplate([
        ('system', """You are a helpful AI assistant specialized in company policies. 
        Your job is to answer employee queries based strictly on the provided policy documents. 
        Always respond in clear, simple, and professional language suitable for employees."""),

        ('user', """Here are the retrieved company policy documents relevant to the query:\n
        {retrieved_docs_from_rag}\n
        Employee Query: {user_query}\n
        Instructions: 
        - Answer the query based ONLY on the documents above.
        - If the answer is not found, politely say that the information is unavailable.
        - Keep the answer concise and easy to understand.
        - Optionally, reference the section or document name if it helps clarity.
        - Do not provide unrelated information.
        Provide your answer below:""")
    ])


def get_chain(llm: GoogleGenerativeAI, prompt_template: ChatPromptTemplate) -> Runnable:
    """
    Create a Runnable chain combining the prompt template, LLM, and output parser.
    """
    return prompt_template | llm | StrOutputParser()


def ask_with_rag(chain: Runnable, query: str, retrieved_docs: list) -> str:
    """
    Generate an AI answer for a query based on RAG retrieved documents.
    """
    docs_text = "\n".join([doc['content'] for doc in retrieved_docs])
    input_mapping = {
        "user_query": query,
        "retrieved_docs_from_rag": docs_text
    }
    return chain.invoke(input_mapping)

# -----------------------------
# Main Execution
# -----------------------------
def main(query: str):
    """
    Full RAG pipeline: load documents, generate embeddings, store/retrieve, and prepare for LLM query.
    """
    # Configuration
    pdf_folder = "data/"
    vector_collection_name = "policy_pal_vector_collection"
    vector_store_directory = "data/policy_pal_vector_store"

    # Step 1: Load PDF documents (example: "Travel Policy", "Expense Policy", "HR Guidelines")
    pdf_chunks = load_pdf_documents(pdf_folder)

    # Step 2: Generate embeddings
    embeddings, embedding_manager = generate_document_embeddings(pdf_chunks)

    # Step 3: Store documents in vector store
    vector_store = store_documents_in_vector_store(pdf_chunks, embeddings, vector_collection_name, vector_store_directory)

    # Step 4: Retrieve relevant documents
    retrieved_docs = query_vector_store(vector_store, embedding_manager, query_text=query, top_k=1, min_score=0.0)

    # Preview retrieved docs
    for doc in retrieved_docs:
        print("\n--- Retrieved Document ---")
        print("ID:", doc['id'])
        print("Similarity Score:", doc['similarity_score'])
        print("Content Preview:", doc['content'][:500], "...")
    return retrieved_docs


if __name__ == "__main__":
    query_text = "Travel reimbursement process for Finance"

    # Step 1-4: Run RAG pipeline
    retrieved_docs = main(query=query_text)
    print(f"Type of 'retrieved_docs' = {type(retrieved_docs)}")

    # Step 5: Initialize LLM
    llm_model = initialize_llm()

    # Step 6: Prepare chain
    prompt_template = get_prompt()
    rag_chain = get_chain(llm_model, prompt_template)

    # Step 7: Ask query
    answer = ask_with_rag(rag_chain, query=query_text, retrieved_docs=retrieved_docs)
    print("\n--- AI Answer ---\n", answer)