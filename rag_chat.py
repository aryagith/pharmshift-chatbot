import os
import warnings
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI # We use the OpenAI client structure
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_FILENAME = "my_document.pdf" # <--- Make sure this matches your actual PDF filename
PDF_PATH = os.path.join(SCRIPT_DIR, PDF_FILENAME)
VECTOR_STORE_PATH = "faiss_index_pdf" # Directory to save/load the index
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
SEARCH_K = 4 # Number of relevant chunks to retrieve

# DeepSeek Configuration (uses OpenAI compatible endpoint)
DEEPSEEK_MODEL_NAME = "deepseek-chat" # Or "deepseek-coder" if appropriate
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

# --- Helper Functions ---

def load_api_key():
    """Loads DeepSeek API key from .env file."""
    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY not found in .env file. Please add it.")
    return api_key

def create_or_load_vector_store(pdf_path, store_path, embedding_model_name):
    """Creates a vector store from PDF or loads if it exists."""
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

    if os.path.exists(store_path):
        print(f"Loading existing vector store from: {store_path}")
        try:
            vectorstore = FAISS.load_local(store_path, embeddings, allow_dangerous_deserialization=True) # Added for newer Langchain versions
            print("Vector store loaded successfully.")
            return vectorstore
        except Exception as e:
            print(f"Error loading vector store: {e}. Recreating...")
            # Optionally remove the corrupted store_path directory here
            # import shutil
            # shutil.rmtree(store_path)


    print(f"Creating new vector store from PDF: {pdf_path}")
    if not os.path.exists(pdf_path):
         raise FileNotFoundError(f"PDF file not found at: {pdf_path}")

    # 1. Load PDF
    print("Loading PDF...")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    if not pages:
        raise ValueError("Could not load any pages from the PDF. Is it empty or corrupted?")
    print(f"Loaded {len(pages)} pages.")

    # 2. Split Text
    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    docs = text_splitter.split_documents(pages)
    if not docs:
        raise ValueError("Text splitting resulted in zero documents. Check PDF content and splitter settings.")
    print(f"Split into {len(docs)} chunks.")

    # 3. Create Embeddings & Vector Store
    print("Creating embeddings and vector store (this may take a while)...")
    vectorstore = FAISS.from_documents(docs, embeddings)
    print("Vector store created.")

    # 4. Save Vector Store (Optional but recommended)
    try:
        print(f"Saving vector store to: {store_path}")
        vectorstore.save_local(store_path)
        print("Vector store saved.")
    except Exception as e:
        print(f"Error saving vector store: {e}")
        warnings.warn("Could not save the vector store. It will need to be recreated next time.")


    return vectorstore

def setup_rag_chain(api_key, vectorstore):
    """Sets up the RAG chain with DeepSeek LLM."""
    # Initialize DeepSeek LLM using the OpenAI client structure
    llm = ChatOpenAI(
        model=DEEPSEEK_MODEL_NAME,
        api_key=api_key,
        base_url=DEEPSEEK_BASE_URL,
        temperature=0.1, # Lower temperature for more factual answers
        # max_tokens=500 # Optional: limit response length
    )

    # Setup Retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": SEARCH_K})

    # Define a custom prompt template (Optional but good practice)
    prompt_template = """You are an assistant tasked with answering questions based ONLY on the provided context.
If the answer is not found in the context, state that you don't know. Do not make up information.
Be concise and directly answer the question using the context.

Context:
{context}

Question:
{question}

Answer:"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # Create RetrievalQA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", # "stuff" puts all retrieved chunks into the context
        retriever=retriever,
        return_source_documents=True, # Set to True to see which chunks were used
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain

# --- Main Execution ---
if __name__ == "__main__":
    try:
        deepseek_api_key = load_api_key()

        # Create or load the vector store
        vector_store = create_or_load_vector_store(PDF_PATH, VECTOR_STORE_PATH, EMBEDDING_MODEL)

        # Setup the RAG chain
        rag_chain = setup_rag_chain(deepseek_api_key, vector_store)

        print("\n--- PDF RAG Chatbot Ready ---")
        print("Enter your questions about the PDF. Type 'exit' or 'quit' to stop.")

        while True:
            query = input("\nAsk your question: ")
            if query.lower() in ["exit", "quit"]:
                break
            if not query:
                continue

            # Run the chain
            print("Thinking...")
            try:
                result = rag_chain.invoke({"query": query}) # Use invoke for newer Langchain versions

                print("\nAnswer:")
                print(result["result"])

                # Optional: Display source documents
                # print("\nSources:")
                # for i, doc in enumerate(result["source_documents"]):
                #     # Limit printing very long chunks
                #     content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                #     print(f"--- Source {i+1} (Page {doc.metadata.get('page', 'N/A')}) ---")
                #     print(content_preview)
                #     print("-" * 20)

            except Exception as e:
                 print(f"\nAn error occurred while processing the query: {e}")
                 print("Please check your API key, network connection, and the query.")


    except (ValueError, FileNotFoundError) as e:
        print(f"\nError during setup: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

    print("\nChatbot session ended.")