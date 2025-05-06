import os
import warnings
import hashlib # For creating a unique ID for document sets
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader # Using PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import traceback # For detailed error printing

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- List your PDF filenames here ---
PDF_FILENAMES = ["my_document.pdf", "diabetes_care_devices.pdf"] # <--- Add your PDF filenames
# PDF_FILENAMES = ["my_document.pdf"] # Or just one if you prefer for now

# Generate full paths
PDF_PATHS = [os.path.join(SCRIPT_DIR, fname) for fname in PDF_FILENAMES]

# Function to create a unique ID for a set of documents
def get_doc_set_id(filepaths):
    """Creates a short hash ID from a list of filepaths."""
    sorted_names = sorted([os.path.basename(f) for f in filepaths])
    m = hashlib.md5()
    m.update("||".join(sorted_names).encode('utf-8'))
    return m.hexdigest()[:8]

DOC_SET_ID = get_doc_set_id(PDF_PATHS)
VECTOR_STORE_BASE_DIR = "faiss_indexes"
VECTOR_STORE_PATH = os.path.join(SCRIPT_DIR, VECTOR_STORE_BASE_DIR, f"idx_{DOC_SET_ID}_pymupdf")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TEMPERATURE = 0.2
SEARCH_K = 8

# DeepSeek Configuration
DEEPSEEK_MODEL_NAME = "deepseek-chat"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

# --- Helper Functions ---

def load_api_key():
    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY not found in .env file.")
    return api_key

def create_or_load_vector_store(pdf_filepaths, store_path, embedding_model_name):
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

    if os.path.exists(store_path):
        print(f"Loading existing vector store from: {store_path}")
        try:
            vectorstore = FAISS.load_local(store_path, embeddings, allow_dangerous_deserialization=True)
            print("Vector store loaded successfully.")
            return vectorstore
        except Exception as e:
            print(f"Error loading vector store: {e}. Recreating...")

    print(f"Creating new vector store for document set ID: {DOC_SET_ID} using PyMuPDFLoader")
    all_docs_from_all_pdfs = []

    for pdf_path in pdf_filepaths:
        if not os.path.exists(pdf_path):
            warnings.warn(f"PDF file not found at: {pdf_path}. Skipping this file.")
            continue

        print(f"Loading PDF with PyMuPDFLoader: {os.path.basename(pdf_path)}...")
        loader = PyMuPDFLoader(pdf_path)
        try:
            pages = loader.load()
        except Exception as e:
            warnings.warn(f"Could not load PDF {os.path.basename(pdf_path)} using PyMuPDFLoader: {e}. Skipping.")
            continue

        if not pages:
            warnings.warn(f"PyMuPDFLoader returned no pages from PDF: {os.path.basename(pdf_path)}. Skipping.")
            continue
        print(f"Loaded {len(pages)} pages from {os.path.basename(pdf_path)}.")

        for page_doc in pages:
            page_doc.metadata["source_document"] = os.path.basename(pdf_path)
            # PyMuPDFLoader should automatically add 'source' (full path) and 'page_number' (0-indexed).
            # If 'page_number' is missing, it could be an issue with the loader version or the PDF.

        print(f"Splitting text from {os.path.basename(pdf_path)} into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )
        docs = text_splitter.split_documents(pages)
        if not docs:
            warnings.warn(f"Text splitting resulted in zero documents for {os.path.basename(pdf_path)}.")
            continue
        print(f"Split {os.path.basename(pdf_path)} into {len(docs)} chunks.")
        all_docs_from_all_pdfs.extend(docs)

    if not all_docs_from_all_pdfs:
        raise ValueError("No documents were processed from any PDF. Check PDF files, paths, and PyMuPDFLoader compatibility.")

    print(f"Total chunks from all documents: {len(all_docs_from_all_pdfs)}")
    print("Creating embeddings and vector store (this may take a while)...")
    vectorstore = FAISS.from_documents(all_docs_from_all_pdfs, embeddings)
    print("Vector store created.")

    try:
        os.makedirs(os.path.dirname(store_path), exist_ok=True)
        print(f"Saving vector store to: {store_path}")
        vectorstore.save_local(store_path)
        print("Vector store saved.")
    except Exception as e:
        print(f"Error saving vector store: {e}")
        warnings.warn(f"Could not save the vector store at {store_path}.")

    return vectorstore

def setup_rag_chain(api_key, vectorstore):
    llm = ChatOpenAI(
        model=DEEPSEEK_MODEL_NAME,
        api_key=api_key,
        base_url=DEEPSEEK_BASE_URL,
        temperature=TEMPERATURE,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": SEARCH_K}, search_type='mmr')

    prompt_template = """You are an assistant tasked with answering questions based ONLY on the provided context from one or more documents.
If the answer is not found in the context, state that you don't know the answer based on the provided documents. Do not make up information.
Be concise and directly answer the question using the context. If multiple documents provide relevant information, synthesize it.

Context:
{context}

Question:
{question}

Answer:"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain

# --- Main Execution ---
if __name__ == "__main__":
    try:
        deepseek_api_key = load_api_key()

        os.makedirs(os.path.join(SCRIPT_DIR, VECTOR_STORE_BASE_DIR), exist_ok=True)

        vector_store = create_or_load_vector_store(PDF_PATHS, VECTOR_STORE_PATH, EMBEDDING_MODEL)

        rag_chain = setup_rag_chain(deepseek_api_key, vector_store)

        print("\n--- Multi-PDF RAG Chatbot Ready (using PyMuPDFLoader) ---")
        print(f"Using vector store: {VECTOR_STORE_PATH}")
        print("Enter your questions. Type 'exit' or 'quit' to stop.")

        while True:
            query = input("\nAsk your question: ")
            if query.lower() in ["exit", "quit"]:
                break
            if not query:
                continue

            print("Thinking...")
            try:
                result = rag_chain.invoke({"query": query})

                print("\nAnswer:")
                print(result["result"])

                print("\nSources:")
                if "source_documents" in result and result["source_documents"]:
                    for i, doc in enumerate(result["source_documents"]):
                        content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                        
                        # --- MODIFIED/ENHANCED Page Number Logic & Debugging ---
                        source_doc_name = doc.metadata.get('source_document', os.path.basename(doc.metadata.get('source', 'Unknown Document')))
                        
                        page_num_display = 'N/A'
                        page_num_0_indexed = -1 # Default to -1 if not found or not an int

                        # Attempt to get 'page_number' (standard for PyMuPDFLoader in recent langchain)
                        if 'page_number' in doc.metadata:
                            page_num_0_indexed = doc.metadata['page_number']
                        # Fallback: attempt to get 'page' (might be used by other loaders or older versions)
                        elif 'page' in doc.metadata:
                            page_num_0_indexed = doc.metadata['page']
                        
                        # Convert to 1-indexed for display if a valid 0-indexed number was found
                        if isinstance(page_num_0_indexed, int) and page_num_0_indexed >= 0:
                            page_num_display = str(page_num_0_indexed + 1)
                        elif isinstance(page_num_0_indexed, str) and page_num_0_indexed.isdigit(): # Handle if it's a string digit
                            try:
                                page_num_display = str(int(page_num_0_indexed) + 1)
                            except ValueError:
                                pass # Keep as N/A if conversion fails

                        # For debugging: print all metadata for the first source document
                        if i == 0: # Only print for the first source doc to reduce clutter
                            print(f"    [DEBUG] Metadata for first source doc chunk: {doc.metadata}")
                        # --- End MODIFIED/ENHANCED Page Number Logic ---

                        print(f"--- Source {i+1} (File: {source_doc_name}, Page: {page_num_display}) ---")
                        print(content_preview)
                        print("-" * 20)
                else:
                    print("No source documents found for this answer.")

            except Exception as e:
                 print(f"\nAn error occurred while processing the query: {e}")
                 traceback.print_exc()
                 print("Please check your API key, network connection, and the query.")

    except (ValueError, FileNotFoundError) as e:
        print(f"\nError during setup: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        traceback.print_exc()

    print("\nChatbot session ended.")