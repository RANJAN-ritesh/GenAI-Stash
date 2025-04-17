import os
import glob
from typing import List
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain.vectorstores.faiss import FAISS

# Load environment variables
load_dotenv()

# Check for API key
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("Please set GOOGLE_API_KEY environment variable in .env file")

def load_documents() -> List[Document]:
    """Load documents from the docs folder"""
    documents = []
    
    # Create docs directory if it doesn't exist
    if not os.path.exists("docs"):
        os.makedirs("docs")
        print("Created docs directory. Please add documents and run again.")
        exit(0)
    
    # Load all markdown files from docs folder
    for file_path in glob.glob("docs/*.md"):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            documents.append(
                Document(
                    page_content=content,
                    metadata={"source": os.path.basename(file_path)}
                )
            )
    
    if not documents:
        print("No documents found in docs folder.")
        exit(0)
    
    print(f"Loaded {len(documents)} documents")
    return documents

def split_documents(documents: List[Document]) -> List[Document]:
    """Split documents into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks

def create_vector_store(chunks: List[Document]):
    """Create FAISS vector store from document chunks"""
    # Initialize Google's embedding model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Create and save the FAISS index
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    # Create directory if it doesn't exist
    if not os.path.exists("faiss_index"):
        os.makedirs("faiss_index")
    
    # Save the index to disk
    vector_store.save_local("faiss_index")
    print(f"Vector store saved to faiss_index/")

def main():
    print("Starting document ingestion process...")
    documents = load_documents()
    chunks = split_documents(documents)
    create_vector_store(chunks)
    print("Document ingestion complete!")

if __name__ == "__main__":
    main()