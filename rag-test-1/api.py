import os
from typing import List, Dict, Any
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores.faiss import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
load_dotenv()

# Check for API key
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("Please set GOOGLE_API_KEY environment variable in .env file")

# Check if FAISS index exists
if not os.path.exists("faiss_index"):
    print("FAISS index not found. Please run ingest.py first.")

# Initialize FastAPI
app = FastAPI(title="Gemini RAG API")

# Define request model
class QuestionRequest(BaseModel):
    question: str

# Define response model
class AnswerResponse(BaseModel):
    answer: str
    sources: List[str]

# Initialize embeddings and vector store
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# - vector_store = FAISS.load_local("faiss_index", embeddings)
vector_store = FAISS.load_local(
     "faiss_index",
     embeddings,
     allow_dangerous_deserialization=True   # tell LangChain you trust this file
 )
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# Initialize the Gemini model
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2)

# Define prompt template
template = """Answer the question based on the following context. 
If you don't know the answer or cannot find it in the context, just say so.
Don't make up information that's not in the context.

Context: {context}

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)

# Create the RAG chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def extract_sources(docs):
    sources = list(set(doc.metadata["source"] for doc in docs))
    return sources

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest) -> AnswerResponse:
    """Ask a question to the RAG system"""
    if not request.question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        # Get the documents for sources
        docs = retriever.invoke(request.question)
        sources = extract_sources(docs)
        
        # Get the answer
        answer = rag_chain.invoke(request.question)
        
        return AnswerResponse(answer=answer, sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Welcome to the Gemini RAG API. Use /ask endpoint to ask questions."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)