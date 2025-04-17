# Gemini-powered RAG Demo

This project demonstrates a Retrieval-Augmented Generation (RAG) system powered by Google's Gemini AI. It uses a local FAISS vector database to store document embeddings and the Gemini API to generate responses based on retrieved context.

## Setup Instructions

### 1. Clone the repository
```bash
git clone <repository-url>
cd rag-demo
```

### 2. Create a virtual environment
```bash
python -m venv .venv
```

Activate the virtual environment:
- On Windows:
  ```
  .venv\Scripts\activate
  ```
- On macOS/Linux:
  ```
  source .venv/bin/activate
  ```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up your environment variables
Create a `.env` file in the root directory by copying the example:
```bash
cp .env.example .env
```

Edit the `.env` file and add your Google API key:
```
GOOGLE_API_KEY=your_google_api_key_here
```

### 5. Ingest documents
Run the ingestion script to process documents and create the FAISS index:
```bash
python ingest.py
```

### 6. Start the API server
```bash
uvicorn api:app --reload --port 8000
```

The API will be available at `http://localhost:8000`.

## Usage

To ask a question using the API, you can use curl:

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the key features of Python?"}'
```

The response will contain the answer and the source documents used:

```json
{
  "answer": "The key features of Python include...",
  "sources": ["python.md"]
}
```

## Project Structure

- `docs/`: Contains the markdown documents used for knowledge base
- `faiss_index/`: Generated directory containing the FAISS vector index
- `ingest.py`: Script to process documents and build the vector database
- `api.py`: FastAPI server with the RAG implementation
- `requirements.txt`: Project dependencies
- `.env.example`: Example environment variables file

## Customization

To use a different Gemini model, edit the model name in `api.py`:
```python
model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.2)
```

To add more documents, place markdown files in the `docs/` directory and run `python ingest.py` again.