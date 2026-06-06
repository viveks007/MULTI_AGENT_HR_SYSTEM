# Running the Multi-Agent HR System Chat Interface

## Prerequisites

1. Ensure all dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

2. Prepare your HR documents in the `data/` folder:
   - Place PDF files in `data/pdfs/`
   - Place FAQ text/markdown files in `data/faq/`

## Starting the Server

### Method 1: Run with Python directly

```bash
python app.py
```

The server will start on `http://127.0.0.1:8000`

### Method 2: Run with Uvicorn

```bash
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

## Using the Chat Interface

1. Open your browser to `http://127.0.0.1:8000/`
2. Type your HR question in the input field
3. Press Enter or click "Send"
4. The system will:
   - Classify your intent
   - Retrieve relevant documents
   - Generate a response
   - Score the quality

## API Endpoints

### POST /query
Process a user query through the RAG pipeline.

**Request:**
```json
{
  "query": "What is the vacation policy?"
}
```

**Response:**
```json
{
  "query": "What is the vacation policy?",
  "intent": "policy_question",
  "answer": "Based on the company handbook...",
  "source_documents": [...],
  "score": 0.85
}
```

### GET /health
Health check endpoint. Returns `{"status": "ok"}`

## First Time Setup

On first run, if no vector store exists, the system will automatically:
1. Load documents from `data/pdfs/` and `data/faq/`
2. Chunk the documents
3. Generate embeddings
4. Build and save the vector store to `data/vector_store.pkl`

This process may take a few moments depending on document size.

## Troubleshooting

- **Vector store not found:** Place HR documents in `data/pdfs/` or `data/faq/` and restart
- **Module import errors:** Ensure all packages in `requirements.txt` are installed
- **Port already in use:** Change the port in app.py or use `--port 8001`

## Architecture

The chat interface orchestrates 5 agents:
1. **Intent Agent** - Classifies user query intent
2. **Retrieval Agent** - Finds relevant documents from the vector store
3. **Policy Agent** - Decides next action based on intent and context
4. **Response Agent** - Generates human-readable answers
5. **Critic Agent** - Scores response quality (0-1)

All components work together to provide accurate, contextual HR support.
