"""FastAPI application for the Multi-Agent HR System."""

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from agents.critic_agent import CriticAgent
from agents.intent_agent import IntentAgent
from agents.policy_agent import PolicyAgent
from agents.response_agent import ResponseAgent
from agents.retrieval_agent import RetrievalAgent
from rag.ingestion import ingest

app = FastAPI(title="Multi-Agent HR System", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

VECTOR_STORE_PATH = "data/vector_store.pkl"


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    query: str
    intent: str
    answer: str
    source_documents: Optional[list] = None
    score: float = 0.0


def ensure_vector_store():
    """Ensure the vector store exists; ingest if missing."""
    if not Path(VECTOR_STORE_PATH).exists():
        print("Vector store not found. Running ingestion...")
        try:
            ingest(data_dir="data", persist_path=VECTOR_STORE_PATH)
            print(f"Vector store created at {VECTOR_STORE_PATH}")
        except Exception as e:
            print(f"Ingestion failed: {e}")


@app.on_event("startup")
async def startup():
    """Initialize agents and vector store on startup."""
    global intent_agent, retrieval_agent, policy_agent, response_agent, critic_agent
    
    ensure_vector_store()
    
    intent_agent = IntentAgent()
    retrieval_agent = RetrievalAgent(VECTOR_STORE_PATH)
    policy_agent = PolicyAgent()
    response_agent = ResponseAgent()
    critic_agent = CriticAgent()
    
    print("Agents initialized successfully.")


@app.get("/", response_class=HTMLResponse)
async def get_chat_ui():
    """Serve a simple chat UI."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Multi-Agent HR Chat</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; }
            .chat-container { border: 1px solid #ccc; padding: 20px; border-radius: 5px; }
            .query-input { width: 100%; padding: 10px; font-size: 16px; }
            .submit-btn { padding: 10px 20px; margin-top: 10px; background: #007bff; color: white; border: none; cursor: pointer; border-radius: 3px; }
            .response { margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 5px; }
            .intent { color: #666; font-size: 12px; margin-top: 10px; }
        </style>
    </head>
    <body>
        <h1>HR Assistant Chat</h1>
        <div class="chat-container">
            <input type="text" id="queryInput" class="query-input" placeholder="Ask your HR question...">
            <button class="submit-btn" onclick="askQuestion()">Send</button>
            <div id="response"></div>
        </div>
        <script>
            async function askQuestion() {
                const q = document.getElementById('queryInput').value.trim();
                if (!q) return;
                document.getElementById('response').innerHTML = '<p>Loading...</p>';
                try {
                    const r = await fetch('/query', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({query: q})
                    });
                    if (!r.ok) throw new Error('Request failed');
                    const d = await r.json();
                    document.getElementById('response').innerHTML = '<div class="response"><strong>Answer:</strong><p>' + d.answer + '</p><div class="intent">Intent: ' + d.intent + '</div></div>';
                } catch (e) {
                    document.getElementById('response').innerHTML = '<p style="color: red;">Error: ' + e.message + '</p>';
                }
            }
            document.getElementById('queryInput').addEventListener('keypress', (e) => {
                if (e.key === 'Enter') askQuestion();
            });
        </script>
    </body>
    </html>
    """


@app.post("/query", response_model=QueryResponse)
async def query_hr_system(request: QueryRequest) -> QueryResponse:
    """Process a user query through the multi-agent RAG pipeline."""
    query = request.query.strip()
    
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    try:
        intent = intent_agent.predict_intent(query)
        retrieved_docs = retrieval_agent.retrieve(query, top_k=3)
        action = policy_agent.decide(intent, retrieved_docs)
        answer = response_agent.generate(query, retrieved_docs)
        score = critic_agent.score(answer, query)
        
        return QueryResponse(
            query=query,
            intent=intent,
            answer=answer,
            source_documents=retrieved_docs,
            score=score,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
