import os
import time
import secrets
import hashlib
import httpx
import asyncio
import json
import logging
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from sqlalchemy import create_engine, Column, String, Integer, BigInteger, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import OperationalError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Ollama API Server",
    description="Self-hosted LLM API with Ollama + FastAPI. API key protected.",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database Setup
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    logger.error("DATABASE_URL environment variable is not set")

engine = None
SessionLocal = None
Base = declarative_base()

class APIKey(Base):
    __tablename__ = "api_keys"
    id = Column(Integer, primary_key=True, index=True)
    key_hash = Column(String(64), unique=True, index=True)
    name = Column(String(100))
    created_at = Column(BigInteger, default=lambda: int(time.time()))

def init_db():
    global engine, SessionLocal
    if not DATABASE_URL:
        return
    try:
        engine = create_engine(DATABASE_URL, pool_pre_ping=True)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        Base.metadata.create_all(bind=engine)
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")

@app.on_event("startup")
async def startup_event():
    init_db()

# Security
security = HTTPBearer()

def get_db():
    if SessionLocal is None:
        raise HTTPException(status_code=503, detail="Database not initialized")
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security), db = Depends(get_db)):
    key = credentials.credentials
    key_hash = hashlib.sha256(key.encode()).hexdigest()
    try:
        db_key = db.query(APIKey).filter(APIKey.key_hash == key_hash).first()
        if not db_key:
            raise HTTPException(status_code=401, detail="Invalid API Key")
        return db_key
    except OperationalError:
        raise HTTPException(status_code=503, detail="Database connection error")

# Pydantic models for API
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: Optional[bool] = False

class APIKeyCreate(BaseModel):
    name: str
    rate_limit: Optional[int] = None

class APIKeyRevoke(BaseModel):
    key_hash: str

# Ollama Proxy
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MASTER_KEY = os.getenv("MASTER_KEY", "ollama-master-key-change-me")

@app.get("/health")
async def health_check():
    ollama_ready = False
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{OLLAMA_HOST}/api/tags", timeout=1.0)
            ollama_ready = (resp.status_code == 200)
    except Exception:
        pass
    return {
        "status": "healthy",
        "ollama": ollama_ready,
        "database": SessionLocal is not None
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, api_key = Depends(verify_api_key)):
    async def generate():
        async with httpx.AsyncClient(timeout=None) as client:
            ollama_req = {
                "model": request.model,
                "messages": [m.dict() for m in request.messages],
                "stream": request.stream
            }
            async with client.stream("POST", f"{OLLAMA_HOST}/api/chat", json=ollama_req) as response:
                if response.status_code != 200:
                    error_detail = await response.aread()
                    raise HTTPException(status_code=response.status_code, detail=error_detail.decode())
                async for line in response.aiter_lines():
                    if not line: continue
                    data = json.loads(line)
                    chunk = {
                        "id": f"chatcmpl-{secrets.token_hex(12)}",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": data.get("message", {}).get("content", "")},
                            "finish_reason": "stop" if data.get("done") else None
                        }]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"

    if request.stream:
        return StreamingResponse(generate(), media_type="text/event-stream")
    
    async with httpx.AsyncClient(timeout=None) as client:
        ollama_req = {
            "model": request.model,
            "messages": [m.dict() for m in request.messages],
            "stream": False
        }
        resp = await client.post(f"{OLLAMA_HOST}/api/chat", json=ollama_req)
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        data = resp.json()
        return {
            "id": f"chatcmpl-{secrets.token_hex(12)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": data.get("message"),
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        }

# Admin Endpoints
async def verify_master_key(x_master_key: str = Header(...)):
    if x_master_key != MASTER_KEY:
        raise HTTPException(status_code=403, detail="Invalid Master Key")
    return True

@app.post("/admin/keys", dependencies=[Depends(verify_master_key)])
async def create_api_key(key_data: APIKeyCreate, db = Depends(get_db)):
    new_key = secrets.token_urlsafe(32)
    key_hash = hashlib.sha256(new_key.encode()).hexdigest()
    db_key = APIKey(key_hash=key_hash, name=key_data.name)
    db.add(db_key)
    db.commit()
    db.refresh(db_key)
    return {"api_key": f"ollama_{new_key}", "warning": "Save this key now!", "key_id": db_key.id}

@app.get("/admin/keys", dependencies=[Depends(verify_master_key)])
async def list_api_keys(db = Depends(get_db)):
    keys = db.query(APIKey).all()
    return [{"id": k.id, "name": k.name, "key_hash": k.key_hash, "created_at": k.created_at} for k in keys]

@app.post("/admin/keys/revoke", dependencies=[Depends(verify_master_key)])
async def revoke_api_key(revoke_data: APIKeyRevoke, db = Depends(get_db)):
    key_to_revoke = db.query(APIKey).filter(APIKey.key_hash == revoke_data.key_hash).first()
    if not key_to_revoke:
        raise HTTPException(status_code=404, detail="API Key not found")
    db.delete(key_to_revoke)
    db.commit()
    return {"message": "API Key revoked successfully"}

@app.get("/v1/models")
async def list_models(api_key = Depends(verify_api_key)):
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{OLLAMA_HOST}/api/tags")
        if resp.status_code != 200:
            return {"object": "list", "data": []}
        ollama_models = resp.json().get("models", [])
        data = []
        for m in ollama_models:
            data.append({
                "id": m.get("name"),
                "object": "model",
                "created": int(time.time()),
                "owned_by": "ollama"
            })
        return {"object": "list", "data": data}

@app.post("/api/generate")
async def ollama_generate(request: Dict[str, Any], api_key = Depends(verify_api_key)):
    async with httpx.AsyncClient(timeout=None) as client:
        resp = await client.post(f"{OLLAMA_HOST}/api/generate", json=request)
        return resp.json()

@app.get("/api/models")
async def ollama_models(api_key = Depends(verify_api_key)):
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{OLLAMA_HOST}/api/tags")
        return resp.json()

@app.post("/api/pull")
async def ollama_pull(request: Dict[str, Any], api_key = Depends(verify_api_key)):
    async with httpx.AsyncClient(timeout=None) as client:
        resp = await client.post(f"{OLLAMA_HOST}/api/pull", json=request)
        return resp.json()

@app.get("/UI", response_class=HTMLResponse)
@app.get("/", response_class=HTMLResponse)
async def index():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>Ollama API Server</title>
    <link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>🤖</text></svg>">
    <style>
        body { font-family: system-ui, sans-serif; max-width: 900px; margin: 40px auto; padding: 20px; line-height: 1.6; background: #f4f4f9; color: #333; }
        .card { background: white; padding: 25px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 25px; border: 1px solid #eef; }
        h1, h2, h3 { color: #2c3e50; margin-top: 0; }
        code { background: #f8f9fa; padding: 3px 6px; border-radius: 4px; font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace; font-size: 0.9em; color: #e83e8c; }
        .code-block { background: #1e1e2e; color: #cdd6f4; padding: 20px; border-radius: 8px; margin: 15px 0; overflow-x: auto; font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace; font-size: 0.85em; position: relative; }
        .warning { background: #fff3cd; color: #856404; border: 1px solid #ffeeba; padding: 15px; border-radius: 8px; margin: 20px 0; display: flex; align-items: center; gap: 10px; }
        .endpoint { background: #f8f9fa; padding: 12px 15px; border-radius: 6px; margin-bottom: 8px; display: flex; align-items: center; border-left: 4px solid #2196f3; }
        .method { background: #28a745; color: white; padding: 4px 10px; border-radius: 4px; margin-right: 15px; font-size: 0.75em; font-weight: 800; text-transform: uppercase; letter-spacing: 0.5px; }
        .url { font-weight: 600; color: #0056b3; font-family: monospace; }
        .status { display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 10px; }
        .online { background: #4caf50; box-shadow: 0 0 8px rgba(76, 175, 80, 0.5); }
        .offline { background: #f44336; box-shadow: 0 0 8px rgba(244, 67, 54, 0.5); }
        #chat-box { height: 400px; overflow-y: auto; border: 1px solid #e1e4e8; padding: 20px; margin-bottom: 15px; background: #ffffff; border-radius: 8px; display: flex; flex-direction: column; gap: 12px; }
        .message { max-width: 85%; padding: 12px 16px; border-radius: 12px; font-size: 0.95em; position: relative; word-wrap: break-word; }
        .user { background: #007bff; color: white; align-self: flex-end; border-bottom-right-radius: 2px; }
        .assistant { background: #f1f3f5; color: #212529; align-self: flex-start; border-bottom-left-radius: 2px; border: 1px solid #dee2e6; }
        .input-group { display: flex; gap: 10px; margin-top: 10px; }
        input[type="text"] { flex: 1; padding: 12px 15px; border: 2px solid #e1e4e8; border-radius: 8px; font-size: 1em; outline: none; transition: border-color 0.2s; }
        input[type="text"]:focus { border-color: #2196f3; }
        button { background: #2196f3; color: white; border: none; padding: 12px 25px; border-radius: 8px; font-weight: 600; cursor: pointer; transition: background 0.2s; }
        button:hover { background: #1976d2; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; border-radius: 8px; overflow: hidden; }
        th, td { border: 1px solid #e1e4e8; padding: 12px 15px; text-align: left; }
        th { background-color: #f8f9fa; font-weight: 600; color: #495057; }
        .container { display: flex; flex-direction: column; gap: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="card">
            <h1>Ollama API Server</h1>
            <p>Status: <span id="status-dot" class="status offline"></span><span id="status-text">Checking system status...</span></p>
            <h3>Interactive Playground</h3>
            <div id="chat-box"></div>
            <div class="input-group">
                <input type="text" id="user-input" placeholder="Ask anything to the LLM..." onkeypress="if(event.key==='Enter') sendMessage()">
                <button onclick="sendMessage()">Send Message</button>
            </div>
        </div>

        <div class="card">
            <h2>Authentication</h2>
            <p>All API requests must include your API key in the <code>Authorization</code> header.</p>
            <div class="code-block">Authorization: Bearer YOUR_API_KEY</div>
            <div class="warning">
                <span>⚠️</span>
                <span>Keep your API keys secure. They grant direct access to your LLM resources.</span>
            </div>
        </div>

        <div class="card">
            <h2>Key Management (Admin)</h2>
            <p style="color:#6c757d;margin-bottom:15px">Manage your access keys using the Master Key in the <code>X-Master-Key</code> header.</p>
            <div class="endpoint"><span class="method" style="background:#6f42c1">POST</span><span class="url">/admin/keys</span> - Create new API key</div>
            <div class="code-block">Headers: X-Master-Key: your-master-key<br>Body: {"name": "production-app", "rate_limit": 5000}<br><br>Response: {"api_key": "ollama_xxxxx", "warning": "Save this key now!"}</div>
            <div class="endpoint"><span class="method" style="background:#17a2b8">GET</span><span class="url">/admin/keys</span> - List active keys</div>
            <div class="endpoint"><span class="method" style="background:#dc3545">POST</span><span class="url">/admin/keys/revoke</span> - Revoke access</div>
        </div>

        <div class="card">
            <h2>OpenAI-Compatible API</h2>
            <p>Use your favorite OpenAI libraries by pointing the base URL to this server.</p>
            <h3>Chat Completions</h3>
            <div class="endpoint"><span class="method">POST</span><span class="url">/v1/chat/completions</span></div>
            <div class="code-block">curl -X POST <span class="base-url"></span>/v1/chat/completions \<br>  -H "Authorization: Bearer YOUR_API_KEY" \<br>  -H "Content-Type: application/json" \<br>  -d '{<br>    "model": "qwen2.5:0.5b",<br>    "messages": [{"role": "user", "content": "Hello!"}],<br>    "stream": true<br>  }'</div>
            <h3>List Models</h3>
            <div class="endpoint"><span class="method" style="background:#17a2b8">GET</span><span class="url">/v1/models</span></div>
        </div>

        <div class="card">
            <h2>Ollama Native API</h2>
            <div class="endpoint"><span class="method">POST</span><span class="url">/api/generate</span></div>
            <div class="endpoint"><span class="method" style="background:#17a2b8">GET</span><span class="url">/api/models</span></div>
            <div class="endpoint"><span class="method" style="background:#6f42c1">POST</span><span class="url">/api/pull</span></div>
        </div>

        <div class="card">
            <h2>Available Models</h2>
            <table>
                <thead>
                    <tr><th>Model Name</th><th>Size</th><th>Performance</th><th>Best For</th></tr>
                </thead>
                <tbody>
                    <tr><td><code>qwen2.5:0.5b</code></td><td>~300MB</td><td>Ultra Fast</td><td>Lightweight tasks</td></tr>
                    <tr><td><code>tinyllama:latest</code></td><td>~600MB</td><td>Very Fast</td><td>Quick inference</td></tr>
                    <tr><td><code>llama3.2:1b</code></td><td>~1.3GB</td><td>Fast</td><td>General reasoning</td></tr>
                    <tr><td><code>mistral:7b</code></td><td>~4.1GB</td><td>Standard</td><td>High-quality output</td></tr>
                </tbody>
            </table>
        </div>
    </div>

    <script>
        document.querySelectorAll(".base-url").forEach(el=>el.textContent=window.location.origin);
        async function checkStatus() {
            try {
                const res = await fetch("/health");
                const data = await res.json();
                const dot = document.getElementById("status-dot");
                const text = document.getElementById("status-text");
                if (data.ollama) {
                    dot.className = "status online";
                    text.textContent = "Ollama Online & Ready";
                } else {
                    dot.className = "status offline";
                    text.textContent = "Ollama Initializing...";
                }
            } catch (e) {
                console.error(e);
            }
        }
        async function sendMessage() {
            const input = document.getElementById("user-input");
            const text = input.value.trim();
            if (!text) return;
            
            const key = prompt("Please enter your API Key to authenticate:");
            if (!key) return;

            input.value = "";
            const chatBox = document.getElementById("chat-box");
            
            const userDiv = document.createElement("div");
            userDiv.className = "message user";
            userDiv.textContent = text;
            chatBox.appendChild(userDiv);
            chatBox.scrollTop = chatBox.scrollHeight;

            const assistantMsg = document.createElement("div");
            assistantMsg.className = "message assistant";
            assistantMsg.textContent = "Thinking...";
            chatBox.appendChild(assistantMsg);

            try {
                const res = await fetch("/v1/chat/completions", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                        "Authorization": `Bearer ${key}`
                    },
                    body: JSON.stringify({
                        model: "qwen2.5:0.5b",
                        messages: [{role: "user", content: text}]
                    })
                });
                
                if (!res.ok) {
                    const err = await res.json();
                    throw new Error(err.detail || "Authentication failed or server error");
                }

                const data = await res.json();
                assistantMsg.textContent = data.choices[0].message.content;
            } catch (e) {
                assistantMsg.textContent = "Error: " + e.message;
                assistantMsg.style.color = "#dc3545";
            }
            chatBox.scrollTop = chatBox.scrollHeight;
        }
        
        checkStatus();
        setInterval(checkStatus, 10000);
    </script>
</body>
</html>"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
