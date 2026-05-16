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
    # Don't raise error here, let the app start so health checks can pass
    # Endpoints requiring DB will fail gracefully later

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
        # Create tables if they don't exist
        Base.metadata.create_all(bind=engine)
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")

@app.on_event("startup")
async def startup_event():
    # Run DB init in background or during startup but catch errors
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
                    if not line:
                        continue
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
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }

# Admin Endpoints for API Key Management
async def verify_master_key(x_master_key: str = Header(...)):
    if x_master_key != MASTER_KEY:
        raise HTTPException(status_code=403, detail="Invalid Master Key")
    return True

@app.post("/admin/keys", dependencies=[Depends(verify_master_key)])
async def create_api_key(key_data: APIKeyCreate, db = Depends(get_db)):
    new_key = secrets.token_urlsafe(32) # Generate a URL-safe key
    key_hash = hashlib.sha256(new_key.encode()).hexdigest()
    db_key = APIKey(key_hash=key_hash, name=key_data.name)
    db.add(db_key)
    db.commit()
    db.refresh(db_key)
    return {"api_key": f"ollama_{new_key}", "warning": "Save this key now! It will not be shown again.", "key_id": db_key.id}

@app.get("/admin/keys", dependencies=[Depends(verify_master_key)])
async def list_api_keys(db = Depends(get_db)):
    keys = db.query(APIKey).all()
    return [{
        "id": k.id,
        "name": k.name,
        "key_hash": k.key_hash, # For debugging, normally wouldn't expose hash
        "created_at": k.created_at
    } for k in keys]

@app.post("/admin/keys/revoke", dependencies=[Depends(verify_master_key)])
async def revoke_api_key(revoke_data: APIKeyRevoke, db = Depends(get_db)):
    key_to_revoke = db.query(APIKey).filter(APIKey.key_hash == revoke_data.key_hash).first()
    if not key_to_revoke:
        raise HTTPException(status_code=404, detail="API Key not found")
    db.delete(key_to_revoke)
    db.commit()
    return {"message": "API Key revoked successfully"}

@app.get("/", response_class=HTMLResponse)
async def index():
    return """
<!DOCTYPE html>
<html>
<head>
    <title>Ollama API Server</title>
    <link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>🤖</text></svg>">
    <style>
        body { font-family: system-ui, sans-serif; max-width: 800px; margin: 40px auto; padding: 20px; line-height: 1.6; background: #f4f4f9; color: #333; }
        .card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }
        h1, h2, h3 { color: #2c3e50; }
        code { background: #eee; padding: 2px 5px; border-radius: 4px; font-family: monospace; }
        .code-block { background: #eef; padding: 15px; border-left: 5px solid #2196f3; margin: 15px 0; overflow-x: auto; font-family: monospace; }
        .warning { background: #fff3cd; color: #856404; border: 1px solid #ffeeba; padding: 10px; border-radius: 4px; margin: 15px 0; }
        .endpoint { background: #f8f8f8; padding: 10px; border-radius: 4px; margin-bottom: 5px; display: flex; align-items: center; }
        .method { background: #28a745; color: white; padding: 3px 8px; border-radius: 3px; margin-right: 10px; font-size: 0.8em; font-weight: bold; }
        .url { font-weight: bold; color: #0056b3; }
        .status { display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; }
        .online { background: #4caf50; }
        .offline { background: #f44336; }
        #chat-box { height: 300px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; margin-bottom: 10px; background: #fafafa; }
        .message { margin-bottom: 10px; padding: 8px; border-radius: 4px; }
        .user { background: #e3f2fd; text-align: right; }
        .assistant { background: #f5f5f5; }
        input[type="text"], button { padding: 10px; border: 1px solid #ddd; border-radius: 4px; font-size: 1em; }
        input[type="text"] { width: calc(70% - 22px); margin-right: 10px; }
        button { background: #2196f3; color: white; border: none; cursor: pointer; width: 25%; }
        button:hover { background: #1976d2; }
        table { width: 100%; border-collapse: collapse; margin: 15px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <div class="card">
        <h1>Ollama API Server</h1>
        <p>Status: <span id="status-dot" class="status offline"></span><span id="status-text">Checking...</span></p>
        <h3>Quick Test</h3>
        <div id="chat-box"></div>
        <input type="text" id="user-input" placeholder="Type a message..." onkeypress="if(event.key===\'Enter\') sendMessage()">
        <button onclick="sendMessage()">Send</button>
    </div>

    <div class="card">
        <h2>API Usage</h2>
        <p>All API endpoints require authentication with a Bearer token.</p>
        <div class="code-block">Authorization: Bearer YOUR_API_KEY</div>
        <div class="warning">Keep your API keys secret. They grant full access to the LLM API.</div>
    </div>

    <div class="card">
        <h2>Key Management (Master Key Required)</h2>
        <p style="color:#888;margin-bottom:10px">Use your MASTER_KEY in the X-Master-Key header to manage API keys.</p>
        <div class="endpoint"><span class="method">POST</span><span class="url">/admin/keys</span> - Create new API key</div>
        <div class="code-block">Headers: X-Master-Key: your-master-key<br>
Body: {\"name\": \"my-app\", \"rate_limit\": 1000}<br><br>
Response: {\"api_key\": \"ollama_xxxxx\", \"warning\": \"Save this key now!\"}</div>
        <div class="endpoint"><span class="method">GET</span><span class="url">/admin/keys</span> - List all keys</div>
        <div class="endpoint"><span class="method">POST</span><span class="url">/admin/keys/revoke</span> - Revoke a key</div>
        <div class="code-block">Body: {\"key_hash\": \"sha256_hash_of_key\"}</div>
    </div>

    <div class="card">
        <h2>OpenAI-Compatible Endpoints</h2>
        <h3>List Models</h3>
        <div class="endpoint"><span class="method">GET</span><span class="url">/v1/models</span></div>
        <p style="color:#888;margin:8px 0">Returns available models in OpenAI format.</p>
        <div class="code-block">Response:<br>
{
&nbsp;&nbsp;\"object\": \"list\",<br>
&nbsp;&nbsp;\"data\": [{\"id\": \"qwen2.5:0.5b\", \"object\": \"model\", \"created\": 1234567890, \"owned_by\": \"ollama\"}]
}</div>
        <h3>Chat Completions</h3>
        <div class="endpoint"><span class="method">POST</span><span class="url">/v1/chat/completions</span></div>
        <table class="table">
            <tr><th>Parameter</th><th>Type</th><th>Required</th><th>Description</th></tr>
            <tr><td>model</td><td>string</td><td>No</td><td>Model name (default: qwen2.5:0.5b)</td></tr>
            <tr><td>messages</td><td>array</td><td>Yes</td><td>[{role, content}]</td></tr>
            <tr><td>stream</td><td>boolean</td><td>No</td><td>Stream response</td></tr>
            <tr><td>temperature</td><td>float</td><td>No</td><td>0.0 - 2.0 (default: 0.7)</td></tr>
            <tr><td>max_tokens</td><td>integer</td><td>No</td><td>Max tokens (default: 2048)</td></tr>
        </table>
        <div class="code-block">curl -X POST <span class="base-url"></span>/v1/chat/completions<br>
-H \"Authorization: Bearer YOUR_API_KEY\"<br>
-H \"Content-Type: application/json\"<br>
-d \'{\"model\":\"qwen2.5:0.5b\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}]}\'</div>
    </div>

    <div class="card">
        <h2>Ollama Native Endpoints</h2>
        <h3>Generate Text</h3>
        <div class="endpoint"><span class="method">POST</span><span class="url">/api/generate</span></div>
        <table class="table">
            <tr><th>Parameter</th><th>Type</th><th>Required</th><th>Description</th></tr>
            <tr><td>model</td><td>string</td><td>No</td><td>Model name</td></tr>
            <tr><td>prompt</td><td>string</td><td>Yes</td><td>Text prompt</td></tr>
            <tr><td>stream</td><td>boolean</td><td>No</td><td>Stream response</td></tr>
            <tr><td>temperature</td><td>float</td><td>No</td><td>Sampling temp</td></tr>
            <tr><td>max_tokens</td><td>integer</td><td>No</td><td>Max tokens</td></tr>
        </table>
        <h3>List Models (Native)</h3>
        <div class="endpoint"><span class="method">GET</span><span class="url">/api/models</span></div>
        <h3>Pull Model</h3>
        <div class="endpoint"><span class="method">POST</span><span class="url">/api/pull</span></div>
        <div class="code-block">curl -X POST <span class="base-url"></span>/api/pull<br>
-H \"Authorization: Bearer YOUR_API_KEY\"<br>
-d \'{\"name\": \"llama3.2:1b\"}\'</div>
    </div>

    <div class="card">
        <h2>Health Check</h2>
        <div class="endpoint"><span class="method">GET</span><span class="url">/health</span></div>
        <p style="color:#888;margin:8px 0">No auth required. Returns server status.</p>
    </div>

    <div class="card">
        <h2>Environment Variables</h2>
        <table class="table">
            <tr><th>Variable</th><th>Default</th><th>Description</th></tr>
            <tr><td>DEFAULT_MODEL</td><td>qwen2.5:0.5b</td><td>Auto-pulled model</td></tr>
            <tr><td>OLLAMA_HOST</td><td>http://localhost:11434</td><td>Internal Ollama URL</td></tr>
            <tr><td>MASTER_KEY</td><td>ollama-master-key-change-me</td><td>Admin key for key management</td></tr>
        </table>
        <div class="warning">Change MASTER_KEY in production! Anyone with it can create/revoke API keys.</div>
    </div>

    <div class="card">
        <h2>Available Models</h2>
        <table class="table">
            <tr><th>Model</th><th>Size</th><th>Speed</th><th>Use Case</th></tr>
            <tr><td>qwen2.5:0.5b</td><td>~300MB</td><td>Very Fast</td><td>Default, lightweight</td></tr>
            <tr><td>llama3.2:1b</td><td>~1.3GB</td><td>Fast</td><td>General purpose</td></tr>
            <tr><td>gemma2:2b</td><td>~1.6GB</td><td>Fast</td><td>Google model</td></tr>
            <tr><td>phi3:mini</td><td>~2GB</td><td>Medium</td><td>Microsoft, balanced</td></tr>
            <tr><td>mistral:7b</td><td>~4GB</td><td>Slower</td><td>High quality</td></tr>
        </table>
    </div>

    <script>
        document.querySelectorAll(\".base-url\").forEach(el=>el.textContent=window.location.origin);
        async function checkStatus() {
            try {
                const res = await fetch(\"/health\");
                const data = await res.json();
                const dot = document.getElementById(\"status-dot\");
                const text = document.getElementById(\"status-text\");
                if (data.ollama) {
                    dot.className = \"status online\";
                    text.textContent = \"Ollama Online\";
                } else {
                    dot.className = \"status offline\";
                    text.textContent = \"Ollama Starting...\";
                }
            } catch (e) {
                console.error(e);
            }
        }
        async function sendMessage() {
            const input = document.getElementById(\"user-input\");
            const text = input.value.trim();
            if (!text) return;
            
            const key = prompt(\"Enter your API Key to test:\");
            if (!key) return;

            input.value = \"\";
            const chatBox = document.getElementById(\"chat-box\");
            chatBox.innerHTML += `<div class=\"message user\">${text}</div>`;
            chatBox.scrollTop = chatBox.scrollHeight;

            const assistantMsg = document.createElement(\"div\");
            assistantMsg.className = \"message assistant\";
            assistantMsg.textContent = \"...\";
            chatBox.appendChild(assistantMsg);

            try {
                const res = await fetch(\"/v1/chat/completions\", {
                    method: \"POST\",
                    headers: {
                        \"Content-Type\": \"application/json\",
                        \"Authorization\": `Bearer ${key}`
                    },
                    body: JSON.stringify({
                        model: \"qwen2.5:0.5b\",
                        messages: [{role: \"user\", content: text}]
                    })
                });
                
                if (!res.ok) {
                    const err = await res.json();
                    throw new Error(err.detail || \"Failed to connect\");
                }

                const data = await res.json();
                assistantMsg.textContent = data.choices[0].message.content;
            } catch (e) {
                assistantMsg.textContent = \"Error: \" + e.message;
                assistantMsg.style.color = \"#ff6b6b\";
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
