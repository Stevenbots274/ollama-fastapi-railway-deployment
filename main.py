import os
import time
import secrets
import hashlib
import httpx
import asyncio
import json
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from sqlalchemy import create_all, create_engine, Column, String, Integer, BigInteger, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

app = FastAPI(
    title="Ollama API Server",
    description="Self-hosted LLM API with Ollama + FastAPI. API key protected.",
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "qwen2.5:0.5b")
MASTER_KEY = os.getenv("MASTER_KEY", "ollama-master-key-change-me")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://neondb_owner:npg_1imPJgOw5qBc@ep-divine-fog-ajro56fz-pooler.c-3.us-east-2.aws.neon.tech/Dailymotion%20?sslmode=require")

# ============ DATABASE SETUP ============

Base = declarative_base()

class APIKey(Base):
    __tablename__ = "api_keys"
    key_hash = Column(String, primary_key=True)
    name = Column(String)
    created = Column(BigInteger)
    rate_limit = Column(Integer)
    usage_count = Column(Integer, default=0)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create tables on startup
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ============ AUTHENTICATION ============

security = HTTPBearer(auto_error=False)

def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security), db = Depends(get_db)):
    if not credentials:
        raise HTTPException(status_code=401, detail="Missing Authorization header. Use: Bearer YOUR_API_KEY")
    token = credentials.credentials
    key_hash = hash_key(token)
    
    db_key = db.query(APIKey).filter(APIKey.key_hash == key_hash).first()
    if not db_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    # Increment usage count
    db_key.usage_count += 1
    db.commit()
    
    return token

def verify_master_key(x_master_key: str = Header(None)):
    if not x_master_key:
        raise HTTPException(status_code=401, detail="Missing X-Master-Key header")
    if x_master_key != MASTER_KEY:
        raise HTTPException(status_code=403, detail="Invalid master key")
    return x_master_key

# ============ REQUEST MODELS ============

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    stream: bool = False
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2048

class GenerateRequest(BaseModel):
    model: Optional[str] = None
    prompt: str
    stream: bool = False
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2048

class PullModelRequest(BaseModel):
    name: str

class DeleteModelRequest(BaseModel):
    name: str

class CreateKeyRequest(BaseModel):
    name: str
    rate_limit: Optional[int] = 1000  # requests per hour

class RevokeKeyRequest(BaseModel):
    key_hash: str

# ============ PUBLIC ENDPOINTS ============

@app.get("/")
def root():
    return {
        "service": "Ollama API Server",
        "version": "2.1.0",
        "storage": "PostgreSQL (Shared)",
        "auth": "Bearer token required for API access",
        "documentation": "/docs",
        "web_ui": "/ui",
        "health": "/health",
        "endpoints": {
            "chat": "POST /v1/chat/completions",
            "generate": "POST /api/generate",
            "models": "GET /v1/models",
            "create_key": "POST /admin/keys (master key)",
            "list_keys": "GET /admin/keys (master key)"
        }
    }

@app.get("/health")
async def health():
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
            return {"status": "ok", "ollama": "connected", "database": "connected"}
    except:
        return {"status": "degraded", "ollama": "not ready", "database": "connected"}

# ============ KEY MANAGEMENT (MASTER KEY PROTECTED) ============

@app.post("/admin/keys")
def create_key(req: CreateKeyRequest, master: str = Depends(verify_master_key), db = Depends(get_db)):
    raw_key = "ollama_" + secrets.token_urlsafe(32)
    key_hash = hash_key(raw_key)
    
    new_key = APIKey(
        key_hash=key_hash,
        name=req.name,
        created=int(time.time()),
        rate_limit=req.rate_limit,
        usage_count=0
    )
    db.add(new_key)
    db.commit()
    
    return {
        "api_key": raw_key,
        "name": req.name,
        "warning": "Save this key now - it will not be shown again!"
    }

@app.get("/admin/keys")
def list_keys(master: str = Depends(verify_master_key), db = Depends(get_db)):
    keys = db.query(APIKey).all()
    return {"keys": [{"name": k.name, "created": k.created, "rate_limit": k.rate_limit, "usage_count": k.usage_count, "key_hash": k.key_hash} for k in keys]}

@app.post("/admin/keys/revoke")
def revoke_key(req: RevokeKeyRequest, master: str = Depends(verify_master_key), db = Depends(get_db)):
    db_key = db.query(APIKey).filter(APIKey.key_hash == req.key_hash).first()
    if db_key:
        db.delete(db_key)
        db.commit()
        return {"status": "revoked"}
    raise HTTPException(status_code=404, detail="Key not found")

# ============ PROTECTED API ENDPOINTS ============

@app.get("/v1/models")
async def list_models_v1(api_key: str = Depends(verify_api_key)):
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{OLLAMA_HOST}/api/tags", timeout=30)
            data = r.json()
            models = []
            for m in data.get("models", []):
                models.append({
                    "id": m["name"],
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "ollama"
                })
            return {"object": "list", "data": models}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Ollama error: {str(e)}")

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest, api_key: str = Depends(verify_api_key)):
    model = req.model or DEFAULT_MODEL
    try:
        payload = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in req.messages],
            "stream": req.stream,
            "options": {
                "temperature": req.temperature,
                "num_predict": req.max_tokens
            }
        }

        if req.stream:
            async def streamer():
                async with httpx.AsyncClient(timeout=None) as client:
                    async with client.stream("POST", f"{OLLAMA_HOST}/api/chat", json=payload) as r:
                        async for line in r.aiter_lines():
                            if line:
                                yield line + "\n"
            return StreamingResponse(streamer(), media_type="text/event-stream")

        async with httpx.AsyncClient(timeout=300) as client:
            r = await client.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=300)
            data = r.json()
            content = data.get("message", {}).get("content", "")
            if not content and not req.stream:
                content = data.get("response", "")
            
            return {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content
                    },
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Ollama error: {str(e)}")

@app.post("/api/generate")
async def generate(req: GenerateRequest, api_key: str = Depends(verify_api_key)):
    model = req.model or DEFAULT_MODEL
    try:
        payload = {
            "model": model, "prompt": req.prompt, "stream": req.stream,
            "options": {"temperature": req.temperature, "num_predict": req.max_tokens}
        }

        if req.stream:
            async def streamer():
                async with httpx.AsyncClient(timeout=None) as client:
                    async with client.stream("POST", f"{OLLAMA_HOST}/api/generate", json=payload) as r:
                        async for line in r.aiter_lines():
                            if line: yield line + "\n"
            return StreamingResponse(streamer(), media_type="application/x-ndjson")

        async with httpx.AsyncClient(timeout=300) as client:
            r = await client.post(f"{OLLAMA_HOST}/api/generate", json=payload)
            return r.json()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Ollama error: {str(e)}")

@app.get("/api/models")
async def ollama_models(api_key: str = Depends(verify_api_key)):
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{OLLAMA_HOST}/api/tags", timeout=30)
            return r.json()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Ollama error: {str(e)}")

@app.post("/api/pull")
async def pull_model(req: PullModelRequest, api_key: str = Depends(verify_api_key)):
    try:
        payload = {"name": req.name, "stream": False}
        async with httpx.AsyncClient(timeout=600) as client:
            r = await client.post(f"{OLLAMA_HOST}/api/pull", json=payload)
            return r.json()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Ollama error: {str(e)}")

@app.post("/api/delete")
async def delete_model(req: DeleteModelRequest, api_key: str = Depends(verify_api_key)):
    try:
        payload = {"name": req.name}
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.request("DELETE", f"{OLLAMA_HOST}/api/delete", json=payload)
            if r.status_code == 200:
                return {"status": "success", "message": f"Model {req.name} deleted"}
            return r.json()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Ollama error: {str(e)}")

# ============ WEB UI ============

@app.get("/ui", response_class=HTMLResponse)
def web_ui():
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Ollama API Server</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;background:#0f0f23;color:#e0e0e0;min-height:100vh}
.container{max-width:900px;margin:0 auto;padding:40px 20px}
h1{font-size:2.5rem;margin-bottom:10px;background:linear-gradient(90deg,#00d4ff,#7b2cbf);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.subtitle{color:#888;margin-bottom:40px}
.card{background:#1a1a2e;border:1px solid #2a2a4a;border-radius:12px;padding:24px;margin-bottom:20px}
.card h2{color:#00d4ff;margin-bottom:16px;font-size:1.2rem}
.endpoint{background:#0f0f1a;border-left:3px solid #00d4ff;padding:12px 16px;margin:8px 0;border-radius:0 8px 8px 0;font-family:"Courier New",monospace;font-size:.9rem}
.method{color:#7ee787;font-weight:bold;margin-right:8px}
.url{color:#dcdcaa}
.status{display:inline-block;padding:4px 12px;border-radius:20px;font-size:.85rem;font-weight:600}
.status.ok{background:#1a472a;color:#7ee787}
.status.warn{background:#4a3a1a;color:#ffa500}
.chat-box{height:400px;overflow-y:auto;background:#0f0f1a;border-radius:8px;padding:16px;margin-bottom:16px}
.message{margin-bottom:12px;padding:12px;border-radius:8px;max-width:80%}
.message.user{background:#1a3a5c;margin-left:auto}
.message.assistant{background:#2a2a4a}
.input-row{display:flex;gap:10px}
input{flex:1;background:#0f0f1a;border:1px solid #2a2a4a;color:#e0e0e0;padding:12px;border-radius:8px;font-size:1rem}
button{background:linear-gradient(90deg,#00d4ff,#7b2cbf);color:white;border:none;padding:12px 24px;border-radius:8px;cursor:pointer;font-weight:600;font-size:1rem}
button:hover{opacity:.9}
.code-block{background:#0f0f1a;border-radius:8px;padding:16px;overflow-x:auto;font-family:"Courier New",monospace;font-size:.85rem;color:#dcdcaa;margin:10px 0}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}
.nav{display:flex;gap:10px;margin-bottom:30px;flex-wrap:wrap}
.nav a{color:#00d4ff;text-decoration:none;padding:8px 16px;border:1px solid #2a2a4a;border-radius:8px;font-size:.9rem}
.nav a:hover{background:#1a1a2e}
.key-input{width:100%;margin-bottom:10px}
.alert{background:#1a472a;border:1px solid #2a5a3a;color:#7ee787;padding:12px;border-radius:8px;margin-bottom:16px;display:none}
.alert.error{background:#4a1a1a;border-color:#5a2a2a;color:#ff6b6b}
@media(max-width:600px){.grid{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="container">
<div class="nav">
<a href="/ui">Dashboard</a>
<a href="/api-docs">API Docs</a>
<a href="/docs">Swagger UI</a>
<a href="/redoc">ReDoc</a>
</div>
<h1>Ollama API Server</h1>
<p class="subtitle">Self-hosted LLM with FastAPI + Ollama + PostgreSQL Keys</p>

<div class="card">
<h2>API Key Required</h2>
<p style="color:#888;margin-bottom:12px">All API endpoints require a Bearer token. Enter your key below to test:</p>
<input type="password" id="apiKeyInput" class="key-input" placeholder="Enter your API key (ollama_xxxxxxxx...)" value="">
<div class="alert" id="keyAlert"></div>
</div>

<div class="card">
<h2>Connection Status</h2>
<div id="status">Checking...</div>
<div style="margin-top:10px;font-size:.9rem;color:#888">Base URL: <span id="baseUrl"></span></div>
</div>

<div class="card">
<h2>API Endpoints</h2>
<div class="endpoint"><span class="method">GET</span><span class="url">/health</span> - Health check</div>
<div class="endpoint"><span class="method">GET</span><span class="url">/v1/models</span> - List models</div>
<div class="endpoint"><span class="method">POST</span><span class="url">/v1/chat/completions</span> - Chat (OpenAI-compatible)</div>
<div class="endpoint"><span class="method">POST</span><span class="url">/api/generate</span> - Generate text</div>
</div>

<div class="card">
<h2>Test Chat</h2>
<div class="chat-box" id="chatBox"></div>
<div class="input-row">
<input type="text" id="chatInput" placeholder="Type a message..." onkeypress="if(event.key==='Enter')sendChat()">
<button onclick="sendChat()">Send</button>
</div>
</div>

<div class="grid">
<div class="card">
<h2>Python Example</h2>
<div class="code-block">
import requests<br><br>
url = "<span class='base-url'></span>/v1/chat/completions"<br>
headers = {<br>
&nbsp;&nbsp;"Content-Type": "application/json",<br>
&nbsp;&nbsp;"Authorization": "Bearer YOUR_API_KEY"<br>
}<br>
data = {<br>
&nbsp;&nbsp;"model": "qwen2.5:0.5b",<br>
&nbsp;&nbsp;"messages": [{"role": "user", "content": "Hello!"}]<br>
}<br><br>
res = requests.post(url, json=data, headers=headers)<br>
print(res.json())
</div>
</div>
<div class="card">
<h2>cURL Example</h2>
<div class="code-block">
curl -X POST <span class='base-url'></span>/v1/chat/completions \<br>
&nbsp;&nbsp;-H "Content-Type: application/json" \<br>
&nbsp;&nbsp;-H "Authorization: Bearer YOUR_API_KEY" \<br>
&nbsp;&nbsp;-d '{<br>
&nbsp;&nbsp;&nbsp;&nbsp;"model": "qwen2.5:0.5b",<br>
&nbsp;&nbsp;&nbsp;&nbsp;"messages": [{"role": "user", "content": "Hello!"}]<br>
&nbsp;&nbsp;}'
</div>
</div>
</div>
</div>

<script>
const baseUrl = window.location.origin;
document.querySelectorAll('.base-url').forEach(el => el.textContent = baseUrl);
document.getElementById('baseUrl').textContent = baseUrl;

async function checkStatus() {
    const statusEl = document.getElementById('status');
    try {
        const res = await fetch('/health');
        const data = await res.json();
        if (data.status === 'ok') {
            statusEl.innerHTML = '<span class="status ok">ONLINE</span> Ollama + Database connected';
        } else {
            statusEl.innerHTML = '<span class="status warn">DEGRADED</span> Ollama is starting...';
        }
    } catch (e) {
        statusEl.innerHTML = '<span class="status warn">OFFLINE</span> Server unreachable';
    }
}

async function sendChat() {
    const input = document.getElementById('chatInput');
    const key = document.getElementById('apiKeyInput').value;
    const chatBox = document.getElementById('chatBox');
    const alert = document.getElementById('keyAlert');
    
    if (!key) {
        alert.textContent = 'Please enter an API key first';
        alert.style.display = 'block';
        alert.className = 'alert error';
        return;
    }
    
    const text = input.value.trim();
    if (!text) return;
    
    alert.style.display = 'none';
    input.value = '';
    chatBox.innerHTML += `<div class="message user">${text}</div>`;
    chatBox.scrollTop = chatBox.scrollHeight;
    
    const assistantMsg = document.createElement('div');
    assistantMsg.className = 'message assistant';
    assistantMsg.textContent = '...';
    chatBox.appendChild(assistantMsg);
    
    try {
        const res = await fetch('/v1/chat/completions', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${key}`
            },
            body: JSON.stringify({
                model: 'qwen2.5:0.5b',
                messages: [{role: 'user', content: text}]
            })
        });
        
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Failed to connect');
        }
        
        const data = await res.json();
        assistantMsg.textContent = data.choices[0].message.content;
    } catch (e) {
        assistantMsg.textContent = 'Error: ' + e.message;
        assistantMsg.style.color = '#ff6b6b';
    }
    chatBox.scrollTop = chatBox.scrollHeight;
}

checkStatus();
setInterval(checkStatus, 10000);
</script>
</body>
</html>
"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
