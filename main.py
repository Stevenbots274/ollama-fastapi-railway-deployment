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

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Ollama API Server",
    description="Self-hosted LLM API with Ollama + FastAPI. API key protected.",
    version="2.0.0",
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
DATABASE_URL = os.getenv("DATABASE_URL")

# Azure OpenAI Configuration
# Use AZURE_OPENAI_MODEL as the base endpoint if it looks like a URL
AZURE_OPENAI_MODEL_ENV = os.getenv("AZURE_OPENAI_MODEL", "").strip()
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY", "").strip()
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o").strip()
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview").strip()

# Smart URL handling: If AZURE_OPENAI_MODEL is a URL, it's likely the resource endpoint
if AZURE_OPENAI_MODEL_ENV.startswith("http"):
    if "/openai/v1" in AZURE_OPENAI_MODEL_ENV:
        AZURE_OPENAI_ENDPOINT = AZURE_OPENAI_MODEL_ENV.split("/openai/v1")[0]
    else:
        AZURE_OPENAI_ENDPOINT = AZURE_OPENAI_MODEL_ENV

# If AZURE_OPENAI_ENDPOINT is the project URL, we still need the resource URL
# But the Azure SDK usually wants the .openai.azure.com one
if "services.ai.azure.com" in AZURE_OPENAI_ENDPOINT and not AZURE_OPENAI_MODEL_ENV.startswith("http"):
    # Fallback or warning if needed, but we'll try to use what's given
    pass

USE_AZURE = bool(AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_KEY)

if USE_AZURE:
    from openai import AzureOpenAI
    try:
        azure_client = AzureOpenAI(
            api_key=AZURE_OPENAI_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT
        )
        logger.info("Azure OpenAI client initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize Azure client: {e}")
        azure_client = None
        USE_AZURE = False
else:
    azure_client = None

# Database Setup
Base = declarative_base()

class APIKey(Base):
    __tablename__ = "api_keys"
    id = Column(Integer, primary_key=True)
    key_hash = Column(String(64), unique=True, index=True)
    name = Column(String(100))
    created_at = Column(BigInteger)

engine = None
SessionLocal = None

def init_db():
    global engine, SessionLocal
    if not DATABASE_URL:
        logger.error("DATABASE_URL not set. API key management will fail.")
        return
    try:
        engine = create_engine(DATABASE_URL)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        Base.metadata.create_all(bind=engine)
        logger.info("Database initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")

@app.on_event("startup")
async def startup_event():
    init_db()
    asyncio.create_task(keep_warm_background())

async def keep_warm_background():
    """Aggressive background task that pings Ollama every 2 minutes with actual inference."""
    await asyncio.sleep(30)  # Start sooner
    while True:
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                # Use actual chat endpoint for the heartbeat to ensure model stays in GPU/RAM
                payload = {
                    "model": DEFAULT_MODEL,
                    "messages": [{"role": "user", "content": "heartbeat"}],
                    "stream": False,
                    "options": {"num_predict": 1}
                }
                await client.post(f"{OLLAMA_HOST}/api/chat", json=payload)
                logger.info(f"Aggressive keep-warm heartbeat sent for {DEFAULT_MODEL}")
        except Exception as e:
            logger.warning(f"Keep-warm heartbeat failed: {e}")
        await asyncio.sleep(120)  # Ping every 2 minutes (more frequent)

def get_db():
    if SessionLocal is None:
        raise HTTPException(status_code=503, detail="Database not initialized")
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Security
security = HTTPBearer(auto_error=False)

def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security), db = Depends(get_db)):
    if not credentials:
        raise HTTPException(status_code=401, detail="Missing Authorization header. Use: Bearer YOUR_API_KEY")
    token = credentials.credentials
    
    # Allow Master Key for all endpoints
    if token == MASTER_KEY:
        return token
        
    key_hash = hash_key(token)
    key_record = db.query(APIKey).filter(APIKey.key_hash == key_hash).first()
    if not key_record:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return token

def verify_master_key(x_master_key: str = Header(None)):
    if not x_master_key:
        raise HTTPException(status_code=401, detail="Missing X-Master-Key header")
    if x_master_key != MASTER_KEY:
        raise HTTPException(status_code=403, detail="Invalid master key")
    return x_master_key

# Request Models
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

class CreateKeyRequest(BaseModel):
    name: str

class RevokeKeyRequest(BaseModel):
    key_hash: str

# Endpoints
@app.get("/health")
async def health():
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
            if r.status_code == 200:
                return {"status": "ok", "ollama": "connected", "auth": "enabled"}
    except:
        pass
    return {"status": "degraded", "ollama": "not ready", "auth": "enabled"}

@app.get("/ping")
async def ping():
    """Lightweight keep-alive endpoint for external pingers. No auth required."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{OLLAMA_HOST}/api/tags")
            if r.status_code == 200:
                return {"status": "alive", "timestamp": int(time.time())}
    except:
        pass
    return {"status": "starting", "timestamp": int(time.time())}

@app.post("/warmup")
async def warmup():
    """Send a tiny inference to keep the model loaded in RAM. No auth required."""
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            payload = {
                "model": DEFAULT_MODEL,
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": False,
                "options": {"num_predict": 1}
            }
            r = await client.post(f"{OLLAMA_HOST}/api/chat", json=payload)
            return {"status": "warm", "model": DEFAULT_MODEL, "timestamp": int(time.time())}
    except Exception as e:
        return {"status": "error", "detail": str(e)}

@app.get("/", response_class=HTMLResponse)
@app.get("/ui", response_class=HTMLResponse)
@app.get("/UI", response_class=HTMLResponse)
async def web_ui():
    return HTML_CONTENT

@app.get("/api-docs", response_class=HTMLResponse)
async def api_docs():
    return API_DOCS_CONTENT

@app.get("/v1/models")
async def list_models(api_key: str = Depends(verify_api_key)):
    models = []
    
    # Add Azure model if configured
    if USE_AZURE and AZURE_OPENAI_DEPLOYMENT_NAME:
        models.append({
            "id": AZURE_OPENAI_DEPLOYMENT_NAME,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "azure"
        })
        
    # Add Ollama models
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
            if r.status_code == 200:
                data = r.json()
                for m in data.get("models", []):
                    models.append({
                        "id": m["name"],
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "ollama"
                    })
    except Exception as e:
        logger.error(f"Ollama tags error: {e}")
        if not models:
            return {"object": "list", "data": []}
            
    return {"object": "list", "data": models}

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest, api_key: str = Depends(verify_api_key)):
    model = req.model or DEFAULT_MODEL
    
    # Check if we should use Azure
    use_azure_for_this = USE_AZURE and (req.model == AZURE_OPENAI_DEPLOYMENT_NAME or not req.model)
    
    if use_azure_for_this:
        if not azure_client:
            logger.error("Azure client is not initialized.")
            raise HTTPException(status_code=500, detail="Azure client not initialized. Check your environment variables.")
        try:
            messages = [{"role": m.role, "content": m.content} for m in req.messages]
            logger.info(f"Sending request to Azure OpenAI: {AZURE_OPENAI_DEPLOYMENT_NAME}")
            
            if req.stream:
                response = azure_client.chat.completions.create(
                    model=AZURE_OPENAI_DEPLOYMENT_NAME,
                    messages=messages,
                    temperature=req.temperature,
                    max_tokens=req.max_tokens,
                    stream=True
                )
                
                async def azure_streamer():
                    try:
                        for chunk in response:
                            if chunk.choices:
                                yield f"data: {json.dumps(chunk.model_dump())}\n\n"
                        yield "data: [DONE]\n\n"
                    except Exception as e:
                        logger.error(f"Azure streaming error: {e}")
                        yield f"data: {json.dumps({'error': str(e)})}\n\n"
                return StreamingResponse(azure_streamer(), media_type="text/event-stream")
            else:
                response = azure_client.chat.completions.create(
                    model=AZURE_OPENAI_DEPLOYMENT_NAME,
                    messages=messages,
                    temperature=req.temperature,
                    max_tokens=req.max_tokens,
                    stream=False
                )
                return response.model_dump()
        except Exception as e:
            logger.error(f"Azure OpenAI error: {e}")
            raise HTTPException(status_code=500, detail=f"Azure error: {str(e)}")
            
    # Fallback to Ollama
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
                    try:
                        async with client.stream("POST", f"{OLLAMA_HOST}/api/chat", json=payload) as response:
                            async for line in response.aiter_lines():
                                if line:
                                    data = json.loads(line)
                                    chunk = {
                                        "id": f"chatcmpl-{int(time.time())}",
                                        "object": "chat.completion.chunk",
                                        "created": int(time.time()),
                                        "model": model,
                                        "choices": [{
                                            "index": 0,
                                            "delta": {"content": data.get("message", {}).get("content", "")},
                                            "finish_reason": "stop" if data.get("done") else None
                                        }]
                                    }
                                    yield f"data: {json.dumps(chunk)}\n\n"
                            yield "data: [DONE]\n\n"
                    except Exception as e:
                        yield f"data: {json.dumps({'error': str(e)})}\n\n"
            return StreamingResponse(streamer(), media_type="text/event-stream")
        else:
            async with httpx.AsyncClient(timeout=120) as client:
                r = await client.post(f"{OLLAMA_HOST}/api/chat", json=payload)
                data = r.json()
                return {
                    "id": f"chatcmpl-{int(time.time())}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": data.get("message", {}).get("content", "")
                        },
                        "finish_reason": "stop"
                    }]
                }
    except Exception as e:
        logger.error(f"Chat completion error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

@app.post("/api/generate")
async def generate(req: GenerateRequest, api_key: str = Depends(verify_api_key)):
    model = req.model or DEFAULT_MODEL
    try:
        payload = {
            "model": model,
            "prompt": req.prompt,
            "stream": req.stream,
            "options": {
                "temperature": req.temperature,
                "num_predict": req.max_tokens
            }
        }
        if req.stream:
            async def streamer():
                async with httpx.AsyncClient(timeout=None) as client:
                    async with client.stream("POST", f"{OLLAMA_HOST}/api/generate", json=payload) as response:
                        async for line in response.aiter_lines():
                            if line:
                                yield line + "\n"
            return StreamingResponse(streamer(), media_type="application/x-ndjson")
        else:
            async with httpx.AsyncClient(timeout=120) as client:
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
        async with httpx.AsyncClient(timeout=300) as client:
            payload = {"name": req.name, "stream": False}
            r = await client.post(f"{OLLAMA_HOST}/api/pull", json=payload)
            return r.json()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Ollama error: {str(e)}")

# Admin Endpoints
@app.post("/admin/keys")
async def create_key(req: CreateKeyRequest, master: str = Depends(verify_master_key), db = Depends(get_db)):
    raw_key = "ollama_" + secrets.token_urlsafe(32)
    key_hash = hash_key(raw_key)
    new_key = APIKey(
        key_hash=key_hash,
        name=req.name,
        created_at=int(time.time())
    )
    db.add(new_key)
    db.commit()
    return {
        "api_key": raw_key,
        "name": req.name,
        "warning": "Save this key now - it will not be shown again!"
    }

@app.get("/admin/keys")
async def list_keys(master: str = Depends(verify_master_key), db = Depends(get_db)):
    keys = db.query(APIKey).all()
    return {"keys": [{"name": k.name, "key_hash": k.key_hash, "created_at": k.created_at} for k in keys]}

@app.post("/admin/keys/revoke")
async def revoke_key(req: RevokeKeyRequest, master: str = Depends(verify_master_key), db = Depends(get_db)):
    key_record = db.query(APIKey).filter(APIKey.key_hash == req.key_hash).first()
    if key_record:
        db.delete(key_record)
        db.commit()
        return {"status": "revoked"}
    raise HTTPException(status_code=404, detail="Key not found")

HTML_CONTENT = """<!DOCTYPE html>
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
table{width:100%;border-collapse:collapse;margin-top:10px}
th,td{text-align:left;padding:12px;border-bottom:1px solid #2a2a4a}
th{color:#00d4ff;font-size:.9rem}
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
<div class="endpoint"><span class="method">GET</span><span class="url">/api/models</span> - List models (native)</div>
<div class="endpoint"><span class="method">POST</span><span class="url">/api/pull</span> - Pull model</div>
</div>

<div class="card">
<h2>Test Chat</h2>
<div class="chat-box" id="chatBox"></div>
<div class="input-row">
<input type="text" id="chatInput" placeholder="Type a message..." onkeypress="if(event.key==='Enter')sendChat()">
<button id="sendBtn" onclick="sendChat()">Send</button>
</div>
</div>

<div class="card">
<h2>Available Models</h2>
<table id="modelsTable">
<thead>
<tr><th>Model</th><th>ID</th><th>Created</th></tr>
</thead>
<tbody id="modelsBody">
<tr><td colspan="3">Enter API key to load models...</td></tr>
</tbody>
</table>
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
curl -X POST <span class='base-url'></span>/v1/chat/completions<br>
-H "Content-Type: application/json"<br>
-H "Authorization: Bearer YOUR_API_KEY"<br>
-d '{"model":"qwen2.5:0.5b","messages":[{"role":"user","content":"Hello!"}]}'
</div>
</div>
</div>
</div>
<script>
const baseUrl = window.location.origin;
document.getElementById('baseUrl').textContent = baseUrl;
document.querySelectorAll('.base-url').forEach(el=>el.textContent=baseUrl);

function getApiKey(){return document.getElementById('apiKeyInput').value.trim();}

async function checkHealth(){
try{
const res = await fetch(baseUrl + '/health');
const data = await res.json();
const el = document.getElementById('status');
if(data.ollama === 'connected'){
el.innerHTML = '<span class="status ok">Ollama Connected | Auth Enabled</span>';
}else{
el.innerHTML = '<span class="status warn">Ollama Starting...</span>';
}
}catch{
document.getElementById('status').innerHTML = '<span class="status warn">Server Starting...</span>';
}
}
checkHealth();
setInterval(checkHealth, 5000);

async function loadModels() {
    const key = getApiKey();
    if(!key) return;
    try {
        const res = await fetch(baseUrl + '/v1/models', {
            headers: {'Authorization': 'Bearer ' + key}
        });
        if(res.ok) {
            const data = await res.json();
            const body = document.getElementById('modelsBody');
            body.innerHTML = '';
            data.data.forEach(m => {
                const row = `<tr><td>${m.id}</td><td>${m.id}</td><td>${new Date(m.created * 1000).toLocaleString()}</td></tr>`;
                body.innerHTML += row;
            });
        }
    } catch(e) {}
}
document.getElementById('apiKeyInput').addEventListener('change', loadModels);

async function sendChat(){
const key = getApiKey();
if(!key){alert("Please enter your API key first");return;}
const input = document.getElementById('chatInput');
const btn = document.getElementById('sendBtn');
const box = document.getElementById('chatBox');
const text = input.value.trim();
if(!text) return;

input.value = '';
box.innerHTML += `<div class="message user">${text}</div>`;
box.scrollTop = box.scrollHeight;

try {
    btn.disabled = true;
    const res = await fetch(baseUrl + '/v1/chat/completions', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + key
        },
        body: JSON.stringify({
            model: "qwen2.5:0.5b",
            messages: [{role: "user", content: text}],
            stream: true
        })
    });
    
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let assistantMsg = document.createElement('div');
    assistantMsg.className = 'message assistant';
    box.appendChild(assistantMsg);
    
    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value);
        const lines = chunk.split('\\n');
        for (const line of lines) {
            if (line.startsWith('data: ')) {
                const dataStr = line.slice(6);
                if (dataStr === '[DONE]') break;
                try {
                    const data = JSON.parse(dataStr);
                    const content = data.choices[0].delta.content || '';
                    assistantMsg.textContent += content;
                    box.scrollTop = box.scrollHeight;
                } catch (e) {}
            }
        }
    }
} catch(e) {
    box.innerHTML += `<div class="message assistant error">Error: ${e.message}</div>`;
} finally {
    btn.disabled = false;
    box.scrollTop = box.scrollHeight;
}
}
</script>
</body>
</html>"""

API_DOCS_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>API Documentation - Ollama API Server</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;background:#0f0f23;color:#e0e0e0;min-height:100vh}
.container{max-width:900px;margin:0 auto;padding:40px 20px}
h1{font-size:2.5rem;margin-bottom:10px;background:linear-gradient(90deg,#00d4ff,#7b2cbf);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.nav{display:flex;gap:10px;margin-bottom:30px;flex-wrap:wrap}
.nav a{color:#00d4ff;text-decoration:none;padding:8px 16px;border:1px solid #2a2a4a;border-radius:8px;font-size:.9rem}
.nav a:hover{background:#1a1a2e}
.card{background:#1a1a2e;border:1px solid #2a2a4a;border-radius:12px;padding:24px;margin-bottom:20px}
.endpoint{background:#0f0f1a;border-left:3px solid #00d4ff;padding:12px 16px;margin:8px 0;border-radius:0 8px 8px 0;font-family:"Courier New",monospace;font-size:.9rem}
.method{color:#7ee787;font-weight:bold;margin-right:8px}
.url{color:#dcdcaa}
.table{width:100%;border-collapse:collapse;margin:16px 0}
.table th,.table td{text-align:left;padding:12px;border-bottom:1px solid #2a2a4a}
.table th{color:#00d4ff;font-size:.9rem}
.code-block{background:#0f0f1a;border-radius:8px;padding:16px;overflow-x:auto;font-family:"Courier New",monospace;font-size:.85rem;color:#dcdcaa;margin:10px 0}
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
<h1>API Documentation</h1>

<div class="card">
<h2>Authentication</h2>
<p>All endpoints (except /health) require an API key passed as a Bearer token in the Authorization header.</p>
<div class="code-block">Authorization: Bearer ollama_xxxxxxxxxxxxxxxx</div>
</div>

<div class="card">
<h2>OpenAI Compatible Endpoints</h2>
<h3>Chat Completions</h3>
<div class="endpoint"><span class="method">POST</span><span class="url">/v1/chat/completions</span></div>
<table class="table">
<tr><th>Parameter</th><th>Type</th><th>Required</th><th>Description</th></tr>
<tr><td>model</td><td>string</td><td>No</td><td>Model name (e.g. qwen2.5:0.5b)</td></tr>
<tr><td>messages</td><td>array</td><td>Yes</td><td>List of message objects</td></tr>
<tr><td>stream</td><td>boolean</td><td>No</td><td>Stream response</td></tr>
</table>
<div class="code-block">curl -X POST <span class="base-url"></span>/v1/chat/completions<br>
-H "Authorization: Bearer YOUR_API_KEY"<br>
-H "Content-Type: application/json"<br>
-d '{"model":"qwen2.5:0.5b","messages":[{"role":"user","content":"Hello!"}]}'</div>
</div>

<div class="card">
<h2>Key Management (Admin)</h2>
<p>Requires <code>X-Master-Key</code> header.</p>
<h3>Create Key</h3>
<div class="endpoint"><span class="method">POST</span><span class="url">/admin/keys</span></div>
<div class="code-block">curl -X POST <span class="base-url"></span>/admin/keys<br>
-H "X-Master-Key: YOUR_MASTER_KEY"<br>
-H "Content-Type: application/json"<br>
-d '{"name": "my-new-key"}'</div>
</div>

</div>
<script>
document.querySelectorAll('.base-url').forEach(el=>el.textContent=window.location.origin);
</script>
</body>
</html>"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
