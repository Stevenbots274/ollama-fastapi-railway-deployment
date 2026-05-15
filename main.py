import os
import time
import secrets
import hashlib
import httpx
import asyncio
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

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

# Persistent API key storage
import json
DB_PATH = os.path.join(os.getenv("OLLAMA_MODELS", "/root/.ollama/models"), "api_keys.json")

def load_keys():
    if os.path.exists(DB_PATH):
        try:
            with open(DB_PATH, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_keys(keys):
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with open(DB_PATH, "w") as f:
        json.dump(keys, f)

API_KEYS: Dict[str, Dict[str, Any]] = load_keys()

security = HTTPBearer(auto_error=False)

def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not credentials:
        raise HTTPException(status_code=401, detail="Missing Authorization header. Use: Bearer YOUR_API_KEY")
    token = credentials.credentials
    key_hash = hash_key(token)
    if key_hash not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")
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
        "version": "2.0.0",
        "auth": "Bearer token required for API access",
        "documentation": "/docs",
        "redoc": "/redoc",
        "web_ui": "/ui",
        "api_docs": "/api-docs",
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
            return {"status": "ok", "ollama": "connected", "auth": "enabled"}
    except:
        return {"status": "degraded", "ollama": "not ready", "auth": "enabled"}

# ============ KEY MANAGEMENT (MASTER KEY PROTECTED) ============

@app.post("/admin/keys")
def create_key(req: CreateKeyRequest, master: str = Depends(verify_master_key)):
    raw_key = "ollama_" + secrets.token_urlsafe(32)
    key_hash = hash_key(raw_key)
    API_KEYS[key_hash] = {
        "name": req.name,
        "created": int(time.time()),
        "rate_limit": req.rate_limit,
        "usage_count": 0,
        "key_hash": key_hash
    }
    save_keys(API_KEYS)
    return {
        "api_key": raw_key,
        "name": req.name,
        "warning": "Save this key now - it will not be shown again!"
    }

@app.get("/admin/keys")
def list_keys(master: str = Depends(verify_master_key)):
    return {"keys": list(API_KEYS.values())}

@app.post("/admin/keys/revoke")
def revoke_key(req: RevokeKeyRequest, master: str = Depends(verify_master_key)):
    if req.key_hash in API_KEYS:
        del API_KEYS[req.key_hash]
        save_keys(API_KEYS)
        return {"status": "revoked"}
    raise HTTPException(status_code=404, detail="Key not found")

# ============ PROTECTED API ENDPOINTS ============

@app.get("/v1/models")
def list_models(api_key: str = Depends(verify_api_key)):
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=30)
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
            from fastapi.responses import StreamingResponse
            async def streamer():
                async with httpx.AsyncClient(timeout=None) as client:
                    async with client.stream("POST", f"{OLLAMA_HOST}/api/chat", json=payload) as r:
                        async for line in r.aiter_lines():
                            if line:
                                yield line + "\n"
            return StreamingResponse(streamer(), media_type="text/event-stream")

        async with httpx.AsyncClient(timeout=300) as client:
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
        raise HTTPException(status_code=503, detail=f"Ollama error: {str(e)}")

@app.post("/api/generate")
def generate(req: GenerateRequest, api_key: str = Depends(verify_api_key)):
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
        r = requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, stream=req.stream, timeout=120)

        if req.stream:
            from fastapi.responses import StreamingResponse
            def streamer():
                for line in r.iter_lines():
                    if line:
                        yield line.decode("utf-8") + "\n"
            return StreamingResponse(streamer(), media_type="application/x-ndjson")

        return r.json()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Ollama error: {str(e)}")

@app.get("/api/models")
def ollama_models(api_key: str = Depends(verify_api_key)):
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=30)
        return r.json()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Ollama error: {str(e)}")

@app.post("/api/pull")
def pull_model(req: PullModelRequest, api_key: str = Depends(verify_api_key)):
    try:
        payload = {"name": req.name, "stream": False}
        r = requests.post(f"{OLLAMA_HOST}/api/pull", json=payload, timeout=600)
        return r.json()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Ollama error: {str(e)}")

@app.post("/api/delete")
def delete_model(req: DeleteModelRequest, api_key: str = Depends(verify_api_key)):
    try:
        payload = {"name": req.name}
        r = requests.delete(f"{OLLAMA_HOST}/api/delete", json=payload, timeout=30)
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
<p class="subtitle">Self-hosted LLM with FastAPI + Ollama + API Keys</p>

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
function showAlert(msg,isError){
const el=document.getElementById('keyAlert');
el.textContent=msg;
el.style.display='block';
if(isError){el.classList.add('error');}else{el.classList.remove('error');}
}
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
async function sendChat(){
const key = getApiKey();
if(!key){showAlert("Please enter your API key above",true);return;}
const input = document.getElementById('chatInput');
const box = document.getElementById('chatBox');
const msg = input.value.trim();
if(!msg) return;
box.innerHTML += `<div class="message user">${msg}</div>`;
input.value = '';
box.scrollTop = box.scrollHeight;
try{
const res = await fetch(baseUrl + '/v1/chat/completions', {
method: 'POST',
headers: {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key},
body: JSON.stringify({model: 'qwen2.5:0.5b', messages: [{role: 'user', content: msg}]})
});
if(res.status===401){showAlert("Invalid API key!",true);return;}
const data = await res.json();
const reply = data.choices?.[0]?.message?.content || 'No response';
box.innerHTML += `<div class="message assistant">${reply}</div>`;
box.scrollTop = box.scrollHeight;
showAlert("Message sent successfully",false);
}catch(e){
box.innerHTML += `<div class="message assistant" style="color:#ff6b6b">Error: ${e.message}</div>`;
}
}
</script>
</body>
</html>"""

@app.get("/api-docs", response_class=HTMLResponse)
def api_docs():
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>API Documentation - Ollama Server</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;background:#0f0f23;color:#e0e0e0;min-height:100vh}
.container{max-width:1000px;margin:0 auto;padding:40px 20px}
h1{font-size:2.5rem;margin-bottom:10px;background:linear-gradient(90deg,#00d4ff,#7b2cbf);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.subtitle{color:#888;margin-bottom:40px}
.card{background:#1a1a2e;border:1px solid #2a2a4a;border-radius:12px;padding:24px;margin-bottom:20px}
.card h2{color:#00d4ff;margin-bottom:16px;font-size:1.2rem}
.card h3{color:#e0e0e0;margin:20px 0 10px;font-size:1rem}
.endpoint{background:#0f0f1a;border-left:3px solid #00d4ff;padding:12px 16px;margin:8px 0;border-radius:0 8px 8px 0;font-family:"Courier New",monospace;font-size:.9rem}
.method{color:#7ee787;font-weight:bold;margin-right:8px}
.url{color:#dcdcaa}
.code-block{background:#0f0f1a;border-radius:8px;padding:16px;overflow-x:auto;font-family:"Courier New",monospace;font-size:.85rem;color:#dcdcaa;margin:10px 0}
.table{width:100%;border-collapse:collapse;margin:10px 0}
.table th,.table td{padding:10px;text-align:left;border-bottom:1px solid #2a2a4a;font-size:.9rem}
.table th{color:#00d4ff;font-weight:600}
.table td{color:#ccc}
.nav{display:flex;gap:10px;margin-bottom:30px;flex-wrap:wrap}
.nav a{color:#00d4ff;text-decoration:none;padding:8px 16px;border:1px solid #2a2a4a;border-radius:8px;font-size:.9rem}
.nav a:hover{background:#1a1a2e}
.note{background:#1a3a5c;border-left:3px solid #00d4ff;padding:12px 16px;margin:10px 0;border-radius:0 8px 8px 0;font-size:.9rem;color:#e0e0e0}
.warning{background:#4a3a1a;border-left:3px solid #ffa500;padding:12px 16px;margin:10px 0;border-radius:0 8px 8px 0;font-size:.9rem;color:#ffa500}
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
<p class="subtitle">Complete reference for Ollama FastAPI Server endpoints</p>

<div class="card">
<h2>Authentication</h2>
<p style="color:#888;margin-bottom:10px">All API endpoints (except /health and UI pages) require a Bearer token.</p>
<div class="code-block">Authorization: Bearer YOUR_API_KEY</div>
<div class="warning">Keep your API keys secret. They grant full access to the LLM API.</div>
</div>

<div class="card">
<h2>Key Management (Master Key Required)</h2>
<p style="color:#888;margin-bottom:10px">Use your MASTER_KEY in the X-Master-Key header to manage API keys.</p>
<div class="endpoint"><span class="method">POST</span><span class="url">/admin/keys</span> - Create new API key</div>
<div class="code-block">Headers: X-Master-Key: your-master-key<br>
Body: {"name": "my-app", "rate_limit": 1000}<br><br>
Response: {"api_key": "ollama_xxxxx", "warning": "Save this key now!"}</div>
<div class="endpoint"><span class="method">GET</span><span class="url">/admin/keys</span> - List all keys</div>
<div class="endpoint"><span class="method">POST</span><span class="url">/admin/keys/revoke</span> - Revoke a key</div>
<div class="code-block">Body: {"key_hash": "sha256_hash_of_key"}</div>
</div>

<div class="card">
<h2>OpenAI-Compatible Endpoints</h2>
<h3>List Models</h3>
<div class="endpoint"><span class="method">GET</span><span class="url">/v1/models</span></div>
<p style="color:#888;margin:8px 0">Returns available models in OpenAI format.</p>
<div class="code-block">Response:<br>
{<br>
&nbsp;&nbsp;"object": "list",<br>
&nbsp;&nbsp;"data": [{"id": "qwen2.5:0.5b", "object": "model", "created": 1234567890, "owned_by": "ollama"}]<br>
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
-H "Authorization: Bearer YOUR_API_KEY"<br>
-H "Content-Type: application/json"<br>
-d '{"model":"qwen2.5:0.5b","messages":[{"role":"user","content":"Hello!"}]}'</div>
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
-H "Authorization: Bearer YOUR_API_KEY"<br>
-d '{"name": "llama3.2:1b"}'</div>
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

</div>
<script>
document.querySelectorAll('.base-url').forEach(el=>el.textContent=window.location.origin);
</script>
</body>
</html>"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
