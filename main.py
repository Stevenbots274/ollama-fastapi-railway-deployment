
import os
import json
import asyncio
import uvicorn
from typing import List, Dict, Any, Union, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel, Field

from sqlalchemy import create_engine, Column, Integer, String, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

import bcrypt
import secrets
import httpx

# Conditional import for AzureOpenAI
USE_AZURE = os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT")
if USE_AZURE:
    from openai import AzureOpenAI

# --- App Setup ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    asyncio.create_task(keep_warm_background())
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Environment Variables ---

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "tinyllama:latest")
MASTER_KEY = os.getenv("MASTER_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

# --- Azure OpenAI Configuration ---

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENTS_STR = os.getenv("AZURE_OPENAI_DEPLOYMENTS", "gpt-4o-mini,gpt-35-turbo,gpt-4")
AZURE_OPENAI_DEPLOYMENTS = [d.strip() for d in AZURE_OPENAI_DEPLOYMENTS_STR.split(",")]
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

azure_client = None
if USE_AZURE:
    azure_client = AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT
    )

# --- Database Setup ---

Base = declarative_base()

class APIKey(Base):
    __tablename__ = "api_keys"
    id = Column(Integer, primary_key=True, index=True)
    hashed_key = Column(String, unique=True, index=True)
    is_master = Column(Boolean, default=False)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Background Task for Model Warm-up ---

async def keep_warm_background():
    while True:
        try:
            # Ping Ollama to keep the default model warm
            async with httpx.AsyncClient() as client:
                await client.post(f"{OLLAMA_HOST}/api/generate", json={
                    "model": DEFAULT_MODEL,
                    "prompt": "Hi",
                    "stream": False,
                    "options": {"num_predict": 1}
                }, timeout=30.0)
            print(f"Heartbeat: {DEFAULT_MODEL} kept warm.")
        except Exception as e:
            print(f"Heartbeat failed: {e}")
        await asyncio.sleep(300) # Ping every 5 minutes

# --- Security ---

security = HTTPBearer()

def hash_key(key: str) -> str:
    return bcrypt.hashpw(key.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

def verify_api_key(api_key: str, db: Session = Depends(get_db)) -> bool:
    if MASTER_KEY and api_key == MASTER_KEY:
        return True
    # Iterate through all stored keys and verify
    for stored_key_obj in db.query(APIKey).all():
        try:
            if bcrypt.checkpw(api_key.encode("utf-8"), stored_key_obj.hashed_key.encode("utf-8")):
                return True
        except ValueError:
            # Handle cases where hashed_key might be malformed or not a bcrypt hash
            continue
    return False

def verify_master_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> bool:
    if not MASTER_KEY:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Master key not set")
    if credentials.credentials == MASTER_KEY:
        return True
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid master key")

# --- Pydantic Models ---

class ChatMessage(BaseModel):
    role: str
    content: Optional[Union[str, List[Dict[str, Any]]]] = None # For multimodal input
    images: Optional[List[str]] = None # For base64 image data

class ChatRequest(BaseModel):
    model: str = Field(..., example="tinyllama:latest")
    messages: List[ChatMessage]
    stream: bool = False
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None

class GenerateRequest(BaseModel):
    model: str = Field(..., example="tinyllama:latest")
    prompt: str
    stream: bool = False
    system: Optional[str] = None
    template: Optional[str] = None
    context: Optional[List[int]] = None
    raw: Optional[bool] = False
    format: Optional[str] = None
    options: Optional[Dict[str, Any]] = None

class PullModelRequest(BaseModel):
    name: str

class CreateKeyRequest(BaseModel):
    key: Optional[str] = None

class RevokeKeyRequest(BaseModel):
    key: str

# --- Endpoints ---

@app.get("/health", tags=["Health Check"])
async def health_check():
    ollama_status = "disconnected"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{OLLAMA_HOST}/api/tags", timeout=5.0)
            if response.status_code == 200:
                ollama_status = "connected"
    except httpx.RequestError:
        pass
    return {"status": "ok", "ollama": ollama_status, "auth": "enabled" if MASTER_KEY else "disabled"}

@app.get("/ping", tags=["Health Check"])
async def ping():
    return {"message": "pong"}

@app.get("/warmup", tags=["Health Check"])
async def warmup_status():
    return {"message": f"Keeping {DEFAULT_MODEL} warm."}

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
@app.get("/ui", response_class=HTMLResponse, include_in_schema=False)
@app.get("/UI", response_class=HTMLResponse, include_in_schema=False)
async def get_root():
    return HTML_CONTENT

@app.get("/api-docs", response_class=HTMLResponse, include_in_schema=False)
async def get_api_docs():
    return API_DOCS_CONTENT

@app.get("/v1/models", tags=["Models"])
async def get_models(request: Request, db: Session = Depends(get_db)):
    api_key_header = request.headers.get("Authorization")
    if api_key_header and api_key_header.startswith("Bearer "):
        api_key = api_key_header.split(" ")[1]
        if not verify_api_key(api_key, db):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    else:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing Authorization header")

    ollama_models = []
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{OLLAMA_HOST}/api/tags", timeout=10.0)
            response.raise_for_status()
            data = response.json()
            ollama_models = [{
                "id": model["name"],
                "object": "model",
                "created": 1678901234,
                "owned_by": "ollama",
            } for model in data.get("models", [])]
    except httpx.RequestError:
        pass
    except httpx.HTTPStatusError as e:
        print(f"Error fetching Ollama models: {e}")

    azure_models = []
    if USE_AZURE:
        for deployment_name in AZURE_OPENAI_DEPLOYMENTS:
            azure_models.append({
                "id": deployment_name,
                "object": "model",
                "created": 1678901234, # Placeholder
                "owned_by": "azure",
            })

    return {"data": ollama_models + azure_models, "object": "list"}

@app.post("/v1/chat/completions", tags=["Chat"])
async def chat_completions(req: ChatRequest, request: Request, db: Session = Depends(get_db)):
    api_key_header = request.headers.get("Authorization")
    if api_key_header and api_key_header.startswith("Bearer "):
        api_key = api_key_header.split(" ")[1]
        if not verify_api_key(api_key, db):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    else:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing Authorization header")

    requested_model = req.model

    # Prepare messages for Ollama or Azure
    ollama_messages = []
    azure_messages = []
    for msg in req.messages:
        ollama_msg_content = msg.content if isinstance(msg.content, str) else ""
        ollama_msg = {"role": msg.role, "content": ollama_msg_content}
        
        azure_content_parts = []
        if isinstance(msg.content, str):
            if msg.content:
                azure_content_parts.append({"type": "text", "text": msg.content})
        elif isinstance(msg.content, list):
            azure_content_parts.extend(msg.content)

        if msg.images:
            # Ollama expects base64 encoded images directly in the message
            ollama_msg["images"] = msg.images
            # Azure expects image URLs or base64 data in a specific content block format
            for img_url_or_base64 in msg.images:
                if img_url_or_base64.startswith("http") or img_url_or_base64.startswith("data:image"): # Assume it's a URL or base64 data URI
                    azure_content_parts.append({"type": "image_url", "image_url": {"url": img_url_or_base64}})
                else:
                    # If it's just base64 string without data URI prefix, add it
                    azure_content_parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_url_or_base64}"}})
        
        azure_msg = {"role": msg.role, "content": azure_content_parts}
        ollama_messages.append(ollama_msg)
        azure_messages.append(azure_msg)

    # Route to Azure OpenAI if the model is in AZURE_OPENAI_DEPLOYMENTS
    if USE_AZURE and requested_model in AZURE_OPENAI_DEPLOYMENTS:
        if req.stream:
            async def azure_streaming_generator():
                try:
                    stream = await azure_client.chat.completions.create(
                        model=requested_model,
                        messages=azure_messages,
                        temperature=req.temperature,
                        max_tokens=req.max_tokens,
                        stream=True
                    )
                    for chunk in stream:
                        if chunk.choices and chunk.choices[0].delta.content is not None:
                            yield f"data: {json.dumps(chunk.model_dump())}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    print(f"Azure streaming error: {e}")
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"
            return StreamingResponse(azure_streaming_generator(), media_type="text/event-stream")
        else:
            response = await azure_client.chat.completions.create(
                model=requested_model,
                messages=azure_messages,
                temperature=req.temperature,
                max_tokens=req.max_tokens,
                stream=False
            )
            return response.model_dump()
    else:
        # Route to Ollama
        ollama_request_data = {
            "model": requested_model,
            "messages": ollama_messages,
            "stream": req.stream,
            "options": {"temperature": req.temperature}
        }
        if req.max_tokens:
            ollama_request_data["options"]["num_predict"] = req.max_tokens

        if req.stream:
            async def ollama_streaming_generator():
                try:
                    async with httpx.AsyncClient(timeout=None) as client:
                        async with client.stream("POST", f"{OLLAMA_HOST}/api/chat", json=ollama_request_data) as response:
                            response.raise_for_status()
                            async for chunk in response.aiter_bytes():
                                try:
                                    # Ollama sends newline-delimited JSON objects
                                    for line in chunk.decode().splitlines():
                                        if line.strip():
                                            yield f"data: {line}\n\n"
                                except json.JSONDecodeError:
                                    print(f"Could not decode JSON from chunk: {chunk.decode()}")
                                    continue
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    print(f"Ollama streaming error: {e}")
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"
            return StreamingResponse(ollama_streaming_generator(), media_type="text/event-stream")
        else:
            async with httpx.AsyncClient() as client:
                response = await client.post(f"{OLLAMA_HOST}/api/chat", json=ollama_request_data, timeout=None)
                response.raise_for_status()
                return response.json()

@app.post("/api/generate", tags=["Generate (Ollama specific)"])
async def generate_completion(req: GenerateRequest, request: Request, db: Session = Depends(get_db)):
    api_key_header = request.headers.get("Authorization")
    if api_key_header and api_key_header.startswith("Bearer "):
        api_key = api_key_header.split(" ")[1]
        if not verify_api_key(api_key, db):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    else:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing Authorization header")

    ollama_request_data = req.model_dump(exclude_unset=True)
    ollama_request_data["model"] = req.model

    if req.stream:
        async def ollama_generate_streaming_generator():
            try:
                async with httpx.AsyncClient(timeout=None) as client:
                    async with client.stream("POST", f"{OLLAMA_HOST}/api/generate", json=ollama_request_data) as response:
                        response.raise_for_status()
                        async for chunk in response.aiter_bytes():
                            try:
                                for line in chunk.decode().splitlines():
                                    if line.strip():
                                        yield f"data: {line}\n\n"
                            except json.JSONDecodeError:
                                print(f"Could not decode JSON from chunk: {chunk.decode()}")
                                continue
                yield "data: [DONE]\n\n"
            except Exception as e:
                print(f"Ollama generate streaming error: {e}")
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
        return StreamingResponse(ollama_generate_streaming_generator(), media_type="text/event-stream")
    else:
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{OLLAMA_HOST}/api/generate", json=ollama_request_data, timeout=None)
            response.raise_for_status()
            return response.json()

@app.get("/admin/keys", tags=["Admin"], response_model=List[Dict[str, Any]])
async def get_api_keys(master_key_valid: bool = Depends(verify_master_key), db: Session = Depends(get_db)):
    keys = db.query(APIKey).all()
    return [{
        "id": key.id,
        "hashed_key": key.hashed_key,
        "is_master": key.is_master
    } for key in keys]

@app.post("/admin/keys", tags=["Admin"])
async def create_api_key(req: CreateKeyRequest, master_key_valid: bool = Depends(verify_master_key), db: Session = Depends(get_db)):
    new_key = req.key if req.key else secrets.token_urlsafe(32)
    hashed_new_key = hash_key(new_key)
    db_key = APIKey(hashed_key=hashed_new_key)
    db.add(db_key)
    db.commit()
    db.refresh(db_key)
    return {"message": "API key created successfully", "key": new_key, "hashed_key": hashed_new_key}

@app.delete("/admin/keys", tags=["Admin"])
async def revoke_api_key(req: RevokeKeyRequest, master_key_valid: bool = Depends(verify_master_key), db: Session = Depends(get_db)):
    target_key = None
    for stored_key_obj in db.query(APIKey).all():
        try:
            if bcrypt.checkpw(req.key.encode("utf-8"), stored_key_obj.hashed_key.encode("utf-8")):
                target_key = stored_key_obj
                break
        except Exception:
            continue
    
    if target_key:
        db.delete(target_key)
        db.commit()
        return {"message": "API key revoked successfully"}
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="API key not found")

HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <title>Ollama API Server</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; max-width: 800px; margin: 40px auto; padding: 0 20px; color: #333; }
        pre { background: #f4f4f4; padding: 15px; border-radius: 5px; overflow-x: auto; }
        code { font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace; font-size: 0.9em; }
        .status { display: inline-block; padding: 4px 8px; border-radius: 4px; font-size: 0.8em; font-weight: bold; }
        .status-ok { background: #e6ffed; color: #22863a; }
        h1 { border-bottom: 1px solid #eaecef; padding-bottom: 0.3em; }
        a { color: #0366d6; text-decoration: none; }
        a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <h1>Ollama API Server <span class="status status-ok">Active</span></h1>
    <p>This is a high-performance wrapper for Ollama, providing OpenAI-compatible endpoints and API key authentication.</p>
    
    <h3>Available Endpoints</h3>
    <ul>
        <li><code>GET /health</code> - Server health check</li>
        <li><code>GET /v1/models</code> - List available models</li>
        <li><code>POST /v1/chat/completions</code> - OpenAI-compatible chat</li>
        <li><code>POST /api/generate</code> - Ollama-native generation</li>
        <li><code>GET /api-docs</code> - API Documentation</li>
    </ul>

    <h3>Quick Start</h3>
    <pre><code>curl https://ollama-fastapi-railway-deployment-qst2ba.fly.dev/v1/chat/completions \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "tinyllama:latest",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'</code></pre>

    <p><a href="/docs">Swagger UI</a> | <a href="/api-docs">API Guide</a></p>
</body>
</html>
"""

API_DOCS_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <title>API Documentation - Ollama Server</title>
    <style>
        body { font-family: -apple-system, sans-serif; line-height: 1.6; max-width: 900px; margin: 40px auto; padding: 0 20px; color: #24292e; }
        h1, h2, h3 { border-bottom: 1px solid #eaecef; padding-bottom: 0.3em; }
        pre { background: #f6f8fa; padding: 16px; border-radius: 6px; overflow: auto; }
        code { font-family: monospace; background: rgba(27,31,35,0.05); padding: 0.2em 0.4em; border-radius: 3px; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #dfe2e5; padding: 8px 12px; text-align: left; }
        th { background: #f6f8fa; }
    </style>
</head>
<body>
    <h1>API Documentation</h1>
    
    <h2>Authentication</h2>
    <p>All API requests must include a Bearer token in the <code>Authorization</code> header:</p>
    <pre><code>Authorization: Bearer ollama_...</code></pre>

    <h2>Chat Completions (OpenAI Compatible)</h2>
    <p><code>POST /v1/chat/completions</code></p>
    <table>
        <tr><th>Field</th><th>Type</th><th>Description</th></tr>
        <tr><td>model</td><td>string</td><td>Model name (e.g., tinyllama:latest)</td></tr>
        <tr><td>messages</td><td>array</td><td>List of message objects</td></tr>
        <tr><td>stream</td><td>boolean</td><td>Enable streaming (SSE)</td></tr>
    </table>

    <h2>Admin API</h2>
    <p>Admin endpoints require the <code>X-Master-Key</code> header.</p>
    <ul>
        <li><code>GET /admin/keys</code> - List all keys</li>
        <li><code>POST /admin/keys</code> - Create a new key</li>
        <li><code>DELETE /admin/keys/{hash}</code> - Revoke a key</li>
    </ul>
</body>
</html>
"""
