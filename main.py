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
from openai import AsyncAzureOpenAI, AzureOpenAI

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
DATABASE_URL = os.getenv("DATABASE_URL")

MODEL_TO_MACHINE_MAP = {
    "qwen2.5:0.5b": os.getenv("QWEN_MACHINE_URL", ""),
    "tinyllama:latest": os.getenv("TINYLLAMA_MACHINE_URL", ""),
    "llama3:latest": os.getenv("LLAMA3_MACHINE_URL", ""),
}

# =============================================================================
# AZURE OPENAI CONFIGURATION
# =============================================================================
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip().rstrip("/")
AZURE_OPENAI_KEY = (os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_KEY") or "").strip()
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview").strip()

AZURE_DEPLOYMENTS_STR = os.getenv("AZURE_OPENAI_DEPLOYMENTS", "gpt-4o-mini,gpt-35-turbo,gpt-4").strip()
AZURE_OPENAI_DEPLOYMENTS = []
if AZURE_DEPLOYMENTS_STR:
    AZURE_OPENAI_DEPLOYMENTS = [d.strip() for d in AZURE_DEPLOYMENTS_STR.split(",") if d.strip()]

EMBEDDING_PATTERNS = ["embedding", "embed", "text-embedding", "davinci-embed", "ada-embed"]
VISION_ONLY_PATTERNS = ["gpt-image", "dall-e", "image-generation"]

def is_chat_model(deployment_name: str) -> bool:
    name_lower = deployment_name.lower()
    for pattern in EMBEDDING_PATTERNS + VISION_ONLY_PATTERNS:
        if pattern in name_lower:
            return False
    return True

def is_embedding_model(deployment_name: str) -> bool:
    name_lower = deployment_name.lower()
    for pattern in EMBEDDING_PATTERNS:
        if pattern in name_lower:
            return True
    return False

USE_AZURE = False
azure_async_client = None
azure_chat_deployments = []
azure_embedding_deployments = []

if AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_KEY:
    try:
        # Sync client for startup validation only
        sync_client = AzureOpenAI(
            api_key=AZURE_OPENAI_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT
        )
        
        validated_chat = []
        validated_embed = []

        for dep_name in AZURE_OPENAI_DEPLOYMENTS:
            is_chat = is_chat_model(dep_name)
            is_embed = is_embedding_model(dep_name)

            try:
                if is_embed:
                    sync_client.embeddings.create(model=dep_name, input=["test"])
                    validated_embed.append(dep_name)
                    logger.info(f"Azure embedding deployment validated: {dep_name}")
                elif is_chat:
                    sync_client.chat.completions.create(
                        model=dep_name,
                        messages=[{"role": "user", "content": "hi"}],
                        max_tokens=1
                    )
                    validated_chat.append(dep_name)
                    logger.info(f"Azure chat deployment validated: {dep_name}")
            except Exception as e:
                logger.warning(f"Azure deployment test failed for {dep_name}: {e}")

        if validated_chat or validated_embed:
            USE_AZURE = True
            azure_chat_deployments = validated_chat
            azure_embedding_deployments = validated_embed
            # Async client for actual endpoint handling
            azure_async_client = AsyncAzureOpenAI(
                api_key=AZURE_OPENAI_KEY,
                api_version=AZURE_OPENAI_API_VERSION,
                azure_endpoint=AZURE_OPENAI_ENDPOINT
            )
            logger.info(f"Azure OpenAI ready. Chat: {validated_chat}, Embeddings: {validated_embed}")
    except Exception as e:
        logger.error(f"Failed to initialize Azure client: {e}")

# =============================================================================
# DATABASE SETUP
# =============================================================================
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
        engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_recycle=300)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        Base.metadata.create_all(bind=engine)
        logger.info("Database initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")

@app.on_event("startup")
async def startup_event():
    init_db()
    asyncio.create_task(ensure_models_loaded())
    asyncio.create_task(keep_warm_background())

async def ensure_models_loaded():
    await asyncio.sleep(60)
    models_to_ensure = ["qwen2.5:0.5b", "tinyllama:latest", "llama3:latest"]
    process_group = os.getenv("FLY_PROCESS_GROUP", "")
    if process_group == "qwen":
        primary = "qwen2.5:0.5b"
    elif process_group == "tinyllama":
        primary = "tinyllama:latest"
    elif process_group == "llama3":
        primary = "llama3:latest"
    else:
        primary = DEFAULT_MODEL
    
    # Always ensure primary model is loaded
    try:
        async with httpx.AsyncClient(timeout=300) as client:
            await client.post(f"{OLLAMA_HOST}/api/pull", json={"name": primary, "stream": False})
            logger.info(f"Ensured primary model is loaded: {primary}")
    except Exception as e:
        logger.warning(f"Could not ensure model {primary}: {e}")

async def keep_warm_background():
    await asyncio.sleep(30)
    process_group = os.getenv("FLY_PROCESS_GROUP", "")
    if process_group == "qwen":
        warm_model = "qwen2.5:0.5b"
    elif process_group == "tinyllama":
        warm_model = "tinyllama:latest"
    elif process_group == "llama3":
        warm_model = "llama3:latest"
    else:
        warm_model = DEFAULT_MODEL
    
    while True:
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                payload = {
                    "model": warm_model,
                    "messages": [{"role": "user", "content": "heartbeat"}],
                    "stream": False,
                    "options": {"num_predict": 1}
                }
                await client.post(f"{OLLAMA_HOST}/api/chat", json=payload)
                logger.info(f"Keep-warm heartbeat sent for {warm_model}")
        except Exception as e:
            logger.warning(f"Keep-warm heartbeat failed: {e}")
        await asyncio.sleep(120)

def get_db():
    if SessionLocal is None:
        raise HTTPException(status_code=503, detail="Database not initialized")
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# =============================================================================
# SECURITY
# =============================================================================
security = HTTPBearer(auto_error=False)

def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security), db = Depends(get_db)):
    if not credentials:
        raise HTTPException(status_code=401, detail="Missing Authorization header. Use: Bearer YOUR_API_KEY")
    token = credentials.credentials
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

# =============================================================================
# REQUEST MODELS
# =============================================================================
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

class EmbeddingRequest(BaseModel):
    model: Optional[str] = None
    input: List[str]

class PullModelRequest(BaseModel):
    name: str

class CreateKeyRequest(BaseModel):
    name: str

class RevokeKeyRequest(BaseModel):
    key_hash: str

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def messages_to_prompt(messages: List[ChatMessage]) -> str:
    prompt_parts = []
    for msg in messages:
        prompt_parts.append(f"{msg.role.capitalize()}: {msg.content}")
    prompt_parts.append("Assistant:")
    return "\n\n".join(prompt_parts)

async def ensure_model_present(model: str):
    """Checks if a model is present locally, if not pulls it."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(f"{OLLAMA_HOST}/api/tags")
            if r.status_code == 200:
                models = [m.get("name") for m in r.json().get("models", [])]
                if model in models:
                    return True
        
        logger.info(f"Model {model} not found locally. Pulling...")
        async with httpx.AsyncClient(timeout=300) as client:
            await client.post(f"{OLLAMA_HOST}/api/pull", json={"name": model, "stream": False})
            return True
    except Exception as e:
        logger.warning(f"Failed to ensure model {model}: {e}")
        return False

async def get_ollama_base_url(model: str):
    """Returns the local Ollama host or a peer machine URL if configured."""
    # Check if model is handled by another machine
    peer_url = MODEL_TO_MACHINE_MAP.get(model)
    if peer_url:
        # If it's not the current machine's primary model, proxy it
        process_group = os.getenv("FLY_PROCESS_GROUP", "")
        is_primary = False
        if process_group == "qwen" and model == "qwen2.5:0.5b": is_primary = True
        elif process_group == "tinyllama" and model == "tinyllama:latest": is_primary = True
        elif process_group == "llama3" and model == "llama3:latest": is_primary = True
        
        if not is_primary:
            logger.info(f"Routing request for {model} to peer: {peer_url}")
            return peer_url
            
    return OLLAMA_HOST

async def ollama_chat_generate(model: str, messages: List[ChatMessage], stream: bool, temperature: float, max_tokens: int):
    # Ensure model is available (pull if missing)
    await ensure_model_present(model)
    
    # Determine which host to use (local or peer)
    target_host = await get_ollama_base_url(model)
    
    payload = {
        "model": model,
        "messages": [{"role": m.role, "content": m.content} for m in messages],
        "stream": stream,
        "options": {"temperature": temperature, "num_predict": max_tokens}
    }

    if stream:
        async def generator():
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("POST", f"{target_host}/api/chat", json=payload) as response:
                    if response.status_code != 200:
                        error_text = await response.aread()
                        yield f"data: {json.dumps({'error': error_text.decode()})}\n\n"
                        yield "data: [DONE]\n\n"
                        return
                    
                    async for line in response.aiter_lines():
                        if not line: continue
                        try:
                            data = json.loads(line)
                            chunk = {
                                "id": f"ollama-{int(time.time())}",
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {"content": data.get("message", {}).get("content", "")},
                                    "finish_reason": data.get("done_reason") if data.get("done") else None
                                }]
                            }
                            yield f"data: {json.dumps(chunk)}\n\n"
                            if data.get("done"):
                                yield "data: [DONE]\n\n"
                        except: continue
        return generator()
    else:
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(f"{target_host}/api/chat", json=payload)
            if r.status_code != 200:
                # Fallback to /api/generate
                prompt = messages_to_prompt(messages)
                gen_payload = {
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": temperature, "num_predict": max_tokens}
                }
                r = await client.post(f"{target_host}/api/generate", json=gen_payload)
                if r.status_code != 200:
                    raise HTTPException(status_code=r.status_code, detail=r.text)
                gen_data = r.json()
                content = gen_data.get("response", "")
                prompt_tokens = 0
                completion_tokens = gen_data.get("eval_count", 0)
            else:
                ollama_data = r.json()
                content = ollama_data.get("message", {}).get("content", "")
                prompt_tokens = ollama_data.get("prompt_eval_count", 0)
                completion_tokens = ollama_data.get("eval_count", 0)
            
            return {
                "id": f"ollama-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                }
            }

# =============================================================================
# ENDPOINTS
# =============================================================================
@app.get("/health")
async def health():
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
            if r.status_code == 200:
                return {
                    "status": "ok",
                    "ollama": "connected",
                    "auth": "enabled",
                    "azure": USE_AZURE,
                    "azure_chat_deployments": azure_chat_deployments,
                    "azure_embedding_deployments": azure_embedding_deployments
                }
    except: pass
    return {
        "status": "degraded",
        "ollama": "not ready",
        "auth": "enabled",
        "azure": USE_AZURE,
        "azure_chat_deployments": azure_chat_deployments,
        "azure_embedding_deployments": azure_embedding_deployments
    }

@app.get("/ping")
async def ping():
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{OLLAMA_HOST}/api/tags")
            if r.status_code == 200:
                return {"status": "alive", "timestamp": int(time.time()), "azure": USE_AZURE}
    except: pass
    return {"status": "starting", "timestamp": int(time.time()), "azure": USE_AZURE}

@app.post("/warmup")
async def warmup():
    process_group = os.getenv("FLY_PROCESS_GROUP", "")
    if process_group == "qwen":
        warm_model = "qwen2.5:0.5b"
    elif process_group == "tinyllama":
        warm_model = "tinyllama:latest"
    elif process_group == "llama3":
        warm_model = "llama3:latest"
    else:
        warm_model = DEFAULT_MODEL

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            payload = {
                "model": warm_model,
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": False,
                "options": {"num_predict": 1}
            }
            await client.post(f"{OLLAMA_HOST}/api/chat", json=payload)
            return {"status": "warm", "model": warm_model, "timestamp": int(time.time())}
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
async def v1_models(token: str = Depends(verify_api_key)):
    models = []
    # Add Azure chat models
    if USE_AZURE:
        for deployment in azure_chat_deployments:
            models.append({"id": deployment, "object": "model", "created": int(time.time()), "owned_by": "azure-openai"})
    
    # Add Ollama models
    configured_ollama_models = ["qwen2.5:0.5b", "tinyllama:latest", "llama3:latest"]
    present_model_names = []
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
            if r.status_code == 200:
                data = r.json()
                present_model_names = [m.get("name") for m in data.get("models", []) if m.get("name")]
    except Exception as e:
        logger.error(f"Ollama tags error: {e}")

    all_ollama_models = list(set(configured_ollama_models + present_model_names))
    for model_name in all_ollama_models:
        models.append({"id": model_name, "object": "model", "created": int(time.time()), "owned_by": "ollama"})
    
    return {"object": "list", "data": models}

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest, api_key: str = Depends(verify_api_key)):
    model = req.model or DEFAULT_MODEL
    
    if USE_AZURE and model in azure_chat_deployments:
        try:
            messages = [{"role": m.role, "content": m.content} for m in req.messages]
            if req.stream:
                response = await azure_async_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=req.temperature,
                    max_tokens=req.max_tokens,
                    stream=True
                )
                async def azure_streamer():
                    try:
                        async for chunk in response:
                            if chunk.choices:
                                yield f"data: {json.dumps(chunk.model_dump())}\n\n"
                        yield "data: [DONE]\n\n"
                    except Exception as e:
                        logger.error(f"Azure streaming error: {e}")
                        yield f"data: {json.dumps({'error': str(e)})}\n\n"
                        yield "data: [DONE]\n\n"
                return StreamingResponse(azure_streamer(), media_type="text/event-stream")
            else:
                response = await azure_async_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=req.temperature,
                    max_tokens=req.max_tokens,
                    stream=False
                )
                return response.model_dump()
        except Exception as e:
            logger.error(f"Azure OpenAI error: {e}. Falling back to Ollama.")

    # Ollama path
    if req.stream:
        gen = await ollama_chat_generate(model, req.messages, True, req.temperature, req.max_tokens)
        return StreamingResponse(gen, media_type="text/event-stream")
    else:
        result = await ollama_chat_generate(model, req.messages, False, req.temperature, req.max_tokens)
        return result

@app.post("/v1/embeddings")
async def create_embeddings(req: EmbeddingRequest, api_key: str = Depends(verify_api_key)):
    model = req.model or (azure_embedding_deployments[0] if azure_embedding_deployments else DEFAULT_MODEL)
    
    if USE_AZURE and model in azure_embedding_deployments:
        try:
            response = await azure_async_client.embeddings.create(model=model, input=req.input)
            return response.model_dump()
        except Exception as e:
            logger.error(f"Azure embedding error: {e}")

    # Ollama fallback
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(f"{OLLAMA_HOST}/api/embed", json={"model": model, "input": req.input})
            if r.status_code != 200:
                raise HTTPException(status_code=r.status_code, detail=f"Ollama error: {r.text}")
            data = r.json()
            embeddings = [{"object": "embedding", "embedding": emb, "index": i} for i, emb in enumerate(data.get("embeddings", []))]
            return {
                "object": "list",
                "data": embeddings,
                "model": model,
                "usage": {"prompt_tokens": 0, "total_tokens": 0}
            }
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate")
async def generate(req: GenerateRequest, api_key: str = Depends(verify_api_key)):
    model = req.model or DEFAULT_MODEL
    payload = {
        "model": model,
        "prompt": req.prompt,
        "stream": req.stream,
        "options": {"temperature": req.temperature, "num_predict": req.max_tokens}
    }
    if req.stream:
        async def streamer():
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("POST", f"{OLLAMA_HOST}/api/generate", json=payload) as response:
                    async for line in response.aiter_lines():
                        if line: yield line + "\n"
        return StreamingResponse(streamer(), media_type="application/x-ndjson")
    else:
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(f"{OLLAMA_HOST}/api/generate", json=payload)
            return r.json()

@app.get("/api/models")
async def ollama_models_list(api_key: str = Depends(verify_api_key)):
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{OLLAMA_HOST}/api/tags")
        return r.json()

@app.post("/api/pull")
async def pull_model(req: PullModelRequest, api_key: str = Depends(verify_api_key)):
    async def streamer():
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", f"{OLLAMA_HOST}/api/pull", json={"name": req.name, "stream": True}) as response:
                async for line in response.aiter_lines():
                    if line: yield line + "\n"
    return StreamingResponse(streamer(), media_type="application/x-ndjson")

# =============================================================================
# ADMIN
# =============================================================================
@app.post("/admin/keys")
async def create_key(req: CreateKeyRequest, master: str = Depends(verify_master_key), db = Depends(get_db)):
    raw_key = "ollama_" + secrets.token_urlsafe(32)
    key_hash = hash_key(raw_key)
    new_key = APIKey(key_hash=key_hash, name=req.name, created_at=int(time.time()))
    db.add(new_key)
    db.commit()
    return {"key": raw_key, "name": req.name, "key_hash": key_hash}

@app.get("/admin/keys")
async def list_keys(master: str = Depends(verify_master_key), db = Depends(get_db)):
    keys = db.query(APIKey).all()
    return [{"name": k.name, "key_hash": k.key_hash, "created_at": k.created_at} for k in keys]

@app.post("/admin/keys/revoke")
async def revoke_key(req: RevokeKeyRequest, master: str = Depends(verify_master_key), db = Depends(get_db)):
    key_record = db.query(APIKey).filter(APIKey.key_hash == req.key_hash).first()
    if key_record:
        db.delete(key_record)
        db.commit()
        return {"status": "revoked"}
    raise HTTPException(status_code=404, detail="Key not found")

# =============================================================================
# UI CONTENT
# =============================================================================
HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ollama API Server</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        body { background-color: #0f172a; color: #e2e8f0; font-family: 'Inter', sans-serif; }
        .glass { background: rgba(30, 41, 59, 0.7); backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.1); }
        .glow-green { box-shadow: 0 0 15px rgba(34, 197, 94, 0.3); }
        .glow-blue { box-shadow: 0 0 15px rgba(59, 130, 246, 0.3); }
        pre { background: #1e293b; padding: 1rem; border-radius: 0.5rem; overflow-x: auto; }
        .status-dot { height: 10px; width: 10px; border-radius: 50%; display: inline-block; }
        .dot-online { background-color: #22c55e; }
        .dot-offline { background-color: #ef4444; }
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: #0f172a; }
        ::-webkit-scrollbar-thumb { background: #334155; border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: #475569; }
    </style>
</head>
<body class="min-h-screen p-4 md:p-8">
    <div class="max-w-6xl mx-auto space-y-8">
        <!-- Header -->
        <header class="flex flex-col md:flex-row justify-between items-center gap-4 glass p-6 rounded-2xl glow-blue">
            <div class="flex items-center gap-4">
                <div class="bg-blue-600 p-3 rounded-xl">
                    <i class="fas fa-robot text-2xl"></i>
                </div>
                <div>
                    <h1 class="text-2xl font-bold">Ollama API Server</h1>
                    <p class="text-slate-400 text-sm">Enterprise-grade LLM Gateway</p>
                </div>
            </div>
            <div class="flex items-center gap-6">
                <div class="flex items-center gap-2 glass px-4 py-2 rounded-full">
                    <span id="ollama-status-dot" class="status-dot dot-offline"></span>
                    <span id="ollama-status-text" class="text-sm font-medium">Ollama: Checking...</span>
                </div>
                <a href="/docs" target="_blank" class="bg-slate-700 hover:bg-slate-600 px-4 py-2 rounded-lg text-sm transition-colors">
                    <i class="fas fa-book mr-2"></i>API Docs
                </a>
            </div>
        </header>

        <div class="grid grid-cols-1 lg:grid-cols-3 gap-8">
            <!-- Left Column: Models & Keys -->
            <div class="lg:col-span-1 space-y-8">
                <!-- API Key Section -->
                <section class="glass p-6 rounded-2xl space-y-4">
                    <h2 class="text-lg font-semibold flex items-center gap-2">
                        <i class="fas fa-key text-yellow-500"></i> Authentication
                    </h2>
                    <div class="space-y-2">
                        <label class="text-xs text-slate-400 uppercase tracking-wider">Your API Key</label>
                        <div class="relative">
                            <input type="password" id="api-key-input" placeholder="Enter Bearer Token" 
                                class="w-full bg-slate-900 border border-slate-700 rounded-lg px-4 py-2 focus:ring-2 focus:ring-blue-500 outline-none transition-all">
                            <button onclick="toggleKeyVisibility()" class="absolute right-3 top-2 text-slate-500 hover:text-slate-300">
                                <i class="fas fa-eye" id="key-toggle-icon"></i>
                            </button>
                        </div>
                        <p class="text-[10px] text-slate-500 italic">Keys are stored locally in your browser for this session.</p>
                    </div>
                </section>

                <!-- Model List Section -->
                <section class="glass p-6 rounded-2xl space-y-4">
                    <div class="flex justify-between items-center">
                        <h2 class="text-lg font-semibold flex items-center gap-2">
                            <i class="fas fa-layer-group text-blue-500"></i> Available Models
                        </h2>
                        <button onclick="refreshModels()" class="text-slate-400 hover:text-white transition-colors">
                            <i class="fas fa-sync-alt" id="refresh-icon"></i>
                        </button>
                    </div>
                    <div id="model-list" class="space-y-2 max-h-[400px] overflow-y-auto pr-2">
                        <div class="animate-pulse flex space-x-4">
                            <div class="flex-1 space-y-4 py-1">
                                <div class="h-4 bg-slate-700 rounded w-3/4"></div>
                                <div class="h-4 bg-slate-700 rounded"></div>
                            </div>
                        </div>
                    </div>
                </section>
            </div>

            <!-- Right Column: Chat & Console -->
            <div class="lg:col-span-2 space-y-8">
                <!-- Live Chat Box -->
                <section class="glass rounded-2xl flex flex-col h-[600px] overflow-hidden glow-green">
                    <div class="p-4 border-b border-slate-700 flex justify-between items-center bg-slate-800/50">
                        <div class="flex items-center gap-3">
                            <i class="fas fa-comments text-green-500"></i>
                            <span class="font-medium">Live Inference Console</span>
                        </div>
                        <div id="selected-model-badge" class="text-xs bg-blue-600/30 text-blue-400 px-2 py-1 rounded border border-blue-500/30">
                            No model selected
                        </div>
                    </div>
                    
                    <div id="chat-messages" class="flex-1 overflow-y-auto p-6 space-y-4 bg-slate-900/30">
                        <div class="flex gap-4">
                            <div class="h-8 w-8 rounded-full bg-blue-600 flex items-center justify-center flex-shrink-0">
                                <i class="fas fa-robot text-xs"></i>
                            </div>
                            <div class="bg-slate-800 p-4 rounded-2xl rounded-tl-none max-w-[80%]">
                                <p class="text-sm">Hello! Select a model and enter your API key to start testing the server.</p>
                            </div>
                        </div>
                    </div>

                    <div class="p-4 bg-slate-800/50 border-t border-slate-700">
                        <form id="chat-form" class="flex gap-2">
                            <input type="text" id="chat-input" placeholder="Type your prompt here..." 
                                class="flex-1 bg-slate-900 border border-slate-700 rounded-xl px-4 py-3 focus:ring-2 focus:ring-green-500 outline-none transition-all">
                            <button type="submit" class="bg-green-600 hover:bg-green-500 px-6 py-3 rounded-xl font-medium transition-all flex items-center gap-2">
                                <span>Send</span>
                                <i class="fas fa-paper-plane"></i>
                            </button>
                        </form>
                    </div>
                </section>

                <!-- Code Examples -->
                <section class="glass p-6 rounded-2xl space-y-4">
                    <h2 class="text-lg font-semibold flex items-center gap-2">
                        <i class="fas fa-code text-purple-500"></i> Implementation
                    </h2>
                    <div class="flex gap-4 border-b border-slate-700 mb-4">
                        <button onclick="switchTab('curl')" class="pb-2 border-b-2 border-blue-500 px-2 text-sm font-medium" id="tab-curl">cURL</button>
                        <button onclick="switchTab('python')" class="pb-2 border-transparent border-b-2 hover:border-slate-500 px-2 text-sm font-medium text-slate-400" id="tab-python">Python</button>
                    </div>
                    <div id="code-curl" class="block">
                        <pre><code class="text-blue-300">curl -X POST http://this-server/v1/chat/completions \\
  -H "Authorization: Bearer YOUR_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{
    "model": "qwen2.5:0.5b",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'</code></pre>
                    </div>
                    <div id="code-python" class="hidden">
                        <pre><code class="text-green-300">import openai

client = openai.OpenAI(
    base_url="http://this-server/v1",
    api_key="YOUR_API_KEY"
)

response = client.chat.completions.create(
    model="qwen2.5:0.5b",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)</code></pre>
                    </div>
                </section>
            </div>
        </div>

        <footer class="text-center text-slate-500 text-sm py-8 border-t border-slate-800">
            <p>Powered by Ollama, FastAPI, and Railway/Fly.io Deployment</p>
        </footer>
    </div>

    <script>
        let selectedModel = "";
        const messagesContainer = document.getElementById('chat-messages');
        const chatForm = document.getElementById('chat-form');
        const chatInput = document.getElementById('chat-input');
        const apiKeyInput = document.getElementById('api-key-input');

        // Load API key from local storage
        if (localStorage.getItem('ollama_api_key')) {
            apiKeyInput.value = localStorage.getItem('ollama_api_key');
        }

        apiKeyInput.addEventListener('input', (e) => {
            localStorage.setItem('ollama_api_key', e.target.value);
        });

        function toggleKeyVisibility() {
            const icon = document.getElementById('key-toggle-icon');
            if (apiKeyInput.type === 'password') {
                apiKeyInput.type = 'text';
                icon.classList.replace('fa-eye', 'fa-eye-slash');
            } else {
                apiKeyInput.type = 'password';
                icon.classList.replace('fa-eye-slash', 'fa-eye');
            }
        }

        async function checkStatus() {
            try {
                const res = await fetch('/health');
                const data = await res.json();
                const dot = document.getElementById('ollama-status-dot');
                const text = document.getElementById('ollama-status-text');
                
                if (data.status === 'ok') {
                    dot.classList.replace('dot-offline', 'dot-online');
                    text.innerText = 'Ollama: Online';
                } else {
                    dot.classList.replace('dot-online', 'dot-offline');
                    text.innerText = 'Ollama: ' + (data.ollama || 'Disconnected');
                }
            } catch (e) {
                console.error('Status check failed', e);
            }
        }

        async function refreshModels() {
            const icon = document.getElementById('refresh-icon');
            icon.classList.add('fa-spin');
            
            const list = document.getElementById('model-list');
            const apiKey = apiKeyInput.value;
            
            if (!apiKey) {
                list.innerHTML = '<p class="text-xs text-yellow-500 p-2">Enter API key to see models</p>';
                icon.classList.remove('fa-spin');
                return;
            }

            try {
                const res = await fetch('/v1/models', {
                    headers: { 'Authorization': `Bearer ${apiKey}` }
                });
                
                if (!res.ok) throw new Error('Auth failed');
                
                const data = await res.json();
                list.innerHTML = '';
                
                data.data.forEach(model => {
                    const div = document.createElement('div');
                    const isAzure = model.owned_by === 'azure-openai';
                    div.className = `p-3 rounded-xl border border-slate-700 cursor-pointer transition-all hover:bg-slate-800 flex justify-between items-center ${selectedModel === model.id ? 'bg-blue-600/20 border-blue-500' : ''}`;
                    div.onclick = () => selectModel(model.id);
                    
                    div.innerHTML = `
                        <div class="flex flex-col">
                            <span class="text-sm font-medium">${model.id}</span>
                            <span class="text-[10px] text-slate-500 uppercase">${model.owned_by}</span>
                        </div>
                        ${isAzure ? '<i class="fas fa-cloud text-blue-400 text-xs"></i>' : '<i class="fas fa-microchip text-slate-500 text-xs"></i>'}
                    `;
                    list.appendChild(div);
                });

                if (!selectedModel && data.data.length > 0) {
                    selectModel(data.data[0].id);
                }
            } catch (e) {
                list.innerHTML = '<p class="text-xs text-red-500 p-2">Failed to load models. Check API key.</p>';
            } finally {
                icon.classList.remove('fa-spin');
            }
        }

        function selectModel(id) {
            selectedModel = id;
            document.getElementById('selected-model-badge').innerText = id;
            refreshModels(); // Update visual selection
        }

        function appendMessage(role, content) {
            const div = document.createElement('div');
            div.className = 'flex gap-4 ' + (role === 'user' ? 'flex-row-reverse' : '');
            
            const icon = role === 'user' ? 'fa-user' : 'fa-robot';
            const color = role === 'user' ? 'bg-green-600' : 'bg-blue-600';
            const rounded = role === 'user' ? 'rounded-tr-none' : 'rounded-tl-none';
            
            div.innerHTML = `
                <div class="h-8 w-8 rounded-full ${color} flex items-center justify-center flex-shrink-0">
                    <i class="fas ${icon} text-xs"></i>
                </div>
                <div class="bg-slate-800 p-4 rounded-2xl ${rounded} max-w-[80%]">
                    <p class="text-sm whitespace-pre-wrap">${content}</p>
                </div>
            `;
            messagesContainer.appendChild(div);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
            return div.querySelector('p');
        }

        chatForm.onsubmit = async (e) => {
            e.preventDefault();
            const text = chatInput.value.trim();
            const apiKey = apiKeyInput.value;
            
            if (!text || !selectedModel || !apiKey) return;
            
            chatInput.value = '';
            appendMessage('user', text);
            
            const responseEl = appendMessage('assistant', '...');
            let fullResponse = "";

            try {
                const res = await fetch('/v1/chat/completions', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': `Bearer ${apiKey}`
                    },
                    body: JSON.stringify({
                        model: selectedModel,
                        messages: [{ role: 'user', content: text }],
                        stream: true
                    })
                });

                if (!res.ok) {
                    const err = await res.json();
                    responseEl.innerText = "Error: " + (err.detail || err.error?.message || "Request failed");
                    return;
                }

                const reader = res.body.getReader();
                const decoder = new TextDecoder();
                responseEl.innerText = "";

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    
                    const chunk = decoder.decode(value);
                    const lines = chunk.split('\\n');
                    
                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            const dataStr = line.slice(6);
                            if (dataStr === '[DONE]') continue;
                            try {
                                const data = JSON.parse(dataStr);
                                const content = data.choices[0].delta.content || "";
                                fullResponse += content;
                                responseEl.innerText = fullResponse;
                                messagesContainer.scrollTop = messagesContainer.scrollHeight;
                            } catch (e) {}
                        }
                    }
                }
            } catch (e) {
                responseEl.innerText = "Connection error: " + e.message;
            }
        };

        function switchTab(tab) {
            document.getElementById('code-curl').className = tab === 'curl' ? 'block' : 'hidden';
            document.getElementById('code-python').className = tab === 'python' ? 'block' : 'hidden';
            
            document.getElementById('tab-curl').className = tab === 'curl' ? 'pb-2 border-b-2 border-blue-500 px-2 text-sm font-medium' : 'pb-2 border-transparent border-b-2 hover:border-slate-500 px-2 text-sm font-medium text-slate-400';
            document.getElementById('tab-python').className = tab === 'python' ? 'pb-2 border-b-2 border-blue-500 px-2 text-sm font-medium' : 'pb-2 border-transparent border-b-2 hover:border-slate-500 px-2 text-sm font-medium text-slate-400';
        }

        // Init
        checkStatus();
        setInterval(checkStatus, 15000);
        setTimeout(refreshModels, 1000);
    </script>
</body>
</html>
"""

API_DOCS_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>API Documentation - Ollama Server</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body { background-color: #0f172a; color: #e2e8f0; }
        .glass { background: rgba(30, 41, 59, 0.7); backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.1); }
        code { background: #1e293b; padding: 0.2rem 0.4rem; border-radius: 0.25rem; color: #38bdf8; }
        pre { background: #1e293b; padding: 1rem; border-radius: 0.5rem; overflow-x: auto; margin: 1rem 0; border: 1px solid #334155; }
    </style>
</head>
<body class="p-8">
    <div class="max-w-4xl mx-auto space-y-8">
        <header class="glass p-8 rounded-2xl">
            <h1 class="text-3xl font-bold mb-2">API Documentation</h1>
            <p class="text-slate-400">Reference for OpenAI-compatible and native Ollama endpoints.</p>
        </header>

        <section class="glass p-8 rounded-2xl space-y-6">
            <h2 class="text-xl font-semibold border-b border-slate-700 pb-2">Authentication</h2>
            <p>All requests (except health/ping) require a Bearer token in the <code>Authorization</code> header.</p>
            <pre>Authorization: Bearer YOUR_API_KEY</pre>
        </section>

        <section class="glass p-8 rounded-2xl space-y-6">
            <h2 class="text-xl font-semibold border-b border-slate-700 pb-2">Endpoints</h2>
            
            <div class="space-y-4">
                <div class="border-l-4 border-blue-500 pl-4">
                    <h3 class="font-mono font-bold text-blue-400">POST /v1/chat/completions</h3>
                    <p class="text-sm text-slate-400">OpenAI-compatible chat completion. Routes to Azure if model matches deployment name.</p>
                </div>

                <div class="border-l-4 border-green-500 pl-4">
                    <h3 class="font-mono font-bold text-green-400">GET /v1/models</h3>
                    <p class="text-sm text-slate-400">Lists all available Ollama models and Azure deployments.</p>
                </div>

                <div class="border-l-4 border-purple-500 pl-4">
                    <h3 class="font-mono font-bold text-purple-400">POST /admin/keys</h3>
                    <p class="text-sm text-slate-400 text-yellow-500/80 italic">Requires X-Master-Key header.</p>
                    <pre>curl -X POST http://server/admin/keys \\
  -H "X-Master-Key: YOUR_MASTER_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{"name": "New Project Key"}'</pre>
                </div>
            </div>
        </section>
    </div>
</body>
</html>
"""

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
