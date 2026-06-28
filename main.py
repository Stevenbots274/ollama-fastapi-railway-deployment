import os
import time
import secrets
import hashlib
import httpx
import asyncio
import json
import logging
from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Union
from sqlalchemy import create_engine, Column, String, Integer, BigInteger, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import OperationalError
from openai import AsyncAzureOpenAI

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Ollama API Server",
    description="Self-hosted LLM API with Ollama + FastAPI. API key protected.",
    version="2.2.0",
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
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "tinyllama:latest")
MASTER_KEY = os.getenv("MASTER_KEY", "ollama-master-key-change-me")
DATABASE_URL = os.getenv("DATABASE_URL")

# =============================================================================
# AZURE OPENAI CONFIGURATION
# =============================================================================
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip().rstrip("/")
AZURE_OPENAI_KEY = (os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_KEY") or "").strip()
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview").strip()

AZURE_DEPLOYMENTS_STR = os.getenv("AZURE_OPENAI_DEPLOYMENTS", "").strip()
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
        azure_async_client = AsyncAzureOpenAI(
            api_key=AZURE_OPENAI_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            azure_endpoint=AZURE_OPENAI_ENDPOINT
        )
        for dep_name in AZURE_OPENAI_DEPLOYMENTS:
            if is_embedding_model(dep_name):
                azure_embedding_deployments.append(dep_name)
            elif is_chat_model(dep_name):
                azure_chat_deployments.append(dep_name)
        if azure_chat_deployments or azure_embedding_deployments:
            USE_AZURE = True
            logger.info(f"Azure OpenAI configured. Chat: {azure_chat_deployments}, Embeddings: {azure_embedding_deployments}")
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
        logger.warning("DATABASE_URL not set. Using in-memory fallback for API keys.")
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
    asyncio.create_task(keep_warm_background())

async def keep_warm_background():
    await asyncio.sleep(30)
    while True:
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                payload = {
                    "model": DEFAULT_MODEL,
                    "messages": [{"role": "user", "content": "heartbeat"}],
                    "stream": False,
                    "options": {"num_predict": 1}
                }
                await client.post(f"{OLLAMA_HOST}/api/chat", json=payload)
                logger.info(f"Keep-warm heartbeat sent for {DEFAULT_MODEL}")
        except Exception as e:
            logger.warning(f"Keep-warm heartbeat failed: {e}")
        await asyncio.sleep(120)

def get_db():
    if SessionLocal is None:
        raise HTTPException(status_code=503, detail="Database not initialized. Set DATABASE_URL.")
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# =============================================================================
# IN-MEMORY FALLBACK FOR API KEYS (when DB is not available)
# =============================================================================
_in_memory_keys = {}

def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()

security = HTTPBearer()

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not credentials:
        raise HTTPException(status_code=401, detail="Missing Authorization header. Use: Bearer YOUR_API_KEY")
    token = credentials.credentials
    if token == MASTER_KEY:
        return token
    key_hash = hash_key(token)

    # Try database first
    if SessionLocal is not None:
        try:
            db = SessionLocal()
            key_record = db.query(APIKey).filter(APIKey.key_hash == key_hash).first()
            db.close()
            if key_record:
                return token
        except Exception as e:
            logger.warning(f"DB lookup failed, falling back to memory: {e}")

    # Fallback to in-memory
    if key_hash in _in_memory_keys:
        return token

    raise HTTPException(status_code=401, detail="Invalid API key")

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
    input: Union[str, List[str]]

class PullModelRequest(BaseModel):
    name: str

class CreateKeyRequest(BaseModel):
    name: str

class RevokeKeyRequest(BaseModel):
    key_hash: Optional[str] = None
    key_name: Optional[str] = None

# =============================================================================
# OLLAMA MODEL AVAILABILITY CACHE
# =============================================================================
_ollama_models_cache = []
_ollama_cache_time = 0

async def get_available_ollama_models() -> List[str]:
    global _ollama_models_cache, _ollama_cache_time
    if time.time() - _ollama_cache_time < 30 and _ollama_models_cache:
        return _ollama_models_cache

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(f"{OLLAMA_HOST}/api/tags")
            if r.status_code == 200:
                data = r.json()
                models = [m.get("name") for m in data.get("models", []) if m.get("name")]
                _ollama_models_cache = models
                _ollama_cache_time = time.time()
                return models
    except Exception as e:
        logger.warning(f"Failed to fetch Ollama models: {e}")

    return _ollama_models_cache or []

def is_model_available(model_name: str) -> bool:
    """Check if a model is available (either Azure or Ollama)."""
    if USE_AZURE and model_name in azure_chat_deployments:
        return True
    if model_name in _ollama_models_cache:
        return True
    return False

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def messages_to_prompt(messages: List[ChatMessage]) -> str:
    prompt_parts = []
    for msg in messages:
        prompt_parts.append(f"{msg.role.capitalize()}: {msg.content}")
    prompt_parts.append("Assistant:")
    return "\n\n".join(prompt_parts)

async def ollama_chat_generate(model: str, messages: List[ChatMessage], stream: bool, temperature: float, max_tokens: int):
    payload = {
        "model": model,
        "messages": [{"role": m.role, "content": m.content} for m in messages],
        "stream": stream,
        "options": {"temperature": temperature, "num_predict": max_tokens}
    }

    if stream:
        async def generator():
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("POST", f"{OLLAMA_HOST}/api/chat", json=payload) as response:
                    if response.status_code != 200:
                        error_text = await response.aread()
                        try:
                            err_json = json.loads(error_text)
                            err_msg = err_json.get("error", error_text.decode())
                        except:
                            err_msg = error_text.decode()
                        yield f"data: {json.dumps({'error': err_msg})}\n\n"
                        yield "data: [DONE]\n\n"
                        return

                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            content = ""
                            if "message" in data and isinstance(data["message"], dict):
                                content = data["message"].get("content", "")
                            elif "response" in data:
                                content = data["response"]

                            chunk = {
                                "id": f"ollama-{int(time.time())}",
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {"content": content},
                                    "finish_reason": data.get("done_reason") if data.get("done") else None
                                }]
                            }
                            yield f"data: {json.dumps(chunk)}\n\n"
                            if data.get("done"):
                                yield "data: [DONE]\n\n"
                        except Exception as e:
                            logger.debug(f"Stream parse error: {e}")
                            continue

        return generator()
    else:
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(f"{OLLAMA_HOST}/api/chat", json=payload)

            if r.status_code != 200:
                # Try fallback to /api/generate
                error_detail = r.text
                try:
                    err_json = r.json()
                    error_detail = err_json.get("error", r.text)
                except:
                    pass

                prompt = messages_to_prompt(messages)
                gen_payload = {
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": temperature, "num_predict": max_tokens}
                }
                r = await client.post(f"{OLLAMA_HOST}/api/generate", json=gen_payload)
                if r.status_code != 200:
                    raise HTTPException(status_code=r.status_code, detail=error_detail)
                gen_data = r.json()
                content = gen_data.get("response", "")
                prompt_tokens = 0
                completion_tokens = gen_data.get("eval_count", 0)
            else:
                ollama_data = r.json()
                content = ""
                if "message" in ollama_data and isinstance(ollama_data["message"], dict):
                    content = ollama_data["message"].get("content", "")
                elif "response" in ollama_data:
                    content = ollama_data["response"]
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
    ollama_status = "not_ready"
    ollama_models = []
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
            if r.status_code == 200:
                ollama_status = "connected"
                data = r.json()
                ollama_models = [m.get("name") for m in data.get("models", []) if m.get("name")]
    except Exception as e:
        logger.warning(f"Health check Ollama error: {e}")

    return {
        "status": "ok",
        "ollama": ollama_status,
        "auth": "enabled",
        "azure": USE_AZURE,
        "azure_chat_deployments": azure_chat_deployments,
        "azure_embedding_deployments": azure_embedding_deployments,
        "available_ollama_models": ollama_models,
        "default_model": DEFAULT_MODEL
    }

@app.get("/ping")
async def ping():
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{OLLAMA_HOST}/api/tags")
            if r.status_code == 200:
                return {"status": "alive", "timestamp": int(time.time()), "azure": USE_AZURE}
    except Exception as e:
        logger.warning(f"Ping error: {e}")
    return {"status": "starting", "timestamp": int(time.time()), "azure": USE_AZURE}

@app.post("/warmup")
async def warmup():
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            payload = {
                "model": DEFAULT_MODEL,
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": False,
                "options": {"num_predict": 1}
            }
            r = await client.post(f"{OLLAMA_HOST}/api/chat", json=payload)
            if r.status_code == 200:
                return {"status": "warm", "model": DEFAULT_MODEL, "timestamp": int(time.time())}
            else:
                return {"status": "error", "detail": f"Ollama returned {r.status_code}: {r.text[:200]}", "model": DEFAULT_MODEL}
    except Exception as e:
        return {"status": "error", "detail": str(e), "model": DEFAULT_MODEL}

@app.get("/", response_class=HTMLResponse)
@app.get("/ui", response_class=HTMLResponse)
@app.get("/UI", response_class=HTMLResponse)
async def web_ui():
    return HTMLResponse(content=HTML_CONTENT)

@app.get("/api-docs", response_class=HTMLResponse)
async def api_docs():
    return HTMLResponse(content=API_DOCS_CONTENT)

@app.get("/v1/models")
async def v1_models(token: str = Depends(verify_api_key)):
    models = []

    # Add Azure chat models
    if USE_AZURE:
        for deployment in azure_chat_deployments:
            models.append({"id": deployment, "object": "model", "created": int(time.time()), "owned_by": "azure-openai"})

    # Add Ollama models - ONLY actually available ones
    available_models = await get_available_ollama_models()
    for model_name in available_models:
        models.append({"id": model_name, "object": "model", "created": int(time.time()), "owned_by": "ollama"})

    return {"object": "list", "data": models}

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest, api_key: str = Depends(verify_api_key)):
    model = req.model or DEFAULT_MODEL

    # Check Azure first
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
            logger.error(f"Azure OpenAI error: {e}")
            raise HTTPException(status_code=502, detail=f"Azure error: {str(e)}")

    # Ollama path
    try:
        if req.stream:
            gen = await ollama_chat_generate(model, req.messages, True, req.temperature, req.max_tokens)
            return StreamingResponse(gen, media_type="text/event-stream")
        else:
            result = await ollama_chat_generate(model, req.messages, False, req.temperature, req.max_tokens)
            return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ollama chat error: {e}")
        raise HTTPException(status_code=502, detail=f"Ollama error: {str(e)}")

@app.post("/v1/embeddings")
async def create_embeddings(req: EmbeddingRequest, api_key: str = Depends(verify_api_key)):
    model = req.model or (azure_embedding_deployments[0] if azure_embedding_deployments else DEFAULT_MODEL)

    # Normalize input to list
    input_data = req.input
    if isinstance(input_data, str):
        input_data = [input_data]

    if USE_AZURE and model in azure_embedding_deployments:
        try:
            response = await azure_async_client.embeddings.create(model=model, input=input_data)
            return response.model_dump()
        except Exception as e:
            logger.error(f"Azure embedding error: {e}")
            raise HTTPException(status_code=502, detail=f"Azure embedding error: {str(e)}")

    # Ollama fallback
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(f"{OLLAMA_HOST}/api/embed", json={"model": model, "input": input_data})
            if r.status_code != 200:
                try:
                    err_json = r.json()
                    err_msg = err_json.get("error", r.text)
                except:
                    err_msg = r.text
                raise HTTPException(status_code=r.status_code, detail=f"Ollama error: {err_msg}")

            data = r.json()
            embeddings = [{"object": "embedding", "embedding": emb, "index": i} for i, emb in enumerate(data.get("embeddings", []))]
            return {
                "object": "list",
                "data": embeddings,
                "model": model,
                "usage": {"prompt_tokens": 0, "total_tokens": 0}
            }
    except HTTPException:
        raise
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

    try:
        if req.stream:
            async def streamer():
                async with httpx.AsyncClient(timeout=None) as client:
                    async with client.stream("POST", f"{OLLAMA_HOST}/api/generate", json=payload) as response:
                        if response.status_code != 200:
                            error_text = await response.aread()
                            try:
                                err_json = json.loads(error_text)
                                err_msg = err_json.get("error", error_text.decode())
                            except:
                                err_msg = error_text.decode()
                            yield json.dumps({"error": err_msg}) + "\n"
                            return
                        async for line in response.aiter_lines():
                            if line:
                                yield line + "\n"
            return StreamingResponse(streamer(), media_type="application/x-ndjson")
        else:
            async with httpx.AsyncClient(timeout=120) as client:
                r = await client.post(f"{OLLAMA_HOST}/api/generate", json=payload)
                if r.status_code != 200:
                    try:
                        err_json = r.json()
                        err_msg = err_json.get("error", r.text)
                    except:
                        err_msg = r.text
                    raise HTTPException(status_code=r.status_code, detail=err_msg)
                return r.json()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Generate error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models")
async def ollama_models_list(api_key: str = Depends(verify_api_key)):
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(f"{OLLAMA_HOST}/api/tags")
            if r.status_code != 200:
                raise HTTPException(status_code=r.status_code, detail=f"Ollama error: {r.text}")
            return r.json()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/pull")
async def pull_model(req: PullModelRequest, api_key: str = Depends(verify_api_key)):
    async def streamer():
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", f"{OLLAMA_HOST}/api/pull", json={"name": req.name, "stream": True}) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    try:
                        err_json = json.loads(error_text)
                        err_msg = err_json.get("error", error_text.decode())
                    except:
                        err_msg = error_text.decode()
                    yield json.dumps({"error": err_msg}) + "\n"
                    return
                async for line in response.aiter_lines():
                    if line:
                        yield line + "\n"
    return StreamingResponse(streamer(), media_type="application/x-ndjson")

# =============================================================================
# ADMIN
# =============================================================================
@app.post("/admin/keys")
async def create_key(req: CreateKeyRequest, master: str = Depends(verify_master_key)):
    raw_key = "ollama_" + secrets.token_urlsafe(32)
    key_hash = hash_key(raw_key)

    # Try database first
    if SessionLocal is not None:
        try:
            db = SessionLocal()
            new_key = APIKey(key_hash=key_hash, name=req.name, created_at=int(time.time()))
            db.add(new_key)
            db.commit()
            db.close()
            return {"key": raw_key, "name": req.name, "key_hash": key_hash}
        except Exception as e:
            logger.warning(f"DB create failed, using memory fallback: {e}")

    # Fallback to in-memory
    _in_memory_keys[key_hash] = {"name": req.name, "created_at": int(time.time())}
    return {"key": raw_key, "name": req.name, "key_hash": key_hash}

@app.get("/admin/keys")
async def list_keys(master: str = Depends(verify_master_key)):
    keys = []

    # Try database first
    if SessionLocal is not None:
        try:
            db = SessionLocal()
            db_keys = db.query(APIKey).all()
            for k in db_keys:
                keys.append({"name": k.name, "key_hash": k.key_hash, "created_at": k.created_at})
            db.close()
        except Exception as e:
            logger.warning(f"DB list failed, using memory fallback: {e}")

    # Add in-memory keys
    for key_hash, info in _in_memory_keys.items():
        keys.append({"name": info["name"], "key_hash": key_hash, "created_at": info["created_at"]})

    return keys

@app.post("/admin/keys/revoke")
async def revoke_key(req: RevokeKeyRequest, master: str = Depends(verify_master_key)):
    revoked = False

    if req.key_hash:
        # Try database
        if SessionLocal is not None:
            try:
                db = SessionLocal()
                key_record = db.query(APIKey).filter(APIKey.key_hash == req.key_hash).first()
                if key_record:
                    db.delete(key_record)
                    db.commit()
                    revoked = True
                db.close()
            except Exception as e:
                logger.warning(f"DB revoke by hash failed: {e}")

        # Fallback to in-memory
        if req.key_hash in _in_memory_keys:
            del _in_memory_keys[req.key_hash]
            revoked = True

    elif req.key_name:
        # Try database
        if SessionLocal is not None:
            try:
                db = SessionLocal()
                key_record = db.query(APIKey).filter(APIKey.name == req.key_name).first()
                if key_record:
                    db.delete(key_record)
                    db.commit()
                    revoked = True
                db.close()
            except Exception as e:
                logger.warning(f"DB revoke by name failed: {e}")

        # Fallback to in-memory
        to_delete = [h for h, info in _in_memory_keys.items() if info["name"] == req.key_name]
        for h in to_delete:
            del _in_memory_keys[h]
            revoked = True
    else:
        raise HTTPException(status_code=422, detail="Provide either key_hash or key_name")

    if revoked:
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
        .dot-warn { background-color: #f59e0b; }
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
                    <p class="text-slate-400 text-sm">Enterprise-grade LLM Gateway v2.2</p>
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
                        <pre><code class="text-blue-300">curl -X POST http://this-server/v1/chat/completions \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tinyllama:latest",
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
    model="tinyllama:latest",
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

                if (data.status === 'ok' && data.ollama === 'connected') {
                    dot.classList.replace('dot-offline', 'dot-online');
                    dot.classList.replace('dot-warn', 'dot-online');
                    text.innerText = 'Ollama: Online (' + (data.available_ollama_models?.length || 0) + ' models)';
                } else {
                    dot.classList.replace('dot-online', 'dot-warn');
                    dot.classList.replace('dot-offline', 'dot-warn');
                    text.innerText = 'Ollama: ' + (data.ollama || 'Checking...');
                }
            } catch (e) {
                const dot = document.getElementById('ollama-status-dot');
                const text = document.getElementById('ollama-status-text');
                dot.classList.replace('dot-online', 'dot-offline');
                dot.classList.replace('dot-warn', 'dot-offline');
                text.innerText = 'Ollama: Disconnected';
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
                    const isOllama = model.owned_by === 'ollama';
                    div.className = `p-3 rounded-xl border border-slate-700 cursor-pointer transition-all hover:bg-slate-800 flex justify-between items-center ${selectedModel === model.id ? 'bg-blue-600/20 border-blue-500' : ''}`;
                    div.onclick = () => selectModel(model.id);

                    let iconClass = isAzure ? 'fa-cloud text-blue-400' : 'fa-microchip text-slate-500';

                    div.innerHTML = `
                        <div class="flex flex-col">
                            <span class="text-sm font-medium">${model.id}</span>
                            <span class="text-[10px] text-slate-500 uppercase">${model.owned_by}</span>
                        </div>
                        <i class="fas ${iconClass} text-xs"></i>
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
            refreshModels();
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
                    const lines = chunk.split('\n');

                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            const dataStr = line.slice(6);
                            if (dataStr === '[DONE]') continue;
                            try {
                                const data = JSON.parse(dataStr);
                                if (data.error) {
                                    responseEl.innerText = "Error: " + data.error;
                                    return;
                                }
                                const content = data.choices?.[0]?.delta?.content || "";
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
            <p>Admin endpoints require <code>X-Master-Key</code> header.</p>
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
                    <h3 class="font-mono font-bold text-purple-400">POST /v1/embeddings</h3>
                    <p class="text-sm text-slate-400">Create embeddings. Supports Azure and Ollama backends.</p>
                </div>

                <div class="border-l-4 border-yellow-500 pl-4">
                    <h3 class="font-mono font-bold text-yellow-400">POST /api/generate</h3>
                    <p class="text-sm text-slate-400">Native Ollama generate endpoint.</p>
                </div>

                <div class="border-l-4 border-red-500 pl-4">
                    <h3 class="font-mono font-bold text-red-400">POST /admin/keys</h3>
                    <p class="text-sm text-slate-400 text-yellow-500/80 italic">Requires X-Master-Key header.</p>
                    <pre>curl -X POST http://server/admin/keys \
  -H "X-Master-Key: YOUR_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{"name": "New Project Key"}'</pre>
                </div>

                <div class="border-l-4 border-orange-500 pl-4">
                    <h3 class="font-mono font-bold text-orange-400">POST /admin/keys/revoke</h3>
                    <p class="text-sm text-slate-400">Revoke an API key by hash or name.</p>
                    <pre>curl -X POST http://server/admin/keys/revoke \
  -H "X-Master-Key: YOUR_MASTER_KEY" \
  -H "Content-Type: application/json" \
  -d '{"key_hash": "..."}'  # or {"key_name": "..."}</pre>
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
