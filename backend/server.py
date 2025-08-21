from fastapi import FastAPI, APIRouter, HTTPException, Request
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone, date, time

# ============================
# Env & App Bootstrap
# ============================
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection (MUST use MONGO_URL from env)
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Main FastAPI app and router with /api prefix
app = FastAPI()
api_router = APIRouter(prefix="/api")

# CORS (origins from env)
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================
# Helpers: UUID, Time, Mongo serialization
# ============================

def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def prepare_for_mongo(data: Dict[str, Any]) -> Dict[str, Any]:
    # Convert date/time objects to primitives before storing
    d = dict(data)
    if isinstance(d.get('timestamp'), datetime):
        d['timestamp'] = d['timestamp'].isoformat()
    if isinstance(d.get('date'), date):
        d['date'] = d['date'].isoformat()
    if isinstance(d.get('time'), time):
        d['time'] = d['time'].strftime('%H:%M:%S')
    return d


def parse_from_mongo(item: Dict[str, Any]) -> Dict[str, Any]:
    d = dict(item)
    # Parse timestamp to ISO string if stored differently
    ts = d.get('timestamp')
    if isinstance(ts, datetime):
        d['timestamp'] = ts.isoformat()
    elif isinstance(ts, str):
        # keep as ISO string for JSON
        d['timestamp'] = ts
    # ignore Mongo _id
    d.pop('_id', None)
    return d

# ============================
# Models
# ============================
class StatusCheck(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: str = Field(default_factory=now_utc_iso)


class StatusCheckCreate(BaseModel):
    client_name: str


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatMessage(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    user_message: str
    assistant_response: str
    timestamp: str = Field(default_factory=now_utc_iso)


class ChatResponse(BaseModel):
    id: str
    message: str
    response: str
    session_id: str
    timestamp: str


# ============================
# LLM via Emergent Integrations (playbook): unify client
# ============================
# We must use Emergent Universal LLM Key (EMERGENT_LLM_KEY)
# Model: gpt-4.1 via OpenAI provider through the integration layer
try:
    # Lazy import pattern inside functions to avoid import errors if missing during startup
    import importlib
    unify_spec = importlib.util.find_spec("unify")
    if unify_spec is None:
        # Try alternative module name
        unifyai_spec = importlib.util.find_spec("unifyai")
        if unifyai_spec is None:
            UNIFY_AVAILABLE = False
            unify = None  # type: ignore
        else:
            UNIFY_AVAILABLE = True
            import unifyai as unify  # type: ignore
    else:
        UNIFY_AVAILABLE = True
        import unify  # type: ignore
except Exception:
    UNIFY_AVAILABLE = False
    unify = None  # type: ignore


def get_llm_client():
    api_key = os.environ.get("EMERGENT_LLM_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Missing EMERGENT_LLM_KEY on server")
    if not UNIFY_AVAILABLE:
        raise HTTPException(status_code=500, detail="LLM integration not installed. Please contact admin.")
    # Use Unify abstraction to target OpenAI gpt-4.1
    # The identifier format follows the integration playbook: "gpt-4.1@openai"
    return unify.Unify("gpt-4.1@openai", api_key=api_key)


async def run_completion(prompt: str) -> str:
    """Execute a completion request using the Unify client. The library supports awaitable generate."""
    client = get_llm_client()
    try:
        # Some unify versions expose async generate; fall back to sync if needed
        gen = getattr(client, 'generate', None)
        if gen is None:
            raise RuntimeError("LLM client missing generate()")
        result = gen(prompt)
        if hasattr(result, "__await__"):
            text = await result
        else:
            text = result
        if not isinstance(text, str):
            text = str(text)
        return text
    except Exception as e:
        logger.exception("LLM generation failed")
        raise HTTPException(status_code=500, detail=f"Chat completion failed: {e}")


# ============================
# Routes
# ============================
@api_router.get("/")
async def root():
    return {"message": "Hello World"}


@api_router.get("/health")
async def health():
    return {"status": "ok", "service": "unis-ai"}


@api_router.post("/status", response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate):
    status_obj = StatusCheck(client_name=input.client_name)
    _ = await db.status_checks.insert_one(prepare_for_mongo(status_obj.model_dump()))
    return status_obj


@api_router.get("/status", response_model=List[StatusCheck])
async def get_status_checks():
    status_checks = await db.status_checks.find().to_list(1000)
    return [StatusCheck(**parse_from_mongo(sc)) for sc in status_checks]


@api_router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest, request: Request):
    # Basic validation (short and safe)
    msg = (payload.message or "").strip()
    if not msg:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    session_id = payload.session_id or str(uuid.uuid4())

    # Prepare minimal context from recent 8 messages
    history_docs = await db.chat_messages.find({"session_id": session_id}).sort("timestamp", -1).limit(8).to_list(length=8)
    history_docs = list(reversed(history_docs))
    context_lines = []
    for h in history_docs:
        h = parse_from_mongo(h)
        context_lines.append(f"User: {h.get('user_message','')}")
        context_lines.append(f"Assistant: {h.get('assistant_response','')}")

    full_prompt = "\n".join(context_lines + [f"User: {msg}", "Assistant:"])

    # Call LLM
    completion = await run_completion(full_prompt)

    # Persist
    chat_msg = ChatMessage(
        session_id=session_id,
        user_message=msg,
        assistant_response=completion,
    )
    await db.chat_messages.insert_one(prepare_for_mongo(chat_msg.model_dump()))

    return ChatResponse(
        id=chat_msg.id,
        message=msg,
        response=completion,
        session_id=session_id,
        timestamp=chat_msg.timestamp,
    )


@api_router.get("/chat/history/{session_id}")
async def chat_history(session_id: str):
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id required")
    docs = await db.chat_messages.find({"session_id": session_id}).sort("timestamp", 1).to_list(length=200)
    messages = [parse_from_mongo(d) for d in docs]
    return {"session_id": session_id, "messages": messages}


# Mount router
app.include_router(api_router)


@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()