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

# OpenAI (official) async client
try:
    from openai import AsyncOpenAI
except Exception:  # library may not be installed yet; we guard when used
    AsyncOpenAI = None  # type: ignore

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
# LLM via OpenAI official SDK (using OPENAI_API_KEY)
# ============================

def get_openai_client() -> AsyncOpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Missing OPENAI_API_KEY on server")
    if AsyncOpenAI is None:
        raise HTTPException(status_code=500, detail="OpenAI SDK not installed on server")
    return AsyncOpenAI(api_key=api_key)


async def run_completion(history_lines: List[str], user_msg: str) -> str:
    client = get_openai_client()
    # Build messages with minimal system prompt + recent context
    messages = []
    system_prompt = "You are Unis AI, a helpful, concise assistant."
    messages.append({"role": "system", "content": system_prompt})

    # Convert context pairs into alternating messages where possible
    # history_lines contains lines like "User: ..." and "Assistant: ..."
    # We'll compress them to messages
    for line in history_lines:
        if line.startswith("User: "):
            messages.append({"role": "user", "content": line.replace("User: ", "", 1)})
        elif line.startswith("Assistant: "):
            messages.append({"role": "assistant", "content": line.replace("Assistant: ", "", 1)})

    messages.append({"role": "user", "content": user_msg})

    try:
        resp = await client.chat.completions.create(
            model="gpt-4.1",
            messages=messages,
            temperature=0.7,
        )
        content = resp.choices[0].message.content if resp and resp.choices else ""
        return content or ""
    except Exception as e:
        logger.exception("OpenAI generation failed")
        raise HTTPException(status_code=500, detail=f"Chat completion failed: {e}")


# ============================
# Routes
# ============================
@api_router.get("/")
async def root():
    return {"message": "Hello World"}


@api_router.get("/health")
async def health():
    # indicate LLM readiness too
    ok = True
    detail = "ok"
    try:
        _ = os.environ.get("OPENAI_API_KEY")
        if not _:
            ok = False
            detail = "missing OPENAI_API_KEY"
    except Exception:
        ok = False
        detail = "key check failed"
    return {"status": "ok" if ok else "degraded", "detail": detail, "service": "unis-ai"}


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

    # Call LLM
    completion = await run_completion(context_lines, msg)

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