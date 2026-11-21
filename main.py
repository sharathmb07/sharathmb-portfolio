import json
import os
from pathlib import Path
from typing import List, Literal, Optional

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "portfolio_data.json"
FRONTEND_PATH = BASE_DIR.parent / "frontend" / "index.html"

with DATA_PATH.open("r", encoding="utf-8") as f:
    PORTFOLIO_DATA = json.load(f)

REFUSAL_MESSAGE = (
    "I can only answer questions about Sharath M B's portfolio — including his "
    "AI & Data Science skills, projects (Local Multi-Agent RAG System using GraphRAG "
    "and RAG using LLM (n8n)), education, certifications, and contact links. "
    "What would you like to know about Sharath?"
)

SYSTEM_PROMPT = f"""You are Sharath M B's dedicated AI Portfolio Assistant.
Your ONLY purpose is to help recruiters, engineers, and visitors learn about Sharath M B based STRICTLY on the verified portfolio JSON below.

STRICT RULES:
1. PORTFOLIO-ONLY SCOPE: If the user asks about anything unrelated to Sharath M B's portfolio (such as general coding help, math homework, weather, news, politics, jokes, recipes, movies, or other people), you MUST refuse politely and reply ONLY with:
"{REFUSAL_MESSAGE}"
2. ZERO FABRICATION: Never invent internships, work experience, metrics, projects, skills, or certifications that are not in the JSON below.
3. STAY CONCISE & HELPFUL: Answer directly in 2-4 clear sentences using natural recruiter-friendly language.
4. GREETINGS ALLOWED: If the user says hello/hi/thanks, greet them briefly and invite them to ask about Sharath's skills, projects, certifications, or contact details.

VERIFIED PORTFOLIO DATA:
{json.dumps(PORTFOLIO_DATA, indent=2)}
"""

# Fast keyword pre-filter for clearly off-topic prompts to save LLM tokens
PORTFOLIO_ALLOW_KEYWORDS = {
    "sharath", "portfolio", "skill", "skills", "stack", "tech", "python", "fastapi",
    "rag", "graphrag", "n8n", "llm", "ai", "agent", "agentic", "gemini", "chroma",
    "chromadb", "langchain", "bigquery", "sql", "docker", "git", "github", "aws",
    "project", "projects", "cert", "certification", "certifications", "oracle",
    "skill india", "sololearn", "education", "degree", "vtu", "college", "graduate",
    "graduation", "resume", "cv", "contact", "email", "linkedin", "hire", "hiring",
    "intern", "internship", "role", "job", "recruiter", "about", "who", "location",
    "karnataka", "india", "hello", "hi", "hey", "thanks", "thank", "help", "work",
    "experience", "background", "multi-agent", "automation", "cloud"
}


def is_likely_in_scope(message: str) -> bool:
    text = message.lower().strip()
    if len(text) <= 18 and any(g in text for g in ("hi", "hello", "hey", "thanks", "thank")):
        return True
    return any(kw in text for kw in PORTFOLIO_ALLOW_KEYWORDS)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., max_length=2000)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1500)
    history: List[ChatMessage] = Field(default_factory=list)


class ChatResponse(BaseModel):
    reply: str
    in_scope: bool
    provider: str


app = FastAPI(title="Sharath M B — Portfolio AI Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def call_gemini(message: str, history: List[ChatMessage]) -> str:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
    if not api_key or api_key == "your_gemini_api_key_here":
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is not configured in backend/.env.",
        )

    contents = []
    for turn in history[-6:]:
        role = "user" if turn.role == "user" else "model"
        contents.append({"role": role, "parts": [{"text": turn.content}]})
    contents.append({"role": "user", "parts": [{"text": message}]})

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    payload = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": contents,
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 350,
        },
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(url, params={"key": api_key}, json=payload)
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Gemini API error: {resp.text}")
        data = resp.json()

    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except (KeyError, IndexError) as exc:
        raise HTTPException(status_code=502, detail="Unexpected Gemini response format") from exc


async def call_ollama(message: str, history: List[ChatMessage]) -> str:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    model = os.getenv("OLLAMA_MODEL", "llama3.2").strip()

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history[-6:]:
        messages.append({"role": turn.role, "content": turn.content})
    messages.append({"role": "user", "content": message})

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{base_url}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {"temperature": 0.2},
            },
        )
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Ollama error: {resp.text}")
        data = resp.json()

    return data.get("message", {}).get("content", "").strip() or REFUSAL_MESSAGE


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "provider": os.getenv("LLM_PROVIDER", "gemini"),
        "owner": PORTFOLIO_DATA["name"],
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    provider = os.getenv("LLM_PROVIDER", "gemini").lower().strip()

    # Layer 1: Fast guardrail against clearly off-topic prompts
    if not is_likely_in_scope(req.message):
        return ChatResponse(reply=REFUSAL_MESSAGE, in_scope=False, provider="guardrail")

    # Layer 2: Grounded LLM reasoning with strict system-prompt scope enforcement
    if provider == "ollama":
        reply = await call_ollama(req.message, req.history)
    else:
        reply = await call_gemini(req.message, req.history)

    in_scope = REFUSAL_MESSAGE[:40].lower() not in reply.lower()
    return ChatResponse(reply=reply, in_scope=in_scope, provider=provider)


@app.get("/")
async def serve_frontend():
    if not FRONTEND_PATH.exists():
        raise HTTPException(status_code=404, detail="frontend/index.html not found")
    return FileResponse(FRONTEND_PATH)
