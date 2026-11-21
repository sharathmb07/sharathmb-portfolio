import json
import os
from pathlib import Path
from typing import List, Literal

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "portfolio_data.json"
FRONTEND_PATH = BASE_DIR / "public" / "index.html"

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


def fallback_portfolio_answer(message: str) -> str:
    q = message.lower()
    if any(k in q for k in ("project", "graphrag", "rag", "n8n", "multi-agent")):
        return (
            "Sharath showcases two core AI projects: Local Multi-Agent RAG System using GraphRAG "
            "(knowledge-graph retrieval with local multi-agent LLM reasoning) and RAG using LLM (n8n) "
            "(automated workflow with n8n, Gemini API, and ChromaDB at github.com/sharathmb07/RAG-using-LLM-n8n-)."
        )
    if any(k in q for k in ("skill", "stack", "tech", "python", "fastapi", "aws", "docker", "git")):
        return (
            "Sharath's technical stack covers Generative AI & LLMs (Gemini API, Google AI Studio, BigQuery ML), "
            "RAG & Agentic Systems (GraphRAG, LangChain, ChromaDB), Backend & Automation (Python, FastAPI, n8n, SQL, Docker), "
            "and Cloud & Developer Tools (Git, GitHub, AWS)."
        )
    if any(k in q for k in ("cert", "oracle", "sololearn", "skill india", "education", "vtu", "degree")):
        return (
            "Sharath is pursuing a B.E. in Artificial Intelligence and Data Science at VTU (Expected 2027). "
            "His certifications include Oracle Agentic AI Certified Foundations Associate, Skill India Digital Hub "
            "(AI & Data Engineering, Agentic AI, BigQuery ML), and Sololearn (Programming & Software Design)."
        )
    if any(k in q for k in ("contact", "email", "linkedin", "github", "hire", "resume", "cv")):
        return (
            "You can contact Sharath at smb81980@gmail.com, connect on linkedin.com/in/sharathmb07, "
            "or explore his repositories at github.com/sharathmb07."
        )
    return (
        "Sharath M B is an AI & Data Science Engineer (B.E. at VTU, Expected 2027) focused on Generative AI, "
        "GraphRAG, Agentic AI, and n8n automation. Ask me about his projects, skills, Oracle certification, or contact links!"
    )


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
    model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite").strip()
    if not api_key or api_key == "your_gemini_api_key_here":
        return fallback_portfolio_answer(message)

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
            "maxOutputTokens": 512,
            "thinkingConfig": {"thinkingLevel": "minimal"},
        },
    }

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                url,
                headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
                json=payload,
            )
        if resp.status_code != 200:
            return fallback_portfolio_answer(message)
        data = resp.json()
        parts = data["candidates"][0]["content"]["parts"]
        text_parts = [p["text"] for p in parts if "text" in p and not p.get("thought")]
        return "\n".join(text_parts).strip() or fallback_portfolio_answer(message)
    except (httpx.TimeoutException, httpx.HTTPError, KeyError, IndexError):
        return fallback_portfolio_answer(message)


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "provider": os.getenv("LLM_PROVIDER", "gemini"),
        "model": os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
        "owner": PORTFOLIO_DATA["name"],
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not is_likely_in_scope(req.message):
        return ChatResponse(reply=REFUSAL_MESSAGE, in_scope=False, provider="guardrail")

    reply = await call_gemini(req.message, req.history)
    in_scope = REFUSAL_MESSAGE[:40].lower() not in reply.lower()
    return ChatResponse(reply=reply, in_scope=in_scope, provider="gemini")


@app.get("/")
async def serve_frontend():
    if not FRONTEND_PATH.exists():
        raise HTTPException(status_code=404, detail="public/index.html not found")
    return FileResponse(FRONTEND_PATH)
