# Sharath M B — Full-Stack Portfolio

Apple-inspired Burgundy & Off-White 3D portfolio connected to a FastAPI + Gemini AI assistant with strict portfolio-only guardrails.

## Deploy to Vercel (3 Steps)
1. Push this entire folder to a GitHub repo under `github.com/sharathmb07` (do NOT upload `.env`).
2. Go to Vercel -> **Add New -> Project** -> Import your GitHub repo (Vercel auto-detects FastAPI via root `main.py` and `requirements.txt`).
3. Add these 3 **Environment Variables** in Vercel and click **Deploy**:
   - `LLM_PROVIDER` = `gemini`
   - `GEMINI_MODEL` = `gemini-2.5-flash`
   - `GEMINI_API_KEY` = your key from Google AI Studio

## Run Locally
```bash
pip install -r requirements.txt
cp backend/.env.example backend/.env
uvicorn main:app --reload --port 8000
```
