# demo/ — hosted web demo (testing only)

This folder contains a **self-contained web demo** of RecentAgent that can run
on Vercel as a Python serverless function, so people can try the agent's
behavior in a browser. The real app stays local-first and untouched: the
Streamlit UI, document upload, Ollama/Groq/OpenAI providers and the full tool
set all keep working exactly as documented in the main README.

> ⚠️ **Testing demo only — not the product.** It runs on a free-tier Gemini key,
> serves the four sample documents as its knowledge base, keeps no state between
> cold starts, and includes none of your local documents.

## What it is

| File | Purpose |
|---|---|
| `index.py` | FastAPI app: serves the page and exposes `POST /api/chat` — a faithful port of the **ReAct loop** (same prompts, same parsing), with tools: `knowledge_base_search` (TF-IDF from scratch), `calculator` (same AST sandbox), `wikipedia` (public REST API), `current_datetime`. Gemini is called through the OpenAI-compatible endpoint using the `GEMINI_API` env var. Standard library only, apart from FastAPI itself. |
| `index.html` | The UI — a pixel-faithful replica of the **local Streamlit app's design** (sidebar with Configuration / Agent settings / Knowledge Base sections, wide chat area, reasoning-trace and sources expanders, bottom chat input). The knowledge-base panel shows the 4 sample documents (demo data; upload/paste/delete/clear are shown but locked — they work in the real project, not in the demo) |
| `logo.png` | Copy of the project logo served by the demo (sidebar brand image) |
| `favicon.png`, `favicon-32.png`, `apple-touch-icon.png` | Browser-tab icons (navy tile + logo), served by `index.py` |
| `sample_docs/` | The same four sample documents used by the local app |
| `requirements.txt` | `fastapi` only |
| `vercel.json` | 60s function timeout for the agent loop |
| `deploy.mjs` | Deploys this folder to Vercel via the REST API |

## Deploying (or updating) the demo

From the **repo root**, with `VERCEL_API_TOKEN` and `GEMINI_API` set in your
environment (values are never printed or committed):

```bash
node demo/deploy.mjs
```

## Differences vs the local app

- **No web_search tool** — DuckDuckGo scraping is unreliable from serverless IPs.
- **No document upload** — the knowledge base is the four sample documents.
- **Conversation history resets** whenever the serverless instance goes cold.
- **Free-tier throttling**: under sustained load the Gemini free tier soft-throttles
  (requests return empty completions). The demo retries with backoff and shows a
  friendly message asking to try again in a moment — this is a free-tier limit,
  not a bug. Check `/api/health` and `/api/llm-debug` to verify the model from
  the deployment's own network position.
- **LLM fixed to Gemini** (`gemini-flash-lite-latest`) via the encrypted
  `GEMINI_API` project environment variable.
