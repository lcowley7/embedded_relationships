"""Async Gemini client used by every generator. Key: env GEMINI_API_KEY, else ~/.api/gemini_api_key."""
import os, asyncio
from pathlib import Path

def load_api_key():
    k = os.environ.get("GEMINI_API_KEY")
    if k: return k.strip()
    p = Path.home() / ".api" / "gemini_api_key"
    if p.exists(): return p.read_text().strip()
    raise SystemExit("No Gemini key")

GEN_MODEL = "gemini-3.5-flash"

from google import genai
from google.genai import types
_call_count = 0
async def gemini(prompts, *, temperature=1.0, max_output_tokens=1400, concurrency=8, thinking_budget=0):
    global _call_count
    client = genai.Client(api_key=load_api_key())
    cfg = types.GenerateContentConfig(temperature=temperature, max_output_tokens=max_output_tokens,
        thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget))
    sem = asyncio.Semaphore(concurrency); out = [""] * len(prompts)
    async def one(i, pr):
        async with sem:
            for a in range(4):
                try:
                    r = await client.aio.models.generate_content(model=GEN_MODEL, contents=pr, config=cfg)
                    cand = (r.candidates or [None])[0]
                    parts = (cand.content.parts if (cand and cand.content and cand.content.parts) else []) or []
                    out[i] = "".join(pt.text for pt in parts if getattr(pt, "text", None)); return
                except Exception as e:
                    if a == 3: out[i] = f"[ERROR: {e}]"
                    else: await asyncio.sleep(1.5 * (a + 1))
    await asyncio.gather(*(one(i, p) for i, p in enumerate(prompts)))
    _call_count += len(prompts); print(f"[gemini] +{len(prompts)} calls ({_call_count} total)")
    return out
