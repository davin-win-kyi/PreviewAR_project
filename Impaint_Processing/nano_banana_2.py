import os
import json
import traceback
from typing import Dict, Any, List
from io import BytesIO

import requests
from PIL import Image
from dotenv import load_dotenv

from google import genai
from google.genai import types


load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise SystemExit("CRITICAL ERROR: GEMINI_API_KEY not found in .env file.")

client = genai.Client(api_key=GEMINI_API_KEY)


def rank_images_from_urls(
    image_urls: List[str],
    target_object: str,
    model_name: str = "gemini-3.1-flash-lite-preview",
) -> Dict[str, Any]:

    instruction = f"""
You are an expert image analyst. Rank the provided images of a {target_object}
based on which angle reveals the maximum geometric surface area (faces) of the {target_object}.

Instructions:
1) Prioritize Geometry: Focus ONLY on the underlying structure of the {target_object}.
2) View Preference: Favor high-angle three-quarter views (Top, Front, Side planes).
3) Ranking Logic:
   - Rank 1: Shows 3+ planes (Top, Front, Side).
   - Rank 2: Shows 2 planes (Front, Side).
   - Rank 3: Flat/2D profile view.
4) Completeness: The entire {target_object} must be in the frame.

Return STRICT JSON only:
{{
  "best_image_url": "URL",
  "reasoning": "explanation",
  "scores": {{ "URL": {{"exposure_score": 1-10, "notes": "..."}} }}
}}
""".strip()

    parts: List[types.Part] = [types.Part.from_text(text=instruction)]

    ok = 0
    for url in image_urls:
        try:
            resp = requests.get(
                url,
                timeout=15,
                headers={
                    # Amazon sometimes blocks "python-requests" UA
                    "User-Agent": "Mozilla/5.0",
                    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
                    "Referer": "https://www.amazon.com/",
                },
            )
            resp.raise_for_status()

            # Determine mime type; fall back to jpeg if unknown
            mime = (resp.headers.get("Content-Type") or "image/jpeg").split(";")[0].strip()

            # Optional: sanity check that bytes are actually an image
            # (Amazon sometimes returns HTML/error pages)
            if not mime.startswith("image/"):
                raise RuntimeError(f"Non-image content-type: {mime}")

            parts.append(types.Part.from_bytes(data=resp.content, mime_type=mime))
            parts.append(types.Part.from_text(text=f"Identifier: {url}"))
            ok += 1

        except Exception as e:
            print(f"⚠️ Could not load image from {url}: {e}")

    if ok == 0:
        return {"best_image_url": "", "reasoning": "No valid images could be processed.", "scores": {}}

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=parts,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0,
                max_output_tokens=512,
            ),
        )
        raw = (response.text or "").strip()
        return json.loads(raw)

    except Exception as e:
        status = getattr(locals().get("resp", None), "status_code", None)
        ctype = None
        text_head = None
        try:
            if "resp" in locals() and resp is not None:
                ctype = resp.headers.get("Content-Type")
                # show a short preview (safe) of what we actually got
                text_head = resp.text[:200]
        except Exception:
            pass

        print(f"⚠️ Could not load image from {url}: {type(e).__name__}: {e}")
        print(f"    status={status} content-type={ctype}")
        if text_head:
            print(f"    body[:200]={text_head!r}")