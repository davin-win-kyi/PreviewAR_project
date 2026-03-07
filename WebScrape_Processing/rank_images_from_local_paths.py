import json
import traceback
from typing import Dict, Any, List

from PIL import Image
from google import genai
from google.genai import types
from dotenv import load_dotenv
import os

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def rank_images_from_local_paths(
    image_paths: List[str],
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
  "best_image_path": "path",
  "reasoning": "explanation",
  "scores": {{ "path": {{"exposure_score": 1-10, "notes": "..."}} }}
}}
""".strip()

    contents: List[Any] = [instruction]
    valid = 0
    MAX_RES = 2048

    for path in image_paths:
        try:
            img = Image.open(path)

            if max(img.size) > MAX_RES:
                img.thumbnail((MAX_RES, MAX_RES))

            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")

            contents.append(img)
            contents.append(f"Identifier: {path}")
            valid += 1

        except Exception as e:
            print(f"⚠️ Could not load local image {path}: {e}")

    if valid == 0:
        return {
            "best_image_path": "",
            "reasoning": "No valid local images could be processed.",
            "scores": {},
        }

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.0,
            ),
        )

        raw = (response.text or "").strip()
        print("\n[DEBUG raw Gemini output]")
        print(raw)
        print("[END DEBUG]\n")
        return json.loads(raw)

    except Exception as e:
        traceback.print_exc()
        return {
            "best_image_path": "",
            "reasoning": f"API Error: {e}",
            "scores": {},
        }