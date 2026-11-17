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
    model_name: str = "gemini-3.1-pro-preview",
) -> Dict[str, Any]:
    instruction = f"""
You are an expert product-image ranking system.

Task:
Given multiple images, choose the single best image of the target object: {target_object}.

Goal:
Select the image that best exemplifies the target object category by showing one PRIMARY instance of the full {target_object} in the clearest, most canonical, least obstructed, and most informative way.

Definition of PRIMARY instance:
The PRIMARY instance is the single instance of the target object in an image that is largest, most central, and most visually prominent.

Definition of exemplifies the target object:
An image exemplifies the target object well if a human viewer would immediately recognize the PRIMARY object as a clear, representative example of a full {target_object}, rather than a part, accessory, attachment, decoration, or related object.

IMPORTANT:
Always evaluate only ONE PRIMARY instance of the target object per image.
Do NOT combine evidence across multiple objects in the same image.

Use STRICT priority ordering.
Do NOT average criteria together.
Apply the following criteria in order, from most important to least important:

1) Exemplifies the full target object (highest priority)
   - Prefer images where the PRIMARY object is clearly the full, assembled, whole {target_object}.
   - Strongly penalize images that mainly show only a part, accessory, attachment, replacement component, or related object.
   - Prefer the full object over any isolated component.
   - Example: if the target object is a lamp, prefer the full lamp over only a bulb, shade, cord, or base.

2) Default / Canonical Configuration
   - Prefer images where the PRIMARY {target_object} appears in its normal, standard, intended resting or display configuration.
   - Penalize images where it is folded, collapsed, disassembled, unusually opened, partially assembled, deformed, flipped into an unnatural pose, or shown in a clearly non-standard state.

3) Least Obstruction
   - Prefer images where the PRIMARY {target_object} is obstructed the least by other objects, people, hands, labels, overlays, text, decor, cables, or clutter.
   - Penalize overlap, blockage, and visual interference.

4) Most Visible Faces of the PRIMARY object
   - Prefer images where the PRIMARY {target_object} reveals the most distinct geometric faces or planes of that same single object.
   - Example: top + front + side is better than front + side, and front + side is better than a flat frontal or profile view.
   - Count visible faces only for the ONE PRIMARY object.
   - Do NOT reward an image because multiple objects of the same category are visible.
   - If two target objects are visible, do NOT combine their visible faces.

5) Fullness in Frame
   - Prefer images where the full PRIMARY {target_object} is inside the frame and not cropped off.

Additional rules:
- Evaluate only one PRIMARY target object per image.
- If multiple instances of the target object appear, judge only the most prominent single instance.
- Ignore background objects except when they obstruct the PRIMARY object.
- Prefer clean product-like views over busy lifestyle scenes when higher-priority criteria are otherwise similar.
- If an image contains both the full target object and one of its components, evaluate the full target object, not the component.
- The winning image should be the one that best satisfies the ordered priorities above.

Tie-breaking:
- First tie-break: better exemplar of the full target object
- Second tie-break: more canonical/default configuration
- Third tie-break: less obstruction
- Fourth tie-break: more visible faces of the PRIMARY object
- Fifth tie-break: fuller object in frame

Scoring guidance:
- exemplar_score: 1 means mostly a part/accessory/ambiguous object, 10 means a very clear full representative example of the target object
- default_configuration_score: 1 means highly non-standard, 10 means perfectly canonical
- obstruction_score: 1 means heavily obstructed, 10 means nearly unobstructed
- visible_faces_score: 1 means weak flat/profile view, 10 means strong informative multi-face view of the PRIMARY object
- fullness_score: 1 means heavily cropped, 10 means fully visible in frame

Return STRICT JSON only.
Do not include markdown.
Do not include code fences.
Do not include any extra text before or after the JSON.
Return exactly one JSON object, not an array.
Do not wrap the object in [].

Use the exact provided identifier string as the key in "scores" and as the value of "best_image_path".

Return JSON in exactly this structure:
{{
  "best_image_path": "one of the provided identifiers",
  "reasoning": "brief explanation of why this image won based on the ordered criteria",
  "scores": {{
    "IDENTIFIER_1": {{
      "exemplar_score": 1,
      "default_configuration_score": 1,
      "obstruction_score": 1,
      "visible_faces_score": 1,
      "fullness_score": 1,
      "rank_reason": "brief notes"
    }},
    "IDENTIFIER_2": {{
      "exemplar_score": 1,
      "default_configuration_score": 1,
      "obstruction_score": 1,
      "visible_faces_score": 1,
      "fullness_score": 1,
      "rank_reason": "brief notes"
    }}
  }}
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
            elif img.mode != "RGB":
                img = img.convert("RGB")

            contents.append(img)
            contents.append(f"IMAGE_IDENTIFIER={path}")
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

        result = json.loads(raw)

        # Handle the case where the model returns [{...}] instead of {...}
        if isinstance(result, list):
            if len(result) == 0:
                raise ValueError("Model output was an empty JSON array.")
            if len(result) == 1 and isinstance(result[0], dict):
                result = result[0]
            else:
                raise ValueError(
                    "Model output was a JSON array, but not a single-object array."
                )

        if not isinstance(result, dict):
            raise ValueError("Model output was not a JSON object.")

        if "best_image_path" not in result:
            raise ValueError("Missing 'best_image_path' in model output.")

        if "reasoning" not in result:
            raise ValueError("Missing 'reasoning' in model output.")

        if "scores" not in result or not isinstance(result["scores"], dict):
            raise ValueError("Missing or invalid 'scores' in model output.")

        # Optional sanity check: best_image_path should be one of the scored identifiers
        if result["best_image_path"] and result["best_image_path"] not in result["scores"]:
            print(
                f"⚠️ Warning: best_image_path '{result['best_image_path']}' "
                "was not present as a key in scores."
            )

        return result

    except Exception as e:
        traceback.print_exc()
        return {
            "best_image_path": "",
            "reasoning": f"API Error: {e}",
            "scores": {},
        }