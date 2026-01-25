import os
import json
from openai import OpenAI

from dotenv import load_dotenv  # type: ignore
load_dotenv()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def infer_positive_and_negative_prompt(
    image_url: str,
    classes: list[str],
    model: str = "gpt-5.2",
) -> dict:
    prompt_text = f"""
You are a vision model helping generate prompts for image inpainting / segmentation.

Given:
- An image of a product.
- Candidate product class names: {classes}

Tasks:
1) Choose the SINGLE best name from the candidate list that describes the MAIN product.
2) Identify OTHER objects visible in the image (not the main product).
3) Produce:
   - positive_prompt: the chosen product name, followed by a period.
   - negative_prompt: other objects as short generic nouns or noun phrases,
     separated by a period and a space, each ending with a period.

Rules:
- The positive_prompt MUST come from the provided class list (exact match).
- Do NOT include the main product or its synonyms in the negative prompt.
- Keep negative objects generic (e.g., pillow, rug, plant, table, lamp, wall art, person).
- Deduplicate negatives.
- Return STRICT JSON ONLY with the exact schema below.
- No markdown, no commentary, no extra keys.

Schema:
{{
  "positive_prompt": "object.",
  "negative_prompt": "object1. object2. object3."
}}
"""

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url,
                            "detail": "high"
                        },
                    },
                ],
            }
        ],
    )

    content = resp.choices[0].message.content.strip()
    return json.loads(content)


# Example usage
best_image_path = "https://m.media-amazon.com/images/I/71v0slLGRlL._AC_SL1500_.jpg"
classes = ["sectional", "couch", "sofa", "loveseat", "recliner"]

result = infer_positive_and_negative_prompt(
    best_image_path,
    classes,
    model="gpt-5.2"
)

print(result)
