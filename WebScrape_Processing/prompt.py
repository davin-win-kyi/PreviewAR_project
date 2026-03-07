
import os
import json
import requests
from io import BytesIO
from PIL import Image

from google import genai
from google.genai import types

from dotenv import load_dotenv
load_dotenv()

# Create Gemini client (reads GEMINI_API_KEY automatically)
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))


def infer_positive_and_negative_prompt(
    image_path: str,
    class_name: str,
    model_name: str = "gemini-3.1-flash-lite-preview",
) -> dict:

    prompt_text = f"""
You are a vision model helping generate prompts for image inpainting.

Given:
- An image of a product.
- Product class name: "{class_name}"

Tasks:
1) Identify objects that are PHYSICALLY ON, TOUCHING, or OCCLUDING the main product
(e.g., pillows on a couch, a blanket draped over it, a person sitting on it).
Do NOT include background objects or items that are merely nearby or behind.

2) Produce:
- positive_prompt: the provided product class name, followed by a period.
- negative_prompt: ONLY the objects from step (1), as short generic nouns or noun phrases,
  separated by a period and a space, each ending with a period.

Rules:
- The positive_prompt MUST be exactly the provided product class name.
- Do NOT include the main product or its synonyms in the negative prompt.
- Only include objects that physically intersect or rest on the main product.
- Keep negative objects generic (e.g., pillow, blanket, person).
- Deduplicate negatives.
- If there are NO such objects, return an empty string for negative_prompt.
- Return STRICT JSON ONLY with the schema below.

Schema:
{{
  "positive_prompt": "object.",
  "negative_prompt": "object1. object2."
}}
"""

    # Load image (URL or local)
    if image_path.startswith(("http://", "https://")):
        response = requests.get(image_path)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
    else:
        img = Image.open(image_path)

    # Convert if needed
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")

    # Gemini request
    response = client.models.generate_content(
        model=model_name,
        contents=[
            types.Part.from_text(text=prompt_text),
            img,
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        ),
    )

    content = (response.text or "").strip()
    return json.loads(content)


# Example usage
if __name__ == "__main__":

    my_path = "/Users/davinwinkyi/gemini_ranking_test/test_images_2/image_01.jpg"

    result = infer_positive_and_negative_prompt(
        image_path=my_path,
        class_name="sectional",
    )

    print(result)
