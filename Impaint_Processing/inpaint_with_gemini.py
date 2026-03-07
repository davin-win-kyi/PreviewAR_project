import os
import sys
import uuid
from io import BytesIO
from pathlib import Path
from typing import List

from PIL import Image
from google import genai
from google.genai.types import GenerateContentConfig

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.text_utils import format_object_list
from utils.mask_utils import combine_images_with_mask

# Default uploads directory
UPLOADS_DIR = "uploads"
os.makedirs(UPLOADS_DIR, exist_ok=True)


def replace_black_with_white(
    img: Image.Image,
    threshold: int = 10,
) -> Image.Image:
    """
    Replaces black (or near-black) pixels with white.

    threshold:
        0  -> only pure (0,0,0)
        10 -> replaces very dark pixels too
    """
    img = img.convert("RGB")
    pixels = img.load()

    width, height = img.size

    for y in range(height):
        for x in range(width):
            r, g, b = pixels[x, y]

            # Replace pure or near black
            if r <= threshold and g <= threshold and b <= threshold:
                pixels[x, y] = (255, 255, 255)

    return img



def build_inpaint_prompt(target_class: str, classes_in_crop: str) -> str:
    """
    Builds a structured inpainting prompt that:
    - fills masked regions with consistent object texture
    - removes unwanted objects
    - ensures a pure white background
    """

    other_objects = classes_in_crop

    if other_objects:
        prompt = (
            f"Fill the masked region with the same texture, material, and geometry as the rest of the {target_class}. "
            f"The final image must contain only the {target_class}. "
            f"There must be no gaps, holes, or missing parts in the {target_class}. "
            f"Do not include the following objects: {other_objects}. "
            f"The background must be a clean pure white studio background with no objects, shadows, or textures."
        )
    else:
        prompt = (
            f"Fill the masked region with the same texture, material, and geometry as the rest of the {target_class}. "
            f"The final image must contain only the {target_class}. "
            f"There must be no gaps, holes, or missing parts in the {target_class}. "
            f"The background must be a clean pure white studio background with no objects, shadows, or textures."
        )

    return prompt

def recontext_masked_area(
    combined_image_path: str,
    prompt: str,
    inpainted_output_path: str | None = None,
) -> Image.Image:
    """
    Sends the combined image + prompt to Gemini to in-paint the masked area.
    """
    client = genai.Client()

    full_prompt = prompt
    print("[gemini] prompt:", full_prompt)

    image = Image.open(combined_image_path)

    if inpainted_output_path is None:
        inpainted_output_path = combined_image_path

    response = client.models.generate_content(
        model="gemini-3.1-flash-image-preview",
        contents=[full_prompt, image],
        config=GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"],
            candidate_count=1,
        ),
    )

    for part in response.candidates[0].content.parts:
        if part.text is not None:
            print("[gemini] text:", part.text)
        elif part.inline_data is not None:
            image = Image.open(BytesIO(part.inline_data.data))

            # Replace black pixels with white
            image = replace_black_with_white(image, threshold=10)

            image.save(inpainted_output_path)

            print(f"[gemini] Inpainted image saved to {inpainted_output_path}")

    return image


def inpaint_image_with_mask(
    original_image_path: str,
    prompt: str,
    combined_output_path: str | None = None,
    inpainted_output_path: str | None = None,
) -> tuple[str, str]:
    """
    1. Combines original + mask as described above (object kept, background -> BW mask).
    2. Sends that image + prompt to Gemini to in-paint the masked (white) area.

    Returns:
        (combined_path, inpainted_path)
    """
    if combined_output_path is None:
        combined_output_path = os.path.join(UPLOADS_DIR, f"{uuid.uuid4()}_combined.png")
    if inpainted_output_path is None:
        inpainted_output_path = os.path.join(UPLOADS_DIR, f"{uuid.uuid4()}_inpainted.png")

    print("[pipeline] Combining original image and mask...")
    combined_path = combine_images_with_mask(
        original_path=original_image_path,
        output_path=combined_output_path,
        resize_mask=True,
    )

    print("[pipeline] Sending combined image to Gemini for inpainting...")
    recontext_masked_area(combined_path, prompt, inpainted_output_path=inpainted_output_path)

    return combined_path, inpainted_output_path

def get_inpainted_image_path(
    target_class,
    classes_in_crop,
    original_image_path,
):  
    # Build prompt
    prompt = build_inpaint_prompt(target_class, classes_in_crop)
    print("[summary] inpaint prompt:", prompt)

    # Output path for inpainted image
    inpainted_output_path = os.path.join(UPLOADS_DIR, "inpainted_result.png")

    # Directly inpaint the original image (no mask combining)
    print("[pipeline] Sending image to Gemini for inpainting...")
    recontext_masked_area(
        combined_image_path=original_image_path,
        prompt=prompt,
        inpainted_output_path=inpainted_output_path
    )

    return {
        "target_class": target_class,
        "classes_in_crop": classes_in_crop,
        "prompt": prompt,
        "inpainted_image_path": inpainted_output_path,
    }
