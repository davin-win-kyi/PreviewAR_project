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
    - reconstructs missing parts of the target object
    - removes unwanted objects
    - places the final object on a pure white studio background
    """

    other_objects = classes_in_crop.strip()

    base_prompt = (
        f"Edit the image so the final result contains only one complete {target_class}. "
        f"Reconstruct and fill any masked, missing, or occluded regions so they match the {target_class}'s "
        f"existing texture, material, color, shape, and geometry. "
        f"The {target_class} must look complete, natural, and fully assembled, with no gaps, holes, missing parts, or artifacts. "
        f"Keep the object centered and fully visible. "
        f"Place the object on a clean pure white studio background. "
        f"The background must be solid white and contain no other objects, no decorations, no shadows, no text, no reflections, and no visible texture."
    )

    if other_objects:
        prompt = (
            base_prompt
            + f" Remove any non-target objects, including: {other_objects}. "
            + f"Do not include any object other than the single {target_class}."
        )
    else:
        prompt = (
            base_prompt
            + f" Do not include any object other than the single {target_class}."
        )

    return prompt


from io import BytesIO
from PIL import Image
from google import genai
from google.genai.types import GenerateContentConfig

def recontext_masked_area(
    combined_image_path: str,
    prompt: str,
    inpainted_output_path: str | None = None,
) -> Image.Image:
    """
    Sends the combined image + prompt to Gemini to inpaint the masked area.
    Saves and returns the generated image directly, without replacing black pixels.
    """
    client = genai.Client()

    full_prompt = prompt
    print("[gemini] prompt:", full_prompt)

    input_image = Image.open(combined_image_path)

    if inpainted_output_path is None:
        inpainted_output_path = combined_image_path

    response = client.models.generate_content(
        model="gemini-3-pro-image-preview",
        contents=[full_prompt, input_image],
        config=GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"],
            candidate_count=1,
        ),
    )

    output_image = input_image

    for part in response.candidates[0].content.parts:
        if getattr(part, "text", None) is not None:
            print("[gemini] text:", part.text)
        elif getattr(part, "inline_data", None) is not None:
            output_image = Image.open(BytesIO(part.inline_data.data)).convert("RGB")
            output_image.save(inpainted_output_path)
            print(f"[gemini] Inpainted image saved to {inpainted_output_path}")
            break

    return output_image

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
