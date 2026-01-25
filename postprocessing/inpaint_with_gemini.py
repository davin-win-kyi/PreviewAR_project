import os
import uuid
from io import BytesIO
from typing import List

from PIL import Image
from google import genai
from google.genai.types import GenerateContentConfig

from ..utils.text_utils import format_object_list
from ..utils.mask_utils import combine_images_with_mask


# Default uploads directory
UPLOADS_DIR = "uploads"
os.makedirs(UPLOADS_DIR, exist_ok=True)


def build_inpaint_prompt(target_class: str, classes_in_crop: List[str]) -> str:
    """
    Builds a structured inpainting prompt like:

    "can you fill the mask with the same texture and fabric as the rest of the <target_object>
     with only the <target_object> and no white gaps or holes in the <target_object>.
     You must not include the following objects in the image <other_object#1>, ..., <other_object#n>."
    """
    other_objects = sorted({
        c for c in classes_in_crop
        if c.lower() != target_class.lower()
    })

    if other_objects:
        others_str = format_object_list(other_objects=other_objects)
        prompt = (
            f"can you fill the mask with the same texture and fabric as the rest of the {target_class} "
            f"with only the {target_class} and no white gaps or holes in the {target_class}. "
            f"Also make sure to not include {others_str}."
        )
    else:
        prompt = (
            f"Can you fill the mask with the same texture and fabric as the rest of the {target_class} "
            f"with only the {target_class} and no white gaps or holes in the {target_class}. "
            f"There are no other objects that should be included."
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
        model="gemini-2.5-flash-image-preview",
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
            image.save(inpainted_output_path)
            print(f"[gemini] Inpainted image saved to {inpainted_output_path}")

    return image


def inpaint_image_with_mask(
    original_image_path: str,
    mask_image_path: str,
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
        mask_path=mask_image_path,
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
    mask_image_path
):  
    # Build prompt
    prompt = build_inpaint_prompt(target_class, classes_in_crop)
    print("[summary] inpaint prompt:", prompt)

    combined_output_path = os.path.join(UPLOADS_DIR, "combined_manual_mask.png")
    inpainted_output_path = os.path.join(UPLOADS_DIR, "inpainted_manual_mask.png")

    combined_path, inpainted_path = inpaint_image_with_mask(
        original_image_path=original_image_path,
        mask_image_path=mask_image_path,
        prompt=prompt,
        combined_output_path=combined_output_path,
        inpainted_output_path=inpainted_output_path,
    )

    return {
        "target_class": target_class,
        "classes_in_crop": classes_in_crop,
        "prompt": prompt,
        "combined_image_path": combined_path,
        "inpainted_image_path": inpainted_path,
    }
