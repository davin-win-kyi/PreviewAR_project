#!/usr/bin/env python3
"""
Nano-banana: Full inpainting pipeline using manual masks.

This script orchestrates the complete inpainting workflow by:
1. Running MaskGenerationRunner to detect target objects
2. Building intelligent prompts based on detected objects
3. Using the modularized inpainting functions to generate the final result
"""
from dotenv import load_dotenv

# Import MaskGenerationRunner

# Import modularized inpainting functions
from Impaint_Processing.inpaint_with_gemini import get_inpainted_image_path

# -------------------------------------------------------------------
# Environment setup
# -------------------------------------------------------------------
load_dotenv()


# -------------------------------------------------------------------
# High-level: use original + manual mask, plus MaskGenerationRunner for prompt
# -------------------------------------------------------------------
def run_full_inpaint_with_manual_mask(
    original_image_path: str,
    product_url: str,
    conf: float = 0.25,
    iou: float = 0.45,
    use_gpt: bool = True,
    adjustment_factor: int = -15,
) -> dict:
    """
    Full end-to-end inpainting pipeline:
      1) Detect target_class and classes_in_crop (currently hardcoded)
      2) Build the structured prompt from those classes
      3) Send the original image directly to Gemini for inpainting
      4) Return paths + prompt
    
    Args:
        original_image_path: Path to the original image
        product_url: URL of the product (for future MaskGenerationRunner integration)
        conf: Confidence threshold for detection
        iou: IOU threshold for detection
        use_gpt: Whether to use GPT in MaskGenerationRunner
        adjustment_factor: Adjustment factor for MaskGenerationRunner
    
    Returns:
        Dictionary containing:
            - target_class: The detected target object class
            - classes_in_crop: List of all detected classes in the crop
            - prompt: The generated inpainting prompt
            - inpainted_image_path: Path to the final inpainted image
    """
    print("[nano-banana] Running MaskGenerationRunner...")

    target_class = "couch" 
    classes_in_crop = ["pillow"] 

    if not target_class:
        raise RuntimeError("MaskGenerationRunner did not return a target_class.")

    print(f"[nano-banana] target_class: {target_class}")
    print(f"[nano-banana] classes_in_crop: {classes_in_crop}")

    # 2-4) Use the modularized inpainting pipeline
    print("[nano-banana] Starting inpainting pipeline...")
    result = get_inpainted_image_path(
        target_class=target_class,
        classes_in_crop=classes_in_crop,
        original_image_path=original_image_path,
    )

    return result


# -------------------------------------------------------------------
# Main: set variables and run
# -------------------------------------------------------------------
if __name__ == "__main__":
    # === SET THESE VARIABLES ===
    ORIGINAL_IMAGE_PATH = "/Users/calebhu/Documents/Research/PreviewAR_project/image.png"
    PRODUCT_URL = (
        "https://www.amazon.com/MODNEST-Modular-Sectional-Boneless-Assembly/dp/"
        "B0FCYC86TT/ref=sr_1_5_sspa?crid=3RIV4C6CTWYQA&dib=eyJ2IjoiMSJ9."
        "mCR5SjVr3IuoofQI97UxVmePO3nKmQbrqGkH6q7BhCWvXZfA2gaJDgyWsF100Jp3IznRSrEL8WKrwF2Xtlr-Q6YdVwugk_h-"
        "vhluo-EvhxJBqShb2gctTmjV71AXRyHOoPE6xC5K1iS8ITO3gdrhSf93HanYw7yk5iuIDU0gQFvfQLiPHo05ZX5PuYYk5As943eAeCxhe_d7i07UPtixVaCT_4yDty6lWukpMHvQmtssdgjG_"
        "zKvPqz8uqzZ5oAjIljfU3T1fj2lJKgKnqjlWxjA504G0RVwfRlQuUKr5bM.bAdATDJIW3OxMs6M16JiHRG9zAvYg7_B-SoRvPV4i5s&dib_tag=se&keywords=couch&qid=1762565138&"
        "sprefix=cou%2Caps%2C264&sr=8-5-spons&sp_csd=d2lkZ2V0TmFtZT1zcF9hdGY&th=1"
    )      # used by MaskGenerationRunner

    print("=" * 60)
    print("Starting Nano-Banana Inpainting Pipeline")
    print("=" * 60)

    result = run_full_inpaint_with_manual_mask(
        original_image_path=ORIGINAL_IMAGE_PATH,
        product_url=PRODUCT_URL,
        conf=0.25,
        iou=0.45,
        use_gpt=True,
        adjustment_factor=-15,
    )

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Target class: {result['target_class']}")
    print(f"Classes in crop: {result['classes_in_crop']}")
    print(f"\nPrompt used:\n{result['prompt']}")
    print(f"\nInpainted image: {result['inpainted_image_path']}")
    print("=" * 60)
