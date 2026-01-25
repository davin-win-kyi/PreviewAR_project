import numpy as np
from PIL import Image


def combine_images_with_mask(
    original_path: str,
    mask_path: str,
    output_path: str,
    resize_mask: bool = True,
    white_threshold: int = 250,
) -> str:
    """
    Creates an image that:
      - Keeps ALL non-white pixels from the original image.
      - On original white background pixels:
          * if mask is white ( > 0 ) -> set to pure white (255,255,255)
          * if mask is black ( == 0 ) -> set to pure black (0,0,0)

    So:
      - The couch / target object (non-white) stays unchanged.
      - The white background becomes a black/white mask, indicating where to inpaint.

    Args:
        original_path: Path to original RGB/RGBA image with white background.
        mask_path: Path to mask image (white where you want the inpaint region).
        output_path: Where to save result.
        resize_mask: Resize mask to match original size if needed.
        white_threshold: Pixels above this on all channels are treated as “white”.

    Returns:
        output_path
    """
    try:
        # Load original as RGB
        original = Image.open(original_path).convert("RGB")
        orig_size = original.size

        # Load mask as grayscale
        mask = Image.open(mask_path).convert("L")
        if resize_mask and mask.size != orig_size:
            mask = mask.resize(orig_size, Image.NEAREST)
            print(f"[combine] Resized mask to {orig_size}")

        # Convert to numpy
        orig_np = np.array(original)           # (H, W, 3)
        mask_np = np.array(mask)               # (H, W)

        # Identify white background in original (within a threshold)
        white_bg = (
            (orig_np[:, :, 0] >= white_threshold) &
            (orig_np[:, :, 1] >= white_threshold) &
            (orig_np[:, :, 2] >= white_threshold)
        )

        # Mask “on” where mask image is non-zero (white)
        mask_on = mask_np > 0

        # Start with original image
        out = orig_np.copy()

        # On white background:
        #  - where mask_on -> white
        #  - where not mask_on -> black
        bg_and_mask = white_bg & mask_on
        bg_and_not_mask = white_bg & ~mask_on

        out[bg_and_mask] = [255, 255, 255]
        out[bg_and_not_mask] = [0, 0, 0]

        # Save
        out_img = Image.fromarray(out, mode="RGB")
        out_img.save(output_path)
        print(f"[combine] Combined original+mask image saved to {output_path}")
        return output_path

    except FileNotFoundError:
        raise FileNotFoundError("Error: One of the files was not found.")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred in combine_images_with_mask: {e}")
        