# Image_Processing

Grounded-SAM 2 segmentation for stage 1. Given the chosen photo and the prompts from `WebScrape_Processing/prompt.py`, it finds the product and the objects covering it, then builds the composite image that [Impaint_Processing/](../Impaint_Processing/) sends to Gemini. See the [main README](../README.md) for the full pipeline.

## Usage

```python
from Image_Processing.Grounded_SAM2_Config import GroundedSAM2Config, build_models
from Image_Processing.Get_Impainting_Images import main as impaint_main

device, sam2_predictor, processor, grounding_model = build_models(
    GroundedSAM2Config(sam2_checkpoint="grounded_sam2/checkpoints/sam2.1_hiera_large.pt")
)

index_json = impaint_main(
    image_path="photo.jpg",
    positive_prompt="couch.",
    negative_prompt="pillow. blanket.",
    device=device,
    sam2_predictor=sam2_predictor,
    processor=processor,
    grounding_model=grounding_model,
)
```

`impaint_main` returns the path of `sam_outputs/targets/impainting_index.json`, and stage 1 uses the first target's `final_composite`. If you don't pass prebuilt models, `impaint_main` loads them for that call and frees them afterwards. `preprocess_main.py` builds them once and reuses them for every product.

## How it works

1. **Load models** (`Grounded_SAM2_Config.py`). `build_models` loads Grounding DINO (`IDEA-Research/grounding-dino-tiny` from Hugging Face) and SAM 2.1 Hiera-L (`sam2.1_hiera_large.pt` with `configs/sam2.1/sam2.1_hiera_l.yaml`). It also turns on bfloat16 autocast for the whole process and enables TF32 on Ampere and newer GPUs.
2. **Detect and segment** (`Grounded_SAM2_Run.py`). `run_pos_neg_and_write` runs Grounding DINO (box threshold 0.4, text threshold 0.3) and then SAM 2 on the detected boxes, once for the positive prompt and once for the negative prompt.
3. **Build composites** (`Generate_Impainting_Images.py`). For each positive detection (a "target"), `generate_impainting_images`:
   - crops the photo to the target's box, plus `crop_pad` pixels;
   - builds the **inpaint mask** from the negative boxes inside the crop, each shrunk by 12 px, keeping only the part within the target's mask dilated by 20 px, so that only objects on the product are marked;
   - writes the **final composite**: original pixels inside the target mask but outside the inpaint mask, white inside the inpaint mask, and black everywhere else.

## Output

Each call writes to [sam_outputs/](../sam_outputs/):

```
sam_outputs/
├── positive_prompt/            bbox.jpg, segmented_overlay.jpg, results.json (boxes, scores, RLE masks)
├── negative_prompt/            the same, for the negative prompt
└── targets/
    ├── impainting_index.json   the targets and their file paths
    └── target_000/
        ├── crop.jpg            photo cropped to the target
        ├── positive_mask.png   target mask
        ├── inpaint_mask.png    area for Gemini to fill
        ├── final_mask.png      target mask minus inpaint mask
        ├── final_composite.png image sent to Gemini
        ├── preview_whited.png  crop with the inpaint area painted white
        ├── final_overlay.jpg   final mask drawn over the crop
        └── target.json         crop box, prompts, RLE masks, pixel counts
```

## Notes

- SAM 2 is imported as `grounded_sam2.sam2`, so run from the repository root with the edited `grounded_sam2/` described in the [main README's setup section](../README.md#setup).
- The default `sam2_checkpoint` is an absolute path on the original development machine; pass your own.
- `force_cpu=True` runs without a GPU, slowly.
- `requirements.txt` lists this folder's dependencies.
