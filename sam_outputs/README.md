# sam_outputs

Scratch output from [Image_Processing/](../Image_Processing/) for the most recent photo it segmented. Every product in stage 1 overwrites it, so it's for inspecting the last segmentation, not a record of past runs. [Image_Processing/README.md](../Image_Processing/README.md#output) describes each file.

```
sam_outputs/
├── positive_prompt/    detections for the product
├── negative_prompt/    detections for objects covering it
└── targets/
    ├── impainting_index.json
    └── target_NNN/     crop, masks, and the composite sent to Gemini
```

The folder is never cleared. If an earlier photo had more targets than the current one, its extra `target_NNN/` folders stay behind, and only `targets/impainting_index.json` shows which targets are current. The checked-in index lists one target (`target_000`); `target_001` to `target_007` are left over from an earlier photo.
