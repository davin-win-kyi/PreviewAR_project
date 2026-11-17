# WebScrape_Processing

Code that works with product listings before segmentation. It has two parts:

- **Stage 1 helpers**, used by `preprocess_main.py`: Gemini-based photo ranking and segmentation-prompt writing.
- **A single-URL scraper**, used by the older `main.py`: opens a product page in Chrome, extracts the product's details with GPT-5, and picks the best product photo.

See the [main README](../README.md) for how these fit into the pipeline.

The web scraper that builds the pipeline's input, `webscrape_output/all_products.json`, isn't in this folder or anywhere else in this repository. If you'd like to use it, email [davinkyi@gmail.com](mailto:davinkyi@gmail.com).

## Stage 1 helpers

| File | Function | What it does |
|---|---|---|
| `rank_images_from_local_paths.py` | `rank_images_from_local_paths(image_paths, target_object)` | Sends all of a product's photos to Gemini (`gemini-3.1-pro-preview`, temperature 0) in one request and returns `best_image_path`, `reasoning`, and per-photo `scores`. Photos are compared on strict priorities: shows the whole object rather than a part or accessory, then normal configuration, least obstruction, most visible faces, and fully in frame. Photos larger than 2048 px are downscaled first. |
| `prompt.py` | `infer_positive_and_negative_prompt(image_path, class_name)` | Asks Gemini (`gemini-3.1-flash-lite-preview`) for Grounding DINO prompts. `positive_prompt` is the class name (e.g. `"couch."`); `negative_prompt` lists only objects on, touching, or covering the product (e.g. `"pillow. blanket."`), or is empty. `image_path` can be a file path or a URL. |

If the Gemini call fails, `rank_images_from_local_paths` returns an empty `best_image_path` instead of raising, and `preprocess_main.py` marks the product as failed.

## Single-URL scraper

`get_best_image_url(url)` in `best_image_selector.py` runs the whole scrape:

1. `extract_url_info.extract_with_gpt5(url)` asks GPT-5 for the company name and product type names, using the page title and domain.
2. `generic_web_scraper` opens the page in Chrome with Selenium (getting past Amazon's bot-check page if it appears). It saves the page as `page.html` and a filtered copy that keeps mainly `img`, `span`, and `td` elements as `page_filtered.html`, both in the working directory, and has GPT-5 extract the title, description, attributes, image URLs, and candidate dimension strings.
3. GPT-5 then picks the length, width, and height from the candidate dimensions, expands the product names into aliases (e.g. couch → sofa), and ranks the image URLs.
4. The chosen photo and the full result are saved to `<output_dir>/best_image.png` and `<output_dir>/selection_result.json`. `output_dir` defaults to `outputs/` in the working directory.

Run it from the repository root:

```bash
python -m WebScrape_Processing.best_image_selector "<product URL>" --output-dir outputs
```

`webscrape_processing.py` wraps the same steps in `fetch_url_info_and_best_image(url)` for `main.py`, but it currently fails to import. It adds this folder to `sys.path` and imports `best_image_selector` as a top-level module, which breaks that file's relative imports (`from .extract_url_info import ...`).

## Other files

| Path | Contents |
|---|---|
| `outputs/` | `best_image.png` and `selection_result.json` from a sample scraper run |
| `page.html`, `page_filtered.html` | HTML saved by a scraper run from this folder |
| `requirements.txt` | Dependencies for this folder |

## Configuration

- `GEMINI_API_KEY` for the stage 1 helpers. Both create their Gemini client when imported, so the key must be set before importing them.
- `OPENAI_API_KEY` for the scraper.
- Google Chrome for Selenium. The scraper opens a visible browser window.
