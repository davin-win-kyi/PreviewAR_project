# webscrape_output

The input to the batch pipeline. Stage 1 (`preprocess_main.py`) reads `all_products.json`; pass `--input-json` to use a different file. See the [main README](../README.md) for the full pipeline.

The web scraper that builds this folder isn't included in this repository. **If you'd like to use it, email [davinkyi@gmail.com](mailto:davinkyi@gmail.com).** Otherwise, produce the folder with your own scraper using the format below.

The checked-in file contains a single test product (`bed#1`).

## Layout

```
webscrape_output/
├── all_products.json       every product, in pipeline order
└── <name>/                 one folder per product, e.g. bed#1/
    ├── product.json        that product's entry
    └── images/
        └── image_01.jpg    downloaded listing photos
```

The pipeline reads only `all_products.json` and the photos it points to.

## `all_products.json`

```json
{
  "products": [
    {
      "url": "https://www.example.com/products/123",
      "object_type": "couch",
      "site": "amazon",
      "name": "couch#1",
      "standardized_dimensions": { "l": 84.0, "w": 35.0, "h": 33.5 },
      "images": [
        {
          "url": "https://www.example.com/images/123-1.jpg",
          "local_path": "webscrape_output/couch#1/images/image_01.jpg"
        }
      ],
      "runtime_seconds": 10.65
    }
  ],
  "average_runtime_seconds": 10.65
}
```

A top-level JSON list of products also works.

| Field | Used for |
|---|---|
| `url` | Identifies the product in later outputs. |
| `object_type` | The product category. It's the target for photo ranking, the default class name for segmentation prompts, and part of every output file name. |
| `standardized_dimensions` | Listed dimensions **in inches**, as `l`/`w`/`h` or `length`/`width`/`height`. Carried through to `mesh_outputs/meshes.json` and used as ground truth in stage 3. |
| `images[].local_path` | Photo files to rank, relative to the repository root or absolute. Entries whose file is missing are skipped. |
| `images[].url` | The photo's original URL (not used). |
| `site` | Source website, carried through to later outputs. |
| `name` | Not used: stage 1 recomputes names as `<object_type>#<n>` from the order of `products`. |
| `product_names` | Optional list. If present, its first entry replaces `object_type` as the class name for prompt writing. |
| `runtime_seconds`, `average_runtime_seconds` | Scraping time (not used). |

Products are numbered within their category in the order they appear. Appending products is safe. Reordering or removing products renumbers the ones after them, and stages 1 and 2 will then match existing outputs to the wrong product.
