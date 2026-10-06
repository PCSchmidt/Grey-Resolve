# Third-Party Notices

Grey-Resolve is a research prototype. Pretrained model weights are **not**
redistributed in this repository; they are downloaded at runtime from their
original sources and remain subject to their own terms.

## Dependencies and model weights

| Component | What we use | Code license | Weights terms | Notes |
|---|---|---|---|---|
| InsightFace (`buffalo_l`, w600k_r50 ArcFace) | Embedding backbone (primary) | MIT | Non-commercial (explicit grant in `MODEL.LICENSE`) | Weights downloaded at runtime; never committed |
| AdaFace (IR-50 WebFace4M) | Backbone fallback | MIT | Follows training-data terms (WebFace600M: research only) | BGR input convention |
| MagFace | FIQA (feature magnitude) | Apache-2.0 | Silent on weights | Quality score only |
| CR-FIQA | Cited benchmark baseline only | none published | none published | Do not vendor code or weights |
| FAISS (`faiss-cpu`) | Vector index (HNSW) | MIT (Patent grant) | n/a | |
| FastAPI / Pydantic / Uvicorn | REST API | MIT / MIT / BSD-3-Clause | n/a | |
| NumPy / PyYAML | Core | BSD-3-Clause / MIT | n/a | |

Full license analysis with citations: [`docs/BACKBONE_LICENSES.md`](docs/BACKBONE_LICENSES.md).

## Data

No datasets are included in this repository. The earlier coursework dataset
(LFW-derived, research-only) is never redistributed here; loaders expect
user-supplied local data, and the Phase 2 scenario dataset is fully synthetic.

## Non-commercial notice

Several model weights above are licensed for non-commercial research use only.
Using this repository's code with those weights is subject to those terms.
For commercial use, obtain a license from the respective owners (InsightFace
recognition-oss-pack / MSU for AdaFace).
