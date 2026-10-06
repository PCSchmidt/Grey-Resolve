# Grey-Resolve — Face Embedding Backbone: License & Public-Use Comparison

**Date of review:** 2026-10-06
**Scope:** Licensing and public-use terms of four candidate pretrained face-recognition / face-quality (FIQA) ecosystems, for a **public GitHub portfolio repo** ("defense-framed" research prototype, synthetic/public data only, non-commercial).
**Method:** LICENSE files, README license sections, model-release pages, and maintainer statements were fetched directly from the upstream GitHub repositories (raw files + GitHub API) and Hugging Face model cards. Every claim below cites a URL. This is a practical license review, not legal advice.

> **Naming note:** two candidate URLs given in the task do not exist. The official repos are
> **AdaFace → `github.com/mk-minchul/AdaFace`** (not `minsuk00/AdaFace` or `vyra/AdaFace`) and
> **CR-FIQA → `github.com/fdbtrs/CR-FIQA`** (not `tensorboy/CR-FIQA` or `minivits/crfiqa`).

---

## Summary table

| Ecosystem | (a) Code license | (b) Pretrained weights terms | (c) Commercial use | (d) Public portfolio repo (non-commercial, synthetic/public data) | (e) Distribution |
|---|---|---|---|---|---|
| **InsightFace** (buffalo_l / antelopev2 / w600k_r50) | **MIT** (stated in README; no root LICENSE file) | **Non-commercial research only** — explicit, signed machine-readable `MODEL.LICENSE` (`"grant": "non-commercial"`); manual + auto downloads both covered | Code: **yes**. Weights: **no**, unless licensed separately (contact `recognition-oss-pack@insightface.ai`) | **Low risk** — explicit grant covers non-commercial research; do not commit weights; document terms in repo | GitHub Releases ("model-zoo", e.g. `buffalo_l.zip` ~289 MB), auto-download via `pip install insightface`; ONNX; **no registration wall** |
| **AdaFace** (`mk-minchul/AdaFace`) | **MIT** (LICENSE, author-confirmed) | **No weight license file.** HF cards: *"follow the license of the training dataset"* (WebFace4M/12M, MS1MV2, CASIA-WebFace, VGGFace2 — all research-oriented). Maintainer redirects commercial questions to **MSU Technologies** | Code: **yes**. Weights: **no** without MSU license | **Low–moderate risk** — fine for a non-commercial demo if weights are not committed and dataset terms are documented | `.pth` checkpoints on Google Drive + Hugging Face (`minchul/cvlface_adaface_*`); **no registration wall**; not pip-installable |
| **MagFace** (`IrvingMeng/MagFace`) | **Apache-2.0** (LICENSE) | **No weight license statement.** Weights on Google Drive/Baidu, trained on **MS1MV2 / CASIA-WebFace** → inherit research-only dataset restrictions | Code: **yes**. Weights: **no** (dataset-derived) | **Low–moderate risk** — Apache code is safe to vendor/reuse with attribution; treat weights as research-only | Google Drive + Baidu (Baidu needs CN account); PyTorch checkpoints; no pip; no official ONNX |
| **CR-FIQA** (`fdbtrs/CR-FIQA`) | **NONE** (no LICENSE file; GitHub: "No license" → all rights reserved) | **No weight license.** Google Drive weights, trained on **MS1MV2 (L) / CASIA-WebFace (S)**; README says to follow InsightFace dataset licenses | Code: **no legal grant at all**. Weights: **no** | **Do not vendor code or weights** — highest risk of the four. Local-only use is common academic practice but unlicensed | Google Drive folders; PyTorch 1.7-era research code; no pip; no ONNX |

**Headline finding:** none of the four ecosystems offers commercially-usable pretrained weights. All released weights trace back to research-only face datasets (WebFace600K/WebFace4M, MS1MV2, CASIA-WebFace, VGGFace2). The differentiator is *clarity of terms*: InsightFace is the only one with an explicit, signed, machine-readable non-commercial grant. AdaFace and MagFace are informal-but-usable for research; CR-FIQA's code has no license at all and must not be copied into a public repo.

---

## 1. InsightFace / ArcFace model zoo (buffalo_l, antelopev2, w600k_r50)

### (a) Code license
- **MIT.** Root README "License" section: *"The code of InsightFace is released under the MIT License. There is no limitation for both academic and commercial usage."*
  - https://github.com/deepinsight/insightface/blob/master/README.md#license
  - Repeated in the Python-package model guide: https://github.com/deepinsight/insightface/blob/master/python-package/docs/model_zoo.md#model-licenses
- Caveat: there is **no root LICENSE file** (GitHub license detection reports none); the README statement is the license grant. Subdirectories carry their own files (e.g. `detection/scrfd/LICENSE`).

### (b) Pretrained weights license / terms
- **Non-commercial research only.** Model zoo banner: *"ALL models are available for non-commercial research purposes only."*
  - https://github.com/deepinsight/insightface/blob/master/model_zoo/README.md
- Root README: *"The training data containing the annotation (and the models trained with these data) are available for non-commercial research purposes only. Both manual-downloading models from our github repo and auto-downloading models with our python-library follow the above license policy (which is for non-commercial research purposes only)."*
  - https://github.com/deepinsight/insightface/blob/master/README.md#license
- The repo now ships **signed, machine-readable model license manifests**. `buffalo_l`'s `MODEL.LICENSE` is a JSON with `"issuer": "InsightFace"`, `"license_id": "buffalo_l-public-v1"`, **`"grant": "non-commercial"`**, plus a signature. Same pattern for `antelopev2`, `buffalo_m/s/sc`, `raccoon_s/l`:
  - https://github.com/deepinsight/insightface/blob/master/server/backend/insightface_server/licensing/defaults/buffalo_l/MODEL.LICENSE
  - https://github.com/deepinsight/insightface/blob/master/server/backend/insightface_server/licensing/defaults/antelopev2/MODEL.LICENSE
- Note on "InsightFace NonCommercial" / CC BY-NC-SA wording: community mirrors sometimes label these weights CC BY-NC-SA 4.0 or "InsightFace NonCommercial". The **authoritative current terms** are the repo README + the signed `MODEL.LICENSE` manifests above, which grant **non-commercial** use. Treat any other label on third-party mirrors as unofficial.
- `2025-11-24` README update: *"For open-sourced face recognition models (e.g., buffalo_l package), please contact `recognition-oss-pack@insightface.ai` for licensing."* — i.e., commercial or other licensing is handled case-by-case by InsightFace.
  - https://github.com/deepinsight/insightface/blob/master/README.md#license

### (c) Commercial use allowed?
- **Code: yes (MIT).** **Weights: no** — non-commercial research only, unless InsightFace grants a separate license (email above).

### (d) Public GitHub portfolio repo — risk assessment
- **Low risk for the planned non-commercial research-prototype use**, provided:
  - weights are **not committed** to the repo (download at runtime / document the upstream link),
  - the repo states the weights are third-party, **non-commercial research only**, and links to the upstream license text,
  - demo data is synthetic/public, and the "defense" framing is clearly fictional and non-operational.
- The intended use (public, non-commercial, research, synthetic/public data) sits squarely inside the explicit non-commercial grant — this is the *only* candidate where that is spelled out by the rights holder.

### (e) Practical notes
- `pip install -U insightface` (PyPI package `insightface`; PyPI metadata carries no license classifier, README carries the MIT statement): https://pypi.org/project/insightface/
- Weights auto-download from the GitHub release **"InsightFace Model Zoo"** into `~/.insightface/models/<name>/` — e.g. `buffalo_l.zip` (288 MB), `antelopev2.zip`, `raccoon_s/l`, etc. No registration wall, no Google Drive.
  - https://github.com/deepinsight/insightface/releases/tag/model-zoo
- `buffalo_l` = SCRFD-10GF detector + **ResNet50@WebFace600K** recognition model (the "w600k_r50" ArcFace backbone) + 2D/3D alignment + gender/age. Accuracy per model zoo: IJB-C(E4) 97.25, LFW 99.83.
  - https://github.com/deepinsight/insightface/blob/master/model_zoo/README.md
- All models are **ONNX**, run on onnxruntime (CPU/CUDA/CoreML) → easiest integration path.

---

## 2. AdaFace (`mk-minchul/AdaFace`, CVPR 2022)

### (a) Code license
- **MIT**, LICENSE file: *"Copyright (c) 2022 Minchul Kim"*.
  - https://github.com/mk-minchul/AdaFace/blob/master/LICENSE
- Author-confirmed: issue #4 ("About the license for this model") — *"The license file has been added and it is MIT license."*
  - https://github.com/mk-minchul/AdaFace/issues/4

### (b) Pretrained weights license / terms
- **No license file covers the weights.** Weights are hosted externally (Google Drive in the README "Pretrained Models" table: R18/R50/R100 × CASIA-WebFace / VGGFace2 / WebFace4M / MS1MV2).
  - https://github.com/mk-minchul/AdaFace#pretrained-models
- The maintainer's Hugging Face model cards (`minchul/cvlface_adaface_*`) state the operative rule: *"Please cite the original paper and follow the license of the training dataset."* The HF cards carry **no license tag**.
  - https://huggingface.co/minchul/cvlface_adaface_ir50_webface4m
- The maintainer declines to grant commercial rights himself; issues #174 ("Commercial use of pretrained weights") and #175 both get the reply: *"As this research is conducted at Michigan State University, MSU technologies (https://innovationcenter.msu.edu/contact/) will be able to answer your questions on its commercial usage."*
  - https://github.com/mk-minchul/AdaFace/issues/174
  - https://github.com/mk-minchul/AdaFace/issues/175
- Net effect: weights inherit **research-only / non-commercial** terms of the training datasets (WebFace4M/WebFace12M and MS1MV2 come from the InsightFace data zoo — non-commercial research; CASIA-WebFace and VGGFace2 are research-only), and **commercial rights are controlled by MSU**.

### (c) Commercial use allowed?
- **Code: yes (MIT).** **Weights: not without a license from MSU Technologies**; "frozen feature extractor in a commercial product" was explicitly not granted in issue #174.

### (d) Public GitHub portfolio repo — risk assessment
- **Low–moderate risk** for a non-commercial research demo: the academic norm covers it, and the maintainer's stated rule ("follow the license of the training dataset") is documentable. But the terms are **informal and implicit** — nothing grants research use in as many words as InsightFace does. Keep weights out of the repo, cite the paper, document the dataset-terms chain.

### (e) Practical notes
- Checkpoints (`.pth`) on **Google Drive** (README table) and **Hugging Face** `minchul/cvlface_adaface_*` (IR18/IR50/IR101/ViT × casia/vgg2/webface4m/webface12m/ms1mv2/ms1mv3) — no registration wall, `hf_hub_download` works.
  - https://huggingface.co/models?search=adaface
- Not pip-installable as a standalone package (clone the repo; the related `mk-minchul/CVLface` toolkit loads the HF checkpoints).
- **BGR input convention** (unlike InsightFace's RGB) — a real integration gotcha: README "Pretrained Models" section.
- ONNX export: no official export, but community converters exist (e.g. `yakhyo/adaface-onnx`, MIT): https://github.com/yakhyo/adaface-onnx
- Quality: strongest of the candidates on **low-quality/degraded imagery** (IJB-S, TinyFace); e.g. R100/WebFace4M IJB-C TAR@FAR=1e-4 = 97.66.

---

## 3. MagFace (`IrvingMeng/MagFace`, CVPR 2021 Oral)

### (a) Code license
- **Apache-2.0** (LICENSE file in repo; GitHub detects `Apache-2.0`).
  - https://github.com/IrvingMeng/MagFace/blob/master/LICENSE
- Note the README disclaimer: the repo is an *"official but abridged version"* of a private codebase.

### (b) Pretrained weights license / terms
- **No weight license statement anywhere in the README.** Weights are in the "Model Zoo" table on Google Drive / Baidu: iResNet100/50 trained on **MS1MV2**, iResNet18 trained on **CASIA-WebFace**.
  - https://github.com/IrvingMeng/MagFace#model-zoo
- The Apache-2.0 grant covers the repo ("the Work"); externally hosted checkpoints are not clearly covered. Weights are trained on InsightFace-distributed MS1MV2 / CASIA-WebFace, which are **non-commercial research only** (InsightFace policy link in §1), so treat the weights as **research-only**.

### (c) Commercial use allowed?
- **Code: yes (Apache-2.0).** **Weights: effectively no** (no grant from the authors + research-only training data).

### (d) Public GitHub portfolio repo — risk assessment
- **Low–moderate risk**, same profile as AdaFace, but the **Apache-2.0 code is the safest of the four to copy/adapt** (with attribution + NOTICE). Do not commit checkpoints; document that the quality scorer is a research-only third-party model.

### (e) Practical notes
- Distribution: Google Drive (Baidu mirrors require a Chinese account); PyTorch checkpoints; toy inference notebook `inference/examples.ipynb`; no pip package; no official ONNX export.
- **Dual use:** embedding *and* face-quality score (feature magnitude is the paper's quality measure) → one network can serve both the backbone and the FIQA role.
- IJB-B/IJB-C eval data requires applying to NIST first (README).

---

## 4. CR-FIQA (`fdbtrs/CR-FIQA`, CVPR 2023)

### (a) Code license
- **None.** The repository has **no LICENSE file**; GitHub reports "No license" → default **all rights reserved**. The code is not open source in the legal sense and **cannot legally be copied or redistributed** in a public repo.
  - https://github.com/fdbtrs/CR-FIQA (repo root contains only `README.md`, code, configs — no license file)

### (b) Pretrained weights license / terms
- **No weight license.** CR-FIQA(L) and CR-FIQA(S) checkpoints are Google Drive folders linked from the README.
  - https://github.com/fdbtrs/CR-FIQA#pretrained-model
- Training data: MS1MV2 (CR-FIQA(L)) and CASIA-WebFace (CR-FIQA(S)); the README itself says to download them *"on strictly follow the licence distribution"* from InsightFace — so weights inherit **non-commercial research-only** dataset terms.
- Some community forks (e.g. `songuyenerza/CR-FIQA_Tuning`) add MIT headers, but that only covers the fork's changes, not the original code or weights.

### (c) Commercial use allowed?
- **No.** There is no license grant for code or weights at all.

### (d) Public GitHub portfolio repo — risk assessment
- **Do not vendor its code or weights.** Highest-risk item of the four. Referencing its published results in docs is fine; running it locally as an unredistributed evaluation dependency is common academic practice but remains unlicensed. For anything shipped in the repo, **reimplement from the paper** (https://arxiv.org/abs/2112.06592) or use a properly-licensed FIQA.

### (e) Practical notes
- PyTorch 1.7-era research code, Google Drive weights, no pip, no ONNX. Quality is the best published option (README: ranked 1st/2nd at NIST FATE quality, https://pages.nist.gov/frvt/html/frvt_quality.html), which is why it keeps appearing in comparisons despite the license hole.

---

## Cross-cutting rules for the Grey-Resolve public repo

1. **Never commit model weights** to the repo. Add a `scripts/download_models.py` (or docs instructions) that fetches them at runtime and prints the upstream license notice.
2. Add **`THIRD_PARTY_NOTICES.md`** quoting the weight terms and linking the URLs above, with a retrieval date (2026-10-06).
3. License **our code** permissively (MIT or Apache-2.0) but state clearly that third-party pretrained models are **non-commercial research only** and not covered by our code license.
4. Keep the demo data **synthetic/public** and the "defense" framing explicitly fictional; avoid any wording that implies operational capability or endorsement.
5. **If the project ever goes commercial:** obtain a license from InsightFace (`recognition-oss-pack@insightface.ai`) and/or MSU (AdaFace), or retrain an embedding on truly synthetic data. Note that even DigiFace-1M (synthetic) is R-UDA **non-commercial** (https://github.com/microsoft/DigiFace1M/blob/main/LICENSE), so "retrain on synthetic data" still needs a permissive synthetic corpus check.

---

## Recommendation for Grey-Resolve Phase 1

Priority order applied: **weights license clarity → model quality → ease of use**. Since no candidate has commercially-permissive weights, priority 1 becomes "clearest terms that unambiguously allow our stated non-commercial public-repo use".

### Primary backbone: **InsightFace `buffalo_l` (ArcFace R50, WebFace600K = w600k_r50)**
- **Why (priority 1):** the only ecosystem with an explicit, signed, machine-readable non-commercial grant (`MODEL.LICENSE`, `"grant": "non-commercial"`) that directly covers a public non-commercial research repo. Predictable, self-service terms — no informal inference from dataset licenses.
- **Why (priority 3):** pip-installable, ONNX-native, detector (SCRFD) + alignment + embeddings bundled → fastest path to a working Phase-1 pipeline.
- **Quality (priority 2):** strong (IJB-C(E4) 97.25, LFW 99.83) — adequate for Phase 1.
- **Conditions:** weights downloaded at runtime only; repo + notices state non-commercial research use; cite upstream.

### Fallback / low-quality upgrade: **AdaFace (IR-50 WebFace4M; IR-101 if compute allows)**
- MIT code; **best of the candidates on degraded/low-quality imagery** (IJB-S, TinyFace), which matters for a defense-framed scenario. Use if Phase-1 evaluation on degraded imagery shows a gap versus buffalo_l, or as an A/B second backbone.
- Weights terms are informal (dataset license + MSU controls commercial use) — kept as fallback, not primary. Source: HF `minchul/cvlface_adaface_ir50_webface4m` (or the Google Drive links in the README); respect the BGR input convention.

### FIQA: **MagFace feature magnitude (primary), AdaFace feature-norm (zero-cost fallback)**
- **Primary:** MagFace quality score — **Apache-2.0 code is the cleanest license for code reuse** of all four; feature magnitude is a well-established face-quality proxy and it doubles as a second embedding if needed. Weights: research-only (MS1MV2), downloaded at runtime, not committed.
- **Fallback:** AdaFace's feature-norm quality proxy — costs nothing extra if AdaFace is already in the pipeline (it is the paper's own quality approximation).
- **Excluded:** CR-FIQA — best published accuracy, but its repo has **no license**; do not ship its code or weights. Cite it as the benchmark reference only; reimplement from the paper (arXiv 2112.06592) if CR-FIQA-level quality becomes a hard requirement, and prefer reaching out to the author (`fdbtrs`) first.

### One-line summary
Ship Phase 1 on **InsightFace buffalo_l (explicit non-commercial weights grant, pip + ONNX)**, keep **AdaFace** as the quality-focused fallback for degraded imagery, use **MagFace magnitude (Apache-2.0 code)** for face quality, and stay away from **CR-FIQA** code/weights (unlicensed) — with all weights downloaded at runtime and full third-party notices in the repo.

---

## Sources

**InsightFace**
- https://github.com/deepinsight/insightface/blob/master/README.md (License section)
- https://github.com/deepinsight/insightface/blob/master/model_zoo/README.md (non-commercial banner, model pack table)
- https://github.com/deepinsight/insightface/blob/master/python-package/docs/model_zoo.md (model licenses)
- https://github.com/deepinsight/insightface/blob/master/server/backend/insightface_server/licensing/defaults/buffalo_l/MODEL.LICENSE
- https://github.com/deepinsight/insightface/blob/master/server/backend/insightface_server/licensing/defaults/antelopev2/MODEL.LICENSE
- https://github.com/deepinsight/insightface/releases/tag/model-zoo (weight distribution)
- https://pypi.org/project/insightface/

**AdaFace**
- https://github.com/mk-minchul/AdaFace/blob/master/LICENSE (MIT)
- https://github.com/mk-minchul/AdaFace#pretrained-models (Google Drive weight table)
- https://huggingface.co/minchul/cvlface_adaface_ir50_webface4m ("follow the license of the training dataset")
- https://github.com/mk-minchul/AdaFace/issues/4 (MIT confirmation)
- https://github.com/mk-minchul/AdaFace/issues/174 (commercial use → MSU)
- https://github.com/mk-minchul/AdaFace/issues/175 (commercial use → MSU)
- https://github.com/yakhyo/adaface-onnx (community ONNX, MIT)

**MagFace**
- https://github.com/IrvingMeng/MagFace/blob/master/LICENSE (Apache-2.0)
- https://github.com/IrvingMeng/MagFace#model-zoo (weights distribution, training data)

**CR-FIQA**
- https://github.com/fdbtrs/CR-FIQA (README; no LICENSE file present)
- https://github.com/fdbtrs/CR-FIQA#pretrained-model (Google Drive weights)
- https://arxiv.org/abs/2112.06592 (paper)
- https://pages.nist.gov/frvt/html/frvt_quality.html (NIST FATE quality ranking)

**Cross-cutting**
- https://github.com/deepinsight/insightface/blob/master/recognition/_datasets_/README.md (training datasets)
- https://github.com/microsoft/DigiFace1M/blob/main/LICENSE (R-UDA non-commercial; synthetic data note)
