# Modal MedSigLIP deployment

A Modal app that hosts `google/medsiglip-448` on an A10G GPU behind five HTTPS
endpoints. It exists so the API service can call MedSigLIP without downloading
6 GB of gated weights on every Render dyno and without needing a local GPU.

## Endpoints

| Route            | Method | Body                                                    | Purpose |
| ---------------- | ------ | ------------------------------------------------------- | --- |
| `/healthz`       | GET    | —                                                       | liveness |
| `/info`          | GET    | —                                                       | model card metadata, `embedding_dim=1152`, cold-start seconds |
| `/embed`         | POST   | `{"dicom_b64": "..."}` OR `{"pixels_b64": "..."}`       | 1×1152 embedding |
| `/embed_batch`   | POST   | `{"dicoms_b64": [...]}` OR `{"pixels_b64": [...]}` (≤32) | N×1152 embeddings |
| `/zero_shot`     | POST   | `{"dicom_b64"\|"pixels_b64": "...", "prompts": [...]}`  | SigLIP sigmoid probs per prompt |

`dicom_b64` triggers DICOM parsing on the server (pydicom Modality LUT +
percentile windowing + MONOCHROME1 invert). `pixels_b64` accepts PNG or JPEG
bytes that the server just opens with PIL and converts to RGB — no clinical
preprocessing.

## Client (repo-side)

Use the drop-in wrapper in `src/oncology_arbiter/models/medsiglip_modal_client.py`:

```python
from oncology_arbiter.models.medsiglip_modal_client import get_medsiglip_client

client = get_medsiglip_client()  # env-driven
res = client.run("path/to/mammogram.dcm")     # MedSigLipResult with sigmoid probs
emb = client.embed_dicom("path/to/scan.png")  # 1152-d vector
```

`get_medsiglip_client()` returns `MedSigLipModalClient` when
`MEDSIGLIP_BACKEND=modal`, otherwise it returns the local `MedSigLip` class.

## Environment variables

| Variable | Default | Purpose |
| --- | --- | --- |
| `MEDSIGLIP_BACKEND`         | `local` | `modal` selects the Modal client |
| `MODAL_MEDSIGLIP_URL`       | — (required for `modal`) | Base URL, e.g. `https://crispro-test--medsiglip` |
| `MEDSIGLIP_MODAL_TIMEOUT`   | `300`   | Per-call HTTP timeout in seconds |

## Deploy recipe

```bash
# One-time secret setup on Modal:
#   modal secret create medsiglip-hf-token HF_TOKEN=hf_...
export MODAL_PROFILE=crispro-test
python3 -m modal deploy deploy/modal/medsiglip_app.py

# If containers are holding old source after a code change, force a refresh:
python3 -m modal app stop -y medsiglip-448
python3 -m modal deploy deploy/modal/medsiglip_app.py
```

## Container timings (measured 2026-07-06 on A10G)

* Image build (cached): ~5s
* Cold-start (first request loads weights): ~88s (from `/info.load_seconds`)
* Warm single `/embed` on 14 MB DICOM: 1.38s
* Warm `/embed_batch` on 16×PNG (1024×1024): ~3.4s server-side
* Full CBIS-DDSM (3086 mammograms, chunks of 16): ~11 min end-to-end from a
  warm container

## Caveats

* MedSigLIP was pretrained on chest X-ray, dermatology, ophthalmology, and
  histopathology. **Mammography is off-label.** Zero-shot probabilities are
  weak and are only useful for `MedSigLipResult` display purposes. Use the
  `embed_*` endpoints + a supervised head (Track V) for real classification.
* Server-side `embed_batch` caps at 32 inputs per call. The client chunks to
  16 by default for HTTP payload margin.
