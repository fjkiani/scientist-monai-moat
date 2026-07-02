# CBIS-DDSM DICOM fixtures

Five real screening mammograms pulled from a public Hugging Face mirror of CBIS-DDSM:

- Source dataset: `helloerikaaa/cbis-ddsm-r`
- License: **CC-BY-NC 4.0** (non-commercial research use)
- Total size: ~120 MB (excluded from git via `.gitignore`)
- Metadata inventory: `_dicom_metadata.json` (versioned in git)

## Files

| Filename | View | Laterality (filename) | `BodyPartExamined` | Rows × Cols | Bits |
|---|---|---|---|---|---|
| `Calc-Test_P_00038_LEFT_CC.dcm` | CC | LEFT | `"Left Breast"` | 4616 × 3016 | 16 |
| `Calc-Test_P_00038_RIGHT_CC.dcm` | CC | RIGHT | `"Right Breast"` | 4688 × 2744 | 16 |
| `Calc-Test_P_00038_LEFT_MLO.dcm` | MLO | LEFT | `"Left Breast"` | 4728 × 3064 | 16 |
| `Calc-Test_P_00038_RIGHT_MLO.dcm` | MLO | RIGHT | `"Right Breast"` | 4720 × 2928 | 16 |
| `Mass-Test_P_00016_LEFT_CC.dcm` | CC | LEFT | `"BREAST"` (no side) | 4006 × 1846 | 16 |

## Real-data properties discovered

1. **Heterogeneous DICOM tag population**: Calc- fixtures have `BodyPartExamined` including
   laterality (`"Left Breast"`); the Mass- fixture has bare `"BREAST"` with no side. Neither
   `ImageLaterality` nor `ViewPosition` is populated in this mirror — orientation is in
   `PatientOrientation`.
2. **Bit-depth sparse packing**: `BitsStored=16` but pixel values only occupy the bottom 8 bits
   (max ≈ 187/65535 ≈ 0.003 after 16-bit normalization). Preprocessing MUST use adaptive
   Otsu thresholding, not a hardcoded background threshold.
3. **Orientation heterogeneity**: Calc- fixtures are in acquisition orientation (chest wall
   on either side per laterality); the Mass- fixture is pre-mirrored. Content-only
   laterality detection is unreliable on this dataset — filename or DICOM-tag hints are
   authoritative.
4. **Pectoral pre-cropping**: All 4 MLO fixtures are cropped tightly enough that the
   pectoral triangle is absent from the top corners. `remove_pectoral_mlo` is a no-op on
   these fixtures; its correctness is verified in `tests/unit/test_pectoral_removal_synthetic.py`
   against a synthetic MLO with a known pectoral triangle.

## Re-downloading

Run from repo root:

```bash
python tests/fixtures/download_cbis_ddsm_fixtures.py
```

This uses `huggingface_hub.hf_hub_download` and writes the five DICOMs into this directory.
No authentication required (public dataset).
