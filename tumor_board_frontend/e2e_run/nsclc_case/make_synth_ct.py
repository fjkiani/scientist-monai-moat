"""Synthesize a 16-slice CT volume with a planted +50-HU nodule and write
it out as a valid DICOM series that the arbiter's CT reader can ingest.

Matches the shape/spacing/HU profile documented in the `demo_provenance`
block of `demo_samples/nsclc.json`:
    * 16 slices, 128 × 128 pixels
    * 1.0 × 1.0 × 2.5 mm spacing
    * gantry -1200 HU, lung -800 HU, body -50 HU, planted nodule +50 HU
    * 9 × 9 × 5 hyperdense blob at volume center, slices 6..10

This is not synthetic data pretending to be a patient CT; it's a
detector-input-shaped calibration phantom. Running LUNA16 on it exercises
the same code path used on real LIDC-IDRI. Everything downstream is real.
"""
from __future__ import annotations

import argparse
import pathlib
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid, CTImageStorage


SHAPE = (16, 128, 128)              # z, y, x
SPACING = (2.5, 1.0, 1.0)           # slice_thickness_mm, row_mm, col_mm
NODULE_HU = 50
BODY_HU = -50
LUNG_HU = -800
GANTRY_HU = -1200
NODULE_SHAPE = (7, 34, 34)          # z, y, x   (spans slices 5..11; ~34 mm in-plane → HIGH bucket per Fleischner)
STUDY_UID = generate_uid()
SERIES_UID = generate_uid()


def build_volume() -> np.ndarray:
    """Build the CT volume (int16 HU) matching the demo_provenance profile."""
    vol = np.full(SHAPE, GANTRY_HU, dtype=np.int16)

    # Body ring: interior square minus lung region
    vol[:, 20:108, 20:108] = BODY_HU

    # Lung parenchyma: two ovoid regions (left/right lung) at slices 3..12
    zz, yy, xx = np.ogrid[: SHAPE[0], : SHAPE[1], : SHAPE[2]]
    # Left lung, centered around (y=64, x=44)
    left = ((yy - 64) ** 2 / 30**2 + (xx - 44) ** 2 / 20**2) <= 1
    right = ((yy - 64) ** 2 / 30**2 + (xx - 84) ** 2 / 20**2) <= 1
    lung_slice_mask = (zz >= 3) & (zz <= 12)
    vol[np.broadcast_to(lung_slice_mask, vol.shape) & (left | right)] = LUNG_HU

    # Planted MASS: NODULE_SHAPE centered at left-lung center. A >30 mm
    # diameter mass exercises the HIGH-risk NCCN-lite branch (LUNG-1 mass
    # branch). We plant an ellipsoid so the diameter isn't a perfect cube.
    cz, cy, cx = 8, 64, 44   # left-lung center
    dz, dy, dx = NODULE_SHAPE
    zz, yy, xx = np.ogrid[: SHAPE[0], : SHAPE[1], : SHAPE[2]]
    mass = (
        ((zz - cz) ** 2) / (dz / 2) ** 2
        + ((yy - cy) ** 2) / (dy / 2) ** 2
        + ((xx - cx) ** 2) / (dx / 2) ** 2
    ) <= 1
    vol[mass] = NODULE_HU
    return vol


def _make_slice_dataset(vol_slice: np.ndarray, slice_idx: int) -> FileDataset:
    """Wrap one axial slice in a minimally-valid CT DICOM."""
    fm = FileMetaDataset()
    fm.MediaStorageSOPClassUID = CTImageStorage
    fm.MediaStorageSOPInstanceUID = generate_uid()
    fm.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(
        filename_or_obj=f"slice_{slice_idx:03d}.dcm",
        dataset={},
        file_meta=fm,
        preamble=b"\0" * 128,
    )
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = fm.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = STUDY_UID
    ds.SeriesInstanceUID = SERIES_UID

    ds.PatientID = "SYN-0001"
    ds.PatientName = "Anon^Synthetic"
    ds.Modality = "CT"
    ds.Rows = SHAPE[1]
    ds.Columns = SHAPE[2]
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1  # signed
    ds.RescaleIntercept = 0
    ds.RescaleSlope = 1

    ds.PixelSpacing = [SPACING[1], SPACING[2]]
    ds.SliceThickness = SPACING[0]
    ds.SpacingBetweenSlices = SPACING[0]

    # ImagePositionPatient in mm; z ordered feet → head (increasing).
    z_mm = slice_idx * SPACING[0]
    ds.ImagePositionPatient = [0.0, 0.0, z_mm]
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.InstanceNumber = slice_idx + 1

    ds.PixelData = vol_slice.astype(np.int16).tobytes()
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    return ds


def write_series(out_dir: pathlib.Path) -> pathlib.Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    vol = build_volume()
    for z in range(SHAPE[0]):
        ds = _make_slice_dataset(vol[z], z)
        (out_dir / f"slice_{z:03d}.dcm").write_bytes(b"")   # touch
        ds.save_as(out_dir / f"slice_{z:03d}.dcm")
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/e2e_synth_ct/SYN-0001/STUDY/CT_SYN",
                    help="output series directory")
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    write_series(out)
    n_files = sum(1 for _ in out.iterdir())
    print(f"wrote {n_files} DICOM slices to {out}")
    print(f"volume shape: {SHAPE}, spacing (z,y,x): {SPACING} mm")
    print(f"planted nodule: {NODULE_SHAPE} @ +{NODULE_HU} HU, slices 6..10")


if __name__ == "__main__":
    main()
