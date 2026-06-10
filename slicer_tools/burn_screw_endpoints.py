# Burn screw entry/tip endpoints (from a Slicer planner CSV) into a CT DICOM
# series as small high-HU spheres, writing a derived series for navigation
# software such as Mazor.
#
# Two interchangeable backends do the DICOM I/O + geometry:
#   --backend pydicom    (default; pure-python, no extra install)
#   --backend simpleitk  (optional; only if SimpleITK is installed)
# Both share the same physical-space sphere-burning math, so they burn the
# identical set of voxels; they differ only in how the series is read/written.
#
# Example:
# python slicer_tools/burn_screw_endpoints.py \
#     --dicom-dir /path/to/ct_dicom_series \
#     --csv /path/to/screw_line_coordinates.csv \
#     --out-dir /path/to/burned_dicom_export

import argparse
import copy
import csv
import os
import time
import uuid
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError
from pydicom.uid import ExplicitVRLittleEndian


UID_ROOT = "1.2.826.0.1.3680043.2.1125"

# Output rescale convention (standards-correct CT; matches the pipeline's
# nifti_utils HU convention).  stored = (HU - intercept) / slope.
OUT_RESCALE_SLOPE = 1.0
OUT_RESCALE_INTERCEPT = -1024.0


# ---------------------------------------------------------------------------
# Shared helpers (backend-independent)
# ---------------------------------------------------------------------------

def make_uid(suffix: str = "") -> str:
    base = f"{UID_ROOT}.{time.strftime('%Y%m%d%H%M%S')}.{uuid.uuid4().int % 10**12}"
    if suffix:
        base = f"{base}.{suffix}"
    return base[:64].rstrip(".")


def ras_to_lps(point_ras: Sequence[float]) -> Tuple[float, float, float]:
    return (-float(point_ras[0]), -float(point_ras[1]), float(point_ras[2]))


def parse_points_csv(csv_path: str) -> List[Dict[str, object]]:
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        required = [
            "line_name",
            "entry_ras_x",
            "entry_ras_y",
            "entry_ras_z",
            "tip_ras_x",
            "tip_ras_y",
            "tip_ras_z",
        ]
        missing = [c for c in required if c not in (reader.fieldnames or [])]
        if missing:
            raise RuntimeError(
                "CSV is missing expected columns: " + ", ".join(missing)
            )

        for r in reader:
            try:
                row = {
                    "line_name": r.get("line_name", ""),
                    "entry_ras": (
                        float(r["entry_ras_x"]),
                        float(r["entry_ras_y"]),
                        float(r["entry_ras_z"]),
                    ),
                    "tip_ras": (
                        float(r["tip_ras_x"]),
                        float(r["tip_ras_y"]),
                        float(r["tip_ras_z"]),
                    ),
                }
                rows.append(row)
            except Exception as e:
                raise RuntimeError(f"Failed parsing CSV row {r}: {e}") from e

    if not rows:
        raise RuntimeError("CSV contained no coordinate rows.")
    return rows


# ---------------------------------------------------------------------------
# Backend-neutral volume representation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Geometry:
    """LPS image geometry, mirroring SimpleITK's image grid exactly.

    Index convention is (i, j, k) = (column, row, slice); the HU array is
    indexed [k, j, i].  ``direction`` columns are [X, Y, normal] where X is the
    column-index direction (ImageOrientationPatient[0:3]), Y the row-index
    direction (IOP[3:6]).  ``spacing`` is (PixelSpacing[1], PixelSpacing[0],
    slice_gap).
    """

    origin: np.ndarray        # (3,) LPS
    spacing: np.ndarray       # (3,) (i, j, k)
    direction: np.ndarray     # (3, 3) columns [X, Y, normal]
    shape_zyx: Tuple[int, int, int]   # (nslices, nrows, ncols)

    def index_to_physical(self, ijk) -> np.ndarray:
        ijk = np.asarray(ijk, dtype=float)
        return self.origin + self.direction @ (self.spacing * ijk)

    def physical_to_continuous_index(self, p_lps) -> np.ndarray:
        p = np.asarray(p_lps, dtype=float)
        m = self.direction * self.spacing      # direction @ diag(spacing)
        return np.linalg.solve(m, p - self.origin)


@dataclass
class Volume:
    hu_zyx: np.ndarray        # int32 HU, shape == geometry.shape_zyx
    geometry: Geometry
    source_slices: object     # pydicom: sorted list[Dataset]; sitk: (reader, file_names)


# ---------------------------------------------------------------------------
# Shared burn math (operates only on an HU array + Geometry)
# ---------------------------------------------------------------------------

def burn_sphere_into_array(
    hu_zyx: np.ndarray,
    geom: Geometry,
    center_lps: Sequence[float],
    radius_mm: float,
    burn_value: float,
) -> int:
    spacing = geom.spacing            # (i, j, k)
    nk, nj, ni = geom.shape_zyx
    center_idx_cont = geom.physical_to_continuous_index(center_lps)   # (i, j, k)

    rx = int(np.ceil(radius_mm / spacing[0])) + 1
    ry = int(np.ceil(radius_mm / spacing[1])) + 1
    rz = int(np.ceil(radius_mm / spacing[2])) + 1

    cx, cy, cz = center_idx_cont

    imin = max(0, int(np.floor(cx)) - rx)
    imax = min(ni - 1, int(np.ceil(cx)) + rx)
    jmin = max(0, int(np.floor(cy)) - ry)
    jmax = min(nj - 1, int(np.ceil(cy)) + ry)
    kmin = max(0, int(np.floor(cz)) - rz)
    kmax = min(nk - 1, int(np.ceil(cz)) + rz)

    touched = 0
    radius_sq = radius_mm * radius_mm

    for k in range(kmin, kmax + 1):
        for j in range(jmin, jmax + 1):
            for i in range(imin, imax + 1):
                p = geom.index_to_physical((i, j, k))
                dx = p[0] - center_lps[0]
                dy = p[1] - center_lps[1]
                dz = p[2] - center_lps[2]
                if dx * dx + dy * dy + dz * dz <= radius_sq:
                    hu_zyx[k, j, i] = burn_value
                    touched += 1

    return touched


def burn_all_points(
    vol: Volume,
    points_rows: List[Dict[str, object]],
    radius_mm: float,
    burn_value: float,
):
    arr = vol.hu_zyx.copy()

    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        burn_value_cast = np.array(np.clip(burn_value, info.min, info.max), dtype=arr.dtype).item()
    else:
        burn_value_cast = np.array(burn_value, dtype=arr.dtype).item()

    summary = []
    total_voxels = 0

    for row in points_rows:
        for label, ras_key in (("entry", "entry_ras"), ("tip", "tip_ras")):
            center_lps = ras_to_lps(row[ras_key])
            voxels = burn_sphere_into_array(arr, vol.geometry, center_lps, radius_mm, burn_value_cast)
            total_voxels += voxels
            summary.append(
                {
                    "line_name": row["line_name"],
                    "point_type": label,
                    "center_ras": row[ras_key],
                    "center_lps": center_lps,
                    "voxels_burned": voxels,
                }
            )

    burned = Volume(hu_zyx=arr, geometry=vol.geometry, source_slices=vol.source_slices)
    return burned, summary, total_voxels


# ---------------------------------------------------------------------------
# pydicom backend (default)
# ---------------------------------------------------------------------------

def _scan_series(dicom_dir: str) -> Dict[str, List[str]]:
    """Map SeriesInstanceUID -> list of geometry-bearing image file paths.

    Mirrors SimpleITK/GDCM: only files that are readable DICOM image slices
    (with ImagePositionPatient + ImageOrientationPatient + Rows/Columns) are
    grouped; non-image objects and non-DICOM files are skipped.
    """
    groups: Dict[str, List[str]] = {}
    for name in os.listdir(dicom_dir):
        path = os.path.join(dicom_dir, name)
        if not os.path.isfile(path):
            continue
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True, force=False)
        except (InvalidDicomError, OSError):
            continue
        if not all(getattr(ds, t, None) is not None for t in
                   ("SeriesInstanceUID", "ImagePositionPatient",
                    "ImageOrientationPatient", "Rows", "Columns")):
            continue
        groups.setdefault(str(ds.SeriesInstanceUID), []).append(path)
    return groups


class PydicomBackend:
    name = "pydicom"

    def choose_series_id(self, dicom_dir: str, requested_series_id: str = None) -> str:
        series_ids = sorted(_scan_series(dicom_dir).keys())
        if not series_ids:
            raise RuntimeError(f'No DICOM series found in "{dicom_dir}".')

        if requested_series_id:
            if requested_series_id not in series_ids:
                raise RuntimeError(
                    "Requested series-id not found.\nAvailable series IDs:\n" + "\n".join(series_ids)
                )
            return requested_series_id

        if len(series_ids) > 1:
            raise RuntimeError(
                "Multiple DICOM series found in the folder. Re-run with --series-id and choose one of:\n"
                + "\n".join(series_ids)
            )

        return series_ids[0]

    def read_series(self, dicom_dir: str, series_id: str) -> Volume:
        paths = _scan_series(dicom_dir).get(series_id, [])
        if not paths:
            raise RuntimeError(f'No image slices for series "{series_id}" in "{dicom_dir}".')

        slices = [pydicom.dcmread(p, force=False) for p in paths]

        # Keep the dominant geometrically-consistent subgroup (matches GDCM's
        # filtering of stray slices, e.g. a localizer at a different orientation).
        def consistency_key(ds):
            return (
                tuple(np.round(np.asarray(ds.ImageOrientationPatient, float), 5)),
                int(ds.Rows), int(ds.Columns),
                tuple(np.round(np.asarray(ds.PixelSpacing, float), 6)),
            )

        from collections import Counter
        keys = [consistency_key(s) for s in slices]
        dominant = Counter(keys).most_common(1)[0][0]
        slices = [s for s, k in zip(slices, keys) if k == dominant]

        # Reject compressed pixel data up front with an actionable message.
        ts = slices[0].file_meta.TransferSyntaxUID
        if ts.is_compressed:
            try:
                _ = slices[0].pixel_array  # probe: can pixels be decoded?
            except Exception:
                raise RuntimeError(
                    f"Source series uses a compressed transfer syntax "
                    f"({ts.name}); pydicom needs a codec to decode it. Install one of:\n"
                    f"  pip install pylibjpeg pylibjpeg-libjpeg\n"
                    f"  pip install python-gdcm\n"
                    f"or re-run with --backend simpleitk."
                )

        iop = np.asarray(slices[0].ImageOrientationPatient, float)
        x_dir, y_dir = iop[:3], iop[3:6]
        normal = np.cross(x_dir, y_dir)

        ipps = np.array([np.asarray(s.ImagePositionPatient, float) for s in slices])
        proj = ipps @ normal
        order = np.argsort(proj)
        slices = [slices[i] for i in order]
        ipps = ipps[order]
        proj = proj[order]

        gaps = np.diff(proj)
        if len(gaps):
            if np.any(np.abs(gaps) < 1e-4):
                raise RuntimeError("Duplicate/overlapping slice positions detected in series.")
            gap = float(np.mean(gaps))
            if gaps.std() > max(1e-3 * abs(gap), 1e-3):
                print(f"WARNING: non-uniform slice spacing (gap std={gaps.std():.4f} mm); "
                      f"using mean {gap:.4f} mm.")
        else:
            gap = float(getattr(slices[0], "SliceThickness", 1.0) or 1.0)

        ps = np.asarray(slices[0].PixelSpacing, float)   # [row(y), col(x)]
        origin = ipps[0].astype(float)
        direction = np.column_stack([x_dir, y_dir, normal])
        rows, cols = int(slices[0].Rows), int(slices[0].Columns)
        spacing = np.array([ps[1], ps[0], gap], dtype=float)
        geom = Geometry(origin=origin, spacing=spacing, direction=direction,
                        shape_zyx=(len(slices), rows, cols))

        hu = np.empty((len(slices), rows, cols), dtype=np.int32)
        for idx, ds in enumerate(slices):
            slope = float(getattr(ds, "RescaleSlope", 1) or 1)
            inter = float(getattr(ds, "RescaleIntercept", 0) or 0)
            hu[idx] = np.rint(ds.pixel_array.astype(np.float64) * slope + inter).astype(np.int32)

        return Volume(hu_zyx=hu, geometry=geom, source_slices=slices)

    def write_series(self, vol: Volume, output_dir: str,
                     series_description_suffix: str = "Burned-in screw endpoints") -> None:
        os.makedirs(output_dir, exist_ok=True)
        slices = vol.source_slices
        nk = vol.geometry.shape_zyx[0]

        new_series_uid = make_uid()
        mod_date = time.strftime("%Y%m%d")
        mod_time = time.strftime("%H%M%S")
        orig_desc = getattr(slices[0], "SeriesDescription", "") or ""
        series_description = (orig_desc + " " + series_description_suffix).strip()

        x_dir = vol.geometry.direction[:, 0]
        y_dir = vol.geometry.direction[:, 1]
        iop = [float(v) for v in (*x_dir, *y_dir)]

        for k in range(nk):
            ds = copy.deepcopy(slices[k])

            pr = int(getattr(ds, "PixelRepresentation", 1))
            out_dtype = np.int16 if pr == 1 else np.uint16
            info = np.iinfo(out_dtype)
            stored_f = (vol.hu_zyx[k].astype(np.float64) - OUT_RESCALE_INTERCEPT) / OUT_RESCALE_SLOPE
            if stored_f.min() < info.min or stored_f.max() > info.max:
                raise RuntimeError(
                    f"Stored pixel value out of {out_dtype.__name__} range on slice {k} "
                    f"([{stored_f.min():.0f}, {stored_f.max():.0f}])."
                )
            stored = np.rint(stored_f).astype(out_dtype)

            ds.PixelData = stored.tobytes()
            ds["PixelData"].VR = "OW"
            ds.Rows, ds.Columns = int(stored.shape[0]), int(stored.shape[1])
            ds.BitsAllocated = 16
            ds.BitsStored = 16
            ds.HighBit = 15
            ds.PixelRepresentation = pr
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = getattr(ds, "PhotometricInterpretation", "MONOCHROME2")
            ds.RescaleSlope = OUT_RESCALE_SLOPE
            ds.RescaleIntercept = OUT_RESCALE_INTERCEPT
            ds.RescaleType = getattr(ds, "RescaleType", "HU")

            ds.ImageType = ["DERIVED", "SECONDARY"]
            ds.SeriesInstanceUID = new_series_uid
            sop_uid = make_uid(str(k + 1))
            ds.SOPInstanceUID = sop_uid
            ds.file_meta.MediaStorageSOPInstanceUID = sop_uid
            ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
            ds.InstanceNumber = k + 1
            ds.SeriesDate = mod_date
            ds.SeriesTime = mod_time
            ds.InstanceCreationDate = mod_date
            ds.InstanceCreationTime = mod_time
            ds.SeriesDescription = series_description
            ds.ImageOrientationPatient = iop
            pos = vol.geometry.index_to_physical((0, 0, k))
            ds.ImagePositionPatient = [float(pos[0]), float(pos[1]), float(pos[2])]

            ds.save_as(os.path.join(output_dir, f"{k + 1:04d}.dcm"), enforce_file_format=True)


# ---------------------------------------------------------------------------
# SimpleITK backend (optional; lazy import) -- reproduces the original tool
# exactly, and serves as the validation oracle.
# ---------------------------------------------------------------------------

def _sitk_write_derived(image_lps, series_reader, output_dir, series_description_suffix):
    import SimpleITK as sitk

    os.makedirs(output_dir, exist_ok=True)

    writer = sitk.ImageFileWriter()
    writer.KeepOriginalImageUIDOn()

    tags_to_copy = [
        "0010|0010", "0010|0020", "0010|0030",
        "0020|000d", "0020|0010", "0008|0020", "0008|0030",
        "0008|0050", "0008|0060", "0020|0052",
    ]

    modification_time = time.strftime("%H%M%S")
    modification_date = time.strftime("%Y%m%d")
    new_series_uid = make_uid()

    direction = image_lps.GetDirection()
    orientation = "\\".join(
        map(str, (direction[0], direction[3], direction[6],
                  direction[1], direction[4], direction[7]))
    )

    original_series_desc = ""
    if series_reader.HasMetaDataKey(0, "0008|103e"):
        original_series_desc = series_reader.GetMetaData(0, "0008|103e")
    series_description = (original_series_desc + " " + series_description_suffix).strip()

    shared_tags = [
        (k, series_reader.GetMetaData(0, k))
        for k in tags_to_copy
        if series_reader.HasMetaDataKey(0, k)
    ] + [
        ("0008|0008", "DERIVED\\SECONDARY"),
        ("0008|0021", modification_date),
        ("0008|0031", modification_time),
        ("0008|103e", series_description),
        ("0020|000e", new_series_uid),
        ("0020|0037", orientation),
    ]

    extract = sitk.ExtractImageFilter()
    size = list(image_lps.GetSize())
    size[2] = 0
    extract.SetSize(size)

    depth = image_lps.GetDepth()
    for k in range(depth):
        extract.SetIndex([0, 0, k])
        image_slice = extract.Execute(image_lps)

        for tag, value in shared_tags:
            image_slice.SetMetaData(tag, str(value))

        instance_date = time.strftime("%Y%m%d")
        instance_time = time.strftime("%H%M%S")
        position = image_lps.TransformIndexToPhysicalPoint((0, 0, k))
        position_str = "\\".join(map(str, position))

        image_slice.SetMetaData("0008|0012", instance_date)
        image_slice.SetMetaData("0008|0013", instance_time)
        image_slice.SetMetaData("0020|0032", position_str)
        image_slice.SetMetaData("0020|0013", str(k + 1))
        image_slice.SetMetaData("0008|0018", make_uid(str(k + 1)))

        writer.SetFileName(os.path.join(output_dir, f"{k + 1:04d}.dcm"))
        writer.Execute(image_slice)


class SimpleitkBackend:
    name = "simpleitk"

    def choose_series_id(self, dicom_dir: str, requested_series_id: str = None) -> str:
        import SimpleITK as sitk
        series_ids = list(sitk.ImageSeriesReader.GetGDCMSeriesIDs(dicom_dir) or [])
        if not series_ids:
            raise RuntimeError(f'No DICOM series found in "{dicom_dir}".')
        if requested_series_id:
            if requested_series_id not in series_ids:
                raise RuntimeError(
                    "Requested series-id not found.\nAvailable series IDs:\n" + "\n".join(series_ids)
                )
            return requested_series_id
        if len(series_ids) > 1:
            raise RuntimeError(
                "Multiple DICOM series found in the folder. Re-run with --series-id and choose one of:\n"
                + "\n".join(series_ids)
            )
        return series_ids[0]

    def read_series(self, dicom_dir: str, series_id: str) -> Volume:
        import SimpleITK as sitk
        file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(dicom_dir, series_id)
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(file_names)
        reader.MetaDataDictionaryArrayUpdateOn()
        reader.LoadPrivateTagsOn()
        image = reader.Execute()

        hu = sitk.GetArrayFromImage(image).astype(np.int32)
        geom = Geometry(
            origin=np.asarray(image.GetOrigin(), float),
            spacing=np.asarray(image.GetSpacing(), float),
            direction=np.asarray(image.GetDirection(), float).reshape(3, 3),
            shape_zyx=hu.shape,
        )
        return Volume(hu_zyx=hu, geometry=geom, source_slices=(reader, list(file_names)))

    def write_series(self, vol: Volume, output_dir: str,
                     series_description_suffix: str = "Burned-in screw endpoints") -> None:
        import SimpleITK as sitk
        reader, _file_names = vol.source_slices
        img = sitk.GetImageFromArray(vol.hu_zyx)
        img.SetOrigin(tuple(float(v) for v in vol.geometry.origin))
        img.SetSpacing(tuple(float(v) for v in vol.geometry.spacing))
        img.SetDirection(tuple(float(v) for v in vol.geometry.direction.flatten()))
        _sitk_write_derived(img, reader, output_dir, series_description_suffix)


def get_backend(name: str):
    if name == "pydicom":
        return PydicomBackend()
    if name == "simpleitk":
        try:
            import SimpleITK  # noqa: F401
        except ImportError:
            raise SystemExit(
                "--backend simpleitk requires SimpleITK, which is not installed.\n"
                "Install it (pip install SimpleITK) or use the default --backend pydicom."
            )
        return SimpleitkBackend()
    raise SystemExit(f"Unknown backend: {name!r}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Burn endpoint spheres into a CT DICOM series using screw entry/tip coordinates exported from Slicer."
    )
    parser.add_argument("--dicom-dir", required=True, help="Folder containing the source CT DICOM series.")
    parser.add_argument("--csv", required=True, help="CSV exported from the Slicer screw planning tool.")
    parser.add_argument("--out-dir", required=True, help="Folder to write the derived burned-in DICOM series.")
    parser.add_argument("--series-id", default=None, help="Optional Series Instance UID if the folder contains multiple series.")
    parser.add_argument("--radius-mm", type=float, default=1.0, help="Sphere radius in mm. Default: 1.0")
    parser.add_argument("--burn-value", type=float, default=3000.0, help="Voxel value (HU) to burn in. Default: 3000")
    parser.add_argument("--backend", choices=["pydicom", "simpleitk"], default="pydicom",
                        help="DICOM I/O backend. Default: pydicom (no SimpleITK needed).")
    args = parser.parse_args()

    backend = get_backend(args.backend)
    series_id = backend.choose_series_id(args.dicom_dir, args.series_id)
    vol = backend.read_series(args.dicom_dir, series_id)
    points_rows = parse_points_csv(args.csv)

    burned, summary, total_voxels = burn_all_points(
        vol, points_rows, radius_mm=args.radius_mm, burn_value=args.burn_value
    )

    backend.write_series(burned, args.out_dir)

    print(f"Backend: {args.backend}")
    print(f"Input series ID: {series_id}")
    print(f"Input slices: {vol.geometry.shape_zyx[0]}")
    print(f"CSV rows read: {len(points_rows)}")
    print(f"Sphere radius (mm): {args.radius_mm}")
    print(f"Burn value (HU): {args.burn_value}")
    print(f"Total endpoint spheres attempted: {len(points_rows) * 2}")
    print(f"Total voxels burned: {total_voxels}")
    print(f"Output DICOM folder: {args.out_dir}")

    for item in summary[:10]:
        print(
            f"{item['line_name']} | {item['point_type']} | "
            f"RAS={item['center_ras']} | LPS={item['center_lps']} | voxels={item['voxels_burned']}"
        )
    if len(summary) > 10:
        print(f"... plus {len(summary) - 10} more endpoint summaries.")


if __name__ == "__main__":
    main()
