"""NIfTI volume helpers: resampling, cropping, and metal-threshold estimation.

Utilities for working with CT volumes in RAS — resampling to an isotropic pitch, bounding-box
cropping, and Otsu-based estimation of the metal/bone Hounsfield thresholds used by screw
detection (`compute_metal_threshold`).
"""

import logging

import numpy as np
import nibabel as nib
from scipy import ndimage

log = logging.getLogger(__name__)

# 12-bit DICOM: unsigned 0-4095, rescale intercept -1024 → max HU = 3071
HU_CLIP = 3071


def resample_to_pitch(img: nib.Nifti1Image,
                      target_pitch: tuple[float, float, float]
                      ) -> nib.Nifti1Image:
    """Resample a NIfTI image to a target voxel pitch via linear interpolation.

    Assumes canonical orientation (diagonal-dominant affine). Short-circuits
    when the current pitch already matches the target within 1%. Output affine
    preserves the world origin and adjusts the per-axis voxel scales using the
    pitch actually achieved (which may drift from the request by sub-percent
    due to integer rounding of the output shape).
    """
    pitch = np.abs(np.diag(img.affine[:3, :3]))
    target = np.asarray(target_pitch, dtype=float)

    if np.allclose(pitch, target, rtol=0.01):
        return img

    zoom = pitch / target
    resampled = ndimage.zoom(img.get_fdata(), zoom, order=1)

    achieved = pitch * (np.array(img.shape[:3]) / np.array(resampled.shape[:3]))
    new_affine = img.affine.copy()
    new_affine[:3, :3] = img.affine[:3, :3] @ np.diag(achieved / pitch)

    log.info('resampled %s @ %s mm → %s @ %s mm',
             tuple(img.shape[:3]), np.round(pitch, 3).tolist(),
             tuple(resampled.shape[:3]), np.round(achieved, 3).tolist())

    return nib.Nifti1Image(resampled, new_affine)


def _otsu_threshold(values: np.ndarray, nbins: int = 256) -> float:
    """Otsu's method: threshold maximizing inter-class variance."""
    counts, edges = np.histogram(values, bins=nbins)
    centers = (edges[:-1] + edges[1:]) / 2
    p = counts / counts.sum()
    w = np.cumsum(p)
    mu = np.cumsum(p * centers)
    mu_total = mu[-1]
    valid = (w > 0) & (w < 1)
    sigma_b = np.zeros_like(w)
    sigma_b[valid] = (mu_total * w[valid] - mu[valid]) ** 2 / (w[valid] * (1 - w[valid]))
    return float(centers[np.argmax(sigma_b)])


def compute_metal_threshold(data: np.ndarray, floor: int = 500) -> int:
    """Adaptive metal detection threshold via Otsu on upper HU tail.

    Uses the metal class mean minus 2 sigma minus 200 HU offset.
    At native resolution (~0.5mm) this gives ~2200-2300 HU.
    At coarse resolution (4mm) this drops to ~1000-1100 HU,
    capturing partial-volume metal voxels.
    """
    data.clip(-1024, HU_CLIP)

    upper = data[data > floor].ravel()
    if len(upper) < 100:
        log.warning('Too few voxels above %d HU floor (%d); using default 2200',
                     floor, len(upper))
        return 2200

    # Seed: all voxels likely containing metal (including partial-volume).
    # Fixed at 2700 HU rather than a percentile — dense cortical bone
    # rarely exceeds ~1800 HU, so >= 2700 is overwhelmingly metal.
    # At coarse resolution a percentile-based seed (p99.9) collapses
    # onto the pure-metal pile-up near 3071, excluding partial-volume
    # voxels and causing Otsu to split within the metal class.
    METAL_SEED_HU = 2700
    n_metal_seed = int(np.sum(upper >= METAL_SEED_HU))

    # If metal is < 5% of the upper tail, Otsu can't resolve it —
    # it finds an intra-bone split instead. Narrow the window so
    # metal reaches ~5%, giving Otsu enough representation.
    metal_frac = n_metal_seed / len(upper)
    if metal_frac < 0.05:
        target_n = int(n_metal_seed / 0.05)
        cutoff = np.partition(upper, -target_n)[-target_n]
        analysis = upper[upper >= cutoff]
        log.info('Metal underrepresented (%.1f%%); narrowing to top %d '
                 'voxels (>= %.0f HU)',
                 100 * metal_frac, len(analysis), cutoff)
    else:
        analysis = upper

    otsu = _otsu_threshold(analysis)
    metal = analysis[analysis > otsu]
    mu_m = float(np.mean(metal))
    sigma_m = float(np.std(metal))
    threshold = int(mu_m - 2 * sigma_m - 200)
    threshold = max(threshold, floor)  # never below floor

    log.info('Adaptive metal threshold: %d HU '
             '(Otsu=%d, mu_metal=%.0f, sigma_metal=%.0f)',
             threshold, int(otsu), mu_m, sigma_m)
    return threshold


def nonzero_box(label: nib.Nifti1Image) -> nib.Nifti1Image:
    """
    :param label:
    :return:
    """
    data = label.get_fdata()
    nz = np.nonzero(data)
    start = np.array([nz[0].min(), nz[1].min(), nz[2].min()])
    stop = np.array([nz[0].max() + 1, nz[1].max() + 1, nz[2].max() + 1])
    sub_data = data[start[0]:stop[0], start[1]:stop[1], start[2]:stop[2]]
    sub_affine = label.affine.copy()
    sub_affine[:3, 3:4] += sub_affine[:3, :3] @ start.reshape(3, 1)

    return nib.Nifti1Image(sub_data, sub_affine)
