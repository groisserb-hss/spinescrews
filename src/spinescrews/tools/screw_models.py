"""Screw geometry models and plan parsing.

Defines the `Screw` class hierarchy (fixed / headless / polyaxial / skip) and `parse_preop_plan()`,
which reads a 3D Slicer screw-plan CSV into the typed `Screw` objects the pipeline registers and
scores against.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import yaml

from bg3dtools.pointclouds.fitting import align_axes, project_to_line, project_to_plane
from bg3dtools.transforms_unified import (
    transform_points_forward,
    spherical_to_cartesian, cartesian_to_spherical,
)
from bg3dtools.mesh.generate import build_cylinder_capped, generate_icosahedron
from bg3dtools.mesh.utils import join_meshes

from spinescrews.tools import dimR, dimA, dimS, possible_levels


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid with clipping."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _screw_volume(shaft_rad: float, shaft_len: float,
                  head_rad: float, head_len: float) -> float:
    """Approximate screw volume (mm^3): cylindrical shaft + open head."""
    return np.pi * shaft_rad**2 * shaft_len + 0.6 * np.pi * head_rad**2 * head_len


# ---------------------------------------------------------------------------
# Screw plan I/O
# ---------------------------------------------------------------------------

def parse_preop_plan(csv_file: str) -> tuple[list[str], list[Screw]]:
    """Parse a 3DSlicer-format CSV screw plan.

    Expected columns: line_name, screw_type, entry_ras_x/y/z, tip_ras_x/y/z,
    length_mm, cylinder_radius_mm.  Optional: head_rad, head_len.

    Returns (levels, screws) sorted distal-to-proximal, L before R.
    """
    plan = pd.read_csv(csv_file)
    plan.columns = [s.strip().lower() for s in plan.columns]

    # load screw names
    screw_names = [s.strip() for s in plan['line_name'].to_list()]
    bad_prefix = [n for n in screw_names if n[0] not in ('S', 'L', 'T', 'C')]
    if bad_prefix:
        raise ValueError('screw names must start with S, L, T, or C; got: %s' % bad_prefix)
    bad_suffix = [n for n in screw_names if n[-1] not in ('L', 'R')]
    if bad_suffix:
        raise ValueError('screw names must end with L or R; got: %s' % bad_suffix)

    # screw types
    screw_types = [s.strip().lower() for s in plan['screw_type'].to_list()]
    bad_types = [t for t in screw_types if t not in ('skip', 'fixed', 'headless', 'poly')]
    if bad_types:
        raise ValueError('unknown screw types: %s (allowed: skip, fixed, headless, poly)' % bad_types)

    # load screw positions/geometry
    entry_pts = np.column_stack((plan.entry_ras_x, plan.entry_ras_y, plan.entry_ras_z)).astype(np.float32)
    tip_pts = np.column_stack((plan.tip_ras_x, plan.tip_ras_y, plan.tip_ras_z)).astype(np.float32)
    screw_rads = plan.cylinder_radius_mm.astype(np.float32)
    screw_lengths = plan.length_mm.astype(np.float32)

    # sort into distal-to-proximal order (use possible_levels for correct
    # anatomical ordering — seg_val integers are non-monotonic for T13=28)
    level_rank = {lvl: i for i, lvl in enumerate(possible_levels)}
    side_val = {'L': 0, 'R': 1}

    numeric_code = np.zeros(len(screw_names))
    for ii, name in enumerate(screw_names):
        numeric_code[ii] = 10 * level_rank[name[:-1]] + side_val[name[-1]]

    # sort and reorder (ascending: most caudal first, L before R)
    sort_order = np.argsort(numeric_code)
    screw_names = [screw_names[s] for s in sort_order]
    screw_types = [screw_types[s] for s in sort_order]
    screw_lengths, screw_rads = screw_lengths[sort_order], screw_rads[sort_order]
    entry_pts, tip_pts = entry_pts[sort_order], tip_pts[sort_order]

    # create list of screw objects
    screws = [Screw.create(v, n, l, r, e, t) for v, n, l, r, e, t in
              zip(screw_types, screw_names, screw_lengths, screw_rads, entry_pts, tip_pts)]

    # override head sizes from plan if columns present
    if 'head_rad' in plan.columns and 'head_len' in plan.columns:
        for screw, l, r in zip(screws, plan['head_len'][sort_order], plan['head_rad'][sort_order]):
            if getattr(screw, 'head_len', 0) > 0:
                screw.head_len = float(l)
                screw.head_rad = float(r)

    levels = [s.name[:-1] for s in screws]  # T1, T2, etc
    l0, l1 = levels[::2], levels[1::2]
    mismatched = [(a, b) for a, b in zip(l0, l1) if a != b]
    if mismatched:
        raise ValueError('screw levels not paired correctly: %s' % mismatched)
    levels = l0
    return levels, screws


def sanity_check_plan(screws: list[Screw], length_thresh: float = 0.5) -> None:
    """Validate screw plan geometry and auto-detect LPS / supine-prone orientation.

    Checks distal-to-proximal ordering, L/R pairing, entry-to-tip distance vs
    planned length, and anterior/lateral consistency.  Converts LPS coordinates
    to RAS in-place when detected.

    Raises ValueError with all collected errors if any check fails.
    """
    entry_pts = np.vstack([S.planned_entry for S in screws])
    tip_pts = np.vstack([S.planned_tip for S in screws])

    # --- Step 1: orientation-independent checks ---
    errors = []
    # row order should be from inferior to superior
    if not np.all(np.diff(entry_pts[::2, dimS]) > 8):
        errors.append('left screw entry points not in distal-to-proximal order')
    if not np.all(np.diff(entry_pts[1::2, dimS]) > 8):
        errors.append('right screw entry points not in distal-to-proximal order')
    if not np.all(np.diff(tip_pts[::2, dimS]) > 8):
        errors.append('left screw tip points not in distal-to-proximal order')
    if not np.all(np.diff(tip_pts[1::2, dimS]) > 8):
        errors.append('right screw tip points not in distal-to-proximal order')

    for left_screw, right_screw in zip(screws[::2], screws[1::2]):
        if left_screw.name[-1] != 'L':
            errors.append('%s: expected left screw (name ending with L)' % left_screw.name)
        if right_screw.name[-1] != 'R':
            errors.append('%s: expected right screw (name ending with R)' % right_screw.name)
        if left_screw.name[:-1] != right_screw.name[:-1]:
            errors.append('screw names should be paired: %s vs %s' % (left_screw.name, right_screw.name))
        if left_screw.type == 'skip' or right_screw.type == 'skip':
            continue
        if left_screw.type != right_screw.type:
            errors.append('screw types should be paired: %s (%s) vs %s (%s)' %
                          (left_screw.name, left_screw.type, right_screw.name, right_screw.type))

    for screw in screws:
        if screw.type == 'skip':
            continue
        tip_to_tail_len = np.linalg.norm(screw.planned_tip - screw.planned_entry)
        if abs(tip_to_tail_len - screw.shaft_len) >= length_thresh:
            errors.append('%s : planned screw len (%.1f) should match entry-to-tip distance (%.1f)' %
                          (screw.name, screw.shaft_len, tip_to_tail_len))

    # --- Step 2: LPS detection via A-axis ---
    # Pedicle screw tips are always anterior to entries; in RAS mean(tip_A - entry_A) > 0
    mean_a_diff = np.mean(tip_pts[:, dimA] - entry_pts[:, dimA])
    if mean_a_diff < 0:
        log.info('Pre-op plan is in LPS coordinates')
        for screw in screws:
            screw.planned_entry[0:2] *= -1
            screw.planned_tip[0:2] *= -1
        # recompute arrays after LPS→RAS fix
        entry_pts = np.vstack([S.planned_entry for S in screws])
        tip_pts = np.vstack([S.planned_tip for S in screws])

    # --- Step 3: orientation detection via R-axis ---
    # In RAS, supine: left screws at negative R, right at positive R
    # Prone: left screws at positive R, right at negative R
    mean_left_R = np.mean(entry_pts[::2, dimR])
    mean_right_R = np.mean(entry_pts[1::2, dimR])
    if mean_left_R < mean_right_R:
        flip = 1
        log.info('Detected orientation: supine')
    else:
        flip = -1
        log.info('Detected orientation: prone')

    # --- Step 4: flip-dependent validation ---
    for left_screw, right_screw in zip(screws[::2], screws[1::2]):
        if left_screw.type == 'skip' or right_screw.type == 'skip':
            continue
        if flip * (right_screw.planned_entry[dimR] - left_screw.planned_entry[dimR]) <= 5:
            errors.append('%s/%s: left entry should be left of right entry' % (left_screw.name, right_screw.name))
        if flip * (right_screw.planned_tip[dimR] - left_screw.planned_tip[dimR]) <= 0:
            errors.append('%s/%s: left tip should be left of right tip' % (left_screw.name, right_screw.name))

    for screw in screws:
        if screw.type == 'skip':
            continue
        if flip * (screw.planned_tip[dimA] - screw.planned_entry[dimA]) <= screw.shaft_len * 0.3:
            errors.append('%s: tip should be anterior to entry point' % screw.name)

    if errors:
        raise ValueError('screw plan validation failed:\n  ' + '\n  '.join(errors))
    log.info('all planned screw tests passed')


# ---------------------------------------------------------------------------
# Screw class hierarchy
# ---------------------------------------------------------------------------

class Screw(ABC):
    """Abstract base for pedicle screw models.

    Stores planned and detected entry/tip positions in RAS world coordinates.
    Subclasses implement head geometry, cost functions, and mesh generation.
    """

    def __init__(self, type: str, name: str,
                 shaft_len: float, shaft_rad: float,
                 planned_entry: np.ndarray, planned_tip: np.ndarray) -> None:
        """Initialize screw with planned geometry.

        Parameters
        ----------
        type : str
            One of 'skip', 'poly', 'headless', 'fixed'.
        name : str
            Level + side identifier (e.g. 'T11L', 'LSR').
        shaft_len : float
            Shaft length in mm (valid range: 10-100).
        shaft_rad : float
            Shaft radius in mm (valid range: 1-10).
        planned_entry, planned_tip : np.ndarray
            (3,) RAS coordinates.
        """
        ABC.__init__(self)
        if np.isfinite(shaft_len) and not (10 < shaft_len < 100):
            raise ValueError('shaft_len=%.1f out of range; expecting mm units (10-100)' % shaft_len)
        if np.isfinite(shaft_rad) and not (1 < shaft_rad < 10):
            raise ValueError('shaft_rad=%.1f out of range; expecting mm units (1-10)' % shaft_rad)
        if name[-1] not in ('L', 'R'):
            raise ValueError('screw name should end with L or R; got %r' % name)
        level = name[:-1]
        if level not in possible_levels:
            raise ValueError('unknown level %r (from screw name %r)' % (level, name))

        self.name = name
        self.level = level
        self.type = type
        self.shaft_len = shaft_len
        self.shaft_rad = shaft_rad
        self.planned_entry = planned_entry
        self.planned_tip = planned_tip

        self.detected_entry = np.full(3, np.nan)
        self.detected_tip = np.full(3, np.nan)

    def __copy__(self) -> Screw:
        """Shallow copy preserving planned and detected endpoints."""
        new_screw = Screw.create(self.type, self.name, self.shaft_len, self.shaft_rad,
                                 self.planned_entry.copy(), self.planned_tip.copy())
        new_screw.detected_entry = self.detected_entry.copy()
        new_screw.detected_tip = self.detected_tip.copy()
        return new_screw

    @staticmethod
    def create(type: str, name: str, shaft_len: float, shaft_rad: float,
               planned_entry: np.ndarray, planned_tip: np.ndarray,
               detected_entry: np.ndarray | None = None,
               detected_tip: np.ndarray | None = None,
               head_vec: np.ndarray | None = None) -> Screw:
        """Factory: create the correct Screw subclass by type string.

        Head-size defaults per type: headless (4mm/8mm), fixed (7mm/17mm),
        poly (7mm/17mm).
        """
        if type == 'skip':
            new_screw = SkipScrew(name, shaft_len, shaft_rad, planned_entry, planned_tip)
        elif type == 'poly':
            new_screw = PolyAxial(name, shaft_len, shaft_rad, planned_entry, planned_tip)
        elif type == 'headless':
            new_screw = FixedHead(name, shaft_len, shaft_rad, planned_entry, planned_tip,
                                  head_rad=4., head_len=8.)
            new_screw.type = 'headless'
        elif type == 'fixed':
            new_screw = FixedHead(name, shaft_len, shaft_rad, planned_entry, planned_tip,
                                  head_rad=7., head_len=17.)
            new_screw.type = 'fixed'
        else:
            raise ValueError(f'unknown screw type {type}')

        if (detected_entry is not None) and (detected_tip is not None):
            planned_len = np.linalg.norm(planned_tip - planned_entry)
            detected_len = np.linalg.norm(detected_tip - detected_entry)
            if abs(planned_len - shaft_len) >= 0.5:
                raise ValueError('planned screw length (%.1f) should match shaft length (%.1f)' % (planned_len, shaft_len))
            if abs(detected_len - shaft_len) >= 0.5:
                raise ValueError('detected screw length (%.1f) should match shaft length (%.1f)' % (detected_len, shaft_len))

            new_screw.detected_entry = detected_entry
            new_screw.detected_tip = detected_tip
        if head_vec is not None:
            new_screw.head_vec = head_vec

        return new_screw

    def axis(self, planned: bool = True) -> np.ndarray:
        """Unit vector from entry to tip."""
        if planned:
            entry_pt = self.planned_entry
            tip_pt = self.planned_tip
        else:
            entry_pt = self.detected_entry
            tip_pt = self.detected_tip

        vec = tip_pt - entry_pt
        mag = np.linalg.norm(vec)
        if mag < 1e-6:
            return np.array([1, 1, 1]) / np.sqrt(3)
        vec /= mag

        return vec

    def center(self, planned: bool = True) -> np.ndarray:
        """Midpoint between entry and tip."""
        if planned:
            return (self.planned_entry + self.planned_tip) / 2
        else:
            return (self.detected_entry + self.detected_tip) / 2

    @property
    @abstractmethod
    def volume(self) -> float:
        """Approximate screw volume in mm^3."""

    @staticmethod
    def build_cylinder(r: float, pt0: np.ndarray, pt1: np.ndarray,
                       N: int = 128) -> tuple[np.ndarray, np.ndarray]:
        """Build a capped cylinder mesh between two endpoints.

        Parameters
        ----------
        r : float
            Cylinder radius.
        pt0, pt1 : np.ndarray
            Start and end points (3,).
        N : int
            Points along the cylinder axis.
        """
        h = np.linalg.norm(pt1 - pt0)

        # unit cylinder (radius 1, length 1)
        nC = max(32, int((N * h) / (2 * np.pi * r)))
        verts, faces = build_cylinder_capped(N, nC)
        verts[:, :2] *= r
        verts[:, 2] *= h
        verts = verts[:, [2, 0, 1]]  # put axis on x-axis

        # rotate cylinder to match axis
        pt2 = np.array([0, 0, 1])  # this will break if axis = [0, 0, 1]
        M = align_axes(pt0, pt1, pt2)
        verts = transform_points_forward(M, verts)

        return verts, faces

    @staticmethod
    def inside_cylinder(pts: np.ndarray, pt0: np.ndarray, pt1: np.ndarray,
                        r: float, return_distances: bool = False
                        ) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Test which points lie inside a cylinder.

        Parameters
        ----------
        pts : np.ndarray
            (N, 3) query points.
        pt0, pt1 : np.ndarray
            Cylinder endpoints (3,).
        r : float
            Cylinder radius.
        return_distances : bool
            If True, also return (radial_d, axial_d).
        """
        pt0, pt1 = pt0.reshape(3), pt1.reshape(3)
        axis = pt1 - pt0
        h = np.linalg.norm(axis)
        axis /= h

        # distance to axis line
        radial_d = project_to_line((pt0, axis), pts)[1]

        # distance to plane of pt0
        plane = np.concatenate([axis, -np.sum(pt0 * axis).reshape(1)])
        axial_d = project_to_plane(plane, pts)[1]

        # compute mask of points inside cylinder
        head_mask = (radial_d < r) & (axial_d > 0) & (axial_d < h)

        if return_distances:
            return head_mask, radial_d, axial_d
        else:
            return head_mask

    def fit_cloud(self, pts: np.ndarray, weights: np.ndarray | None = None,
                  max_evals: int | None = None,
                  ) -> tuple[bool, np.ndarray, dict]:
        """Fit screw parameters to a point cloud via least-squares.

        Parameters
        ----------
        pts : np.ndarray
            (N, 3) metal point cloud.
        weights : np.ndarray or None
            Per-point weights in [0, 1]; sqrt(w) scales residuals.
        max_evals : int or None
            Max function evaluations (default: 5000 * n_params).

        Returns
        -------
        success : bool
            Optimizer convergence flag.
        mask : np.ndarray
            Boolean inlier mask over pts.
        fit_metrics : dict
            Keys: n_points, inlier_ratio, cost, converged.
        """
        n_pts = len(pts)
        sqrt_w = np.sqrt(weights) if weights is not None else None

        def cost_fun(params):
            """Compute weighted residuals for least_squares optimizer."""
            res = self.fit_cost(params, pts)
            if sqrt_w is not None:
                res = res.copy()
                res[:n_pts] *= sqrt_w
            return res

        init_params = self.pack_params(planned=False)

        n_residuals = len(cost_fun(init_params))
        if n_residuals == 0:
            log.warning('%s: no points assigned — keeping ICP result', self.name)
            return False, np.zeros(0, dtype=bool), {
                'n_points': 0, 'inlier_ratio': 0.0, 'cost': 0.0, 'converged': False,
            }
        initial_cost = float(np.sum(cost_fun(init_params)**2))
        method = 'lm' if n_residuals >= len(init_params) else 'trf'
        if method == 'trf':
            log.warning('%s: only %d residuals for %d params — using TRF instead of LM',
                        self.name, n_residuals, len(init_params))

        # run the optimization
        max_evals = (5000 * len(init_params)) if max_evals is None else max_evals
        optimized = least_squares(cost_fun, init_params, method=method, max_nfev=max_evals)
        opt_err = float(np.sum(cost_fun(optimized.x)**2))

        if optimized.success:
            log.debug("%s: %s nfev %d, cost %.0f -> %.0f, optimality %.3f"
                      % (self.name, optimized.message, optimized.nfev, initial_cost, opt_err, optimized.optimality))
        else:
            log.warning('%s: did not converge (%s) — using partial result (cost %.0f -> %.0f)'
                        % (self.name, optimized.message.rstrip('.'), initial_cost, opt_err))
        opt_params = optimized.x

        # Inlier mask uses UNWEIGHTED cost
        raw_cost = self.fit_cost(opt_params, pts)
        if self.type == 'poly':
            raw_cost = raw_cost[:-1]
        mask = raw_cost < 0.5
        inlier_ratio = float(np.sum(mask) / len(mask)) if len(mask) > 0 else 0.0
        self.parse_params(opt_params, planned=False)

        fit_metrics = {
            'n_points': n_pts,
            'inlier_ratio': inlier_ratio,
            'cost': opt_err,
            'converged': bool(optimized.success),
        }
        return optimized.success, mask, fit_metrics

    @abstractmethod
    def fit_cost(self, params: np.ndarray, pts: np.ndarray, t: float = 4.,
                 ) -> np.ndarray:
        """Cost vector for least-squares screw fitting.

        Parameters
        ----------
        params : np.ndarray
            Packed screw parameters (from pack_params).
        pts : np.ndarray
            (N, 3) point cloud.
        t : float
            Sigmoid steepness controlling the inside/outside boundary sharpness.

        Returns
        -------
        np.ndarray
            Per-point cost in [0, 1]; 0 = inside screw, 1 = outside.
            PolyAxial appends one extra element for bend-angle penalty.
        """

    def pack_params(self, planned: bool = True) -> np.ndarray:
        """Pack screw state into a flat parameter vector.

        Layout: [entry_x, entry_y, entry_z, theta, phi].
        PolyAxial extends with head angles.
        """
        shaft_vec = self.axis(planned)
        shaft_ang = cartesian_to_spherical(shaft_vec)

        if planned:
            return np.concatenate([self.planned_entry, shaft_ang[:2]])
        else:
            return np.concatenate([self.detected_entry, shaft_ang[:2]])

    def parse_params(self, params: np.ndarray, planned: bool = True) -> None:
        """Unpack parameter vector into screw entry/tip positions."""
        entry_pt = params[:3]
        shaft_ang = params[3:]
        shaft_vec = spherical_to_cartesian(shaft_ang)

        if planned:
            self.planned_entry = entry_pt
            self.planned_tip = entry_pt + self.shaft_len * shaft_vec
        else:
            self.detected_entry = entry_pt
            self.detected_tip = entry_pt + self.shaft_len * shaft_vec

    def _to_yaml_dict(self) -> dict:
        """Serialize common screw fields to a dict for YAML output."""
        return {
            'name': self.name, 'type': self.type,
            'shaft_len': self.shaft_len, 'shaft_rad': self.shaft_rad,
            'planned_entry': self.planned_entry.tolist(),
            'planned_tip': self.planned_tip.tolist(),
            'detected_entry': self.detected_entry.tolist(),
            'detected_tip': self.detected_tip.tolist(),
        }

    def _from_yaml_dict(self, data: dict) -> None:
        """Load common screw fields from a parsed YAML dict."""
        self.shaft_len = data['shaft_len']
        self.shaft_rad = data['shaft_rad']
        self.planned_entry = np.array(data['planned_entry'])
        self.planned_tip = np.array(data['planned_tip'])
        self.detected_entry = np.array(data['detected_entry'])
        self.detected_tip = np.array(data['detected_tip'])

    def save_to_yaml(self, filename: str) -> None:
        """Write screw state to a YAML file."""
        with open(filename, 'w') as f:
            yaml.dump(self._to_yaml_dict(), f, default_flow_style=False)

    def load_from_yaml(self, filename: str) -> None:
        """Load screw state from a YAML file."""
        with open(filename, 'r') as f:
            data = yaml.safe_load(f)
        if self.type != data['type']:
            raise ValueError('screw type mismatch for %s : should be %s but is %s' % (self.name, self.type, data['type']))
        if self.name != data['name']:
            raise ValueError('screw name mismatch for %s : should be %s but is %s' % (self.name, self.name, data['name']))
        self._from_yaml_dict(data)

    @abstractmethod
    def build_mesh(self, planned: bool) -> tuple[np.ndarray, np.ndarray]:
        """Build a triangle mesh for the screw.  Returns (vertices, faces)."""


class SkipScrew(Screw):
    """Placeholder for a screw position that was not instrumented."""

    def __init__(self, name: str = 'SKIP', shaft_len: float = np.nan, shaft_rad: float = np.nan,
                 planned_entry: np.ndarray = (np.nan, np.nan, np.nan),
                 planned_tip: np.ndarray = (np.nan, np.nan, np.nan)) -> None:
        """Initialize a placeholder for a screw position that was not instrumented."""
        Screw.__init__(self, 'skip', name, shaft_len, shaft_rad, planned_entry, planned_tip)

    @property
    def volume(self) -> float:
        """Always 0 — no physical screw present."""
        return 0

    def fit_cloud(self, pts: np.ndarray, weights: np.ndarray | None = None,
                  max_evals: int | None = None) -> tuple[bool, np.ndarray, dict]:
        """Not applicable — raises ValueError."""
        raise ValueError('SkipScrew cannot be fit to point cloud')

    def fit_cost(self, params: np.ndarray, pts: np.ndarray, t: float = 4.,
                 ) -> np.ndarray:
        """Not applicable — raises ValueError."""
        raise ValueError('SkipScrew has no cost function')

    def build_mesh(self, planned: bool) -> tuple[np.ndarray, np.ndarray]:
        """Build a simple cylinder from planned endpoints if geometry is finite."""
        if not planned:
            raise ValueError('SkipScrew has no detected points')
        if np.isfinite(self.shaft_rad) and np.isfinite(self.shaft_len) and \
            np.all(np.isfinite(self.planned_entry)) and np.all(np.isfinite(self.planned_tip)):
            return self.build_cylinder(self.shaft_rad, self.planned_entry, self.planned_tip)
        else:
            return np.ndarray([0, 3], dtype=np.float32), np.ndarray([0, 3], dtype=np.int32)


class FixedHead(Screw):
    """Screw model with cylindrical shaft and spherical head.

    Used for headless and fixed screw types. Head dimensions are set
    via Screw.create() defaults (headless: 4mm rad / 8mm len,
    fixed: 7mm rad / 17mm len) and optionally overridden by CSV plan.
    """

    def __init__(self, name: str, shaft_len: float, shaft_rad: float,
                 planned_entry: np.ndarray, planned_tip: np.ndarray,
                 head_rad: float = 0., head_len: float = 0.) -> None:
        """Initialize a fixed-head screw with shaft and spherical head geometry."""
        super().__init__('headless', name, shaft_len, shaft_rad, planned_entry, planned_tip)
        self.head_rad = head_rad
        self.head_len = head_len

    def __copy__(self) -> FixedHead:
        """Shallow copy preserving shaft, head, and detected geometry."""
        new_screw = FixedHead(self.name, self.shaft_len, self.shaft_rad,
                              self.planned_entry.copy(), self.planned_tip.copy(),
                              self.head_rad, self.head_len)
        new_screw.type = self.type  # preserve 'headless' or 'fixed'
        new_screw.detected_entry = self.detected_entry.copy()
        new_screw.detected_tip = self.detected_tip.copy()
        return new_screw

    @property
    def volume(self) -> float:
        """Approximate volume (shaft cylinder + head sphere)."""
        return _screw_volume(self.shaft_rad, self.shaft_len, self.head_rad, self.head_len)

    def head_center(self, planned: bool = True) -> np.ndarray:
        """Center of the spherical head, offset behind entry along the axis."""
        if planned:
            return self.planned_entry - self.axis(planned) * self.head_len / 2
        else:
            return self.detected_entry - self.axis(planned) * self.head_len / 2

    def fit_cost(self, params: np.ndarray, pts: np.ndarray, t: float = 4.,
                 ) -> np.ndarray:
        """Cost vector: per-point sigmoid distance to shaft + head geometry."""
        entry_pt, shaft_ang = params[:3], params[3:]
        shaft_vec = spherical_to_cartesian(shaft_ang)
        tip_pt = entry_pt + self.shaft_len * shaft_vec

        # compute distance to axis
        _, h_r, h_d = self.inside_cylinder(pts, entry_pt, tip_pt, self.shaft_rad, return_distances=True)
        h_r = self.shaft_rad - h_r  # boundary at head_rad
        h_d = (self.shaft_len / 2) - np.abs(h_d - self.shaft_len / 2)  # boundary at 0
        shaft_w = 1 - (_sigmoid(t * h_r) * _sigmoid(t * h_d))

        # distance to head
        if self.head_len > 0:
            d = np.linalg.norm(pts - self.head_center().reshape([1, 3]), axis=1)
            head_w = 1 - _sigmoid(t * (self.head_len / 2 - d))
            w = np.minimum(head_w, shaft_w)
        else:
            w = shaft_w

        return w

    def _to_yaml_dict(self) -> dict:
        """Add head_len and head_rad to the base YAML dict."""
        d = super()._to_yaml_dict()
        d['head_len'] = self.head_len
        d['head_rad'] = self.head_rad
        return d

    def _from_yaml_dict(self, data: dict) -> None:
        """Load head_len and head_rad in addition to base fields."""
        super()._from_yaml_dict(data)
        self.head_len = data.get('head_len', self.head_len)
        self.head_rad = data.get('head_rad', self.head_rad)

    def build_mesh(self, planned: bool) -> tuple[np.ndarray, np.ndarray]:
        """Build shaft cylinder + head sphere mesh."""
        sphere_v, sphere_f = generate_icosahedron()
        sphere_v *= self.head_rad
        sphere_v += self.head_center(planned)

        if planned:
            cyl_v, cyl_f = self.build_cylinder(self.shaft_rad, self.planned_entry, self.planned_tip)
        else:
            cyl_v, cyl_f = self.build_cylinder(self.shaft_rad, self.detected_entry, self.detected_tip)

        v, f = join_meshes(cyl_v, cyl_f, sphere_v, sphere_f)
        return v, f


class PolyAxial(Screw):
    """Poly-axial screw with articulating head and separate shaft."""

    def __init__(self, name: str,
                 shaft_len: float, shaft_rad: float = 2.25,
                 planned_entry: np.ndarray = np.zeros(3),
                 planned_tip: np.ndarray = np.zeros(3),
                 head_len: float = 17., head_rad: float = 7.) -> None:
        """Initialize a poly-axial screw with articulating head and shaft."""
        Screw.__init__(self, 'poly', name, shaft_len, shaft_rad, planned_entry, planned_tip)
        self.head_len = head_len
        self.head_rad = head_rad
        self.head_vec = np.full(3, np.nan)
        self.joint_offset = 4

        self.ransac_iters = 1000
        self.min_bend = 4. * np.pi / 5.

    def __copy__(self) -> PolyAxial:
        """Shallow copy preserving shaft, head, RANSAC, and detected geometry."""
        new_screw = PolyAxial(self.name, self.shaft_len, self.shaft_rad,
                              self.planned_entry.copy(), self.planned_tip.copy(),
                              self.head_len, self.head_rad)
        new_screw.detected_entry = self.detected_entry.copy()
        new_screw.detected_tip = self.detected_tip.copy()
        new_screw.head_vec = self.head_vec.copy()
        new_screw.joint_offset = self.joint_offset
        new_screw.ransac_iters = self.ransac_iters
        new_screw.min_bend = self.min_bend
        return new_screw

    @property
    def volume(self) -> float:
        """Approximate volume (shaft cylinder + head cylinder)."""
        return _screw_volume(self.shaft_rad, self.shaft_len, self.head_rad, self.head_len)

    def head_pts(self, planned: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Compute the two endpoints of the head cylinder."""
        if planned:
            entry_pt = self.planned_entry
            head_vec = -self.axis(planned)
        else:
            entry_pt = self.detected_entry
            head_vec = self.head_vec
            if np.any(np.isnan(self.head_vec)):
                head_vec = -self.axis(planned)

        shaft_axis = self.axis(planned)

        joint_pt = entry_pt - self.joint_offset * shaft_axis
        head_a = joint_pt - self.joint_offset * head_vec
        head_b = joint_pt + (self.head_len - self.joint_offset) * head_vec
        return head_a, head_b

    def build_mesh(self, planned: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Build shaft + head cylinder mesh."""
        if planned:
            entry_pt = self.planned_entry
            tip_pt = self.planned_tip
        else:
            entry_pt = self.detected_entry
            tip_pt = self.detected_tip

        head_a, head_b = self.head_pts(planned)
        shaft_v, shaft_f = Screw.build_cylinder(self.shaft_rad, entry_pt, tip_pt)
        head_v, head_f = Screw.build_cylinder(self.head_rad, head_a, head_b)

        # combine cylinders
        verts = np.vstack([shaft_v, head_v])
        faces = np.vstack([shaft_f, head_f + len(shaft_v)])
        return verts, faces

    def pack_params(self, planned: bool = True) -> np.ndarray:
        """Pack params: [entry, shaft_theta, shaft_phi, head_theta, head_phi]."""
        shaft_vec = self.axis(planned)
        shaft_ang = cartesian_to_spherical(shaft_vec)

        if planned or np.any(np.isnan(self.head_vec)):
            head_vec = -shaft_vec
        else:
            head_vec = self.head_vec
        head_ang = cartesian_to_spherical(head_vec)

        neck_pt = self.planned_entry if planned else self.detected_entry

        params = np.concatenate([neck_pt.flatten(), shaft_ang[:2], head_ang[:2]])

        return params

    def parse_params(self, params: np.ndarray, planned: bool = True) -> None:
        """Unpack params including head vector."""
        entry_pt = params[:3]
        shaft_ang = params[3:5]
        shaft_vec = spherical_to_cartesian(shaft_ang)
        head_ang = params[5:]
        head_vec = spherical_to_cartesian(head_ang)

        if planned:
            self.planned_entry = entry_pt
            self.planned_tip = entry_pt + self.shaft_len * shaft_vec
            self.head_vec = head_vec
        else:
            self.detected_entry = entry_pt
            self.detected_tip = entry_pt + self.shaft_len * shaft_vec
            self.head_vec = head_vec

    def fit_cost(self, params: np.ndarray, pts: np.ndarray, t: float = 3.,
                 ) -> np.ndarray:
        """Cost vector: per-point distance to shaft + head, plus bend penalty.

        Returns (N+1,) array: N per-point costs plus one bend-angle cost.
        """
        entry_pt, shaft_ang, head_ang = params[:3], params[3:5], params[5:7]
        shaft_vec = spherical_to_cartesian(shaft_ang)
        head_vec = spherical_to_cartesian(head_ang)

        # prevent bending polyaxial head too much
        bend_angle = np.arccos(shaft_vec.dot(head_vec).clip(-1, 1))
        b = (self.min_bend - bend_angle) * 100
        bend_cost = 1000 * _sigmoid(b)  # weighted sigmoid

        # compute distance to each axis
        joint_pt = entry_pt - self.joint_offset * shaft_vec
        head_a = joint_pt - self.joint_offset * head_vec
        head_b = joint_pt + (self.head_len - self.joint_offset) * head_vec
        _, h_r, h_d = self.inside_cylinder(pts, head_a, head_b, self.head_rad, return_distances=True)

        h_r = self.head_rad - h_r  # boundary at head_rad
        h_d = (self.head_len / 2) - np.abs(h_d - self.head_len / 2)   # boundary at 0
        head_w = 1 - (_sigmoid(t * h_r) * _sigmoid(t * h_d))

        tip_pt = entry_pt + self.shaft_len * shaft_vec
        _, s_r, s_d = self.inside_cylinder(pts, entry_pt, tip_pt, self.shaft_rad, return_distances=True)
        s_r = 2 * (self.shaft_rad - s_r) / self.shaft_rad  # boundary at shaft_rad
        s_d = (self.shaft_len / 2) - np.abs(s_d - self.shaft_len / 2)
        shaft_w = 1 - (_sigmoid(t * s_r) * _sigmoid(t * s_d))

        w = np.minimum(head_w, 2 * shaft_w)
        w = np.concatenate([w, bend_cost * np.ones(1)])
        return w

    def _to_yaml_dict(self) -> dict:
        """Add head geometry and head_vec to the base YAML dict."""
        d = super()._to_yaml_dict()
        d['head_len'] = self.head_len
        d['head_rad'] = self.head_rad
        d['head_vec'] = self.head_vec.tolist()
        return d

    def _from_yaml_dict(self, data: dict) -> None:
        """Load head geometry and head_vec in addition to base fields."""
        super()._from_yaml_dict(data)
        self.head_len = data.get('head_len', self.head_len)
        self.head_rad = data.get('head_rad', self.head_rad)
        if 'head_vec' in data:
            self.head_vec = np.array(data['head_vec'])


# ---------------------------------------------------------------------------
# Multi-screw joint fitting
# ---------------------------------------------------------------------------

def _extract_endpoints(params: np.ndarray, screw: Screw,
                       offset: int) -> tuple[np.ndarray, np.ndarray]:
    """Extract entry and tip from packed params for a single screw."""
    entry = params[offset:offset + 3]
    shaft_ang = params[offset + 3:offset + 5]
    shaft_vec = spherical_to_cartesian(shaft_ang)
    tip = entry + screw.shaft_len * shaft_vec
    return entry, tip


def _separation_residuals(params: np.ndarray, screws: list[Screw],
                          num_params: list[int],
                          planned_entry_dists: list[float],
                          planned_tip_dists: list[float],
                          pitch: float = 0.5,
                          w_entry_compress: float = 10.,
                          w_entry_stretch: float = 1.,
                          w_tip_compress: float = 5.,
                          w_tip_stretch: float = 0.5) -> np.ndarray:
    """Pairwise separation prior: penalize screws bunching up or stretching apart.

    Asymmetric: compression (r < 1) uses 1/r - 1 (prohibitive as r -> 0),
    stretching (r > 1) uses r - 1 (gentle linear).
    Entry gets tighter constraints than tip.
    Weights scale down with pitch (coarser voxels -> wider tolerance).
    """
    scale = 0.5 / pitch  # 1.0 at native, 0.25 at 2mm
    offsets = np.concatenate([[0], np.cumsum(num_params[:-1])])
    endpoints = [_extract_endpoints(params, s, o) for s, o in zip(screws, offsets)]

    residuals = []
    pair_idx = 0
    for i in range(len(screws)):
        for j in range(i + 1, len(screws)):
            for kind, w_c, w_s, d_plan in [
                ('entry', w_entry_compress * scale, w_entry_stretch * scale, planned_entry_dists[pair_idx]),
                ('tip',   w_tip_compress * scale,   w_tip_stretch * scale,   planned_tip_dists[pair_idx]),
            ]:
                ep = 0 if kind == 'entry' else 1
                d_curr = np.linalg.norm(endpoints[i][ep] - endpoints[j][ep])
                r = d_curr / max(d_plan, 1e-6)
                r = max(r, 1e-6)  # avoid division by zero

                if r < 1:
                    residuals.append(w_c * (1.0 / r - 1.0))
                else:
                    residuals.append(w_s * (r - 1.0))
            pair_idx += 1

    return np.array(residuals)


def multi_screw_cloud_fit(screws: list[Screw], pts: np.ndarray,
                          pitch: float = 0.5, max_evals: int | None = None,
                          ftol: float = 0.01) -> tuple[bool, np.ndarray]:
    """Joint optimization of multiple screws with pairwise separation priors.

    Parameters
    ----------
    screws : list[Screw]
        Screws to fit jointly (detected positions used as initialization).
    pts : np.ndarray
        (N, 3) metal point cloud.
    pitch : float
        Voxel pitch for scaling separation weights.
    max_evals : int or None
        Max function evaluations.
    ftol : float
        Relative cost tolerance for convergence.

    Returns
    -------
    success : bool
        Optimizer convergence flag.
    mask : np.ndarray
        Boolean inlier mask over pts.
    """
    nP = len(pts)
    num_params = [len(screw.pack_params()) for screw in screws]

    # precompute planned pairwise distances
    planned_entry_dists = []
    planned_tip_dists = []
    for i in range(len(screws)):
        for j in range(i + 1, len(screws)):
            planned_entry_dists.append(np.linalg.norm(screws[i].planned_entry - screws[j].planned_entry))
            planned_tip_dists.append(np.linalg.norm(screws[i].planned_tip - screws[j].planned_tip))

    def cost_fun(params):
        """Combined per-point min-distance cost across all screws plus separation priors."""
        w = np.inf * np.ones(nP)
        r = []
        oo = 0
        for n, screw in zip(num_params, screws):
            w_ = screw.fit_cost(params[oo:oo+n], pts)
            w = np.minimum(w, w_[:nP])
            r.append(w_[nP:])
            oo += n
        stacked_r = np.concatenate(r)

        sep = _separation_residuals(params, screws, num_params,
                                    planned_entry_dists, planned_tip_dists,
                                    pitch=pitch)

        return np.concatenate([w, stacked_r, sep])

    init_params = np.concatenate([screw.pack_params(planned=False) for screw in screws])
    initial_cost = float(np.sum(cost_fun(init_params)**2))

    max_evals = (5000 * len(init_params)) if max_evals is None else max_evals
    optimized = least_squares(cost_fun, init_params, method='trf', max_nfev=max_evals, verbose=2, ftol=ftol)
    opt_err = float(np.sum(cost_fun(optimized.x)**2))

    log.debug("multi_screw: %s nfev %d, cost %.0f -> %.0f, optimality %.3f"
               % (optimized.message, optimized.nfev, initial_cost, opt_err, optimized.optimality))
    if not optimized.success:
        log.warning('multi_screw_cloud_fit did not converge (%s) — using partial result (cost %.0f -> %.0f)'
                    % (optimized.message.rstrip('.'), initial_cost, opt_err))
    opt_params = optimized.x

    mask = cost_fun(opt_params)[:nP] < 0.5
    # parse params for all screws
    for n, screw in zip(num_params, screws):
        screw.parse_params(opt_params[:n], planned=False)
        opt_params = opt_params[n:]

    return optimized.success, mask
