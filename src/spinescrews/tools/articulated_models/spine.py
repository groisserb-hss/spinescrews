"""
Spine articulated model.

This module provides the Spine class for representing and manipulating
segmented spine meshes with kinematic chain deformations.
"""

import numpy as np
from .base_unified import Articulated
import bg3dtools.transforms_unified as transforms
from bg3dtools.pytorch import to_numpy


class Spine(Articulated):
    """
    Articulated spine model with per-vertebra segmentation.

    Parameters
    ----------
    rest_verts : (N, 3) ndarray or list of ndarrays
        Vertex coordinates. If list, each element is one vertebra.
    labels : (N,) ndarray, optional
        Segment labels for each vertex.
    default_aff : (J, 4, 4) ndarray or tuple, optional
        Default joint transforms.
    faces : (M, 3) ndarray or list, optional
        Face indices.
    trunk : list of int, optional
        Parent indices for kinematic chain.
    landmarks : ndarray, optional
        Anatomical landmarks.
    backend : str, optional
        Array backend: 'numpy' (default) or 'torch'.
    """

    def __init__(self, rest_verts, labels=None, default_aff=None, faces=None,
                 trunk=None, landmarks=None, backend='numpy'):
        """Initialize articulated spine from per-vertebra vertex lists or a labeled vertex array."""
        if trunk is None:
            nJ = len(rest_verts) if isinstance(rest_verts, list) else np.max(labels)+1
            trunk = list(np.arange(nJ, dtype=int) - 1)
        Articulated.__init__(self, trunk, backend)
        bk = self._bk

        if isinstance(rest_verts, list):
            if backend == 'numpy':
                self.verts = [v.copy() for v in rest_verts]
            else:
                self.verts = [bk.array(v) for v in rest_verts]
            labels = [ii * np.ones(len(v)) for ii, v in enumerate(rest_verts)]
            self.labels = np.concatenate(labels).astype(int)
        else:
            assert isinstance(labels, np.ndarray)
            if backend == 'numpy':
                self.verts = [np.ascontiguousarray(rest_verts[labels == ll].copy()) for ll in range(len(trunk))]
            else:
                self.verts = [bk.array(np.ascontiguousarray(rest_verts[labels == ll])) for ll in range(len(trunk))]
            self.labels = labels.copy()

        if isinstance(default_aff, tuple):
            default_twist, default_trans = default_aff
            aff = transforms.rel_params_to_aff(self.trunk, default_twist, default_trans)
            self.default_aff = bk.array(aff) if backend != 'numpy' else aff
        elif default_aff is not None:
            assert default_aff.shape == (len(self.trunk), 4, 4)
            self.default_aff = bk.array(default_aff) if backend != 'numpy' else default_aff.copy()
        else:
            aff = transforms.make_aff(np.zeros([len(self.trunk), 3]), np.zeros([len(self.trunk), 3]))
            self.default_aff = bk.array(aff) if backend != 'numpy' else aff

        if isinstance(faces, list):
            all_faces = []
            offset = 0
            for ii in range(self.nJ):
                all_faces.append(faces[ii] + offset)
                offset += len(self.verts[ii])
            self.faces = np.ascontiguousarray(np.concatenate(all_faces))
        elif isinstance(faces, np.ndarray):
            assert np.max(faces) == len(self.labels) - 1
            self.faces = faces.copy()
        else:
            assert faces is None
            self.faces = None

        assert landmarks is None or len(landmarks) == self.num_bones
        if landmarks is not None and backend != 'numpy':
            self.landmarks = [bk.array(lm) for lm in landmarks]
        else:
            self.landmarks = landmarks

    @classmethod
    def with_backend(cls, model, backend):
        """Create a copy of the model using a different compute backend (numpy/torch/jax)."""
        verts = [to_numpy(v) for v in model.verts]
        labels = model.labels.copy()
        default_aff = to_numpy(model.default_aff) if model.default_aff is not None else None
        faces = model.faces.copy() if model.faces is not None else None
        trunk = model.trunk.copy()
        landmarks = [to_numpy(lm) for lm in model.landmarks] if model.landmarks is not None else None
        return cls(verts, labels=labels, default_aff=default_aff, faces=faces,
                   trunk=trunk, landmarks=landmarks, backend=backend)

    @property
    def nV(self):
        """Total number of vertices across all bone segments."""
        return np.sum([len(v) for v in self.verts])

    @property
    def num_bones(self):
        """Number of bones (vertebral levels) in the spine model."""
        return len(self.trunk)

    def _interpret_affine(self, twist, trans):
        """Convert various input formats (twist+trans, vectorized, matrices) to absolute 4x4 affines."""
        bk = self._bk
        if twist is None:
            abs_aff = self.default_aff
        elif trans is None and twist.ndim == 3:  # parameters are absolute affine matrices
            abs_aff = bk.copy(twist)
        elif trans is None and twist.ndim == 1:  # parameters are vectorized
            abs_twist, abs_trans = self.parse_params(twist)
            abs_aff = transforms.make_aff(abs_twist, abs_trans, bk)
        else:
            assert twist.ndim == 2 and trans.ndim == 2
            abs_aff = transforms.make_aff(twist, trans, bk)
        return abs_aff

    def _auto_scale(self, abs_aff):
        """Compute scale factor matching posed spine length to default spine length."""
        bk = self._bk
        _, default_trans = transforms.extract_params(self.default_aff, bk)
        default_diffs = default_trans[1:] - default_trans[:-1]
        default_len = bk.sum(bk.linalg.norm(default_diffs, axis=-1))

        _, pose_trans = transforms.extract_params(abs_aff, bk)
        pose_diffs = pose_trans[1:] - pose_trans[:-1]
        pose_len = bk.sum(bk.linalg.norm(pose_diffs, axis=-1))

        s = pose_len / default_len
        return s

    def build_model(self, twist=None, trans=None, scale=None):
        """Pose all bone vertices using the given transforms and scale."""
        bk = self._bk
        abs_aff = self._interpret_affine(twist, trans)
        s = self._auto_scale(abs_aff) if scale is None else scale

        parts = []
        for jj in range(self.num_bones):
            parts.append(transforms.transform_points_forward(abs_aff[jj], self.verts[jj] * s, bk))
        return bk.concatenate(parts, axis=0)

    def build_landmarks(self, twist=None, trans=None, scale=1.0):
        """Pose cortical landmark points using the given transforms and scale."""
        bk = self._bk
        abs_aff = self._interpret_affine(twist, trans)
        s = self._auto_scale(abs_aff) if scale is None else scale
        assert 0.5 < float(s) < 2.0

        posed_lm = []
        for jj in range(self.num_bones):
            posed_lm.append(transforms.transform_points_forward(abs_aff[jj], self.landmarks[jj] * s, bk))

        return posed_lm

    def vectorize_params(self, twist=None, trans=None):
        """Flatten twist and translation arrays into a single 1D parameter vector."""
        bk = self._bk
        if trans is None:
            twist, trans = transforms.extract_params(twist, bk)
        return bk.concatenate((twist, trans), axis=1).flatten()

    def parse_params(self, vectorized):
        """Reshape 1D parameter vector back into (num_bones, 3) twist and trans arrays."""
        A = vectorized.reshape(self.num_bones, 6)
        return A[:, :3], A[:, 3:]
