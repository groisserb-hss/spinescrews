"""
Unified base classes for articulated models supporting both numpy and pytorch backends.

Usage:
    # Numpy (default - backward compatible)
    model = Linear(trunk, faces, verts, bweights, shapespace)

    # PyTorch
    model = Linear(trunk, faces, verts, bweights, shapespace, backend='torch')

For training with gradient descent, use TorchLinear which wraps parameters
as nn.Parameters.
"""
from __future__ import annotations
from os.path import join, splitext, sep
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union, Iterator

import numpy as np
import bg3dtools.transforms_unified as transforms
from bg3dtools.pytorch import TorchBackend, ArrayLike, to_numpy, to_torch


# =============================================================================
# Abstract Pose and Shape dataclasses
# =============================================================================

@dataclass
class Pose(ABC):
    """
    Abstract base for pose parameters.

    Subclasses should define model-specific pose attributes.
    All implementations must support iteration for backward compatibility
    with tuple unpacking.
    """

    @abstractmethod
    def __iter__(self) -> Iterator:
        """Allow tuple-style unpacking."""
        pass

    def to_numpy(self) -> 'Pose':
        """Convert all arrays to numpy. Override in subclass."""
        return self

    def to_torch(self, device=None, dtype=None) -> 'Pose':
        """Convert all arrays to torch tensors. Override in subclass."""
        return self

    def add_batch_dim(self) -> 'Pose':
        """Add batch dimension if not present. Override in subclass."""
        return self


@dataclass
class Shape(ABC):
    """
    Abstract base for shape parameters.

    Subclasses should define model-specific shape attributes.
    All implementations must support iteration for backward compatibility
    with tuple unpacking.
    """

    @abstractmethod
    def __iter__(self) -> Iterator:
        """Allow tuple-style unpacking."""
        pass

    def to_numpy(self) -> 'Shape':
        """Convert all arrays to numpy. Override in subclass."""
        return self

    def to_torch(self, device=None, dtype=None) -> 'Shape':
        """Convert all arrays to torch tensors. Override in subclass."""
        return self

    def add_batch_dim(self) -> 'Shape':
        """Add batch dimension if not present. Override in subclass."""
        return self


class Articulated(ABC):
    """
    Base class for articulated models.

    The backend parameter determines whether to use numpy or torch for array operations.
    For a given instance, the backend is fixed.
    """

    def __init__(self, trunk, backend='numpy'):
        """Initialize kinematic chain from parent-index list and select array backend."""
        ABC.__init__(self)
        assert isinstance(trunk, list)
        self.trunk = trunk.copy()
        self._backend_name = backend

        # Set up backend
        if backend == 'numpy':
            self._bk = np
        else:
            self._bk = TorchBackend()

        self._branch_level = np.array([self.__follow_to_root(trunk, tt, 0) for tt in range(len(trunk))])
        assert np.all(self._branch_level >= 0)

    @property
    def backend(self):
        """The array backend (numpy module or TorchBackend instance)."""
        return self._bk

    @property
    def backend_name(self):
        """String name of the backend ('numpy' or 'torch')."""
        return self._backend_name

    @property
    def nJ(self):
        """Number of joints (bones) in the kinematic chain."""
        return len(self.trunk)

    @classmethod
    @abstractmethod
    def with_backend(cls, model: 'Articulated', backend: str) -> 'Articulated':
        """
        Create a new instance with a different backend.

        This classmethod creates a copy of the model using the specified backend.
        All model parameters are converted to the target backend's array type.

        Parameters
        ----------
        model : Articulated
            Source model instance to copy from.
        backend : str
            Target backend: 'numpy' or 'torch'.

        Returns
        -------
        Articulated
            New model instance with the specified backend.

        Examples
        --------
        >>> numpy_model = SMPL.init_from_file('model.mat', backend='numpy')
        >>> torch_model = SMPL.with_backend(numpy_model, backend='torch')
        """
        pass

    @abstractmethod
    def nV(self):
        """Total number of vertices in the model."""

    @abstractmethod
    def build_model(self, pose, shape):
        """Pose and shape the model, returning final vertex positions."""

    def rel_params_to_aff(self, rel_twist=None, rel_trans=None):
        """
        Convert relative twist/translation parameters to absolute affine transforms.

        Args:
            rel_twist: [nJ, 3] axis-angle relative rotations
            rel_trans: [nJ, 3] relative translations

        Returns:
            abs_affs: [nJ, 4, 4] absolute affine transforms
        """
        bk = self._bk
        num_bones = len(self.trunk)

        if rel_twist is None:
            rel_twist = bk.zeros((num_bones, 3))
        if rel_trans is None:
            rel_trans = bk.zeros((num_bones, 3))

        return transforms.rel_params_to_aff(self.trunk, rel_twist, rel_trans, bk)

    def aff_to_rel_params(self, abs_affs):
        """
        Convert absolute affine transforms to relative twist/translation parameters.

        Args:
            abs_affs: [nJ, 4, 4] absolute affine transforms

        Returns:
            rel_theta: [nJ, 3] axis-angle relative rotations
            rel_tran: [nJ, 3] relative translations
        """
        return transforms.aff_to_rel_params(self.trunk, abs_affs, self._bk)

    @staticmethod
    def __follow_to_root(trunk, entry_point, count):
        """Recursively count hops from entry_point to root (-1) in the kinematic tree."""
        if count > len(trunk):
            return -1
        if entry_point == -1:
            return count - 1
        return Articulated.__follow_to_root(trunk, trunk[entry_point], count + 1)


class Linear(Articulated):
    """
    Basic linear model with backend support:
        - Articulations modeled with Linear Blend Skinning
        - Low-dimensional shape space implemented with Principal Component Analysis

    Args:
        trunk: nJ-element list of segment connectivity
        faces: [nF x 3] array connectivity
        verts: [nV x 3] array vertex positions in cartesian coordinates
        bweights: [nV x nJ] blend weights
        shapespace: [nV x 3 x nL] PCA shape basis
        backend: 'numpy' (default) or 'torch'
    """

    def __init__(self, trunk, faces, verts, bweights, shapespace, backend='numpy'):
        """Initialize Linear model with mesh, blend weights, and PCA shape space."""
        Articulated.__init__(self, trunk, backend)
        bk = self._bk

        # Store model data - copy to avoid mutation
        if backend == 'numpy':
            self.verts = verts.copy()
            self.faces = faces.copy()
            self.blend_weights = bweights.copy()
            self.shapespace = shapespace.copy()
        else:
            self.verts = bk.array(verts)
            self.faces = faces.copy()  # faces stay as numpy for indexing
            self.blend_weights = bk.array(bweights)
            self.shapespace = bk.array(shapespace)

        # Face weights computed from blend weights (always numpy for now)
        self.face_weights = (bweights[faces[:, 0]] + bweights[faces[:, 1]] + bweights[faces[:, 2]]) / 3

    def map_model(self, new_faces: np.ndarray, joint_subset: list, vertex_map):
        """Map model parameters to a new (aligned) mesh topology."""
        nW, nV = vertex_map.shape
        assert nV == self.nV

        # extract subset of joints to keep
        subset_arr = np.array(joint_subset)
        new_trunk = [-1] + [np.where(subset_arr == self.trunk[jj])[0][0] for jj in joint_subset[1:]]

        # Convert to numpy for mapping operations if needed
        verts = self._bk.to_numpy(self.verts) if self._backend_name == 'torch' else self.verts
        bweights = self._bk.to_numpy(self.blend_weights) if self._backend_name == 'torch' else self.blend_weights
        shapespace = self._bk.to_numpy(self.shapespace) if self._backend_name == 'torch' else self.shapespace

        new_verts = vertex_map @ verts
        new_bweights = vertex_map @ bweights[:, joint_subset]

        # map shapespace
        new_shapespace = np.zeros([nW, 3, self.nL], dtype=shapespace.dtype)
        for ll in range(self.nL):
            new_shapespace[:, :, ll] = vertex_map @ shapespace[:, :, ll]

        return new_trunk, new_faces, new_verts, new_bweights, new_shapespace

    @property
    def nV(self):
        """Total number of vertices in the template mesh."""
        return self.verts.shape[0]

    @property
    def nL(self):
        """Number of PCA shape basis components."""
        return self.shapespace.shape[-1]

    @abstractmethod
    def save(self, filename):
        """Serialize model to disk."""

    @abstractmethod
    def get_joints(self, verts, init=False):
        """Compute joint positions from vertex positions."""

    @abstractmethod
    def compress_shape(self, shapeV):
        """Project full vertex displacements into PCA latent coefficients."""

    def expand_shape(self, intrinsic_shape=None, center_root=True):
        """
        Convert latent shape to full mesh vertices using PCA latent space.

        Parameters
        ----------
        intrinsic_shape : tuple or list, optional
            [latent_coeffs, joints] where latent_coeffs is either:
            - [k] or [batch, k] latent shape coefficients (k <= nL, zero-padded if shorter)
            - [nV, 3] or [batch, nV, 3] full vertex positions
        center_root : bool, optional
            If True, center vertices on root joint. Default is True.

        Returns
        -------
        shapeV : ArrayLike
            [nV, 3] or [batch, nV, 3] vertex positions.
        """
        bk = self._bk

        if intrinsic_shape is None:
            return bk.copy(self.verts)

        _ = iter(intrinsic_shape)  # latent_shape must be wrapped in iterator
        subj_shape = intrinsic_shape[0]

        # Check if batched and determine if it's full vertices or latent coefficients
        is_batched = False
        is_full_verts = False

        if subj_shape.ndim == 1:
            # [k] latent shape (k <= nL)
            is_batched = False
            is_full_verts = False
        elif subj_shape.ndim == 2:
            if subj_shape.shape[-1] == 3 and subj_shape.shape[0] == self.nV:
                # [nV, 3] full vertices
                is_full_verts = True
            else:
                # [batch, k] batched latent shape (k <= nL)
                is_batched = True
        elif subj_shape.ndim == 3:
            # [batch, nV, 3] batched full vertices
            is_batched = True
            is_full_verts = True

        if is_full_verts:
            shapeV = subj_shape
        else:
            # Pad betas to nL if shorter
            input_size = subj_shape.shape[-1]
            if input_size < self.nL:
                pad_size = self.nL - input_size
                padding = ((0, 0), (0, pad_size)) if is_batched else ((0, pad_size),)
                subj_shape = bk.pad(subj_shape, padding, mode='constant', constant_values=0)

            if is_batched:
                # Batched PCA expansion: verts + einsum(shapespace, latent_coeffs)
                # shapespace: [nV, 3, nL], subj_shape: [batch, nL] -> [batch, nV, 3]
                shapeV = bk.expand_dims(self.verts, axis=0) + bk.einsum('vdl,bl->bvd', self.shapespace, subj_shape)
            else:
                # Unbatched PCA expansion
                shapeV = self.verts + bk.einsum('vdl,l->vd', self.shapespace, subj_shape)

        # center on root joint
        if center_root:
            joints = self.get_joints(shapeV) if intrinsic_shape[-1] is None else intrinsic_shape[-1]
            if is_batched:
                shapeV = shapeV - joints[:, 0:1, :]
            else:
                shapeV = shapeV - joints[0]

        return shapeV

    def project_to_betas(self, verts):
        """Project full vertex positions to PCA shape coefficients.

        Inverse of ``expand_shape``: given full verts, recover the PCA betas
        by projecting the displacement from the template onto each shape basis
        vector.

        Parameters
        ----------
        verts : ArrayLike
            [nV, 3] or [batch, nV, 3] vertex positions.

        Returns
        -------
        betas : ArrayLike
            [nL] or [batch, nL] shape coefficients.
        """
        bk = self._bk
        delta = verts - self.verts
        norms_sq = bk.sum(self.shapespace ** 2, axis=(0, 1))  # [nL]
        if delta.ndim == 2:
            return bk.einsum('vdl,vd->l', self.shapespace, delta) / norms_sq
        else:
            return bk.einsum('vdl,bvd->bl', self.shapespace, delta) / norms_sq

    def linear_blend_skin(self, rest_joints, thetas, points, bweights=None, init_theta=None, init_joints=None):
        """
        Apply Linear Blend Skinning to transform points from rest pose to posed position.

        Args:
            rest_joints: [nJ x 3] joint positions in rest pose
            thetas: [nJ x 3] axis-angle rotations for each joint
            points: [nV x 3] vertices to transform
            bweights: Optional [nV x nJ] blend weights (defaults to self.blend_weights)
            init_theta: Optional [nJ x 3] initial pose to transform from
            init_joints: Optional [nJ x 3] initial joint positions

        Returns:
            posed_points: [nV x 3] transformed vertices
        """
        bk = self._bk

        if bweights is None:
            bweights = self.blend_weights
        assert points.shape[0] == bweights.shape[0]

        # Compute relative joint offsets (avoid in-place for autograd)
        if self._backend_name == 'torch':
            import torch
            root = rest_joints[0:1, :]
            offsets = rest_joints[1:, :] - rest_joints[self.trunk[1:], :]
            rel_joints = torch.cat([root, offsets], dim=0)
        else:
            rel_joints = bk.copy(rest_joints)
            rel_joints[1:, :] = rest_joints[1:, :] - rest_joints[self.trunk[1:], :]

        # Get rest and posed affine transforms
        rest_G = transforms.rel_params_to_aff(self.trunk, init_theta, rel_joints, bk)
        pose_G = transforms.rel_params_to_aff(self.trunk, thetas, rel_joints, bk)

        if init_joints is not None:
            points = points - init_joints[0:1, :]
            init_joints = init_joints - init_joints[0:1, :]
            rest_G[:, :3, -1] = init_joints

        # Compose transforms: pose_G @ inv(rest_G)
        composed_G = [pose_G[ii] @ transforms.inverse(rest_G[ii], bk) for ii in range(self.nJ)]
        composed_G = bk.stack(composed_G, axis=-1)

        # Weight and apply transforms
        if self._backend_name == 'numpy':
            G_w = composed_G.dot(bweights.T)
            v_h = np.hstack((points, np.ones((points.shape[0], 1))))
            posed_homog = (G_w[:, 0, :] * v_h[:, 0] + G_w[:, 1, :] * v_h[:, 1] +
                          G_w[:, 2, :] * v_h[:, 2] + G_w[:, 3, :] * v_h[:, 3]).T
        else:
            import torch
            G_w = bk.einsum('nmj,vj->vnm', composed_G, bweights)
            v_h = torch.nn.functional.pad(points, (0, 1), value=1)
            posed_homog = bk.einsum('vnm,vm->vn', G_w, v_h)

        posed_homog = posed_homog / posed_homog[:, 3, None]
        return posed_homog[:, 0:3]

    def inverse_kinematics(self, vertices):
        """
        Estimate pose based on surface mesh (used to initialize gradient descent).
        NOTE: This method requires numpy - uses rigid_reg which has no torch equivalent.

        Args:
            vertices: [nV x 3] posed vertex positions

        Returns:
            rel_theta: [nJ x 3] estimated relative rotations
            rel_tran: [nJ x 3] estimated relative translations
        """
        # This is numpy-only as it uses rigid_reg
        if self._backend_name == 'torch':
            verts = self._bk.to_numpy(self.verts)
            bweights = self._bk.to_numpy(self.blend_weights)
            vertices = self._bk.to_numpy(vertices)
        else:
            verts = self.verts
            bweights = self.blend_weights

        num_bones = len(self.trunk)
        abs_affs = np.zeros((num_bones, 4, 4))

        for jj in range(num_bones):
            segW = bweights[:, jj]
            sigW = segW[segW > .01]
            seg_idx = segW > np.percentile(sigW, 75)

            source = verts[seg_idx, :]
            dest = vertices[seg_idx, :]
            abs_affs[jj] = transforms.rigid_reg(source, dest)

        return transforms.aff_to_rel_params(self.trunk, abs_affs, np)

    @staticmethod
    def rest_joints(trunk, twist, posed_joints):
        """
        Compute rest pose joint locations from posed joints and twist parameters.

        Args:
            trunk: List of parent indices
            twist: [nJ x 3] twist vectors of segments
            posed_joints: [nJ x 3] absolute positions of joints

        Returns:
            rel_joints: [nJ x 3] relative joint positions
            rest_joints: [nJ x 3] absolute joint locations in rest pose
        """
        rotAff = transforms.rel_params_to_aff(trunk, twist, None, np)
        absTwist, _ = transforms.extract_params(rotAff, np)
        absAff = transforms.make_aff(absTwist, posed_joints, np)

        _, rel_joints = transforms.aff_to_rel_params(trunk, absAff, np)
        restAff = transforms.rel_params_to_aff(trunk, None, rel_joints, np)
        _, rest_joints_out = transforms.extract_params(restAff, np)

        return rel_joints, rest_joints_out

    @staticmethod
    def relative_joints(trunk, joints, theta=None):
        """
        Compute relative joint positions from absolute positions.

        Args:
            trunk: List of parent indices
            joints: [nJ x 3] absolute joint positions
            theta: [nJ x 3] joint rotations (default: zeros)

        Returns:
            rel_joints: [nJ x 3] relative joint positions
        """
        if theta is None:
            theta = np.zeros(joints.shape)

        abs_rotation = transforms.rel_params_to_aff(trunk, theta, None, np)
        abs_trans = transforms.make_aff(None, joints, np)

        rel_joints = np.zeros((len(trunk), 3))
        for ii in range(1, len(trunk)):
            abs_aff = transforms.inverse(abs_rotation[ii], np) @ transforms.inverse(abs_trans[trunk[ii]], np)
            rel_joints[ii] = transforms.transform_points_forward(abs_aff, joints[ii:ii+1, :], np).flatten()

        return rel_joints


class SubjData(ABC):
    """
    Data structure to hold subject registration data.
    """

    def __init__(self, scan_files, nV, ext=None, num_rel_paths=3):
        """Initialize subject data from scan file paths and vertex count."""
        ABC.__init__(self)

        root_dir, rel_paths, ext, name = SubjData.parse_file_paths(scan_files, ext, num_rel_paths)

        self.root_dir = root_dir
        self.rel_paths = rel_paths
        self.scan_ext = ext
        self.name = name

        self.nV = nV
        self.shapeV = np.zeros([nV, 3], dtype=np.float32)

        # scan-level data
        self.regV = np.zeros([self.nS, nV, 3], dtype=np.float32)
        self.modelV = np.zeros([self.nS, self.nV, 3], dtype=np.float32)

    @property
    def scan_files(self):
        """Reconstruct full scan file paths from root_dir + relative paths + extension."""
        return [join(self.root_dir, p + self.scan_ext) for p in self.rel_paths]

    @staticmethod
    def parse_file_paths(file_paths: list[str], ext=None, num_rel_paths=3):
        """
        Parse scan file paths into components.
        Expects scans saved in SAR-style directory structure:
        /path/to/data/subj_ID/pose_ID/scan_name.suff
        """
        if len(file_paths) == 0:
            return '', [], '', ''
        scan_path = file_paths[0]

        ext = splitext(scan_path)[1] if ext is None else ext
        for file in file_paths:
            assert file.endswith(ext)
        scan_list = [file.replace(ext, '') for file in file_paths]

        split_path = scan_path.split(sep)
        root_dir = join(*split_path[:-num_rel_paths])
        name = split_path[-num_rel_paths]
        if scan_path[0] == sep and root_dir[0] != sep:
            root_dir = sep + root_dir
        for file in scan_list:
            assert file.startswith(root_dir)

        rel_paths = [file.replace(root_dir + sep, '') for file in scan_list]
        for rel_path in rel_paths:
            assert rel_path.startswith(name)
            assert len(rel_path.split(sep)) == num_rel_paths

        for rel_path, orig_path in zip(rel_paths, file_paths):
            assert join(root_dir, rel_path + ext) == orig_path

        return root_dir, rel_paths, ext, name

    @property
    def nS(self):
        """Number of scans for this subject."""
        return len(self.rel_paths)

    @property
    @abstractmethod
    def nW(self):
        """Number of vertices in the registered (warped) mesh."""

    @property
    @abstractmethod
    def nL(self):
        """Number of PCA shape components."""

    @abstractmethod
    def subset(self, slice):
        """Return a SubjData containing only the specified scan subset."""

    @abstractmethod
    def get_pose(self, slice):
        """Retrieve pose parameters for the given scan index."""

    @abstractmethod
    def set_pose(self, new_pose, slice):
        """Store pose parameters for the given scan index."""

    @property
    @abstractmethod
    def intrinsic(self):
        """Get the subject's intrinsic shape parameters."""

    @intrinsic.setter
    @abstractmethod
    def intrinsic(self, new_shape):
        """Set the subject's intrinsic shape parameters."""

    @abstractmethod
    def write_to_disk(self, base_dir, suffix):
        """Serialize all subject data (shape + per-scan poses) to disk."""

    @staticmethod
    @abstractmethod
    def load(root_dir, name, tag, suffix='.obj'):
        """Load a SubjData instance from previously saved files."""

    @staticmethod
    @abstractmethod
    def write_reg_ply(*args, **kwargs):
        """Write a registered mesh to PLY format."""

    @staticmethod
    @abstractmethod
    def write_subj_ply(*args, **kwargs):
        """Write a subject-specific mesh to PLY format."""

    @staticmethod
    @abstractmethod
    def read_reg_ply(infile):
        """Read a registered mesh from PLY format."""

    @staticmethod
    @abstractmethod
    def read_subj_ply(infile):
        """Read a subject-specific mesh from PLY format."""
