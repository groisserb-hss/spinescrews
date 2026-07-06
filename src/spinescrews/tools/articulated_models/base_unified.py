"""Unified base class for articulated kinematic-chain models.

``Articulated`` defines the kinematic tree (parent-index ``trunk``), selects the
array backend (numpy or torch), and converts between relative and absolute
affine joint parameters. ``Spine`` (spine.py) is the concrete subclass the
pipeline uses.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import bg3dtools.transforms_unified as transforms
from bg3dtools.pytorch import TorchBackend


@dataclass
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


