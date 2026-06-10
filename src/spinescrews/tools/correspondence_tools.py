"""Spectral + anatomical correspondence between a vertebra and its template.

`SpectralDescriptor` augments spectral (functional-map) features with anatomical signatures for
robust vertebra-to-template matching; `load_vertebral_template()` loads and caches the
preprocessed per-level template meshes and labels.
"""

from os.path import isfile

import numpy as np
from spectral_match.correspondence.feature_descriptors import DescriptorClass
from spectral_match.pipeline import FunctionalMapper, Mesh
from spectral_match import SigConfig


class SpectralDescriptor(DescriptorClass):
    def __init__(self, sig_config: SigConfig, scale=50.):
        """Initialize descriptor with spectral + anatomical signatures for vertebra matching."""
        super().__init__(sig_config)
        self.num_extra = 3
        self.scale = scale

    def __call__(self, mesh: Mesh):
        """Compute concatenated spectral and anatomical signatures for a mesh."""
        base_sigs = super().__call__(mesh)
        anatomical_sigs = self.scale * self.anatomical_signature(mesh)
        return np.concatenate([base_sigs, anatomical_sigs], axis=-1)

    @staticmethod
    def anatomical_signature(mesh: Mesh):
        """ compute anatomical signature
        NOTE: this function requires that the mesh has been previously oriented!
        the origin should be in the center of the spinal canal and coordinate system is RAS:
        x = right
        y = anterior
        z = superior
        """

        # using cylindrical coordinates as a proxy for anatomical features
        rho = np.sqrt(np.sum(mesh.v[:, :2] ** 2, axis=1))
        rho /= np.max(rho)

        l = mesh.v[:, 0].copy()
        l -= np.min(l)
        l /= np.max(l)

        z = mesh.v[:, 2].copy()
        z -= np.min(z)
        z /= np.max(np.abs(z))
        return np.column_stack((rho, l, z))


def load_vertebral_template(ply_file, mapper: FunctionalMapper) -> (Mesh, Mesh):
    """Load a template PLY and its preprocessed spectral descriptors (cached as .npz)."""
    raw = Mesh.from_file(ply_file, normalize=True)
    processed_file = ply_file.replace('.ply', '.npz')
    if isfile(processed_file):
        processed = Mesh.from_file(processed_file)
    else:
        processed = mapper.preprocess_mesh(raw.v, raw.f)
        processed.save_np(processed_file)

    return raw, processed
