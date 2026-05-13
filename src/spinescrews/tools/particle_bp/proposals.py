"""Proposal strategies for generating new particle candidates."""

import numpy as np


class RandomWalkProposal:
    """Generate proposals by perturbing existing particles with decaying noise.

    Parameters
    ----------
    noise_std : ndarray, shape (ndim,)
        Initial per-dimension noise in physical units (radians / mm).
    decay : float
        Multiplicative decay per iteration.
    rng : np.random.Generator or None
    """

    def __init__(self, noise_std, decay=0.93, rng=None):
        self.noise_std = np.asarray(noise_std, dtype=np.float64)
        self.decay = decay
        self.rng = rng or np.random.default_rng()

    def __call__(self, pset, graph, iteration):
        """Generate one proposal per existing particle.

        Returns
        -------
        proposals : ndarray, shape (pset.n, ndim)
        """
        sigma = self.noise_std * (self.decay ** iteration)
        noise = self.rng.normal(0, sigma, size=pset.particles.shape)
        return pset.particles + noise


class NeighborProposal:
    """Generate proposals by sampling relative transforms around neighbor particles.

    For each neighbor: pick source particles, apply the neighbor proposal function
    to get the model-predicted position, then add Gaussian noise in physical units.
    This samples diverse relative configurations, following Zuffi & Black (CVPR 2015).

    Parameters
    ----------
    n_proposals : int or float
        If >= 1, absolute number of proposals per neighbor.
        If < 1, proportion of the target node's current particle count
        (e.g. 0.5 means half of target's particles come from each neighbor).
    noise_std : ndarray or None
        Per-dimension standard deviation in physical units (radians for rotation,
        mm for translation).
    rng : np.random.Generator or None
    """

    def __init__(self, n_proposals=0.5, noise_std=None, rng=None):
        self.n_proposals = n_proposals
        self.noise_std = np.asarray(noise_std) if noise_std is not None else None
        self.rng = rng or np.random.default_rng()

    def __call__(self, pset, neighbor_psets, graph, iteration):
        """Generate proposals from neighbor configurations.

        Parameters
        ----------
        pset : ParticleSet
            Target node.
        neighbor_psets : dict
            {neighbor_id: ParticleSet}
        graph : FactorGraph

        Returns
        -------
        proposals : ndarray, shape (m, ndim)
        """
        if graph.neighbor_proposal_fn is None:
            return np.empty((0, pset.ndim), dtype=np.float64)

        if self.noise_std is not None:
            sigma = self.noise_std
        else:
            sigma = 0.02 * graph.param_scale

        # Compute number of proposals per neighbor
        if self.n_proposals < 1:
            n_per_nb = max(1, int(round(self.n_proposals * pset.n)))
        else:
            n_per_nb = int(self.n_proposals)

        proposals = []
        for nb_id, nb_pset in neighbor_psets.items():
            if nb_pset.n == 0:
                continue
            n_source = min(n_per_nb, nb_pset.n)
            source_idx = self.rng.choice(nb_pset.n, size=n_source,
                                         replace=n_source > nb_pset.n)
            for idx in source_idx:
                nb_params = nb_pset.particles[idx]
                base = graph.neighbor_proposal_fn(pset.node_id, nb_id, nb_params)
                noise = self.rng.normal(0, sigma)
                proposals.append(base + noise)

        if proposals:
            return np.array(proposals)
        return np.empty((0, pset.ndim), dtype=np.float64)
