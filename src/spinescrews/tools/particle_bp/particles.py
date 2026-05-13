"""Per-node particle set with diversity-preserving selection."""

import logging
import numpy as np

log = logging.getLogger(__name__)


class ParticleSet:
    """Manages a set of particles (candidate configurations) for one graph node.

    Parameters
    ----------
    node_id : int
        Which graph node this set belongs to.
    ndim : int
        Dimensionality of each particle.
    max_particles : int
        Capacity (used for pre-allocation).
    """

    def __init__(self, node_id, ndim, max_particles):
        self.node_id = node_id
        self.ndim = ndim
        self.max_particles = max_particles
        self.particles = np.empty((0, ndim), dtype=np.float64)
        self.costs = np.empty(0, dtype=np.float64)
        self.messages = {}       # {neighbor_id: (n,) array}
        self.beliefs = np.empty(0, dtype=np.float64)

    @property
    def n(self):
        return len(self.particles)

    def set_particles(self, particles, costs=None):
        """Replace all particles. Resets messages and beliefs."""
        self.particles = np.asarray(particles, dtype=np.float64)
        if costs is not None:
            self.costs = np.asarray(costs, dtype=np.float64)
        else:
            self.costs = np.full(self.n, np.inf)
        self.messages = {}
        self.beliefs = self.costs.copy()

    def add_particles(self, new_particles, new_costs=None):
        """Append new particles (pre-selection)."""
        new_particles = np.asarray(new_particles, dtype=np.float64)
        if len(new_particles) == 0:
            return
        self.particles = np.vstack([self.particles, new_particles])
        if new_costs is not None:
            self.costs = np.concatenate([self.costs, np.asarray(new_costs, dtype=np.float64)])
        else:
            self.costs = np.concatenate([self.costs, np.full(len(new_particles), np.inf)])
        # Extend existing messages with inf for new particles
        for k in self.messages:
            self.messages[k] = np.concatenate([
                self.messages[k], np.zeros(len(new_particles))])
        self.beliefs = np.concatenate([self.beliefs, np.full(len(new_particles), np.inf)])

    def compute_beliefs(self):
        """Recompute beliefs = costs + sum(incoming messages)."""
        self.beliefs = self.costs.copy()
        for msg in self.messages.values():
            self.beliefs += msg[:len(self.beliefs)]

    def select_diverse(self, n_keep, bounds_range):
        """Farthest-point sampling weighted by belief quality.

        1. Normalize particles by bounds_range (so rotation/translation contribute equally).
        2. Start with the lowest-belief (best) particle.
        3. Greedily add the particle maximizing
           ``min_dist_to_selected * exp(-belief / temperature)``
           where temperature = median(beliefs) - min(beliefs) + eps.
        4. Discard the rest.

        Parameters
        ----------
        n_keep : int
            Number of particles to retain.
        bounds_range : ndarray, shape (ndim,)
            Per-dimension range for normalization.
        """
        if self.n <= n_keep:
            return

        # Normalize
        scale = np.maximum(bounds_range, 1e-12)
        normed = self.particles / scale

        # Temperature from belief spread
        bmin = np.min(self.beliefs)
        bmed = np.median(self.beliefs)
        temperature = max(bmed - bmin, 1e-8)

        # Quality weight: prefer low belief
        quality = np.exp(-(self.beliefs - bmin) / temperature)

        # Greedy farthest-point
        selected = [int(np.argmin(self.beliefs))]
        min_dist = np.full(self.n, np.inf)
        for _ in range(n_keep - 1):
            last = normed[selected[-1]]
            d = np.sum((normed - last) ** 2, axis=1)
            min_dist = np.minimum(min_dist, d)
            score = np.sqrt(min_dist) * quality
            score[selected] = -1
            selected.append(int(np.argmax(score)))

        idx = np.array(selected)

        # Warn if selection loses the best particle's belief
        # (should not happen — best is always selected first)
        best_before = np.min(self.beliefs)
        best_after = self.beliefs[idx].min()
        if best_after > best_before + 1e-8:
            log.warning('select_diverse node %d: best belief degraded %.4f -> %.4f '
                        '(lost best particle!)', self.node_id, best_before, best_after)

        n_discarded = self.n - n_keep
        discarded_mask = np.ones(self.n, dtype=bool)
        discarded_mask[idx] = False
        n_good_discarded = int(np.sum(
            self.beliefs[discarded_mask] < np.median(self.beliefs[idx])))
        if n_good_discarded > n_keep // 4:
            log.debug('select_diverse node %d: discarded %d particles with '
                      'belief < selected median (aggressive pruning)',
                      self.node_id, n_good_discarded)

        self.particles = self.particles[idx]
        self.costs = self.costs[idx]
        for k in self.messages:
            self.messages[k] = self.messages[k][idx]
        self.beliefs = self.beliefs[idx]

    def select_message_preserving(self, n_keep, M_stacked, m_full):
        """Message-preserving particle selection (Pacheco et al. 2014).

        Greedily selects a subset of particles that minimizes the worst-case
        distortion of outgoing messages to all neighbors (Eq. 12-14,
        adapted to the min-sum semiring).

        The best-belief particle is always included first to ensure monotonic
        convergence of the overall solution.

        Parameters
        ----------
        n_keep : int
            Number of particles to retain.
        M_stacked : ndarray, shape (total_neighbor_particles, n)
            Vertically stacked message foundation matrices for all neighbors.
            ``M_stacked[a, b] = pw(t, x_t^b, s, x_s^a) + src_score_ts[b]``
        m_full : ndarray, shape (total_neighbor_particles,)
            Full messages computed from all particles: row-wise min of
            M_stacked.
        """
        if self.n <= n_keep:
            return

        n_t = self.n

        # Force-include the best-belief particle
        best_idx = int(np.argmin(self.beliefs))
        selected = [best_idx]
        m_approx = M_stacked[:, best_idx].copy()

        unselected = np.ones(n_t, dtype=bool)
        unselected[best_idx] = False

        for _ in range(n_keep - 1):
            # Worst-approximated row (largest gap between approx and full)
            error = m_approx - m_full
            a_k = int(np.argmax(error))

            # Best unselected particle for that row
            row = M_stacked[a_k, :]
            row_masked = np.where(unselected, row, np.inf)
            b_k = int(np.argmin(row_masked))

            selected.append(b_k)
            unselected[b_k] = False
            m_approx = np.minimum(m_approx, M_stacked[:, b_k])

        final_error = float(np.max(m_approx - m_full))
        log.debug('select_msg node %d: kept %d/%d, max msg error=%.4f',
                  self.node_id, n_keep, n_t, final_error)

        idx = np.array(selected)
        self.particles = self.particles[idx]
        self.costs = self.costs[idx]
        for k in self.messages:
            self.messages[k] = self.messages[k][idx]
        self.beliefs = self.beliefs[idx]
