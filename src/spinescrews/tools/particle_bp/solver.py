"""Particle belief propagation solver (D-PMP)."""

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import minimize

from .particles import ParticleSet
from .proposals import RandomWalkProposal, NeighborProposal

log = logging.getLogger(__name__)


@dataclass
class SolverConfig:
    """Configuration for ParticleBPSolver."""
    n_iterations: int = 30
    particles_per_node: int = 20
    total_budget: int | None = None   # adaptive redistribution if set
    n_min: int = 5                    # min particles per node (adaptive)
    n_max: int = 50                   # max particles per node (adaptive)
    refine_iterations: int = 4        # Nelder-Mead maxiter per particle
    random_walk_noise_std: np.ndarray | None = None  # physical units (rad/mm)
    random_walk_decay: float = 0.97
    neighbor_proposals: float = 0.5  # proportion of target node's particles
    neighbor_noise_std: np.ndarray | None = None  # physical units (rad/mm)
    convergence_tol: float = 1e-4
    convergence_patience: int = 5
    fixed_budget: dict[int, int] | None = None  # nodes with fixed particle count
    n_workers: int = 0                # thread pool size (0 = cpu_count)
    seed: int | None = None


@dataclass
class SolverResult:
    """Output of ParticleBPSolver.solve()."""
    best_params: dict          # {node_id: (ndim,) ndarray}
    best_costs: dict           # {node_id: float belief}
    global_cost: float
    n_iterations: int
    converged: bool
    history: list = field(default_factory=list)


class ParticleBPSolver:
    """Particle-based max-product belief propagation on a tree-structured graph.

    Implements D-PMP (Discrete-continuous Mixture of Particles Message Passing)
    from Zuffi & Black, CVPR 2015.

    Parameters
    ----------
    graph : FactorGraph
        The factor graph defining the problem.
    config : SolverConfig or None
        Solver configuration. Defaults to SolverConfig().
    """

    def __init__(self, graph, config=None):
        self.graph = graph
        self.config = config or SolverConfig()
        self.rng = np.random.default_rng(self.config.seed)

        # Build proposal strategies
        rw_std = (self.config.random_walk_noise_std
                  if self.config.random_walk_noise_std is not None
                  else 0.05 * graph.param_scale)
        self._random_walk = RandomWalkProposal(
            noise_std=rw_std,
            decay=self.config.random_walk_decay,
            rng=self.rng)
        self._neighbor_proposal = NeighborProposal(
            n_proposals=self.config.neighbor_proposals,
            noise_std=self.config.neighbor_noise_std,
            rng=self.rng)

        # Thread pool for parallel node evaluation
        n_workers = self.config.n_workers or max((os.cpu_count() or 4) - 2, 1)
        self._pool = ThreadPoolExecutor(max_workers=n_workers)

    def solve(self, initial_particles=None, initial_center=None,
              initial_noise=None, stage_callbacks=None):
        """Run D-PMP optimization.

        Parameters
        ----------
        initial_particles : dict or None
            {node_id: (n, ndim) array} of starting particles.
        initial_center : ndarray or None
            (n_nodes, ndim) center for initialization (e.g. from ICP).
        initial_noise : ndarray or None
            (ndim,) std for random perturbation around center.
        stage_callbacks : list of callable or None
            Each callback is called between stages with signature
            ``callback(best_params, psets)`` and may modify graph state
            (e.g. rebuild cost closures for coarse-to-fine).

        Returns
        -------
        SolverResult
        """
        g = self.graph
        cfg = self.config

        # Initialize particle sets
        psets = {}
        for i in range(g.n_nodes):
            psets[i] = ParticleSet(i, g.ndim, cfg.particles_per_node)

        if initial_particles is not None:
            for i, pts in initial_particles.items():
                psets[i].set_particles(pts)
        elif initial_center is not None:
            noise = initial_noise if initial_noise is not None else (
                0.1 * g.param_scale)
            for i in range(g.n_nodes):
                pts = initial_center[i] + self.rng.normal(
                    0, noise, size=(cfg.particles_per_node, g.ndim))
                # Always include the center itself
                pts[0] = initial_center[i]
                psets[i].set_particles(pts)

        # Evaluate initial local costs
        self._evaluate_costs(psets)

        # Get message schedule
        forward, backward = g.message_schedule()

        # Build stage list: initial stage + callback stages
        stages = [None]  # None = no callback for initial stage
        if stage_callbacks:
            stages.extend(stage_callbacks)

        history = []
        total_iters = 0
        converged = False

        for stage_idx, callback in enumerate(stages):
            if callback is not None:
                best_params = self._extract_best(psets)
                callback(best_params, psets)
                # Re-evaluate all costs after callback (closures may have changed)
                self._evaluate_costs(psets)

            patience_counter = 0
            no_improve_counter = 0
            prev_global = np.inf
            target_counts = {i: cfg.particles_per_node for i in range(g.n_nodes)}

            for it in range(cfg.n_iterations):
                # 1. Forward-backward message passing
                self._run_bp(forward, backward, psets)

                # 2. Compute beliefs
                for ps in psets.values():
                    ps.compute_beliefs()

                # 3. Resample: random walk + neighbor proposals
                n_proposals = 0
                n_nodes_improved = 0
                best_before = {i: float(np.min(psets[i].costs)) for i in range(g.n_nodes)}
                n_before = {i: psets[i].n for i in range(g.n_nodes)}
                for i in range(g.n_nodes):
                    rw = self._random_walk(psets[i], g, total_iters)
                    nb_psets = {nb: psets[nb] for nb in g.neighbors(i)}
                    np_prop = self._neighbor_proposal(psets[i], nb_psets, g, total_iters)
                    psets[i].add_particles(rw)
                    psets[i].add_particles(np_prop)
                    n_proposals += len(rw) + len(np_prop)

                # 4. Evaluate costs for new particles only
                self._evaluate_new_costs(psets)

                # Count nodes where proposals found a better local cost
                for i in range(g.n_nodes):
                    if float(np.min(psets[i].costs)) < best_before[i] - 1e-6:
                        n_nodes_improved += 1

                # 5. Refine newly generated particles
                refine_improvement = self._refine(psets, n_before)

                # 6. Re-run BP on expanded set
                self._run_bp(forward, backward, psets)

                # 7. Recompute beliefs
                for ps in psets.values():
                    ps.compute_beliefs()

                # Snapshot beliefs before selection for diagnostics
                pre_select_best = {i: float(np.min(ps.beliefs))
                                   for i, ps in psets.items()}
                pre_select_counts = {i: ps.n for i, ps in psets.items()}

                # 8. Adaptive budget + message-preserving selection
                target_counts = self._redistribute_budget(psets)
                self._select_particles(psets, target_counts)

                # 9. Log + convergence
                global_cost = sum(float(np.min(ps.beliefs)) for ps in psets.values())
                total_iters += 1

                # Per-node diagnostics
                per_node_best = {}
                per_node_local = {}
                per_node_msg = {}
                for i, ps in psets.items():
                    best_idx = int(np.argmin(ps.beliefs))
                    per_node_best[i] = float(ps.beliefs[best_idx])
                    per_node_local[i] = float(ps.costs[best_idx])
                    msg_total = sum(float(m[best_idx]) for m in ps.messages.values())
                    per_node_msg[i] = msg_total

                history.append({
                    'stage': stage_idx, 'iteration': it,
                    'global_cost': global_cost,
                    'per_node_best': per_node_best,
                    'per_node_local': per_node_local,
                    'per_node_msg': per_node_msg,
                })

                if it % 5 == 0 or it == cfg.n_iterations - 1:
                    rw_decay = cfg.random_walk_decay ** total_iters
                    log.info('D-PMP s%d i%02d: cost=%.4f  rw_decay=%.4f  '
                             'proposals: %d/%d nodes improved',
                             stage_idx, it, global_cost, rw_decay,
                             n_nodes_improved, g.n_nodes)
                    counts = [psets[i].n for i in range(g.n_nodes)]
                    log.info('  particles: %s  total=%d', counts, sum(counts))

                # Convergence: only a meaningful cost decrease resets patience.
                # Increases and tiny decreases all count as stagnation.
                improvement = prev_global - global_cost  # positive = decreased
                if improvement > cfg.convergence_tol:
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= cfg.convergence_patience:
                        log.info('D-PMP converged at stage %d iter %d '
                                 '(no improvement > %.1e for %d iters)',
                                 stage_idx, it, cfg.convergence_tol,
                                 patience_counter)
                        converged = True
                        break
                prev_global = global_cost

        # Extract final result
        best_params = self._extract_best(psets)
        best_costs = {i: float(np.min(ps.beliefs)) for i, ps in psets.items()}
        global_cost = sum(best_costs.values())

        return SolverResult(
            best_params=best_params,
            best_costs=best_costs,
            global_cost=global_cost,
            n_iterations=total_iters,
            converged=converged,
            history=history,
        )

    def _evaluate_costs(self, psets):
        """Evaluate local costs for all particles in all nodes (parallel)."""
        g = self.graph

        def _eval(item):
            i, ps = item
            if ps.n > 0:
                ps.costs = g.local_cost_fn(i, ps.particles)

        list(self._pool.map(_eval, psets.items()))

    def _evaluate_new_costs(self, psets):
        """Evaluate local costs only for new particles (parallel)."""
        g = self.graph

        def _eval_new(item):
            i, ps = item
            mask = ~np.isfinite(ps.costs)
            if mask.any():
                ps.costs[mask] = g.local_cost_fn(i, ps.particles[mask])

        list(self._pool.map(_eval_new, psets.items()))

    def _run_bp(self, forward, backward, psets):
        """Run forward and backward message passing."""
        for src, dst in forward:
            self._send_message(src, dst, psets)
        for src, dst in backward:
            self._send_message(src, dst, psets)

    def _send_message(self, src, dst, psets):
        """Send min-sum message from src to dst.

        msg[p] = min_q (pairwise(src_q, dst_p) + src_score[q])

        where src_score[q] = src.costs[q] + sum(messages to src, excluding dst).
        """
        g = self.graph
        src_ps = psets[src]
        dst_ps = psets[dst]

        if src_ps.n == 0 or dst_ps.n == 0:
            return

        # src_score = local cost + incoming messages (excluding from dst)
        src_score = src_ps.costs.copy()
        for nb_id, msg in src_ps.messages.items():
            if nb_id != dst:
                src_score += msg[:len(src_score)]

        # Pairwise cost matrix: (n_src, n_dst)
        n_src, n_dst = src_ps.n, dst_ps.n
        if g.pairwise_cost_matrix_fn is not None:
            pw_matrix = g.pairwise_cost_matrix_fn(
                src, src_ps.particles, dst, dst_ps.particles)
        else:
            pw_matrix = np.empty((n_src, n_dst))
            for q in range(n_src):
                for p in range(n_dst):
                    pw_matrix[q, p] = g.pairwise_cost_fn(
                        src, src_ps.particles[q], dst, dst_ps.particles[p])

        # msg[p] = min_q (pw_matrix[q, p] + src_score[q])
        msg = np.min(pw_matrix + src_score[:, None], axis=0)

        # Normalize to prevent drift (subtract minimum)
        msg -= msg.min()

        dst_ps.messages[src] = msg

    def _refine(self, psets, n_before):
        """Nelder-Mead refinement on newly generated particles (parallel).

        Each new particle (random walk or neighbor proposal) is refined once
        at generation time, then never again. This follows the D-PMP paper
        where local optimization pulls proposals toward data evidence.

        Returns dict with refinement stats, or None if skipped.
        """
        g = self.graph
        cfg = self.config
        if cfg.refine_iterations <= 0:
            return None

        best_per_node = self._extract_best(psets)

        def _refine_node(item):
            i, ps = item
            start = n_before[i]
            if start >= ps.n:
                return 0, 0, 0.0
            ni, na, td = 0, 0, 0.0
            for idx in range(start, ps.n):
                old_cost = ps.costs[idx]

                def _cost(p, _i=i, _best=best_per_node):
                    c = float(g.local_cost_fn(_i, p.reshape(1, -1))[0])
                    for nb in g.neighbors(_i):
                        c += g.pairwise_cost_fn(_i, p, nb, _best[nb])
                    return c

                result = minimize(_cost, ps.particles[idx],
                                  method='Nelder-Mead',
                                  options={'maxiter': cfg.refine_iterations,
                                           'adaptive': True,
                                           'xatol': 1e-4, 'fatol': 1e-5})
                refined = result.x
                new_cost = float(g.local_cost_fn(i, refined.reshape(1, -1))[0])
                ps.particles[idx] = refined
                ps.costs[idx] = new_cost
                na += 1
                delta = new_cost - old_cost
                if delta < -1e-6:
                    ni += 1
                td += delta
            return ni, na, td

        results = list(self._pool.map(_refine_node, psets.items()))
        n_improved = sum(r[0] for r in results)
        n_attempted = sum(r[1] for r in results)
        total_delta = sum(r[2] for r in results)

        return {'n_improved': n_improved, 'n_attempted': n_attempted,
                'total_delta': total_delta}

    def _redistribute_budget(self, psets):
        """Allocate particles proportional to best belief cost.

        Nodes with higher best-belief (worse fit) get more particles;
        well-converged nodes shed to minimum.  Nodes in ``fixed_budget``
        always keep their prescribed count and are excluded from the
        adaptive pool.
        """
        cfg = self.config
        n = self.graph.n_nodes
        fixed = cfg.fixed_budget or {}

        if cfg.total_budget is None:
            alloc = {i: cfg.particles_per_node for i in range(n)}
            for node_id, n_fixed in fixed.items():
                alloc[node_id] = n_fixed
            return alloc

        # Reserve budget for fixed nodes, distribute the rest adaptively
        fixed_total = sum(fixed.get(i, 0) for i in range(n) if i in fixed)
        adaptive_ids = [i for i in range(n) if i not in fixed]
        n_adaptive = len(adaptive_ids)

        if n_adaptive == 0:
            return {i: fixed[i] for i in range(n)}

        best_beliefs = np.array([psets[i].beliefs.min() if psets[i].n > 0 else 0.0
                                 for i in adaptive_ids])
        total = best_beliefs.sum()

        adaptive_budget = cfg.total_budget - fixed_total
        pool = adaptive_budget - n_adaptive * cfg.n_min
        if pool <= 0 or total < 1e-12:
            alloc_arr = np.full(n_adaptive, cfg.n_min)
        else:
            alloc_arr = cfg.n_min + pool * (best_beliefs / total)
            alloc_arr = np.clip(alloc_arr, cfg.n_min, cfg.n_max).astype(int)

            # Distribute remainder to worst-fit adaptive nodes
            remainder = adaptive_budget - int(alloc_arr.sum())
            if remainder > 0:
                order = np.argsort(-best_beliefs)
                for k in range(min(remainder, n_adaptive)):
                    if alloc_arr[order[k]] < cfg.n_max:
                        alloc_arr[order[k]] += 1

        alloc = {}
        for k, i in enumerate(adaptive_ids):
            alloc[i] = int(alloc_arr[k])
        for node_id, n_fixed in fixed.items():
            alloc[node_id] = n_fixed

        return alloc

    def _select_particles(self, psets, target_counts):
        """Message-preserving particle selection for all nodes.

        For each node t, builds the stacked message foundation matrix M_t
        (Pacheco et al. 2014) and selects particles that minimize worst-case
        message distortion to all neighbors.

        Falls back to belief-sorted selection for isolated nodes or nodes
        whose neighbors have no particles.
        """
        g = self.graph

        for t in range(g.n_nodes):
            n_keep = target_counts[t]
            ps_t = psets[t]

            if ps_t.n <= n_keep:
                continue

            neighbors = g.neighbors(t)

            # Fallback: no neighbors or all neighbors empty
            if not neighbors or all(psets[s].n == 0 for s in neighbors):
                idx = np.argsort(ps_t.beliefs)[:n_keep]
                ps_t.particles = ps_t.particles[idx]
                ps_t.costs = ps_t.costs[idx]
                for k in ps_t.messages:
                    ps_t.messages[k] = ps_t.messages[k][idx]
                ps_t.beliefs = ps_t.beliefs[idx]
                continue

            # Build stacked message foundation matrix.
            # For message t→s: M_ts[a, b] = pw(t, x_t^b, s, x_s^a) + src_score_ts[b]
            # where src_score_ts[b] = cost_t[b] + sum_{k != s} msg_kt[b].
            blocks = []
            for s in neighbors:
                ps_s = psets[s]
                if ps_s.n == 0:
                    continue

                # Source score excluding message from s
                src_score = ps_t.costs.copy()
                for nb_id, msg in ps_t.messages.items():
                    if nb_id != s:
                        src_score += msg[:len(src_score)]

                # Pairwise cost: (n_t, n_s)
                if g.pairwise_cost_matrix_fn is not None:
                    pw_matrix = g.pairwise_cost_matrix_fn(
                        t, ps_t.particles, s, ps_s.particles)
                else:
                    pw_matrix = np.empty((ps_t.n, ps_s.n))
                    for q in range(ps_t.n):
                        for p in range(ps_s.n):
                            pw_matrix[q, p] = g.pairwise_cost_fn(
                                t, ps_t.particles[q], s, ps_s.particles[p])

                # M_ts shape (n_s, n_t): transpose pw so rows=dst particles,
                # cols=src particles, then add src_score along columns
                M_ts = pw_matrix.T + src_score[np.newaxis, :]
                blocks.append(M_ts)

            if not blocks:
                idx = np.argsort(ps_t.beliefs)[:n_keep]
                ps_t.particles = ps_t.particles[idx]
                ps_t.costs = ps_t.costs[idx]
                for k in ps_t.messages:
                    ps_t.messages[k] = ps_t.messages[k][idx]
                ps_t.beliefs = ps_t.beliefs[idx]
                continue

            M_stacked = np.vstack(blocks)
            m_full = M_stacked.min(axis=1)

            ps_t.select_message_preserving(n_keep, M_stacked, m_full)

    def _extract_best(self, psets):
        """Extract best particle per node (lowest belief)."""
        best = {}
        for i, ps in psets.items():
            if ps.n > 0:
                best[i] = ps.particles[np.argmin(ps.beliefs)].copy()
            else:
                best[i] = np.zeros(self.graph.ndim)
        return best
