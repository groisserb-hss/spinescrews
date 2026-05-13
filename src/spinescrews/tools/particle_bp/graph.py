"""Factor graph topology, cost function storage, and message schedule."""

import numpy as np
from collections import deque


class FactorGraph:
    """Tree-structured factor graph for particle belief propagation.

    Parameters
    ----------
    n_nodes : int
        Number of nodes in the graph.
    ndim : int
        Dimensionality of each node's parameter vector.
    edges : list of (int, int)
        Undirected edges.
    local_cost_fn : callable
        ``(node_id, params_batch) -> costs``  where params_batch is (N, ndim)
        and costs is (N,).  Must be vectorized.
    pairwise_cost_fn : callable
        ``(node_i, params_i, node_j, params_j) -> float``  scalar cost for a
        single pair of configurations.
    neighbor_proposal_fn : callable or None
        ``(node_i, node_j, params_j) -> params_i``  propose a configuration
        for node_i given neighbor node_j's configuration.
    param_scale : ndarray, shape (ndim,), optional
        Per-dimension scale for normalizing particle distances in
        ``select_diverse``.  If None, all dimensions weighted equally.
    """

    def __init__(self, n_nodes, ndim, edges,
                 local_cost_fn, pairwise_cost_fn, neighbor_proposal_fn=None,
                 pairwise_cost_matrix_fn=None, param_scale=None):
        self.n_nodes = n_nodes
        self.ndim = ndim
        self.edges = [(int(a), int(b)) for a, b in edges]
        self.local_cost_fn = local_cost_fn
        self.pairwise_cost_fn = pairwise_cost_fn
        self.neighbor_proposal_fn = neighbor_proposal_fn
        self.pairwise_cost_matrix_fn = pairwise_cost_matrix_fn
        self.param_scale = (np.asarray(param_scale, dtype=np.float64)
                            if param_scale is not None
                            else np.ones(ndim))

        # Build adjacency list
        self._adj = [[] for _ in range(n_nodes)]
        for a, b in self.edges:
            self._adj[a].append(b)
            self._adj[b].append(a)

    def neighbors(self, node_id):
        """Return list of neighbor node IDs."""
        return self._adj[node_id]

    @property
    def is_chain(self):
        """True if the graph is a simple chain (path graph)."""
        if len(self.edges) != self.n_nodes - 1:
            return False
        return all(len(nb) <= 2 for nb in self._adj)

    @property
    def is_tree(self):
        """True if the graph is a tree (connected, no cycles)."""
        if len(self.edges) != self.n_nodes - 1:
            return False
        # BFS connectivity check
        visited = set()
        q = deque([0])
        visited.add(0)
        while q:
            u = q.popleft()
            for v in self._adj[u]:
                if v not in visited:
                    visited.add(v)
                    q.append(v)
        return len(visited) == self.n_nodes

    def message_schedule(self):
        """Compute forward (leaf→root) and backward (root→leaf) edge lists.

        The root is chosen as the middle node of the longest path (diameter).
        For a chain 0-1-...-n this gives balanced passes.

        Returns
        -------
        forward : list of (src, dst)
            Edges ordered leaf-to-root.
        backward : list of (src, dst)
            Edges ordered root-to-leaf (reverse of forward).
        """
        # Find diameter endpoints via double BFS
        def _bfs_farthest(start):
            dist = {start: 0}
            q = deque([start])
            farthest = start
            while q:
                u = q.popleft()
                for v in self._adj[u]:
                    if v not in dist:
                        dist[v] = dist[u] + 1
                        q.append(v)
                        if dist[v] > dist[farthest]:
                            farthest = v
            return farthest, dist

        end1, _ = _bfs_farthest(0)
        end2, dist_from_end1 = _bfs_farthest(end1)

        # Reconstruct diameter path
        path = [end2]
        visited = {end2}
        while path[-1] != end1:
            for v in self._adj[path[-1]]:
                if v not in visited and dist_from_end1[v] < dist_from_end1[path[-1]]:
                    path.append(v)
                    visited.add(v)
                    break

        root = path[len(path) // 2]

        # BFS from root to get parent pointers → leaf-to-root ordering
        parent = {root: None}
        order = []
        q = deque([root])
        while q:
            u = q.popleft()
            for v in self._adj[u]:
                if v not in parent:
                    parent[v] = u
                    order.append(v)
                    q.append(v)

        # Forward: leaf-to-root (reverse BFS order)
        forward = [(v, parent[v]) for v in reversed(order)]
        # Backward: root-to-leaf (BFS order)
        backward = [(parent[v], v) for v in order]

        return forward, backward

