"""Adaptive longest-edge bisection for a closed triangle mesh, carrying a rest map.

Used by the erosion unwind (`vertebrae._unwind_erosion`), which has to grow structures erosion deleted outright (a
transverse process cut off with the bar around its foramen, a spinous-process tip): with fixed connectivity a few
triangles end up stretched tens of times, so the mesh is refined where it stretches.

`split_long_edges` splits every edge longer than `max_len` (current geometry), pass by pass, until none is left. A
pass splits an edge only when it is the longest over-long edge of each face it borders, so a face is split at most
once per pass (2 faces -> 4); the globally longest over-long edge always qualifies, so every pass makes progress.
Splits only append vertices (the originals keep their indices) and never change topology (V+1, E+3, F+2 per split:
the Euler characteristic, hence the genus, is unchanged).

The new vertex sits at the midpoint of the edge's CURRENT positions, so the surface does not move. The rest map
`R` (n x n0 sparse) writes every vertex as a combination of the n0 original vertices: a new vertex's row is the
mean of its endpoints' rows, so `R @ V0` places each vertex on the starting mesh. Downstream, `R` carries
correspondences defined on the starting mesh's vertices over to the refined mesh.
"""

import numpy as np
from scipy import sparse


def mesh_edges(faces):
    """Unique undirected edges and, per half-edge (face f, slot i: F[f,i] -> F[f,(i+1)%3]), its edge id (nF, 3)."""
    half_edges = np.stack([faces, np.roll(faces, -1, axis=1)], axis=2).reshape(-1, 2)    # row 3f+i
    edges, edge_id = np.unique(np.sort(half_edges, axis=1), axis=0, return_inverse=True)
    return edges, edge_id.reshape(-1, 3)


def euler_characteristic(faces):
    """V - E + F over the vertices the faces use (0 for a closed genus-1 surface)."""
    edges, _ = mesh_edges(faces)
    return len(np.unique(faces)) - len(edges) + len(faces)


def split_long_edges(verts, faces, rest_map, max_len, max_passes=100):
    """Bisect edges longer than max_len until none remain.

    :param verts: (n, 3) current vertex positions
    :param faces: (nF, 3) triangles of a closed, consistently oriented mesh
    :param rest_map: (n, n0) sparse map from the original vertices (identity before any split)
    :param max_len: split threshold, in the units of verts
    :return: (verts, faces, rest_map, n_split) -- appended vertices, rebuilt faces, extended rest map
    """
    V = np.asarray(verts, np.float64)
    F = np.asarray(faces, np.int64)
    R = sparse.csr_matrix(rest_map)
    n_split = 0
    for _ in range(max_passes):
        E, fe = mesh_edges(F)
        length = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)
        flagged = length > max_len
        if not flagged.any():
            return V, F, R, n_split
        # each face picks its longest flagged edge (ties broken by edge id through a vanishing bias)
        score = np.where(flagged, length, -np.inf)[fe] + 1e-12 * fe / len(E)
        slot = np.argmax(score, axis=1)
        rows = np.arange(len(F))
        has = np.isfinite(score[rows, slot])
        choice = np.where(has, fe[rows, slot], -1)
        # an edge is split when every face bordering it chose it
        borders = np.bincount(fe.ravel(), minlength=len(E))
        chosen = np.bincount(choice[has], minlength=len(E))
        split = flagged & (chosen == borders)
        new_id = np.full(len(E), -1, np.int64)
        new_id[split] = len(V) + np.arange(split.sum())
        a, b = E[split, 0], E[split, 1]
        V = np.vstack([V, 0.5 * (V[a] + V[b])])
        R = sparse.vstack([R, 0.5 * (R[a] + R[b])]).tocsr()
        # rebuild the faces bordering a split edge: (v_i, v_i+1, o) -> (v_i, m, o) + (m, v_i+1, o)
        hit = has & split[np.maximum(choice, 0)]
        fh, sh = np.where(hit)[0], slot[hit]
        vi, vj, vo = F[fh, sh], F[fh, (sh + 1) % 3], F[fh, (sh + 2) % 3]
        m = new_id[choice[hit]]
        F = np.vstack([F[~hit], np.column_stack([vi, m, vo]), np.column_stack([m, vj, vo])])
        n_split += int(split.sum())
    raise RuntimeError('split_long_edges: edges still longer than %g after %d passes' % (max_len, max_passes))
