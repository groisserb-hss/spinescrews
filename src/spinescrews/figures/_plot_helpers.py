from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from bg3dtools.mesh.generate import build_cylinder_capped, pointcloud_to_splatted_mesh
import igl


def plot_error_clouds(out_file, group_names, data_list, color_list, symbol_list, S):
    """Scatter plot of 2D error points with PCA ellipses, used by group_statistics.py."""
    if np.isscalar(S):
        S = [-S, S]

    assert len(data_list) == len(color_list) == len(symbol_list)

    # Remove non-finite elements
    data_list = [data[np.all(np.isfinite(data), axis=1), :] for data in data_list]

    fig = plt.figure(figsize=(8, 8.6))
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')

    # set grid, ticks, and background color
    plt.xlim(S)
    plt.ylim(S)
    plt.xticks(np.arange(S[0], S[1] + 1))
    plt.yticks(np.arange(S[0], S[1] + 1))
    ax.set_facecolor('none')
    plt.gcf().patch.set_facecolor('none')
    ax.set_axisbelow(True)
    ax.grid(True, zorder=0)

    # plot target
    plot_target([2, 4])

    # plot ellipses
    for data, color in zip(data_list, color_list):
        ellipse = compute_ellipse(data)
        plt.fill(ellipse[:, 0], ellipse[:, 1], color=color, alpha=0.3, zorder=2)  # 0.5 makes it semi-transparent
        plt.plot(ellipse[:, 0], ellipse[:, 1], color=color, linewidth=1.5, zorder=3)  # Outline for the ellipse

    # scatter data
    for name, data, symbol, color in zip(group_names, data_list, symbol_list, color_list):
        g = max(S)
        plt.scatter(data[:, 0], data[:, 1], s=round(200 / g), c=[color], marker=symbol, alpha=0.7, edgecolors=[color], zorder=4, label=name)
        plt.scatter(np.mean(data[:, 0]), np.mean(data[:, 1]), s=round(6000 / g), c=[color], marker='*', alpha=0.8, edgecolors='k', zorder=5)

    fig.legend(loc='lower center', ncol=len(group_names), fontsize=10, frameon=False)
    fig.tight_layout(rect=[0, 0.05, 1, 1])

    fig.savefig(f"{out_file}.png", transparent=True, dpi=600)
    plt.close(fig)


def plot_error_bundles(out_name, group_names, entry_pts, tip_pts, color_list, symbol_list):
    """
    Plot lines connecting entry and tip points

    :param out_name:
    :param entry_pts: list of [N x 2] arrays
    :param tip_pts: list of [N x 2] arrays
    :param color_list:
    :param symbol_list:
    :param S:
    :return:
    """

    assert len(entry_pts) == len(tip_pts) == len(color_list) == len(symbol_list)

    # Remove non-finite elements
    good_pts = [np.all(np.isfinite(np.column_stack([e, t])), axis=1) for e, t in zip(entry_pts, tip_pts)]
    entry_pts = [e[good] for e, good in zip(entry_pts, good_pts)]
    tip_pts = [t[good] for t, good in zip(tip_pts, good_pts)]

    fig = plt.figure(figsize=(8, 2.5))
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')

    # set grid, ticks, and background color
    ax.set_facecolor('none')
    plt.gcf().patch.set_facecolor('none')
    ax.set_axisbelow(True)
    ax.grid(True, zorder=0)

    # find start/stop coordinates for 95% CI regions
    roi_x = [np.linspace(np.percentile(e[:,0], 50), np.percentile(t[:,0], 50),
                         100) for e, t in zip(entry_pts, tip_pts)]
    roi_top, roi_bottom = [], []
    for e, t, r in zip(entry_pts, tip_pts, roi_x):
        I = project_lines(r, e, t, d=0)
        m = np.mean(I[:,:, 1], axis=1)
        s = np.std(I[:,:, 1], axis=1)
        roi_top.append(m + 2*s)
        roi_bottom.append(m - 2*s)

    # scatter data
    for name, entry, tip, symbol, color in zip(group_names, entry_pts, tip_pts, symbol_list, color_list):
        plt.scatter(entry[:, 0], entry[:, 1], s=20, c=[color], marker=symbol, alpha=0.7, edgecolors=[color], zorder=4, label=name)
        plt.scatter(tip[:, 0], tip[:, 1], s=20, c=[color], marker=symbol, alpha=0.7, edgecolors=[color], zorder=4)
    # shaded roi
    for r, f1, f2, c in zip(roi_x[-1::-1], roi_top[-1::-1], roi_bottom[-1::-1], color_list[-1::-1]):
        draw_roi_between_functions(r, f1, f2, color=c)

    fig.legend(loc='lower center', ncol=len(group_names), fontsize=10, frameon=False)
    fig.tight_layout(rect=[0, 0.12, 1, 1])

    fig.savefig(f"{out_name}.png", transparent=True, dpi=600)
    plt.close(fig)


def plot_target(rings):
    """Draw concentric target rings and crosshair at origin."""
    for r in rings:
        t = np.linspace(0, 2 * np.pi, 1000)
        x = np.cos(t) * r
        y = np.sin(t) * r
        plt.plot(x, y, color=[0.5, 0.5, 0.5], linewidth=4, zorder=1)

    p1 = min(rings) - 0.4
    p2 = max(rings) + 0.4
    plt.plot([p1, p2], [0, 0], color=[0.5, 0.5, 0.5], linewidth=4, zorder=1)
    plt.plot([-p1, -p2], [0, 0], color=[0.5, 0.5, 0.5], linewidth=4, zorder=1)
    plt.plot([0, 0], [p1, p2], color=[0.5, 0.5, 0.5], linewidth=4, zorder=1)
    plt.plot([0, 0], [-p1, -p2], color=[0.5, 0.5, 0.5], linewidth=4, zorder=1)
    plt.scatter(0, 0, s=1000, c='k', marker='+', linewidths=2, zorder=1)


def compute_ellipse(data, p=180, ref_ang=None):
    """Fit a PCA-based 2-sigma ellipse to 2D point data; optionally track orientation via ref_ang."""
    # Compute variance along major/minor axis of spread
    pca = PCA(n_components=2)
    pca.fit(data)

    # Use one of the components as orientation (here: 2nd component, as in your code)
    v = pca.components_[1]  # shape (2,)
    angle = np.arctan2(-v[0], v[1])  # your original convention

    if ref_ang is not None:
        # Candidate with flipped sign (add π)
        alt_angle = angle + np.pi
        if np.abs(alt_angle - ref_ang) < np.abs(angle - ref_ang):
            angle = alt_angle

    spread = 2 * np.sqrt(pca.explained_variance_)

    # Build ellipse
    t = np.linspace(0, 2 * np.pi, p)
    x = np.cos(t) * spread[0]
    y = np.sin(t) * spread[1]

    # Rotate ellipse
    T = np.array([[np.cos(angle), np.sin(angle)],
                  [-np.sin(angle), np.cos(angle)]])
    ellipse = np.dot(np.column_stack((x, y)), T) + np.mean(data, axis=0)

    if ref_ang is not None:
        return ellipse, angle
    return ellipse


def project_lines(r, p1, p2, d=0):
    """
    :param r: nI one-dimensional vector
    :param p1: [N x D] array
    :param p2: [N x D] array
    :param d: int dimension of "independent variable"
    :return: [nI x N x D]
    """
    N, D = p1.shape
    nI = len(r)
    assert p1.shape == p2.shape

    Y = np.full((nI, N, D), np.nan, np.float32)
    for ii, x in enumerate(r):
        dx = p2[:, d] - p1[:, d]
        p = (x - p1[:, d]) / dx
        Y[ii] = p1 + (p2 - p1) * p[:, None]
    return Y


def draw_roi_between_functions(x, y1, y2, color='C0'):
    """
    f1, f2: callables, f1(x) >= f2(x) over [x_min, x_max]
    x_min, x_max: range to draw
    n: number of sample points
    """

    # Translucent interior
    plt.fill_between(x, y1, y2, color=color, alpha=0.3, zorder=2)

    # Opaque-ish border
    plt.plot(x, y1, color=color, linewidth=1.5, zorder=3)
    plt.plot(x, y2, color=color, linewidth=1.5, zorder=3)

    # Optional: draw vertical edges at the ends if you want a fully closed ROI outline
    plt.plot([x[0], x[0]], [y2[0], y1[0]], color=color, linewidth=1., zorder=3)
    plt.plot([x[-1], x[-1]], [y2[-1], y1[-1]], color=color, linewidth=1., zorder=3)


def plot_3d_error_clouds(basename, group_names, entry_pts, tip_pts, colors):
    """
    :param out_dir:
    :param group_names: list of group names
    :param entry_pts: list of [N x 3] arrays
    :param tip_pts: list of [N x 3] arrays
    :param colors: list of colors
    :return:
    """
    # remove NaNs
    good_pts = [np.all(np.isfinite(np.column_stack([e, t])), axis=1) for e, t in zip(entry_pts, tip_pts)]
    entry_pts = [e[good] for e, good in zip(entry_pts, good_pts)]
    tip_pts = [t[good] for t, good in zip(tip_pts, good_pts)]

    d = 1  # long screw axis (A-P)
    steps, ticks = 500, 120
    _, funnel_f = igl.cylinder(ticks, steps)

    for g, e, t in zip(group_names, entry_pts, tip_pts):
        # range along screw axes
        ap_range = np.linspace(np.percentile(e[:, d], 50), np.percentile(t[:, d], 50), steps)
        I = project_lines(ap_range, e, t, d=d)  # S x N x 3

        funnel_v, ref_ang = [], 0
        for ii, (pts, r) in enumerate(zip(I, ap_range)):
            verts = np.full((ticks, 3), r, np.float32)
            verts[:, [0, 2]], ref_ang = compute_ellipse(pts[:, [0, 2]], ticks, ref_ang)
            funnel_v.append(verts)
        funnel_v = np.concatenate(funnel_v, axis=0)
        igl.write_triangle_mesh('%s_%s_funnel.ply' % (basename, g), funnel_v, funnel_f)

        e_v, e_f = pointcloud_to_splatted_mesh(e, 0.33)
        igl.write_triangle_mesh('%s_%s_entry-pts.ply' % (basename, g), e_v, e_f)

        t_v, t_f = pointcloud_to_splatted_mesh(t, 0.33)
        igl.write_triangle_mesh('%s_%s_tip-pts.ply' % (basename, g), t_v, t_f)

        # make cylinders to connect tip and tail