import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import seaborn as sns
from matplotlib import rcParams
from matplotlib.ticker import MultipleLocator, MaxNLocator, FuncFormatter
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mc

# Font
rcParams['font.family'] = 'Helvetica'

def lighten_color(color, amount=0):
    
    c = np.array(mc.to_rgb(color))
    return tuple(c + (1 - c) * amount)

# Input file
current_folder = os.path.dirname(os.path.abspath(__file__))
excel_path = os.path.join(
    current_folder,
    "Summery of burstness ICS and total interaction time (both).xlsx"
)

# Load Excel sheets
burstness_df = pd.read_excel(excel_path, sheet_name="Burstness")
interaction_df = pd.read_excel(excel_path, sheet_name="Total Interaction time")

burstness_numeric = burstness_df.apply(pd.to_numeric, errors='coerce')
interaction_numeric = interaction_df.apply(pd.to_numeric, errors='coerce')

burstness_long = burstness_numeric.stack().reset_index()
interaction_long = interaction_numeric.stack().reset_index()

merged_df = pd.DataFrame({
    'Cluster': burstness_long['level_1'],
    'Burstness': burstness_long[0],
    'Interaction Time': interaction_long[0]
}).dropna()

# Plot ranges
x_min, x_max = 0, 12
y_min, y_max = 0, 600

# Histogram settings
x_bin_width = 1.0
y_bin_width = 50.0

# KDE settings
ADD_KDE_FLOOR = True
KDE_GRID_N = 100
KDE_LEVELS = [0.05, 0.25, 0.50, 0.75]
KDE_LW = 0.30
KDE_ALPHA = 1.0

KDE_FILL_LEVELS = np.array([0.05, 0.25, 0.50, 0.75, 1.00])
KDE_FILL_ALPHA_MIN = 0.15
KDE_FILL_ALPHA_MAX = 0.60

MIN_POINTS_FOR_KDE = 3

# Generate plots
clusters = merged_df['Cluster'].unique()
palette = sns.color_palette("dark", len(clusters))

for i, cluster in enumerate(clusters):
    data = merged_df[merged_df['Cluster'] == cluster]
    if len(data) < 2:
        print(f"Skipping {cluster} (insufficient data)")
        continue

    x = data['Burstness'].values
    y = data['Interaction Time'].values

    # Histogram
    x_bins = np.arange(x_min, x_max + x_bin_width, x_bin_width)
    y_bins = np.arange(y_min, y_max + y_bin_width, y_bin_width)

    counts, x_edges, y_edges = np.histogram2d(x, y, bins=[x_bins, y_bins])

    xpos, ypos = np.meshgrid(x_edges[:-1], y_edges[:-1], indexing="ij")
    xpos, ypos = xpos.ravel(), ypos.ravel()
    zpos = np.zeros_like(xpos)

    dx = x_bin_width * np.ones_like(zpos)
    dy = y_bin_width * np.ones_like(zpos)

    # Convert counts to fractions
    total_points = len(x)   # valid numeric rows in this cluster/column
    dz = (counts / total_points).ravel()

    mask = dz > 0
    xpos, ypos, zpos = xpos[mask], ypos[mask], zpos[mask]
    dx, dy, dz = dx[mask], dy[mask], dz[mask]

    safe_cluster_name = str(cluster).replace(" ", "_").replace("/", "_")

    
    # KDE floor plot
    fig_kde = plt.figure(figsize=(1.8, 1.8))
    ax_kde = fig_kde.add_subplot(111, projection="3d")

    if ADD_KDE_FLOOR and len(x) >= MIN_POINTS_FOR_KDE:
        kde = gaussian_kde(np.vstack([x, y]))

        xg = np.linspace(x_min, x_max, KDE_GRID_N)
        yg = np.linspace(y_min, y_max, KDE_GRID_N)
        Xg, Yg = np.meshgrid(xg, yg)

        Zk = kde(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(Xg.shape)
        Zk /= Zk.max()

        n_bands = len(KDE_FILL_LEVELS) - 1
        band_alphas = np.linspace(KDE_FILL_ALPHA_MIN, KDE_FILL_ALPHA_MAX, n_bands)

        for (lo, hi), a in zip(zip(KDE_FILL_LEVELS[:-1], KDE_FILL_LEVELS[1:]), band_alphas):
            ax_kde.contourf(
                Xg, Yg, Zk,
                zdir="z", offset=0,
                levels=[lo, hi],
                colors=[palette[i]],
                alpha=float(a)
            )

        ax_kde.contour(
            Xg, Yg, Zk,
            zdir="z", offset=0,
            levels=KDE_LEVELS,
            colors=[palette[i]],
            linewidths=KDE_LW,
            alpha=KDE_ALPHA
        )

    ax_kde.set_xlim(x_min, x_max)
    ax_kde.set_ylim(y_min, y_max)
    ax_kde.set_zlim(0, max(0.05, dz.max()) * 1.1)

    ax_kde.xaxis.set_major_locator(MultipleLocator(2))
    ax_kde.yaxis.set_major_locator(MultipleLocator(100))
    ax_kde.zaxis.set_major_locator(MaxNLocator(nbins=4))
    ax_kde.zaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:.2f}"))

    ax_kde.tick_params(labelsize=5)
    ax_kde.grid(True)

    # White tick labels
    ax_kde.tick_params(axis='x', labelcolor='white')
    ax_kde.tick_params(axis='y', labelcolor='white')
    ax_kde.tick_params(axis='z', labelcolor='white')

    for axis in [ax_kde.xaxis, ax_kde.yaxis, ax_kde.zaxis]:
        axis._axinfo["grid"]["linewidth"] = 0.3
        axis._axinfo["grid"]["color"] = (0.75, 0.75, 0.75, 0.7)

    ax_kde.set_box_aspect((1, 1, 0.6))
    ax_kde.view_init(elev=28, azim=-55)

    out_kde = os.path.join(current_folder, f"{safe_cluster_name}_KDE_floor.png")
    plt.savefig(out_kde, dpi=500, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig_kde)

    print(f"Saved KDE floor: {out_kde}")

    
    # KDE floor plot (black)
    fig_kde_black = plt.figure(figsize=(1.8, 1.8))
    ax_kde_black = fig_kde_black.add_subplot(111, projection="3d")

    if ADD_KDE_FLOOR and len(x) >= MIN_POINTS_FOR_KDE:
        kde = gaussian_kde(np.vstack([x, y]))

        xg = np.linspace(x_min, x_max, KDE_GRID_N)
        yg = np.linspace(y_min, y_max, KDE_GRID_N)
        Xg, Yg = np.meshgrid(xg, yg)

        Zk = kde(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(Xg.shape)
        Zk /= Zk.max()

        n_bands = len(KDE_FILL_LEVELS) - 1
        band_alphas = np.linspace(KDE_FILL_ALPHA_MIN, KDE_FILL_ALPHA_MAX, n_bands)

        for (lo, hi), a in zip(zip(KDE_FILL_LEVELS[:-1], KDE_FILL_LEVELS[1:]), band_alphas):
            ax_kde_black.contourf(
                Xg, Yg, Zk,
                zdir="z", offset=0,
                levels=[lo, hi],
                colors=[palette[i]],
                alpha=float(a)
            )

        ax_kde_black.contour(
            Xg, Yg, Zk,
            zdir="z", offset=0,
            levels=KDE_LEVELS,
            colors=[palette[i]],
            linewidths=KDE_LW,
            alpha=KDE_ALPHA
        )

    ax_kde_black.set_xlim(x_min, x_max)
    ax_kde_black.set_ylim(y_min, y_max)
    ax_kde_black.set_zlim(0, max(0.05, dz.max()) * 1.1)

    ax_kde_black.xaxis.set_major_locator(MultipleLocator(2))
    ax_kde_black.yaxis.set_major_locator(MultipleLocator(100))
    ax_kde_black.zaxis.set_major_locator(MaxNLocator(nbins=4))
    ax_kde_black.zaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:.2f}"))

    ax_kde_black.tick_params(labelsize=5)

    # Black style
    fig_kde_black.patch.set_facecolor('black')
    ax_kde_black.set_facecolor('black')

    ax_kde_black.tick_params(axis='x', colors='white')
    ax_kde_black.tick_params(axis='y', colors='white')
    ax_kde_black.tick_params(axis='z', colors='white')

    for axis in [ax_kde_black.xaxis, ax_kde_black.yaxis, ax_kde_black.zaxis]:
        axis._axinfo["grid"]["linewidth"] = 0.3
        axis._axinfo["grid"]["color"] = (1, 1, 1, 0.3)

    ax_kde_black.set_box_aspect((1, 1, 0.6))
    ax_kde_black.view_init(elev=28, azim=-55)

    out_kde_black = os.path.join(current_folder, f"{safe_cluster_name}_KDE_floor_black.png")
    plt.savefig(out_kde_black, dpi=500, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig_kde_black)

    print(f"Saved KDE floor black: {out_kde_black}")

    
    # Transparent bar plot
    fig_bar = plt.figure(figsize=(1.8, 1.8))
    fig_bar.patch.set_alpha(0)

    ax_bar = fig_bar.add_subplot(111, projection="3d")
    ax_bar.patch.set_alpha(0)

    light_bar_color = lighten_color(palette[i], amount=0.45)

    bars = ax_bar.bar3d(
        xpos, ypos, zpos,
        dx, dy, dz,
        color=light_bar_color,
        edgecolor="black",
        linewidth=0.3,
        alpha=0.9,  
        shade=True
    )
    bars.set_zsort("max")

    ax_bar.set_xlim(x_min, x_max)
    ax_bar.set_ylim(y_min, y_max)
    ax_bar.set_zlim(0, dz.max() * 1.1)

    # Remove axis visuals
    ax_bar.set_xticks([])
    ax_bar.set_yticks([])
    ax_bar.set_zticks([])

    ax_bar.grid(False)

    for axis in [ax_bar.xaxis, ax_bar.yaxis, ax_bar.zaxis]:
        axis.pane.set_visible(False)
        axis.line.set_alpha(0)
        axis._axinfo["grid"]["linewidth"] = 0
        axis._axinfo["axisline"]["linewidth"] = 0

    ax_bar.set_box_aspect((1, 1, 0.6))
    ax_bar.view_init(elev=28, azim=-55)

    out_bar = os.path.join(current_folder, f"{safe_cluster_name}_3D_columns_transparent.png")
    plt.savefig(
        out_bar,
        dpi=500,
        bbox_inches="tight",
        pad_inches=0.02,
        transparent=True
    )
    plt.close(fig_bar)

    print(f"Saved transparent bars: {out_bar}")