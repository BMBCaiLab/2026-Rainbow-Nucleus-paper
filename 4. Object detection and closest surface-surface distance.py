import os
import re
import glob
import numpy as np
import pandas as pd
import tifffile
import cc3d
import multiprocessing as mp

from skimage.measure import regionprops
from scipy.ndimage import distance_transform_edt, binary_erosion


# Settings
PX_PER_UM_XY = 28.20
UM_PER_SLICE_Z = 0.13

UM_PER_PX_XY = 1.0 / PX_PER_UM_XY
SPACING_UM = (UM_PER_SLICE_Z, UM_PER_PX_XY, UM_PER_PX_XY)

CONNECTIVITY = 26 # 26-way connnectivity

OUT_OBJECTS = "0. objects and center.xlsx"
OUT_DISTS = "0. min_surf to surf distance.xlsx"

COUNT_CHANNELS = ["C1", "C3", "C4", "C5"]

# Number of CPU cores used for multiprocessing
N_CORES = 6

# Distance pairs to calculate as source channel to target channel
DIST_PAIRS = [
    ("C4", "C2", "C4_to_C2_Nucleolus_to_NS"),
    ("C5", "C2", "C5_to_C2_PML_to_NS"),
    ("C3", "C2", "C3_to_C2_CB_to_NS"),
    ("C1", "C2", "C1_to_C2_HLB_to_NS"),
    ("C5", "C4", "C5_to_C4_PML_to_Nucleolus"),
    ("C3", "C4", "C3_to_C4_CB_to_Nucleolus"),
    ("C1", "C4", "C1_to_C4_HLB_to_Nucleolus"),
    ("C3", "C5", "C3_to_C5_CB_to_PML"),
    ("C1", "C5", "C1_to_C5_HLB_to_PML"),
    ("C1", "C3", "C1_to_C3_HLB_to_CB"),
    ("C4", "C5", "C4_to_C5_Nucleolus_to_PML"),
    ("C4", "C3", "C4_to_C3_Nucleolus_to_CB"),
    ("C4", "C1", "C4_to_C1_Nucleolus_to_HLB"),
    ("C5", "C3", "C5_to_C3_PML_to_CB"),
    ("C5", "C1", "C5_to_C1_PML_to_HLB"),
    ("C3", "C1", "C3_to_C1_CB_to_HLB"),
]


# Find thresholded mask files
def make_fname(folder: str, Cn: int, cell: int):
    patterns = [
        f"C{Cn}_*New-01({cell})*thresholded_output.tif",
        f"C{Cn}_*New-02({cell})*thresholded_output.tif",
    ]

    candidates = []
    for pattern in patterns:
        candidates.extend(glob.glob(os.path.join(folder, pattern)))

    candidates = sorted(list(dict.fromkeys(candidates)))

    if not candidates:
        print(f"Not found: C{Cn}, cell {cell:02d}")
        return None

    selected = candidates[0]
    print(f"Found: {os.path.basename(selected)}")

    return selected


# Image processing functions
def load_stack_TZYX(path):
    arr = tifffile.imread(path)

    if arr.ndim == 3:
        return arr[None, ...]

    if arr.ndim == 4:
        return arr

    raise ValueError(f"Wrong image shape: {arr.shape}")


def as_bool(mask):
    return mask > 0


def label_3d(mask):
    return cc3d.connected_components(mask.astype(np.uint8), connectivity=CONNECTIVITY)


def px_to_um_centroid(z, y, x):
    return (
        z * UM_PER_SLICE_Z,
        y * UM_PER_PX_XY,
        x * UM_PER_PX_XY,
    )


def boundary_voxels(mask):
    eroded = binary_erosion(
        mask,
        structure=np.ones((3, 3, 3), dtype=bool),
        iterations=1,
        border_value=0,
    )
    return mask & (~eroded)


def min_surface_distance_um(obj_mask, tgt_mask):
    if not np.any(tgt_mask):
        return float("nan")

    if np.any(obj_mask & tgt_mask):
        return 0.0

    distance_map = distance_transform_edt(~tgt_mask, sampling=SPACING_UM)
    obj_boundary = boundary_voxels(obj_mask)

    if not np.any(obj_boundary):
        coords = np.argwhere(obj_mask)
        return float(distance_map[tuple(coords[0])])

    return float(np.min(distance_map[obj_boundary]))


def excel_safe_sheet_name(name):
    return re.sub(r'[:\\/?*\[\]]', "_", name)[:31]


# Process one folder
def run_one_folder(folder):
    print(f"Folder: {folder}")

    hits = glob.glob(os.path.join(folder, "C1_New-*-Airyscan Processing*thresholded_output.tif"))

    if not hits:
        print("No thresholded masks found")
        return

    cell_ids = set()

    for hit in hits:
        match = re.search(r"C1_.*\((\d+)\).*thresholded_output", os.path.basename(hit))
        if match:
            cell_ids.add(int(match.group(1)))

    if not cell_ids:
        print("No cell IDs detected")
        return

    print(f"Detected cells: {sorted(cell_ids)}")

    centroid_rows = []
    pair_rows = {sheet: [] for (_, _, sheet) in DIST_PAIRS}

    for cell in sorted(cell_ids):
        print(f"Processing cell {cell:02d}")

        stacks = {}

        for cn in range(1, 6):
            path = make_fname(folder, cn, cell)

            if path is None:
                raise FileNotFoundError(f"Missing C{cn} mask for cell {cell:02d}")

            stacks[f"C{cn}"] = load_stack_TZYX(path)

        T = stacks["C1"].shape[0]

        for t in range(T):
            masks = {ch: as_bool(stacks[ch][t]) for ch in stacks}
            labels = {ch: label_3d(masks[ch]) for ch in masks}

            for ch in COUNT_CHANNELS:
                for prop in regionprops(labels[ch]):
                    z, y, x = prop.centroid
                    z_um, y_um, x_um = px_to_um_centroid(z, y, x)

                    centroid_rows.append({
                        "cell": cell,
                        "t": t,
                        "channel": ch,
                        "label_id": int(prop.label),
                        "centroid_z_px": float(z),
                        "centroid_y_px": float(y),
                        "centroid_x_px": float(x),
                        "centroid_z_um": float(z_um),
                        "centroid_y_um": float(y_um),
                        "centroid_x_um": float(x_um),
                    })

            for src, tgt, sheet in DIST_PAIRS:
                for prop in regionprops(labels[src]):
                    src_object = labels[src] == prop.label
                    d_um = min_surface_distance_um(src_object, masks[tgt])

                    z, y, x = prop.centroid
                    z_um, y_um, x_um = px_to_um_centroid(z, y, x)

                    pair_rows[sheet].append({
                        "cell": cell,
                        "src_label_id": int(prop.label),
                        "centroid_z_px": float(z),
                        "centroid_y_px": float(y),
                        "centroid_x_px": float(x),
                        "centroid_z_um": float(z_um),
                        "centroid_y_um": float(y_um),
                        "centroid_x_um": float(x_um),
                        "min_surface_distance_um": float(d_um),
                    })

    out_obj = os.path.join(folder, OUT_OBJECTS)
    out_dist = os.path.join(folder, OUT_DISTS)

    df_cent = pd.DataFrame(centroid_rows)

    if df_cent.empty:
        df_cnt = pd.DataFrame(columns=["cell", "t", "channel", "n_objects"])
    else:
        df_cnt = (
            df_cent.groupby(["cell", "t", "channel"])["label_id"]
            .nunique()
            .reset_index(name="n_objects")
        )

    with pd.ExcelWriter(out_obj, engine="openpyxl") as writer:
        df_cnt.to_excel(writer, index=False, sheet_name="counts_C1C3C4C5")
        df_cent.to_excel(writer, index=False, sheet_name="centroids_C1C3C4C5")

    with pd.ExcelWriter(out_dist, engine="openpyxl") as writer:
        for sheet, rows in pair_rows.items():
            df_pair = pd.DataFrame(rows)
            df_pair.to_excel(writer, index=False, sheet_name=excel_safe_sheet_name(sheet))

    print("Saved Excel files")


# Main pipeline
def main():
    base = os.getcwd()
    print(f"Main directory: {base}")

    subfolders = [
        os.path.join(base, name)
        for name in os.listdir(base)
        if os.path.isdir(os.path.join(base, name))
    ]

    subfolders = sorted(subfolders)

    print(f"CPU cores: {N_CORES}")

    with mp.Pool(processes=N_CORES) as pool:
        pool.map(run_one_folder, subfolders)

    print("All folders processed")


if __name__ == "__main__":
    main()
