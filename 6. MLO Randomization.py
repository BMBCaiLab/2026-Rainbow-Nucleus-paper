import numpy as np
import tifffile
import cc3d
from scipy.ndimage import binary_dilation
import matplotlib.pyplot as plt
import pandas as pd
import random
import os
import multiprocessing as mp
import glob

# Settings
structure = np.zeros((3, 3, 3), dtype=np.uint8)
structure[1, :, :] = 1  # Dilate only in XY

def find_file(prefix):
    hits = glob.glob(f"{prefix}_*.tif")
    if len(hits) != 1:
        raise RuntimeError(f"Expected exactly one file for {prefix}_*.tif, found: {hits}")
    return hits[0]

file_paths = {
    'HLB':  find_file("C1"),
    'NS':   find_file("C2"),
    'CB':   find_file("C3"),
    'NoLS': find_file("C4"),
    'PML':  find_file("C5"),
}

nuclear_mask_path = "mask_clean.tif"
n_random = 100  # number of randomizations
N_WORKERS = 6 # CPU numbers used for randomizations

# Analysis functions
def load_binary_tiff(path):
    img = tifffile.imread(path)
    return (img >= 255).astype(np.uint8)

def label_objects(mask):
    return cc3d.connected_components(mask, connectivity=26)

def count_contacts(labelA, maskB):
    contact_count = 0
    for lbl in range(1, labelA.max() + 1):
        obj = (labelA == lbl)
        dilated = binary_dilation(obj, structure=structure, iterations=2)
        if np.any(dilated & maskB):
            contact_count += 1
    return contact_count

def randomize_mask_within_volume(volume_mask, object_mask, return_stats=False):
    random_mask = np.zeros_like(volume_mask, dtype=np.uint8)
    label_map = cc3d.connected_components(object_mask, connectivity=26)

    total = label_map.max()
    success = 0
    failed = 0

    for lbl in range(1, total + 1):
        obj_coords = np.argwhere(label_map == lbl)
        size = np.ptp(obj_coords, axis=0) + 1
        max_offsets = np.array(volume_mask.shape) - size - 1

        # If object bounding box bigger than volume array
        if np.any(max_offsets < 0):
            failed += 1
            continue

        for attempt in range(100):
            offset = [random.randint(0, int(max_offsets[d])) for d in range(3)]
            shifted = obj_coords - obj_coords.min(axis=0) + offset
            if np.all(volume_mask[tuple(shifted.T)]):
                random_mask[tuple(shifted.T)] = 1
                success += 1
                break
        else:
            failed += 1
           
            # print(f"Failed to place object {lbl} after 100 attempts.")

    if return_stats:
        return random_mask, success, failed
    else:
        return random_mask

def run_one_randomization(args):
    
    r, orgs, masks, nuclear_mask = args
    n_orgs = len(orgs)

    counts_r = np.zeros((n_orgs, n_orgs), dtype=int)
    placed_r = np.zeros((n_orgs,), dtype=int)

    # Randomize each source once
    rand_labels = {}
    for i, src in enumerate(orgs):
        rand_mask, success, failed = randomize_mask_within_volume(
            nuclear_mask, masks[src], return_stats=True
        )
        placed_r[i] = success
        rand_labels[src] = label_objects(rand_mask)

    # Count contacts for all pairs
    for i, src in enumerate(orgs):
        for j, tgt in enumerate(orgs):
            if i == j:
                continue
            counts_r[i, j] = count_contacts(rand_labels[src], masks[tgt])

    return r, counts_r, placed_r

# Main analysis

if __name__ == "__main__":
    # Load data
    masks = {name: load_binary_tiff(path) for name, path in file_paths.items()}
    nuclear_mask = load_binary_tiff(nuclear_mask_path)
    labels = {name: label_objects(mask) for name, mask in masks.items()}
    orgs = list(masks.keys())

    # Count number of objects per organelle
    organelle_counts = {name: labels[name].max() for name in orgs}
    df_counts = pd.DataFrame.from_dict(organelle_counts, orient='index', columns=['Object Count'])
    df_counts.to_csv("organelle_object_counts.csv")

    # Real interactions
    real_matrix = np.zeros((len(orgs), len(orgs)), dtype=int)
    for i, src in enumerate(orgs):
        for j, tgt in enumerate(orgs):
            if i != j:
                real_matrix[i, j] = count_contacts(labels[src], masks[tgt])

    # Randomized interactions
    random_counts = np.zeros((len(orgs), len(orgs), n_random), dtype=int)
    placed_counts = np.zeros((len(orgs), n_random), dtype=int)

    ctx = mp.get_context("spawn")  # best for macOS
    work_items = [(r, orgs, masks, nuclear_mask) for r in range(n_random)]

    with ctx.Pool(processes=N_WORKERS) as pool:
        for k, (r, counts_r, placed_r) in enumerate(pool.imap_unordered(run_one_randomization, work_items), start=1):
            random_counts[:, :, r] = counts_r
            placed_counts[:, r] = placed_r
            print(f"Finished randomization {k}/{n_random}")

    # Compute averages and probabilities
    random_mean_matrix = random_counts.mean(axis=2)

    # Total successfully placed source objects
    total_placed_per_src = placed_counts.sum(axis=1).astype(float)  # (len(orgs),)

    # Random contact probability
    random_prob_matrix = np.zeros((len(orgs), len(orgs)), dtype=float)
    for i in range(len(orgs)):
        denom = total_placed_per_src[i]
        if denom <= 0:
            random_prob_matrix[i, :] = np.nan
        else:
            random_prob_matrix[i, :] = random_counts[i, :, :].sum(axis=1) / denom

    # Output
    df_real = pd.DataFrame(real_matrix, index=orgs, columns=orgs)
    df_random_mean = pd.DataFrame(random_mean_matrix, index=orgs, columns=orgs)
    df_random_prob = pd.DataFrame(random_prob_matrix, index=orgs, columns=orgs)

    df_real.to_csv("real_interactions.csv")
    df_random_mean.to_csv("random_mean_interactions.csv")
    df_random_prob.to_csv("random_probability_interactions.csv")

    # Heatmap of real interactions
    plt.figure(figsize=(8, 6))
    plt.imshow(real_matrix, cmap='viridis')
    plt.title("Real Organelle Interactions")
    plt.colorbar(label="Contact Count")
    plt.xticks(range(len(orgs)), orgs)
    plt.yticks(range(len(orgs)), orgs)
    for i in range(len(orgs)):
        for j in range(len(orgs)):
            plt.text(
                j, i, str(real_matrix[i, j]),
                ha='center', va='center',
                color='white' if real_matrix[i, j] > real_matrix.max() * 0.5 else 'black'
            )
    plt.tight_layout()
    plt.savefig("real_interactions_heatmap.png")