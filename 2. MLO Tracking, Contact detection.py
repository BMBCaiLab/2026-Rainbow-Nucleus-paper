import os
import numpy as np
import tifffile
import pandas as pd
from skimage.measure import regionprops
from scipy.ndimage import binary_dilation
import cc3d


CELL_ID = 5728  # change cell ID

# IOU threshold for tracking
iou_threshold = 0.05

# Contact setting: XY dilate 2 pixels (NO Z dilation)
structure = np.zeros((3, 3, 3), dtype=np.uint8)
structure[1, :, :] = 1
DILATE_ITERS_XY = 2  

def make_fname(ch_num: int, cell_id: int) -> str:
    return f"C{ch_num}_cell {cell_id}_thresholded_output.tif"

file_C1 = make_fname(1, CELL_ID)  # HLB
file_C2 = make_fname(2, CELL_ID)  # NS (NOT tracked; treated as union C2_all)
file_C3 = make_fname(3, CELL_ID)  # CB
file_C4 = make_fname(4, CELL_ID)  # Nucleolus
file_C5 = make_fname(5, CELL_ID)  # PML-NB

# Output Directory
input_dir = os.path.dirname(os.path.abspath(file_C1))
output_dir = os.path.join(input_dir, f"Cell {CELL_ID} interactions")
os.makedirs(output_dir, exist_ok=True)


# Load Stacks
stack_C1 = tifffile.imread(file_C1)
stack_C2 = tifffile.imread(file_C2)
stack_C3 = tifffile.imread(file_C3)
stack_C4 = tifffile.imread(file_C4)
stack_C5 = tifffile.imread(file_C5)

n_timepoints = stack_C1.shape[0]

stacks = {
    'C1': stack_C1,
    'C2': stack_C2,
    'C3': stack_C3,
    'C4': stack_C4,
    'C5': stack_C5,
}

all_channels = ['C1', 'C2', 'C3', 'C4', 'C5']
tracked_channels = ['C1', 'C3', 'C4', 'C5']  # C2 is NOT tracked

# Label Objects per Frame
tracked_objects = {ch: {} for ch in all_channels}

for ch, stack in stacks.items():
    for t in range(n_timepoints):
        binary = stack[t] > 0
        labeled = cc3d.connected_components(binary)
        props = regionprops(labeled)

        objs = []
        for prop in props:
            objs.append({
                'label': prop.label,
                'mask': (labeled == prop.label)
            })
        tracked_objects[ch][t] = objs


# IOU Tracking for C1/C3/C4/C5
def compute_iou(mask1, mask2):
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return inter / union if union > 0 else 0.0

object_tracks = {ch: {} for ch in tracked_channels}
next_track_id = 1

for ch in tracked_channels:
    for t in range(n_timepoints):
        objs = tracked_objects[ch][t]
        if t == 0:
            for i in range(len(objs)):
                object_tracks[ch][(t, i)] = next_track_id
                next_track_id += 1
        else:
            prev_objs = tracked_objects[ch][t - 1]
            for i, obj in enumerate(objs):
                best_iou = 0.0
                best_prev_idx = -1
                for j, prev_obj in enumerate(prev_objs):
                    iou = compute_iou(obj['mask'], prev_obj['mask'])
                    if iou > best_iou:
                        best_iou = iou
                        best_prev_idx = j

                if best_iou > iou_threshold and best_prev_idx != -1:
                    track_id = object_tracks[ch][(t - 1, best_prev_idx)]
                else:
                    track_id = next_track_id
                    next_track_id += 1

                object_tracks[ch][(t, i)] = track_id


# Export Tracked Label Stacks(C1, C3, C4, C5)

for ch in tracked_channels:  # ['C1','C3','C4','C5']

    stack_shape = stacks[ch].shape  # (T,Z,Y,X)
    label_stack = np.zeros(stack_shape, dtype=np.uint16)

    for t in range(n_timepoints):
        objs = tracked_objects[ch][t]

        for i, obj in enumerate(objs):
            tid = object_tracks[ch][(t, i)]
            label_stack[t][obj['mask']] = tid

    outname = os.path.join(
        output_dir,
        f"{ch}_tracked_labels_Cell {CELL_ID}.tif"
    )

    tifffile.imwrite(
        outname,
        label_stack,
        imagej=True,
        metadata={'axes': 'TZYX'}
    )


# Build Contacts Per Frame
def get_items_for_channel_at_t(ch, t):
    if ch == 'C2':
        objs = tracked_objects['C2'][t]
        if len(objs) == 0:
            return []
        union = np.zeros_like(objs[0]['mask'], dtype=bool)
        for o in objs:
            union |= o['mask']
        return [('C2_all', union)]
    else:
        objs = tracked_objects[ch][t]
        items = []
        for i, obj in enumerate(objs):
            tid = object_tracks[ch][(t, i)]
            items.append((f"{ch}_{tid}", obj['mask']))
        return items

per_frame_records = []


for t in range(n_timepoints):
    items_by_channel = {ch: get_items_for_channel_at_t(ch, t) for ch in all_channels}

    for i_ch in range(len(all_channels)):
        chA = all_channels[i_ch]
        itemsA = items_by_channel[chA]
        if len(itemsA) == 0:
            continue

        for j_ch in range(i_ch + 1, len(all_channels)):
            chB = all_channels[j_ch]
            itemsB = items_by_channel[chB]
            if len(itemsB) == 0:
                continue

            for ida, mask_a in itemsA:
                dil_a = binary_dilation(mask_a, structure=structure, iterations=DILATE_ITERS_XY)

                for idb, mask_b in itemsB:
                    if np.any(dil_a & mask_b):

                        # ---- DIRECTION 1: A -> B ----
                        pair_key_ab = f"{ida}__{idb}"
                        per_frame_records.append((t, ida, idb, pair_key_ab))

                        # ---- DIRECTION 2: B -> A ----
                        pair_key_ba = f"{idb}__{ida}"
                        per_frame_records.append((t, idb, ida, pair_key_ba))


# Export Per Frame Contact
df_frame = pd.DataFrame(per_frame_records, columns=['time', 'obj1', 'obj2', 'pair_key'])
out_frame_csv = os.path.join(output_dir, f"interactions_per_frame_Cell {CELL_ID}.csv")
df_frame.to_csv(out_frame_csv, index=False)
print(f"Saved per-frame contacts: {out_frame_csv}")


# Convert Per Frame Contact to BOUTS

def boolean_runs(times_sorted):
    if len(times_sorted) == 0:
        return []
    runs = []
    start = prev = times_sorted[0]
    for tt in times_sorted[1:]:
        if tt == prev + 1:
            prev = tt
        else:
            runs.append((start, prev))
            start = prev = tt
    runs.append((start, prev))
    return runs

bout_records = []


for pair_key, sub in df_frame.groupby('pair_key'):
    times = sorted(sub['time'].unique().tolist())
    runs = boolean_runs(times)
    obj1, obj2 = pair_key.split("__", 1)

    for k, (s, e) in enumerate(runs, start=1):
        bout_records.append({
            'pair_key': pair_key,
            'obj1': obj1,
            'obj2': obj2,
            'bout_id': k,
            'start_time': s,
            'end_time': e,
            'duration_frames': e - s + 1,
        })

df_bouts = pd.DataFrame(bout_records)
out_bouts_csv = os.path.join(output_dir, f"interaction_bouts_Cell {CELL_ID}.csv")
df_bouts.to_csv(out_bouts_csv, index=False)