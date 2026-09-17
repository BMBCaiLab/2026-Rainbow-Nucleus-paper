import os
import numpy as np
import tifffile
from skimage.filters import threshold_otsu
from scipy import ndimage as ndi
from skimage.draw import line

# Settings
INPUT_FILE = "ROI 19 58916.tif" #Change accordingly

input_dir = os.path.dirname(os.path.abspath(INPUT_FILE))
OUTPUT_FILE = os.path.join(input_dir, "roi_19_trajectory_channels_TCYX.tif")

# Load image
img = tifffile.imread(INPUT_FILE)

if img.ndim != 4:
    raise ValueError(f"Wrong image dimention")

T, C, Y, X = img.shape

# Extract original channels
input_c1 = img[:, 0].astype(np.uint16)
input_c2 = img[:, 1].astype(np.uint16)

# Analysis functions
def centers_from_otsu(stack):
    centers = []

    for t in range(stack.shape[0]):
        frame = stack[t]

        if frame.max() == frame.min():
            centers.append(None)
            continue

        threshold = threshold_otsu(frame)
        mask = frame > threshold

        if mask.sum() == 0:
            centers.append(None)
        else:
            cy, cx = ndi.center_of_mass(mask.astype(np.uint8))
            centers.append((float(cy), float(cx)))

    return centers


def build_trajectory_stack(centers, T, Y, X, intensity):
   
    trajectory = np.zeros((T, Y, X), dtype=np.uint16)
    points = []

    for t in range(T):
        if t > 0:
            trajectory[t] = trajectory[t - 1].copy()

        point = centers[t]

        if point is not None:
            y = int(round(point[0]))
            x = int(round(point[1]))
            y = np.clip(y, 0, Y - 1)
            x = np.clip(x, 0, X - 1)
            points.append((y, x))

        if len(points) >= 2:
            y0, x0 = points[-2]
            y1, x1 = points[-1]
            rr, cc = line(y0, x0, y1, x1)
            rr = np.clip(rr, 0, Y - 1)
            cc = np.clip(cc, 0, X - 1)
            trajectory[t, rr, cc] = intensity

        if len(points) >= 1:
            y, x = points[-1]
            trajectory[t, y, x] = intensity

    return trajectory

# Link trajectories
c1_centers = centers_from_otsu(input_c1)
c2_centers = centers_from_otsu(input_c2)

traj_c1 = build_trajectory_stack(c1_centers, T, Y, X, np.iinfo(np.uint16).max)
traj_c2 = build_trajectory_stack(c2_centers, T, Y, X, np.iinfo(np.uint16).max)

# Save output
# Output channel order:
# C1 = input C2
# C2 = input C1
# C3 = trajectory of input C2
# C4 = trajectory of input C1
out = np.stack([input_c2, input_c1, traj_c2, traj_c1], axis=1)

tifffile.imwrite(OUTPUT_FILE, out, imagej=True)

print(f"Saved: {OUTPUT_FILE}")
print(f"Output shape: {out.shape}")
