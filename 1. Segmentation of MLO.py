import numpy as np
import tifffile
import pandas as pd
import os
import re

from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import remove_small_objects
from skimage.measure import label, regionprops

# Channel annotation: C1-HLB; C2-NS; C3-CB; C4-nucleolus; C5-PML-NB.
# Input .tif file (T, Z, C, Y, X)
INPUT_TIF = "cell 5737.tif"   # change filename.

gaussian_sigma = 1.5  # Gaussian blur. Default is 1.5.

# Fixed-threshold settings per channel as needed. Default is False.
use_fixed_threshold = {
    1: False,
    2: False,
    3: False,
    4: False,
    5: False,
}

# Only need if use fixed threshold. Defalut is 0.
fixed_threshold = {
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0,
}

# Start timepoint to use fixed threshold. Defalut is 0.
fixed_threshold_start_tp = {
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0,
}

# Channel-specific Otsu threshold offset. Default is 0.
threshold_offset = {
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0,
}

# Size filter (min_voxels) per channel. Default: True, to get rid of pixelated noise and false thresholded objects.
use_size_filter = {
    1: True,
    2: True,
    3: True,
    4: True,
    5: True,
}

channel_min_voxels = {
    1: 20,
    2: 20,
    3: 20,
    4: 3000,
    5: 20,
}

# Intensity filter toggle per channel. Default: False. If True, keep only objects whose center intensity > intensity_min[channel]
use_intensity_filter = {
    1: False,
    2: False,
    3: False,
    4: False,
    5: False,
}

intensity_min = {
    1: 0,
    2: 0,
    3: 0,
    4: 0,
    5: 0,
}


# Main pipeline
def main():
    current_folder = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(current_folder, INPUT_TIF)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Cannot find input TIF file: {INPUT_TIF}")

    stack_5d = tifffile.imread(input_path)

    if stack_5d.ndim != 5:
        raise ValueError(f"Not 5D (T, Z, C, Y, X), got {stack_5d.shape}")

    T, Z, C, Y, X = stack_5d.shape
    print(f"Loaded TIF with shape (T, Z, C, Y, X) = {stack_5d.shape}")

    base_name = os.path.splitext(os.path.basename(INPUT_TIF))[0]

    # Create output folder named after input file
    output_folder = os.path.join(current_folder, base_name)
    os.makedirs(output_folder, exist_ok=True)

    # Save TIFs for individual channels and process each channel
    thresholded_per_channel = {}  # store for composite

    for ch_index in range(C):
        ch_num = ch_index + 1

        # Extract individual channel: (T, Z, Y, X)
        ch_stack = stack_5d[:, :, ch_index, :, :]
        ch_base_name = f"C{ch_num}_{base_name}"
        raw_out_path = os.path.join(output_folder, f"{ch_base_name}.tif")

        tifffile.imwrite(
            raw_out_path,
            ch_stack.astype(np.uint16),
            imagej=True,
            metadata={'axes': 'TZYX'}
        )

        # Run thresholding pipeline on this channel
        thresholded = process_channel_stack(
            ch_stack,
            channel_number=ch_num,
            base_name=ch_base_name,
            output_folder=output_folder
        )

        thresholded_per_channel[ch_num] = thresholded

    # Build composite thresholded image from C1–C5
    composite_channels = []
    for ch_num in range(1, 6):
        composite_channels.append(thresholded_per_channel[ch_num])

    composite_stack = np.stack(composite_channels, axis=2)  # (T, Z, C, Y, X)

    tifffile.imwrite(
        os.path.join(output_folder, "Composite_thresholded_C1C2C3C4C5.tif"),
        composite_stack.astype(np.uint8),
        imagej=True,
        metadata={'axes': 'TZCYX'}
    )

    # Build a TRUE-signal MIP composite from per-channel MIP files
    true_mip_channels = []

    for ch_num in sorted(thresholded_per_channel.keys()):
        ch_base_name = f"C{ch_num}_{base_name}"
        mip_filename = f"{ch_base_name}_mip_timelapse.tif"
        mip_path = os.path.join(output_folder, mip_filename)

        # Each MIP timelapse is (T, Y, X)
        mip_stack = tifffile.imread(mip_path)
        true_mip_channels.append(mip_stack)

    # Stack into shape (T, C, Y, X)
    true_mip_composite = np.stack(true_mip_channels, axis=1)  # axis=1 is channel

    # Save TRUE-signal MIP composite
    true_mip_out = os.path.join(output_folder, f"{base_name}_TRUE_MIP_C1_to_C{len(true_mip_channels)}.tif")
    tifffile.imwrite(
        true_mip_out,
        true_mip_composite.astype(np.uint16),
        imagej=True,
        metadata={'axes': 'TCYX'}
    )


def process_channel_stack(stack_4d, channel_number, base_name, output_folder):

    T = stack_4d.shape[0]

    thresholded_stack = np.zeros_like(stack_4d, dtype=np.uint8)
    mip_stack = []
    otsu_values = []
    thresholded_mip_stack = []  # MIPs of thresholded volumes per timepoint

    for t in range(T):
        print(f"     • Timepoint {t+1}/{T}")
        volume = stack_4d[t]  # (Z, Y, X)

        # 1. Apply Gaussian blur per Z-slice
        blurred = np.stack(
            [gaussian(slice_, sigma=gaussian_sigma, preserve_range=True) for slice_ in volume],
            axis=0
        )

        # 2. MIP and Otsu — reset min/max per timepoint
        mip = np.max(blurred, axis=0)  # (Y, X)
        mip_stack.append(mip.astype(np.uint16))

        # Apply min/max normalization before Otsu
        mip_min = float(mip.min())
        mip_max = float(mip.max())

        mip_norm = (mip - mip_min) / (mip_max - mip_min)

        norm_thresh = threshold_otsu(mip_norm)
        real_thresh = norm_thresh * (mip_max - mip_min) + mip_min
        otsu_values.append(real_thresh)

        # 3. Threshold selection logic: Otsu or fixed threshold
        if use_fixed_threshold.get(channel_number, False) and t >= fixed_threshold_start_tp.get(channel_number, 0):
            base_thresh = fixed_threshold.get(channel_number, real_thresh)
        else:
            base_thresh = real_thresh

        # Optional per-channel offset
        offset = threshold_offset.get(channel_number, 0)
        effective_thresh = base_thresh + offset

        # Apply threshold to 3D blurred volume
        binary = blurred > effective_thresh

        # 4. Optional voxel size filter
        if use_size_filter.get(channel_number, False):
            min_vox = channel_min_voxels.get(channel_number, 0)
            if min_vox > 0:
                binary = remove_small_objects(binary, min_size=min_vox)

        # 5. Optional intensity filter
        if use_intensity_filter.get(channel_number, False):
            thr = intensity_min.get(channel_number, 0)
            labeled = label(binary)
            filtered_binary = np.zeros_like(binary, dtype=bool)
            props = regionprops(labeled, intensity_image=blurred)

            for region in props:
                zc, yc, xc = map(int, region.centroid)
                if blurred[zc, yc, xc] > thr:
                    filtered_binary[labeled == region.label] = True

            binary = filtered_binary

        # MIP of thresholded (binary) volume for this timepoint: shape (Y, X)
        thr_mip = np.max(binary, axis=0)
        thresholded_mip_stack.append((thr_mip.astype(np.uint8) * 255))

        thresholded_stack[t] = (binary.astype(np.uint8) * 255)

    # Export results for this channel
    thresh_out_path = os.path.join(output_folder, f"{base_name}_thresholded_output.tif")
    mip_out_path = os.path.join(output_folder, f"{base_name}_mip_timelapse.tif")
    mip_thresh_out_path = os.path.join(output_folder, f"{base_name}_mip_timelapse_thresholded.tif")
    csv_out_path = os.path.join(output_folder, f"{base_name}_otsu_thresholds.csv")

    tifffile.imwrite(
        thresh_out_path,
        thresholded_stack,
        imagej=True,
        metadata={'axes': 'TZYX'}
    )

    tifffile.imwrite(
        mip_out_path,
        np.stack(mip_stack, axis=0).astype(np.uint16),
        imagej=True,
        metadata={'axes': 'TYX'}
    )

    tifffile.imwrite(
        mip_thresh_out_path,
        np.stack(thresholded_mip_stack, axis=0).astype(np.uint8),
        imagej=True,
        metadata={'axes': 'TYX'}
    )

    df = pd.DataFrame({
        'timepoint': list(range(len(otsu_values))),
        'otsu_threshold': otsu_values
    })
    df.to_csv(csv_out_path, index=False)

    return thresholded_stack


if __name__ == "__main__":
    main()