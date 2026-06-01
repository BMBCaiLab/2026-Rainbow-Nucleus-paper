Rainbow Nucleus MLO Analysis Pipeline

Overview
These code are associated with "Rainbow Nucleus Charts Dynamic Interactome of Membrane-less Organelles"

------------------------------------------------------------
1. Segmentation of MLO.py
------------------------------------------------------------

Input:
    5D TIFF stack (T, Z, C, Y, X)
    C1 = Histone Locus Body (HLB)
    C2 = Nuclear Speckle (NS)
    C3 = Cajal Body (CB)
    C4 = Nucleolus
    C5 = PML Nuclear Body (PML-NB)

Processing:
    - Gaussian blur sigma = 1.5
    - MIP-based Otsu thresholding
    - Optional threshold offset
    - Optional fixed threshold
    - Optional size filtering
    - Optional intensity filtering

Output:
    C1-C5 individual channel TIFFs
    *_thresholded_output.tif
    *_mip_timelapse.tif
    *_mip_timelapse_thresholded.tif
    *_otsu_thresholds.csv
    Composite_thresholded_C1C2C3C4C5.tif

Purpose:
    Generate binary masks for downstream analysis.

------------------------------------------------------------
2. MLO Tracking, Contact detection.py
------------------------------------------------------------

Input:
    - *_thresholded_output.tif, generated from 1. Segmentation of MLO.py

Processing:
    - Labels individual objects
    - Tracks C1, C3, C4, and C5 using IoU matching
    - Assigns persistent track IDs
    - Detects contacts between MLOs
    - Uses dilation for contact detection
    - Converts frame-by-frame contacts into interaction bouts

Output:
    tracked_labels.tif
    interactions_per_frame.csv
    interaction_bouts.csv

Purpose:
    Identify MLO contacts over time from thresholded masks.

------------------------------------------------------------
3. MLO Contact analysis KDE plot.py
------------------------------------------------------------

Input:
    Contact counts and time summary tables. Each contact pair needs to be separated by an empty row.

Processing:
    - 2D histogram generation
    - Kernel density estimation (KDE)
    - 3D visualization

Output:
    *_KDE_floor.png
    *_KDE_floor_black.png (this is used for high contrast for poster, not included in the paper)
    *_3D_columns_transparent.png

Purpose:
    Visualize characteristic interaction dynamics of different MLO pairs.

------------------------------------------------------------
4. Object detection and closest surface-surface distance.py
------------------------------------------------------------

Input:
    - *_thresholded_output.tif, generated from 1. Segmentation of MLO.py

Processing:
    - Labels individual MLOs
    - Counts objects
    - Calculates centroids
    - Calculates minimum 3D surface-to-surface distance

Output:
    0. objects and center.xlsx
    0. min_surf to surf distance.xlsx

Purpose:
    Quantify MLO abundance, spatial organization, and nearest-neighbor relationships.

------------------------------------------------------------
5. MLO trajectory following Actuator perturbation.py
------------------------------------------------------------

Input:
    - Isolated ROI. TOF file from Actuator perturbation file. Input C1 = Cajal Body (CB); C2 = Histone Locus Body (HLB).

Processing:
    - Otsu thresholding
    - Center-of-mass detection
    - Trajectory reconstruction

Output:
    roi_*_trajectory_channels_TCYX.tif

Output channels:
    C1 = original C2
    C2 = original C1
    C3 = trajectory of C2
    C4 = trajectory of C1

Purpose:
    Visualize MLO movement trajectories following perturbation.

------------------------------------------------------------
6. MLO Randomization.py
------------------------------------------------------------

Input:
    - *_thresholded_output.tif, generated from 1. Segmentation of MLO.py. It needs to be single time point.

Processing:
    - Randomly relocates MLO objects within the nuclear mask
    - Preserves object shape and size
    - Repeats randomization N = 100 times
    - Calculates randomized contact frequencies

Output:
    organelle_object_counts.csv
    real_interactions.csv
    random_mean_interactions.csv
    random_probability_interactions.csv
    real_interactions_heatmap.png

Purpose:
    Generate randomized controls to determine whether observed MLO interactions exceed random expectation.

