# feature_extraction.py
import os, csv
import numpy as np
import cv2
import OpenEXR, Imath, array

# Input depth map folder
DEPTH_FOLDER = "depth_maps"   # put your .exr files here
OUTPUT_CSV = "datasets/labels.csv"

os.makedirs("datasets", exist_ok=True)

def load_exr_depth(path):
    """Load depth map from EXR file using OpenEXR bindings"""
    exr_file = OpenEXR.InputFile(path)
    header = exr_file.header()
    dw = header['dataWindow']
    width, height = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

    # EXR stores as RGB(A); depth usually in one channel (R)
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    raw = exr_file.channel('R', pt)   # read Red channel
    depth = np.frombuffer(raw, dtype=np.float32)
    depth.shape = (height, width)

    return depth


def extract_features(depth_map):
    """Extract min depth features from left, center, right, and whole image"""
    h, w = depth_map.shape
    thirds = w // 3

    left = depth_map[:, :thirds]
    center = depth_map[:, thirds:2*thirds]
    right = depth_map[:, 2*thirds:]

    min_all = np.nanmin(depth_map)
    min_left = np.nanmin(left)
    min_center = np.nanmin(center)
    min_right = np.nanmin(right)

    return min_all, min_left, min_center, min_right

def assign_label(features, threshold=5.0):
    """
    Simple rule-based label assignment for training:
    0 = Safe
    1 = Turn Left
    2 = Turn Right
    3 = Stop
    """
    min_all, min_left, min_center, min_right = features

    if min_all > threshold:
        return 0  # Safe
    elif min_center < threshold:
        return 3  # Stop (obstacle ahead)
    elif min_left < threshold:
        return 2  # Turn Right
    elif min_right < threshold:
        return 1  # Turn Left
    else:
        return 0  # Safe by default

def main():
    files = sorted([f for f in os.listdir(DEPTH_FOLDER) if f.endswith(".exr")])
    if not files:
        print(f"No EXR files found in {DEPTH_FOLDER}")
        return

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "min_all", "min_left", "min_center", "min_right", "label"])

        for fname in files:
            path = os.path.join(DEPTH_FOLDER, fname)
            depth_map = load_exr_depth(path)
            features = extract_features(depth_map)
            label = assign_label(features)

            writer.writerow([fname, *features, label])

    print(f"✅ Features saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
