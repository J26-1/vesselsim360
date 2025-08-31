import cv2
import numpy as np
import json
import os
from glob import glob


# -------- SETTINGS --------
RGB_DIR = "rgb"
DEPTH_DIR = "depth_maps"
INTRINSICS_PATH = "intrinsics.json"
EXTRINSICS_PATH = "camera_extrinsics.npy"
OUTPUT_PLY = "reconstruction.ply"

# Blender clipping planes (adjust to your scene!)
NEAR = 0.1
FAR = 100.0

# -------- HELPERS --------
def load_intrinsics(json_path):
    import json
    with open(json_path, "r") as f:
        intr = json.load(f)

    fx = intr["fx"]
    fy = intr["fy"]
    cx = intr["cx"]
    cy = intr["cy"]

    K = np.array([
        [fx, 0,  cx],
        [0,  fy, cy],
        [0,  0,  1]
    ], dtype=np.float32)

    return K, intr["width"], intr["height"]


def load_extrinsics(path):
    return np.load(path)

def read_depth(depth_path):
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    if depth is None:
        raise RuntimeError(f"Failed to load depth {depth_path}")

    # If RGB 16-bit -> take one channel
    if depth.ndim == 3:
        depth = depth[:, :, 0]

    depth = depth.astype(np.float32)

    # ---- Heuristic check ----
    if depth.max() > 100:  
        # Probably already in meters (Blender Z-pass)
        return depth
    else:
        # Probably normalized 0–1 -> rescale to meters
        depth /= depth.max()  # normalize
        depth = NEAR * (1.0 - depth) + FAR * depth
        return depth

def depth_to_points(depth, K, extrinsic):
    h, w = depth.shape
    i, j = np.meshgrid(np.arange(w), np.arange(h))
    pixels = np.stack((i, j, np.ones_like(i)), axis=-1).reshape(-1, 3).T

    # backproject to camera coords
    K_inv = np.linalg.inv(K)
    rays = K_inv @ pixels
    rays = rays / rays[2, :]

    cam_points = rays * depth.reshape(-1)

    # to world coords
    ones = np.ones((1, cam_points.shape[1]))
    homog = np.vstack((cam_points, ones))
    world = extrinsic @ homog
    return world[:3, :].T

def save_ply(filename, points, colors=None):
    with open(filename, "w") as f:
        f.write("ply\nformat ascii 1.0\nelement vertex {}\n".format(len(points)))
        if colors is not None:
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        else:
            f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for i, p in enumerate(points):
            if colors is not None:
                c = colors[i]
                f.write("{} {} {} {} {} {}\n".format(p[0], p[1], p[2], c[0], c[1], c[2]))
            else:
                f.write("{} {} {}\n".format(p[0], p[1], p[2]))

# -------- MAIN --------
if __name__ == "__main__":
    K, W, H = load_intrinsics(INTRINSICS_PATH)
    print("Loaded intrinsics matrix:\n", K)
    extrinsics = load_extrinsics(EXTRINSICS_PATH)
    rgb_files = sorted(glob(os.path.join(RGB_DIR, "*.png")))
    depth_files = sorted(glob(os.path.join(DEPTH_DIR, "*.png")))

    print(f"Loaded intrinsics: {K}")
    print(f"Loaded {len(extrinsics)} extrinsic matrices.")
    print(f"Found {len(rgb_files)} RGB frames.")
    print(f"Found {len(depth_files)} depth maps.")

    all_points, all_colors = [], []

    for idx, (rgb_path, depth_path) in enumerate(zip(rgb_files, depth_files)):
        rgb = cv2.imread(rgb_path)[:, :, ::-1]  # BGR → RGB
        depth = read_depth(depth_path)
        extrinsic = extrinsics[idx]

        pts = depth_to_points(depth, K, extrinsic)

        # Match color to depth
        h, w = depth.shape
        colors = rgb.reshape(-1, 3)

        all_points.append(pts)
        all_colors.append(colors)

        if idx % 25 == 0:
            print(f"Processed frame {idx}/{len(rgb_files)}")

    all_points = np.concatenate(all_points, axis=0)
    all_colors = np.concatenate(all_colors, axis=0)

    save_ply(OUTPUT_PLY, all_points, all_colors)
    print(f"Saved {OUTPUT_PLY}, total {len(all_points)} points.")
