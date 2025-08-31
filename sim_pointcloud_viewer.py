import os, math, time
import numpy as np
import imageio.v2 as imageio
import open3d as o3d

# Sensor-specific parameters
DEPTH_SCALE = 1000.0  # convert mm to meters
W, H = 100, 100
hfov_deg = 70.0

# Intrinsic computation
hfov = math.radians(hfov_deg)
fx = (W / 2) / math.tan(hfov / 2)
fy = fx
cx, cy = (W - 1) / 2.0, (H - 1) / 2.0

DATA_PATH = "datasets/zhaw_depth_dataset.npy"
OUTPUT_MP4 = "pointcloud_recon.mp4"
VIDEO_FPS = 20
SHOW_WINDOW = True  # set False to speed up off-screen rendering

def depth_to_pointcloud(depth_m):
    i, j = np.meshgrid(np.arange(W), np.arange(H))
    Z = depth_m
    mask = Z > 0
    u, v = i[mask], j[mask]
    Z = Z[mask]
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    return np.stack((X, Y, Z), axis=-1)

def main():
    assert os.path.exists(DATA_PATH)
    depth_seq = np.load(DATA_PATH)  # (T, H, W)
    depth_seq = depth_seq.astype(np.float32) / DEPTH_SCALE
    T = len(depth_seq)

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=SHOW_WINDOW, width=640, height=480)
    render_opt = vis.get_render_option()
    render_opt.point_size = 2
    render_opt.background_color = np.array((0, 0, 0), dtype=np.float32)

    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)
    ctr = vis.get_view_control()
    ctr.set_front([0.0, 0.0, -1.0])
    ctr.set_up([0.0, -1.0, 0.0])
    ctr.set_lookat([0, 0, 1.0])
    ctr.set_zoom(0.6)

    writer = imageio.get_writer(OUTPUT_MP4, fps=VIDEO_FPS)
    t0 = time.time()

    for frame in depth_seq:
        pts = depth_to_pointcloud(frame)
        if pts.size == 0:
            continue
        colors = plt.cm.jet((pts[:,2] - pts[:,2].min()) / (pts[:,2].ptp() + 1e-6))[:, :3]
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        vis.update_geometry(pcd)
        vis.poll_events(); vis.update_renderer()
        img = np.asarray(vis.capture_screen_float_buffer(False)) * 255
        writer.append_data(img.astype(np.uint8))
    writer.close()
    vis.destroy_window()
    print(f"Saved pointcloud video to {OUTPUT_MP4}")

if __name__ == "__main__":
    main()

