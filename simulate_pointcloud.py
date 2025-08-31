import numpy as np
import open3d as o3d
import cv2
import os

def main():
    # === Config ===
    depth_path = "fake_depth_dataset_1min.npy"
    out_dir = "output"
    fx, fy = 150, 150   # Focal lengths
    cx, cy = 80, 60     # Principal point
    stride = 2
    video_out = os.path.join(out_dir, "pointcloud_recon.mp4")
    os.makedirs(out_dir, exist_ok=True)

    # === Load dataset ===
    depth_data = np.load(depth_path)
    n_frames, H, W = depth_data.shape
    print(f"Loaded dataset {depth_path}, shape={depth_data.shape}, "
          f"min={depth_data.min():.2f}, max={depth_data.max():.2f}")

    # === Open3D setup ===
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=640, height=480, visible=False)

    pcd = o3d.geometry.PointCloud()
    vis.add_geometry(pcd)

    # Camera control
    ctr = vis.get_view_control()
    render_opt = vis.get_render_option()
    render_opt.background_color = np.array([0, 0, 0])   # Black background
    render_opt.point_size = 2.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_out, fourcc, 20, (640, 480))

    # === Process frames ===
    for i in range(n_frames):
        depth = depth_data[i]

        # Convert depth map to 3D points
        points = []
        for v in range(0, H, stride):
            for u in range(0, W, stride):
                Z = depth[v, u]
                if Z > 0:
                    X = (u - cx) * Z / fx
                    Y = (v - cy) * Z / fy
                    points.append([X, -Y, Z])  # flip Y

        if not points:
            continue

        pcd.points = o3d.utility.Vector3dVector(points)

        # Update visualization
        vis.update_geometry(pcd)

        if i == 0:
            vis.reset_view_point(True)  # set a good initial viewpoint

        vis.poll_events()
        vis.update_renderer()

        # Capture rendered frame
        img = np.asarray(vis.capture_screen_float_buffer(do_render=True))
        img = (img * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        writer.write(img)

        if i % 50 == 0:
            print(f"[Frame {i}/{n_frames}] rendered")

    writer.release()
    vis.destroy_window()
    print(f"✅ Saved point cloud video to {video_out}")

if __name__ == "__main__":
    main()

