"""
render_video.py
Load scene mesh / pcd created by reconstruct.py, render a rotation video and save as MP4.
Usage:
    python render_video.py --scene output/scene_mesh.ply --out output/reconstruction.mp4 --width 1280 --height 720 --frames 180
"""

import argparse
import open3d as o3d
import numpy as np
import imageio
import os
from tqdm import tqdm

def render_rotate(mesh_path, out_path, width=1280, height=720, frames=180):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    # Create visualizer and capture images
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False)
    vis.add_geometry(mesh)

    ctr = vis.get_view_control()
    bbox = mesh.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    extent = np.linalg.norm(bbox.get_extent())
    # set camera
    cam = ctr.convert_to_pinhole_camera_parameters()
    cam.extrinsic = np.eye(4)
    ctr.convert_from_pinhole_camera_parameters(cam)

    # initial distance
    ctr.set_lookat(center)
    ctr.set_up([0, -1, 0])
    ctr.set_front([0, 0, -1])
    ctr.set_zoom(0.8)

    images = []
    for i in tqdm(range(frames)):
        # rotate around center
        angle = 2.0 * np.pi * i / frames
        R = o3d.geometry.get_rotation_matrix_from_xyz([0, angle, 0])
        mesh_rot = mesh.rotate(R, center=center)
        vis.update_geometry(mesh)
        vis.poll_events()
        vis.update_renderer()
        img = vis.capture_screen_float_buffer(False)
        img = (255*np.asarray(img)).astype('uint8')
        images.append(img)
    vis.destroy_window()
    # save video
    imageio.mimwrite(out_path, images, fps=30, quality=8)
    print("Saved video to:", out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, required=True)
    parser.add_argument("--out", type=str, default="output/reconstruction.mp4")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--frames", type=int, default=180)
    args = parser.parse_args()
    render_rotate(args.scene, args.out, args.width, args.height, args.frames)
