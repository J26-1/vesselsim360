import argparse
import os
import numpy as np
import open3d as o3d
from tqdm import tqdm

def depth_to_rgbd(depth, color=None, depth_trunc=10.0):
    # depth: 2D numpy float32 meters
    depth_o3d = o3d.geometry.Image((depth).astype(np.float32))
    if color is None:
        # create dummy gray image
        gray = (np.clip(depth, 0.0, depth_trunc) / depth_trunc * 255).astype(np.uint8)
        color_o3d = o3d.geometry.Image(np.stack([gray,gray,gray], axis=2))
    else:
        # color expected HxWx3 uint8
        color_o3d = o3d.geometry.Image(color.astype(np.uint8))
    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d, depth_scale=1.0, depth_trunc=depth_trunc, convert_rgb_to_intensity=False
    )

def depth_to_pointcloud(depth, intrinsics):
    H, W = depth.shape
    fx, fy, cx, cy = intrinsics
    u = np.arange(W)
    v = np.arange(H)
    uu, vv = np.meshgrid(u, v)
    z = depth
    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy
    pts = np.stack((x, y, z), axis=2).reshape(-1,3)
    valid = (z.reshape(-1) > 0.001) & (z.reshape(-1) < 100.0)
    pts = pts[valid]
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts.astype(np.float32)))
    return pc

def pairwise_registration(source_pc, target_pc, voxel_size):
    # compute FPFH + RANSAC then ICP refinement (optional)
    # For speed, use downsample + ICP (good if frames are close)
    src = source_pc.voxel_down_sample(voxel_size)
    tgt = target_pc.voxel_down_sample(voxel_size)
    src.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
    tgt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*2, max_nn=30))
    # initial alignment with identity
    trans_init = np.identity(4)
    reg = o3d.pipelines.registration.registration_icp(
        src, tgt, max_correspondence_distance=voxel_size*1.5,
        init=trans_init,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane()
    )
    return reg.transformation

def build_poses(depth_stack, intrinsics, stride=1, voxel_size=0.05):
    N = depth_stack.shape[0]
    poses = [np.eye(4)]
    pcs = []
    print("Converting to point clouds...")
    for i in tqdm(range(0, N, stride)):
        pc = depth_to_pointcloud(depth_stack[i], intrinsics)
        pcs.append(pc)
    # pairwise registration
    print("Estimating poses (pairwise ICP)...")
    for i in tqdm(range(1, len(pcs))):
        T = pairwise_registration(pcs[i], pcs[i-1], voxel_size)
        poses.append(poses[-1] @ np.linalg.inv(T))  # accumulate (note ordering)
    return poses, pcs

def fuse_tsdf(depth_stack, poses, intrinsics, out_dir, depth_trunc=10.0, voxel_length=0.02, sdf_trunc=0.04):
    os.makedirs(out_dir, exist_ok=True)
    h, w = depth_stack.shape[1], depth_stack.shape[2]
    fx, fy, cx, cy = intrinsics
    intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)

    tsdf = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor
    )

    print("Integrating frames into TSDF...")
    for i, depth in tqdm(list(enumerate(depth_stack))):
        rgbd = depth_to_rgbd(depth, color=None, depth_trunc=depth_trunc)
        pose = poses[i] if i < len(poses) else poses[-1]
        tsdf.integrate(rgbd, intrinsic_o3d, np.linalg.inv(pose))  # integrate with camera-to-world transform
    mesh = tsdf.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    mesh_path = os.path.join(out_dir, "scene_mesh.ply")
    pcd_path = os.path.join(out_dir, "scene.pcd")
    print("Saving mesh to:", mesh_path)
    o3d.io.write_triangle_mesh(mesh_path, mesh)
    pcd = tsdf.extract_point_cloud()
    o3d.io.write_point_cloud(pcd_path, pcd)
    print("Saved mesh and pointcloud.")
    return mesh, pcd

def main(args):
    depth_stack = np.load(args.depth)  # (N, H, W)
    print("Loaded depth:", depth_stack.shape)
    intrinsics = (args.fx, args.fy, args.cx, args.cy)
    poses, pcs = build_poses(depth_stack, intrinsics, stride=args.stride, voxel_size=args.voxel)
    mesh, pcd = fuse_tsdf(depth_stack, poses, intrinsics, args.out_dir,
                          depth_trunc=args.depth_trunc, voxel_length=args.voxel_length, sdf_trunc=args.sdf_trunc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=str, default="data/fake_depth_dataset_1min.npy")
    parser.add_argument("--out-dir", type=str, default="output")
    parser.add_argument("--fx", type=float, required=True)
    parser.add_argument("--fy", type=float, required=True)
    parser.add_argument("--cx", type=float, required=True)
    parser.add_argument("--cy", type=float, required=True)
    parser.add_argument("--stride", type=int, default=2, help="use every Nth frame to speed up")
    parser.add_argument("--voxel", type=float, default=0.05, help="voxel size for ICP downsample")
    parser.add_argument("--voxel_length", type=float, default=0.02)
    parser.add_argument("--sdf_trunc", type=float, default=0.04)
    parser.add_argument("--depth_trunc", type=float, default=10.0)
    args = parser.parse_args()
    main(args)
