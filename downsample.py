import open3d as o3d

# Load your big ply
pcd = o3d.io.read_point_cloud("reconstruction.ply")

# Downsample (voxel size controls resolution)
downpcd = pcd.voxel_down_sample(voxel_size=0.02)

# Save smaller version
o3d.io.write_point_cloud("reconstruction_down.ply", downpcd)
