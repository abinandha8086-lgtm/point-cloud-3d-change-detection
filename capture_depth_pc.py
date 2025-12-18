#!/usr/bin/python3
import pyrealsense2 as rs
import open3d as o3d
import numpy as np
import cv2
from pathlib import Path
from camera_intrinsics import CAMERA_PARAMS

pc_dir = Path("pointcloud_txts")
pc_dir.mkdir(exist_ok=True)

# 1. Initialize Pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, CAMERA_PARAMS["width"], CAMERA_PARAMS["height"], rs.format.z16, 30)
profile = pipeline.start(config)

# 2. Hardware Tuning for Small Objects
device = profile.get_device()
depth_sensor = device.first_depth_sensor()

# Set to 'High Accuracy' preset if available
if depth_sensor.supports(rs.option.visual_preset):
    depth_sensor.set_option(rs.option.visual_preset, 3) # 3 is often High Accuracy

# 3. Precision Filters
# Threshold is critical: ignore everything further than 50cm to remove room noise
threshold = rs.threshold_filter(min_dist=0.1, max_dist=0.5) 
spatial = rs.spatial_filter()   
temporal = rs.temporal_filter() 
hole_filling = rs.hole_filling_filter()

# IMPORTANT: Depth scale correction
depth_scale = depth_sensor.get_depth_scale()

# Get Intrinsics
intr = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
pinhole_intr = o3d.camera.PinholeCameraIntrinsic(intr.width, intr.height, intr.fx, intr.fy, intr.ppx, intr.ppy)

print(f"=== PRECISION CAPTURE (Scale: {depth_scale}) ===")
print("Focusing on 10cm - 50cm range. Press SPACE to capture.")

try:
    capture_id = 0
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        if not depth_frame: continue

        # Apply Filters
        filtered = threshold.process(depth_frame)
        filtered = spatial.process(filtered)
        filtered = temporal.process(filtered)
        filtered = hole_filling.process(filtered)

        # Convert to numpy and apply METRIC scale
        depth_data = np.asanyarray(filtered.get_data())
        
        # Visualize
        depth_viz = cv2.applyColorMap(cv2.convertScaleAbs(depth_data, alpha=0.1), cv2.COLORMAP_JET)
        cv2.imshow("Capture View", depth_viz)
        
        key = cv2.waitKey(1)
        if key == 32: # SPACE
            # Use Open3D to create the cloud
            depth_o3d = o3d.geometry.Image(depth_data)
            pcd = o3d.geometry.PointCloud.create_from_depth_image(
                depth_o3d, 
                pinhole_intr, 
                depth_scale=1.0/depth_scale, # This forces METERS
                depth_trunc=0.5
            )
            
            # Clean outliers immediately
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            
            save_path = pc_dir / f"pc_{capture_id:04d}.txt"
            np.savetxt(save_path, np.asarray(pcd.points), fmt="%.6f")
            print(f"✓ Saved METRIC point cloud: {save_path}")
            capture_id += 1
        elif key == 27:
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
