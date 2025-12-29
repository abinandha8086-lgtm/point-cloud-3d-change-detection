import cv2
import numpy as np
import open3d as o3d
import os

def process_general_3d_change(ref_path, cur_path):
    img_ref = cv2.imread(ref_path)
    img_cur = cv2.imread(cur_path)
    if img_ref is None or img_cur is None: return None, None, None, None

    gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    gray_cur = cv2.cvtColor(img_cur, cv2.COLOR_BGR2GRAY)
    
    # 1. CLEAN BACKGROUND (Ceiling, Fan, Pipes)
    # We use a very strong bilateral filter to keep the fan/pipes but flatten the ceiling grain
    bg_depth_map = cv2.bilateralFilter(gray_cur, 15, 80, 80)
    height, width = gray_cur.shape
    v, u = np.mgrid[0:height, 0:width]
    
    # Standard scene depth
    z = (255.0 - bg_depth_map.astype(float)) * 0.0015 

    # 2. ROBUST CHANGE DETECTION (Works for any object)
    diff = cv2.absdiff(cv2.GaussianBlur(gray_ref, (11,11), 0), 
                       cv2.GaussianBlur(gray_cur, (11,11), 0))
    _, mask = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    mask = cv2.dilate(mask, np.ones((5,5), np.uint8), iterations=2)

    # 3. THE "REAL OBJECT" FIX
    # Instead of a spike, we give the object a fixed, rounded thickness
    # This ensures a bottle looks like a bottle, not a spike.
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)
    
    # We cap the depth so it doesn't become a long spike
    # 0.15 is the thickness of the object. Increase for 'thicker' objects.
    object_thickness = dist_transform[mask > 0] * 0.15 
    z[mask > 0] = np.mean(z) + 0.2 + object_thickness 

    # 4. PROJECTION
    fx, fy = 650.0, 650.0
    cx, cy = width / 2.0, height / 2.0
    x = (u - cx) * (z + 1.5) / fx
    y = (v - cy) * (z + 1.5) / fy

    # 5. COLORS
    colors = np.full((height, width, 3), [0.75, 0.75, 0.75]) # Light Grey for visibility
    colors[mask > 0] = [1.0, 0.0, 0.0] # Red

    return x, y, z, colors

def main():
    base_dir = "/home/abinandha/3d_pc_change/3DCDNet/depth_captures"
    # Fallback to check for both png and jpg
    for ext in ['.jpg', '.png']:
        ref = os.path.join(base_dir, f"reference{ext}")
        cur = os.path.join(base_dir, f"current{ext}")
        if os.path.exists(ref): break

    x, y, z, colors = process_general_3d_change(ref, cur)
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.column_stack((x.flatten(), y.flatten(), z.flatten())))
    pcd.colors = o3d.utility.Vector3dVector(colors.reshape(-1, 3))
    pcd.estimate_normals()

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Realistic 3D Change Detection")
    vis.add_geometry(pcd)
    vis.get_render_option().light_on = True
    vis.get_render_option().point_size = 3.0
    vis.get_render_option().background_color = np.array([0.1, 0.1, 0.1])
    
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    main()
