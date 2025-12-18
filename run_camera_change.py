import numpy as np
import torch
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
import sys

sys.path.append(".")
from net import Siam3DCDNet

# CONFIGURATION - TUNED FOR SMALL OBJECTS (EARPODS)
PC1_PATH = "pointcloud_txts/pc_0000.txt"
PC2_PATH = "pointcloud_txts/pc_0001.txt"
N_POINTS = 8192  
# DECREASED: 0.005 means 5mm voxels. This preserves the earpod box shape.
VOXEL_SIZE = 0.05
SCALE_CHECK = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_point_cloud(path, target_n):
    # Load raw XYZ data
    try:
        raw_data = np.loadtxt(path)[:, :3].astype(np.float32)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

    # DEBUG: Check if data is in meters. If the max value is > 10, it's likely in mm.
    if np.max(np.abs(raw_data)) > 10.0:
        raw_data /= 1000.0
        
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(raw_data)

    # Clean edges and isolate the small object
   # pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)

    points = np.asarray(pcd.points)
    if len(points) == 0:
        print(f"Warning: {path} resulted in 0 points after preprocessing!")
        return np.zeros((target_n, 3))

    if len(points) >= target_n:
        idx = np.random.choice(len(points), target_n, replace=False)
        points = points[idx]
    else:
        idx = np.random.choice(len(points), target_n, replace=True)
        points = points[idx]
    
    return points

def build_pyramid(pc):
    """
    Creates the point cloud hierarchy required by the 3DCDNet architecture.
    """
    xyz_list, neigh_list, pool_list, up_list = [], [], [], []
    counts = [8192, 2048, 512, 128, 32] 
    
    curr_pc = pc
    hierarchy = [pc]
    for i in range(4):
        # Downsample by 4 at each level
        next_pc = curr_pc[::4][:counts[i+1]]
        hierarchy.append(next_pc)
        curr_pc = next_pc

    for i in range(5):
        p = hierarchy[i]
        xyz_list.append(torch.from_numpy(p).float().unsqueeze(0).to(device))
        
        # Local neighborhood
        nn = NearestNeighbors(n_neighbors=16).fit(p)
        idx = nn.kneighbors(p, return_distance=False)
        neigh_list.append(torch.from_numpy(idx).long().unsqueeze(0).to(device))
        
        if i < 4:
            # Pooling indices
            p_idx = torch.arange(len(hierarchy[i+1])).long().view(1, -1, 1).to(device)
            pool_list.append(p_idx)
            
            # Upsampling indices for skip connections
            p_next = hierarchy[i+1]
            nn_up = NearestNeighbors(n_neighbors=1).fit(p_next)
            u_idx = nn_up.kneighbors(p, return_distance=False)
            up_list.append(torch.from_numpy(u_idx).long().unsqueeze(0).to(device))

    return [xyz_list, neigh_list, pool_list, up_list]

# --- MAIN EXECUTION ---
print("--- Step 1: High-Sensitivity Preprocessing ---")
pc1 = preprocess_point_cloud(PC1_PATH, N_POINTS)
pc2 = preprocess_point_cloud(PC2_PATH, N_POINTS)

# SIAMESE SETUP
in0 = build_pyramid(pc1)
in1 = build_pyramid(pc2)

# Global correspondence (Cross-frame matching)
nn01 = NearestNeighbors(n_neighbors=1).fit(pc2)
idx01 = nn01.kneighbors(pc1, return_distance=False)
nn10 = NearestNeighbors(n_neighbors=1).fit(pc1)
idx10 = nn10.kneighbors(pc2, return_distance=False)
k_idx = [torch.from_numpy(idx01).long().unsqueeze(0).to(device), 
         torch.from_numpy(idx10).long().unsqueeze(0).to(device)]

print("--- Step 2: Inference ---")
model = Siam3DCDNet(3, 64).to(device)
try:
    ckpt = torch.load("./outputs/best_weights/best_net.pth", map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
except Exception as e:
    print(f"Weights not found, using random init: {e}")

model.eval()

with torch.no_grad():
    # out0: Log-probabilities for each point in PC1
    out0, _ = model(in0, in1, k_idx)

# WE LOWER THE SOFTMAX THRESHOLD TO "SEE" WEAKER CHANGES
probs = torch.exp(out0[0]) # Convert log_softmax to probability
change_prob = probs[:, 1].cpu().numpy() # Probability of "change" class

# Threshold: Points with > 35% probability of change are marked red
pred = (change_prob > 0.90).astype(int) 

print(f"--- SUCCESS! Found {np.sum(pred)} change points ---")

# --- Step 3: Heatmap Visualization ---
pcd_out = o3d.geometry.PointCloud()
pcd_out.points = o3d.utility.Vector3dVector(pc1)

# Color logic: Static = Gray, High-Confidence Change = Pure Red, Low-Confidence = Pink/Orange
colors = np.zeros((len(pc1), 3))
for i in range(len(pc1)):
    p = change_prob[i]
    if pred[i] == 1:
        # Detected change: Scale from Pink to Red based on confidence
        colors[i] = [1.0, 1.0 - p, 1.0 - p] 
    else:
        # Static environment: Gray
        colors[i] = [0.6, 0.6, 0.6]

pcd_out.colors = o3d.utility.Vector3dVector(colors)

# Instructions for user
print("\n[VISUALIZATION]")
print("- Gray points: Static")
print("- Red/Pink points:changes")
print("- Use mouse to rotate, scroll to zoom in close to the table surface.")

o3d.visualization.draw_geometries([pcd_out], window_name="Siam3DCDNet Heatmap")
