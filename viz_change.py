import os
import numpy as np
import matplotlib.pyplot as plt


POINTCLOUD_DIR = "pointcloud_txts"
MASK_NAME = "gt_mask_0000.npy"   # from rubiks_project
PC_BEFORE = "pc_0000.txt"        # before
PC_AFTER = "pc_0001.txt"         # after


def load_points():
    pc0 = np.loadtxt(os.path.join(POINTCLOUD_DIR, PC_BEFORE)).astype(np.float32)
    pc1 = np.loadtxt(os.path.join(POINTCLOUD_DIR, PC_AFTER)).astype(np.float32)
    return pc0, pc1


def load_gt_mask(n_points):
    mask_path = os.path.join(POINTCLOUD_DIR, MASK_NAME)
    if not os.path.exists(mask_path):
        raise FileNotFoundError(mask_path)
    m = np.load(mask_path).astype(bool)
    if m.shape[0] != n_points:
        raise ValueError("GT mask length mismatch")
    return m


def visualize_exact_changes():
    pc0, pc1 = load_points()
    gt_mask = load_gt_mask(pc0.shape[0])

    # unchanged: gray, changed: red
    unchanged = pc0[~gt_mask]
    changed = pc0[gt_mask]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Exact Changed Points (GT)")

    ax.scatter(unchanged[:, 0], unchanged[:, 1], unchanged[:, 2],
               c='lightgray', s=3, marker='o')
    ax.scatter(changed[:, 0], changed[:, 1], changed[:, 2],
               c='red', s=5, marker='o')

    ax.set_axis_off()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize_exact_changes()

