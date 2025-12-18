#!/usr/bin/python3
import numpy as np
import torch
from torch.utils.data import DataLoader
import open3d as o3d

import configs as cfg
from dataset import CDDataset
from net import Siam3DCDNet

tcfg = cfg.CONFIGS["Train"]


def main():
    # 1. Build dataset and take first test sample
    ds = CDDataset(
        tcfg.path["test_dataset"],
        tcfg.path["test_txt"],
        tcfg.n_samples,
        "test",
        tcfg.path.prepare_data,
    )
    dl = DataLoader(ds, batch_size=1, shuffle=False)
    batch_data0, batch_data1, dir_name, pc0_name, pc1_name, raw_data = next(iter(dl))

    (
        p0,
        p0_neighbors_idx,
        p0_pool_idx,
        p0_unsam_idx,
        lb0,
        knearest_idx0,
        raw_length0,
    ) = [v for v in batch_data0.values()]

    (
        p1,
        p1_neighbors_idx,
        p1_pool_idx,
        p1_unsam_idx,
        lb1,
        knearest_idx1,
        raw_length1,
    ) = [v for v in batch_data1.values()]

    print("len(p0):", len(p0))
    for i, t in enumerate(p0):
        print(f"p0[{i}] shape:", t.shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move to device
    p0 = [t.to(device, dtype=torch.float32) for t in p0]
    p0_neighbors_idx = [t.to(device, dtype=torch.long) for t in p0_neighbors_idx]
    p0_pool_idx = [t.to(device, dtype=torch.long) for t in p0_pool_idx]
    p0_unsam_idx = [t.to(device, dtype=torch.long) for t in p0_unsam_idx]

    p1 = [t.to(device, dtype=torch.float32) for t in p1]
    p1_neighbors_idx = [t.to(device, dtype=torch.long) for t in p1_neighbors_idx]
    p1_pool_idx = [t.to(device, dtype=torch.long) for t in p1_pool_idx]
    p1_unsam_idx = [t.to(device, dtype=torch.long) for t in p1_unsam_idx]

    knearest_idx = [
        knearest_idx0.to(device, dtype=torch.long),
        knearest_idx1.to(device, dtype=torch.long),
    ]

    # 2. Load pretrained model
    best_model_path = "/home/abinandha/go2_3d_change/data/best_net.pth"
    ckpt = torch.load(best_model_path, map_location=device)
    net = Siam3DCDNet(tcfg.in_dim, tcfg.out_dim).to(device)
    net.load_state_dict(ckpt["model_state_dict"], strict=False)
    net.eval()

    # 3. Forward once
    with torch.no_grad():
        out0, out1 = net(
            [p0, p0_neighbors_idx, p0_pool_idx, p0_unsam_idx],
            [p1, p1_neighbors_idx, p1_pool_idx, p1_unsam_idx],
            knearest_idx,
        )

    print("out0 shape:", out0.shape)  # expect [1, N, 2] or [1, N, 2, 1]

    # 4. Get per-point predictions
    if out0.ndim == 3 and out0.shape[2] == 2:
        # [1, N, 2]
        logits0 = out0[0]                   # [N, 2]
        pred0 = logits0.argmax(dim=1)       # [N]
    elif out0.ndim == 4 and out0.shape[2] == 2:
        # [1, N, 2, 1]
        logits0 = out0[0, :, :, 0]          # [N, 2]
        pred0 = logits0.argmax(dim=1)       # [N]
    elif out0.ndim == 3 and out0.shape[1] == 2:
        # [1, 2, N]
        logits0 = out0[0]                   # [2, N]
        pred0 = logits0.argmax(dim=0)       # [N]
    else:
        raise RuntimeError(f"Unexpected out0 shape: {out0.shape}")

    pred0 = pred0.cpu().numpy()            # [N]

    # 5. Full xyz from finest scale p0[0]: [1, N, 3]
    xyz0 = p0[0].squeeze(0).cpu().numpy()  # [N, 3]

    print("xyz0 shape:", xyz0.shape)
    print("pred0 shape:", pred0.shape)

    # 6. Ensure same length
    N = min(xyz0.shape[0], pred0.shape[0])
    xyz0 = xyz0[:N]
    pred0 = pred0[:N]

    print("Saving", N, "points to xyz0.npy / pred0.npy")
    np.save("xyz0.npy", xyz0.astype(np.float32))
    np.save("pred0.npy", pred0.astype(np.int64))

    # 7. Visualize full cloud
    colors = np.zeros((N, 3), dtype=np.float64)
    colors[pred0 == 1] = [1.0, 0.0, 0.0]
    colors[pred0 == 0] = [0.0, 0.0, 1.0]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz0)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    print("Opening Open3D viewer with full cloud...")
    o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    main()

