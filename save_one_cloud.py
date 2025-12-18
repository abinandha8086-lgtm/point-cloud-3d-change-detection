#!/usr/bin/python3
import numpy as np
import torch
from torch.utils.data import DataLoader

import configs as cfg
from dataset import CDDataset
from net import Siam3DCDNet

tcfg = cfg.CONFIGS["Train"]


def main():
    # 1. Build dataset and get first batch only
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

    # 2. Load model
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

    # 4. Take full processed cloud from first scale p0[0]: [1, 8192, 3]
    proc_xyz0 = p0[0]              # tensor [1, N, 3]
    proc_xyz0 = proc_xyz0.squeeze(0).cpu().numpy()  # [N, 3]
    xyz0 = proc_xyz0               # [N,3], N ~ 8192

    # 5. Predictions from logits: out0: [1, num_classes, N]
    logits0 = out0[0]              # [num_classes, N]
    pred_labels = logits0.argmax(dim=0).cpu().numpy()  # [N]

    # 6. Make sure lengths match
    N = min(xyz0.shape[0], pred_labels.shape[0])
    xyz0 = xyz0[:N]
    pred0 = pred_labels[:N]

    print("Saving", N, "points to xyz0.npy / pred0.npy")
    np.save("xyz0.npy", xyz0.astype(np.float32))
    np.save("pred0.npy", pred0.astype(np.int64))


if __name__ == "__main__":
    main()
