#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import numpy as np
import torch
from torch.utils.data import DataLoader

import configs as cfg
from dataset import CDDataset
from net import Siam3DCDNet

tcfg = cfg.CONFIGS["Train"]


def main():
    # 1. Build test dataset & loader (same as test.py)
    test_dataset = CDDataset(
        tcfg.path["test_dataset"],
        tcfg.path["test_txt"],
        tcfg.n_samples,
        "test",
        tcfg.path.prepare_data,
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    # 2. Load pretrained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Siam3DCDNet(tcfg.in_dim, tcfg.out_dim).to(device)
    best_model_path = "/home/abinandha/3d_pc_change/data/best_net.pth"
    ckpt = torch.load(best_model_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    # 3. Take first test sample only
    inputs16, inputs20, dirname, pc0_name, pc1_name, raw_data = next(iter(test_loader))

    # move tensors to device
    for k in ["xyz", "neighbors_idx", "pool_idx", "unsam_idx", "label", "knearst_idx_in_another_pc"]:
        if isinstance(inputs16[k], list):
            inputs16[k] = [t.to(device) for t in inputs16[k]]
            inputs20[k] = [t.to(device) for t in inputs20[k]]
        else:
            inputs16[k] = inputs16[k].to(device)
            inputs20[k] = inputs20[k].to(device)

    with torch.no_grad():
        pred0, pred1 = model(
            [
                inputs16["xyz"],
                inputs16["neighbors_idx"],
                inputs16["pool_idx"],
                inputs16["unsam_idx"],
            ],
            [
                inputs20["xyz"],
                inputs20["neighbors_idx"],
                inputs20["pool_idx"],
                inputs20["unsam_idx"],
            ],
            [
                inputs16["knearst_idx_in_another_pc"],
                inputs20["knearst_idx_in_another_pc"],
            ],
        )

    # logits -> labels [B, N]
    pred0 = pred0.max(dim=-1)[1].detach().cpu().numpy()[0]
    pred1 = pred1.max(dim=-1)[1].detach().cpu().numpy()[0]

    # xyz for PC0/PC1: [B, N, 3] -> [N, 3]
    xyz0 = inputs16["xyz"][0].detach().cpu().numpy()[0]
    xyz1 = inputs20["xyz"][0].detach().cpu().numpy()[0]

    os.makedirs("outputs/example", exist_ok=True)
    np.save("outputs/example/xyz0.npy", xyz0)
    np.save("outputs/example/xyz1.npy", xyz1)
    np.save("outputs/example/pred0.npy", pred0)
    np.save("outputs/example/pred1.npy", pred1)

    print("Saved:")
    print("  outputs/example/xyz0.npy, pred0.npy  (time 1)")
    print("  outputs/example/xyz1.npy, pred1.npy  (time 2)")
    print("dirname:", dirname[0], "pc0_name:", pc0_name[0], "pc1_name:", pc1_name[0])


if __name__ == "__main__":
    main()
