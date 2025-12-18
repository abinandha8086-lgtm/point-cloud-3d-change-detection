#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import time
import numpy as np

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import configs as cfg
from net import Siam3DCDNet
from dataset import CDDataset

tcfg = cfg.CONFIGS["Train"]


def test_network(tcfg):
    # 1. Build dataset and dataloader
    test_dataset = CDDataset(
        tcfg.path["test_dataset"],
        tcfg.path["test_txt"],
        tcfg.n_samples,
        "test",
        tcfg.path.prepare_data,
    )
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 2. Load pretrained model
    best_model_path = "/home/abinandha/go2_3d_change/data/best_net.pth"
    ckpt = torch.load(best_model_path, map_location="cpu")
    pretrained_dict = ckpt["model_state_dict"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Siam3DCDNet(tcfg.in_dim, tcfg.out_dim).to(device)
    net.load_state_dict(pretrained_dict, strict=False)
    net.eval()

    # 3. Inference loop
    dur = 0.0
    tqdm_loader = tqdm(test_dataloader, total=len(test_dataloader))

    for batch_idx, data in enumerate(tqdm_loader):
        (
            batch_data0,
            batch_data1,
            dir_name,
            pc0_name,
            pc1_name,
            raw_data,
        ) = data

        # unpack dicts from dataset
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

        # move to device
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

        lb0 = lb0.squeeze(-1).to(device, dtype=torch.long)
        lb1 = lb1.squeeze(-1).to(device, dtype=torch.long)

        # 4. Forward
        t0 = time.time()
        with torch.no_grad():
            out0, out1 = net(
                [p0, p0_neighbors_idx, p0_pool_idx, p0_unsam_idx],
                [p1, p1_neighbors_idx, p1_pool_idx, p1_unsam_idx],
                knearest_idx,
            )
        dur += time.time() - t0

        # --- SAVE FULL PROCESSED CLOUD + LABELS FOR FIRST SAMPLE ONLY ---
        if batch_idx == 0:
            # p0[0] is [1, 8192, 3] → full processed cloud for cloud 0
            proc_xyz0 = p0[0]  # tensor [1, N, 3]
            proc_xyz0 = proc_xyz0.squeeze(0).cpu().numpy()  # [N, 3]
            xyz0 = proc_xyz0  # [N, 3]

            # out0: [B, num_classes, N] → logits for cloud 0
            logits0 = out0[0]  # [num_classes, N]
            pred_labels = logits0.argmax(dim=0).cpu().numpy()  # [N]

            # Ensure matching length
            L = min(xyz0.shape[0], pred_labels.shape[0])
            xyz0 = xyz0[:L]
            pred0 = pred_labels[:L]

            np.save("xyz0.npy", xyz0.astype(np.float32))
            np.save("pred0.npy", pred0.astype(np.int64))
            print("Saved FULL xyz0.npy and pred0.npy for visualization.")
        # --- END SAVE BLOCK ---

        # metrics disabled; continue loop

    print("FPS: ", len(test_dataloader) / dur)


if __name__ == "__main__":
    test_network(tcfg)

