#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import numpy as np
from tqdm import tqdm

import configs as cfg

tcfg = cfg.CONFIGS["Train"]


def load_xyz(path):
    """
    Robustly load XYZ from a SHREC-style point file.

    - Skips empty lines.
    - Skips comment/header lines starting with // or #.
    - Skips any line with < 3 numeric values.
    - Uses only the first 3 values (x, y, z).
    """
    xyz_list = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("//") or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                x, y, z = map(float, parts[:3])
                xyz_list.append([x, y, z])
            except ValueError:
                # Non-numeric row; skip
                continue

    if not xyz_list:
        raise ValueError(f"No valid xyz rows found in {path}")

    return np.array(xyz_list, dtype=np.float32)


def process_data(root, split, n=8192):
    """
    Randomly sample n points from each point cloud in
    {root}/{split}/scene_id/point2016.txt and point2020.txt,
    and save them under {root}/{split}_split_plane_{n}_thr_0.5.
    """
    src_dir = os.path.join(root, split)
    dst_dir = os.path.join(root, f"{split}_split_plane_{n}_thr_0.5")
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    scene_ids = os.listdir(src_dir)

    for scene in tqdm(scene_ids):
        scene_path = os.path.join(src_dir, scene)
        if not os.path.isdir(scene_path):
            continue

        p16_path = os.path.join(scene_path, "point2016.txt")
        p20_path = os.path.join(scene_path, "point2020.txt")
        if not (os.path.exists(p16_path) and os.path.exists(p20_path)):
            continue

        # Robust load: xyz only
        p16 = load_xyz(p16_path)  # [N1,3]
        p20 = load_xyz(p20_path)  # [N2,3]

        if p16.ndim == 1:
            p16 = p16.reshape(1, -1)
        if p20.ndim == 1:
            p20 = p20.reshape(1, -1)

        # sample indices
        idx16 = np.random.choice(p16.shape[0], n, replace=p16.shape[0] < n)
        idx20 = np.random.choice(p20.shape[0], n, replace=p20.shape[0] < n)

        p16_xyz = p16[idx16]  # already 3 cols
        p20_xyz = p20[idx20]

        out_scene_dir = os.path.join(dst_dir, scene)
        os.makedirs(out_scene_dir, exist_ok=True)

        sp16 = os.path.join(out_scene_dir, "point2016.txt")
        sp20 = os.path.join(out_scene_dir, "point2020.txt")

        # 3 columns -> 3 formats
        np.savetxt(sp16, p16_xyz, fmt="%.8f %.8f %.8f")
        np.savetxt(sp20, p20_xyz, fmt="%.8f %.8f %.8f")


def generate_removed_plane_dataset_test(root):
    """
    Wrapper for test split, using tcfg.n_samples.
    """
    process_data(root, "test_seg", n=tcfg.n_samples)


def generate_removed_plane_dataset_train(root):
    """
    Wrapper for train split, using tcfg.n_samples.
    """
    process_data(root, "train_seg", n=tcfg.n_samples)

