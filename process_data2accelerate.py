#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import os.path as osp
import time
import random

import numpy as np
import torch
from tqdm import tqdm

import configs as cfg
import utils

tcfg = cfg.CONFIGS["Train"]


def norm_data(p16, p20):
    """Normalize concatenated point pairs to unit sphere."""
    p16_raw, p20_raw = p16, p20
    point_pair = np.vstack((p16, p20))
    idx = np.where(point_pair == 0)
    point_set = point_pair[idx, :]
    point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # center
    dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
    point_set = point_set / (dist + 1e-8)  # scale
    point_pair[idx, :] = point_set
    p16 = point_pair[: tcfg.n_samples, :]
    p20 = point_pair[tcfg.n_samples :, :]
    return p16, p20, p16_raw, p20_raw


class PrepareData:
    def __init__(self, flag, data_path, txt_path, n_samples):
        """
        flag: 'train' / 'val' / 'test'
        data_path: root where split_plane folders live
        txt_path: list file
        """
        super(PrepareData, self).__init__()
        self.flag = flag
        self.data_path = data_path
        self.txt_path = txt_path
        self.n_samples = n_samples
        with open(self.txt_path, "r") as f:
            self.list = f.readlines()
            self.file_size = len(self.list)

    def load_and_pp_data(self):
        for idx in tqdm(range(self.file_size)):
            # where to save prepared numpy file
            save_dir = os.path.join(tcfg.path.prepare_data, self.flag)
            query_path = os.path.join(save_dir, f"{idx}.npy")
            if os.path.exists(query_path):
                continue

            # entries in txt: "subdir/0_1/point2016.txt subdir/0_1/point2020.txt"
            line = self.list[idx].strip()
            left, right = line.split()

            # normalize any backslashes from original Windows txt
            left = left.replace("\\", "/")
            right = right.replace("\\", "/")

            p16_path = osp.join(self.data_path, left)
            p20_path = osp.join(self.data_path, right)

            # directory and filenames (for bookkeeping)
            dir_name = left.split("/")[0]
            pc0_name = left.split("/")[-1]
            pc1_name = right.split("/")[-1]

            p16, p20, p16_raw_length, p20_raw_length = utils.align_length(
                p16_path, p20_path, self.n_samples
            )

            p16_data = p16[:, :-1]
            p20_data = p20[:, :-1]

            if tcfg.norm_data:
                p16_data, p20_data, p16_raw, p20_raw = norm_data(p16_data, p20_data)
            else:
                p16_raw, p20_raw = p16_data, p20_data

            label16, label20 = self.generate_label(p16, p20)
            batch_data16 = self.process_data(p16_data)
            batch_data20 = self.process_data(p20_data)

            p16ofp20 = utils.search_k_neighbors(
                p20_data[:, :3], p16_data[:, :3], tcfg.k_neighbors
            )
            p20ofp16 = utils.search_k_neighbors(
                p16_data[:, :3], p20_data[:, :3], tcfg.k_neighbors
            )

            inputs16 = {}
            inputs20 = {}

            inputs16["xyz"] = [
                torch.from_numpy(data).float() for data in batch_data16[0]
            ]
            inputs16["neighbors_idx"] = [
                torch.from_numpy(data).long() for data in batch_data16[1]
            ]
            inputs16["pool_idx"] = [
                torch.from_numpy(data).long() for data in batch_data16[2]
            ]
            inputs16["unsam_idx"] = [
                torch.from_numpy(data).long() for data in batch_data16[3]
            ]
            inputs16["label"] = torch.from_numpy(label16).long()
            inputs16["knearst_idx_in_another_pc"] = torch.from_numpy(p16ofp20).long()
            inputs16["raw_length"] = p16_raw_length

            inputs20["xyz"] = [
                torch.from_numpy(data).float() for data in batch_data20[0]
            ]
            inputs20["neighbors_idx"] = [
                torch.from_numpy(data).long() for data in batch_data20[1]
            ]
            inputs20["pool_idx"] = [
                torch.from_numpy(data).long() for data in batch_data20[2]
            ]
            inputs20["unsam_idx"] = [
                torch.from_numpy(data).long() for data in batch_data20[3]
            ]
            inputs20["label"] = torch.from_numpy(label20).long()
            inputs20["knearst_idx_in_another_pc"] = torch.from_numpy(p20ofp16).long()
            inputs20["raw_length"] = p20_raw_length

            ppdata = (
                inputs16,
                inputs20,
                dir_name,
                pc0_name,
                pc1_name,
                [p16_raw, p20_raw],
            )

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            np.save(query_path, np.array(ppdata, dtype=object))

    def process_data(self, pc_data, subsam_rate=tcfg.sub_sampling_ratio):
        if pc_data.shape[1] == 3:
            xyz = pc_data
        else:
            xyz = pc_data[:, :3]

        input_points = [pc_data]
        neighbors_idx = []
        pool_idx = []
        upsam_idx = []

        for i in range(tcfg.num_layers):
            k_neigh_idx = utils.search_k_neighbors(
                xyz, xyz, tcfg.k_neighbors
            )

            sub_pc_data = pc_data[: pc_data.shape[0] // subsam_rate[i], :]
            sub_xyz = xyz[: xyz.shape[0] // subsam_rate[i], :]
            sub_idx = k_neigh_idx[: pc_data.shape[0] // subsam_rate[i], :]
            up_idx = utils.search_k_neighbors(sub_xyz, xyz, 1)

            input_points.append(sub_pc_data)
            neighbors_idx.append(k_neigh_idx)
            pool_idx.append(sub_idx)
            upsam_idx.append(up_idx)

            pc_data = sub_pc_data
            xyz = sub_xyz

        inputs_list = [input_points, neighbors_idx, pool_idx, upsam_idx]
        return inputs_list

    def generate_label(self, p16, p20):
        label16 = np.expand_dims(p16[:, -1], 1)
        label20 = np.expand_dims(p20[:, -1], 1)
        return label16, label20


def prepare_data_train():
    pptrain = PrepareData(
        "train",
        tcfg.path["train_dataset"],
        tcfg.path["train_txt"],
        tcfg.n_samples,
    )
    ppval = PrepareData(
        "val",
        tcfg.path["val_dataset"],
        tcfg.path["val_txt"],
        tcfg.n_samples,
    )
    print("processing training data...")
    pptrain.load_and_pp_data()
    print("processing val data...")
    ppval.load_and_pp_data()
    print("data preparation finished.")


def prepare_data_test():
    pptest = PrepareData(
        "test",
        tcfg.path["test_dataset"],
        tcfg.path["test_txt"],
        tcfg.n_samples,
    )
    print("processing testing data...")
    pptest.load_and_pp_data()


if __name__ == "__main__":
    prepare_data_test()

