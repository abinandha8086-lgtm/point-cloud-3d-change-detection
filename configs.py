#!/usr/bin/python
# -*- coding: UTF-8 -*-
import ml_collections as mlc
import numpy as np
import os


def train_cfg():
    """returns training configuration."""

    cfg = mlc.ConfigDict()
    cfg.resume = False
    cfg.display = True
    cfg.print_rate = 5
    cfg.batch_size = 8
    cfg.epoch = 40

    # network setting
    cfg.use_rgb = False  # use 'RGB' or 'XYZRGB' as input dimensions
    if cfg.use_rgb:
        cfg.in_dim = 6
    else:
        cfg.in_dim = 3
    cfg.out_dim = 64
    # down sample rate of the input point clouds of each layer
    cfg.sub_sampling_ratio = [4, 4, 4, 4]
    cfg.down_rate = np.prod(cfg.sub_sampling_ratio)
    cfg.num_layers = len(cfg.sub_sampling_ratio)
    # The k value in LFA module
    cfg.k_neighbors = 16

    # dataset setting
    # the point number of the input point clouds
    cfg.n_samples = 8192
    cfg.remove_plane = True  # if remove the ground plane of PCs
    cfg.plane_threshold = 0.50
    cfg.norm_data = True

    # path config dict
    cfg.path = mlc.ConfigDict()

    # root of SHREC/SLPCCD dataset
    cfg.path.data_root = "/home/abinandha/go2_3d_change/data/SHREC2020-CD"

    # processed (plane-removed) folders
    # actual folders are test_seg_split_plane_8192_thr_0.5 and train_seg_split_plane_8192_thr_0.5
    cfg.path.test_dataset = os.path.join(
        cfg.path.data_root,
        f"test_seg_split_plane_{cfg.n_samples}_thr_{cfg.plane_threshold}",
    )
    cfg.path.train_dataset = os.path.join(
        cfg.path.data_root,
        f"train_seg_split_plane_{cfg.n_samples}_thr_{cfg.plane_threshold}",
    )
    cfg.path.val_dataset = cfg.path.train_dataset

    # saving processed PCs to .npy to accelerate
    cfg.if_prepare_data = True

    # where to write/read prepared .npy data
    cfg.path.prepare_data = "./data"

    # txt files stay inside repo ./data
    cfg.path.save_txt = "./data"
    cfg.path.train_txt = "./data/train.txt"
    cfg.path.val_txt = "./data/val.txt"
    cfg.path.test_txt = "./data/test.txt"

    # outputs stay under ./outputs inside repo
    cfg.path.outputs = "./outputs"
    cfg.path.weights_save_dir = "./outputs/weights"
    cfg.path.best_weights_save_dir = "./outputs/best_weights"
    cfg.path.val_prediction = "./outputs/val_prediction"
    cfg.path.test_prediction = "./outputs/test_prediction"
    cfg.path.test_prediction_PCs = "./outputs/test_prediction_PCs"
    cfg.path.feature = "./outputs/feature"

    # optimizer
    cfg.optimizer = mlc.ConfigDict()
    cfg.optimizer.type = "Adam"
    # (you can add lr, weight_decay etc. here if needed)

    # validation and testing setting
    cfg.save_prediction = True  # if save the prediction results.
    # criterion for selecting models: 'miou' or 'oa'
    cfg.criterion = "miou"

    return cfg


CONFIGS = {
    "Train": train_cfg(),
}

