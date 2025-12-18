import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_utils as pt_utils
from LocalFeatureAggregation import LFA, gather_neighbour

class C3Dnet(nn.Module):
    def __init__(self, in_d, out_d):
        super(C3Dnet, self).__init__()
        self.in_d = in_d
        self.out_d = out_d
        self.fc0 = pt_utils.Conv1d(self.in_d, 64, kernel_size=1, bn=True)
        self.block1 = LFA(64, 128)
        self.block2 = LFA(128, 256)
        self.block3 = LFA(256, 512)
        self.block4 = LFA(512, 1024)
        self.dt = pt_utils.Conv2d(1024, 1024, kernel_size=(1, 1), bn=True)
        self.d4 = pt_utils.Conv2d(1024*2, 512, kernel_size=(1, 1), bn=True)
        self.d3 = pt_utils.Conv2d(512*2, 256, kernel_size=(1, 1), bn=True)
        self.d2 = pt_utils.Conv2d(256*2, 128, kernel_size=(1, 1), bn=True)
        self.d1 = pt_utils.Conv2d(128*2, 64, kernel_size=(1, 1), bn=True)
        self.d0 = pt_utils.Conv2d(64, self.out_d, kernel_size=(1, 1), bn=True)
        
    def forward(self, end_points): 
        xyz, neigh_idx, pool_idx, unsam_idx = end_points
        
        # Encoder
        out0 = self.fc0(xyz[0].permute(0, 2, 1)).unsqueeze(dim=3) 
        
        out1 = self.block1(out0, neigh_idx[0])
        out1p = self.random_sample(out1, pool_idx[0])
        
        out2 = self.block2(out1p, neigh_idx[1])
        out2p = self.random_sample(out2, pool_idx[1])
        
        out3 = self.block3(out2p, neigh_idx[2])
        out3p = self.random_sample(out3, pool_idx[2])
        
        out4 = self.block4(out3p, neigh_idx[3])
        out4p = self.random_sample(out4, pool_idx[3])
        
        # Decoder (U-Net skip connections)
        out = self.dt(out4p)
        out = torch.cat((out, out4p), 1)
        out = self.d4(out)
        
        out = self.nearest_interpolation(out, unsam_idx[3])
        out = torch.cat((out, out3p), 1)
        out = self.d3(out)
        
        out = self.nearest_interpolation(out, unsam_idx[2])
        out = torch.cat((out, out2p), 1)
        out = self.d2(out)
        
        out = self.nearest_interpolation(out, unsam_idx[1])
        out = torch.cat((out, out1p), 1)
        out = self.d1(out)
        
        out = self.nearest_interpolation(out, unsam_idx[0])
        out = self.d0(out)
        
        return out
    
    @staticmethod
    def random_sample(feature, pool_idx):
        # feature: [B, C, N, 1]
        # pool_idx: [B, N_out, 1]
        B, C, N, _ = feature.shape
        # Flatten index for gather
        idx = pool_idx.transpose(1, 2).expand(B, C, -1) # [B, C, N_out]
        pool_features = torch.gather(feature.squeeze(3), 2, idx)
        return pool_features.unsqueeze(3)
    
    @staticmethod
    def nearest_interpolation(feature, interp_idx):
        # feature: [B, C, N_small, 1]
        # interp_idx: [B, N_large, 1]
        B, C, N, _ = feature.shape
        idx = interp_idx.transpose(1, 2).expand(B, C, -1) # [B, C, N_large]
        interpolated_features = torch.gather(feature.squeeze(3), 2, idx)
        return interpolated_features.unsqueeze(3)

class Siam3DCDNet(nn.Module):
    def __init__(self, in_d, out_d):
        super(Siam3DCDNet, self).__init__()
        self.net = C3Dnet(in_d, out_d)
        self.mlp1 = pt_utils.Conv1d(64, 32, kernel_size=1, bn=True)
        self.mlp2 = pt_utils.Conv1d(32, 2, kernel_size=1, bias=False, bn=False, activation=None)
        
    def forward(self, end_points0, end_points1, knearest_idx):
        out0 = self.net(end_points0)
        out1 = self.net(end_points1)
        
        knearest_01, knearest_10 = knearest_idx
        fout0 = self.nearest_feature_difference(out0, out1, knearest_01)
        fout1 = self.nearest_feature_difference(out1, out0, knearest_10)
        
        fout0 = self.mlp1(fout0.squeeze(-1))
        fout1 = self.mlp1(fout1.squeeze(-1))
        fout0 = self.mlp2(fout0)
        fout1 = self.mlp2(fout1)
        
        fout0 = F.log_softmax(fout0.transpose(2, 1), dim=-1)
        fout1 = F.log_softmax(fout1.transpose(2, 1), dim=-1)
        return fout0, fout1
    
    @staticmethod
    def nearest_feature_difference(raw, query, nearest_idx):
        # nearest_idx: [B, N, 1]
        B, C, N, _ = raw.shape
        idx = nearest_idx.transpose(1, 2).expand(B, C, -1)
        nearest_features = torch.gather(query.squeeze(3), 2, idx).unsqueeze(3)
        diff = torch.abs(raw - nearest_features)
        return diff
