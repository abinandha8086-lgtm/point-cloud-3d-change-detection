import torch
import torch.nn as nn
import pytorch_utils as pt_utils

def gather_neighbour(pc, neighbor_idx):
    if pc.dim() == 4:
        pc = pc.squeeze(-1) 
    B, C, N = pc.shape
    K = neighbor_idx.shape[-1]
    pc = pc.transpose(2, 1).contiguous() 
    
    # Safety clamp for indices
    neighbor_idx = torch.clamp(neighbor_idx, 0, N - 1)
    
    batch_indices = torch.arange(B, device=pc.device).view(B, 1, 1)
    features = pc[batch_indices, neighbor_idx, :] 
    return features.permute(0, 3, 1, 2).contiguous()

class SPE(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.mlp2 = pt_utils.Conv2d(d_in, d_out, kernel_size=(1, 1), bn=True)
    def forward(self, feature, neigh_idx):
        f_neigh = gather_neighbour(feature, neigh_idx)
        f_agg2 = self.mlp2(f_neigh)
        return torch.sum(f_agg2, -1, keepdim=True)

class LFE(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.mlp1 = pt_utils.Conv2d(d_in, d_in, kernel_size=(1, 1), bn=True)
        self.mlp2 = pt_utils.Conv2d(d_in, d_out, kernel_size=(1, 1), bn=True)
        self.mlp3 = pt_utils.Conv2d(d_in, d_out, kernel_size=(1, 1), bn=True)
    def forward(self, feature, neigh_idx):
        f_neigh = gather_neighbour(feature, neigh_idx)
        f_neigh = self.mlp1(f_neigh)
        f_neigh = torch.sum(f_neigh, dim=-1, keepdim=True)
        f_neigh = self.mlp2(f_neigh)
        feature = self.mlp3(feature)
        return f_neigh + feature

class LFA(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.spe = SPE(d_in, d_out)
        self.lfe = LFE(d_in, d_out)
        self.mlp = pt_utils.Conv2d(d_out, d_out, kernel_size=(1, 1), bn=True)
    def forward(self, feature, neigh_idx):
        return self.mlp(self.spe(feature, neigh_idx) + self.lfe(feature, neigh_idx))
