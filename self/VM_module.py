import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn
import time

class SAModule(nn.Module):
    """ 
        fast
    """
    def __init__(self, feat_dim, grid=0.2, lam=32) -> None:
        super().__init__()
        self.grid = grid
        self.F = feat_dim
        self.C = lam
        self.conv = nn.Sequential(
            nn.Conv1d(feat_dim, lam, 1),
            nn.BatchNorm1d(lam),
            nn.ReLU()
        )
        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x: torch.Tensor, y: torch.Tensor, z: list):
        s = time.time()
        bx = rnn.pad_sequence(torch.split(x, z), True, 1) # B, T, 3
        by = rnn.pad_sequence(torch.split(y, z), True, -1) # B, T, F
        s1 = time.time()

        bv = torch.div(bx, self.grid).int()
        xyz_max = bv.max(1)[0] - bv.min(1)[0] # B, 3
        xyz_max = torch.cuda.FloatTensor([[1, xyz_max[b, 0], xyz_max[b, 0]*xyz_max[b, 1]] for b in range(len(z))]).unsqueeze(1)
        index_bv = (xyz_max * bv).sum(-1) # B, T
        s2 = time.time()

        index_bv, ind = index_bv.sort(-1)
        bx = torch.gather(bx, 1, ind.unsqueeze(-1).expand(-1, -1, 3))
        by = torch.gather(by, 1, ind.unsqueeze(-1).expand(-1, -1, self.F))
        s3 = time.time()

        _, inversed, cnt = torch.unique_consecutive(index_bv, return_inverse=True, return_counts=True)

        max_v = inversed[:, -1] - inversed[:, 0] + 1
        counts = (max_v - 1).tolist()
        cnt = torch.split(cnt, max_v.tolist())

        max_v = torch.max(max_v).item()
        cnt = torch.stack([F.pad(cnti, pad=(0, max_v-cnti.shape[0]), value=1) for cnti in cnt])
        s4 = time.time()

        by = self.conv(by.transpose(-1, -2)) # B, C, T

        s5 = time.time()
        coordinates = torch.zeros((len(z), max_v, 3), device=bx.device, requires_grad=False)
        output = torch.zeros((len(z), self.C, max_v), device=by.device)

        inversed = inversed - inversed[:, 0].unsqueeze(-1)
        coordinates.scatter_add_(1, inversed.unsqueeze(-1).expand(-1, -1, 3), bx) # B, V, 3
        output.scatter_add_(-1, inversed.unsqueeze(1).expand(-1, self.C, -1), by) # B, C, V
        s6 = time.time()

        coordinates = torch.div(coordinates, cnt.unsqueeze(-1)) # B, V, 3
        output = torch.div(output.transpose(-1,-2), cnt.unsqueeze(-1)) # B, V, C
        s7 = time.time()

        coordinates = torch.cat([coordinates[b, :cnti] for b, cnti in zip(range(len(z)), counts)])
        output = torch.cat([output[b, :cnti] for b, cnti in zip(range(len(z)), counts)])
        
        s8 = time.time()

        # print(f's1-s:{s1 - s}')
        # print(f's2-s1:{s2-s1}')
        # print(f's3-s2:{s3-s2}')
        # print(f's4-s3:{s4-s3}')
        # print(f's5-s4:{s5-s4}')
        # print(f's6-s5:{s6-s5}')
        # print(f's7-s6:{s7-s6}')
        # print(f's8-s7:{s8-s7}')

        return coordinates, output, counts






