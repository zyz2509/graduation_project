import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class attn_2_vote(nn.Module):
    """ 
    input:
        ind_buffer:indices B*(Num_tokens, 3)
        coor_buffer:coordinates B*(Num_tokens, 3)
        feat_buffer:features B*(Num_tokens, Channels)

    output:
        ind_buffer:indices B*(V, 3)
        coor_buffer:coordinates B*(V, V', 3)
        feat_buffer:features B*(V, V', Channels)
      
    """
    def __init__(self, feat_dim, vote_factor, grid_size=0.1) -> None:
        super().__init__()
        self.vote_factor = vote_factor
        self.in_ch = feat_dim
        self.out_ch = feat_dim
        self.grid_size = grid_size
        self.conv1 = nn.Conv1d(self.in_ch, self.out_ch, 1)
        self.conv2 = nn.Conv1d(self.in_ch, self.out_ch, 1)
        self.conv3 = nn.Conv1d(self.in_ch, (3+self.out_ch)*self.vote_factor, 1)
        self.bn1 = nn.BatchNorm1d(self.in_ch)
        self.bn2 = nn.BatchNorm1d(self.in_ch)

    def forward(self, ID:list, CO:list, FT:list):
        ind_buffer = []
        coor_buffer = []
        feat_buffer = []

        for i in range(len(ID)):
            tokens, feat_dim = FT[i].shape
            votes = tokens*self.vote_factor
            log = F.relu(self.bn1(self.conv1(FT[i].unsqueeze(-1)))) # N, C, 1
            log = F.relu(self.bn2(self.conv2(log)))
            log = self.conv3(log).view(tokens, self.vote_factor, 3+feat_dim)
            offset = log[:, :, 0:3]
            coor = (CO[i].unsqueeze(1) + offset).reshape(votes, 3) # tokens, 3
            grid_res = torch.div(offset, self.grid_size).int()
            ind = (ID[i].unsqueeze(1) + grid_res).reshape(votes, 3)
            offset = log[:, :, 3:]
            
            for n in range(tokens):
                unique_dict = {}
                for idx, key in enumerate(grid_res[n]):
                    if key not in unique_dict:
                        unique_dict[key] = []
                    unique_dict[key].append(idx)
                for key, value in unique_dict.items():
                    for id in value:
                        offset[n][id] = offset[n][idx]* len(key) / self.vote_factor

            feat = (FT[i].unsqueeze(1) + offset).reshape(votes, feat_dim)
            dict = {}

            for j in range(votes):
                key = tuple(ind[j].tolist())  # Convert the tensor 'ind[j]' to a tuple
                if key not in dict:
                    dict[key] = []
                dict[key].append((coor[j], feat[j]))

            ind_m_buffer = torch.zeros((len(dict), 3), dtype=torch.int32)
            coor_m_buffer = torch.zeros((len(dict), max(len(v) for v in dict.values()), 3), dtype=torch.float32)
            feat_m_buffer = torch.zeros((len(dict), max(len(v) for v in dict.values()), self.out_ch), dtype=torch.float32)
            mask = []

            for k, (key, value) in enumerate(dict.items()):
                ratio = torch.ceil(torch.div(coor_m_buffer.size(1), len(value))).int()
                if ratio > 5:   
                    continue
                mask.append(k)
                ind_m_buffer[k] = torch.tensor(key, dtype=torch.int32)
                # 提取坐标张量和特征张量
                coor_to_repeat = torch.stack([x[0] for x in value])  
                feat_to_repeat = torch.stack([x[1] for x in value])  
                
                # 使用 repeat 函数重复填充 coor_m_buffer 和 feat_m_buffer
                coor_m_buffer[k] = coor_to_repeat.repeat(ratio, 1)[:coor_m_buffer.size(1)]
                feat_m_buffer[k] = feat_to_repeat.repeat(ratio, 1)[:feat_m_buffer.size(1)]
            
            ind_buffer.append(ind_m_buffer[mask])
            coor_buffer.append(coor_m_buffer[mask])
            feat_buffer.append(feat_m_buffer[mask])

        return ind_buffer, coor_buffer, feat_buffer

torch.cuda.synchronize()
start = time.time()
net = attn_2_vote(256, 9).cuda()
ind_buffer, coor_buffer, feat_buffer = net(torch.randint(0,10,(8,1024,3)).cuda(), 
                                           torch.rand(8,1024,3).cuda(),
                                           torch.rand(8,1024,256).cuda())

torch.cuda.synchronize()

print("ind_buffer:")
for ind_array in ind_buffer:
    print(ind_array.shape)

print("\ncoor_buffer:")
for coor_array in coor_buffer:
    print(coor_array.shape)

print(time.time()-start)