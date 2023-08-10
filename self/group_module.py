import numpy as np
import torch
import torch.nn as nn

def point_2_voxel(point_cloud, voxel_size):
    """
    将点云进行体素化，并只保存非空体素

    参数:
    point_cloud: numpy数组,形状为(B, N, 3),表示B个点云,每个点云有N个点,每个点有3个坐标(x, y, z)。
    voxel_size: 体素的尺寸，单位为米。

    返回:
    ind_buffer: 保存非空体素的体素坐标的列表,每个元素为一个numpy数组,形状为(V, 3),V为对应批次的非空体素数量。
    coor_buffer: 保存实际点坐标的列表,每个元素为一个numpy数组,形状为(V, V_max, 3),V_max为对应批次中所有体素中包含的点的最大数量。
    """
    B, N, _ = point_cloud.shape

    # 初始化保存非空体素的ind_buffer和coor_buffer
    ind_buffer = []
    coor_buffer = []

    # 将点云数据映射到对应的体素中
    for b in range(B):
        voxel_dict = {}

        for n in range(N):
            x, y, z = point_cloud[b, n]
            i, j, k = int(x / voxel_size), int(y / voxel_size), int(z / voxel_size)

            # 将点添加到体素中
            voxel_key = (i, j, k)
            if voxel_key not in voxel_dict:
                voxel_dict[voxel_key] = []

            voxel_dict[voxel_key].append(point_cloud[b, n])

        # 保存非空体素的信息到ind_buffer和coor_buffer
        non_empty_voxels = len(voxel_dict)
        ind_array = np.zeros((non_empty_voxels, 3), dtype=np.int32)
        coor_array = np.zeros((non_empty_voxels, max(len(v) for v in voxel_dict.values()), 3), dtype=np.float32)

        for idx, (key, value) in enumerate(voxel_dict.items()):
            ind_array[idx] = [key[0], key[1], key[2]]
            ratio = np.ceil(coor_array.shape[1]/len(value))
            coor_array[idx, :] = np.tile(value, (int(ratio), 1))[0:coor_array.shape[1]]

        ind_buffer.append(ind_array)
        coor_buffer.append(coor_array)

    return ind_buffer, coor_buffer

class voxel_2_attn(nn.Module):
    """ 
    module to aggregate features from points within the same voxel 

    input:
        coor_buffer(V, V_max, 3):the same as above except shape is (V, V_max, 3)

    output:
        coor_buffer(V, V_max, 3):the same as above except shape is (V, 3)
        feat_buffer(V, V_max, C):features in the shape of (V, C)
       
    """
    def __init__(self, out_ch, feat_dim=3) -> None:
        super().__init__()
        self.in_ch = feat_dim
        self.out_ch = out_ch
        self.conv = nn.Conv1d(self.in_ch, self.out_ch, 1)
        self.bn = nn.BatchNorm1d(self.out_ch)
        self.pooling = nn.AdaptiveAvgPool1d(output_size=1)

    def forward(self, x:np.array):
        coor_buffer = []
        feat_buffer = []

        for coor in x:
            if isinstance(coor, np.array):
                coor = torch.Tensor(coor) # V, 3, V_max
            net = self.bn(self.conv(coor.transpose(2,1))) # V, C, V_max
            net = self.pooling(net).squeeze() # V, C
            coor = self.pooling(coor).squeeze() # V, 3
            coor_buffer.append(coor)
            feat_buffer.append(net)

        return coor_buffer, feat_buffer
    

# 示例用法
# 假设有一个形状为(B, N, 3)的点云数据
point_cloud = np.random.rand(8,100000,3)*10

# 体素尺寸设置为0.1米
voxel_size = 0.1

ind_buffer, coor_buffer = point_2_voxel(point_cloud, voxel_size)

# 打印结果
print("ind_buffer:")
for ind_array in ind_buffer:
    print(ind_array.shape)

print("\ncoor_buffer:")
for coor_array in coor_buffer:
    print(coor_array.shape)