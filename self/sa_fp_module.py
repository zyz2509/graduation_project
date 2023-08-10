import torch
import torch.nn as nn
import time

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def ball_query(radius, nsample, xyz, new_xyz):
    """
    输入:
        radius: 局部区域半径
        nsample: 局部区域中的最大采样点数
        xyz: 所有点，[B, N, 3]
        new_xyz: 查询点，[B, S, 3]，S表示中心点的数量
    返回:
        group_idx: 分组后的点索引，[B, S, nsample]
    """
    device = xyz.device
    B, N, _ = xyz.shape
    _, S, _ = new_xyz.shape

    sqrdists = torch.sum((new_xyz.unsqueeze(2) - xyz.unsqueeze(1))**2, dim=-1)

    # 使用torch.where将group_idx张量中的无效索引（N）替换掉
    group_idx = torch.where(sqrdists > radius ** 2, N, torch.arange(N, dtype=torch.long) \
                            .unsqueeze(0).unsqueeze(0).expand(B, S, N).to(device))

    # 排序并选择每个中心点的前nsample个点
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]

    # mask用于指示具有无效索引（N）的点
    mask = group_idx == N

    # 使用第一个点的索引替换无效索引
    group_idx[mask] = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])[mask]

    return group_idx

def knn_query(k, xyz):
    
    device = xyz.device
    idx = torch.sum((xyz.unsqueeze(2) - xyz.unsqueeze(1))**2, dim=-1).argsort()[:, :, :k]

    return idx


x = (torch.rand(8,2048,3)*10).cuda()
y = (torch.rand(8,1024,3)*10).cuda()
torch.cuda.synchronize
start = time.time()
z = ball_query(0.4, 8, x, y)
print(time.time()-start)
torch.cuda.synchronize             