import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np

import math
import spconv

from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce
from torch_scatter import scatter_softmax, scatter_sum

from pointops2.functions import pointops

from lib import pointnet2_utils as pointutils

LEAKY_RATE = 0.1
use_bn = False
# use_leaky = False
# occ_threshold = 0.8

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B,_,C = points.shape
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx.long(), :]
    
    # points_flatten = points.reshape(-1, C).contiguous()
    # idx_flatten = idx.reshape(-1).contiguous().to(device)
    # new_points = (points_flatten[idx_flatten, :]).reshape(B,-1,C).contiguous()
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
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


def knn_point(k, pos1, pos2):
    '''
    Input:
        k: int32, number of k in k-nn search
        pos1: (batch_size, ndataset, c) float32 array, input points
        pos2: (batch_size, npoint, c) float32 array, query points
    Output:
        val: (batch_size, npoint, k) float32 array, L2 distances
        idx: (batch_size, npoint, k) int32 array, indices to input points
    '''
    B, N, C = pos1.shape
    M = pos2.shape[1]
    pos1 = pos1.view(B,1,N,-1).repeat(1,M,1,1)
    pos2 = pos2.view(B,M,1,-1).repeat(1,1,N,1)
    dist = torch.sum(-(pos1-pos2)**2,-1)
    val,idx = dist.topk(k=k,dim = -1)
    return torch.sqrt(-val), idx


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    mask = group_idx != N
    cnt = mask.sum(dim=-1)
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx, cnt


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx, _ = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    if points != None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points != None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


def index_points_group(points, knn_idx):
    """
    Input:
        points: input points data, [B, N, C]
        knn_idx: sample index data, [B, N, K]
    Return:
        new_points:, indexed points data, [B, N, K, C]
    """
    points_flipped = points.permute(0, 2, 1).contiguous()
    new_points = pointutils.grouping_operation(points_flipped, knn_idx.int()).permute(0, 2, 3, 1)

    return new_points


def group_query(nsample, s_xyz, xyz, s_points):
    """
    Input:
        nsample: scalar
        s_xyz: input points position data, [B, N, C]
        s_points: input points data, [B, N, D]
        xyz: input points position data, [B, S, C]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    B, N, C = s_xyz.shape
    S = xyz.shape[1]
    new_xyz = xyz
    idx = knn_point(nsample, s_xyz, new_xyz)[1]
    grouped_xyz = index_points_group(s_xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    if s_points is not None:
        grouped_points = index_points_group(s_points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm

    return new_points, grouped_xyz_norm


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, mlp2 = None, group_all = False):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.mlp2_convs = nn.ModuleList()
        last_channel = in_channel+3
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1, bias = False))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        if mlp2 != None:
            for out_channel in mlp2:
                self.mlp2_convs.append(nn.Sequential(nn.Conv1d(last_channel, out_channel, 1, bias=False),
                                                    nn.BatchNorm1d(out_channel)))
                last_channel = out_channel
        # if group_all:
        #     self.queryandgroup = pointutils.GroupAll()
        # else:
        #     self.queryandgroup = pointutils.QueryAndGroup(radius, nsample)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, S, C]
            new_points_concat: sample points feature data, [B, S, D']
        """
        device = xyz.device
        B, C, N = xyz.shape
        xyz_t = xyz.permute(0, 2, 1).contiguous()
        # if points != None:
        #     points = points.permute(0, 2, 1).contiguous()

        # 选取邻域点
        if self.group_all == False:
            fps_idx = pointutils.furthest_point_sample(xyz_t, self.npoint)  # [B, N]
            new_xyz = pointutils.gather_operation(xyz, fps_idx)  # [B, C, N]
        else:
            new_xyz = xyz
        new_xyz_t = new_xyz.permute(0,2,1).contiguous()
        points_t = points.permute(0,2,1).contiguous()
        # new_points = self.queryandgroup(xyz_t, new_xyz.transpose(2, 1).contiguous(), points) # [B, 3+C, N, S]
        new_points, grouped_xyz_norm = group_query(self.nsample, xyz_t, new_xyz_t, points_t)# [B, N, S, 3+C]
        new_points = new_points.permute(0,3,2,1).contiguous()
        # new_xyz: sampled points position data, [B, C, npoint]
        # new_points: sampled points data, [B, C+D, npoint, nsample]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, -2)[0]

        for i, conv in enumerate(self.mlp2_convs):
            new_points = F.relu(conv(new_points))
        return new_xyz, new_points, fps_idx


class PointNetSetUpConv(nn.Module):
    def __init__(self, nsample, radius, f1_channel, f2_channel, mlp, mlp2, knn = True):
        super(PointNetSetUpConv, self).__init__()
        self.nsample = nsample
        self.radius = radius
        self.knn = knn
        self.mlp1_convs = nn.ModuleList()
        self.mlp2_convs = nn.ModuleList()
        last_channel = f2_channel+3
        for out_channel in mlp:
            self.mlp1_convs.append(nn.Sequential(nn.Conv2d(last_channel, out_channel, 1, bias=False),
                                                 nn.BatchNorm2d(out_channel),
                                                 nn.ReLU(inplace=False)))
            last_channel = out_channel
        if len(mlp) != 0:
            last_channel = mlp[-1] + f1_channel
        else:
            last_channel = last_channel + f1_channel
        for out_channel in mlp2:
            self.mlp2_convs.append(nn.Sequential(nn.Conv1d(last_channel, out_channel, 1, bias=False),
                                                 nn.BatchNorm1d(out_channel),
                                                 nn.ReLU(inplace=False)))
            last_channel = out_channel

    def forward(self, pos1, pos2, feature1, feature2):
        """
            Feature propagation from xyz2 (less points) to xyz1 (more points)

        Inputs:
            xyz1: (batch_size, 3, npoint1)
            xyz2: (batch_size, 3, npoint2)
            feat1: (batch_size, channel1, npoint1) features for xyz1 points (earlier layers, more points)
            feat2: (batch_size, channel1, npoint2) features for xyz2 points
        Output:
            feat1_new: (batch_size, npoint2, mlp[-1] or mlp2[-1] or channel1+3)

            TODO: Add support for skip links. Study how delta(XYZ) plays a role in feature updating.
        """
        pos1_t = pos1.permute(0, 2, 1).contiguous()
        pos2_t = pos2.permute(0, 2, 1).contiguous()
        B,C,N = pos1.shape
        if self.knn:
            _, idx = pointutils.knn(self.nsample, pos1_t, pos2_t)
        else:
            idx, _ = query_ball_point(self.radius, self.nsample, pos2_t, pos1_t)
        
        pos2_grouped = pointutils.grouping_operation(pos2, idx)
        pos_diff = pos2_grouped - pos1.view(B, -1, N, 1)    # [B,3,N1,S]

        feat2_grouped = pointutils.grouping_operation(feature2, idx)
        feat_new = torch.cat([feat2_grouped, pos_diff], dim = 1)   # [B,C1+3,N1,S]
        for conv in self.mlp1_convs:
            feat_new = conv(feat_new)
        # max pooling
        feat_new = feat_new.max(-1)[0]   # [B,mlp1[-1],N1]
        # concatenate feature in early layer
        if feature1 != None:
            feat_new = torch.cat([feat_new, feature1], dim=1)
        # feat_new = feat_new.view(B,-1,N,1)
        for conv in self.mlp2_convs:
            feat_new = conv(feat_new)
        
        return feat_new

class PointNetFeaturePropogation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropogation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, pos1, pos2, feature1, feature2):
        """
        Input:
            xyz1: input points position data, [B, C, N]
            xyz2: sampled input points position data, [B, C, S]
            points1: input points data, [B, D, N]
            points2: input points data, [B, D, S]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        pos1_t = pos1.permute(0, 2, 1).contiguous()
        pos2_t = pos2.permute(0, 2, 1).contiguous()
        B, C, N = pos1.shape
        
        # dists = square_distance(pos1, pos2)
        # dists, idx = dists.sort(dim=-1)
        # dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]
        dists,idx = pointutils.three_nn(pos1_t,pos2_t)
        dists[dists < 1e-10] = 1e-10
        weight = 1.0 / dists
        weight = weight / torch.sum(weight, -1,keepdim = True)   # [B,N,3]
        interpolated_feat = torch.sum(pointutils.grouping_operation(feature2, idx) * weight.view(B, 1, N, 3), dim = -1) # [B,C,N,3]

        if feature1 != None:
            feat_new = torch.cat([interpolated_feat, feature1], 1)
        else:
            feat_new = interpolated_feat
        
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            feat_new = F.relu(bn(conv(feat_new)))
        return feat_new


'''
为了解决基于距离权重上采样带来的偏差（强假设：局部邻域的运动一致性，但是点云的稠密程度不一致，在低分辨率时容易带来较大的误差），采用了基于VFE的特征提取方法，并利用参考点的VFE与KNN近邻得到的K个点基于估计的SF在目标域和源域分别采样对应的VFE，然后计算KNN的距离差值，以及VFE_SRC和VFE_TGT,即（Delta_dist, VFE_SRC, VFE_TGT）基于CNN网络得到K个加权值，然后得到对应的高密度的点云的对应的粗略场景流信息。
'''
class UpsampleSFFeaturePropogation(nn.Module):
    def __init__(self, nsample, radius, f1_channel, f2_channel, mlp, mlp2, knn = True):
        super(UpsampleSFFeaturePropogation, self).__init__()
        self.nsample = nsample
        self.radius = radius
        self.knn = knn
        self.mlp1_convs = nn.ModuleList()
        self.mlp2_convs = nn.ModuleList()
        last_channel = f2_channel+3
        for out_channel in mlp:
            self.mlp1_convs.append(nn.Sequential(nn.Conv2d(last_channel, out_channel, 1, bias=False),
                                                 nn.BatchNorm2d(out_channel),
                                                 nn.ReLU(inplace=False)))
            last_channel = out_channel
        if len(mlp) != 0:
            last_channel = mlp[-1] + f1_channel
        else:
            last_channel = last_channel + f1_channel
        for out_channel in mlp2:
            self.mlp2_convs.append(nn.Sequential(nn.Conv1d(last_channel, out_channel, 1, bias=False),
                                                 nn.BatchNorm1d(out_channel),
                                                 nn.ReLU(inplace=False)))
            last_channel = out_channel

    def forward(self, pos1, pos2, feats1, feats2, sparse_sf):
        """
            Feature propagation from xyz2 (less points) to xyz1 (more points)

        Inputs:
            xyz1: (batch_size, 3, npoint1)
            xyz2: (batch_size, 3, npoint2)
            feat1: (batch_size, channel1, npoint1) features for xyz1 points (earlier layers, more points)
            feat2: (batch_size, channel1, npoint2) features for xyz2 points
        Output:
            feat1_new: (batch_size, npoint2, mlp[-1] or mlp2[-1] or channel1+3)

            TODO: Add support for skip links. Study how delta(XYZ) plays a role in feature updating.
        """
        pos1_t = pos1.permute(0, 2, 1).contiguous()
        pos2_t = pos2.permute(0, 2, 1).contiguous()
        B,C,N = pos1.shape
        if self.knn:
            _, idx = pointutils.knn(self.nsample, pos1_t, pos2_t)
        else:
            idx, _ = query_ball_point(self.radius, self.nsample, pos2_t, pos1_t)
    
        
        pos2_grouped = pointutils.grouping_operation(pos2, idx)
        pos_diff = pos2_grouped - pos1.view(B, -1, N, 1)    # [B,3,N1,S]

        feat2_grouped = pointutils.grouping_operation(feats2, idx)
        feat_new = torch.cat([feat2_grouped, pos_diff], dim = 1)   # [B,C1+3,N1,S]


class BilinearFeedForward(nn.Module):

    def __init__(self, in_planes1, in_planes2, out_planes):
        super().__init__()
        self.bilinear = nn.Bilinear(in_planes1, in_planes2, out_planes)

    def forward(self, x):
        x = x.contiguous()
        x = self.bilinear(x, x)
        return x


class PointMixerIntraSetLayerPaper(nn.Module):
    expansion = 1

    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
        super().__init__()
        mid_planes = out_planes 
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.nsample = nsample

        self.channelMixMLPs01 = nn.Sequential( # input.shape = [N, K, C]
            nn.Linear(3+in_planes, nsample),
            nn.ReLU(inplace=True),
            BilinearFeedForward(nsample, nsample, nsample))
        
        self.linear_p = nn.Sequential( # input.shape = [N, K, C]
            nn.Linear(3, 3),
            nn.Sequential(
                Rearrange('n k c -> n c k'),
                nn.BatchNorm1d(3),
                Rearrange('n c k -> n k c')),
            nn.ReLU(inplace=True), 
            nn.Linear(3, out_planes))
        self.shrink_p = nn.Sequential(
            Rearrange('n k (a b) -> n k a b', b=nsample),
            Reduce('n k a b -> n k b', 'sum', b=nsample))

        self.channelMixMLPs02 = nn.Sequential( # input.shape = [N, K, C]
            Rearrange('n k c -> n c k'),
            nn.Conv1d(nsample+nsample, mid_planes, kernel_size=1, bias=False),
            nn.BatchNorm1d(mid_planes), 
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_planes, mid_planes//share_planes, kernel_size=1, bias=False),
            nn.BatchNorm1d(mid_planes//share_planes),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_planes//share_planes, out_planes//share_planes, kernel_size=1),
            Rearrange('n c k -> n k c'))

        self.channelMixMLPs03 = nn.Linear(in_planes, out_planes)
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, pxo):
        p, x, o = pxo  # (n, 3), (n, c), (b)

        x_knn, knn_idx = pointops.queryandgroup(
            self.nsample, p, p, x, None, o, o, use_xyz=True, return_idx=True)  # (n, k, 3+c)
        p_r = x_knn[:, :, 0:3]

        energy = self.channelMixMLPs01(x_knn) # (n, k, k)
        
        p_embed = self.linear_p(p_r) # (n, k, out_planes)
        p_embed_shrink = self.shrink_p(p_embed) # (n, k, k)

        energy = torch.cat([energy, p_embed_shrink], dim=-1)
        energy = self.channelMixMLPs02(energy) # (n, k, out_planes/share_planes)
        w = self.softmax(energy)

        x_v = self.channelMixMLPs03(x)  # (n, in_planes) -> (n, k)
        n = knn_idx.shape[0]; knn_idx_flatten = knn_idx.flatten()
        x_v  = x_v[knn_idx_flatten, :].view(n, self.nsample, -1)

        n, nsample, out_planes = x_v.shape
        x_knn = (x_v + p_embed).view(n, nsample, self.share_planes, out_planes//self.share_planes)
        # x_knn = x_v.view(n, nsample, self.share_planes, out_planes//self.share_planes)
        x_knn = (x_knn * w.unsqueeze(2))
        x_knn = x_knn.reshape(n, nsample, out_planes)

        x = x_knn.sum(1)
        return (x, x_knn, knn_idx, p_r)


class PointMixerIntraSetLayerPaperv3(nn.Module):
    expansion = 1

    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
        super().__init__()
        mid_planes = out_planes // 1
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.nsample = nsample

        self.channelMixMLPs01 = nn.Sequential( # input.shape = [N*K, C]
            nn.Linear(3+in_planes, nsample),
            nn.ReLU(inplace=True),
            BilinearFeedForward(nsample, nsample, nsample))
        
        self.linear_p = nn.Sequential( # input.shape = [N*K, C]
            nn.Linear(3, 3),
            nn.BatchNorm1d(3),
            nn.ReLU(inplace=True), 
            nn.Linear(3, out_planes))
        self.shrink_p = nn.Sequential(
            Rearrange('n k (a b) -> n k a b', b=nsample),
            Reduce('n k a b -> (n k) b', 'sum', b=nsample))

        self.channelMixMLPs02 = nn.Sequential( # input.shape = [N*K, C]
            nn.Linear(nsample+nsample, mid_planes, bias=False),
            nn.BatchNorm1d(mid_planes), 
            nn.ReLU(inplace=True),
            nn.Linear(mid_planes, mid_planes//share_planes, bias=False),
            nn.BatchNorm1d(mid_planes//share_planes),
            nn.ReLU(inplace=True),
            nn.Linear(mid_planes//share_planes, out_planes//share_planes, bias=True),
            Rearrange('(n k) c -> n k c', k=nsample))

        self.channelMixMLPs03 = nn.Linear(in_planes, out_planes)
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, pxo) -> torch.Tensor:
        p, x, o = pxo  # (n, 3), (n, c), (b)

        x_knn, knn_idx = pointops.queryandgroup(
            self.nsample, p, p, x, None, o, o, use_xyz=True, return_idx=True)  # (n, k, 3+c)
        p_r = x_knn[:, :, 0:3]
        
        x_knn_flatten = rearrange(x_knn, 'n k c -> (n k) c')
        energy_flatten = self.channelMixMLPs01(x_knn_flatten) # (n*k, k)
        
        n = p_r.shape[0]; 
        p_embed = self.linear_p(p_r.view(-1, 3)) # (n*k, out_planes)
        p_embed = p_embed.view(n, self.nsample, -1)
        p_embed_shrink_flatten = self.shrink_p(p_embed) # (n*k, k)

        energy_flatten = torch.cat([energy_flatten, p_embed_shrink_flatten], dim=-1) # (n*k, 2k)
        energy = self.channelMixMLPs02(energy_flatten) # (n, k, out_planes/share_planes)
        w = self.softmax(energy)

        x_v = self.channelMixMLPs03(x)  # (n, in_planes) -> (n, k)
        n = knn_idx.shape[0]; knn_idx_flatten = knn_idx.flatten()
        x_v  = x_v[knn_idx_flatten, :].view(n, self.nsample, -1)

        n, nsample, out_planes = x_v.shape
        x_knn = (x_v + p_embed).view(n, nsample, self.share_planes, out_planes//self.share_planes)
        x_knn = (x_knn * w.unsqueeze(2))
        x_knn = x_knn.reshape(n, nsample, out_planes)

        x = x_knn.sum(1)
        return (x, x_knn, knn_idx, p_r)


class PointMixerInterSetLayerGroupMLPv3(nn.Module):
    def __init__(self, in_planes, share_planes, nsample=16, use_xyz=False):
        super().__init__()
        self.share_planes = share_planes
        self.linear = nn.Linear(in_planes, in_planes//share_planes) # input.shape = [N*K, C] 
        self.linear_x = nn.Linear(in_planes, in_planes//share_planes) # input.shape = [N*K, C]
        self.linear_p = nn.Sequential( # input.shape = [N*K, C]
            nn.Linear(3, 3, bias=False),
            nn.BatchNorm1d(3),
            nn.ReLU(inplace=True), 
            nn.Linear(3, in_planes))

    def forward(self, input):
        x, x_knn, knn_idx, p_r = input
        N = x_knn.shape[0]

        with torch.no_grad():
            knn_idx_flatten = rearrange(knn_idx, 'n k -> (n k) 1')
        p_r_flatten = rearrange(p_r, 'n k c -> (n k) c')
        p_embed_flatten = self.linear_p(p_r_flatten)
        x_knn_flatten = rearrange(x_knn, 'n k c -> (n k) c')
        x_knn_flatten_shrink = self.linear(x_knn_flatten + p_embed_flatten) # nk c'

        x_knn_prob_flatten_shrink = \
            scatter_softmax(x_knn_flatten_shrink, knn_idx_flatten, dim=0) # (n*nsample, c')
        x_v_knn_flatten = self.linear_x(x_knn_flatten) # (n*nsample, c')
        x_knn_weighted_flatten = x_v_knn_flatten * x_knn_prob_flatten_shrink # (n*nsample, c')

        residual = scatter_sum(x_knn_weighted_flatten, knn_idx_flatten, dim=0, dim_size=N) # (n, c')
        residual = repeat(residual, 'n c -> n (repeat c)', repeat=self.share_planes)
        return x + residual

###########################################################################

class PointMixerBlock(nn.Module):
    def __init__(self, in_planes, planes, share_planes=8, nsample=16, use_xyz=False):
        super().__init__()
        self.expansion = 1
        # assert self.intraLayer is not None
        # assert self.interLayer is not None
        self.transformer2 = nn.Sequential(
            PointMixerIntraSetLayerPaper(planes, planes, share_planes, nsample),
            PointMixerInterSetLayerGroupMLPv3(in_planes, nsample, share_planes)
        )
        self.linear1 = nn.Linear(in_planes, planes, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.linear3 = nn.Linear(planes, planes*self.expansion, bias=False)
        self.bn3 = nn.BatchNorm1d(planes*self.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        p, x, o = pxo  # (n, 3), (n, c), (b)
        identity = x
        x = self.relu(self.bn1(self.linear1(x)))
        x = self.relu(self.bn2(self.transformer2([p, x, o])))
        x = self.bn3(self.linear3(x))
        x = x + identity
        x = self.relu(x)
        return [p, x, o]


class PointMixerBlockPaperInterSetLayerGroupMLPv3(PointMixerBlock):
    expansion = 1
    intraLayer = PointMixerIntraSetLayerPaper
    interLayer = PointMixerInterSetLayerGroupMLPv3

##############################################################################################

class SymmetricTransitionUpBlock(nn.Module):
    def __init__(self, in_planes, in_planes2, out_planes=None, nsample=16):
        super().__init__()
        self.nsample = nsample
        if out_planes is None:
            self.linear1 = nn.Sequential(
                nn.Linear(2*in_planes, in_planes), 
                nn.BatchNorm1d(in_planes), 
                nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(
                nn.Linear(in_planes2, in_planes), 
                nn.ReLU(inplace=True))            
        else:
            self.linear1 = nn.Sequential( # input.shape = [N, L]
                nn.Linear(in_planes, out_planes), 
                nn.BatchNorm1d(out_planes),  
                nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential( # input.shape = [N, L]
                nn.Linear(in_planes2, out_planes), 
                nn.BatchNorm1d(out_planes), 
                nn.ReLU(inplace=True))
            self.channel_shrinker = nn.Sequential( # input.shape = [N*K, L]
                nn.Linear(in_planes2+3, in_planes),
                Rearrange('n k c -> n c k'),
                nn.BatchNorm1d(in_planes),
                Rearrange('n c k -> n k c'),
                # nn.BatchNorm1d(in_planes),
                nn.ReLU(inplace=True),
                nn.Linear(in_planes, 1))
        
    def forward(self, pxo1, pxo2=None):
        if pxo2 is None:
            _, x, o = pxo1  # (n, 3), (n, c), (b)
            x_tmp = []
            for i in range(o.shape[0]):
                if i == 0:
                    s_i, e_i, cnt = 0, o[0], o[0]
                else:
                    s_i, e_i, cnt = o[i-1], o[i], o[i] - o[i-1]
                x_b = x[s_i:e_i, :]
                x_b = torch.cat((x_b, self.linear2(x_b.sum(0, True) / cnt).repeat(cnt, 1)), 1)
                x_tmp.append(x_b)
            x = torch.cat(x_tmp, 0)
            y = self.linear1(x) # this part is the same as TransitionUp module.
        else:
            p1, x1, o1 = pxo1; p2, x2, o2 = pxo2 
            # device = p1.device
            # p2 = p2.to(device)
            # x2 = x2.to(device)
            # o2 = o2.to(device)
            # print(device)
            knn_idx = pointops.knnquery(self.nsample, p1, p2, o1, o2)[0].long()

            with torch.no_grad():
                knn_idx_flatten = rearrange(knn_idx, 'm k -> (m k)')
            p_r = p1[knn_idx_flatten, :].view(len(p2), self.nsample, 3) - p2.unsqueeze(1)
            x2_knn = x2.view(len(p2), 1, -1).repeat(1, self.nsample, 1)
            x2_knn = torch.cat([p_r, x2_knn], dim=-1) # (109, 16, 259) # (m, nsample, 3+c)

            with torch.no_grad():
                knn_idx_flatten = knn_idx_flatten.unsqueeze(-1) # (m*nsample, 1)
            
            # print(x2_knn_flatten.device)
            x2_knn_shrink = self.channel_shrinker(x2_knn) # (m * nsample, 1)
            x2_knn_flatten_shrink = rearrange(x2_knn_shrink, 'm k c -> (m k) c') # c = 3+out_planes
            x2_knn_prob_flatten_shrink = scatter_softmax(x2_knn_flatten_shrink, knn_idx_flatten, dim=0)

            x2_knn_prob_shrink = rearrange(x2_knn_prob_flatten_shrink, '(m k) 1 -> m k 1', k=self.nsample)
            up_x2_weighted = self.linear2(x2).unsqueeze(1) * x2_knn_prob_shrink # (m, nsample, c)
            up_x2_weighted_flatten = rearrange(up_x2_weighted, 'm k c -> (m k) c')
            up_x2 = scatter_sum(up_x2_weighted_flatten, knn_idx_flatten, dim=0, dim_size=len(p1))
            y = self.linear1(x1) + up_x2
        return y

##############################################################################################

class SymmetricTransitionDownBlockPaperv3(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, nsample=16):
        super().__init__()
        self.stride, self.nsample = stride, nsample
        
        if stride != 1:
            self.linear2 = nn.Sequential( # input.shape = [N, L]
                nn.Linear(in_planes, out_planes, bias=False),
                nn.BatchNorm1d(out_planes),
                nn.ReLU(inplace=True))
            self.channel_shrinker = nn.Sequential( # input.shape = [N*K, L]
                nn.Linear(3+in_planes, in_planes, bias=False),
                nn.BatchNorm1d(in_planes),
                nn.ReLU(inplace=True),
                nn.Linear(in_planes, 1))

        else:
            self.linear2 = nn.Sequential(
                nn.Linear(in_planes, out_planes, bias=False),
                nn.BatchNorm1d(out_planes),
                nn.ReLU(inplace=True))
        
    def forward(self, pxo):
        p, x, o = pxo  # (n, 3), (n, c), (b)

        if self.stride != 1:
            n_o, count = [o[0].item() // self.stride], o[0].item() // self.stride
            for i in range(1, o.shape[0]):
                count += (o[i].item() - o[i-1].item()) // self.stride
                n_o.append(count)
            n_o = torch.cuda.IntTensor(n_o)
            idx = pointops.furthestsampling(p, o, n_o)  # (m)
            n_p = p[idx.long(), :]  # (m, 3)

            x_knn, knn_idx = pointops.queryandgroup(
                self.nsample, p, n_p, x, None, o, n_o, use_xyz=True, return_idx=True)  # (m, nsample, 3+c)

            m, k, c = x_knn.shape
            x_knn_flatten = rearrange(x_knn, 'm k c -> (m k) c')
            x_knn_flatten_shrink = self.channel_shrinker(x_knn_flatten) # (m*nsample, 1)
            x_knn_shrink = rearrange(x_knn_flatten_shrink, '(m k) c -> m k c', m=m, k=k)
            x_knn_prob_shrink = F.softmax(x_knn_shrink, dim=1)

            y = self.linear2(x) # (n, c)
            with torch.no_grad():
                knn_idx_flatten = rearrange(knn_idx, 'm k -> (m k)')
            y_knn_flatten = y[knn_idx_flatten, :] # (m*nsample, c)
            y_knn = rearrange(y_knn_flatten, '(m k) c -> m k c', m=m, k=k)
            x_knn_weighted = y_knn * x_knn_prob_shrink # (m, nsample, c_out)
            y = torch.sum(x_knn_weighted, dim=1).contiguous() # (m, c_out)
            
            p, o = n_p, n_o

        else:
            idx = pointops.furthestsampling(p, o, o)  # (m)
            y = self.linear2(x)  # (n, c)

        return [p, y, o, idx]


'''
基于估计的场景流信息在目标域搜索对应的K个邻域信息；先根据参考点的邻域计算一个平均的相关参数，作为稀疏阈值；然后将邻域点按照不同的分辨率Voxelize，构造出不同分辨率级别的VFE特征，依次利用关联操作以及稀疏阈值选择操作，结合scatter操作concat到一起，通过进行加权或者cnn+maxpooling/GRU对应的权重信息；组合不同级别的特信息；分别输出对应的权重和用于估计遮挡信息的掩膜特征。
Method 1: 参考SAC方法构建motion featrue 的选择方法
Method 1: 参考PointMixer方法构建Cost Volume
'''
# class CostVolumeNet(nn.Module):
#     def __init__(self, nsample, in_channel, mid_channel, share_channel, out_channel, use_mask=False,
#         intraLayer='PointMixerIntraSetLayer',
#         interLayer='PointMixerInterSetLayer',
#         transup='SymmetricTransitionUpBlock', 
#         transdown='TransitionDownBlock'):
#         super().__init__()
#         self.nsample = nsample
#         self.mid_channel = mid_channel
#         self.share_channel = share_channel
#         self.out_channel = out_channel
#         if use_mask:
#             self.channelMixMLPs01 = nn.Sequential( # input.shape = [N, K, C]
#                     nn.Linear(1+3+in_channel*2, nsample),
#                     nn.ReLU(inplace=True),
#                     BilinearFeedForward(nsample, nsample, nsample))
#         else:
#             self.channelMixMLPs01 = nn.Sequential( # input.shape = [N, K, C]
#                     nn.Linear(3+in_channel*2, nsample),
#                     nn.ReLU(inplace=True),
#                     BilinearFeedForward(nsample, nsample, nsample))
        
#         self.channelMixMLPs01_2 = nn.Sequential( # input.shape = [N, K, C]
#                 nn.Linear(3+out_channel*2, nsample),
#                 nn.ReLU(inplace=True),
#                 BilinearFeedForward(nsample, nsample, nsample))
        
#         self.linear_p = nn.Sequential( # input.shape = [N, K, C]
#             nn.Linear(3, out_channel//2),
#             nn.Sequential(
#                 Rearrange('n k c -> n c k'),
#                 nn.BatchNorm1d(out_channel//2),
#                 Rearrange('n c k -> n k c')),
#             nn.ReLU(inplace=True), 
#             nn.Linear(out_channel//2, out_channel))
        
#         self.shrink_p = nn.Sequential(
#             Rearrange('n k (a b) -> n k a b', b=nsample),
#             Reduce('n k a b -> n k b', 'sum', b=nsample))
        
#         self.channelMixMLPs02 = nn.Sequential( # input.shape = [N, K, C]
#             Rearrange('n k c -> n c k'),
#             nn.Conv1d(nsample+nsample, mid_channel, kernel_size=1, bias=False),
#             nn.BatchNorm1d(mid_channel), 
#             nn.ReLU(inplace=True),
#             nn.Conv1d(mid_channel, mid_channel//share_channel, kernel_size=1, bias=False),
#             nn.BatchNorm1d(mid_channel//share_channel),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(mid_channel//share_channel, out_channel//share_channel, kernel_size=1),
#             Rearrange('n c k -> n k c'))
#         self.channelMixMLPs02_2 = nn.Sequential( # input.shape = [N, K, C]
#             Rearrange('n k c -> n c k'),
#             nn.Conv1d(nsample+nsample, mid_channel, kernel_size=1, bias=False),
#             nn.BatchNorm1d(mid_channel), 
#             nn.ReLU(inplace=True),
#             nn.Conv1d(mid_channel, mid_channel//share_channel, kernel_size=1, bias=False),
#             nn.BatchNorm1d(mid_channel//share_channel),
#             nn.ReLU(inplace=True),
#             nn.Conv1d(mid_channel//share_channel, out_channel//share_channel, kernel_size=1),
#             Rearrange('n c k -> n k c'))
#         # self.channelMixMLPs02_2 = nn.Linear(out_channel//share_channel, 1)
#         if use_mask:
#             self.channelMixMLPs03 = nn.Linear(1+3+in_channel*2, out_channel)
#         else:
#             self.channelMixMLPs03 = nn.Linear(3+in_channel*2, out_channel)
#         self.channelMixMLPs03_2 = nn.Linear(3+out_channel*2, out_channel)
        
#         self.softmax = nn.Softmax(dim=1)
    
#     def get_patch_features(self, pc1, pc2, channelMixMLPs01, channelMixMLPs02, channelMixMLPs03,  sf=None, use_cross_att=True):
#         '''
#         Input:
#         (dense) pc1: [pos:[n, 3]; feats:[n, c]; cnt_num:[b*n]]
#         (sparse) pc2: [pos:[m, 3]; feats:[m, c]; cnt_num:[b2*n]]
#         s_sf: [n,3], cnt_num: [b]
#         '''
#         p1, feats1, cnt_num1 = pc1
#         p2, feats2, cnt_num2 = pc2 
#         # B,N = pos1.shape
#         # b, m_p, _ = pos2.size()
#         if use_cross_att and sf != None:
#             pos_feats_knn, knn_idx = pointops.queryandgroup(
#                 self.nsample, p2, p1+sf, feats2, None, cnt_num2, cnt_num1, use_xyz=True, return_idx=True)  # (n, k, 3+c)
#         else:
#             pos_feats_knn, knn_idx = pointops.queryandgroup(
#                 self.nsample, p2, p1, feats2, None, cnt_num2, cnt_num1, use_xyz=True, return_idx=True)  # (n, k, 3+c)
#         pos_diff = pos_feats_knn[:, :, 0:3]
#         feats1_groupped = feats1.view(len(p1), 1, -1).repeat(1, self.nsample, 1)
        
#         new_feats = torch.cat([pos_feats_knn, feats1_groupped], dim=-1)
#         energy = channelMixMLPs01(new_feats) # (n, k, k)
        
#         p_embed = self.linear_p(pos_diff) # (n, k, out_planes)
#         p_embed_shrink = self.shrink_p(p_embed) # (n, k, k)

#         energy = torch.cat([energy, p_embed_shrink], dim=-1)
#         energy = channelMixMLPs02(energy) # (n, k, out_planes/share_planes)
#         w = self.softmax(energy)
#         if use_cross_att: # and sf != None
#             new_feats_v = channelMixMLPs03(new_feats)  # (n, in_planes) -> (n, k)
#             n = knn_idx.shape[0]
#             # knn_idx_flatten = knn_idx.flatten()
#             # new_feats_v  = new_feats_v[knn_idx_flatten, :].view(n, self.nsample, -1)
#             n, nsample, out_planes = new_feats_v.shape
#             new_feats = (new_feats_v + p_embed).view(n, self.nsample, self.share_channel, out_planes//self.share_channel)
#             # new_feats = new_feats_v.view(n, self.nsample, self.share_channel, out_planes//self.share_channel)
#             new_feats = (new_feats * w.unsqueeze(2)).squeeze(2)
            
#             # if mask != None:
#             #     mask_groupped = mask[knn_idx.flatten(), :].view(n, self.nsample, -1).repeat(1,1,self.out_channel)
#             #     new_feats = mask_groupped * new_feats
#             new_feats = new_feats.reshape(n, nsample, out_planes)   
            
#         if use_cross_att:
#             new_feats = new_feats.sum(1)
#             return new_feats, pos_feats_knn, w
#         else:
#             # return knn_idx, new_feats, pos_feats_knn, w
#             return knn_idx, pos_feats_knn, w
            

#     def forward(self, pc1, pc2, sf=None, mask=None):
#         '''
#         Input:
#         (dense) pc1: [pos:[n, 3]; feats:[n, c]; cnt_num:[b*n]]
#         (sparse) pc2: [pos:[m, 3]; feats:[m, c]; cnt_num:[b2*n]]
#         s_sf: [n,3], cnt_num: [b]
#         '''
#         p1, feats1, cnt_num1 = pc1
#         p2, feats2, cnt_num2 = pc2 
#         B = None
#         if len(p1.shape) == 3:
#             B = p1.shape[0]
#             p1 = p1.permute(0,2,1).contiguous().view(-1, p1.shape[1])
#             feats1 = feats1.permute(0,2,1).contiguous().view(-1, feats1.shape[1])
#             if mask != None:
#                 feats1 = torch.cat([feats1, mask], dim=-1)
#             pc1 = [p1, feats1, cnt_num1]
        
#         if len(p2.shape) == 3:
#             p2 = p2.permute(0,2,1).contiguous().view(-1, p2.shape[1])
#             feats2 = feats2.permute(0,2,1).contiguous().view(-1, feats2.shape[1])
#             pc2 = [p2, feats2, cnt_num2]
            
#         # B,N = pos1.shape
#         # b, m_p, _ = pos2.size()
        
#         if sf == None:
#             sf = torch.zeros_like(p1)
#             # sf_feats = None
#         else:
#             sf = sf.permute(0,2,1).contiguous().view(-1, sf.shape[1])
            
#         N = p1.shape[0]
#         inter_feats_fwd, _, _ = self.get_patch_features(pc1, pc2, channelMixMLPs01=self.channelMixMLPs01, channelMixMLPs02=self.channelMixMLPs02, channelMixMLPs03=self.channelMixMLPs03, sf=sf, use_cross_att=True)
#         if sf!=None:
#             pc1w = [p1+sf, feats1, cnt_num1]
#         else:
#             pc1w = pc1
#         inter_feats_bwd, _, _ = self.get_patch_features(pc2, pc1w, channelMixMLPs01=self.channelMixMLPs01, channelMixMLPs02=self.channelMixMLPs02, channelMixMLPs03=self.channelMixMLPs03, use_cross_att=True)
#         pc1 = [p1, inter_feats_fwd, cnt_num1]
#         pc2 = [p2, inter_feats_bwd, cnt_num2]
#         # intra_knn_idx, intra_groupped_feats, w = self.get_patch_features(pc1, pc1, use_cross_att=False)
#         patch_to_patch_cost, _, _ = self.get_patch_features(pc1, pc2, channelMixMLPs01=self.channelMixMLPs01_2, channelMixMLPs02=self.channelMixMLPs02_2, channelMixMLPs03=self.channelMixMLPs03_2, sf=sf, use_cross_att=True)
#         # pc1 = [p1, feats1, cnt_num1]
#         # if mask != None:
#         #     pc1_2 = [p1, feats1[:,:-1], cnt_num1]
#         # else:
#         #     pc1_2 = pc1
#         # intra_knn_idx, _, w = self.get_patch_features(pc1_2, pc1, channelMixMLPs01=self.channelMixMLPs01, channelMixMLPs02=self.channelMixMLPs02, channelMixMLPs03=self.channelMixMLPs03, use_cross_att=False)

#         # grouped_point_to_patch_cost = inter_feats[intra_knn_idx.flatten(), :].view(N, self.nsample, -1)
#         # patch_to_patch_cost = (w * grouped_point_to_patch_cost).sum(1)

#         patch_to_patch_cost = patch_to_patch_cost.view(B,-1, patch_to_patch_cost.shape[-1]).permute(0,2,1).contiguous()
#         inter_feats_fwd = inter_feats_fwd.view(B,-1, inter_feats_fwd.shape[-1]).permute(0,2,1).contiguous()
#         inter_feats_bwd = inter_feats_bwd.view(B,-1, inter_feats_bwd.shape[-1]).permute(0,2,1).contiguous()
#         return patch_to_patch_cost, inter_feats_fwd, inter_feats_bwd
        
class CostVolumeNet(nn.Module):
    def __init__(self, nsample, in_channel, mid_channel, share_channel, out_channel, use_mask=False,
        intraLayer='PointMixerIntraSetLayer',
        interLayer='PointMixerInterSetLayer',
        transup='SymmetricTransitionUpBlock', 
        transdown='TransitionDownBlock'):
        super().__init__()
        self.nsample = nsample
        self.mid_channel = mid_channel
        self.share_channel = share_channel
        self.out_channel = out_channel

        self.channelMixMLPs01 = nn.Sequential( # input.shape = [N, K, C]
                nn.Linear(3+in_channel*2, nsample),
                nn.ReLU(inplace=True),
                BilinearFeedForward(nsample, nsample, nsample))
        
        self.channelMixMLPs01_2 = nn.Sequential( # input.shape = [N, K, C]
                nn.Linear(3+in_channel*2+out_channel, nsample),
                nn.ReLU(inplace=True),
                BilinearFeedForward(nsample, nsample, nsample))
        
        self.linear_p = nn.Sequential( # input.shape = [N, K, C]
            nn.Linear(3, out_channel//2),
            nn.Sequential(
                Rearrange('n k c -> n c k'),
                nn.BatchNorm1d(out_channel//2),
                Rearrange('n c k -> n k c')),
            nn.ReLU(inplace=True), 
            nn.Linear(out_channel//2, out_channel))
        
        self.shrink_p = nn.Sequential(
            Rearrange('n k (a b) -> n k a b', b=nsample),
            Reduce('n k a b -> n k b', 'sum', b=nsample))
        
        self.channelMixMLPs02 = nn.Sequential( # input.shape = [N, K, C]
            Rearrange('n k c -> n c k'),
            nn.Conv1d(nsample+nsample, mid_channel//share_channel, kernel_size=1, bias=False),
            nn.BatchNorm1d(mid_channel), 
            nn.ReLU(inplace=True),
            # nn.Conv1d(mid_channel, mid_channel//share_channel, kernel_size=1, bias=False),
            # nn.BatchNorm1d(mid_channel//share_channel),
            # nn.ReLU(inplace=True),
            nn.Conv1d(mid_channel//share_channel, out_channel//share_channel, kernel_size=1),
            Rearrange('n c k -> n k c'))
        self.channelMixMLPs02_2 = nn.Sequential( # input.shape = [N, K, C]
            Rearrange('n k c -> n c k'),
            nn.Conv1d(nsample+nsample, mid_channel//share_channel, kernel_size=1, bias=False),
            nn.BatchNorm1d(mid_channel), 
            nn.ReLU(inplace=True),
            # nn.Conv1d(mid_channel, mid_channel//share_channel, kernel_size=1, bias=False),
            # nn.BatchNorm1d(mid_channel//share_channel),
            # nn.ReLU(inplace=True),
            nn.Conv1d(mid_channel//share_channel, out_channel//share_channel, kernel_size=1),
            Rearrange('n c k -> n k c'))
    
        
        self.channelMixMLPs03 = nn.Linear(3+in_channel*2, out_channel)
        self.channelMixMLPs03_2 = nn.Linear(3+in_channel*2+out_channel, out_channel)
        
        self.softmax = nn.Softmax(dim=1)
        
    def get_patch_features(self, pc1, pc2, channelMixMLPs01, channelMixMLPs02, channelMixMLPs03, sf=None, mask=None, use_cross_att=True):
        '''
        Input:
        (dense) pc1: [pos:[n, 3]; feats:[n, c]; cnt_num:[b*n]]
        (sparse) pc2: [pos:[m, 3]; feats:[m, c]; cnt_num:[b2*n]]
        s_sf: [n,3], cnt_num: [b]
        '''
        p1, feats1, cnt_num1 = pc1
        p2, feats2, cnt_num2 = pc2 
        # B,N = pos1.shape
        # b, m_p, _ = pos2.size()
        if use_cross_att and sf != None:
            pos_feats_knn, knn_idx = pointops.queryandgroup(
                self.nsample, p2, p1+sf, feats2, None, cnt_num2, cnt_num1, use_xyz=True, return_idx=True)  # (n, k, 3+c)
        else:
            pos_feats_knn, knn_idx = pointops.queryandgroup(
                self.nsample, p2, p1, feats2, None, cnt_num2, cnt_num1, use_xyz=True, return_idx=True)  # (n, k, 3+c)
        pos_diff = pos_feats_knn[:, :, 0:3]
        feats1_groupped = feats1.view(len(p1), 1, -1).repeat(1, self.nsample, 1)
        
        new_feats = torch.cat([pos_feats_knn, feats1_groupped], dim=-1)
        energy = channelMixMLPs01(new_feats) # (n, k, k)
        
        p_embed = self.linear_p(pos_diff) # (n, k, out_planes)
        p_embed_shrink = self.shrink_p(p_embed) # (n, k, k)

        energy = torch.cat([energy, p_embed_shrink], dim=-1)
        energy = channelMixMLPs02(energy) # (n, k, out_planes/share_planes)
        w = self.softmax(energy)
        # if use_cross_att: # and sf != None
        new_feats_v = channelMixMLPs03(new_feats)  # (n, in_planes) -> (n, k)
        # n = knn_idx.shape[0]
        n, nsample, out_planes = new_feats_v.shape
        new_feats_fwd = (new_feats_v + p_embed).view(n, self.nsample, self.share_channel, out_planes//self.share_channel)
        new_feats_fwd = (new_feats_fwd * w.unsqueeze(2)).squeeze(2).sum(1)
        
        with torch.no_grad():
            knn_idx_flatten = rearrange(knn_idx, 'm k -> (m k)')
        energy_bwd_flatten_shrink = rearrange(energy, 'm k c -> (m k) c') # c = 3+out_planes
        energy_bwd_prob_flatten_shrink = scatter_softmax(energy_bwd_flatten_shrink, knn_idx_flatten, dim=0)

        new_feats_v_bwd_flatten_shrink = rearrange(new_feats_v + p_embed, 'm k c -> (m k) c') # c = 3+out_planes
        new_feats_weighted_flatten = new_feats_v_bwd_flatten_shrink * energy_bwd_prob_flatten_shrink 
        if mask != None:
            # backward cost
            mask_groupped = mask[knn_idx_flatten, :].view(n, self.nsample, -1).repeat(1,1,self.out_channel)
            new_feats_bwd = mask_groupped * new_feats_bwd
            
        new_feats_bwd = scatter_sum(new_feats_weighted_flatten, knn_idx_flatten, dim=0, dim_size=len(p1))

        new_feats_bwd_groupped = new_feats_bwd[knn_idx_flatten, :].view(n, nsample, out_planes)


        cost_feats = torch.cat([new_feats, new_feats_bwd_groupped], dim=-1)
        cost_energy = self.channelMixMLPs01_2(cost_feats) # (n, k, k)
        cost_energy = torch.cat([cost_energy, p_embed_shrink], dim=-1)
        cost_energy = self.channelMixMLPs02_2(cost_energy) # (n, k, out_planes/share_planes)
        cost_w = self.softmax(energy)
        
        cost_feats_v = self.channelMixMLPs03_2(cost_feats)  # (n, in_planes) -> (n, k)
        cost_feats = (cost_feats_v + p_embed).view(n, self.nsample, self.share_channel, out_planes//self.share_channel)
        cost_feats = (cost_feats * cost_w.unsqueeze(2)).squeeze(2).sum(1) + new_feats_fwd

        return cost_feats, new_feats_fwd, new_feats_bwd, w
     
            

    def forward(self, pc1, pc2, sf=None, mask=None):
        '''
        Input:
        (dense) pc1: [pos:[n, 3]; feats:[n, c]; cnt_num:[b*n]]
        (sparse) pc2: [pos:[m, 3]; feats:[m, c]; cnt_num:[b2*n]]
        s_sf: [n,3], cnt_num: [b]
        '''
        p1, feats1, cnt_num1 = pc1
        p2, feats2, cnt_num2 = pc2 
        B = None
        if len(p1.shape) == 3:
            B = p1.shape[0]
            p1 = p1.permute(0,2,1).contiguous().view(-1, p1.shape[1])
            feats1 = feats1.permute(0,2,1).contiguous().view(-1, feats1.shape[1])
            pc1 = [p1, feats1, cnt_num1]
        if len(p2.shape) == 3:
            p2 = p2.permute(0,2,1).contiguous().view(-1, p2.shape[1])
            feats2 = feats2.permute(0,2,1).contiguous().view(-1, feats2.shape[1])
            pc2 = [p2, feats2, cnt_num2]
            
        if sf == None:
            sf = torch.zeros_like(p1)
            # sf_feats = None
        else:
            sf = sf.permute(0,2,1).contiguous().view(-1, sf.shape[1])
            
        patch_to_patch_cost, inter_feats_fwd, inter_feats_bwd, _ = self.get_patch_features(pc1, pc2, channelMixMLPs01=self.channelMixMLPs01, channelMixMLPs02=self.channelMixMLPs02, channelMixMLPs03=self.channelMixMLPs03,sf=sf, use_cross_att=True)
        # inter_feats_bwd, inter_feats_fwd_res, _, _ = self.get_patch_features(pc2, pc1, channelMixMLPs01=self.channelMixMLPs01, channelMixMLPs02=self.channelMixMLPs02, channelMixMLPs03=self.channelMixMLPs03, use_cross_att=True)
        # inter_feats_fwd = self.linear_p2(inter_feats_fwd + inter_feats_fwd_res)
        # inter_feats_bwd = self.linear_p2(inter_feats_bwd + inter_feats_bwd_res)
        # pc1 = [p1, inter_feats_fwd, cnt_num1]
        # pc2 = [p2, inter_feats_bwd, cnt_num2]
        # patch_to_patch_cost, _, _, _ = self.get_patch_features(pc1, pc2, channelMixMLPs01=self.channelMixMLPs01_2, channelMixMLPs02=self.channelMixMLPs02_2, channelMixMLPs03=self.channelMixMLPs03_2, sf=sf, use_cross_att=True)
        
        patch_to_patch_cost = patch_to_patch_cost.view(B,-1, patch_to_patch_cost.shape[-1]).permute(0,2,1).contiguous()
        inter_feats_fwd = inter_feats_fwd.view(B,-1, inter_feats_fwd.shape[-1]).permute(0,2,1).contiguous()
        inter_feats_bwd = inter_feats_bwd.view(B,-1, inter_feats_bwd.shape[-1]).permute(0,2,1).contiguous()
        return patch_to_patch_cost, inter_feats_fwd, inter_feats_bwd
      

class SceneFlowRegressor(nn.Module):
    def __init__(self, nsample, in_channel,  sfeat_channel,  sf_channel, mid_channel, share_channel, out_channel, channels=[128,128], mlp=[128, 128], use_mask=False, use_leaky=True,
        intraLayer='PointMixerIntraSetLayer',
        interLayer='PointMixerInterSetLayer',
        transup='SymmetricTransitionUpBlock', 
        transdown='TransitionDownBlock'):
        super().__init__()
        self.nsample = nsample
        self.mid_channel = mid_channel
        self.share_channel = share_channel
        self.out_channel = out_channel
        self.sf_channel = sf_channel
   
        self.channelMixMLPs01 = nn.Sequential( # input.shape = [N, K, C]
                nn.Linear(in_channel*2+3, nsample),
                nn.ReLU(inplace=True),
                BilinearFeedForward(nsample, nsample, nsample))
        
        self.channelMixMLPs01_2 = nn.Sequential( # input.shape = [N, K, C]
                nn.Linear(out_channel*2+3, nsample),
                nn.ReLU(inplace=True),
                BilinearFeedForward(nsample, nsample, nsample))
        
        self.linear_p = nn.Sequential( # input.shape = [N, K, C]
            nn.Linear(3, out_channel//2, bias=False),
            nn.Sequential(
                Rearrange('n k c -> n c k'),
                nn.BatchNorm1d(out_channel//2),
                Rearrange('n c k -> n k c')),
            nn.ReLU(inplace=True), 
            nn.Linear(out_channel//2, out_channel))
        
        self.shrink_p = nn.Sequential(
            Rearrange('n k (a b) -> n k a b', b=nsample),
            Reduce('n k a b -> n k b', 'sum', b=nsample))
        
        self.channelMixMLPs02 = nn.Sequential( # input.shape = [N, K, C]
            Rearrange('n k c -> n c k'),
            nn.Conv1d(nsample+nsample, mid_channel, kernel_size=1, bias=False),
            nn.BatchNorm1d(mid_channel), 
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channel, mid_channel//share_channel, kernel_size=1, bias=True),
            # nn.BatchNorm1d(mid_channel//share_channel),
            # nn.ReLU(inplace=True),
            # nn.Conv1d(mid_channel//share_channel, out_channel//share_channel, kernel_size=1),
            Rearrange('n c k -> n k c'))
        self.channelMixMLPs02_2 = nn.Sequential( # input.shape = [N, K, C]
            Rearrange('n k c -> n c k'),
            nn.Conv1d(nsample+nsample, mid_channel, kernel_size=1, bias=False),
            nn.BatchNorm1d(mid_channel), 
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channel, mid_channel//share_channel, kernel_size=1, bias=True),
            # nn.BatchNorm1d(mid_channel//share_channel),
            # nn.ReLU(inplace=True),
            # nn.Conv1d(mid_channel//share_channel, out_channel//share_channel, kernel_size=1),
            Rearrange('n c k -> n k c'))
        
        self.channelMixMLPs03 = nn.Linear(in_channel*2+3, out_channel)
        self.channelMixMLPs03_2 = nn.Linear(out_channel*2+3, out_channel)
        
        self.channelMixMLPs04 = nn.Linear(out_channel, out_channel)
        self.channelMixMLPs04_2 = nn.Linear(out_channel, out_channel)
        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)
        
        self.softmax = nn.Softmax(dim=1)

        self.sfnet = SceneFlowEstimatorPointConv2(feat_ch=sfeat_channel, cost_ch=out_channel, flow_ch=sf_channel, channels =channels, mlp=mlp, share_planes=8, neighbors=8, clamp=[-200, 200], use_leaky = True)
        self.sfnet2 = SceneFlowEstimatorPointConv2(feat_ch=sfeat_channel, cost_ch=out_channel, flow_ch=3, channels =channels, mlp=mlp, share_planes=8, neighbors=8, clamp=[-200, 200], use_leaky = True)
        # self.sfnet = SceneFlowEstimatorResidual(feat_ch=sfeat_channel, cost_ch=out_channel, flow_ch=sf_channel, channels =channels, mlp=mlp, neighbors=9, clamp=[-200, 200], use_leaky = True)
        # self.sfnet2 = SceneFlowEstimatorResidual(feat_ch=sfeat_channel, cost_ch=out_channel, flow_ch=3, channels =channels, mlp=mlp, neighbors=9, clamp=[-200, 200], use_leaky = True)
    
    def get_patch_features(self, pc1, pc2, channelMixMLPs01, channelMixMLPs02, channelMixMLPs03, channelMixMLPs04, sf=None, mask=None, use_cross_att=True, use_bwd=False):
        '''
        Input:
        (dense) pc1: [pos:[n, 3]; feats:[n, c]; cnt_num:[b*n]]
        (sparse) pc2: [pos:[m, 3]; feats:[m, c]; cnt_num:[b2*n]]
        s_sf: [n,3], cnt_num: [b]
        '''
        p1, feats1, cnt_num1 = pc1
        p2, feats2, cnt_num2 = pc2 
        # B,N = pos1.shape
        # b, m_p, _ = pos2.size()
        if use_cross_att and sf != None:
            pos_feats_knn, knn_idx = pointops.queryandgroup(
                self.nsample, p2, p1+sf, feats2, None, cnt_num2, cnt_num1, use_xyz=True, return_idx=True)  # (n, k, 3+c)
        else:
            pos_feats_knn, knn_idx = pointops.queryandgroup(
                self.nsample, p2, p1, feats2, None, cnt_num2, cnt_num1, use_xyz=True, return_idx=True)  # (n, k, 3+c)
            
        pos_diff = p2[knn_idx.view(-1), :].view(p1.shape[0], self.nsample, 3) - p1.unsqueeze(1) #pos_feats_knn[:, :, 0:3]
        pos_feats_knn[:, :, 0:3] = pos_diff
        feats1_groupped = feats1.view(len(p1), 1, -1).repeat(1, self.nsample, 1)
        new_feats = torch.cat([pos_feats_knn, feats1_groupped], dim=-1)
        
        p_embed = self.linear_p(pos_diff) # (n, k, out_planes)
        p_embed_shrink = self.shrink_p(p_embed) # (n, k, k)
        
        energy = channelMixMLPs01(new_feats) # (n, k, k)
        
        energy = torch.cat([energy, p_embed_shrink], dim=-1)
        energy = channelMixMLPs02(energy) # (n, k, out_planes/share_planes)
        w = self.softmax(energy)
        if use_cross_att: # and sf != None
            new_feats_v = channelMixMLPs03(new_feats)  # (n, in_planes) -> (n, k)
            n = knn_idx.shape[0]
            n, nsample, out_planes = new_feats_v.shape
            new_feats_v = (new_feats_v + p_embed).view(n, self.nsample, out_planes)
            new_feats_fwd = (new_feats_v * w).sum(1)

            # new_feats_fwd = self.relu(new_feats_fwd.view(len(p1), 1, -1).repeat(1, self.nsample, 1) + new_feats_v)
            # new_feats_fwd = channelMixMLPs04(new_feats_fwd)
            
            if use_bwd:
                with torch.no_grad():
                    knn_idx_flatten = rearrange(knn_idx, 'm k -> (m k)')
                energy_bwd_flatten_shrink = rearrange(energy, 'm k c -> (m k) c') # c = 3+out_planes
                energy_bwd_prob_flatten_shrink = scatter_softmax(energy_bwd_flatten_shrink, knn_idx_flatten, dim=0)
                
                new_feats_v_bwd_flatten_shrink = rearrange(new_feats_v, 'm k c -> (m k) c') # c = 3+out_planes
                new_feats_weighted_flatten = new_feats_v_bwd_flatten_shrink * energy_bwd_prob_flatten_shrink 
                if mask != None:
                    # backward cost
                    mask_groupped = mask[knn_idx_flatten, :]
                    new_feats_bwd = mask_groupped * new_feats_weighted_flatten
                    
                new_feats_bwd = scatter_sum(new_feats_weighted_flatten, knn_idx_flatten, dim=0, dim_size=len(p1))
                # new_feats_bwd = self.relu(new_feats_bwd.view(len(p1), 1, -1).repeat(1, self.nsample, 1) + new_feats_v)
                # new_feats_bwd = channelMixMLPs04(new_feats_bwd)
                # new_feats_bwd = scatter_max(new_feats_weighted_flatten, knn_idx_flatten, dim=0, dim_size=len(p1))[0]
            else:
                new_feats_bwd = None
            

        if use_cross_att:
            # new_feats_fwd = new_feats_fwd.sum(1)
            # new_feats_fwd = torch.max(new_feats_fwd, dim=1)[0]
            return new_feats_fwd, new_feats_bwd, pos_feats_knn, w
        else:
            return knn_idx, pos_feats_knn, w
            
            
    def forward(self, pc1, pc2, sf=None, sf_feat=None, mask=None):
        '''
        Input:
        (dense) pc1: [pos:[n, 3]; feats:[n, c]; cnt_num:[b*n]]
        (sparse) pc2: [pos:[m, 3]; feats:[m, c]; cnt_num:[b2*n]]
        s_sf: [n,3], cnt_num: [b]
        '''
        p1, feats1, cnt_num1 = pc1
        # feats1 = self.featMLP(feats1)
        p2, feats2, cnt_num2 = pc2 
        # feats2 = self.featMLP(feats2)
        B = None
        if len(p1.shape) == 3:
            B = p1.shape[0]
            p1 = p1.permute(0,2,1).contiguous().view(-1, p1.shape[1])
            feats1 = feats1.permute(0,2,1).contiguous().view(-1, feats1.shape[1])
            pc1 = [p1, feats1, cnt_num1]
            sf_feat = sf_feat.permute(0,2,1).contiguous().view(-1, sf_feat.shape[1])
        if len(p2.shape) == 3:
            p2 = p2.permute(0,2,1).contiguous().view(-1, p2.shape[1])
            feats2 = feats2.permute(0,2,1).contiguous().view(-1, feats2.shape[1])
            pc2 = [p2, feats2, cnt_num2]
            
        if sf != None:
            sf = sf.permute(0,2,1).contiguous().view(-1, sf.shape[1])
            
        inter_feats_fwd, inter_feats_bwd, _, _ = self.get_patch_features(pc1, pc2, channelMixMLPs01=self.channelMixMLPs01, channelMixMLPs02=self.channelMixMLPs02, channelMixMLPs03=self.channelMixMLPs03, channelMixMLPs04=self.channelMixMLPs04, sf=sf, mask=mask, use_cross_att=True, use_bwd=True)
        # inter_feats_bwd, inter_feats_fwd_res, _, _ = self.get_patch_features(pc2, pc1, channelMixMLPs01=self.channelMixMLPs01, channelMixMLPs02=self.channelMixMLPs02, channelMixMLPs03=self.channelMixMLPs03, use_cross_att=True)

        # forward(self, bacth, pc, cost_volume, flow=None, mask=None):

        sf_feat, sf = self.sfnet(B, [p1, sf_feat, cnt_num1], cost_volume=inter_feats_fwd, flow=sf, mask=mask, recurrent_mode=True)

        #  sf_feat, sf = self.sfnet(B, [p1, sf_feat, cnt_num1], cost_volume=inter_feats_fwd, flow=sf, mask=mask, recurrent_mode=True)

        pc1 = [p1, inter_feats_fwd, cnt_num1]
        pc2 = [p2, inter_feats_bwd, cnt_num2]
        inter_feats_fwd, inter_feats_bwd, _, _ = self.get_patch_features(pc1, pc2, channelMixMLPs01=self.channelMixMLPs01_2, channelMixMLPs02=self.channelMixMLPs02_2, channelMixMLPs03=self.channelMixMLPs03_2, channelMixMLPs04=self.channelMixMLPs04_2, sf=sf, mask=mask, use_cross_att=True, use_bwd=True)
        
        # if self.sf_channel == 0:
        sf_feat, sf = self.sfnet2(B, [p1, sf_feat, cnt_num1], cost_volume=inter_feats_fwd, flow=sf, mask=mask, recurrent_mode=True)
        # else:
        #     sf_feat, sf = self.sfnet(B, [p1, sf_feat, cnt_num1], cost_volume=inter_feats_fwd, flow=sf, mask=mask, recurren_mode=True)

        sf_feat = sf_feat.view(B,-1, sf_feat.shape[-1]).permute(0,2,1).contiguous()
        sf = sf.view(B,-1, sf.shape[-1]).permute(0,2,1).contiguous()
        inter_feats_fwd = inter_feats_fwd.view(B,-1, inter_feats_fwd.shape[-1]).permute(0,2,1).contiguous()
        inter_feats_bwd = inter_feats_bwd.view(B,-1, inter_feats_bwd.shape[-1]).permute(0,2,1).contiguous()
        return inter_feats_fwd, inter_feats_bwd, sf_feat, sf

    
'''
根据特征相关性以及遮挡区域掩膜实现场景流上采样估计
'''
class SceneFlowEstimatorPointConv2(nn.Module):
    def __init__(self, feat_ch, cost_ch, flow_ch = 3, channels = [128, 128], mlp = [128, 64], share_planes=8, neighbors = 9, clamp = [-200, 200], use_leaky = True):
        super(SceneFlowEstimatorPointConv2, self).__init__()
        self.clamp = clamp
        self.use_leaky = use_leaky
        self.neighbors = neighbors
        self.pointconv_list = nn.ModuleList()
       
        last_channel = feat_ch + cost_ch + flow_ch

        for _, ch_out in enumerate(channels):
            # in_planes, out_planes, share_planes=8, nsample=16
            pointconv = PointMixerIntraSetLayerPaper(in_planes=last_channel, out_planes=ch_out, share_planes=share_planes, nsample=neighbors)
            self.pointconv_list.append(pointconv)
            last_channel = ch_out 
        
        self.mlp_convs = nn.ModuleList()
        last_channel = last_channel
        for _, ch_out in enumerate(mlp):
            self.mlp_convs.append(Conv1d(last_channel, ch_out))
            last_channel = ch_out

        self.fc = nn.Conv1d(last_channel, 3, 1)

    def forward(self, bacth, pc, cost_volume, flow=None, mask=None, recurrent_mode=False):
        '''
        feats: B C1 N
        cost_volume: B C2 N
        flow: B 3 N
        '''
        xyz, feats, cnt_num = pc
       
        B = None
        if not recurrent_mode:
            if len(pc) == 3:
                B = xyz.shape[0]
                xyz = xyz.permute(0,2,1).contiguous().view(-1, xyz.shape[1])
                if feats != None:
                    feats = feats.permute(0,2,1).contiguous().view(-1, feats.shape[1])
                pc = [pc, feats, cnt_num]
                cost_volume = cost_volume.permute(0,2,1).contiguous().view(-1, cost_volume.shape[1])
                if flow != None:
                    flow = flow.permute(0,2,1).contiguous().view(-1, flow.shape[1])
            
        # xyz, feats, cnt_num = pc
        if flow == None:
            new_points = torch.cat([feats, cost_volume], dim = 1)
        else:
            new_points = torch.cat([feats, cost_volume, flow], dim = 1) #
            

        for _, pointconv in enumerate(self.pointconv_list):
            new_points = pointconv([xyz, new_points, cnt_num])[0]

        new_points = new_points.reshape(bacth, -1, new_points.shape[-1]).permute(0,2,1).contiguous()
    
        for conv in self.mlp_convs:
            new_points = conv(new_points)

        re_flow = self.fc(new_points)

        if recurrent_mode:
            new_points = new_points.permute(0,2,1).contiguous().view(-1, new_points.shape[1])
            re_flow = re_flow.permute(0,2,1).contiguous().view(-1, re_flow.shape[1])

        if flow != None:
            re_flow  = flow + re_flow
        
        return new_points, re_flow.clamp(self.clamp[0], self.clamp[1])
 



'''
方案1：利用FPN网络进行上采样到指定层信息，进行多尺度特征信息组合；
方案2：利用PV-RAFT网络实现VFE的多尺度特征信息组合；
'''
class OcclusisonEstiamtionNet(nn.Module):
    def __init__(self, nsample, voxel_size, resolution, in_channels, num_levels=2, out_channel=256, transup='SymmetricTransitionUpBlock'):
        super(OcclusisonEstiamtionNet, self).__init__()
        self.nsample = nsample
        self.num_levels = num_levels
        self.base_scale = voxel_size
        self.voxel_resolution = resolution
        self.out_channel = out_channel
        self.transup = transup
        self.TransitionUpNetlayers = nn.ModuleList() # nn.Sequential()
        for i, in_channel in enumerate(in_channels):
            self.TransitionUpNetlayers.append(SymmetricTransitionUpBlock(in_channels[-1], in_channel, out_channel, nsample=16))
        # self.TransitionUpNet = SymmetricTransitionUpBlock(in_channel, out_channel, nsample=16)
        last_channel = self.num_levels
        self.fc_occ = nn.Sequential(
            nn.Conv1d(last_channel, last_channel, 1),
            nn.BatchNorm1d(last_channel),
            nn.LeakyReLU(LEAKY_RATE, inplace=True),
            nn.Conv1d(last_channel, 1, 1),
            nn.BatchNorm1d(1),
            nn.Sigmoid()
        )
    
    @staticmethod
    def calculate_corr(fmap1, fmap2):
        npoints, nsample, dim = fmap1.shape
        corr = torch.matmul(fmap1, fmap2.transpose(1, 2))
        corr = corr / torch.sqrt(torch.tensor(dim).float()) # N, K , K
        return corr[:,:,0]
    
    def reshape_inpus(self, pxo):
        p1, feats1, cnt_num1 = pxo
        # p2, feats2, cnt_num2 = pc2 
        # B = None
        if len(p1.shape) == 3:
            B = p1.shape[0]
            p1 = p1.permute(0,2,1).contiguous().view(-1, p1.shape[1])
            feats1 = feats1.permute(0,2,1).contiguous().view(-1, feats1.shape[1])
            pc1 = [p1, feats1, cnt_num1]
        
        return pc1

    def forward(self, pc1s, pc2s, sf):
        '''
        Input:
        (dense) pc1: [pos:[n, 3]; feats:[n, c]; cnt_num:[b*n]]
        (sparse) pc2: [pos:[m, 3]; feats:[m, c]; cnt_num:[b2*n]]
        s_sf: [n,3], cnt_num: [b]
        '''
        assert len(pc1s) == len(pc2s)
        assert len(pc1s) == self.num_levels
        
        for i in range(self.num_levels):
            pc1s[i] = self.reshape_inpus(pc1s[i])
            pc2s[i] = self.reshape_inpus(pc2s[i])
        
        B = sf.shape[0]
        sf = sf.permute(0,2,1).contiguous().view(-1, 3)       
        p1, feats1, cnt_num1 = pc1s[-1]
        p2, feats2, cnt_num2 = pc2s[-1]
        N = p1.shape[0]
        feats1_corr = []
        # feats2_corr = []
        knn_idx = pointops.knnquery(self.nsample, p2, p1+sf, cnt_num2, cnt_num1)[0].long()
        p2_groupped = p2[knn_idx, :].view(N, self.nsample, 3)
        with torch.no_grad():
            dis_voxel = torch.round((p2_groupped - p1.unsqueeze(dim=1)) / self.base_scale)
            valid_scatter = (torch.abs(dis_voxel) <= np.floor(self.voxel_resolution / 2)).all(dim=-1) #[N,K]
            valid_scatter = valid_scatter.detach()
        for i in range(self.num_levels-1):
            up_feats1 = self.TransitionUpNetlayers[i](pc1s[self.num_levels-1], pc1s[i])
            up_feats2 = self.TransitionUpNetlayers[i](pc2s[self.num_levels-1], pc2s[i])
            feats1_groupped = up_feats1.view(len(p1), 1, -1).repeat(1, self.nsample, 1)
            feats2_groupped = up_feats2[knn_idx, :].view(len(p1), self.nsample, -1)
            corr = self.calculate_corr(feats1_groupped, feats2_groupped)
            valid_corr = torch.sum(corr * valid_scatter, dim=-1, keepdim=True)
            feats1_corr.append(valid_corr)
            # feats1_corr.append(up_feats1)
            # feats2_corr.append(up_feats2)
        feats1_groupped = feats1.view(len(p1), 1, -1).repeat(1, self.nsample, 1)
        feats2_groupped = feats2[knn_idx, :].view(len(p1), self.nsample, -1)
        corr = self.calculate_corr(feats1_groupped, feats2_groupped)
        valid_corr = torch.sum(corr * valid_scatter, dim=-1, keepdim=True)
        feats1_corr.append(valid_corr)
        corr_feature = torch.cat(feats1_corr, dim=1)
        # if self.num_levels > 2:
        #     corr_feature = torch.cat(feats1_corr, dim=1)
        # else:
        #     corr_feature = feats1_corr[0]
        occ_mask = self.fc_occ(corr_feature.unsqueeze(-1).contiguous()) #[N,1,1]
        # knn_idx = pointops.knnquery(self.nsample, p1, p1, cnt_num1, cnt_num2)[0].long()
        # knn_idx = pointops.knnquery(self.nsample, p2, p1+sf, cnt_num2, cnt_num1)[0].long()
    
        return occ_mask.squeeze(-1).contiguous()
        
            
class PointSetLayer(nn.Module):
    def __init__(self, nsample, num_levels, voxel_size, resolution, in_channel, hidden_channel, out_channel, use_voxel_mixer=True, transup='SymmetricTransitionUpBlock'):
        super().__init__()
        self.nsample = nsample
        self.use_voxel_mixer = use_voxel_mixer
        self.num_levels = num_levels
        self.voxel_size = voxel_size
    
        self.resolution = resolution
        self.in_channel = in_channel
        self.hidden_channel = hidden_channel
        self.out_channel = out_channel
        # self.n_latent = n_latent

        # self.register_buffer('voxel_size', torch.FloatTensor(voxel_size).view(1, -1))

        # self.input_embed = MLP(self.in_channel, self.hidden_channel, self.out_channel, 2)
        # self.pec = PositionalEncodingFourier(self.hidden_channel, self.out_channel)
        # self.vsa_mlp = MLP_VSA_Layer(self.hidden_channel, self.out_channel)
        # self.pre_mlp = nn.Sequential(
        #     nn.Linear(self.out_channel, self.out_channel),
        #     nn.BatchNorm1d(self.out_channel, eps=1e-3, momentum=0.01),
        #     nn.ReLU(),
        #     nn.Linear(self.out_channel, self.out_channel),
        #     nn.BatchNorm1d(self.out_channel, eps=1e-3, momentum=0.01),
        #     nn.ReLU(),
        #     nn.Linear(self.out_channel, self.out_channel),
        #     nn.BatchNorm1d(self.out_channel, eps=1e-3, momentum=0.01),
        # )
        # # the learnable latent codes can be obsorbed by the linear projection
        # self.score = nn.Linear(self.out_channel, 1)

        # self.post_mlp = nn.Sequential(
        #     nn.Conv1d(self.out_channel * self.num_levels, self.out_channel, 1),
        #     nn.BatchNorm1d(self.out_channel, eps=1e-3, momentum=0.01),
        #     nn.ReLU(),
        #     nn.Conv1d(self.out_channel, self.out_channel, 1 ),
        #     nn.BatchNorm1d(self.out_channel, eps=1e-3, momentum=0.01),
        #     nn.ReLU(),
        #     nn.Conv1d(self.out_channel, self.out_channel, 1),
        #     nn.BatchNorm1d(self.out_channel, eps=1e-3, momentum=0.01)
        # )

        if out_channel is None:
            self.linear1 = nn.Sequential(
                nn.Linear(2*in_channel, in_channel), 
                nn.BatchNorm1d(in_channel), 
                nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential(
                nn.Linear(in_channel, in_channel), 
                nn.ReLU(inplace=True))            
        else:
            self.linear1 = nn.Sequential( # input.shape = [N, L]
                nn.Linear(in_channel, out_channel), 
                nn.BatchNorm1d(out_channel),  
                nn.ReLU(inplace=True))
            self.linear2 = nn.Sequential( # input.shape = [N, L]
                nn.Linear(in_channel, out_channel), 
                nn.BatchNorm1d(out_channel), 
                nn.ReLU(inplace=True))
            self.channel_shrinker = nn.Sequential( # input.shape = [N*K, L]
                nn.Linear(in_channel+3, in_channel),
                Rearrange('n k c -> n c k'),
                nn.BatchNorm1d(in_channel),
                Rearrange('n c k -> n k c'),
                nn.ReLU(inplace=True),
                nn.Linear(in_channel, 1))

    def __call__(self, pos1, pos2, use_voxel=False):
        if use_voxel:
            return self.get_voxel_feature(pos1, pos2)
        else:
            return self.get_symmetric_knn_feature(pos1, pos2)

    def get_voxel_feature(self, pos1, pos2):
        '''
        Input:
        (dense) pc1: [pos:[n, 3]; feats:[n, c]; cnt_num:[b*n]]
        (sparse) pc2: [pos:[m, 3]; feats:[m, c]; cnt_num:[b2*n]]
        s_sf: [n,3], cnt_num: [b]
        '''
        return None
        # p1, feats1, cnt_num1 = pos1
        # p2, feats2, cnt_num2 = pos2 
        # # B,N = pos1.shape
        # # b, m_p, _ = pos2.size()
        # knn_idx = pointops.knnquery(self.nsample, p1, p2, cnt_num1, cnt_num2)[0].long()

        # with torch.no_grad():
        #     knn_idx_flatten = rearrange(knn_idx, 'm k -> (m k)')
        
        # feats2_groupped = feats2.view(len(p2), 1, -1).repeat(1, self.nsample, 1)
        # pos1_groupped = p1[knn_idx_flatten, :].view(len(p2), self.nsample, 3)
        # pos_diff = pos1_groupped - p2.unsqueeze(1) #[m,k,3]
        # feats2_groupped = torch.cat([pos_diff, feats2_groupped], dim=-1) # (109, 16, 259) # (m, nsample, 3+c)
        # corr_feature = []
        # from torch_scatter import scatter_add
        # for i in range(self.num_levels):
        #     with torch.no_grad():
        #         r = self.voxel_size * (2 ** i)
        #         dis_voxel = torch.round(pos_diff / r) #sub-voxel
        #         valid_scatter = (torch.abs(dis_voxel) <= np.floor(self.resolution / 2)).all(dim=-1) #[m,k,3]
        #         dis_voxel = dis_voxel - (-1) # FROM 0 TO X
        #         # cube_idx = dis_voxel[:, :, 0] * (self.resolution ** 2) +\
        #         #     dis_voxel[:, :, 1] * self.resolution + dis_voxel[:, :, 2] #X * R^2:表示每层包含多少个cube；Y*R:表示第几层;Z表示指定层的第几个cube
        #         # cube_idx_scatter = cube_idx.type(torch.int64) * valid_scatter

        #         valid_scatter = valid_scatter.detach()
        #         # cube_idx_scatter = cube_idx_scatter.detach()
            
        #     # encoder
        #     if self.use_voxel_mixer:
        #         # SOLUTION A: Voxel Point Mixer-based
        #         feats_knn_flatten = rearrange(feats2_groupped, 'm k c -> (m k) c') # c = 3+out_planes
        #         feats_knn_flatten_shrink = self.channel_shrinker(feats_knn_flatten) # (m, nsample, 1)
        #         feats_knn_prob_flatten_shrink = scatter_softmax(feats_knn_flatten_shrink, knn_idx_flatten, dim=0) * valid_scatter.view(-1, 1)

        #         feats_knn_prob_shrink = rearrange(feats_knn_prob_flatten_shrink, '(m k) 1 -> m k 1', k=self.nsample)
        #         up_feats_weighted = self.linear2(feats2).unsqueeze(1) * feats_knn_prob_shrink # (m, nsample, c)
        #         up_feats_weighted_flatten = rearrange(up_feats_weighted, 'm k c -> (m k) c')
        #         up_feats = scatter_sum(up_feats_weighted_flatten, knn_idx_flatten, dim=0, dim_size=len(p1))
        #         inp_feats_ = self.linear1(feats1) + up_feats
        #     else:
        #         # SOLUTION B: Transformer-based
        #         inp_feats = self.input_embed(pos1_groupped.view(-1, 3)) #X,Y,Z, shape=[N,C]
        #         pe_raw = (pos_diff - dis_voxel * r) / r
        #         pe_raw = pe_raw.view(-1, 3)
        #         inp_feats = inp_feats + self.pec(pe_raw)
        #         #latent code
        #         inp_feats = self.pre_mlp(inp_feats)
        #         attn = scatter_softmax(self.score(inp_feats), knn_idx_flatten, dim=0) * valid_scatter.view(-1, 1)
        #         pe_raw = pe_raw.view(-1, 3)
        #         dot = (attn[:, :, None] * inp_feats.view(-1, 1, self.out_channel)).view(-1, self.out_channel)
        #         inp_feats_ = scatter_sum(dot, knn_idx_flatten, dim=0)# Max Number:  self.resolution ** 3
        #         # inp_feats_ = scatter_sum(dot, knn_idx_flatten, dim=0) # Max Number:  self.resolution ** 3
            
        #     # if corr.shape[-1] != self.resolution ** 3: #TO ALIGN THE DATA STRUCTURE
        #     #     repair = torch.zeros([b, n_p, self.resolution ** 3 - corr.shape[-1]], device=coords.device)
        #     #     corr = torch.cat([corr, repair], dim=-1)
            
        #     corr_feature.append(inp_feats_.contiguous())
        # corr_feature = self.post_mlp(torch.cat(corr_feature, dim=1))
        # # return self.out_conv(torch.cat(corr_feature, dim=1))
        # return corr_feature


    def get_symmetric_knn_feature(self, d_pos, s_pos):
        '''
        Input:
        (dense) pc1: [pos:[n, 3]; feats:[n, c]; cnt_num:[b]]
        (sparse) pc2: [pos:[m, 3]; feats:[m, c]; cnt_num:[b2]]
        s_sf: [n,3], cnt_num: [b]
        '''
        p1, x1, o1 = d_pos
        p2, x2, o2 = s_pos 
        knn_idx = pointops.knnquery(self.nsample, p1, p2, o1, o2)[0].long()

        with torch.no_grad():
            knn_idx_flatten = rearrange(knn_idx, 'm k -> (m k)')
        p_r = p1[knn_idx_flatten, :].view(len(p2), self.nsample, 3) - p2.unsqueeze(1)
        x2_knn = x2.view(len(p2), 1, -1).repeat(1, self.nsample, 1)
        x2_knn = torch.cat([p_r, x2_knn], dim=-1) # (109, 16, 259) # (m, nsample, 3+c)

        # with torch.no_grad():
        #     knn_idx_flatten = knn_idx_flatten.unsqueeze(-1) # (m*nsample, 1)
        # x2_knn_flatten = rearrange(x2_knn, 'm k c -> (m k) c') # c = 3+out_planes
        x2_knn_shrink = self.channel_shrinker(x2_knn) # (m, nsample, 1)
        x2_knn_flatten_shrink = rearrange(x2_knn_shrink, 'm k c -> (m k) c') # c = 3+out_planes
        x2_knn_prob_flatten_shrink = scatter_softmax(x2_knn_flatten_shrink, knn_idx_flatten, dim=0)

        x2_knn_prob_shrink = rearrange(x2_knn_prob_flatten_shrink, '(m k) 1 -> m k 1', k=self.nsample)
        up_x2_weighted = self.linear2(x2).unsqueeze(1) * x2_knn_prob_shrink # (m, nsample, c)
        up_x2_weighted_flatten = rearrange(up_x2_weighted, 'm k c -> (m k) c')
        up_x2 = scatter_sum(up_x2_weighted_flatten, knn_idx_flatten, dim=0, dim_size=len(p1))
        y = self.linear1(x1) + up_x2

        return y


class SceneFlowUpsampleNet(nn.Module):
    def __init__(self, nsample, num_levels, voxel_size, resolution, sf_channel, in_channel, hidden_channel, out_channel, use_voxel_mixer=True,
        intraLayer='PointMixerIntraSetLayer',
        interLayer='PointMixerInterSetLayer',
        transup='SymmetricTransitionUpBlock', 
        transdown='TransitionDownBlock'):
        super().__init__()
        self.nsample = nsample
        # self.use_voxel_mixer = use_voxel_mixer
        # self.num_levels = num_levels
        # self.voxel_size = voxel_size
    
        # self.resolution = resolution
        # self.in_channel = in_channel
        # self.hidden_channel = hidden_channel
        self.out_channel = out_channel
        self.featwiselayer = PointSetLayer(nsample=nsample, num_levels=num_levels, voxel_size=voxel_size, resolution=resolution, in_channel=in_channel, hidden_channel=hidden_channel, out_channel=out_channel, use_voxel_mixer=use_voxel_mixer)
        
        # self.weightnet = nn.Sequential(
        #     nn.Conv1d(self.out_channel * 2, self.out_channel * 2, 1),
        #     nn.BatchNorm1d(self.out_channel * 2, eps=1e-3, momentum=0.01),
        #     nn.ReLU(),
        #     nn.Conv1d(self.out_channel * 2, self.out_channel, 1),
        #     nn.BatchNorm1d(self.out_channel, eps=1e-3, momentum=0.01),
        #     nn.ReLU(),
        #     nn.Conv1d(self.out_channel, 1, 1),
        # )
        self.weightnet = nn.Sequential(
            nn.Conv1d(self.out_channel, self.out_channel, 1),
            nn.BatchNorm1d(self.out_channel, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Conv1d(self.out_channel, self.out_channel, 1),
            nn.BatchNorm1d(self.out_channel, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Conv1d(self.out_channel, 1, 1),
        )
        
        # last_channel = out_channel + sf_channel
        last_channel = sf_channel + in_channel
        self.sffeatslayer = nn.Sequential(
            nn.Conv1d(last_channel, self.out_channel * 2, 1),
            nn.BatchNorm1d(self.out_channel * 2, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Conv1d(self.out_channel * 2, self.out_channel, 1),
            nn.BatchNorm1d(self.out_channel, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Conv1d(self.out_channel, self.out_channel, 1),
        )
    
    def forward(self, pc1, pc2, s_pc1, sf, sf_feats, mask=None):
        '''
        Input:
        (dense) pc1: [pos:[n, 3]; feats:[n, c]; bacth_num:[b]]
        (sparse) pc2: [pos:[n, 3]; feats:[n, c]; batch_num:[b]]
        sf: [n,3], cnt_num: [b]
        '''
        # p1, _, b_num1 = pc1
        # p2, _, b_num2 = pc2 
        # s_p1, _, s_b_num1 = s_pc1 
        
        p1, feats1, b_num1 = pc1
        p2, feats2, b_num2 = pc2 
        s_p1, s_feats1, s_b_num1 = s_pc1 
        B = None
        if len(p1.shape) == 3:
            B = p1.shape[0]
            p1 = p1.permute(0,2,1).contiguous().view(-1, p1.shape[1])
            feats1 = feats1.permute(0,2,1).contiguous().view(-1, feats1.shape[1])
            pc1 = [p1, feats1, b_num1]
        
        if len(p2.shape) == 3:
            p2 = p2.permute(0,2,1).contiguous().view(-1, p2.shape[1])
            feats2 = feats2.permute(0,2,1).contiguous().view(-1, feats2.shape[1])
            pc2 = [p2, feats2, b_num2]
            
        if len(s_p1.shape) == 3:
            s_p1 = s_p1.permute(0,2,1).contiguous().view(-1, s_p1.shape[1])
            sf = sf.permute(0,2,1).contiguous().view(-1, sf.shape[1])
            s_feats1 = s_feats1.permute(0,2,1).contiguous().view(-1, s_feats1.shape[1])
            s_pc1 = [s_p1, s_feats1, s_b_num1]
        
        # sf_p, _ = po_from_batched_pcd(sf)
        inter_knn_idx = pointops.knnquery(self.nsample, p2, s_p1+sf, b_num2, s_b_num1)[0].long()
        feats2_voxel = self.featwiselayer(pc2, pc2)
        inter_feats1_groupped = feats2_voxel[inter_knn_idx, :].view(len(s_p1), self.nsample, self.out_channel)
        
        intra_knn_idx = pointops.knnquery(self.nsample, p1, s_p1, b_num1, s_b_num1)[0].long()
        feats1_voxel = self.featwiselayer(pc1, pc1)
        intra_feats1_groupped = feats1_voxel[intra_knn_idx, :].view(len(s_p1), self.nsample, self.out_channel)
        # nn_feats = self.weightnet(torch.cat([intra_feats1_groupped, inter_feats1_groupped], dim=-1).permute(0,2,1).contiguous())
        # nn_feats = self.weightnet(intra_feats1_groupped.permute(0,2,1).contiguous()) + self.weightnet(inter_feats1_groupped.permute(0,2,1).contiguous())
        nn_feats = self.weightnet((intra_feats1_groupped - inter_feats1_groupped).permute(0,2,1).contiguous())
        # nn_weight = F.softmax(nn_feats, dim=1)
        with torch.no_grad():
            intra_knn_idx_flatten = rearrange(intra_knn_idx, 'm k -> (m k)')
            intra_knn_idx_flatten = intra_knn_idx_flatten.unsqueeze(-1) # (m*nsample, 1)
        # nn_feats_flatten = rearrange(nn_feats, 'm k c -> (m k) c') # c = 3+out_planes
        nn_feats_prob_flatten_shrink = scatter_softmax(nn_feats.permute(0,2,1).view(-1, nn_feats.shape[1]).contiguous(), intra_knn_idx_flatten, dim=0)

        # x2_knn_prob_shrink = rearrange(x2_knn_prob_flatten_shrink, '(m k) 1 -> m k 1', k=self.nsample)
        # up_x2_weighted = self.linear2(x2).unsqueeze(1) * x2_knn_prob_shrink # (m, nsample, c)
        # up_x2_weighted_flatten = rearrange(up_x2_weighted, 'm k c -> (m k) c')
        # up_x2 = scatter_sum(up_x2_weighted_flatten, knn_idx_flatten, dim=0, dim_size=len(p1))
        # y = self.linear1(x1) + up_x2
            
            
        # knn_idx = pointops.knnquery(self.nsample, s_p1, p1, s_b_num1, b_num1)[0].long()
        # intra_sf_groupped = sf[knn_idx, :].view(len(p1), self.nsample, 3)
        intra_sf_groupped = sf.view(sf.shape[0], 1, -1).repeat(1, self.nsample, 1).view(-1, sf.shape[-1])
        up_sf_p = scatter_sum(intra_sf_groupped * nn_feats_prob_flatten_shrink, intra_knn_idx_flatten, dim=0, dim_size=len(p1))
        # intra_sf_feats_groupped = sf_feats[knn_idx, :].view(len(p1), self.nsample, -1)
        intra_sf_feats_groupped = sf_feats.view(sf.shape[0], 1, -1).repeat(1, self.nsample, 1)
        intra_sf_feats_groupped = torch.cat([intra_sf_feats_groupped, intra_feats1_groupped], dim=-1)
        # intra_sf_feats_groupped = torch.cat([intra_sf_feats_groupped], dim=-1)
        # intra_sf_feats_groupped = intra_sf_feats_groupped + intra_feats1_groupped
        up_sf_feats = self.sffeatslayer(intra_sf_feats_groupped.permute(0,2,1).contiguous())
        up_sf_feats = scatter_sum(up_sf_feats.permute(0,2,1).reshape(-1, up_sf_feats.shape[1]).contiguous() * nn_feats_prob_flatten_shrink, intra_knn_idx_flatten, dim=0, dim_size=len(p1))
        
        if B != None:
            up_sf_feats = up_sf_feats.view(B, -1, up_sf_feats.shape[-1]).permute(0,2,1).contiguous()
            up_sf_p = up_sf_p.view(B, -1, up_sf_p.shape[-1]).permute(0,2,1).contiguous()
        return up_sf_feats, up_sf_p


class UpsampleFlow(nn.Module):
    def forward(self, xyz, sparse_xyz, sparse_flow):
        #import ipdb; ipdb.set_trace()
        B, C, N = xyz.shape
        _, _, S = sparse_xyz.shape

        xyz = xyz.permute(0, 2, 1) # B N 3
        sparse_xyz = sparse_xyz.permute(0, 2, 1) # B S 3
        # if sparse_flow.shape[1] == 3:
        sparse_flow = sparse_flow.permute(0, 2, 1) # B S 3
        
        dist, knn_idx = knn_point(3, sparse_xyz, xyz)
        dist = dist.clamp(min = 1e-10)
        # grouped_xyz_norm = index_points_group(sparse_xyz, knn_idx) - xyz.view(B, N, 1, C)
        # dist = torch.norm(grouped_xyz_norm, dim = 3).clamp(min = 1e-10)
        norm = torch.sum(1.0 / dist, dim = 2, keepdim = True)
        weight = (1.0 / dist) / norm 

        grouped_flow = index_points_group(sparse_flow, knn_idx)
        dense_flow = torch.sum(weight.view(B, N, 3, 1) * grouped_flow, dim = 2)
        return dense_flow.permute(0, 2, 1).contiguous() 
    

class SceneFlowUpsampleNet2(nn.Module):
    def __init__(self, nsample, num_levels, voxel_size, resolution, sf_channel, in_channel, cost_channel, out_channel, use_voxel_mixer=True):
        super().__init__()
        self.nsample = nsample
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.upsample = UpsampleFlow()
        self.deconvnet = Conv1d(cost_channel, out_channel)

        self.weightnet = nn.Sequential(
            nn.Conv1d(out_channel+in_channel, self.out_channel, 1),
            nn.BatchNorm1d(self.out_channel, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Conv1d(self.out_channel, self.out_channel, 1),
            nn.BatchNorm1d(self.out_channel, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Conv1d(self.out_channel, 1, 1),
        )
        
        # last_channel = out_channel + sf_channel
        last_channel = sf_channel + in_channel
        self.sffeatslayer = nn.Sequential(
            nn.Conv1d(last_channel, self.out_channel * 2, 1),
            nn.BatchNorm1d(self.out_channel * 2, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Conv1d(self.out_channel * 2, self.out_channel, 1),
            nn.BatchNorm1d(self.out_channel, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Conv1d(self.out_channel, self.out_channel, 1),
        )

    
    def forward(self, pc1, s_pc1, sf, sf_feats, s_cost):
        '''
        Input:
        (dense) pc1: [pos:[n, 3]; feats:[n, c]; bacth_num:[b]]
        (sparse) pc2: [pos:[n, 3]; feats:[n, c]; batch_num:[b]]
        sf: [n,3], cnt_num: [b]
        '''
        # p1, _, b_num1 = pc1
        # p2, _, b_num2 = pc2 
        # s_p1, _, s_b_num1 = s_pc1 
        
        p1, feats1, b_num1 = pc1
        # p2, feats2, b_num2 = pc2 
        s_p1, s_feats1, s_b_num1 = s_pc1 
        B = None
        cost = self.upsample(p1, s_p1, s_cost)
        cost = self.deconvnet(cost)
        if len(p1.shape) == 3:
            B = p1.shape[0]
            p1 = p1.permute(0,2,1).contiguous().view(-1, p1.shape[1])
            feats1 = feats1.permute(0,2,1).contiguous().view(-1, feats1.shape[1])
            pc1 = [p1, feats1, b_num1]
            cost = cost.permute(0,2,1).contiguous().view(-1, cost.shape[1])

        if len(s_p1.shape) == 3:
            s_p1 = s_p1.permute(0,2,1).contiguous().view(-1, s_p1.shape[1])
            sf = sf.permute(0,2,1).contiguous().view(-1, sf.shape[1])
            s_feats1 = s_feats1.permute(0,2,1).contiguous().view(-1, s_feats1.shape[1])
            s_pc1 = [s_p1, s_feats1, s_b_num1]

        intra_knn_idx = pointops.knnquery(self.nsample, p1, s_p1, b_num1, s_b_num1)[0].long()
        # feats1_voxel = self.featwiselayer(pc1, pc1)
        
        intra_cost_groupped = cost[intra_knn_idx, :].view(len(s_p1), self.nsample, self.out_channel)
        intra_feats1_groupped = feats1[intra_knn_idx, :].view(len(s_p1), self.nsample, self.in_channel)
        intra_cost_groupped = torch.cat([intra_cost_groupped, intra_feats1_groupped], dim=-1)
        nn_feats = self.weightnet(intra_cost_groupped.permute(0,2,1).contiguous())
        # nn_weight = F.softmax(nn_feats, dim=1)
        with torch.no_grad():
            intra_knn_idx_flatten = rearrange(intra_knn_idx, 'm k -> (m k)')
            intra_knn_idx_flatten = intra_knn_idx_flatten.unsqueeze(-1) # (m*nsample, 1)
        # nn_feats_flatten = rearrange(nn_feats, 'm k c -> (m k) c') # c = 3+out_planes
        nn_feats_prob_flatten_shrink = scatter_softmax(nn_feats.permute(0,2,1).view(-1, nn_feats.shape[1]).contiguous(), intra_knn_idx_flatten, dim=0)

        intra_sf_groupped = sf.view(sf.shape[0], 1, -1).repeat(1, self.nsample, 1).view(-1, sf.shape[-1])
        up_sf_p = scatter_sum(intra_sf_groupped * nn_feats_prob_flatten_shrink, intra_knn_idx_flatten, dim=0, dim_size=len(p1))
        
        intra_sf_feats_groupped = sf_feats.view(sf.shape[0], 1, -1).repeat(1, self.nsample, 1)
        intra_sf_feats_groupped = torch.cat([intra_sf_feats_groupped, intra_feats1_groupped], dim=-1)

        up_sf_feats = self.sffeatslayer(intra_sf_feats_groupped.permute(0,2,1).contiguous())
        up_sf_feats = scatter_sum(up_sf_feats.permute(0,2,1).reshape(-1, up_sf_feats.shape[1]).contiguous() * nn_feats_prob_flatten_shrink, intra_knn_idx_flatten, dim=0, dim_size=len(p1))
        
        if B != None:
            up_sf_feats = up_sf_feats.view(B, -1, up_sf_feats.shape[-1]).permute(0,2,1).contiguous()
            up_sf_p = up_sf_p.view(B, -1, up_sf_p.shape[-1]).permute(0,2,1).contiguous()
        return up_sf_feats, up_sf_p


class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, use_leaky=True, bn=use_bn):
        super(Conv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

        self.composed_module = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.BatchNorm1d(out_channels) if bn else nn.Identity(),
            relu
        )

    def forward(self, x):
        x = self.composed_module(x)
        return x
    
'''
根据特征相关性以及遮挡区域掩膜实现场景流上采样估计
'''
class SceneFlowEstimatorPointConv(nn.Module):
    def __init__(self, feat_ch, cost_ch, flow_ch = 3, channels = [128, 128], mlp = [128, 64], share_planes=8, neighbors = 9, clamp = [-200, 200], use_leaky = True):
        super(SceneFlowEstimatorPointConv, self).__init__()
        self.clamp = clamp
        self.use_leaky = use_leaky
        
        self.pointconv_list = nn.ModuleList()
        last_channel = feat_ch + cost_ch + flow_ch

        for _, ch_out in enumerate(channels):
            # in_planes, out_planes, share_planes=8, nsample=16
            pointconv = PointMixerIntraSetLayerPaper(in_planes=last_channel, out_planes=ch_out, share_planes=share_planes, nsample=neighbors)
            self.pointconv_list.append(pointconv)
            last_channel = ch_out 
        
        self.mlp_convs = nn.ModuleList()
        for _, ch_out in enumerate(mlp):
            self.mlp_convs.append(Conv1d(last_channel, ch_out))
            last_channel = ch_out

        self.fc = nn.Conv1d(last_channel, 3, 1)

    def forward(self, bacth, pc, cost_volume, flow = None):
        '''
        feats: B C1 N
        cost_volume: B C2 N
        flow: B 3 N
        '''
        xyz, feats, cnt_num = pc
       
        B = None
        if len(pc) == 3:
            B = xyz.shape[0]
            xyz = xyz.permute(0,2,1).contiguous().view(-1, xyz.shape[1])
            feats = feats.permute(0,2,1).contiguous().view(-1, feats.shape[1])
            pc = [pc, feats, cnt_num]
            cost_volume = cost_volume.permute(0,2,1).contiguous().view(-1, cost_volume.shape[1])
            if flow != None:
                flow = flow.permute(0,2,1).contiguous().view(-1, flow.shape[1])
            
        # xyz, feats, cnt_num = pc
        if flow == None:
            new_points = torch.cat([feats, cost_volume], dim = 1)
        else:
            new_points = torch.cat([feats, cost_volume, flow], dim = 1)

        for _, pointconv in enumerate(self.pointconv_list):
            new_points = pointconv([xyz, new_points, cnt_num])[0]

        new_points = new_points.reshape(bacth, -1, new_points.shape[-1]).permute(0,2,1).contiguous()
        for conv in self.mlp_convs:
            new_points = conv(new_points)

        flow = self.fc(new_points)
        # new_points = new_points.permute(0,2,1).reshape(-1, new_points.shape[1]).contiguous()
        # flow = flow.permute(0,2,1).reshape(-1, flow.shape[1]).contiguous()
        
        return new_points, flow.clamp(self.clamp[0], self.clamp[1])
    

def group(nsample, xyz, points, mask=None):
    """
    Input:
        nsample: scalar
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        # new_xyz: sampled points position data, [B, N, C]
        new_points: sampled points data, [B, npoint, nsample, C+D]
    """
    B, N, C = xyz.shape
    S = N
    new_xyz = xyz
    _, idx = knn_point(nsample, xyz, new_xyz)
    grouped_xyz = index_points_group(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    if points is not None:
        grouped_points = index_points_group(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    
    if mask != None:
        grouped_mask = index_points_group(mask.reshape(B, -1, 1), idx) # [B, npoint, nsample, C]
        new_points = new_points * grouped_mask
        grouped_xyz_norm = grouped_xyz_norm * grouped_mask

    return new_points, grouped_xyz_norm

class SceneFlowEstimatorResidual(nn.Module):
    def __init__(self, feat_ch, cost_ch, flow_ch = 3, channels = [128, 128], mlp = [128, 128], neighbors = 9, clamp = [-200, 200], use_leaky = True):
        super(SceneFlowEstimatorResidual, self).__init__()
        self.clamp = clamp
        self.use_leaky = use_leaky
        self.pointconv_list = nn.ModuleList()
        last_channel = feat_ch + cost_ch + flow_ch
        self.feat_ch = feat_ch
        self.neighbors = neighbors

        for _, ch_out in enumerate(channels):
            pointconv = PointConv(neighbors, last_channel + 3, ch_out, bn=False, use_leaky = True)
            self.pointconv_list.append(pointconv)
            last_channel = ch_out 
        
        self.mlp_convs = nn.ModuleList()
        for _, ch_out in enumerate(mlp):
            self.mlp_convs.append(Conv1d(last_channel, ch_out))
            last_channel = ch_out

        self.fc = nn.Conv1d(last_channel, 3, 1)
        
    with torch.no_grad():
        def knn_point(self, pos1, pos2, occ_mask=None):
            '''
            Input:
                k: int32, number of k in k-nn search
                pos1: (batch_size, ndataset, c) float32 array, input points
                pos2: (batch_size, npoint, c) float32 array, query points
            Output:
                val: (batch_size, npoint, k) float32 array, L2 distances
                idx: (batch_size, npoint, k) int32 array, indices to input points
            '''
            k = self.neighbors
            B, N, C = pos1.shape
            M = pos2.shape[1]
            pos1 = pos1.view(B,1,N,-1).repeat(1,M,1,1)
            pos2 = pos2.view(B,M,1,-1).repeat(1,1,N,1)
            dist = torch.sum(-(pos1-pos2)**2,-1)
            dist_mask = torch.ones_like(dist) * ((~occ_mask) * 1000.0).transpose(1,2).repeat(1, M, 1).contiguous()
            dist = dist + dist_mask
            val,idx = dist.topk(k=k, dim=-1)
            return torch.sqrt(-val), idx
        
    def calculate_corr(self, fmap1, fmap2):
        batch, npoints, nsample, dim = fmap1.shape
        corr = torch.matmul(fmap1, fmap2)
        corr = corr / torch.sqrt(torch.tensor(dim).float()) # B, N, K, K
        corr = torch.softmax(corr, -2) * torch.softmax(corr, -1)
        return corr

    def forward(self, xyz, feats, cost_volume, flow = None, mask=None, occ_threshold=0.5):
        '''
        feats: B C1 N
        cost_volume: B C2 N
        flow: B 3 N
        '''
        B, _, N = xyz.shape
        # new_points = torch.cat([flow, feats, cost_volume], dim = 1)
        if flow is None:
            new_points = torch.cat([feats, cost_volume], dim = 1)
        else:
            new_points = torch.cat([flow, feats, cost_volume], dim = 1)

        for _, pointconv in enumerate(self.pointconv_list):
            new_points = pointconv(xyz, new_points)

        for conv in self.mlp_convs:
            new_points = conv(new_points)

        flow = self.fc(new_points).clamp(self.clamp[0], self.clamp[1]) 
        
        if mask != None:
            occ_mask = mask > occ_threshold
            occ_mask_b = occ_mask.reshape(B, -1, 1)
            xyz_t = xyz.transpose(1,2).contiguous()
            _, intra_knn_idx = self.knn_point(xyz_t, xyz_t, occ_mask_b)
            groupped_feats = pointutils.grouping_operation((feats[:,self.feat_ch//2:,:]).contiguous(), intra_knn_idx)
            flow_grouped = pointutils.grouping_operation(flow, intra_knn_idx)
            corr =  self.calculate_corr(groupped_feats.permute(0,2,3,1).contiguous(), groupped_feats.permute(0,2,1,3).contiguous())
            corr = torch.sum(corr, dim=-1, keepdim=True)
            flow_grouped = (corr.permute(0,3,1,2).contiguous() * flow_grouped).sum(dim=-1)
            flow = flow * occ_mask_b.transpose(1,2).contiguous().float() + (~occ_mask_b).transpose(1,2).contiguous().float() * flow_grouped
            
        return new_points, flow
    

class PointConv(nn.Module):
    def __init__(self, nsample, in_channel, out_channel, weightnet = 16, bn = use_bn, use_leaky = True):
        super(PointConv, self).__init__()
        self.bn = bn
        self.nsample = nsample
        self.weightnet = WeightNet(3, weightnet)
        self.linear = nn.Linear(weightnet * in_channel, out_channel)
        if bn:
            self.bn_linear = nn.BatchNorm1d(out_channel)

        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)


    def forward(self, xyz, points, mask=None):
        """
        PointConv without strides size, i.e., the input and output have the same number of points.
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        B = xyz.shape[0]
        N = xyz.shape[2]
        xyz = xyz.permute(0, 2, 1)
        points = points.permute(0, 2, 1)

        new_points, grouped_xyz_norm = group(self.nsample, xyz, points, mask) # [B, npoint, nsample, C+D]

        grouped_xyz = grouped_xyz_norm.permute(0, 3, 2, 1)
        weights = self.weightnet(grouped_xyz) #BxWxKxN
        new_points = torch.matmul(input=new_points.permute(0, 1, 3, 2), other = weights.permute(0, 3, 2, 1)).view(B, N, -1) #BxNxWxK * BxNxKxC => BxNxWxC -> BxNx(W*C)
        new_points = self.linear(new_points)
        if self.bn:
            new_points = self.bn_linear(new_points.permute(0, 2, 1))
        else:
            new_points = new_points.permute(0, 2, 1)

        new_points = self.relu(new_points)

        return new_points
    

class PointConvD(nn.Module):
    def __init__(self, npoint, nsample, in_channel, out_channel, weightnet = 16, bn = use_bn, use_leaky = True):
        super(PointConvD, self).__init__()
        self.npoint = npoint
        self.bn = bn
        self.nsample = nsample
        self.weightnet = WeightNet(3, weightnet)
        self.linear = nn.Linear(weightnet * in_channel, out_channel)
        if bn:
            self.bn_linear = nn.BatchNorm1d(out_channel)

        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

    def forward(self, xyz, points):
        """
        PointConv with downsampling.
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        #import ipdb; ipdb.set_trace()
        B = xyz.shape[0]
        N = xyz.shape[2]
        xyz_t = xyz.permute(0, 2, 1).contiguous()
        points = points.permute(0, 2, 1).contiguous()

        fps_idx = pointutils.furthest_point_sample(xyz_t, self.npoint)  # [B, N]
        new_xyz = pointutils.gather_operation(xyz, fps_idx)  # [B, C, N]

        new_points, grouped_xyz_norm = group_query(self.nsample, xyz_t, new_xyz.permute(0, 2, 1).contiguous(), points)
        grouped_xyz = grouped_xyz_norm.permute(0, 3, 2, 1)
        weights = self.weightnet(grouped_xyz)
        # B, N, S, C
        new_points = torch.matmul(input=new_points.permute(0, 1, 3, 2), other = weights.permute(0, 3, 2, 1)).view(B, self.npoint, -1)
        new_points = self.linear(new_points)
        if self.bn:
            new_points = self.bn_linear(new_points.permute(0, 2, 1))
        else:
            new_points = new_points.permute(0, 2, 1)

        new_points = self.relu(new_points)

        return new_xyz.contiguous(), new_points.contiguous(), fps_idx


class SceneFlowFusionNet(nn.Module):
    def __init__(self, nsample, f1_channel, f2_channel, mlp, mlp2, occ_threshold=0.5, feats_diff=True):
        super().__init__()
        self.nsample = nsample
        self.occ_threshold = occ_threshold
        self.mlp1_convs = nn.ModuleList()
        self.mlp2_convs = nn.ModuleList()
        last_channel = f2_channel+3
        for out_channel in mlp:
            self.mlp1_convs.append(nn.Sequential(nn.Conv2d(last_channel, out_channel, 1, bias=False),
                                                 nn.BatchNorm2d(out_channel),
                                                 nn.ReLU(inplace=False)))
            last_channel = out_channel
        if feats_diff:
            last_channel = mlp[-1]
        else:
            if len(mlp) != 0:
                last_channel = mlp[-1] + f1_channel
            else:
                last_channel = last_channel + f1_channel
        for out_channel in mlp2:
            self.mlp2_convs.append(nn.Sequential(nn.Conv2d(last_channel, out_channel, 1, bias=False),
                                                 nn.BatchNorm2d(out_channel),
                                                 nn.ReLU(inplace=False)))
            last_channel = out_channel

    with torch.no_grad():
        def knn_point(self, pos1, pos2, occ_mask=None):
            '''
            Input:
                k: int32, number of k in k-nn search
                pos1: (batch_size, ndataset, c) float32 array, input points
                pos2: (batch_size, npoint, c) float32 array, query points
            Output:
                val: (batch_size, npoint, k) float32 array, L2 distances
                idx: (batch_size, npoint, k) int32 array, indices to input points
            '''
            k = self.nsample
            B, N, C = pos1.shape
            M = pos2.shape[1]
            pos1 = pos1.view(B,1,N,-1).repeat(1,M,1,1)
            pos2 = pos2.view(B,M,1,-1).repeat(1,1,N,1)
            dist = torch.sum(-(pos1-pos2)**2,-1)
            dist_mask = torch.ones_like(dist) * ((~occ_mask) * 1000.0).transpose(1,2).repeat(1, M, 1).contiguous()
            dist = dist + dist_mask
            val,idx = dist.topk(k=k, dim=-1)
            return torch.sqrt(-val), idx
    
    def calculate_corr(self, fmap1, fmap2):
        batch, npoints, nsample, dim = fmap1.shape
        corr = torch.matmul(fmap1, fmap2)
        corr = corr / torch.sqrt(torch.tensor(dim).float()) # B, N, K, K
        return corr
 
    def forward(self, pc1, sf, batch, mask=None):
        '''
        Input:
        (dense) pc1: [pos:[n, 3]; feats:[n, c]; bacth_num:[b]]
        sf: [n,3], cnt_num: [b]
        mask: [B,N,1]
        '''
        p1, feat1, b_num1 = pc1
        B, _, N = p1.shape
        
        occ_mask = mask > self.occ_threshold
        # p1_b = p1.reshape(batch, -1, 3)
        occ_mask_b = occ_mask.reshape(batch, -1, 1)
        p1_t = p1.transpose(1,2).contiguous()
        _, intra_knn_idx = self.knn_point(p1_t, p1_t, occ_mask_b)
     
        pos1_grouped = pointutils.grouping_operation(p1, intra_knn_idx)
        pos_diff = pos1_grouped - p1.view(B, -1, N, 1)    # [B,3,N1,S]
        groupped_feats = pointutils.grouping_operation(feat1, intra_knn_idx)
        feat1_knn = feat1.unsqueeze(-1).repeat(1, 1, 1, self.nsample) #[B,C,N,S]
        feat_new = torch.cat([pos_diff, groupped_feats], dim=1) # (109, 16, 259) # (m, nsample, 3+c)
        
        for conv in self.mlp1_convs:
            feat_new = conv(feat_new)
        # max pooling
        feat_diff = feat_new - feat1_knn
        # feat_new = feat_new.view(B,-1,N,1)
        for conv in self.mlp2_convs:
            feat_diff = conv(feat_diff)
        
        feat_new = feat_new + feat_diff
        
        corr =  self.calculate_corr(feat1_knn.permute(0,2,3,1).contiguous(), feat_new.permute(0,2,1,3).contiguous())
        corr = torch.sum(corr, dim=-1, keepdim=True)
        
        # sf_grouped = sf[knn_idx_flatten, :].view(len(p1), self.nsample, 3)
        sf_grouped = pointutils.grouping_operation(sf, intra_knn_idx)
        sf_grouped = (corr.permute(0,3,1,2).contiguous() * sf_grouped).sum(dim=-1)
        # occ_mask = occ_mask.contiguous().view(-1, 1).repeat(1,3) # (B*N, 3)
        # mask_groupped = mask[knn_idx_flatten, :]
        new_sf = sf * occ_mask_b.transpose(1,2).contiguous() + (~occ_mask_b).transpose(1,2).contiguous() * sf_grouped
        
        return new_sf
        

'''
Ref: Voxel Set Transformer
'''
class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class VoxelFeatureExtraction(nn.Module):
    def __init__(self, num_latents, input_dim, output_dim, num_point_features, base_scale, voxel_size, point_cloud_range, grid_size, **kwargs):
        super().__init__()
        self.num_latents = num_latents
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.base_scale = base_scale
        self.resolution = voxel_size

        self.input_embed = MLP(num_point_features, 16, self.input_dim, 2)
        self.pe = PositionalEncodingFourier(64, self.input_dim)
        self.mlp_vsa_layer = MLP_VSA_Layer(self.input_dim * 1, self.num_latents[0])

        self.post_mlp = nn.Sequential(
            nn.Linear(self.input_dim * 16, self.output_dim),
            nn.BatchNorm1d(self.output_dim, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Linear(self.output_dim, self.output_dim),
            nn.BatchNorm1d(self.output_dim, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Linear(self.output_dim, self.output_dim),
            nn.BatchNorm1d(self.output_dim, eps=1e-3, momentum=0.01)
        )

        # self.register_buffer('point_cloud_range', torch.FloatTensor(point_cloud_range).view(1, -1))
        # self.register_buffer('voxel_size', torch.FloatTensor(voxel_size).view(1, -1))
        # self.grid_size = grid_size.tolist()
        
        # a, b, c = voxel_size
        # self.register_buffer('voxel_size_02x', torch.FloatTensor([a * 2, b * 2, c]).view(1, -1))
        # self.register_buffer('voxel_size_04x', torch.FloatTensor([a * 4, b * 4, c]).view(1, -1))
        # self.register_buffer('voxel_size_08x', torch.FloatTensor([a * 8, b * 8, c]).view(1, -1))

        # a, b, c = grid_size
        # self.grid_size_02x = [a // 2, b //  2, c]
        # self.grid_size_04x = [a // 4, b //  4, c]
        # self.grid_size_08x = [a // 8, b //  8, c]
    
    def forward(self, pos1, pos2, feats1, feats2):
        """
            Feature propagation from xyz2 (less points) to xyz1 (more points)
        Inputs:
            xyz1: (batch_size, 3, npoint1)
            xyz2: (batch_size, 3, npoint2)
            feat1: (batch_size, channel1, npoint1) features for xyz1 points (earlier layers, more points)
            feat2: (batch_size, channel1, npoint2) features for xyz2 points
        Output:
            feat1_new: (batch_size, npoint2, mlp[-1] or mlp2[-1] or channel1+3)

            TODO: Add support for skip links. Study how delta(XYZ) plays a role in feature updating.
        """
        pos1_t = pos1.permute(0, 2, 1).contiguous()
        pos2_t = pos2.permute(0, 2, 1).contiguous()
        B,C,N = pos1.shape
        if self.knn:
            _, idx = pointutils.knn(self.nsample, pos1_t, pos2_t)
        else:
            idx, _ = query_ball_point(self.radius, self.nsample, pos2_t, pos1_t)
        
        pos2_grouped = pointutils.grouping_operation(pos2, idx)
        pos_diff = pos2_grouped - pos1.view(B, -1, N, 1)  # [B,3,N1,S]

        corr_feature = []
        for i in range(self.num_levels):
            with torch.no_grad():
                r = self.base_scale * (2 ** i)
                dis_voxel = torch.round(pos_diff / r)
                valid_scatter = (torch.abs(dis_voxel) <= np.floor(self.resolution / 2)).all(dim=1)
                dis_voxel = dis_voxel - (-1) # FROM 0 TO X
                cube_idx = dis_voxel[:, 0, :, :] * (self.resolution ** 2) +\
                    dis_voxel[:, 1, :, :] * self.resolution + dis_voxel[:, 2, :, :] #X * R^2:表示每层包含多少个cube；Y*R:表示第几层;Z表示指定层的第几个cube
                cube_idx_scatter = cube_idx.type(torch.int64) * valid_scatter

                valid_scatter = valid_scatter.detach()
                cube_idx_scatter = cube_idx_scatter.detach()



        pos_vox = pos1.clone()
        pos_vox = pos_diff // self.voxel_size
        pe_raw = (pos_diff - pos_vox * self.voxel_size) / self.voxel_size
        pos_vox, inv_pos_vox = torch.unique(pos_vox, return_inverse=True, dim=0)

        new_feats = pos2_grouped + self.pe(pe_raw)
        new_feats = self.mlp_vsa_layer(new_feats, inv_pos_vox, pos_vox, self.grid_size)
        new_feats = self.post_mlp(new_feats)

       
class MLP_VSA_Layer(nn.Module):
    def __init__(self, dim, n_latents=8):
        super(MLP_VSA_Layer, self).__init__()
        self.dim = dim
        self.k = n_latents 
        self.pre_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim,eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim,eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim,eps=1e-3, momentum=0.01),
        )

        # the learnable latent codes can be obsorbed by the linear projection
        self.score = nn.Linear(dim, n_latents)

        conv_dim = dim * self.k
        self.conv_dim = conv_dim
        
        # conv ffn
        self.conv_ffn = nn.Sequential(           
            nn.Conv2d(conv_dim, conv_dim, 3, 1, 1, groups=conv_dim, bias=False), 
            nn.BatchNorm2d(conv_dim),
            nn.ReLU(),
            nn.Conv2d(conv_dim, conv_dim, 3, 1, 1, groups=conv_dim, bias=False), 
            nn.BatchNorm2d(conv_dim),
            nn.ReLU(),
            # nn.Conv2d(conv_dim, conv_dim, 3, 1, dilation=2, padding=2, groups=conv_dim, bias=False),
            # nn.BatchNorm2d(conv_dim),
            # nn.ReLU(), 
            nn.Conv2d(conv_dim, conv_dim, 1, 1, bias=False), 
         ) 
        
        # decoder
        self.norm = nn.BatchNorm1d(dim,eps=1e-3, momentum=0.01)
        self.mhsa = nn.MultiheadAttention(dim, num_heads=1, batch_first=True) 
        
    def forward(self, inp, inverse, coords, bev_shape):
        
        x = self.pre_mlp(inp)

        # encoder
        attn = scatter_softmax(self.score(x), inverse, dim=0)
        dot = (attn[:, :, None] * x.view(-1, 1, self.dim)).view(-1, self.dim*self.k)
        x_ = scatter_sum(dot, inverse, dim=0)

        # conv ffn
        batch_size = int(coords[:, 0].max() + 1)
        h = spconv.SparseConvTensor(F.relu(x_), coords.int(), bev_shape, batch_size).dense().squeeze(-1)
        h = self.conv_ffn(h).permute(0,2,3,1).contiguous().view(-1, self.conv_dim)
        flatten_indices = coords[:, 0] * bev_shape[0] * bev_shape[1] + coords[:, 1] * bev_shape[1] + coords[:, 2]
        h = h[flatten_indices.long(), :] 
        h = h[inverse, :]
       
        # decoder
        hs = self.norm(h.view(-1,  self.dim)).view(-1, self.k, self.dim)
        hs = self.mhsa(x.view(-1, 1, self.dim), hs, hs)[0]
        hs = hs.view(-1, self.dim)
        
        # skip connection
        return torch.cat([inp, hs], dim=-1)


class PositionalEncodingFourier(nn.Module):
    """
    Positional encoding relying on a fourier kernel matching the one used in the
    "Attention is all of Need" paper. The implementation builds on DeTR code
    https://github.com/facebookresearch/detr/blob/master/models/position_encoding.py
    """
    def __init__(self, hidden_dim=64, dim=128, temperature=10000):
        super().__init__()
        self.token_projection = nn.Linear(hidden_dim * 3, dim)
        self.scale = 2 * math.pi
        self.temperature = temperature
        self.hidden_dim = hidden_dim
        

    def forward(self, pos_embed, max_len=(1, 1, 1)):
        z_embed, y_embed, x_embed = pos_embed.chunk(3, 1)
        z_max, y_max, x_max = max_len
        
        eps = 1e-6
        z_embed = z_embed / (z_max + eps) * self.scale
        y_embed = y_embed / (y_max + eps) * self.scale
        x_embed = x_embed / (x_max + eps) * self.scale

        dim_t = torch.arange(self.hidden_dim, dtype=torch.float32, device=pos_embed.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.hidden_dim)

        pos_x = x_embed / dim_t
        pos_y = y_embed / dim_t
        pos_z = z_embed / dim_t

        pos_x = torch.stack((pos_x[:, 0::2].sin(),
                             pos_x[:, 1::2].cos()), dim=2).flatten(1)
        pos_y = torch.stack((pos_y[:, 0::2].sin(),
                             pos_y[:, 1::2].cos()), dim=2).flatten(1)
        pos_z = torch.stack((pos_z[:, 0::2].sin(),
                             pos_z[:, 1::2].cos()), dim=2).flatten(1)

        pos = torch.cat((pos_z, pos_y, pos_x), dim=1)

        pos = self.token_projection(pos)
        return pos  


def po_from_batched_pcd(pcd):
        # x.shape: (B, 3, N)
        B, C, N = pcd.shape
        assert C == 3
        p = pcd.transpose(1, 2).contiguous().view(-1, 3) # (B*N, 3)
        o = torch.IntTensor([N * i for i in range(1, B + 1)]).to(p.device) # (N, 2N, ..)
        return (p, o)


class WeightNet(nn.Module):
    def __init__(self, in_channel, out_channel, hidden_unit = [8, 8], bn = use_bn):
        super(WeightNet, self).__init__()

        self.bn = bn
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        if hidden_unit is None or len(hidden_unit) == 0:
            self.mlp_convs.append(nn.Conv2d(in_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
        else:
            self.mlp_convs.append(nn.Conv2d(in_channel, hidden_unit[0], 1))
            self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[0]))
            for i in range(1, len(hidden_unit)):
                self.mlp_convs.append(nn.Conv2d(hidden_unit[i - 1], hidden_unit[i], 1))
                self.mlp_bns.append(nn.BatchNorm2d(hidden_unit[i]))
            self.mlp_convs.append(nn.Conv2d(hidden_unit[-1], out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
        
    def forward(self, localized_xyz):
        #xyz : BxCxKxN
        weights = localized_xyz
        for i, conv in enumerate(self.mlp_convs):
            if self.bn:
                bn = self.mlp_bns[i]
                weights =  F.relu(bn(conv(weights)))
            else:
                weights = F.relu(conv(weights))

        return weights


class PointConvFlow(nn.Module):
    def __init__(self, nsample, in_channel, mlp, bn = use_bn, use_leaky = True):
        super(PointConvFlow, self).__init__()
        self.nsample = nsample
        self.bn = bn
        self.mlp_convs = nn.ModuleList()
        if bn:
            self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            if bn:
                self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

        self.weightnet1 = WeightNet(3, last_channel)
        self.weightnet2 = WeightNet(3, last_channel)

        self.relu = nn.ReLU(inplace=True) if not use_leaky else nn.LeakyReLU(LEAKY_RATE, inplace=True)

        # self.queryandgroup = pointutils.QueryAndGroup(radius, nsample)
        
    def forward(self, xyz1, xyz2, points1, points2, occ_mask=None):
        """
        Cost Volume layer for Flow Estimation
        Input:
            xyz1: input points position data, [B, C, N1]
            xyz2: input points position data, [B, C, N2]
            points1: input points data, [B, D, N1]
            points2: input points data, [B, D, N2]
        Return:
            new_points: upsample points feature data, [B, D', N1]
        """
        
        # import ipdb; ipdb.set_trace()
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        _, D1, _ = points1.shape
        _, D2, _ = points2.shape
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)
        points1 = points1.permute(0, 2, 1)
        points2 = points2.permute(0, 2, 1)

        # point-to-patch Volume
        knn_idx = knn_point(self.nsample, xyz2, xyz1)[1] # B, N1, nsample
        neighbor_xyz = index_points_group(xyz2, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)

        grouped_points2 = index_points_group(points2, knn_idx) # B, N1, nsample, D2
        grouped_points1 = points1.view(B, N1, 1, D1).repeat(1, 1, self.nsample, 1)
        new_points = torch.cat([grouped_points1, grouped_points2, direction_xyz], dim = -1) # B, N1, nsample, D1+D2+3
        new_points = new_points.permute(0, 3, 2, 1) # [B, D1+D2+3, nsample, N1]
        for i, conv in enumerate(self.mlp_convs):
            if self.bn:
                bn = self.mlp_bns[i]
                new_points =  self.relu(bn(conv(new_points)))
            else:
                new_points =  self.relu(conv(new_points))

        # weighted sum
        weights = self.weightnet1(direction_xyz.permute(0, 3, 2, 1)) # B C nsample N1 

        point_to_patch_cost = torch.sum(weights * new_points, dim = 2) # B C N

        # Patch to Patch Cost
        knn_idx = knn_point(self.nsample, xyz1, xyz1)[1] # B, N1, nsample
        neighbor_xyz = index_points_group(xyz1, knn_idx)
        direction_xyz = neighbor_xyz - xyz1.view(B, N1, 1, C)

        # weights for group cost
        weights = self.weightnet2(direction_xyz.permute(0, 3, 2, 1)) # B C nsample N1 
        grouped_point_to_patch_cost = index_points_group(point_to_patch_cost.permute(0, 2, 1), knn_idx) # B, N1, nsample, C
        # patch_to_patch_cost = torch.sum(weights * grouped_point_to_patch_cost.permute(0, 3, 2, 1), dim = 2) # B C N

        if occ_mask != None:
            # mask = mask > 0.5
            mask_groupped = index_points_group(occ_mask.reshape(B,N1,1).contiguous(), knn_idx) # B, N1, nsample, 1
            mask_groupped = mask_groupped.repeat(1,1,1,weights.shape[1]).permute(0,3,2,1).contiguous()
            patch_to_patch_cost = torch.sum(mask_groupped * weights * grouped_point_to_patch_cost.permute(0, 3, 2, 1), dim = 2) # B C N
        else:
            patch_to_patch_cost = torch.sum(weights * grouped_point_to_patch_cost.permute(0, 3, 2, 1), dim = 2) # B C N
        
        return patch_to_patch_cost



if __name__ == "__main__":
    print('xxxx')
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    input = torch.randn((8,3,4096)).cuda()
    input2 = torch.randn((8,3,4096)).cuda()
    spare_input = torch.randn((8,3,4096)).cuda()
    sf_label = torch.randn((8,3,4096)).cuda()
    occ_label = torch.randn((8,4096,1)).cuda()
 
    

    # model = UpsampleSetLayer(nsample=16, num_levels=3, voxel_size=0.25, resolution=3.0, in_channel=3, hidden_channel=4, out_channel=16, use_voxel_mixer=True).cuda()
    # model = PointSetLayer(nsample=16, num_levels=3, voxel_size=0.25, resolution=3.0, in_channel=3, hidden_channel=4, out_channel=16, use_voxel_mixer=True).cuda()
    # model = CostVolumeNet(nsample=16, in_channel=3, mid_channel=8, share_channel=2, out_channel=16).cuda()
    # model = OcclusisonEstiamtionNet(nsample=16, voxel_size=0.25, resolution=3.0, in_channel=3, num_levels=2, out_channel=3).cuda()
    model = SceneFlowFusionNet(nsample=5, f1_channel=3, f2_channel=3, mlp=[6,3], mlp2=[6,3], occ_threshold=0.5).cuda()
    
    
    p0, o0 = po_from_batched_pcd(input)
    x0 = p0
    sp0, so0 = po_from_batched_pcd(spare_input)
    sx0 = sp0
    p1, o1 = po_from_batched_pcd(input2)
    x1 = p1
    # output = model([p0, x0, o0], [p1, x1, o1], [sp0, sx0, so0], spare_input)
    # output = model([p0, x0, o0], [p1, x1, o1], [p0, x0, o0])
    # output = model([[p0, x0, o0], [sp0, sx0, so0]], [[p0, x0, o0], [sp0, sx0, so0]], [p0, x0, o0])
    # output = model([[p0, x0, o0], [sp0, sx0, so0]], [[p0, x0, o0], [sp0, sx0, so0]], [p0, x0, o0])
     
    output = model([p0, x0, o0], [sp0, sx0, so0], 8, occ_label)
    print(output.size())
