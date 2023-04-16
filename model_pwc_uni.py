import torch 
import torch.nn as nn 
import torch.nn.functional as F  
import numpy as np  
from util import * 
import time

scale = 1.0
# occ_threshold = 0.80
share_channel = 1
use_voxel_mixer=True
nn_upsample = True
pwc_cv = False
occ_net = False
share_planes = 1


class PointWarping(nn.Module):

    def forward(self, xyz1, xyz2, flow1 = None):
        if flow1 is None:
            return xyz2

        # move xyz1 to xyz2'
        xyz1_to_2 = xyz1 + flow1 

        # interpolate flow
        B, C, N1 = xyz1.shape
        _, _, N2 = xyz2.shape
        xyz1_to_2 = xyz1_to_2.permute(0, 2, 1) # B 3 N1
        xyz2 = xyz2.permute(0, 2, 1) # B 3 N2
        flow1 = flow1.permute(0, 2, 1)

        # 3 nearest neightbor & use 1/dist as the weights
        dist, knn_idx = knn_point(3, xyz1_to_2, xyz2) # group flow 1 around points 2
        dist = dist.clamp(min = 1e-10) 
        # grouped_xyz_norm = index_points_group(xyz1_to_2, knn_idx) - xyz2.view(B, N2, 1, C) # B N2 3 C
        # dist = torch.norm(grouped_xyz_norm, dim = 3).clamp(min = 1e-10) 
        norm = torch.sum(1.0 / dist, dim = 2, keepdim = True) 
        weight = (1.0 / dist) / norm 

        # from points 2 to group flow 1 and got weight, and use these weights and grouped flow to wrap a inverse flow and flow back
        grouped_flow1 = index_points_group(flow1, knn_idx)
        flow2 = torch.sum(weight.view(B, N2, 3, 1) * grouped_flow1, dim = 2)
        warped_xyz2 = (xyz2 - flow2).permute(0, 2, 1) # B 3 N2

        return warped_xyz2


class OccAwareNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa0 = nn.Sequential(Conv1d(3,32), Conv1d(32,32))
        self.sa1 = PointNetSetAbstraction(npoint=2048, radius=0.5, nsample=16, in_channel=32, mlp=[32,32,64], group_all=False)
        self.sa2 = PointNetSetAbstraction(npoint=512, radius=1.0, nsample=16, in_channel=64, mlp=[64,64,128], group_all=False)
        self.sa3 = PointNetSetAbstraction(npoint=256, radius=2.0, nsample=8, in_channel=128, mlp=[128,128,256], group_all=False)
        self.sa4 = PointNetSetAbstraction(npoint=128, radius=4.0, nsample=8, in_channel=256, mlp=[256,256,512], group_all=False)
        
        self.su_sa3 = PointNetSetUpConv(nsample=8, radius=2.4, f1_channel = 256, f2_channel = 512, mlp=[256, 256], mlp2=[256, 256])
        if pwc_cv:
            self.cv3 = PointConvFlow(8, 256 + 256 + 3, [256, 256])
        else:
            self.cv3 = SceneFlowRegressor(nsample=32, in_channel=256, sfeat_channel=256, sf_channel=0, mid_channel=256, share_channel=share_channel, out_channel=256, channels =[256], mlp=[256, 256])
        # self.sfnet3 = SceneFlowEstimatorPointConv(feat_ch=256, cost_ch=256, flow_ch=0, channels =[256, 256], mlp=[128, 128], share_planes=share_planes,neighbors=8, clamp=[-200, 200], use_leaky = True)
        # self.sfnet3 = SceneFlowEstimatorResidual(feat_ch=128+256, cost_ch=256, flow_ch=3, channels =[256, 256], mlp=[128, 128], neighbors=8, clamp=[-200, 200], use_leaky = True)
        
        self.su_sa2 = PointNetSetUpConv(nsample=16, radius=1.2, f1_channel = 128, f2_channel = 256+256, mlp=[128, 128], mlp2=[128, 128])
        # if nn_upsample:
        self.su_sf2 = SceneFlowUpsampleNet(nsample=5, num_levels=3, voxel_size=0.5, resolution=3.0, sf_channel=256, in_channel=128, hidden_channel=4, out_channel=128, use_voxel_mixer=use_voxel_mixer)
        self.cv2 = SceneFlowRegressor(nsample=16, in_channel=128, sfeat_channel=128, sf_channel=3, mid_channel=128, share_channel=share_channel, out_channel=128, channels=[128], mlp=[128, 128])
        
        self.su_sa1 = PointNetSetUpConv(nsample=16, radius=0.6, f1_channel = 64, f2_channel = 128+128, mlp=[64, 64], mlp2=[64, 64])
        
        self.su_sf1 = SceneFlowUpsampleNet(nsample=5, num_levels=3, voxel_size=0.25, resolution=3.0, sf_channel=128, in_channel=64, hidden_channel=4, out_channel=64, use_voxel_mixer=use_voxel_mixer)
        self.cv1 = SceneFlowRegressor(nsample=16, in_channel=64, sfeat_channel=64, sf_channel=3, mid_channel=64, share_channel=share_channel, out_channel=64, channels=[64], mlp=[64, 64])
        
        self.su_sa0 = PointNetSetUpConv(nsample=16, radius=0.6, f1_channel = 32, f2_channel = 64+64, mlp=[64, 64], mlp2=[64, 64])
       
        self.su_sf0 = SceneFlowUpsampleNet(nsample=8, num_levels=3, voxel_size=0.25, resolution=3.0, sf_channel=64, in_channel=64, hidden_channel=4, out_channel=64, use_voxel_mixer=use_voxel_mixer)
        
        self.cv0 = SceneFlowRegressor(nsample=16, in_channel=64, sfeat_channel=64, sf_channel=3, mid_channel=64, share_channel=share_channel, out_channel=64, channels=[64], mlp=[64, 64])
                                                neighbors=8, clamp=[-200, 200], use_leaky = True)

    
        self.upsample = UpsampleFlow()
        # self.deconv4_3 = Conv1d(256, 64)
        self.deconv3_2 = Conv1d(256, 64)
        self.deconv2_1 = Conv1d(128, 32)
        self.deconv1_0 = Conv1d(64, 32)
        #warping
        # self.warping = PointWarping()
        
        self.cv_time = 0.0
        self.su_time = 0.0
        self.occ_time = 0.0
        self.total_time = 0.0

    def po_from_batched_pcd(self, pcd):
        # x.shape: (B, 3, N)
        B, C, N = pcd.shape
        assert C == 3
        p = pcd.transpose(1, 2).contiguous().view(-1, 3) # (B*N, 3)
        o = torch.IntTensor([N * i for i in range(1, B + 1)]).to(p.device) # (N, 2N, ..)
        return (p, o)
    
    def get_downsample_num(self, o, stride):
        n_o, count = [o[0].item() // stride], o[0].item() // stride
        for i in range(1, o.shape[0]):
            count += (o[i].item() - o[i-1].item()) // stride
            n_o.append(count)
        n_o = torch.cuda.IntTensor(n_o)
        
        return n_o
    
    def get_downsample_pts(self, xyz, new_num=4096):
        xyz_t = xyz.permute(0, 2, 1).contiguous()
        fps_idx = pointutils.furthest_point_sample(xyz_t, new_num)  # [B, N]
        new_xyz = pointutils.gather_operation(xyz, fps_idx)  # [B, C, N]
        
        return new_xyz, fps_idx

    def forward(self, pc1, pc2, feats1=None, feats2=None, occ_threshold=0.5):        
        B, _, N = pc1.shape
        time_start = time.time()
        if N > 8192:
            n_pc1, fps_idx_l0 = self.get_downsample_pts(pc1)
            n_pc2, _ = self.get_downsample_pts(pc2)
            _, ind1 = self.po_from_batched_pcd(n_pc1)
            feat1 = n_pc1 
            _, ind2 = self.po_from_batched_pcd(n_pc2)
            feat2 = n_pc2
        else:  
            n_pc1 = pc1
            _, ind1 = self.po_from_batched_pcd(pc1)
        
            n_pc2 = pc2
            _, ind2 = self.po_from_batched_pcd(pc2)
      
            if feats1 == None or feats2 == None:
                feat1 = pc1 
                feat2 = pc2
                
        # pc1_l1, feat1_l1, ind1_l1, fps_idx_l1 = self.ds1([pc1, feat1, ind1])
        feat1_l0 = self.sa0(feat1)
        
        [pc1_l1, feat1_l1, fps_idx_l1] = self.sa1(n_pc1, feat1_l0)
        ind1_l1 = self.get_downsample_num(ind1, n_pc1.shape[-1] // pc1_l1.shape[-1])
        # pc1_l2, feat1_l2, ind1_l2, fps_idx_l2 = self.ds2([pc1_l1, feat1_l1, ind1_l1])
        [pc1_l2, feat1_l2, fps_idx_l2] = self.sa2(pc1_l1, feat1_l1)
        ind1_l2 = self.get_downsample_num(ind1_l1, pc1_l1.shape[-1] // pc1_l2.shape[-1])
        # pc1_l3, feat1_l3, ind1_l3, fps_idx_l3 = self.ds3([pc1_l2, feat1_l2, ind1_l2])
        [pc1_l3, feat1_l3, fps_idx_l3] = self.sa3(pc1_l2, feat1_l2)
        ind1_l3 = self.get_downsample_num(ind1_l2, pc1_l2.shape[-1] // pc1_l3.shape[-1])
        # pc1_l4, feat1_l4, ind1_l4, fps_idx_l4 = self.ds4([pc1_l3, feat1_l3, ind1_l3])
        [pc1_l4, feat1_l4, fps_idx_l4] = self.sa4(pc1_l3, feat1_l3)
        ind1_l4 = self.get_downsample_num(ind1_l3, pc1_l3.shape[-1] // pc1_l4.shape[-1])
        
        feat2_l0 = self.sa0(feat2)
        # pc2_l1, feat2_l1, ind2_l1, _ = self.ds1([pc2, feat2, ind2])
        [pc2_l1, feat2_l1, _] = self.sa1(n_pc2, feat2_l0)
        # pc2_l2, feat2_l2, ind2_l2, _ = self.ds2([pc2_l1, feat2_l1, ind2_l1])
        [pc2_l2, feat2_l2, _] = self.sa2(pc2_l1, feat2_l1)
        # pc2_l3, feat2_l3, ind2_l3, _ = self.ds3([pc2_l2, feat2_l2, ind2_l2])
        [pc2_l3, feat2_l3, _] = self.sa3(pc2_l2, feat2_l2)
        # pc2_l4, feat2_l4, ind2_l4, _ = self.ds4([pc2_l3, feat2_l3, ind2_l3])
        [pc2_l4, feat2_l4, _] = self.sa4(pc2_l3, feat2_l3)
        
        ind2 = ind1
        ind2_l1 = ind1_l1
        ind2_l2 = ind1_l2
        ind2_l3 = ind1_l3
        ind2_l4 = ind1_l4
        

        up_feat1_l3 = self.su_sa3(pc1_l3, pc1_l4, feat1_l3, feat1_l4)
        up_feat2_l3 = self.su_sa3(pc2_l3, pc2_l4, feat2_l3, feat2_l4)
 
        if pwc_cv:
            c_feat_fwd_l3 = self.cv3(pc1_l3, pc2_l3, up_feat1_l3, up_feat2_l3)
        else:
            c_feat_fwd_l3, c_feat_bwd_l3, flow_feat_l3, flow_l3  = self.cv3([pc1_l3, up_feat1_l3, ind1_l3], [pc2_l3, up_feat2_l3, ind2_l3], sf_feat=up_feat1_l3)
       
        up_feat1_l3_c = torch.cat([up_feat1_l3, c_feat_fwd_l3], dim=1)
        up_feat2_l3_c = torch.cat([up_feat2_l3, c_feat_bwd_l3], dim=1)
        up_feat1_l2 = self.su_sa2(pc1_l2, pc1_l3, feat1_l2, up_feat1_l3_c)
        up_feat2_l2 = self.su_sa2(pc2_l2, pc2_l3, feat2_l2, up_feat2_l3_c)
    
       
        if nn_upsample:
            up_flow_feat_l3_2, up_flow_l3_2 = self.su_sf2([pc1_l2, up_feat1_l2, ind1_l2], [pc2_l2, up_feat2_l2, ind2_l2], [pc1_l3, up_feat1_l3, ind1_l3], flow_l3, flow_feat_l3)
        else:
            up_flow_l3_2 = self.upsample(pc1_l2, pc1_l3, flow_l3)
                 
        if pwc_cv:
            c_feat_fwd_l2 = self.cv2(pc1_l2, pc2_l2, up_feat1_l2, up_feat2_l2)
        else:
            c_feat_fwd_l2, c_feat_bwd_l2, flow_feat_l2, flow_l2 = self.cv2([pc1_l2, up_feat1_l2, ind1_l2], [pc2_l2, up_feat2_l2, ind2_l2], up_flow_l3_2, up_flow_feat_l3_2)
        

        up_feat1_l2_c = torch.cat([up_feat1_l2, c_feat_fwd_l2], dim=1)
        up_feat2_l2_c = torch.cat([up_feat2_l2, c_feat_bwd_l2], dim=1)
        
        up_feat1_l1 = self.su_sa1(pc1_l1, pc1_l2, feat1_l1, up_feat1_l2_c)
        up_feat2_l1 = self.su_sa1(pc2_l1, pc2_l2, feat2_l1, up_feat2_l2_c)
        
        if nn_upsample:
            up_flow_feat_l2_1, up_flow_l2_1 = self.su_sf1([pc1_l1, up_feat1_l1, ind1_l1], [pc2_l1, up_feat2_l1, ind2_l1], [pc1_l2, up_feat1_l2, ind1_l2], flow_l2, flow_feat_l2)
        else:
            up_flow_l2_1 = self.upsample(pc1_l1, pc1_l2, flow_l2)
            
        if occ_net:
            # time_start=time.time()
            occ_mask_l1 = self.occnet1([[pc1_l3, up_feat1_l3, ind1_l3], [pc1_l2, up_feat1_l2, ind1_l2], [pc1_l1, up_feat1_l1, ind1_l1]], [[pc2_l3, up_feat2_l3, ind2_l3], [pc2_l2, up_feat2_l2, ind2_l2], [pc2_l1, up_feat2_l1, ind2_l1]], up_flow_l2_1)
            # time_end=time.time()
            # self.cv_time = self.cv_time + time_end - time_start
        else:
            occ_mask_l1 = torch.ones([pc1_l1.shape[0]*pc1_l1.shape[2],1]).to(pc1_l1.device)
        # pc2_l1_warp = self.warping(pc1_l1, pc2_l1, up_flow_l2_1)
            
        if pwc_cv:
            c_feat_fwd_l1 = self.cv1(pc1_l1, pc2_l1, up_feat1_l1, up_feat2_l1, occ_mask_l1)
        else:
            if occ_net:
                c_feat_fwd_l1, c_feat_bwd_l1, flow_feat_l1, flow_l1 = self.cv1([pc1_l1, up_feat1_l1, ind1_l1], [pc2_l1, up_feat2_l1, ind2_l1], up_flow_l2_1, up_flow_feat_l2_1,  mask=occ_mask_l1)
            else:
                c_feat_fwd_l1, c_feat_bwd_l1, flow_feat_l1, flow_l1 = self.cv1([pc1_l1, up_feat1_l1, ind1_l1], [pc2_l1, up_feat2_l1, ind2_l1], up_flow_l2_1, up_flow_feat_l2_1)
        
        up_feat1_l1_c = torch.cat([up_feat1_l1, c_feat_fwd_l1], dim=1)
        up_feat2_l1_c = torch.cat([up_feat2_l1, c_feat_bwd_l1], dim=1)
        
        up_feat1_l0 = self.su_sa0(n_pc1, pc1_l1, feat1_l0, up_feat1_l1_c)
        up_feat2_l0 = self.su_sa0(n_pc2, pc2_l1, feat2_l0, up_feat2_l1_c)
        
        if nn_upsample:
            # time_start=time.time()
            # _, up_flow_l1_0 = self.su_sf0([n_pc1, up_feat1_l0, ind1], [n_pc2, up_feat2_l0, ind2], [pc1_l1, up_feat1_l1, ind1_l1], rf_flow_l1, flow_feat_l1)
            up_flow_feat_l1_0, up_flow_l1_0 = self.su_sf0([n_pc1, up_feat1_l0, ind1], [n_pc2, up_feat2_l0, ind2], [pc1_l1, up_feat1_l1, ind1_l1], flow_l1, flow_feat_l1)
            # time_end=time.time()
            # self.su_time = self.su_time + time_end - time_start
        else:
            up_flow_l1_0 = self.upsample(n_pc1, pc1_l1, flow_l1)
        
        if occ_net:
            # time_start=time.time()
            occ_mask_l0 = self.occnet0([[pc1_l2, up_feat1_l2, ind1_l2], [pc1_l1, up_feat1_l1, ind1_l1], [n_pc1, up_feat1_l0, ind1]], [[pc2_l2, up_feat2_l2, ind2_l2], [pc2_l1, up_feat2_l1, ind2_l1], [n_pc2, up_feat2_l0, ind2]], up_flow_l1_0)
            # time_end=time.time()
            # self.occ_time = self.occ_time + time_end - time_start
        else:
            occ_mask_l0 = torch.ones([n_pc1.shape[0]*n_pc1.shape[2],1]).to(pc1_l1.device)
            
        if pwc_cv:
            c_feat_l0 = self.cv0(n_pc1, n_pc2, up_feat1_l0, up_feat2_l0, occ_mask_l0)
        else:
            if occ_net:
                _, _, _, flow_l0 = self.cv0([n_pc1, up_feat1_l0, ind1], [n_pc2, up_feat2_l0, ind2], up_flow_l1_0, up_flow_feat_l1_0, mask=occ_mask_l0)
            else:
                _, _, _, flow_l0 = self.cv0([n_pc1, up_feat1_l0, ind1], [n_pc2, up_feat2_l0, ind2], up_flow_l1_0, up_flow_feat_l1_0)
       
        time_end=time.time()
        self.total_time = self.total_time + time_end - time_start
        print(self.total_time)

        flow_l0 = flow_l0.transpose(1,2).contiguous()
        flow_l1 = flow_l1.transpose(1,2).contiguous()
        flow_l2 = flow_l2.transpose(1,2).contiguous()
        flow_l3 = flow_l3.transpose(1,2).contiguous()
        occ_mask_l0 = (occ_mask_l0.reshape(B, -1, 1).contiguous()).type(torch.float32)
        occ_mask_l1 = (occ_mask_l1.reshape(B, -1, 1).contiguous()).type(torch.float32)
        
        if N > 4096:
            return [flow_l0, flow_l1, flow_l2], [flow_l0, flow_l1, flow_l2, flow_l3], [occ_mask_l0, occ_mask_l1], [fps_idx_l1, fps_idx_l2, fps_idx_l3]
        else:
            # return [rf_flow_l0, rf_flow_l1], [flow_l0, flow_l1, flow_l2, flow_l3], [occ_mask_l0, occ_mask_l1], [fps_idx_l1, fps_idx_l2, fps_idx_l3]
            return [flow_l0, flow_l1], [flow_l0, flow_l1, flow_l2, flow_l3], [occ_mask_l0, occ_mask_l1], [fps_idx_l1, fps_idx_l2, fps_idx_l3]
        

def multiScaleLoss(pred_rf_flows, pred_flows, gt_flow, pred_occ_masks, gt_occ_masks, fps_idxs, alpha=[0.02, 0.04, 0.08, 0.16], beta=[0.02, 0.04], occ_threshold=0.5):
    # num of scale
    num_scale = len(pred_flows)
    offset = len(fps_idxs) - num_scale + 1
    # offset = 0
    occloss = nn.BCELoss()
    gt_flow = gt_flow.permute(0,2,1).contiguous()
    gt_occ_masks = gt_occ_masks.unsqueeze(-1).contiguous()
    # generate GT list and masks
    gt_flows = [gt_flow]
    gt_masks = [gt_occ_masks]
    for i in range(1, len(fps_idxs)+1):
        fps_idx = fps_idxs[i - 1]
        sub_gt_flow = index_points(gt_flows[-1], fps_idx) / scale
        sub_gt_mask = index_points(gt_masks[-1], fps_idx)
        gt_flows.append(sub_gt_flow)
        gt_masks.append(sub_gt_mask)

    occ_sum=0
    rf_flow_loss = torch.zeros(1).cuda()
    flow_loss = torch.zeros(1).cuda()
    occ_loss = torch.zeros(1).cuda()
    for i in range(num_scale):
        diff_flow = (pred_flows[i] - gt_flows[i + offset])
        # diff_mask = pred_occ_masks[i].permute(0, 2, 1) - gt_masks[i + offset]
        # occ_loss += 1.4*alpha[i] *torch.norm(diff_mask, dim=2).sum(dim=1).mean()
        flow_loss += alpha[i] * torch.norm(diff_flow*gt_masks[i + offset], dim=2).sum(dim=1).mean()

    for i in range(len(pred_occ_masks)):
        diff_rf_flow = (pred_rf_flows[i] - gt_flows[i + offset])
        # diff_mask = pred_occ_masks[i] - gt_masks[i + offset]

        # occ_loss += 1.4*alpha[i] * torch.norm(diff_mask, dim=2).sum(dim=1).mean()
        occ_loss += beta[i] * occloss(pred_occ_masks[i], gt_masks[i + offset])
        rf_flow_loss += alpha[i] * torch.norm(diff_rf_flow*gt_masks[i + offset], dim=2).sum(dim=1).mean()
        
    pred_occ_mask = pred_occ_masks[0] > occ_threshold
    occ_acc = torch.mean((pred_occ_mask.type(torch.float32) - gt_masks[0].type(torch.float32)) ** 2)
    occ_acc = 1.0 - occ_acc.item()
    occ_sum += occ_acc

    return rf_flow_loss, flow_loss, occ_loss, occ_sum


if __name__ == '__main__':
    import os
    from thop import profile
    from thop import clever_format
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    # os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    input = torch.randn((1,3,8192)).cuda()
    sf_label = torch.randn((1,3,8192)).cuda()
    occ_label = torch.randn((1,1,8192)).cuda()

    model = OccAwareNet().cuda()
    total_time =  0.0
    for i in range(10):
        output = model(input, input)
        total_time = total_time + model.total_time
    print(total_time/10.0)
    # macs, params = profile(model, inputs=(input,input,))
    # macs, params = clever_format([macs, params], "%.3f")
    # print(macs, params)
