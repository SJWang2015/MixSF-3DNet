#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os
# os.environ['CUDA_VISIBLE_DEVCICES'] = '2,3'
import argparse
import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, StepLR 
from data import SceneflowDataset
from datasets.generic import Batch

from model_pwc_uni import OccAwareNet, multiScaleLoss
model_name = 'model_pwc_uni.py'
import numpy as np 
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
from time import time 

use_rgb = True

class IOStream:
    def __init__(self, path):
        self.f = open(path, 'a')


    def cprint(self, text):
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def _init_(args):
    if args.rm_history and not args.eval:
        print("Remove history files ...")
        if os.path.exists(args.model_dir + args.exp_name):
            os.system('rm -r ' + args.model_dir + args.exp_name)
        print("Create files ...")
        if not os.path.exists(args.model_dir):
            os.makedirs(args.model_dir)

        if not os.path.exists(args.model_dir + args.exp_name):
            os.makedirs(args.model_dir + args.exp_name)

        if not os.path.exists(args.model_dir +  args.exp_name + '/model'):
            os.makedirs(args.model_dir + args.exp_name + '/model')

        os.system('cp main.py ./' + args.model_dir + args.exp_name + '/main.py.bkp')
        os.system('cp ' + model_name +' ./' + args.model_dir + args.exp_name + '/model.py.bkp')
        os.system('cp util.py ./' + args.model_dir + args.exp_name + '/util.py.bkp')
        os.system('cp data.py ./' + args.model_dir + args.exp_name + '/data.py.bkp')


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.kaiming_normal_(m.weight.data)
    if classname.find('Conv1d') != -1:
        nn.init.kaiming_normal_(m.weight.data)


def scene_flow_metric(pred, labels, mask=None):
    # mask = mask.cpu().numpy()
    error = np.sqrt(np.sum((pred - labels)**2, 2) + 1e-20)

    gtflow_len = np.sqrt(np.sum(labels*labels, 2) + 1e-20) # B,N
    acc1 = np.sum(np.logical_or((error <= 0.05), (error/gtflow_len <= 0.05)), axis=1)
    acc2 = np.sum(np.logical_or((error <= 0.1), (error/gtflow_len <= 0.1)), axis=1)

    mask_sum = np.sum(mask, 1)
    N = pred.shape[1]
    acc1 = np.sum(acc1[mask_sum > 0]) / np.sum(mask_sum[mask_sum > 0])
    # acc1 = np.mean(acc1) #/ N
    acc2 = np.sum(acc2[mask_sum > 0]) / np.sum(mask_sum[mask_sum > 0])
    # acc2 = np.sum(np.mean(acc2) #/ N

    EPE = np.sum(error,1) / N
    EPE = np.sum(EPE[mask_sum > 0]) / np.sum(mask_sum[mask_sum > 0])
    # EPE = np.mean(EPE)
    return EPE, acc1, acc2


def scene_flow_metric_full(pred, labels, mask=None):
    error = np.sqrt(np.sum((pred - labels)**2, 2) + 1e-20)

    gtflow_len = np.sqrt(np.sum(labels*labels, 2) + 1e-20) # B,N
    acc1 = np.sum(np.logical_or((error <= 0.05)*mask, (error/gtflow_len <= 0.05)*mask), axis=1)
    acc2 = np.sum(np.logical_or((error <= 0.1)*mask, (error/gtflow_len <= 0.1)*mask), axis=1)

    mask_sum = np.sum(mask, 1)
    acc1 = acc1[mask_sum > 0] / mask_sum[mask_sum > 0]
    acc1 = np.mean(acc1)
    acc2 = acc2[mask_sum > 0] / mask_sum[mask_sum > 0]
    acc2 = np.mean(acc2)

    EPE = np.sum(error * mask, 1)[mask_sum > 0] / mask_sum[mask_sum > 0]
    EPE = np.mean(EPE)
    if acc2 >= 0.999:
        outlier = np.sum(np.logical_and((error > 0.3) * mask, mask), axis=1)
    else:
        # outlier = np.sum(np.logical_or((err > 0.3)*mask, (err/gtflow_len > 0.1)*mask), axis=1)
        outlier = np.sum(np.logical_or((error > 0.3) * mask, (error > 0.1) * mask), axis=1)
    outlier = outlier[mask_sum > 0] / mask_sum[mask_sum > 0]
    outlier = np.mean(outlier)

    return EPE, acc1, acc2, outlier


@torch.no_grad()
def test_one_epoch(args, net, test_loader, rf_flow_factor, flow_factor, occ_factor, epoch=0):
    net.eval()

    total_loss = 0
    total_flow_loss = 0
    total_rf_flow_loss = 0
    total_occ_loss = 0
    total_epe = 0
    total_acc3d = 0
    total_acc3d_2 = 0
    total_outlier = 0
    np_epe_3d=[]
    np_acc_3d=[]
    np_acc_3d_2=[]
    np_outlier=[]
    rf_total_epe = 0
    rf_total_acc3d = 0
    rf_total_acc3d_2 = 0
    rf_total_outlier = 0
    rf_np_epe_3d=[]
    rf_np_acc_3d=[]
    rf_np_acc_3d_2=[]
    rf_np_outlier=[]
    num_examples = 0
    
    use_savefile = False
    with tqdm(enumerate(test_loader), total=len(test_loader), smoothing=0.9)as pbar:
        # for i, data in tqdm(enumerate(test_loader), total = len(test_loader)):
        for i, data in enumerate(test_loader):
            pc1 = data['sequence'][0]
            pc2 = data['sequence'][1]
            if use_rgb:
                feats1 = data['sequence'][0]
                feats2 = data['sequence'][1]
                feats1 = feats1.cuda().squeeze(1).transpose(2, 1).contiguous()
                feats2 = feats2.cuda().squeeze(1).transpose(2, 1).contiguous()
            mask1 = data['ground_truth'][0]
            flow = data['ground_truth'][1]
            pc1 = pc1.cuda().squeeze(1).transpose(2, 1).contiguous()
            pc2 = pc2.cuda().squeeze(1).transpose(2, 1).contiguous()
            # feat1 = feat1.transpose(2, 1).cuda().contiguous()
            # feat2 = feat2.transpose(2, 1).cuda().contiguous()
            flow = flow.cuda().squeeze(1).transpose(2, 1).contiguous()
            mask = mask1.squeeze(-1).cuda().contiguous()


            batch_size = pc1.size(0)
            num_examples += batch_size

            pred_flow_sum = torch.zeros(pc1.shape[0], pc1.shape[-1], 3).cuda()
            rf_pred_flow_sum = torch.zeros(pc1.shape[0], pc1.shape[-1], 3).cuda()
            pred_mask_sum = torch.zeros(pc1.shape[0], pc1.shape[-1], 3).cuda()
            repeat_num = 1
            if repeat_num > 1:
                for i in range(repeat_num):
                    # print(i)
                    perm = torch.randperm(pc1.shape[2])
                    points1_perm = pc1[:, :, perm]
                    points2_perm = pc2[:, :, perm]
                    if use_rgb:
                        feats1_perm = feats1[:, :, perm]
                        feats2_perm = feats2[:, :, perm]
                        pred_rf_flows, pred_flows, pred_mask, fps_pc1_idxs = net(points1_perm, points2_perm)
                    else:
                        pred_rf_flows, pred_flows, pred_mask, fps_pc1_idxs = net(points1_perm, points2_perm, feats1_perm, feats2_perm)

                    # forward
                    
                    pred_flow_sum[:, perm, :] += pred_flows[0]
                    rf_pred_flow_sum[:, perm, :] += pred_rf_flows[0]
                    pred_mask_sum[:, perm, :] += pred_mask[0]
            else:
                if use_rgb:
                    pred_rf_flows, pred_flows, pred_mask, fps_pc1_idxs = net(pc1, pc2)
                else:
                    pred_rf_flows, pred_flows, pred_mask, fps_pc1_idxs = net(pc1, pc2, feats1, feats2)
                # pred_flow_sum += pred_flows[0]
            
            pred_flow_sum /= repeat_num
            rf_pred_flow_sum /= repeat_num
            pred_mask_sum /= repeat_num

            if use_savefile:
                name_fmt = "{:0>6}".format(str(i)) + '.npz'
                np_src = pc1.permute(0, 2, 1).squeeze(0).cpu().numpy()
                np_tgt = pc2.permute(0, 2, 1).squeeze(0).cpu().numpy()
                gt = flow.squeeze(0).cpu().numpy()
                np_flow = pred_rf_flows[0].cpu().detach().squeeze(0).numpy()
                np_occ = pred_mask[0].cpu().detach().squeeze(0).numpy()
                np_occ2 = pred_mask[1].cpu().detach().squeeze(0).numpy()
                np_fps_idx1 = fps_pc1_idxs[0].cpu().detach().squeeze(0).numpy()
                np_mask = mask.cpu().detach().squeeze(0).numpy()
                np.savez('./results_kitti/' + name_fmt, pos1=np_src, pos2=np_tgt, flow=np_flow, gt=gt, occ_l0=np_occ, occ_l1=np_occ2, occ_gt=np_mask, idx= np_fps_idx1)


            # flow_pred = flow_pred * (1-bg_flag) + ego_flow * bg_flag 

            rf_flow_loss, flow_loss, occ_loss, occ_acc = multiScaleLoss(pred_rf_flows, pred_flows, flow, pred_mask, mask, fps_pc1_idxs)
            loss =  flow_factor * flow_loss + occ_factor * occ_loss

            if repeat_num > 1:
                rf_epe_3d, rf_acc_3d, rf_acc_3d_2, rf_outlier = scene_flow_metric_full(rf_pred_flow_sum.detach().cpu().numpy(), flow.transpose(2,1).contiguous().detach().cpu().numpy(), mask.detach().cpu().numpy())
                epe_3d, acc_3d, acc_3d_2, outlier = scene_flow_metric_full(pred_flow_sum.detach().cpu().numpy(), flow.transpose(2,1).contiguous().detach().cpu().numpy(), mask.detach().cpu().numpy())
            else:
                rf_epe_3d, rf_acc_3d, rf_acc_3d_2, rf_outlier = scene_flow_metric_full(pred_rf_flows[0].detach().cpu().numpy(), flow.transpose(2,1).contiguous().detach().cpu().numpy(), mask.detach().cpu().numpy())
                epe_3d, acc_3d, acc_3d_2, outlier = scene_flow_metric_full(pred_flows[0].detach().cpu().numpy(), flow.transpose(2,1).contiguous().detach().cpu().numpy(), mask.detach().cpu().numpy())
            rf_np_epe_3d.append(rf_epe_3d)
            rf_np_acc_3d.append(rf_acc_3d)
            rf_np_acc_3d_2.append(rf_acc_3d_2)
            rf_np_outlier.append(rf_outlier)
            rf_total_epe += rf_epe_3d * batch_size
            rf_total_acc3d += rf_acc_3d * batch_size
            rf_total_acc3d_2 += rf_acc_3d_2*batch_size
            rf_total_outlier += rf_outlier*batch_size
            
            np_epe_3d.append(epe_3d)
            np_acc_3d.append(acc_3d)
            np_acc_3d_2.append(acc_3d_2)
            np_outlier.append(outlier)
            total_epe += epe_3d * batch_size
            total_acc3d += acc_3d * batch_size
            total_acc3d_2 += acc_3d_2*batch_size
            total_outlier += outlier*batch_size
            # print('batch EPE 3D: %f\tACC 3D: %f\tACC 3D 2: %f' % (epe_3d, acc_3d, acc_3d_2))
            total_loss += loss.item() * batch_size
            pbar.set_postfix({blue('Loss'): '{0:1.5f}'.format(total_loss * 1.0 / num_examples)})  # 输入一个字典，显示实验指标
            pbar.update(1)
    # print("Total time:  %f\t; avg_time: %f\t" % (net.total_time, net.total_time/num_examples))
    boardio.add_scalar('Eval/EPE_3d', total_epe * 1.0 / num_examples, epoch)
    boardio.add_scalar('Eval/ACC_3d', total_acc3d * 1.0 / num_examples, epoch)
    boardio.add_scalar('Eval/ACC_3d_2', total_acc3d_2 * 1.0 / num_examples, epoch)
    boardio.add_scalar('Eval/Outlier', total_outlier * 1.0 / num_examples, epoch)
    boardio.add_scalar('Eval/Mean_loss', total_loss * 1.0 / num_examples, epoch)
    
    boardio.add_scalar('Eval/RF_EPE_3d', rf_total_epe * 1.0 / num_examples, epoch)
    boardio.add_scalar('Eval/RF_ACC_3d', rf_total_acc3d * 1.0 / num_examples, epoch)
    boardio.add_scalar('Eval/RF_ACC_3d_2', rf_total_acc3d_2 * 1.0 / num_examples, epoch)
    boardio.add_scalar('Eval/RF_Outlier', rf_total_outlier * 1.0 / num_examples, epoch)

    # np.savez('./results_active', epe_3d=np_epe_3d, acc_3d=np_acc_3d, acc_3d_2=np_acc_3d_2) 
    return total_loss * 1.0 / num_examples, total_epe * 1.0 / num_examples, total_acc3d * 1.0 / num_examples, total_acc3d_2 * 1.0 / num_examples, total_outlier * 1.0 / num_examples, rf_total_epe * 1.0 / num_examples, rf_total_acc3d * 1.0 / num_examples, rf_total_acc3d_2 * 1.0 / num_examples, rf_total_outlier * 1.0 / num_examples


def train_one_epoch(args, net, train_loader, opt, rf_flow_factor, flow_factor, occ_factor, epoch=0):
    net.train()
    num_examples = 0
    total_loss = 0
    total_flow_loss = 0
    total_rf_flow_loss = 0
    total_occ_loss = 0
    epoch_loss = 0.0
    occ_sum = 0.0
    data_size = len(train_loader)
    step = args.step

    with tqdm(enumerate(train_loader), total=len(train_loader)) as pbar:
        # for i, data in tqdm(enumerate(train_loader), total = len(train_loader)):
        for i, data in enumerate(train_loader):
            pc1 = data['sequence'][0]
            pc2 = data['sequence'][1]
            mask1 = data['ground_truth'][0]
            flow = data['ground_truth'][1]
            pc1 = pc1.cuda().squeeze(1).transpose(2, 1).contiguous()
            pc2 = pc2.cuda().squeeze(1).transpose(2, 1).contiguous()
            flow = flow.cuda().squeeze(1).transpose(2, 1).contiguous()
            mask = mask1.cuda().squeeze(-1).contiguous()

            if use_rgb:
                feats1 = data['sequence'][0]
                feats2 = data['sequence'][1]
                feats1 = feats1.cuda().squeeze(1).transpose(2, 1).contiguous()
                feats2 = feats2.cuda().squeeze(1).transpose(2, 1).contiguous()

            batch_size = pc1.size(0)
            opt.zero_grad()
            num_examples += batch_size
            if use_rgb:
                pred_rf_flows, pred_flows, pred_mask, fps_pc1_idxs = net(pc1, pc2)
            else:
                pred_rf_flows, pred_flows, pred_mask, fps_pc1_idxs = net(pc1, pc2, feats1, feats2)
            
            rf_flow_loss, flow_loss, occ_loss, occ_acc = multiScaleLoss(pred_rf_flows, pred_flows, flow, pred_mask, mask, fps_pc1_idxs)
            # loss = rf_flow_factor * rf_flow_loss + flow_factor * flow_loss + occ_factor * occ_loss
            loss = flow_factor * flow_loss + occ_factor * occ_loss
            # loss = torch.mean(mask1 * torch.sum((flow_pred - flow) ** 2, 1) / 2.0)
            loss.backward()

            opt.step()
            total_loss += loss.item() * batch_size
            total_rf_flow_loss += rf_flow_loss.item() * batch_size
            total_flow_loss += flow_loss.item() * batch_size
            total_occ_loss += occ_loss.item() * batch_size

            if (i + 1) % step == 0:
                epoch_cnt = epoch * np.round(data_size / step) + (i + 1) / step - 1
                # print("%s: %d, %s: %f" % (blue('Epoch'), epoch_cnt, blue('mean loss'), epoch_loss / step / batch_size))
                out_str = 'Train/Epoch' + str(args.step) + '_loss'
                # out_str = 'Train/Epoch100' + '_loss'
                boardio.add_scalar(out_str, epoch_loss / step / batch_size, epoch_cnt)
                epoch_loss = 0.0
            pbar.set_postfix({blue('Loss'): '{0:1.5f}'.format(total_loss * 1.0 / num_examples)})
            pbar.update(1)
            
            occ_sum += occ_acc
            
    # final_occ_acc = occ_sum / len(train_loader)
    total_rf_flow_loss = total_rf_flow_loss / num_examples
    total_flow_loss = total_flow_loss / num_examples
    total_occ_loss = total_occ_loss / num_examples
    
    boardio.add_scalar('Train/Total_Loss', total_loss * 1.0 / num_examples, epoch)
    boardio.add_scalar('Train/Flow_Loss', total_flow_loss, epoch)
    boardio.add_scalar('Train/Refine_Flow_Loss', total_rf_flow_loss, epoch)
    boardio.add_scalar('Train/Occlusion_Loss', total_occ_loss, epoch)
    for param_group in opt.param_groups:
        lr = float(param_group['lr'])
        break
    boardio.add_scalar('Train/learning_rate', lr, epoch)
    return total_loss * 1.0 / num_examples, total_rf_flow_loss, total_flow_loss, total_occ_loss


def test(args, net, test_loader, boardio, textio):
    #TODO: ADD THE OCCLUSION PREDICTION RATIO
    rf_flow_factor = 1.0
    flow_factor = 1.0
    occ_factor = 10.0
    test_loss, epe, acc, acc_2, outlier, rf_epe, rf_acc, rf_acc_2, rf_outlier = test_one_epoch(args, net, test_loader, rf_flow_factor, flow_factor, occ_factor)
    textio.cprint('==FINAL TEST')
    textio.cprint('Mean test loss: %f\tEPE 3D:%f\tACC 3D:%f\tACC 3D 2:%f\tOutlier:%f'%(test_loss, epe, acc, acc_2, outlier))
    textio.cprint('Mean refine test loss: %f\tRF_EPE 3D:%f\tRF_ACC 3D:%f\tRF_ACC 3D 2:%f\tRF_Outlier:%f'%(test_loss, rf_epe, rf_acc, rf_acc_2, rf_outlier))


def train(args, net, train_loader, test_loader, boardio, textio, device_ids):
    if args.use_sgd:
        print('Use SGD')
        opt = optim.SGD(net.parameters(), lr = args.lr * 100, momentum=args.momentum, weight_decay = 1e-4)
    else:
        opt = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    # scheduler = StepLR(opt, 16, gamma=args.momentum)
    scheduler = StepLR(opt, 40, gamma=args.momentum)
    # opt = nn.DataParallel(opt, device_ids=device_ids)

    best_test_loss = np.inf
    rf_flow_factor = 1.0
    flow_factor = 1.0
    for epoch in range(args.epochs):
        ## occlusion weight update
        if args.dataset == 'FlowNet3D':
            occ_factor = min(0.4, 0.3+epoch*0.001)
            if epoch >=50:
                occ_factor = 0.6
            if epoch >=75:
                occ_factor = 0.1
                occ_threshold = 0.70
            if epoch >=150:
                occ_threshold = 0.80
        else:
            occ_factor = 0.03
            occ_threshold = 0.90
        occ_factor = 10.0
            
        textio.cprint('==epoch: %d==' % ( epoch ))
        train_loss, train_rf_flow_loss, train_flow_loss, train_occ_loss = train_one_epoch(args, net, train_loader, opt, rf_flow_factor, flow_factor, occ_factor, epoch)
        # textio.cprint('mean train EPE loss: %f' % (train_loss))
        textio.cprint('Mean train loss: %f\tRf_flow_loss:%f\tFlow_loss:%f\tOcc_loss:%f'%(train_loss, train_rf_flow_loss, train_flow_loss, train_occ_loss))

        test_loss, epe, acc, acc_2, outlier, rf_epe, rf_acc, rf_acc_2, rf_outlier = test_one_epoch(args, net, test_loader, rf_flow_factor, flow_factor, occ_factor, epoch)
        textio.cprint('%s: %f\tEPE 3D: %f\tACC 3D: %f\tACC 3D 2: %f\tOutlier: %f' % (blue('mean test loss'), test_loss, epe, acc, acc_2, outlier))
        textio.cprint('%s: %f\tRF_EPE 3D:%f\tRF_ACC 3D:%f\tRF_ACC 3D 2:%f\tRF_Outlier:%f'% (blue('mean refine test loss'), test_loss, rf_epe, rf_acc, rf_acc_2, rf_outlier))

        if epe < best_test_loss:
            best_test_loss = epe
            # textio.cprint('best test loss untill now: %f' % best_test_loss)
            if torch.cuda.device_count() > 1:
                torch.save(net.module.state_dict(), 'checkpoints/%s/model/model.best.t7' % args.exp_name)
            else:
                torch.save(net.state_dict(), 'checkpoints/%s/model/model.best.t7' % args.exp_name)
        textio.cprint('best test loss untill now: %f' % best_test_loss)
        scheduler.step()

    
def main():
    parser = argparse.ArgumentParser(description='Occlusion Aware Scene Flow Estimation')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N', help='Name of the experiment')
    parser.add_argument('--model', type=str, default='OccAwareNet', metavar='N', choices=['OccAwareNet'],
                        help='Model to use, [OccAwareNet]')
    parser.add_argument('--num_points', type=int, default=4096,
                        help='Point Number [default: 2048]')
    parser.add_argument('--dropout', type=float, default=0.5, metavar='N',
                        help='Dropout ratio in transformer')
    parser.add_argument('--batch_size', type=int, default=8, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', action='store_true', default=False,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1234, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='evaluate the model')
    parser.add_argument('--unseen', type=bool, default=False, metavar='N',
                        help='Whether to test on unseen category')
    parser.add_argument('--dataset', type=str, default='HPLFlowNet', metavar='N',
                        help='Name of the dataset mode:[HPLFlowNet, FlowNet3D]')
    parser.add_argument('--dataset_cls', type=str, default='FT3D',
                        metavar='N', choices=['Kitti', 'FT3D'],
                        help='dataset to use: [Kitti, FT3D]')
    parser.add_argument('--dataset_path', type=str, default='/public_dataset_nas/flownet3d/data_processed_maxcut_35_20k_2k_8192/data_processed_maxcut_35_20k_2k_8192/', metavar='N',
                        help='dataset to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--model_dir', type=str, default='checkpoints/', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--n_workers', type=int, default=1, metavar='S',
                        help='the number of worker loaders (default: 1)')
    parser.add_argument('--random_dataset', action='store_true', default=False,
                        help='Whether to remove the history exp directories')
    parser.add_argument('--pretrained', action='store_true', default=False, 
                        help='load pretrained model for training')
    parser.add_argument('--rm_history', type=bool, default=False, metavar='N',
                        help='Whether to remove the history exp directories')
    parser.add_argument('--step', type=int, default=100, metavar='S',
                        help='the interval of tensorboard logs(default: 100)')

    args = parser.parse_args()

    # os.environ['CUDA_VISIBLE_DEVCICES'] = '0,2,3'
    device_ids = [0,1,2,3]
    # os.environ['CUDA_VISIBLE_DEVCICES'] = '1,2'
    # device_ids = [0,1]
    if args.eval:
        device_ids = [0]
    
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.n_workers)

    global blue 
    blue = lambda x: '\033[1;32m' + x + '\033[0m'
    global red
    red = lambda x: '\033[1;35m' + x + '\033[0m'
    global boardio
    _init_(args)
    if args.eval:
        file_dir = args.model_dir + args.exp_name + '/eval/'
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
    else:
        file_dir = args.model_dir + args.exp_name
    boardio = SummaryWriter(log_dir=file_dir)
    global textio

    textio = IOStream(file_dir + '/run.log')
    textio.cprint(str(args))

    if args.dataset == 'HPLFlowNet':
        if args.dataset_cls == 'FT3D':
            from datasets.flyingthings3d_hplflownet import FT3D
            args.dataset_path = '/dataset/public_dataset_nas/flownet3d/FlyingThings3D_subset_processed_35m'
            # lr_lambda = lambda epoch: 1.0 if epoch < 50 else 0.1
            # dataset = FT3D(root_dir=args.dataset_path, nb_points=args.n_points, mode="train")
            # test_dataset = FT3D(root_dir=args.dataset_path, nb_points=args.n_points, mode=mode)
        elif args.dataset_cls == 'Kitti':
            from datasets.kitti_hplflownet import Kitti
            args.dataset_path = '/home2/wangsj/Dataset/KITTI_processed_occ_final/'
            # dataset = Kitti(root_dir=args.dataset_path, nb_points=args.n_points, mode="train")
            # test_dataset = Kitti(root_dir=args.dataset_path, nb_points=args.n_points, mode=mode)
    elif args.dataset == 'FlowNet3D':
        if args.dataset_cls == 'FT3D':
            from datasets.flyingthings3d_flownet3d import FT3D
            args.dataset_path = '/dataset/public_dataset_nas/flownet3d/data_processed_maxcut_35_20k_2k_8192/data_processed_maxcut_35_20k_2k_8192'
            # lr_lambda = lambda epoch: 1.0 if epoch < 340 else 0.1
            # dataset = FT3D(root_dir=args.dataset_path, nb_points=args.n_points, mode="train")
            # test_dataset = FT3D(root_dir=args.dataset_path, nb_points=args.n_points, mode=mode)
        elif args.dataset_cls == 'Kitti':
            from datasets.kitti_flownet3d import Kitti
            args.dataset_path = '/home2/wangsj/Dataset/kitti_rm_ground'
    else:
        raise ValueError("Invalid dataset name: " + args.dataset)

    use_test = True
    mode = "test" if use_test else "val"
    assert mode == "val" or mode == "test", "Problem with mode " + mode
    if args.dataset_cls == 'FT3D':
        dataset = FT3D(root_dir=args.dataset_path, nb_points=args.num_points, mode="train")
        test_dataset = FT3D(root_dir=args.dataset_path, nb_points=args.num_points, mode=mode)
    elif args.dataset_cls == 'Kitti':
        dataset = Kitti(root_dir=args.dataset_path, nb_points=args.num_points, mode="train")
        test_dataset = Kitti(root_dir=args.dataset_path, nb_points=args.num_points, mode="val")

    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers, pin_memory=True,
        collate_fn=Batch, drop_last=True, timeout=0, persistent_workers=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers, pin_memory=True,
        collate_fn=Batch, drop_last=True, timeout=0, persistent_workers=True)

    
    if args.model == 'OccAwareNet':
        net = OccAwareNet()
        # net.apple(weight_init)

        net = nn.DataParallel(net, device_ids=device_ids)
        net = net.cuda()
        print("Let's use ", len(device_ids), " GPUs!")

        if args.eval:
            if args.model_path == '':
                model_path = 'checkpoints' + '/' + args.exp_name + '/model/model.best.t7'
            else:
                model_path = args.model_path
            print(model_path)
            if not os.path.exists(model_path):
                print("can't find pretrained model")
                return
            
            net_dict = net.state_dict()
            pretrained_dict = torch.load(model_path)

            
            for k, v in pretrained_dict.items():
                if 'module' != k[:6]:
                    name = 'module.' + k[:]  # add `module.`  
                else:
                    name = k
                #     name = k[7:] # remove `module.`
                net_dict[name] = v
            # for k, v in pretrained_dict.items():
            #     # name = 'module.' + k[:]  # remove `module.`
            #     name = 'module.' + k[:]  # remove `module.`
            #     net_dict[name] = v
            # net_dict = net.state_dict()
            # pretrained_dict = torch.load(model_path)
            #
            print('Update the neural network with the pretrained model.')
            net.load_state_dict(net_dict, strict=True)  
    else:
        raise Exception('Not implemented')
    if args.eval:
        test(args, net, test_loader, boardio, textio)
    else:
        if args.pretrained:
            net_dict = net.state_dict()
            if args.model_path == '':
                model_path = 'pretrained/model.best.t7'
                if len(device_ids) > 1:
                    pretrained_dict = torch.load(model_path)
                else:
                    pretrained_dict = torch.load(model_path)
                    # pretrained_dict = torch.load(model_path, map_location={'cuda:0': 'cuda:2'})
            else:
                pretrained_dict = torch.load(args.model_path + 'model.best.t7')

            net_dict = net.state_dict()
            # pretrained_dict = torch.load(model_path)
            #
            for k, v in pretrained_dict.items():
                # if 'sfnet' in  k[:]:
                #     continue
                name = 'module.' + k[:]  # remove `module.`
                net_dict[name] = v
            net.load_state_dict(net_dict, strict=False)
            print('Update the neural network with the pretrained model.')
        train(args, net, train_loader, test_loader, boardio, textio, device_ids)


    print('FINISH')
    # boardio.close()


if __name__ == '__main__':
    main()