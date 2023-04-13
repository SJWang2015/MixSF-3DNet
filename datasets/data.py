#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import glob
import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import minkowski

import torch


# Part of the code is referred from: https://github.com/charlesq34/pointnet

def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))


def load_data(partition):
    # download()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, '../kitti_rm_ground')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.05):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    return pointcloud


def farthest_subsample_points(pointcloud1, pointcloud2, num_subsampled_points=768):
    pointcloud1 = pointcloud1.T
    pointcloud2 = pointcloud2.T
    num_points = pointcloud1.shape[0]
    nbrs1 = NearestNeighbors(n_neighbors=num_subsampled_points, algorithm='auto',
                             metric=lambda x, y: minkowski(x, y), n_jobs=1).fit(pointcloud1)
    random_p1 = np.random.random(size=(1, 3)) + np.array([[500, 500, 500]]) * np.random.choice([1, -1, 1, -1])
    idx1 = nbrs1.kneighbors(random_p1, return_distance=False).reshape((num_subsampled_points,))
    nbrs2 = NearestNeighbors(n_neighbors=num_subsampled_points, algorithm='auto',
                             metric=lambda x, y: minkowski(x, y), n_jobs=1).fit(pointcloud2)
    random_p2 = random_p1  # np.random.random(size=(1, 3)) + np.array([[500, 500, 500]]) * np.random.choice([1, -1, 2, -2])
    idx2 = nbrs2.kneighbors(random_p2, return_distance=False).reshape((num_subsampled_points,))
    return pointcloud1[idx1, :].T, pointcloud2[idx2, :].T


class ModelNet40(Dataset):
    def __init__(self, num_points, num_subsampled_points=768, partition='train', gaussian_noise=False, unseen=False,
                 factor=4):
        self.data, self.label = load_data(partition)
        self.num_points = num_points
        self.partition = partition
        self.gaussian_noise = gaussian_noise
        self.unseen = unseen
        self.label = self.label.squeeze()
        self.factor = factor
        self.num_subsampled_points = num_subsampled_points
        if num_points != num_subsampled_points:
            self.subsampled = True
        else:
            self.subsampled = False
        if self.unseen:
            ######## simulate testing on first 20 categories while training on last 20 categories
            if self.partition == 'test':
                self.data = self.data[self.label >= 20]
                self.label = self.label[self.label >= 20]
            elif self.partition == 'train':
                self.data = self.data[self.label < 20]
                self.label = self.label[self.label < 20]

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        # if self.gaussian_noise:
        #     pointcloud = jitter_pointcloud(pointcloud)
        if self.partition != 'train':
            np.random.seed(item)
        anglex = np.random.uniform() * np.pi / self.factor
        angley = np.random.uniform() * np.pi / self.factor
        anglez = np.random.uniform() * np.pi / self.factor

        cosx = np.cos(anglex)
        cosy = np.cos(angley)
        cosz = np.cos(anglez)
        sinx = np.sin(anglex)
        siny = np.sin(angley)
        sinz = np.sin(anglez)
        Rx = np.array([[1, 0, 0],
                       [0, cosx, -sinx],
                       [0, sinx, cosx]])
        Ry = np.array([[cosy, 0, siny],
                       [0, 1, 0],
                       [-siny, 0, cosy]])
        Rz = np.array([[cosz, -sinz, 0],
                       [sinz, cosz, 0],
                       [0, 0, 1]])
        # 生成旋转矩阵
        R_ab = Rx.dot(Ry).dot(Rz)
        R_ba = R_ab.T
        # 生成平移向量
        translation_ab = np.array([np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5),
                                   np.random.uniform(-0.5, 0.5)])
        translation_ba = -R_ba.dot(translation_ab)

        pointcloud1 = pointcloud.T
        rotation_ab = Rotation.from_euler('zyx', [anglez, angley, anglex])
        pointcloud2 = rotation_ab.apply(pointcloud1.T).T + np.expand_dims(translation_ab, axis=1)

        euler_ab = np.asarray([anglez, angley, anglex])
        euler_ba = -euler_ab[::-1]

        pointcloud1 = np.random.permutation(pointcloud1.T).T
        pointcloud2 = np.random.permutation(pointcloud2.T).T

        if self.gaussian_noise:
            pointcloud1 = jitter_pointcloud(pointcloud1)
            pointcloud2 = jitter_pointcloud(pointcloud2)

        if self.subsampled:
            pointcloud1, pointcloud2 = farthest_subsample_points(pointcloud1, pointcloud2,
                                                                 num_subsampled_points=self.num_subsampled_points)

        return pointcloud1.astype('float32'), pointcloud2.astype('float32'), R_ab.astype('float32'), \
               translation_ab.astype('float32'), R_ba.astype('float32'), translation_ba.astype('float32'), \
               euler_ab.astype('float32'), euler_ba.astype('float32')

    def __len__(self):
        return self.data.shape[0]


class SceneflowDataset(Dataset):
    def __init__(self, npoints=2048, root='data/data_processed_maxcut_35_20k_2k_8192', train=True, augment=False,
                 remove_ground=False, cache=None, sampledata=True):
        self.npoints = npoints
        self.train = train
        self.augment = augment
        self.remove_ground = remove_ground
        self.root = root

        if self.train:
            self.datapath = glob.glob(os.path.join(self.root, 'TRAIN*.npz'))
            if sampledata:
                data_lens = len(self.datapath)
                sample_idx1 = np.random.choice(data_lens, data_lens // 2, replace=False)
                self.datapath = list(np.asarray(self.datapath)[sample_idx1])
            if self.augment:
                self.train_gt_T_path = glob.glob(os.path.join('./', 'TRAIN*.npz'))
                self.gt_T = (np.load(self.train_gt_T_path[0]))['gt'].astype('float32')
        else:
            self.datapath = glob.glob(os.path.join(self.root, 'TEST*.npz'))
            # if sampledata:
            #     data_lens = len(self.datapath)
            #     sample_idx1 = np.random.choice(data_lens, data_lens // 2, replace=False)
            #     self.datapath = list(np.asarray(self.datapath)[sample_idx1])
            if self.augment:
                self.test_gt_T_path = glob.glob(os.path.join('./', 'TEST*.npz'))
                self.gt_T = (np.load(self.test_gt_T_path[0]))['gt'].astype('float32')

        if cache is None:
            self.cache = {}
        else:
            self.cache = cache

        self.cache_size = 30000

        ###### deal with one bad datapoint with nan value
        self.datapath = [d for d in self.datapath if 'TRAIN_C_0140_left_0006-0' not in d]
        ######

    def __getitem__(self, index):
        if index in self.cache:
            pos1, pos2, color1, color2, flow, mask1 = self.cache[index]
        else:
            fn = self.datapath[index]
            with open(fn, 'rb') as fp:
                data = np.load(fp)
                pos1 = data['points1'].astype('float32')
                pos2 = data['points2'].astype('float32')
                color1 = data['color1'].astype('float32')
                color2 = data['color2'].astype('float32')
                # color1 = np.zeros_like(pos1)
                # color2 = np.zeros_like(pos2)
                if self.augment:
                    try:
                        if np.abs(np.linalg.norm(self.gt_T[index, :, :] - np.eye(4))) >= 1e-4:
                            if self.train:
                                pos2 = pos2 @ self.gt_T[index, :3, :3] + self.gt_T[index, :3, 2]
                            else:
                                pos2 = pos2 @ self.gt_T[index, :3, :3] + self.gt_T[index, :3, 2]
                    except:
                        print('index')

                if 'gt' in data.files:
                    flow = data['gt'].astype('float32')
                else:
                    flow = data['flow'].astype('float32')
                mask1 = data['valid_mask1']
                if self.remove_ground:
                    not_ground = np.logical_not(pos1[:, 1] < -1.4)
                    # print(np.min(pos1[:, 1]))
                    pos1 = pos1[not_ground]
                    flow = flow[not_ground]
                    not_ground = np.logical_not(pos2[:, 1] < -1.4)
                    # print(np.min(pos2[:, 1]))
                    pos2 = pos2[not_ground]
                # mask1 = np.ones_like(pos1)[:,0].astype('float32')

            if len(self.cache) < self.cache_size:
                if self.augment:
                    self.cache[index] = (pos1, pos2, color1, color2, flow, mask1, self.gt_T[index, :, :])
                else:
                    self.cache[index] = (pos1, pos2, color1, color2, flow, mask1)

        if self.train:
            n1 = pos1.shape[0]
            sample_idx1 = np.random.choice(n1, self.npoints, replace=False)
            n2 = pos2.shape[0]
            sample_idx2 = np.random.choice(n2, self.npoints, replace=False)

            pos1 = pos1[sample_idx1, :]
            pos2 = pos2[sample_idx2, :]
            color1 = color1[sample_idx1, :]
            color2 = color2[sample_idx2, :]
            flow = flow[sample_idx1, :]
            mask1 = mask1[sample_idx1]
        else:
            pos1 = pos1[:self.npoints, :]
            pos2 = pos2[:self.npoints, :]
            color1 = color1[:self.npoints, :]
            color2 = color2[:self.npoints, :]
            flow = flow[:self.npoints, :]
            mask1 = mask1[:self.npoints]

        pos1_center = np.mean(pos1, 0)
        pos1 -= pos1_center
        pos2 -= pos1_center

        pos1 = torch.from_numpy(pos1)
        pos2 = torch.from_numpy(pos2)
        color1 = torch.from_numpy(color1)
        color2 = torch.from_numpy(color2)
        flow = torch.from_numpy(flow)
        mask1 = torch.from_numpy(mask1)
        if self.augment:
            gt_T = torch.from_numpy(self.gt_T[index, :, :])
            return pos1, pos2, color1, color2, flow, mask1, gt_T
        else:
            return pos1, pos2, color1, color2, flow, mask1

    def __len__(self):
        return len(self.datapath)


if __name__ == '__main__':
    val_dir = '/sustech_points/Flyingthings3d/data_processed_maxcut_35_500_100_8192'
    val_dir = '/sustech_points/Flyingthings3d/FlyingThings3D_subset_processed_35m'
    train = SceneflowDataset(npoints=4096, root=val_dir, train=True, augment=True)
    for data in train:
        print(data[0].shape)
        break
