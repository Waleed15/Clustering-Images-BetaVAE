import argparse
import os
import random as rn
import zarr
import pandas as pd
import sys
import cv2
import pickle as pk
import logging
import umap
import umap.plot
import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import torch
import torchvision
from torch.utils.data.dataloader import DataLoader

import hdbscan as hd
from sklearn.cluster import KMeans, SpectralClustering
from scipy.optimize import linear_sum_assignment as linear_assignment
from time import time
from sklearn.decomposition import PCA

from beta_vae import BetaVAE
from reconstruction import ReconstructionNet
from tqdm import tqdm

def get_parser():
    parser = argparse.ArgumentParser(
        description='Beta VAE',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--exp_name', type=str, default=None, metavar='N', help='Name of the experiment')
    parser.add_argument('--task', type=str, default='reconstruct', metavar='N',
                        choices=['reconstruct', 'classify'], help='Experiment task, [reconstruct, classify]')
    parser.add_argument('--encoder', type=str, default='dgcnn_cls', metavar='N', choices=['foldnet', 'dgcnn_cls', 'dgcnn_seg'], help='Encoder to use, [foldingnet, dgcnn_cls, dgcnn_seg]')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--feat_dims', type=int, default=512, metavar='N', help='Number of dims for feature ')
    parser.add_argument('--k', type=int, default=20, metavar='N', help='Num of nearest neighbors to use for KNN')
    parser.add_argument('--shape', type=str, default='sphere', metavar='N',
                        choices=['plane', 'sphere', 'gaussian'], help='Shape of points to input decoder, [plane, sphere, gaussian]')
    parser.add_argument('--dataset', type=str, default='shapenetcorev2', metavar='N',
                        choices=['shapenetcorev2','modelnet40', 'modelnet10'], help='Encoder to use, [shapenetcorev2,modelnet40, modelnet10]')
    parser.add_argument('--use_rotate', action='store_true', help='Rotate the pointcloud before training')
    parser.add_argument('--use_translate', action='store_true', help='Translate the pointcloud before training')
    parser.add_argument('--use_jitter', action='store_true', help='Jitter the pointcloud before training')
    parser.add_argument('--dataset_root', type=str, default='../dataset', help="Dataset root path")
    parser.add_argument('--gpu', type=str, help='Id of gpu device to be used', default='0')
    parser.add_argument('--workers', type=int, help='Number of data loading workers', default=16)
    parser.add_argument('--epochs', type=int, default=200, metavar='N', help='Number of episode to train ')
    parser.add_argument('--snapshot_interval', type=int, default=10, metavar='N', help='Save snapshot interval ')
    parser.add_argument('--no_cuda', action='store_true', help='Enables CUDA training')
    parser.add_argument('--eval', action='store_true', help='Evaluate the model')
    parser.add_argument('--num_points', type=int, default=2048, help='Num of points to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N', help='Path to load model')
    parser.add_argument('--n_clusters', default=128, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--gamma', default=10, type=int)
    parser.add_argument('--pretrain_epochs', default=100, type=int)
    parser.add_argument('--ae_weights', default=None)
    parser.add_argument('--save_dir', default='results/n2d')
    parser.add_argument('--umap_dim', default=2, type=int)
    parser.add_argument('--umap_neighbors', default=10, type=int)
    parser.add_argument('--umap_min_dist', default="0.00", type=str)
    parser.add_argument('--umap_metric', default='euclidean', type=str)
    parser.add_argument('--cluster', default='GMM', type=str)
    parser.add_argument('--eval_all', default=False, action='store_true')
    parser.add_argument('--manifold_learner', default='UMAP', type=str)
    parser.add_argument('--visualize', default=True, action='store_true')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    print("Process Started")

    args = get_parser()

    path = os.getcwd()
    path_pick = os.path.join(path, r'..\..\..\..\..\Data\JOIM27SA\JOIM27SA.pkl')
    image_arr = os.path.join(path, r'..\image_data.npy')
    pcd_arr = os.path.join(path, r'..\pcd_data.npy')
    path_m_images = os.path.join(path, r'..\jupiter_sit_cluster\Scripts\model_images.pth')
    path_m_pcd = os.path.join(path, r'..\jupiter_sit_cluster\Scripts\model_pcd.pth')
    
    print("Loading Images")
    image_temp = np.load(image_arr) # load
    img_tensor = torch.from_numpy(image_temp)
    img_tensor = img_tensor.permute(1, 0, 2, 3)
    
    print("Loading BetaVAE")
    # model_images = BetaVAE(in_channels=1,
    #                 latent_dim=128,
    #                 loss_type='H',
    #                 gamma=args.gamma)
    # # ///                 max_capacity=25,
    # # ///                 Capacity_max_iter=10000)
    # model_images = model_images.double()
    model_images = torch.load(path_m_images, map_location=torch.device('cpu'))
    features = []
    mu_encode, log_var = model_images.encode(img_tensor)
    feature_images = model_images.reparameterize(mu = mu_encode, logvar = log_var)
    feature_images = feature_images.detach().numpy()
    
    
#     print("Loading Point Clouds")
#     pcd_data = np.load(pcd_arr) # load
#     pcd_data_tensor = torch.from_numpy(pcd_data)
#     # for training
    
#     print("Loading FoldingNet AutoEncoder")
#     model_pcd = ReconstructionNet(args)
#     model_pcd = model_pcd.double()
#     model_pcd.load_state_dict(torch.load(path_m_pcd))
#     feature_pcd = model_pcd.encoder(pcd_data_tensor)
#     feature_pcd = feature_pcd.detach().numpy()
    
#     print("Fusing Features")
#     features = np.concatenate((feature_images, feature_pcd), axis=1)
#     ###########Shuffling features in each row#######################
#     features = [np.random.shuffle(x) for x in features]
#     features = np.asarray(features)
    
    print("Reducing Dimensions PCA")
    pca = PCA(n_components=200)
    fit = pca.fit_transform(feature_images)
    
    md = float(args.umap_min_dist)
    hle = umap.UMAP(random_state=0,
                metric=args.umap_metric,
                n_components=args.umap_dim,
                n_neighbors=args.umap_neighbors,
                min_dist=md).fit_transform(feature_images)
    hle_test = umap.UMAP(random_state=0,
                metric=args.umap_metric,
                n_components=args.umap_dim,
                n_neighbors=args.umap_neighbors,
                min_dist=md).fit(feature_images)
    umap.plot.points(hle_test)
    
    
    
    
    
    
    
    
    
