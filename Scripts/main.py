import argparse
import os
import random as rn
import zarr
import pandas as pd
import sys
import cv2
import pickle as pk
import logging
import h5py
import torch
import torchvision
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader

from beta_vae import BetaVAE
from reconstruction import ReconstructionNet
from tqdm import tqdm
import wandb

wandb.init(project="dgcnn_cls-dbvae-10-project")


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

def run(Epochs, loader, model, optimizer, text):
    print("Training Started")
    if text == 'fldnet':
        losses = []
        epochses = []
        for epoch in range(Epochs):
            for point in loader:
                reconstructed, _ = model(point)
                loss = model.get_loss(point, reconstructed)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lossin = loss.detach().numpy()
            wandb.log({"loss-fold": lossin})

            # Optional
            wandb.watch(model)
            epochses.append(epoch)
            losses.append(lossin)
            print("Epoch [{}], loss: {:.4f}".format(epoch,float(lossin)))
        print('Saving CSV file for loss for model_pcd')
        # dictionary of lists
        diction = {'Epoch': epochses,
                   'Loss': losses}
        # creating a dataframe from a dictionary
        df = pd.DataFrame(diction)
        #print(df)
        df.to_csv('model_pcd' + '.csv', encoding='utf-8', index=False)
    elif text == 'vae':
        losses = []
        epochses = []
        for epoch in range(Epochs):
            for image in loader:
                # Output of Autoencoder
                reconstructed = model(image)
                # Calculating the loss function
                loss = model.loss_function(*reconstructed, M_N=0.00025)
                # The gradients are set to zero,
                # the gradient is computed and stored.
                # .step() performs parameter update
                optimizer.zero_grad()
                loss['loss'].backward()
                optimizer.step()
                lossin = loss['loss'].detach().numpy()
            wandb.log({"loss-vae": lossin})

            # Optional
            wandb.watch(model)
            # Storing the losses in a list for plotting
            losses.append(lossin)
            epochses.append(epoch)
            print("Epoch [{}], loss: {:.4f}".format(epoch, float(lossin)))
        # dictionary of lists
        print('Saving CSV file for loss for model_images')
        diction = {'Epoch': epochses,
                   'Loss': losses}
        # creating a dataframe from a dictionary
        df = pd.DataFrame(diction)
        # print(df)
        df.to_csv('model_images' + '.csv', encoding='utf-8', index=False)

if __name__ == '__main__':
    print("Process Started")

    args = get_parser()

    path = os.getcwd()
    path_pick = os.path.join(path, r'..\..\..\..\..\Data\JOIM27SA\JOIM27SA.pkl')
    image_arr = os.path.join(path, r'..\image_data.npy')
    pcd_arr = os.path.join(path, r'..\pcd_data.npy')

    df = pd.read_pickle(path_pick)

    pointcloud_seq = df.LIVOX_FRONT_LIDAR_seq.to_numpy()
    image_seq = df.frame.to_numpy()

    print("Loading Images")
    image_temp = np.load(image_arr) # load
    img_tensor = torch.from_numpy(image_temp)
    img_tensor = img_tensor.permute(1, 0, 2, 3)
    # for training
    loader_images = DataLoader(dataset=img_tensor,
                        batch_size=args.batch_size,
                        shuffle=True)
    print("Loading BetaVAE")
    model_images = BetaVAE(in_channels=1,
                    latent_dim=128,
                    loss_type='H',
                    gamma=args.gamma)
    # ///                 max_capacity=25,
    # ///                 Capacity_max_iter=10000)
    optimizer_images = torch.optim.Adam(model_images.parameters(),
                                 lr=1e-3,
                                 weight_decay=1e-5)
    wandb.config = {
        "learning_rate-B": 1e-3,
        "learning_rate-fold": 0.0001*16/args.batch_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size
    }
    model_images = model_images.double()
    run(args.epochs, loader_images, model_images, optimizer_images, text = 'vae')
    torch.save(model_images, 'model\model_images.pth')

    print("Loading Point Clouds")
    pcd_data = np.load(pcd_arr) # load
    pcd_data_tensor = torch.from_numpy(pcd_data)
    # for training
    loader_pcd = DataLoader(dataset=pcd_data_tensor,
                        batch_size=args.batch_size,
                        shuffle=True)

    print("Loading FoldingNet AutoEncoder")
    model_pcd = ReconstructionNet(args)
    optimizer_pcd = torch.optim.Adam(model_pcd.parameters(),
                                 lr=0.0001*16/args.batch_size, betas=(0.9,0.999),
                                 weight_decay=1e-6)
    model_pcd = model_pcd.double()
    run(args.epochs, loader_pcd, model_pcd, optimizer_pcd, text = 'fldnet')
    torch.save(model_pcd, 'model\model_pcd.pth')

    print('Done')


