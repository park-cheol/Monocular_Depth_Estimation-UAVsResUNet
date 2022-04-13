import argparse
from tqdm import tqdm
import os
import cv2
import errno
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.utils.data

from data.dataset_Kitti import KITTIDataset
from data.dataset_nyu import NYUV2Dataset
from models.resunet import ResUnet
from models.resunet_plus import ResUnetPlusPlus
from utils.losses import *

parser = argparse.ArgumentParser()
parser.add_argument('--data', default='./datasets', help='path to images')
parser.add_argument('--dataset-name', default='kitti', choices=['kitti', 'nyu'], help='path to images')
parser.add_argument('--filenames-file', type=str, help='path to the filenames text file', required=True)

# Dataset Option
parser.add_argument('--in-channels', default=3, help="Image Resolution")
parser.add_argument('--input-height', type=int, default=352, help='input height')
parser.add_argument('--input-width', type=int, default=704, help='input width')
parser.add_argument('--do-random-rotate', action='store_true', help='if set, will perform random rotation for augmentation')
parser.add_argument('--degree', type=float, default=2.5, help='random rotation maximum degree')
parser.add_argument('--do-kb-crop', action='store_true', help='if set, crop input images as kitti benchmark images')
parser.add_argument('--save-lpg', help='if set, save outputs from lpg layers', action='store_true')

# model option
parser.add_argument('--model', type=str, default='resunet_plus', choices=['resunet', 'resunet_plus'], help='Backbone Model[resunet or resunet_plus]')
parser.add_argument('--max-depth', type=float, default=80, help='maximum depth in estimation')

parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--resume', default=None, type=str, metavar='PATH', help="model_args.resume")


def test():
    args = parser.parse_args()
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create models
    if args.model == 'resunet_plus':
        model = ResUnetPlusPlus(args=args, channel=args.in_channels).cuda(args.gpu)
    elif args.model == 'resunet':
        model = ResUnet(args=args, channel=args.in_channels).cuda(args.gpu)
    else:
        raise Exception('model error')

    print("=> loading checkpoint '{}'".format(args.resume))
    if args.gpu is None:
        checkpoint = torch.load(args.resume)
    else:
        # Map model to be loaded to specified single gpu.
        loc = 'cuda:{}'.format(args.gpu)
        checkpoint = torch.load(args.resume, map_location=loc)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    num_test_samples = get_num_lines(args.filenames_file)
    with open(args.filenames_file) as f:
        lines = f.readlines()

    if args.dataset_name == 'nyu':
        test_dataset = NYUV2Dataset(args=args, is_train=False)
    elif args.dataset_name == 'kitti':
        test_dataset = KITTIDataset(args=args, is_train=False)
    else:
        raise Exception("no have dataset")

    dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                             shuffle=False,
                                             num_workers=1, pin_memory=True)

    pred_depths = []

    with torch.no_grad():
        for _, sample in enumerate(tqdm(dataloader)):
            image = torch.autograd.Variable(sample['image']).cuda(args.gpu, non_blocking=True)

            pred_depth = model(image)
            pred_depths.append(pred_depth.cpu().numpy().squeeze())

    print('Done. ')

    save_name = 'result_' + args.dataset_name
    print("saveing result pngs...")
    if not os.path.exists(os.path.dirname(save_name)):
        try:
            os.mkdir(save_name)
            os.mkdir(save_name + '/raw')
            os.mkdir(save_name + '/cmap')
            os.mkdir(save_name + '/rgb')
            os.mkdir(save_name + '/gt')
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    for s in tqdm(range(num_test_samples)):
        if args.dataset_name == 'kitti':
            date_drive = lines[s].split('/')[1]
            filename_pred_png = save_name + '/raw/' + date_drive + '_' + lines[s].split()[0].split('/')[-1].replace(
                '.jpg', '.png')
            filename_cmap_png = save_name + '/cmap/' + date_drive + '_' + lines[s].split()[0].split('/')[
                -1].replace('.jpg', '.png')
            filename_image_png = save_name + '/rgb/' + date_drive + '_' + lines[s].split()[0].split('/')[-1]
        elif args.dataset_name == 'kitti_benchmark':
            filename_pred_png = save_name + '/raw/' + lines[s].split()[0].split('/')[-1].replace('.jpg', '.png')
            filename_cmap_png = save_name + '/cmap/' + lines[s].split()[0].split('/')[-1].replace('.jpg', '.png')
            filename_image_png = save_name + '/rgb/' + lines[s].split()[0].split('/')[-1]
        else:
            scene_name = lines[s].split()[0].split('/')[0]
            filename_pred_png = save_name + '/raw/' + scene_name + '_' + lines[s].split()[0].split('/')[1].replace(
                '.jpg', '.png')
            filename_cmap_png = save_name + '/cmap/' + scene_name + '_' + lines[s].split()[0].split('/')[1].replace(
                '.jpg', '.png')
            filename_gt_png = save_name + '/gt/' + scene_name + '_' + lines[s].split()[0].split('/')[1].replace(
                '.jpg', '.png')
            filename_image_png = save_name + '/rgb/' + scene_name + '_' + lines[s].split()[0].split('/')[1]

        rgb_path = os.path.join('/data1/KITTI/', lines[s].split()[0])
        image = cv2.imread(rgb_path)
        if args.dataset == 'nyu':
            gt_path = os.path.join('/data1/NYU_depth_V2/test' + lines[s].split()[1])
            gt = cv2.imread(gt_path, -1).astype(np.float32) / 1000.0  # Visualization purpose only
            gt[gt == 0] = np.amax(gt)

        pred_depth = pred_depths[s]

        if args.dataset_name == 'kitti' or args.dataset_name == 'kitti_benchmark':
            pred_depth_scaled = pred_depth * 256.0
        else:
            pred_depth_scaled = pred_depth * 1000.0

        pred_depth_scaled = pred_depth_scaled.astype(np.uint16)
        cv2.imwrite(filename_pred_png, pred_depth_scaled, [cv2.IMWRITE_PNG_COMPRESSION, 0])

        if args.save_lpg:
            cv2.imwrite(filename_image_png, image[10:-1 - 9, 10:-1 - 9, :])
            if args.dataset_name == 'nyu':
                plt.imsave(filename_gt_png, np.log10(gt[10:-1 - 9, 10:-1 - 9]), cmap='Greys')
                pred_depth_cropped = pred_depth[10:-1 - 9, 10:-1 - 9]
                plt.imsave(filename_cmap_png, np.log10(pred_depth_cropped), cmap='Greys')
            else:
                plt.imsave(filename_cmap_png, np.log10(pred_depth), cmap='Greys')

        return


def get_num_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return len(lines)


if __name__ == '__main__':
    test()
