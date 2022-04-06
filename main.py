import argparse
import datetime
import os
import random
import time
import warnings
import numpy as np
from tqdm import tqdm

import torch.optim
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.nn.parallel
import torch.optim
import torch.utils
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter

from data.dataset import NYUV2Dataset
from data.dataset_Kitti import KITTIDataset
from models.resunet import ResUnet
from models.resunet_plus import ResUnetPlusPlus
from utils.losses import *
from utils.metrics import compute_errors
from utils.utils import DistributedSamplerNoEvenlyDivisible

parser = argparse.ArgumentParser()
parser.add_argument("--start-epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--epochs", type=int, default=80, help="number of epochs of training")
parser.add_argument('--data', default='./datasets', help='path to images')
parser.add_argument('--dataset-name', default='nyu', choices=['kitti', 'nyu'], help='path to images')
parser.add_argument("--batch-size", type=int, default=64, help="size of the batches")

parser.add_argument("-j", "--workers", type=int, default=4, help="number of cpu threads to use during batch generation")
parser.add_argument("--weight-decay", type=float, default=1e-2, help="weight decay")
parser.add_argument("--eps", type=float, default=1e-3, help="adamw eps parameters")
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--end-learning-rate', type=float, default=-1, help='end learning rate')

# Dataset Option
parser.add_argument('--in-channels', default=3, help="Image Resolution")
parser.add_argument('--input-height', type=int, default=416, help='input height')
parser.add_argument('--input-width', type=int, default=544, help='input width')
parser.add_argument('--do-random-rotate', action='store_true', help='if set, will perform random rotation for augmentation')
parser.add_argument('--degree', type=float, default=2.5, help='random rotation maximum degree')
parser.add_argument('--do-kb-crop', action='store_true', help='if set, crop input images as kitti benchmark images')

# model option
parser.add_argument('--model', type=str, default='resunet_plus', choices=['resunet', 'resunet_plus'], help='Backbone Model[resunet or resunet_plus]')
parser.add_argument('--variance-focus', type=float, default=0.85, help='lambda in paper: [0, 1], higher value more focus on minimizing variance of error')
parser.add_argument('--max-depth', type=float, default=10, help='maximum depth in estimation')
parser.add_argument('--min-depth-eval', type=float, default=1e-3, help='minimum depth for evaluation')
parser.add_argument('--max-depth-eval', type=float, default=80, help='maximum depth for evaluation')
parser.add_argument('--eigen-crop', action='store_true', help='if set, crops according to Eigen NIPS14')
parser.add_argument('--garg-crop', action='store_true', help='if set, crops according to Garg  ECCV16')

parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--print-freq', type=int, default=1)
parser.add_argument('--seed', type=int, default=None, help='random seed (default: None)')
parser.add_argument('--resume', default=None, type=str, metavar='PATH', help="model_args.resume")
parser.add_argument('--evaluate', '-e', default=False, action='store_true')

# Distributed
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    # global best_acc1
    args.gpu = gpu
    summary = SummaryWriter()

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # create models
    if args.model == 'resunet_plus':
        model = ResUnetPlusPlus(args=args, channel=args.in_channels)
    elif args.model == 'resunet':
        model = ResUnet(args=args, channel=args.in_channels)
    else:
        raise Exception('model error')

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("[Total Params]: ", n_params)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            print("this")
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)

    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()

    global global_step
    global_step = 0
    best_eval_measures_lower_better = torch.zeros(6).cpu() + 1e3
    best_eval_measures_higher_better = torch.zeros(3).cpu()
    best_eval_steps = np.zeros(9, dtype=np.int32)

    # Optimizer
    # criterion = silog_loss(args.variance_focus).cuda(args.gpu)
    # criterion = nn.MSELoss().cuda(args.gpu)
    criterion = SiLogLoss().cuda(args.gpu)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.lr)
    # optimizer = torch.optim.AdamW(model.parameters(),
    #                               lr=args.lr,
    #                               weight_decay=args.weight_decay,
    #                               eps=args.eps)
    # Resume
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)

            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            try:
                best_eval_measures_higher_better = checkpoint['best_eval_measures_higher_better'].cpu()
                best_eval_measures_lower_better = checkpoint['best_eval_measures_lower_better'].cpu()
                best_eval_steps = checkpoint['best_eval_steps']
            except KeyError:
                print("Could not load values for online evaluation")
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    if args.dataset_name == 'nyu':
        train_dataset = NYUV2Dataset(args=args, is_train=True)
        test_dataset = NYUV2Dataset(args=args, is_train=False)
    elif args.dataset_name == 'kitti':
        train_dataset = KITTIDataset(args=args, is_train=True)
        test_dataset = KITTIDataset(args=args, is_train=False)
    else:
        raise Exception("no have dataset")
    # Sampler
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        test_sampler = DistributedSamplerNoEvenlyDivisible(test_dataset, shuffle=False)
        print("Sampler")
    else:
        train_sampler = None
        test_sampler = None

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=(train_sampler is None),
                                               num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                              shuffle=False,
                                              num_workers=1, pin_memory=True, sampler=test_sampler)

    end_learning_rate = args.end_learning_rate if args.end_learning_rate != -1 else 0.1 * args.lr

    # Train
    eval_metrics = ['silog', 'abs_rel', 'log10', 'rms', 'sq_rel', 'log_rms', 'd1', 'd2', 'd3']
    for epoch in range(args.start_epoch, args.epochs):

        if args.distributed:
            train_sampler.set_epoch(epoch)

        train(train_loader, model, criterion, optimizer, epoch, args, summary, end_learning_rate)

        eval_measures = evaluate(test_loader, model, args, summary, ngpus_per_node, epoch+1)

        if eval_measures is not None:
            for i in range(9):
                measure = eval_measures[i]
                is_best = False
                if i < 6 and measure < best_eval_measures_lower_better[i]:
                    best_eval_measures_lower_better[i] = measure.item()
                    is_best = True
                elif i >= 6 and measure > best_eval_measures_higher_better[i - 6]:
                    best_eval_measures_higher_better[i - 6] = measure.item()
                    is_best = True

                if is_best:
                    print('New best for {}. Saving model'.format(eval_metrics[i]))
                    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                                and args.rank % ngpus_per_node == 0):
                        torch.save({
                            'epoch': epoch + 1,
                            'model': model.module.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'best_eval_measures_higher_better': best_eval_measures_higher_better,
                            'best_eval_measures_lower_better': best_eval_measures_lower_better,
                        }, "saved_models/checkpoint_%d.pth" % (epoch + 1))


def train(train_loader, model, criterion, optimizer, epoch, args, summary, end_lr):
    global global_step
    model.train()
    start_time = time.time()
    half_epoch = args.epochs // 2

    for i, batch in enumerate(train_loader):
        for param_group in optimizer.param_groups:
            if global_step < 2019 * half_epoch:
                current_lr = (1e-4 - 3e-5) * (global_step /
                                              2019 / half_epoch) ** 0.9 + 3e-5
            else:
                current_lr = (3e-5 - 1e-4) * (global_step /
                                              2019 / half_epoch - 1) ** 0.9 + 1e-4
            param_group['lr'] = current_lr

        image = torch.autograd.Variable(batch['image']).cuda(args.gpu, non_blocking=True)
        depth_gt = torch.autograd.Variable(batch['depth']).cuda(args.gpu, non_blocking=True)
        # focal = torch.autograd.Variable(batch['image']).cuda(args.gpu, non_blocking=True)

        depth_est = model(image)

        # if args.dataset_name == 'nyu':
        #     mask = depth_gt > 0.1
        # else:
        #     mask = depth_gt > 1.0

        # loss = criterion(depth_est, depth_gt, mask.to(torch.bool))
        loss = criterion(depth_est, depth_gt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # for param_group in optimizer.param_groups:
        #     current_lr = (args.lr - end_lr) * (1 - global_step / len(train_loader)) ** 0.9 + end_lr
        #     param_group['lr'] = current_lr
        niter = epoch * len(train_loader) + i
        if args.gpu == 0:
            summary.add_scalar('Train/Loss', loss.item(), niter)
            # summary.add_image('Train/Depth_est', depth_est, niter)
            # summary.add_image('Train/Depth_gt', depth_gt, niter)

        if i % args.print_freq == 0:
            print(f"Epoch [{epoch + 1}][{i}/{len(train_loader)}] | Loss: {loss: .4f} |")

        global_step += 1

    elapse = datetime.timedelta(seconds=time.time() - start_time)
    print(f"걸린 시간: ", elapse)


def evaluate(test_loader, model, args, summary, ngpus_per_node, epoch):
    model.eval()
    eval_measures = torch.zeros(10).cuda(args.gpu)

    for i, eval_batch in tqdm(enumerate(test_loader)):
        with torch.no_grad():
            image = torch.autograd.Variable(eval_batch['image']).cuda(args.gpu, non_blocking=True)
            depth_gt = torch.autograd.Variable(eval_batch['depth'])
            has_valid_depth = eval_batch['has_valid_depth']
            # focal = torch.autograd.Variable(eval_batch['focal']).cuda(args.gpu, non_blocking=True)
            if not has_valid_depth: # KITTI 중에 없는 것이 45개
                # print('Invalid depth. continue.')
                continue

            pred_depth = model(image)

            pred_depth = pred_depth.cpu().numpy().squeeze()
            depth_gt = depth_gt.cpu().numpy().squeeze()

        if args.do_kb_crop:
            height, width = depth_gt.shape
            top_margin = int(height - 352)
            left_margin = int((width - 1216) / 2)
            pred_depth_uncropped = np.zeros((height, width), dtype=np.float32)
            pred_depth_uncropped[top_margin:top_margin + 352, left_margin:left_margin + 1216] = pred_depth
            pred_depth = pred_depth_uncropped

        pred_depth[pred_depth < args.min_depth_eval] = args.min_depth_eval
        pred_depth[pred_depth > args.max_depth_eval] = args.max_depth_eval
        pred_depth[np.isinf(pred_depth)] = args.max_depth_eval
        pred_depth[np.isnan(pred_depth)] = args.min_depth_eval

        valid_mask = np.logical_and(depth_gt > args.min_depth_eval, depth_gt < args.max_depth_eval)

        if args.garg_crop or args.eigen_crop:
            gt_height, gt_width = depth_gt.shape
            eval_mask = np.zeros(valid_mask.shape)

            if args.garg_crop:
                eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height),
                int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

            elif args.eigen_crop:
                if args.dataset_name == 'kitti':
                    eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height),
                    int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
                else:
                    eval_mask[45:471, 41:601] = 1

            valid_mask = np.logical_and(valid_mask, eval_mask)
        measures = compute_errors(depth_gt[valid_mask], pred_depth[valid_mask])

        eval_measures[:9] += torch.tensor(measures).cuda(args.gpu)
        eval_measures[9] += 1

    if args.multiprocessing_distributed:
        group = dist.new_group([i for i in range(ngpus_per_node)])
        dist.all_reduce(tensor=eval_measures, op=dist.ReduceOp.SUM, group=group)

    if not args.multiprocessing_distributed or args.gpu == 0:
        eval_measures_cpu = eval_measures.cpu()
        cnt = eval_measures_cpu[9].item()
        eval_measures_cpu /= cnt
        print('Computing errors for {} eval samples'.format(int(cnt)))
        print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format('silog', 'abs_rel', 'log10', 'rms',
                                                                                     'sq_rel', 'log_rms', 'd1', 'd2',
                                                                                     'd3'))
        for i in range(8):
            print('{:7.3f}, '.format(eval_measures_cpu[i]), end='')
        print('{:7.3f}'.format(eval_measures_cpu[8]))

        if args.gpu == 0:
            summary.add_scalar('eval/silog', eval_measures_cpu[0].item(), epoch)
            summary.add_scalar('eval/abs_rel', eval_measures_cpu[1].item(), epoch)
            summary.add_scalar('eval/log10', eval_measures_cpu[2].item(), epoch)
            summary.add_scalar('eval/rms', eval_measures_cpu[3].item(), epoch)
            summary.add_scalar('eval/sq_rel', eval_measures_cpu[4].item(), epoch)
            summary.add_scalar('eval/log_rms', eval_measures_cpu[5].item(), epoch)
            summary.add_scalar('eval/d1', eval_measures_cpu[6].item(), epoch)
            summary.add_scalar('eval/d2', eval_measures_cpu[7].item(), epoch)
            summary.add_scalar('eval/d3', eval_measures_cpu[8].item(), epoch)

        return eval_measures_cpu

    return None


if __name__ == "__main__":
    main()
