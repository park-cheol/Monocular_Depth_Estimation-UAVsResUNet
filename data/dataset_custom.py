import os
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CustomDataset(Dataset):

    def __init__(self, args, test_files, gt_files, size):
        super(CustomDataset, self).__init__()
        self.args = args
        self.size = size

        self.resize = transforms.Resize(size=size)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.totensor = transforms.ToTensor()

        self.test_files = sorted(test_files)
        self.gt_files = sorted(gt_files)

    def __getitem__(self, idx):
        test_files = self.test_files[idx]
        gt_files = self.gt_files[idx]
        image = Image.open(test_files)
        depth_gt = Image.open(gt_files).convert('L')

        # image, depth_gt = self.random_crop(image, depth_gt, self.size[0], self.size[1])
        image, depth_gt = self.resize(image), self.resize(depth_gt)
        image = np.asarray(image, dtype=np.float32) / 255.0
        depth_gt = np.asarray(depth_gt, dtype=np.float32)
        depth_gt = np.expand_dims(depth_gt, axis=2)
        image, depth_gt = self.totensor(image), self.totensor(depth_gt)
        image = self.normalize(image)
        sample = {'image': image, 'depth': depth_gt}

        return sample

    def __len__(self):
        return len(self.gt_files)

    def random_crop(self, img, depth, height, width):
        # print(img.shape, depth.shape)
        # assert img.shape[0] >= height
        # assert img.shape[1] >= width
        # assert img.shape[0] == depth.shape[0]
        # assert img.shape[1] == depth.shape[1]
        # x = random.randint(0, img.shape[1] - width)
        # y = random.randint(0, img.shape[0] - height)
        x = img.shape[1]//2
        y = img.shape[0]//2
        img = img[y:y + height, x:x + width, :]
        depth = depth[y:y + height, x:x + width, :]
        return img, depth
