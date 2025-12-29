# Data loading based on https://github.com/NVIDIA/flownet2-pytorch

import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import logging
import os
import re
import copy
import math
import random
from pathlib import Path
from glob import glob
import os.path as osp

from core.utils import frame_utils
from core.utils.augmentor import FlowAugmentor, SparseFlowAugmentor

class StereoDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False, reader=None):
        self.augmentor = None
        self.sparse = sparse
        self.aug_params = aug_params
        self.img_pad = aug_params.pop("img_pad", None) if aug_params is not None else None
        if aug_params is not None and "crop_size" in aug_params:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        if reader is None:
            self.disparity_reader = frame_utils.read_gen
        else:
            self.disparity_reader = reader        

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.disparity_list = []
        self.image_list = []
        self.extra_info = []

    def __getitem__(self, index):

        if self.is_test:
            img1 = frame_utils.read_gen(self.image_list[index][0])
            img2 = frame_utils.read_gen(self.image_list[index][1])
            img1 = np.array(img1).astype(np.uint8)[..., :3]
            img2 = np.array(img2).astype(np.uint8)[..., :3]
            img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
            img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
            return img1, img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        disp = self.disparity_reader(self.disparity_list[index])
        
        if isinstance(disp, tuple):
            disp, valid = disp
        else:
            valid = disp < 1024

        img1 = frame_utils.read_gen(self.image_list[index][0])
        img2 = frame_utils.read_gen(self.image_list[index][1])

        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)
        disp = np.array(disp).astype(np.float32)
        flow = np.stack([disp, np.zeros_like(disp)], axis=-1)

        if len(img1.shape) == 2:
            img1 = np.tile(img1[...,None], (1, 1, 3))
            img2 = np.tile(img2[...,None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]

        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:

                img1, img2, flow = self.augmentor(img1, img2, flow)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        if self.sparse:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1024) & (flow[1].abs() < 1024)

        if self.img_pad is not None:

            padH, padW = self.img_pad
            img1 = F.pad(img1, [padW]*2 + [padH]*2)
            img2 = F.pad(img2, [padW]*2 + [padH]*2)

        flow = flow[:1]

        return self.image_list[index] + [self.disparity_list[index]], img1, img2, flow, valid.float()

    def __mul__(self, v):
        copy_of_self = copy.deepcopy(self)
        copy_of_self.flow_list = v * copy_of_self.flow_list
        copy_of_self.image_list = v * copy_of_self.image_list
        copy_of_self.disparity_list = v * copy_of_self.disparity_list
        copy_of_self.extra_info = v * copy_of_self.extra_info
        return copy_of_self
        
    def __len__(self):
        return len(self.image_list)

class SCARD(StereoDataset):
    def __init__(self, aug_params=None, root='..\..\..\datasets\scard', image_set='train'):
        super(SCARD, self).__init__(aug_params, sparse=True, reader=frame_utils.readDispKITTI)
        assert os.path.exists(root), f"Root path {root} does not exist."

        # Define the path to the training or testing set
        data_path = os.path.join(root, image_set)
        assert os.path.exists(data_path), f"Image set path {data_path} does not exist."

        # Loop through dataset_n folders
        for dataset_id in range(1, 8):  # dataset_1 to dataset_9
            dataset_folder = os.path.join(data_path, f'dataset_{dataset_id}')
            assert os.path.exists(dataset_folder), f"Dataset folder {dataset_folder} does not exist."

            # Loop through keyframe_m folders
            for keyframe_id in range(1, 6):  # keyframe_1 to keyframe_5
                keyframe_folder = os.path.join(dataset_folder, f'keyframe_{keyframe_id}', 'data')
                #assert os.path.exists(keyframe_folder), f"Keyframe folder {keyframe_folder} does not exist."
                if not os.path.exists(keyframe_folder):
                    print(f"Skipping {keyframe_folder}: data folder does not exist.")
                    continue

                left_folder = os.path.join(keyframe_folder, 'left')
                left_folder = os.path.normpath(left_folder)
                #print(left_folder)
                if not os.path.exists(left_folder):
                    print(f"Skipping {keyframe_folder}: 'left' folder not found.")
                    continue

                disp_folder = os.path.join(keyframe_folder, 'disparity')
                disp_folder = os.path.normpath(disp_folder)
                #print(left_folder)
                if not os.path.exists(disp_folder):
                    print(f"Skipping {keyframe_folder}: 'disp' folder not found.")
                    continue

                # Paths for left, right, and disparity images
                left_images = sorted(glob(os.path.join(keyframe_folder, 'left', 'frame_data*.png')))
                right_images = sorted(glob(os.path.join(keyframe_folder, 'right', 'frame_data*.png')))
                disparity_images = sorted(glob(os.path.join(keyframe_folder, 'disparity', 'frame_data*.tiff')))

                # Ensure that all three lists have the same number of files
                assert len(left_images) == len(right_images) == len(disparity_images), (
                    f"Mismatched file counts in {keyframe_folder}: "
                    f"left({len(left_images)}), right({len(right_images)}), disparity({len(disparity_images)})"
                )

                # Append image pairs and disparity maps to the lists

                for left, right, disp in zip(left_images, right_images, disparity_images):
                    self.image_list += [[left, right]]
                    self.disparity_list += [disp]

        print(f"Loaded {len(self.image_list)} image pairs and {len(self.disparity_list)} disparity maps from {root}.")

class SCARD_test(StereoDataset):
    def __init__(self, aug_params=None, root='..\..\..\datasets\scard', split='train'):

        super(SCARD_test, self).__init__(aug_params, sparse=True, reader=frame_utils.readDispKITTI)
        self.root = root
        self.split = split

        self._add_scard()

    def _add_scard(self):

        original_length = len(self.disparity_list)

        root = osp.join(self.root, self.split)  # e.g., scard/test
        #dataset_folders = sorted(glob(osp.join(root, 'dataset_*')))  # e.g., scard/test/dataset_1

        for dataset_id in range(1, 10):  # dataset_8 to dataset_9
            dataset_folder = os.path.join(root, f'dataset_{dataset_id}')
            assert os.path.exists(dataset_folder), f"Dataset folder {dataset_folder} does not exist."

            # Loop through keyframe_m folders
            for keyframe_id in range(1, 6):  # keyframe_1 to keyframe_5
                keyframe_folder = os.path.join(dataset_folder, f'keyframe_{keyframe_id}', 'data')
                #assert os.path.exists(keyframe_folder), f"Keyframe folder {keyframe_folder} does not exist."
                if not os.path.exists(keyframe_folder):
                    print(f"Skipping {keyframe_folder}: data folder does not exist.")
                    continue

                left_folder = os.path.join(keyframe_folder, 'left')
                left_folder = os.path.normpath(left_folder)
                #print(left_folder)
                if not os.path.exists(left_folder):
                    print(f"Skipping {keyframe_folder}: 'left' folder not found.")
                    continue

                disp_folder = os.path.join(keyframe_folder, 'disparity')
                disp_folder = os.path.normpath(disp_folder)
                #print(left_folder)
                if not os.path.exists(disp_folder):
                    print(f"Skipping {keyframe_folder}: 'disp' folder not found.")
                    continue

                # Paths for left, right, and disparity images
                left_images = sorted(glob(os.path.join(keyframe_folder, 'left', 'frame_data*.png')))
                right_images = sorted(glob(os.path.join(keyframe_folder, 'right', 'frame_data*.png')))
                disparity_images = sorted(glob(os.path.join(keyframe_folder, 'disparity', 'frame_data*.tiff')))

                # Ensure that all three lists have the same number of files
                assert len(left_images) == len(right_images) == len(disparity_images), (
                    f"Mismatched file counts in {keyframe_folder}: "
                    f"left({len(left_images)}), right({len(right_images)}), disparity({len(disparity_images)})"
                )

                # Append image pairs and disparity maps to the lists

                for left, right, disp in zip(left_images, right_images, disparity_images):
                    self.image_list += [[left, right]]
                    self.disparity_list += [disp]

        logging.info(f"Added {len(self.disparity_list) - original_length} samples from scard dataset")
  
def fetch_dataloader(args):
    """ Create the data loader for the corresponding trainign set """

    aug_params = {'crop_size': args.image_size, 'min_scale': args.spatial_scale[0], 'max_scale': args.spatial_scale[1], 'do_flip': False, 'yjitter': not args.noyjitter}
    if hasattr(args, "saturation_range") and args.saturation_range is not None:
        aug_params["saturation_range"] = args.saturation_range
    if hasattr(args, "img_gamma") and args.img_gamma is not None:
        aug_params["gamma"] = args.img_gamma
    if hasattr(args, "do_flip") and args.do_flip is not None:
        aug_params["do_flip"] = args.do_flip

    train_dataset = None
    for dataset_name in args.train_datasets:
        if 'scard' in dataset_name:
            new_dataset = SCARD(aug_params)
            logging.info(f"Adding {len(new_dataset)} samples from scard")

        train_dataset = new_dataset if train_dataset is None else train_dataset + new_dataset

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, 
        pin_memory=True, shuffle=True, num_workers=8, drop_last=True)

    logging.info('Training with %d image pairs' % len(train_dataset))
    return train_loader

