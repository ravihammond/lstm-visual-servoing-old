from __future__ import print_function

import torch
import torchvision
from torchvision import transforms
from torch.utils import data

import numpy as np
from PIL import Image
import os
import csv
import sys
import cv2
import pprint
from random import randint

# Dataset containing all seque
class SequenceDataset(data.Dataset):
    def __init__(self, seq_paths, pref_seq_len, transform):
        # List of valid sequence paths to load 
        self._seq_paths = seq_paths
        self._pref_seq_length = pref_seq_len
        self._transform = transform

        self._vel_filename = "velocities.csv"

        # Sequence => velocity mappings
        self.map_outputs()
        self.init_intervals()

    # Create a dictionary that maps sequence paths to their velocities
    def map_outputs(self):
        self._seq_to_vel = {}

        for i, seq_path in enumerate(self._seq_paths):
            with open(os.path.join(seq_path, self._vel_filename), "r") as csvfile:
                reader = csv.reader(csvfile)
                next(reader)

                vel_list = [list(map(float, row)) for row in reader] 
                vel = np.array(vel_list)
                np.set_printoptions(precision=4)

                y_vel = torch.FloatTensor(vel[:,:6])
                y_claw = torch.FloatTensor(vel[:,6])
                X_coords = torch.FloatTensor(vel[:,7:10])

                self._seq_to_vel[seq_path] = (y_vel, y_claw, X_coords)

    # Create list of sequence directory paths to the start and stop intervals
    def init_intervals(self):
        self._seq_intervals = []

        for seq_path in self._seq_paths:
            # Count number of frames
            frame_count = 0
            with open(os.path.join(seq_path, self._vel_filename), "r") as f:
                reader = csv.reader(f, delimiter = ",")
                data = list(reader)
                frame_count = len(data) - 1

            # Number of sequences
            num_seq = frame_count / self._pref_seq_length

            # If current seq length <= preffered seq length, only add one sequence
            if num_seq == 0:
                self._seq_intervals.append((seq_path, 1, frame_count))
                continue

            # Loop through number of seqeunces needed, and generate their start/end indexes
            for i in range(num_seq):
                index = randint(1, frame_count - self._pref_seq_length + 1)
                self._seq_intervals.append((seq_path, index, index + self._pref_seq_length - 1))

    # Get length of dataset
    def __len__(self):
        return len(self._seq_intervals)

    # Get a single sample from dataset
    def __getitem__(self, index):
        # Select sequence
        seq_path, start, end = self._seq_intervals[index]

        # Extract the frames from the start and end of the interval
        images_list = []
        for i in range(start, end + 1):
            file_name = "%06d.png" % i
            image_path = os.path.join(seq_path, file_name)
            img = np.array(Image.open(image_path))
            img = self._transform(img)
            images_list.append(img)
        # Extract subseqence from large recording
        outputs = self._seq_to_vel[seq_path]

        # Convert list of images to tensor
        X_img = torch.stack(images_list)
        X_coords = outputs[2][start - 1:end]

        y_vel = outputs[0][start - 1:end]
        y_claw = outputs[1][start - 1:end]

        return X_img, X_coords, y_vel, y_claw

    def create_claw_vector(self, val):
        return torch.Float

if __name__ == "__main__":
    import random
    from matplotlib import pyplot as plt
    from torchvision import datasets, models, transforms
    from torch.utils.data import DataLoader

    seq_dirs = ['training_data/rubbish_bin/19-02-2019_18-51-09_good']

    # random.shuffle(sequence_dirs)
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])

    dataset = SequenceDataset(seq_dirs, 500, data_transforms)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1) 

    for X_img, X_coords, y_vel, y_claw in dataloader:
        X_img = torch.squeeze(X_img)
        X_coords = torch.squeeze(X_coords)
        y_vel = torch.squeeze(y_vel)
        y_claw = torch.squeeze(y_claw)
        for i in range(len(X_img)):
            vel_str = np.array2string(np.array(y_vel[i]), precision=2, floatmode='fixed')
            claw_str = np.array2string(np.array(y_claw[i]), precision=0, floatmode='fixed')
            coords_str = np.array2string(np.array(X_coords[i]), precision=2, floatmode='fixed')
            output_str = vel_str + ' ' + claw_str + ' ' + coords_str
            img = X_img[i,:,:,:].numpy().transpose(1,2,0).copy()
            img *= 40
            img += 127
            img = img.astype("uint8")
            h,w,ch = img.shape

            plt.cla()
            plt.imshow(img)
            plt.title(output_str)
            plt.pause(0.001)

