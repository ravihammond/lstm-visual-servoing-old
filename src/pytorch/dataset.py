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
        self._seq_to_vel = self.seq_to_velocity_mappings()
        self.gen_aug_seq_intervals()

    # Create a dictionary that maps sequence paths to their velocities
    def seq_to_velocity_mappings(self):
        seq_to_vel = {}

        for seq_path in self._seq_paths:
            with open(os.path.join(seq_path, self._vel_filename), "r") as f:
                reader = csv.reader(f, delimiter = ",")
                velocities = list(reader)
                del velocities[0]
                print("list")
                print(velocities[50])
                print("\nnp")
                print(np.array(velocities)[50])
                print("\ntensor")
                print(torch.FloatTensor(np.array(velocities))[50])
                seq_to_vel[seq_path] = torch.FloatTensor(np.array(velocities))
            sys.exit("exit")

        return seq_to_vel

    # Create list of sequence directory paths to the start and stop intervals
    def gen_aug_seq_intervals(self):
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
        return len(self._sequence_dirs)

    # Get a single sample from dataset
    def __getitem__(self, index):
        # Select sequence
        seq_interval = self._seq_intervals[index]
        print("Sequence interval")
        print(seq_interval)

        # Extract the frames from the start and end of the interval
        images_list = []
        for i in range(seq_interval[1], seq_interval[2] + 1):
            file_name = "%06d.png" % i
            image_path = os.path.join(seq_interval[0], file_name)
            img = np.array(Image.open(image_path))
            img = self._transform(img)
            images_list.append(img)

        # Convert list of images to tensor
        X = torch.stack(images_list)
        print("X shape")
        print(X.shape)

        # Extract subseqence from large recording
        sub_seq = self._seq_to_vel[seq_interval[0]][seq_interval[1]-1:seq_interval[2]]

        # Convert subsequence to tensor
        y = torch.stack(sub_seq)
        print("y shape")
        print(y.shape)

        print(y[0])

        print
        return X, y

if __name__ == "__main__":
    import random
    from matplotlib import pyplot as plt
    from torchvision import datasets, models, transforms

    seq_dirs = ['training_data/test/06-02-2019_22-29-13', 
                'training_data/test/07-02-2019_18-06-59', 
                'training_data/test/06-02-2019_21-49-58', 
                'training_data/test/07-02-2019_16-28-08', 
                'training_data/test/06-02-2019_22-26-42', 
                'training_data/test/07-02-2019_16-23-57', 
                'training_data/test/06-02-2019_22-05-56', 
                'training_data/test/07-02-2019_17-47-36', 
                'training_data/test/06-02-2019_22-01-19']

    # random.shuffle(sequence_dirs)
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])

    dataset = SequenceDataset(seq_dirs, 100, data_transforms)

    # cv2.namedWindow("img",0)
    while True:
        for X, y in dataset:
            for i in range(len(X)):
                img = X[i,:,:,:].numpy().transpose(1,2,0).copy()
                img *= 50
                img += 127
                img = img.astype("uint8")
                # print(img.shape)
                # print(img)
                h,w,ch = img.shape

                plt.cla()
                plt.imshow(img)
                plt.title(str(y[i]))
                plt.pause(1)
                # cv2.imshow("img",img)
                # cv2.waitKey(300)


