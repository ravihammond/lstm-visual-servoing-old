from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, models, transforms

import numpy as np
import matplotlib.pyplot as plt
import os
import random
import time
import copy

from dataset import SequenceDataset
from model import LSTMController
from utils import memReport, cpuStats

class TrainManager():
    def __init__(self, training_dir=None, models_dir=None, split=None, epochs=None, seq_paths=[]):
        self._training_dir = training_dir
        self._models_dir = models_dir
        self._split = split
        self._epochs = epochs
        self._seq_paths = seq_paths

        self._num_workers = 3
        self._lr = 0.0001
        self._hidden_dim = 500
        self._middle_out_dim = 500
        self._pref_seq_length = 100

        self._plot_title = "seqlen: %d, h: %d, lr: %f, mid: %d" % (
                self._pref_seq_length, self._hidden_dim, self._lr, self._middle_out_dim)
        self._model = LSTMController(self._hidden_dim, self._middle_out_dim).cuda()
        self._dataloaders = {}
        self._dataset_sizes = {}
        self._dataset_names = ['train', 'val']

    def load_data(self):
        # Shuffle and split the sequences into train, validation, and test set
        random.shuffle(self._seq_paths)
        threshold = int(len(self._seq_paths) * self._split)
        partitions = {'train': self._seq_paths[threshold:],
                     'val': self._seq_paths[:threshold]}

        print(partitions)

        data_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
        ])

        image_datasets = {x: SequenceDataset(partitions[x], self._pref_seq_length, data_transforms) 
                for x in self._dataset_names}

        # self._dataloaders = {x: DataLoader(image_datasets[x], batch_size=1, shuffle=True,
                                # num_workers=self._num_workers) for x in self._partition_names}

        # self._dataset_sizes = {x: len(image_datasets[x]) for x in self._partition_names}

    def train_model(self):
        self._criterion = nn.MSELoss()
        optimizer = optim.Adam(self._model.parameters(), lr=self._lr)

        since = time.time()
        best_loss = float('inf')

        train_loss_list = []
        val_loss_list = []

        for epoch in range(self._epochs):
           # Set model to training mode
            self._model.train()  
            t = time.time() - since

            # Iterate over data.
            for i, (X, y) in enumerate(self._dataloaders['train']):
                optimizer.zero_grad()
                self._model.init_hidden()
                            
                X = torch.squeeze(X.cuda())
                predict = self._model(X.cuda())

                loss = self._criterion(predict, y.cuda())
                self._model.train()  
                loss.backward(retain_graph=True)
                optimizer.step()

                print("[Epoch: %d/%d, Sample: %d] loss: %0.4f, time: %s      " % (
                    epoch + 1, self._epochs, i + 1, loss.item(), self.get_time(time.time() - since)), 
                    end='\r', flush=True)

            # print training and validation loss
            train_loss = self.test('train', since)
            val_loss = self.test('val', since)
            print('Epoch: %d, train loss: %0.4f, val loss: %0.4f' % (epoch + 1, train_loss, val_loss))
                   

            # plot training and validation loss
            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)
            self.plot_loss(train_loss_list, val_loss_list)

            # save this model if it has the lowest validation loss
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(self._model, os.path.join(self._models_dir, "model.pt"))

        # print total time taken to train
        print('Training complete in %s     ' % self.get_time(time.time() - since))

    def test(self, phase, since):
        loss = 0.0
        self._model.eval()  
        with torch.no_grad():
            for i, (X, y) in enumerate(self._dataloaders[phase]):
                X = torch.squeeze(X).cuda()
                self._model.init_hidden()
                predict = self._model(X.cuda())
                loss += self._criterion(predict, y.cuda()).item()

                print('Calculating %s loss: %0.1f%%, time: %s%s' % (phase, 
                    ((i + 1) / self._dataset_sizes[phase]) * 100, self.get_time(time.time() - since), 
                    ' ' * 20), end='\r', flush=True)

        return loss / self._dataset_sizes[phase]

    def plot_loss(self, train_loss_list, val_loss_list):
        plt.cla()
        plt.plot(train_loss_list, color="blue", label="Training Loss")
        plt.plot(val_loss_list, color="green", dashes=[6,2], label="Validation Loss")
        plt.legend(loc="best")
        plt.title(self._plot_title)
        plt.ylabel("Loss")
        plt.xlabel("Epochs")
        plt.draw()
        plt.pause(0.1)
        plt.savefig(os.path.join(self._models_dir, "plot"))

    def get_time(self, t):
        return '%0.2d:%0.2d:%0.2d' % (t // 3600, (t % 3600) // 60, ((t % 3600) % 60))

