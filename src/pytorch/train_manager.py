#!/usr/bin/env python -W ignore::DeprecationWarning
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms

import numpy as np
import matplotlib.pyplot as plt
import os
import random
import time
import copy
import sys
import pprint

from dataset import SequenceDataset
from model import LSTMController
from control_loss import ControlLoss

class TrainManager():
    def __init__(self, training_dir=None, models_dir=None, split=None, epochs=None, seq_paths=[]):
        self._training_dir = training_dir
        self._models_dir = models_dir
        self._split = split
        self._epochs = epochs
        self._seq_paths = seq_paths

        self._num_workers = 3
        self._lr = 0.0001
        self._hidden_dim = 2000
        self._middle_out_dim = 500
        self._pref_seq_length = 600
        self._avg_pool = False

        self._plot_title = "seqlen: %d, h: %d, lr: %f, mid: %d, avgpool: %s" % (
                self._pref_seq_length, self._hidden_dim, self._lr, self._middle_out_dim,
                "True" if self._avg_pool else "False")
        self._model = LSTMController(self._hidden_dim, self._middle_out_dim, self._avg_pool).cuda()
        self._dataloaders = {}
        self._dataset_sizes = {}
        self._dataset_names = ['train', 'val']

    def load_data(self):
        # Shuffle and split the sequences into train, validation, and test set
        random.shuffle(self._seq_paths)
        threshold = int(len(self._seq_paths) * self._split)

        
        partitions = {'train': self._seq_paths[threshold:],
                     'val': self._seq_paths[:threshold]}

        data_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
        ])

        image_datasets = {x: SequenceDataset(partitions[x], self._pref_seq_length, data_transforms) 
                for x in self._dataset_names}

        self._dataloaders = {x: DataLoader(image_datasets[x], batch_size=1, shuffle=True,
                                num_workers=self._num_workers) for x in self._dataset_names}

        self.gen_dataset_intervals()

    def gen_dataset_intervals(self):
        self._dataset_sizes = {}
        for x in self._dataset_names:
            self._dataloaders[x].dataset.init_intervals() 
            self._dataset_sizes[x] = self._dataloaders[x].dataset.__len__()

    def train_model(self):
        # self.vel_criterion = nn.MSELoss()
        self.vel_criterion = ControlLoss()
        self.claw_criterion = nn.MSELoss()

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self._model.parameters()), lr=self._lr)

        since = time.time()
        best_loss = float('inf')

        train_loss_list = []
        val_loss_list = []

        for epoch in range(self._epochs):
           # Set model to training mode
            self._model.train()  
            t = time.time() - since

            self.gen_dataset_intervals()

            # Iterate over data.
            for i, (X, y_vel, y_open, y_close) in enumerate(self._dataloaders['train']):
                X = Variable(torch.squeeze(X)).cuda()
                y_vel = Variable(torch.squeeze(y_vel)).cuda()
                y_open = Variable(torch.squeeze(y_open)).cuda()
                y_close = Variable(torch.squeeze(y_close)).cuda()

                optimizer.zero_grad()
                self._model.init_hidden()
                            
                out_vel, out_open, out_close = self._model(X)

                loss_vel = self.vel_criterion(out_vel, y_vel)
                loss_open = self.claw_criterion(out_open, y_open)
                loss_close = self.claw_criterion(out_close, y_close)
                loss = loss_vel + loss_open + loss_close

                loss.backward(retain_graph=True)
                optimizer.step()

                print("[Epoch: %d/%d, Sample: %0.2f%%] loss: %0.4f, time: %s      " % (
                    epoch + 1, self._epochs, (float(i + 1) / self._dataset_sizes["train"]) * 100, 
                    loss[0], self.get_time(time.time() - since)), end='\r')
                sys.stdout.flush()

            # print training and validation loss
            train_loss = self.test('train', since)
            val_loss = self.test('val', since)
            print('Epoch: %d, train loss: %0.4f, val loss: %0.4f' % (epoch + 1, train_loss, val_loss))
                   
            # plot training and validation loss
            train_loss_list.append(train_loss)
            val_loss_list.append(val_loss)
            self.plot_loss(train_loss_list, val_loss_list)

            # save as latest model, and save as best if it has the lowest validation loss
            torch.save(self._model, os.path.join(self._models_dir, "last_model.pt"))
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(self._model, os.path.join(self._models_dir, "model.pt"))

        # print total time taken to train
        print('Training complete in %s     ' % self.get_time(time.time() - since))

    def test(self, phase, since):
        loss = 0.0
        self._model.eval()  
        for i, (X, y_vel, y_open, y_close) in enumerate(self._dataloaders[phase]):
            X = Variable(torch.squeeze(X), volatile=True).cuda()
            y_vel = Variable(torch.squeeze(y_vel), volatile=True).cuda()
            y_open = Variable(torch.squeeze(y_open), volatile=True).cuda()
            y_close = Variable(torch.squeeze(y_close), volatile=True).cuda()

            self._model.init_hidden()
            out_vel, out_open, out_close = self._model(X)

            loss_vel = self.vel_criterion(out_vel, y_vel)
            loss_open = self.claw_criterion(out_open, y_open)
            loss_close = self.claw_criterion(out_close, y_close)
            loss += loss_vel + loss_open + loss_close

            print('Calculating %s loss: %0.1f%%, time: %s%s' % (phase, 
                (float(i + 1) / self._dataset_sizes[phase]) * 100, self.get_time(time.time() - since), 
                ' ' * 20), end='\r')
            sys.stdout.flush()

        return (loss / self._dataset_sizes[phase]).data[0]

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

