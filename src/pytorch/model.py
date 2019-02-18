import torch
from torch.autograd import Variable
import torch.nn as nn
import torchvision
from torchvision import models
import sys

class LSTMController(nn.Module):
    def __init__(self, hidden_dim, middle_out_dim, avg_pool):
        super(LSTMController, self).__init__()
        resnet_cutoff = -1 if avg_pool else -2
        res18_out_dim = 512 if avg_pool else 512*7*7
        self._hidden_dim = hidden_dim

        self.init_hidden()

        self._res18_model = self.init_resnet(resnet_cutoff)
        self._fc1 = nn.Linear(res18_out_dim, middle_out_dim)
        self._lstm = nn.LSTM(middle_out_dim, hidden_dim)
        self._fc2 = nn.Linear(hidden_dim, 6)
        self._fc3 = nn.Linear(hidden_dim, 2)
        self._fc4 = nn.Linear(hidden_dim, 2)

        self._sigmoid = nn.Sigmoid()
        self._tanh = nn.Tanh()

    def init_hidden(self):
        self._hidden = (Variable(torch.zeros(1, 1, self._hidden_dim).cuda()),
                       Variable(torch.zeros(1, 1, self._hidden_dim).cuda()))

    def init_resnet(self, cutoff):
        res18_model = models.resnet18(pretrained=True)
        res18_model = nn.Sequential(*list(res18_model.children())[:cutoff]).cuda()

        for param in res18_model.parameters():
            param.requires_grad = False

        return res18_model

    def forward(self, X):
        X = self._res18_model(X.float()).detach()

        X = self._sigmoid(self._fc1(X.view(len(X), -1)))

        lstm_out, self._hidden = self._lstm(X.view(len(X), 1, -1), self._hidden)
        lstm_out = lstm_out.view(len(X), -1)

        out_vel = self._tanh(self._fc2(lstm_out))
        out_open = self._fc3(lstm_out)
        out_close = self._fc4(lstm_out)
        
        return out_vel, out_open, out_close

if __name__ == "__main__" :
    print(models.__file__)
    lstm = LSTMController(500,500).cuda()
    lstm(torch.randn((100,3,244,244)).cuda())

