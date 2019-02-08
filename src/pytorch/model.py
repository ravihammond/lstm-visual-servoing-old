import torch
import torch.nn as nn
import torchvision
from torchvision import models

class LSTMController(nn.Module):
    def __init__(self, hidden_dim, middle_out_dim):
        super(LSTMController, self).__init__()
        self.hidden_dim = hidden_dim

        self.init_resnet()
        # res18_out_dim = 512 * 7 * 7
        res18_out_dim = 512
        self.middle_out_dim = middle_out_dim

        self.middle = nn.Linear(res18_out_dim, self.middle_out_dim)

        self.lstm = nn.LSTM(self.middle_out_dim, self.hidden_dim)
        self.hiddenToVel = nn.Linear(self.hidden_dim, 8)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.init_hidden()

    # def init_resnet(self):
        # self.res18_model = models.resnet18(pretrained=True)
        # self.res18_model = nn.Sequential(*list(self.res18_model.children())[:-2]).cuda()

    def init_resnet(self):
        self.res18_model = models.resnet18(pretrained=True)
        self.res18_model = nn.Sequential(*list(self.res18_model.children())[:-1]).cuda()

    def init_hidden(self):
        self.hidden = (torch.zeros(1, 1, self.hidden_dim).cuda(),
                       torch.zeros(1, 1, self.hidden_dim).cuda())

    def forward(self, X):
        with torch.no_grad():
            X = self.res18_model(X.float())

        X = self.middle(X.view(len(X), -1))
        X = self.sigmoid(X)

        lstm_out, self.hidden = self.lstm(X.reshape(len(X), 1, -1), self.hidden)

        vel_space = self.hiddenToVel(lstm_out.view(len(X), -1))
        X = self.tanh(vel_space)

        return vel_space

if __name__ == "__main__" :
    print(models.__file__)
    lstm = LSTMController(500,500).cuda()
    lstm(torch.randn((100,3,244,244)).cuda())


