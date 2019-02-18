import torch
import torch.nn as nn

class ControlLoss():
    def __init__(self):
        self.cos_sim = nn.CosineSimilarity()
        self.p_dist = nn.PairwiseDistance()

    def __call__(self, input1, input2):

        input1_v = input1[:,0:3]
        input1_r = input1[:,3:6]

        input2_v = input2[:,0:3]
        input2_r = input2[:,3:6]

        norm1_v = torch.norm(input1_v,p=2,dim=1)
        norm1_r = torch.norm(input1_r,p=2,dim=1)
        norm2_v = torch.norm(input2_v,p=2,dim=1)
        norm2_r = torch.norm(input2_r,p=2,dim=1)


        cos_sim_v = 1.0-self.cos_sim(input1_v,input2_v)
        cos_sim_v *= (norm1_v > 0.01).float() * (norm2_v > 0.01).float()
        dist_v = self.p_dist(input1_v,input2_v).squeeze_()


        cos_sim_r = 1.0-self.cos_sim(input1_r,input2_r)
        cos_sim_r *= (norm1_r > 0.01).float() * (norm2_r > 0.01).float()
        dist_r = self.p_dist(input1_r,input2_r).squeeze_()


        loss_v = 0.9*cos_sim_v + 0.1 * dist_v
        loss_r = 0.9*cos_sim_r + 0.1 * dist_r

        loss_joined = loss_v + loss_r
        loss = torch.mean(loss_joined)

        return loss

