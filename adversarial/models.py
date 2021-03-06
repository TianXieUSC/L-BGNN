from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import init

REAL_LABEL = 1
FAKE_LABEL = 0


class Discriminator(nn.Module):
    def __init__(self, infeat, hidfeat, outfeat, dropout):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(infeat, hidfeat),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidfeat, outfeat),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.main(input)
        return output


def _weights_init(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        init.constant_(m.bias, 0)


class AdversarialLearning(object):
    def __init__(self, netG, infeat, hidfeat, learning_rate, weight_decay,
                 dropout, device, outfeat):
        self.iteration_cnt = 0
        self.log_interval = 100

        self.netD = Discriminator(infeat, hidfeat, outfeat, dropout).to(device)
        self.netD.apply(_weights_init)
        self.netG = netG

        self.optimizerG = optim.SGD(self.netG.parameters(),
                                    lr=learning_rate,
                                    weight_decay=weight_decay)

        self.optimizerD = optim.Adam(self.netD.parameters(),
                                     lr=learning_rate,
                                     weight_decay=weight_decay)

        self.device = device

    def _loss(self, logits, labels):
        criterion = nn.BCELoss()
        loss = criterion(logits, labels)
        return loss

    def forward_backward(self, real_data, netG_output, step, epoch, iter):
        real = real_data
        fake = netG_output
        ############################
        # (1) Update D network: maximize log_embedding(D(x)) + log_embedding(1 - D(G(z)))
        ###########################
        # train with real
        label_real = torch.full((real.size()[0], 1), REAL_LABEL, device=self.device, requires_grad=False)
        label_fake = torch.full((real.size()[0], 1), FAKE_LABEL, device=self.device, requires_grad=False)

        self.optimizerD.zero_grad()

        output = self.netD(real)
        lossD_real = self._loss(output, label_real)

        # train with fake
        output = self.netD(fake.detach())  # calculate the gradient for discriminator
        lossD_fake = self._loss(output, label_fake)

        lossD = lossD_real + lossD_fake
        lossD.backward()
        self.optimizerD.step()

        ############################
        # (2) Update G network: maximize log_embedding(D(G(z)))
        ###########################
        self.optimizerG.zero_grad()
        output = self.netD(fake)
        lossG = self._loss(output, label_real)
        lossG.backward()
        self.optimizerG.step()

        # if self.iteration_cnt % self.log_interval == 0:
        #     print("Layer depth: %s, Epoch: %s, Iterations: %s, dis loss: %s, gen loss: %s" % (
        #         step, epoch, iter, lossD.item(), lossG.item()))
        # self.iteration_cnt += 1

        return lossD.item(), lossG.item()
