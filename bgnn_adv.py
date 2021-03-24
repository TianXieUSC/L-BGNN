from __future__ import division
from __future__ import print_function

import os

import numpy as np
import torch
import scipy.sparse as sp

from adversarial.models import AdversarialLearning
from gcn.models import GCN, twoLayersGCN
from utils import sparse_mx_to_torch_sparse_tensor
from utils import batch_normalization
from dataloader import normalize_feat


class BGNNAdversarial(object):
    def __init__(self, dataloader, model_config):
        self.dataloader = dataloader
        self.model_config = model_config
        self.batch_iter_one = self.model_config['set_one_size'] // self.model_config['batch_size'] + 1
        self.batch_iter_two = self.model_config['set_two_size'] // self.model_config['batch_size'] + 1

        self.agg_layers, self.adv_layers = self._layer_initialize()

    def _layer_initialize(self):
        agg_layers = []
        adversarial_layers = []
        for i in range(self.model_config['layer_depth']):
            # initialize the two aggregators
            agg_one = GCN(self.model_config['attr_one_dim'], self.model_config['attr_two_dim']).to(
                self.model_config['device'])
            agg_two = GCN(self.model_config['attr_two_dim'], self.model_config['attr_one_dim']).to(
                self.model_config['device'])
            agg_layers.append([agg_one, agg_two])

            # initialize the adversarial operations
            adv_one = AdversarialLearning(agg_one,
                                          self.model_config['attr_two_dim'],
                                          self.model_config['disc_hid_dim'],
                                          self.model_config['learning_rate'],
                                          self.model_config['weight_decay'],
                                          self.model_config['dropout'],
                                          self.model_config['device'],
                                          outfeat=1)
            adv_two = AdversarialLearning(agg_two,
                                          self.model_config['attr_one_dim'],
                                          self.model_config['disc_hid_dim'],
                                          self.model_config['learning_rate'],
                                          self.model_config['weight_decay'],
                                          self.model_config['dropout'],
                                          self.model_config['device'],
                                          outfeat=1)
            adversarial_layers.append([adv_one, adv_two])
        return agg_layers, adversarial_layers

    def _layer_inference(self, agg_layer, adv_layer, set_one_emb, set_two_emb, depth):
        batch_size = self.model_config['batch_size']
        inc_one = self.dataloader.inc_one
        inc_two = self.dataloader.inc_two

        # the training of set two nodes
        for i in range(self.model_config['epoch']):
            for j in range(self.batch_iter_two):
                batch_two_emb = set_two_emb[j * batch_size:(j + 1) * batch_size]
                batch_two_inc = inc_two[j * batch_size:(j + 1) * batch_size]
                batch_two_inc = sparse_mx_to_torch_sparse_tensor(batch_two_inc).to(device=self.model_config['device'])

                agg_output = agg_layer[0](set_one_emb, batch_two_inc)
                lossD, lossG = adv_layer[0].forward_backward(batch_two_emb, agg_output, step=depth, epoch=i, iter=j)
                print('Set two, Layer depth: {}, Epoch: {}, Batch number: {}, '
                      'Finished percentage: {:.2f}, LossD: {:.4f}, LossG: {:.4f}'
                      .format(depth, i, j, j / self.batch_iter_two, lossD, lossG))

            print('')
            # the training of set one nodes
            for j in range(self.batch_iter_one):
                batch_one_emb = set_one_emb[j * batch_size:(j + 1) * batch_size]
                batch_one_inc = inc_one[j * batch_size:(j + 1) * batch_size]
                batch_one_inc = sparse_mx_to_torch_sparse_tensor(batch_one_inc).to(device=self.model_config['device'])

                agg_output = agg_layer[1](set_two_emb, batch_one_inc)
                lossD, lossG = adv_layer[1].forward_backward(batch_one_emb, agg_output, step=depth, epoch=i, iter=j)
                print('Set one, Layer depth: {}, Epoch: {}, Batch number: {}, '
                      'Finished percentage: {:.2f}, LossD: {:.4f}, LossG: {:.4f}'
                      .format(depth, i, j, j / self.batch_iter_one, lossD, lossG))
            print('')
        print('Training done.')
        infer_set_one_emb = torch.FloatTensor([]).to(device=self.model_config['device'])
        infer_set_two_emb = torch.FloatTensor([]).to(device=self.model_config['device'])
        for j in range(self.batch_iter_two):
            batch_two_inc = inc_two[j * batch_size:(j + 1) * batch_size]
            batch_two_inc = sparse_mx_to_torch_sparse_tensor(batch_two_inc).to(device=self.model_config['device'])
            agg_output = agg_layer[0](set_one_emb, batch_two_inc)
            infer_set_two_emb = torch.cat((infer_set_two_emb, agg_output.detach()), dim=0)

        for j in range(self.batch_iter_one):
            batch_one_inc = inc_one[j * batch_size:(j + 1) * batch_size]
            batch_one_inc = sparse_mx_to_torch_sparse_tensor(batch_one_inc).to(device=self.model_config['device'])
            agg_output = agg_layer[1](set_two_emb, batch_one_inc)
            infer_set_one_emb = torch.cat((infer_set_one_emb, agg_output.detach()), dim=0)
        return infer_set_one_emb, infer_set_two_emb

    def adversarial_learning(self):
        set_one_emb = torch.FloatTensor(self.dataloader.set_one).to(device=self.model_config['device'])
        set_two_emb = torch.FloatTensor(self.dataloader.set_two).to(device=self.model_config['device'])
        for i in range(self.model_config['layer_depth']):
            infer_set_one_emb, infer_set_two_emb = self._layer_inference(self.agg_layers[i], self.adv_layers[i],
                                                                         set_one_emb, set_two_emb, i)
            # set_one_emb = torch.FloatTensor(batch_normalization(infer_set_one_emb))
            # set_two_emb = torch.FloatTensor(batch_normalization(infer_set_two_emb))

            # if i != self.model_config['layer_depth'] - 1:
            #     set_one_emb = torch.FloatTensor(normalize_feat(infer_set_one_emb.numpy()))
            #     set_two_emb = torch.FloatTensor(normalize_feat(infer_set_two_emb.numpy()))
            set_one_emb, set_two_emb = infer_set_one_emb, infer_set_two_emb

        # saving embeddings to file
        print('Start saving files...')
        self._save_emb(set_one_emb, set_two_emb)

    def _save_emb(self, set_one_emb, set_two_emb):
        output_file = './output/bgnn-adv/' + self.dataloader.dataset
        if not os.path.exists(output_file):
            os.makedirs(output_file)
        sparse_one_emb = sp.csr_matrix(set_one_emb.numpy())
        sparse_two_emb = sp.csr_matrix(set_two_emb.numpy())
        sp.save_npz(output_file + '/set_one.npz', sparse_one_emb)
        sp.save_npz(output_file + '/set_two.npz', sparse_two_emb)
