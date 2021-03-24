import argparse
import logging
import os
import random

import numpy as np
import torch
import conf
import setproctitle
from bgnn_mlp import BGNNMLP
from bgnn_adv import BGNNAdversarial
from conf import (MODEL, BATCH_SIZE, EPOCHS, LEARNING_RATE,
                  WEIGHT_DECAY, DROPOUT, HIDDEN_DIMENSIONS, GCN_OUTPUT_DIM, ENCODER_HIDDEN_DIMENSIONS,
                  DECODER_HIDDEN_DIMENSIONS, MLP_HIDDEN_DIMENSIONS, LATENT_DIMENSIONS, LAYER_DEPTH)
from dataloader import DataLoader

import calendar
import time

setproctitle.setproctitle('BGNN')
os.environ['CUDA_VISIBLE_DEVICES'] = ' '


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='tencent', required=True)
    parser.add_argument('--model', type=str, default='adv', choices=MODEL, required=True)
    parser.add_argument('--seed', type=int, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=EPOCHS,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=WEIGHT_DECAY,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--dis_hidden', type=int, default=HIDDEN_DIMENSIONS,
                        help='Number of hidden units for discriminator in GAN model.')
    parser.add_argument('--dropout', type=float, default=DROPOUT,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--gpu', type=int, default=0,
                        help='Whether to use CPU or GPU')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='batch size')
    parser.add_argument('--gcn_output_dim', type=int, default=GCN_OUTPUT_DIM,
                        help='The output dimensions of GCN.')
    parser.add_argument('--rank', type=int, default=-1,
                        help='process ID for MPI Simple AutoML')
    parser.add_argument('--encoder_hidfeat', type=int, default=ENCODER_HIDDEN_DIMENSIONS,
                        help='Number of hidden units for encoder in VAE')
    parser.add_argument('--decoder_hidfeat', type=int, default=DECODER_HIDDEN_DIMENSIONS,
                        help='Number of hidden units for mlp in VAE')
    parser.add_argument('--vae_hidfeat', type=int, default=MLP_HIDDEN_DIMENSIONS,
                        help='Number of hidden units for latent representation in VAE')
    parser.add_argument('--latent_hidfeat', type=int, default=LATENT_DIMENSIONS,
                        help='Number of latent units for latent representation in VAE')
    parser.add_argument('--layer_depth', type=int, default=LAYER_DEPTH, help='Number of layer depth')
    parser.add_argument('--learning_type', type=str, default='inference', required=False)

    return parser.parse_args()


def main():
    args = parse_args()

    dataset = args.dataset
    model_name = args.model

    args.seed = random.randint(0, 1000000)

    output_folder = None
    if model_name == "adv":
        output_folder = conf.output_folder_bgnn_adv + "/" + str(dataset)
    elif model_name == "mlp":
        output_folder = conf.output_folder_bgnn_mlp + "/" + str(dataset)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    seed_file = output_folder + "/random_seed.txt"
    fs = open(seed_file, 'w')
    wstr = "%s" % str(args.seed)
    fs.write(wstr + "\n")
    fs.close()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")

    torch.autograd.set_detect_anomaly(True)

    # load the bipartite graph data
    dataloader = DataLoader(args.dataset, link_pred=True, rec=False)
    model_config = {
        "attr_one_dim": dataloader.set_one.shape[1],
        "attr_two_dim": dataloader.set_two.shape[1],
        "set_one_size": dataloader.set_one.shape[0],
        "set_two_size": dataloader.set_two.shape[0],
        "epoch": args.epochs,
        "layer_depth": args.layer_depth,
        "device": device,
        "disc_hid_dim": args.dis_hidden,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "batch_size": args.batch_size
    }

    if args.model == 'adv':
        # start training
        bgnn = BGNNAdversarial(dataloader, model_config)
        bgnn.adversarial_learning()
    elif args.model == 'mlp':
        bgnn = BGNNMLP()
        bgnn.relation_learning()

    with open('./output_model_tuning_link_prediction.txt', 'a') as f:
        f.write("epoch: {}, layer_depth: {}, disc_hid_dim: {}, "
                "learning_rate: {}, weight_decay: {}, dropout: {}, batch_size: {}\n".format(model_config['epoch'],
                                                                                          model_config['layer_depth'],
                                                                                          model_config['disc_hid_dim'],
                                                                                          model_config['learning_rate'],
                                                                                          model_config['weight_decay'],
                                                                                          model_config['dropout'],
                                                                                          model_config['batch_size']))
    f.close()


if __name__ == '__main__':
    main()
