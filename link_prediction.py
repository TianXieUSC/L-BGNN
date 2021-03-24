import argparse

import scipy.sparse as sp
import torch
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import copy

from torch.nn import Bilinear
from dataloader import DataLoader
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

"""
Link prediction model.
    The probability of predicting as an edge is:
        p(u, v) = sigmoid(u^TWv)
        W is a learnable matrix representing the correlation between u and v.
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='amazon')
    parser.add_argument('--neg_ratio', type=float, default=1.)
    return parser.parse_args()


def negative_sampling_edges(num_node_one, num_node_two, edges, ratio):
    # sampling negative edges for training
    row, col = edges[:, 0], edges[:, 1]
    adj = np.zeros((num_node_one, num_node_two))
    adj[row, col] = 1
    neg_row, neg_col = np.where(adj == 0)
    seed = 1
    np.random.seed(seed=seed)
    samples = np.random.randint(len(neg_row), size=int(len(row) * ratio))
    sample_row, sample_col = torch.tensor(neg_row[samples]), torch.tensor(neg_col[samples])
    return sample_row, sample_col


def accuracy(preds, labels):
    """Multilabel accuracy with masking."""
    preds = preds > 0.5
    labels = labels > 0.5
    correct = torch.eq(preds, labels).type(torch.float)
    return correct.mean()


class LinkPrediction(torch.nn.Module):
    def __init__(self, num_feat_one, num_feat_two):
        super(LinkPrediction, self).__init__()
        self.bilinear = Bilinear(num_feat_one, num_feat_two, out_features=1, bias=False)

    def forward(self, feat_one, feat_two):
        return torch.sigmoid(self.bilinear(feat_one, feat_two))


args = parse_args()
dataset = args.dataset
method = 'bgnn'
dataloader = DataLoader(dataset, link_pred=True)
seed = 1
np.random.seed(seed=seed)

# load embeddings
if method == 'bgnn':
    emb_path = './output/bgnn-adv/' + dataset
    set_one_emb = sp.load_npz(emb_path + '/set_one.npz').toarray()
    set_two_emb = sp.load_npz(emb_path + '/set_two.npz').toarray()
elif method == 'metapath2vec':
    emb_path = '/Users/tian/Documents/P4_Bipartite_Graph_Representation/journel_version/baselines/metapath2vec/' \
               + dataset + '/link_prediction'
    set_one_emb = sp.load_npz(emb_path + '/set_one.npz').toarray()
    set_two_emb = sp.load_npz(emb_path + '/set_two.npz').toarray()
elif method == 'bine':
    emb_path = '/Users/tian/Documents/P4_Bipartite_Graph_Representation/journel_version/baselines/BiNE/data/my_' \
               + dataset + '/link_prediction'
    set_one_emb = sp.load_npz(emb_path + '/set_one.npz').toarray()
    set_two_emb = sp.load_npz(emb_path + '/set_two.npz').toarray()
elif method == 'hegan':
    emb_path = '/Users/tian/Documents/P4_Bipartite_Graph_Representation/journel_version/baselines/HeGAN/results/' \
               + dataset + '/link_prediction'
    set_one_emb = sp.load_npz(emb_path + '/set_one.npz').toarray()
    set_two_emb = sp.load_npz(emb_path + '/set_two.npz').toarray()
else:
    assert False, "Wrong model!"
print("method: {}, dataset: {}".format(method, dataset))

# load train and test edges
train_edges = torch.tensor(dataloader.train_edges)
test_edges_pos = torch.tensor(dataloader.test_edges)
set_one_emb = torch.FloatTensor(set_one_emb)
set_two_emb = torch.FloatTensor(set_two_emb)

set_one_pos = F.embedding(train_edges[:, 0], set_one_emb)
set_two_pos = F.embedding(train_edges[:, 1], set_two_emb)

# negative sampling: all \ {train, test}
neg_ratio = args.neg_ratio
edges = torch.cat((train_edges, test_edges_pos), dim=0)
neg_row, neg_col = negative_sampling_edges(dataloader.inc_one.shape[0], dataloader.inc_one.shape[1], edges, neg_ratio)

# select 20% as testing, 80% as training
test_idx = test_edges_pos.shape[0]
test_edges_neg = torch.stack((neg_row[:test_idx], neg_col[:test_idx])).T
neg_row = neg_row[test_idx:]
neg_col = neg_col[test_idx:]
set_one_neg = F.embedding(neg_row, set_one_emb)
set_two_neg = F.embedding(neg_col, set_two_emb)

X_one = torch.cat((set_one_pos, set_one_neg), dim=0)
X_two = torch.cat((set_two_pos, set_two_neg), dim=0)

lb_pos = torch.ones(set_one_pos.shape[0])
lb_neg = torch.zeros(set_one_neg.shape[0])
label = torch.cat((lb_pos, lb_neg), dim=0).reshape(-1, 1)

# sample training, validation and testing
training_ratio = 0.8
train_mask_ind = np.random.choice(X_one.shape[0], int(X_one.shape[0] * training_ratio), replace=False)
val_mask_ind = np.setdiff1d(range(X_one.shape[0]), train_mask_ind)

train_mask = np.zeros(X_one.shape[0], dtype=np.bool)
train_mask[train_mask_ind] = True
val_mask = np.zeros(X_one.shape[0], dtype=np.bool)
val_mask[val_mask_ind] = True

model_args = {
    "num_feat_one": set_one_emb.shape[1],
    "num_feat_two": set_two_emb.shape[1],
    "learning_rate": 0.001,
    "weight_decay": 1e-4,
    "epoch": 500,
    "early_stop": 10
}
model = LinkPrediction(model_args['num_feat_one'], model_args['num_feat_two'])
optimizer = optim.Adam(model.parameters(), lr=model_args['learning_rate'], weight_decay=model_args['weight_decay'])

previous_val_loss = np.inf
early_stop = 0
best_model = copy.deepcopy(model)

if torch.cuda.is_available():
    X_one = X_one.cuda()
    X_two = X_two.cuda()
    label = label.cuda()
    model = model.cuda()

for i in range(model_args['epoch']):
    model.train()
    optimizer.zero_grad()

    y_prob = model(X_one, X_two)
    train_loss = F.binary_cross_entropy(y_prob[train_mask], label[train_mask])

    train_loss.backward()
    optimizer.step()

    train_acc = accuracy(y_prob[train_mask], label[train_mask])
    val_loss = F.binary_cross_entropy(y_prob[val_mask], label[val_mask])
    val_acc = accuracy(y_prob[val_mask], label[val_mask])
    if val_loss.item() > previous_val_loss:
        if early_stop == 0:
            best_model = copy.deepcopy(model)
        early_stop += 1
    else:
        early_stop = 0
    previous_val_loss = val_loss.item()

    print('epoch: {}, train loss: {:.4f}, train acc: {:.4f}, '
          'val loss: {:.4f}, val accuracy: {:.4f}'.format(i, train_loss, train_acc, val_loss, val_acc))
    if early_stop == model_args['early_stop']:
        break

if early_stop != model_args['early_stop']:
    best_model = copy.deepcopy(model)
best_model.eval()
best_model = best_model.cpu()

f = open('./output_model_tuning_link_prediction.txt', 'a')
# the negative examples are random sampled from no existed edge.
test_edges = torch.cat((test_edges_pos[:, :2], test_edges_neg), dim=0)
set_one_test = F.embedding(test_edges[:, 0], set_one_emb)
set_two_test = F.embedding(test_edges[:, 1], set_two_emb)
lb_pos = torch.ones(test_edges_pos.shape[0])
lb_neg = torch.zeros(test_edges_neg.shape[0])
label = torch.cat((lb_pos, lb_neg), dim=0).reshape(-1, 1)

y_prob = best_model(set_one_test, set_two_test).detach()
test_acc = accuracy(y_prob, label)

# auc
y_prob = y_prob.numpy().squeeze()
label = label.numpy().squeeze()
auc = roc_auc_score(label, y_prob)

# f1 score
y_prob[y_prob >= 0.5] = 1
y_prob[y_prob < 0.5] = 0
f1score_binary = f1_score(label, y_prob, average='binary')
f1score_micro = f1_score(label, y_prob, average='micro')
f1score_macro = f1_score(label, y_prob, average='macro')
print("test accuracy: {:.4f}, binary (default) f1-score: {:.4f}, "
      "micro f1-score: {:.4f}, macro f1-score: {:.4f}, auc score: {:.4f}".
      format(test_acc, f1score_binary, f1score_micro, f1score_macro, auc))
f.write("test accuracy: {:.4f}, binary (default) f1-score: {:.4f}, "
        "micro f1-score: {:.4f}, macro f1-score: {:.4f}\n".
        format(test_acc, f1score_binary, f1score_micro, f1score_macro))
f.write('\n')
f.close()
