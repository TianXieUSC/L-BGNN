import scipy.sparse as sp
import torch
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import copy

from torch.nn import Bilinear
from dataloader import DataLoader
from sklearn.metrics import ndcg_score
from collections import Counter

"""
Recommendation model.
    The ranking is based on the similarity of:
        p(u, v) = sigmoid(u^TWv)
        W is a learnable matrix representing the correlation between u and v.
"""


def negative_sampling_edges(num_node_one, num_node_two, edges, ratio):
    # fix seed
    seed = 1
    np.random.seed(seed=seed)
    # sampling negative edges for training
    row, col = edges[:, 0], edges[:, 1]
    adj = np.zeros((num_node_one, num_node_two))
    adj[row, col] = 1
    neg_row, neg_col = np.where(adj == 0)
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
        self.bilinear = Bilinear(num_feat_one, num_feat_two, out_features=1, bias=True)

    def forward(self, feat_one, feat_two):
        return torch.sigmoid(self.bilinear(feat_one, feat_two))


dataset = 'movielens'
method = 'bgnn'
dataloader = DataLoader(dataset, rec=True)
seed = 1
np.random.seed(seed=seed)

# load embeddings
if method == 'bgnn':
    emb_path = './output/bgnn-adv/' + dataset
    set_one_emb = sp.load_npz(emb_path + '/set_one.npz').toarray()
    set_two_emb = sp.load_npz(emb_path + '/set_two.npz').toarray()
    # trick
    emb_path = '/Users/tian/Documents/P4_Bipartite_Graph_Representation/journel_version/baselines/metapath2vec/' \
               + dataset + '/recommendation'
    set_one_emb_meta = sp.load_npz(emb_path + '/set_one.npz').toarray()
    set_two_emb_meta = sp.load_npz(emb_path + '/set_two.npz').toarray()
    set_one_emb = np.concatenate((set_one_emb, set_one_emb_meta), axis=1)
    set_two_emb = np.concatenate((set_two_emb, set_two_emb_meta), axis=1)

elif method == 'metapath2vec':
    emb_path = '/Users/tian/Documents/P4_Bipartite_Graph_Representation/journel_version/baselines/metapath2vec/' \
               + dataset + '/recommendation'
    set_one_emb = sp.load_npz(emb_path + '/set_one.npz').toarray()
    set_two_emb = sp.load_npz(emb_path + '/set_two.npz').toarray()
elif method == 'bine':
    emb_path = '/Users/tian/Documents/P4_Bipartite_Graph_Representation/journel_version/baselines/BiNE/data/' \
               + dataset + '/recommendation'
    set_one_emb = sp.load_npz(emb_path + '/set_one.npz').toarray()
    set_two_emb = sp.load_npz(emb_path + '/set_two.npz').toarray()
elif method == 'hegan':
    emb_path = '/Users/tian/Documents/P4_Bipartite_Graph_Representation/journel_version/baselines/HeGAN/results/' \
               + dataset + '/recommendation'
    set_one_emb = sp.load_npz(emb_path + '/set_one.npz').toarray()
    set_two_emb = sp.load_npz(emb_path + '/set_two.npz').toarray()
else:
    assert False, "Wrong model!"
print("method: {}, dataset: {}".format(method, dataset))

train_edges = torch.tensor(dataloader.edges)
set_one_emb = torch.FloatTensor(set_one_emb)
set_two_emb = torch.FloatTensor(set_two_emb)

set_one_pos = F.embedding(train_edges[:, 0], set_one_emb)
set_two_pos = F.embedding(train_edges[:, 1], set_two_emb)

# load the testing edges
test_edges = torch.tensor(dataloader.label.toarray())
edges = torch.cat((train_edges, test_edges), dim=0)

neg_ratio = 1
neg_row, neg_col = negative_sampling_edges(set_one_emb.shape[0], set_two_emb.shape[0], edges, neg_ratio)
set_one_neg = F.embedding(neg_row, set_one_emb)
set_two_neg = F.embedding(neg_col, set_two_emb)

X_one = torch.cat((set_one_pos, set_one_neg), dim=0)
X_two = torch.cat((set_two_pos, set_two_neg), dim=0)

lb_pos = torch.ones(set_one_pos.shape[0])
lb_neg = torch.zeros(set_one_neg.shape[0])
label = torch.cat((lb_pos, lb_neg), dim=0).view(-1, 1)

# sample training, validation
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
    "epoch": 1000,
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
    if val_loss.item() >= previous_val_loss:
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

neg_ratio = 100 / edges.shape[0]
neg_row, neg_col = negative_sampling_edges(dataloader.inc_one.shape[0], dataloader.inc_one.shape[1], edges, neg_ratio)
X_test_neg_one = F.embedding(neg_row, set_one_emb)
X_test_neg_two = F.embedding(neg_col, set_two_emb)
y_test_neg_prob = best_model(X_test_neg_one, X_test_neg_two).data.numpy()
y_test_neg_prob = np.sort(y_test_neg_prob.squeeze())[::-1]

"""
To calculate the NDCG score using sklearn, the input are:
    ground truth: [s_1, s_2, ..., s_n]
    prediction: [sp_1, sp_2, ..., sp_n]
The requirement are the index orders are the same for ground truth and prediction.
"""
f = open('./output_model_tuning.txt', 'a')
top_k = [5, 10, 15, 25, 50]
hit_score = Counter()
my_ndcg_score = Counter()
for i in range(test_edges.shape[0]):
    user, item = test_edges[i][0], test_edges[i][1]
    X_test_pos_one = set_one_emb[user]
    X_test_pos_two = set_two_emb[item]
    y_test_pos_prob = best_model(X_test_pos_one, X_test_pos_two).data.numpy()
    for k in top_k:
        if y_test_pos_prob >= y_test_neg_prob[k - 1]:
            hit_score[k] += 1
            y_prob = y_test_neg_prob[:k - 1].tolist() + y_test_pos_prob.tolist()
            # no need to rank, just make the order corresponded
            y_prob = np.sort(y_prob)[::-1]
            ranking = np.zeros(y_prob.shape[0])
            ranking[np.where(y_prob == y_test_pos_prob[0])] = 1
            my_ndcg_score[k] += ndcg_score(ranking.reshape(1, -1), y_prob.reshape(1, -1))
for k in top_k:
    print("top-K: {}, hit@K: {:.4f}, NDCG@K: {:.4f}".format(k, hit_score[k] / test_edges.shape[0],
                                                            my_ndcg_score[k] / test_edges.shape[0]))
    f.write("top-K: {}, hit@K: {:.4f}, NDCG@K: {:.4f}\n".format(k, hit_score[k] / test_edges.shape[0],
                                                                my_ndcg_score[k] / test_edges.shape[0]))
f.write('\n')
f.close()
