import torch
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import torch.optim as optim
import copy
from dataloader import DataLoader
from sklearn.datasets import load_iris


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


class LogisticRegression(torch.nn.Module):
    def __init__(self, num_feat, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(num_feat, num_classes)

    def forward(self, feat):
        x = self.linear(feat)
        return F.log_softmax(x, dim=1)


dataset = 'dblp'
dataloader = DataLoader(dataset)
seed = 1
np.random.seed(seed=seed)

# load embeddings
method = 'bgnn'
if method == 'bgnn':
    emb_path = './output/bgnn-adv/' + dataset
    set_one_emb = sp.load_npz(emb_path + '/set_one.npz')
    set_two_emb = sp.load_npz(emb_path + '/set_two.npz')

    set_one_emb = torch.FloatTensor(set_one_emb.toarray())
    set_two_emb = torch.FloatTensor(set_two_emb.toarray())
    X_two = set_two_emb

    # load labels
    label = dataloader.label.toarray().squeeze()
    label = torch.tensor(label)

elif method == 'bine':
    emb_path = '/Users/tian/Documents/P4_Graph_Representation/journel_version/baselines/BiNE/data/my_dblp/set_two.npz'
    set_two_emb = sp.load_npz(emb_path).toarray()
    X_two = torch.FloatTensor(set_two_emb)

    # load labels
    label = dataloader.label.toarray().squeeze()
    label = torch.tensor(label)

elif method == 'metapath2vec':
    emb_path = '/Users/tian/Documents/P4_Graph_Representation/journel_version/baselines/metapath2vec/dblp'
    set_two_emb = sp.load_npz(emb_path + '/set_two.npz').toarray()
    X_two = torch.FloatTensor(set_two_emb)
    mask = sp.load_npz(emb_path + '/mask.npz').toarray().squeeze()

    # load labels
    label = dataloader.label.toarray().squeeze()
    label = torch.tensor(label)
    label = label[mask]

# # sample test
# X, y = load_iris(return_X_y=True)
# X_two = torch.FloatTensor(X)
# label = torch.tensor(y)

# sample training, validation and testing
train_ratio = 0.7
val_ratio = 0.1
train_mask_ind = np.random.choice(X_two.shape[0], int(X_two.shape[0] * train_ratio), replace=False)
val_mask_ind = np.random.choice(np.setdiff1d(range(X_two.shape[0]), train_mask_ind), int(X_two.shape[0] * val_ratio),
                                replace=False)
test_mask_ind = np.setdiff1d(range(X_two.shape[0]), np.concatenate((train_mask_ind, val_mask_ind)))

train_mask = np.zeros(X_two.shape[0], dtype=np.bool)
train_mask[train_mask_ind] = True
val_mask = np.zeros(X_two.shape[0], dtype=np.bool)
val_mask[val_mask_ind] = True
test_mask = np.zeros(X_two.shape[0], dtype=np.bool)
test_mask[test_mask_ind] = True

model_args = {
    "num_feat_two": X_two.shape[1],
    "class": len(np.unique(label)),
    "learning_rate": 0.001,
    "weight_decay": 0,
    "epoch": 1000,
    "early_stop": 50
}

model = LogisticRegression(model_args['num_feat_two'], model_args['class'])
optimizer = optim.Adam(model.parameters(), lr=model_args['learning_rate'], weight_decay=model_args['weight_decay'])

previous_val_loss = np.inf
early_stop = 0
best_model = copy.deepcopy(model)
for i in range(model_args['epoch']):
    model.train()
    optimizer.zero_grad()
    y_log_prob = model(X_two)
    train_loss = F.nll_loss(y_log_prob[train_mask], label[train_mask])
    train_loss.backward()
    optimizer.step()

    train_acc = accuracy(y_log_prob[train_mask], label[train_mask])
    val_loss = F.nll_loss(y_log_prob[val_mask], label[val_mask])
    val_acc = accuracy(y_log_prob[val_mask], label[val_mask])

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
y_log_prob = best_model(X_two[test_mask])
test_acc = accuracy(y_log_prob, label[test_mask])
print("test accuracy: {:.4f}".format(test_acc))
