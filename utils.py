import torch
import numpy as np
import torch.utils.data as d


class Feeder(d.DataLoader):
    def __init__(self, feat_one, feat_two, label):
        self.feat_one = feat_one
        self.feat_two = feat_two
        self.label = label

    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, item):
        feat_one = self.feat_one[item]
        feat_two = self.feat_two[item]
        label = self.label[item]
        return feat_one, feat_two, label


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def batch_normalization(feat):
    # normalize based on each channel
    # (d - mean) / std
    mean = torch.mean(feat, dim=0)
    std = torch.std(feat, dim=0)
    return (feat - mean) / std
