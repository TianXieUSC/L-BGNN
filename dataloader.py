import scipy.sparse as sp
import numpy as np
import torch
from utils import sparse_mx_to_torch_sparse_tensor


def load_imdb_dataset():
    path = '/Users/tian/Documents/P4_Bipartite_Graph_Representation/journel_version/datasets/imdb/preprocessed'
    user = sp.load_npz(path + '/user.npz')
    movie = sp.load_npz(path + '/movie.npz')
    edges = sp.load_npz(path + '/edges.npz')
    return user, movie, None, edges


def load_movielens_pretrain_dataset(link_pred=False, rec=False):
    ## shift PPMI pretrain
    # path = '/Users/tian/Documents/P4_Bipartite_Graph_Representation/journel_version/datasets/pretrained/movielens'
    # metapath2vec pretrain
    path = '/Users/tian/Documents/P4_Bipartite_Graph_Representation/journel_version/baselines/metapath2vec/movielens/recommendation'
    user = sp.load_npz(path + '/set_one.npz')
    movie = sp.load_npz(path + '/set_two.npz')
    label = None
    path = '/Users/tian/Documents/P4_Bipartite_Graph_Representation/journel_version/datasets/pretrained/movielens'
    if rec:
        path += '/recommendation'
        # the labels are the masked edges, one edge per user
        label = sp.load_npz(path + '/test.npz')
        edges = sp.load_npz(path + '/edges.npz')
    elif link_pred:
        path += '/link_prediction'
        edges = []
        edges.append(sp.load_npz(path + '/train_edges.npz'))
        edges.append(sp.load_npz(path + '/test_edges.npz'))
    elif link_pred and rec:
        assert False, "Wrong task."
    else:
        edges = sp.load_npz(path + '/edges.npz')
    return user, movie, label, edges


def load_movielens_dataset(link_pred=False, rec=False):
    path = '/Users/tian/Documents/P4_Bipartite_Graph_Representation/journel_version/datasets/ml-1m/preprocessed'
    user = sp.load_npz(path + '/user.npz')
    movie = sp.load_npz(path + '/movie.npz')
    label = None
    if rec:
        path += '/recommendation'
        # the labels are the masked edges, one edge per user
        label = sp.load_npz(path + '/test.npz')
        edges = sp.load_npz(path + '/edges.npz')
    elif link_pred:
        path += '/link_prediction'
        edges = []
        edges.append(sp.load_npz(path + '/train_edges.npz'))
        edges.append(sp.load_npz(path + '/test_edges.npz'))
    elif link_pred and rec:
        assert False, "Wrong task."
    else:
        edges = sp.load_npz(path + '/edges.npz')
    return user, movie, label, edges


def load_dblp_dataset(link_pred=False):
    path = '/Users/tian/Documents/P4_Bipartite_Graph_Representation/journel_version/datasets/pretrained/dblp'
    author = sp.load_npz(path + '/two.npz')
    paper = sp.load_npz(path + '/one.npz')
    label = None
    if link_pred:
        path += '/link_prediction'
        edges = []
        edges.append(sp.load_npz(path + '/train_edges.npz'))
        edges.append(sp.load_npz(path + '/test_edges.npz'))
    else:
        edges = sp.load_npz(path + '/edges.npz')
    return paper, author, label, edges


def load_wiki_dataset(link_pred=False):
    path = '/Users/tian/Documents/P4_Bipartite_Graph_Representation/journel_version/datasets/pretrained/wiki'
    one = sp.load_npz(path + '/one.npz')
    two = sp.load_npz(path + '/two.npz')
    label = None
    if link_pred:
        path += '/link_prediction'
        edges = []
        edges.append(sp.load_npz(path + '/train_edges.npz'))
        edges.append(sp.load_npz(path + '/test_edges.npz'))
    else:
        edges = sp.load_npz(path + '/edges.npz')
    return one, two, label, edges


def load_amazon_dataset(link_pred=False):
    path = '/Users/tian/Documents/P4_Bipartite_Graph_Representation/journel_version/datasets/pretrained/amazon'
    one = sp.load_npz(path + '/one_v2.npz')
    two = sp.load_npz(path + '/two_v2.npz')
    label = None
    if link_pred:
        path += '/link_prediction'
        edges = []
        edges.append(sp.load_npz(path + '/train_edges.npz'))
        edges.append(sp.load_npz(path + '/test_edges.npz'))
    else:
        edges = sp.load_npz(path + '/edges.npz')
    return one, two, label, edges


def load_yelp_dataset(link_pred=False):
    path = '/Users/tian/Documents/P4_Bipartite_Graph_Representation/journel_version/datasets/pretrained/yelp'
    one = sp.load_npz(path + '/one.npz')
    two = sp.load_npz(path + '/two.npz')
    label = None
    if link_pred:
        path += '/link_prediction'
        edges = []
        edges.append(sp.load_npz(path + '/train_edges.npz'))
        edges.append(sp.load_npz(path + '/test_edges.npz'))
    else:
        edges = sp.load_npz(path + '/edges.npz')
    return one, two, label, edges


def normalize_adj(adj):
    # random walk normalization
    adj = adj.tocoo().astype(float)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -1).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).tocsr()


def normalize_feat(feat):
    # x / sum(x)
    row_sum = np.sum(feat, axis=1).astype(float)
    r_inv = np.power(row_sum, -1)
    r_inv[np.isinf(r_inv)] = 0
    return feat * r_inv.reshape(feat.shape[0], -1)


class DataLoader(object):
    def __init__(self, dataset, link_pred=False, rec=False):
        if dataset == 'dblp':
            self.set_one, self.set_two, self.label, self.edges = load_dblp_dataset(link_pred=link_pred)
        elif dataset == 'imdb':
            self.set_one, self.set_two, self.label, self.edges = load_imdb_dataset()
        elif dataset == 'movielens':
            self.set_one, self.set_two, self.label, self.edges = load_movielens_dataset(link_pred=link_pred, rec=rec)
        elif dataset == 'wiki':
            self.set_one, self.set_two, self.label, self.edges = load_wiki_dataset(link_pred=link_pred)
        elif dataset == 'amazon':
            self.set_one, self.set_two, self.label, self.edges = load_amazon_dataset(link_pred=link_pred)
        elif dataset == 'yelp':
            self.set_one, self.set_two, self.label, self.edges = load_yelp_dataset(link_pred=link_pred)
        else:
            assert False, "Wrong dataset!"

        self.dataset = dataset

        # build the incidence matrix for two sets
        if link_pred:
            train_edges, test_edges = self.edges
            print('Link prediction task; train edges: {}, test edges: {}'.format(train_edges.shape[0],
                                                                                 test_edges.shape[0]))
            self.train_edges = train_edges.toarray()
            self.test_edges = test_edges.toarray()
            self.inc_one = sp.csr_matrix(
                (np.ones(self.train_edges.shape[0]), (self.train_edges[:, 0], self.train_edges[:, 1])),
                shape=(self.set_one.shape[0], self.set_two.shape[0]))
            self.inc_two = self.inc_one.T
        else:
            self.edges = self.edges.toarray()
            self.inc_one = sp.csr_matrix((np.ones(self.edges.shape[0]), (self.edges[:, 0], self.edges[:, 1])),
                                         shape=(self.set_one.shape[0], self.set_two.shape[0]))
            self.inc_two = self.inc_one.T
        self.set_one = self.set_one.toarray()
        self.set_two = self.set_two.toarray()

        # normalize the adjacency matrix
        self.inc_one = normalize_adj(self.inc_one)
        self.inc_two = normalize_adj(self.inc_two)

        # normalize the feature matrix
        # self.set_one = normalize_feat(self.set_one)
        # self.set_two = normalize_feat(self.set_two)
        print('Data loaded.')


if __name__ == '__main__':
    user, movie, edges = load_imdb_dataset()
    print(user.shape)
    print(movie.shape)
    print(edges.shape)
