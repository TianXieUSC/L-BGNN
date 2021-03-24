from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import pandas as pd
import argparse

# figure form
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

# default parameters
pca_dimensions = 50
tsne_dimensions = 2
num_clusters = {'cora': 7,
                'citeseer': 6,
                'pubmed': 3,
                'tencent': 2,
                'yelp': 3}


def t_sne(data, _pca=True):
    X = data
    if _pca and X.shape[1] > pca_dimensions:
        pca = PCA(n_components=pca_dimensions)
        X = pca.fit_transform(X)
    X_embedded = TSNE(n_components=tsne_dimensions).fit_transform(X)
    return X_embedded


if __name__ == '__main__':
    emb_path = './output/bgnn-adv/yelp'
    # # metapath2vec
    # emb_path = '/Users/tian/Documents/P4_Bipartite_Graph_Representation/journel_version/baselines/metapath2vec/yelp/link_prediction'
    # bine
    # emb_path = '/Users/tian/Documents/P4_Bipartite_Graph_Representation/journel_version/baselines/BiNE/data/yelp/link_prediction'
    # # hegan
    # emb_path = '/Users/tian/Documents/P4_Bipartite_Graph_Representation/journel_version/baselines/HeGAN/results/yelp/link_prediction'
    set_one_emb = sp.load_npz(emb_path + '/set_one.npz').toarray()

    label_path = '/Users/tian/Documents/P4_Bipartite_Graph_Representation/journel_version/datasets/pretrained/yelp'
    label = sp.load_npz(label_path + '/business_label.npz').toarray()

    data = set_one_emb[label[:, 0]]
    labels = label[:, 1]
    x = t_sne(data)

    for i in range(num_clusters['yelp']):
        x_cat = x[np.where(labels == i)]
        plt.scatter(x_cat[:, 0], x_cat[:, 1], s=3)
    # plt.legend(['labels: {}'.format(i) for i in range(len(set(labels)))])
    # plt.title('Yelp', fontsize=20, weight='bold')
    plt.axis('off')
    plt.show()
