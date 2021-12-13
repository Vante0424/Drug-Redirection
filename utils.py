# The MIT License
#
# Copyright (c) 2016 Thomas Kipf
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import pickle as pkl
import sys

import networkx as nx
import numpy as np
import scipy.sparse as sp


# 处理index文件并返回index矩阵
def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))  # line.strip()表示删除掉数据中的换行符
    return index  # 返回一个index列表


# 创建mask并返回mask矩阵
def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)  # 返回类型布尔


# 读取数据
def load_data(dataset_str):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => 训练实例的特征向量
        the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => 测试实例的特征向量
        the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => 有标签的+无无标签训练实例的特征向量
        the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => 训练实例的标签，独热编码，numpy.ndarray类的实例
        the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => 测试实例的标签，独热编码，numpy.ndarray类的实例
        the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => 有标签的+无无标签训练实例的标签，独热编码，numpy.ndarray类的实例
        the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => 图数据，collections.defaultdict类的实例，格式为 {index：[index_of_neighbor_nodes]}
        a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict object;
    ind.dataset_str.test.index => 测试实例的id
        the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module. 上述文件必须都用python的pickle模块存储

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    print('utils--------------------------------------------------show')
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):  # len(names)=7
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):  # 查看版本
                '''
                python的pickle模块实现了基本的数据序列和反序列化。
                通过pickle模块的序列化操作我们能够将程序中运行的对象信息保存到文件中去，永久存储。
                通过pickle模块的反序列化操作，我们能够从文件中创建上一次程序保存的对象。
                '''
                objects.append(pkl.load(f, encoding='latin1'))  # 从文件f中重构python对象
                '''
                pickle.load(file)
                从file中读取一个字符串，并将它重构为原来的python对象。
                file：类文件对象，有read()和readline()接口。
                '''
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)

    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))  # 得到一个1行index列表
    # print('test_idx_recoder:', format(test_idx_reorder))

    test_idx_range = np.sort(test_idx_reorder)  # 对列表index进行按行排序（index列表是1行n列）
    # print('test_idx_range:', format(test_idx_range))

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    print(allx.shape)
    print(tx.shape)
    features = sp.vstack((allx, tx)).tolil()  # 转换为链表稀疏矩阵lil
    features = sp.lil_matrix(allx)
    # print('features:', format(features))

    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))  # 从字典生成图
    # print('graph:', format(graph))
    # print()
    # print('adj:', format(adj))

    labels = sp.vstack((ally, ty)).A
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    # print('labels:', format(labels))
    idx_test = test_idx_range.tolist()
    # print('idx_test:', format(idx_test))

    idx_train = range(y.shape[0])
    # print('idx_train:', format(idx_train))  # idx_train: range(0, 140)

    idx_val = range(y.shape[0], y.shape[0]+500)
    # print('idx_val:', format(idx_val))  # idx_val: range(140, 640)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])
    # print('train_mask:', format(train_mask))  # train_mask: [ True  True  True ... False False False]
    # print('val_mask:', format(val_mask))  # val_mask: [False False False ... False False False]
    # print('test_mask:', format(test_mask))  # test_mask: [False False False ...  True  True  True]

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)

    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
    # print('y_train:', format(y_train))
    # print('y_val:', format(y_val))
    # print('y_test:', format(y_test))

    return adj, features, labels, train_mask, val_mask, test_mask


# 处理特征：将特征进行归一化并返回tuple (coords, values, shape)
def preprocess_features(features):
    # print('啦啦啦啦啦啦啦啦啦啦啦啦啦')
    """Row-normalize feature matrix and convert to tuple representation"""
    # 归一化函数实现的方式：对传入特征矩阵的每一行分别求和，取到数后就是每一行非零元素归一化的值，然后与传入特征矩阵进行点乘。
    rowsum = np.array(features.sum(1))  # 会得到一个（2708, 1）的矩阵
    r_inv = np.power(rowsum, -1).flatten()  # 得到（2708, ）的元组
    # 在计算倒数的时候存在一个问题，如果原来的值为0，则其倒数为无穷大，因此需要对d_inv_sqrt中无穷大的值进行修正，更改为0
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


# 图归一化并返回
def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


# 处理得到GCN中的归一化矩阵并返回
def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return adj_normalized
