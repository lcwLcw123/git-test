# -*- code = utf-8 -*-
# @Time: 2022/3/30 13:11
# @Author: Chen Zigeng
# @File:DGI_FIND_4.py
# @Software:PyCharm
import os
import copy
import json
import pickle
import argparse
import numpy as np
import scipy.sparse as sp

import torch
from models.GCN_dgl import GCN
from models.GAT_dgl import GAT
from models.GSAGE_dgl import GraphSAGE
from models.JKNet_dgl import JKNet
import warnings
warnings.filterwarnings("ignore")




def sample_graph_det(adj_orig, A_pred, remove_pct, add_pct):
    if remove_pct == 0 and add_pct == 0:
        return copy.deepcopy(adj_orig)
    orig_upper = sp.triu(adj_orig, 1)
    # 以稀疏格式返回矩阵的上三角部分。返回矩阵A的第1个对角线上或上方的元素
    n_edges = orig_upper.nnz
    # Number of nonzero matrix elements矩阵中的非零元素，两个节点之间有边的个数
    edges = np.asarray(orig_upper.nonzero()).T
    # edges 是2*n维的数据
    # asarray（）转化为array格式，nonzero（）输出非零元素的坐标--->得到存在edges的坐标
    if remove_pct:
        n_remove = int(n_edges * remove_pct / 100)
        pos_probs = A_pred[edges.T[0], edges.T[1]]
        # 取出从源点到终点的edge 概率
        e_index_2b_remove = np.argpartition(pos_probs, n_remove)[:n_remove]
        # 找出前n_remove概率最小的边
        mask = np.ones(len(edges), dtype=bool)
        mask[e_index_2b_remove] = False
        edges_pred = edges[mask]
    #    edges_pred中False 就是需要移除的
    # 只有索引是true的-才会被选出—形成的新的数组
    else:
        edges_pred = edges

    if add_pct:
        n_add = int(n_edges * add_pct / 100)
        # deep copy to avoid modifying A_pred
        A_probs = np.array(A_pred)
        # make the probabilities of the lower half to be zero (including diagonal)
        A_probs[np.tril_indices(A_probs.shape[0])] = 0
        # make the probabilities of existing edges to be zero
        A_probs[edges.T[0], edges.T[1]] = 0
        all_probs = A_probs.reshape(-1)
        e_index_2b_add = np.argpartition(all_probs, -n_add)[-n_add:]
        new_edges = []
        for index in e_index_2b_add:
            i = int(index / A_probs.shape[0])
            j = index % A_probs.shape[0]
            new_edges.append([i, j])
        edges_pred = np.concatenate((edges_pred, new_edges), axis=0)
    adj_pred = sp.csr_matrix((np.ones(len(edges_pred)), edges_pred.T), shape=adj_orig.shape)
    # 生成新的邻接矩阵 edges_pred：有边的个数 edges_pred：提供 row 和 col adj_orig.shape：按照原adj格式 没有的地方补
    adj_pred = adj_pred + adj_pred.T
    return adj_pred

def distance(a,b):
    res = 0
    for i in range(a.shape[0]):
        res -= (a[i]-b[i])*(a[i]-b[i])*1000
    return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='single')
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--gnn', type=str, default='gcn')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--i', type=str, default='2')


    args = parser.parse_args()

    if args.gpu == '-1':
        gpu = -1
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        gpu = 0

#-------------------------------------------------------------------------------------------------


    tvt_nids = pickle.load(open(f'data/graphs/{args.dataset}_tvt_nids.pkl', 'rb'))
    adj_orig = pickle.load(open(f'data/graphs/{args.dataset}_adj.pkl', 'rb'))
    features = pickle.load(open(f'data/graphs/{args.dataset}_features.pkl', 'rb'))
    labels = pickle.load(open(f'data/graphs/{args.dataset}_labels.pkl', 'rb'))
    if sp.issparse(features):  # 是否为稀疏矩阵
        features = torch.FloatTensor(features.toarray())

    params_all = json.load(open('DLauto/best_parameters.json', 'r'))  # 读取jason数据

    params = params_all['GAugM'][args.dataset][args.gnn]  #
    i = params['i']
    #A_pred = pickle.load(open(f'data/edge_probabilities/{args.dataset}_graph_{i}_logits.pkl', 'rb'))
    #A_pred = pickle.load(open(f'data/edge_probabilities/{args.dataset}_graph_{6}_logits.pkl', 'rb'))
    a = np.loadtxt('data/edge_probabilities/Vecdgi9.emb')
    # A_pred = np.array([[0 for i in range(a.shape[0])] for j in range(a.shape[0])])
    # for i in range(a.shape[0]):
    #     for j in range(a.shape[0]):
    #         A_pred[i][j] = distance(a[i], a[j])
    # for i in range(a.shape[0]):
    #     A_pred[i][i] = -1000
    #np.savetxt('./data/edge_probabilities/SDNE_apred_mse.txt', A_pred)


    A_pred = a @ a.T
    A_pred = torch.tensor(A_pred)
    A_pred = torch.sigmoid(A_pred)
    A_pred = A_pred.numpy()
    # print(type(A_pred))
    # print(A_pred.shape)
    # print(A_pred)
    #adj_pred = sample_graph_det(adj_orig, A_pred, params['rm_pct'], params['add_pct'])
    best_acc = 0
    for num in range(10,40):
        adj_pred = sample_graph_det(adj_orig, A_pred, 0, num)
        #                 "rm_pct": 2,删除概率
        #                 "add_pct": 57


        gnn = args.gnn
        if gnn == 'gcn':
            GNN = GCN
        elif gnn == 'gat':
            GNN = GAT
        elif gnn == 'gsage':
            GNN = GraphSAGE
        elif gnn == 'jknet':
            GNN = JKNet



        accs = []
        for j in range(30):

            gnn = GNN(adj_pred, adj_pred, features, labels, tvt_nids, print_progress=False, cuda=0, epochs=200)
            acc, _, _ = gnn.fit()
            #print(gnn.device)
            accs.append(acc)
            #print("********************************************************************************")
            #print("range{}     acc:{}".format(j, acc))
            #print("********************************************************************************")
        cur_acc = np.mean(accs)
        if cur_acc > best_acc:
            best_acc = cur_acc
        print('add:',num)
        print(f'Micro F1: {np.mean(accs):.6f}, std: {np.std(accs):.6f}')
    print("beat_acc:",best_acc)