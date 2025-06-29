"""
# -*- coding: utf-8 -*-
# @Author : Sun JJ
# @File : GCN_Model.py
# @Time : 2022/5/9 9:16
# code is far away from bugs with the god animal protecting
#         ┌─┐       ┌─┐
#      ┌──┘ ┴───────┘ ┴──┐
#      │                 │
#      │       ───       │
#      │  ─┬┘       └┬─  │
#      │                 │
#      │       ─┴─       │
#      │                 │
#      └───┐         ┌───┘
#          │         │
#          │         │
#          │         │
#          │         └──────────────┐
#          │                        │
#          │                        ├─┐
#          │                        ┌─┘
#          │                        │
#          └─┐  ┐  ┌───────┬──┐  ┌──┘
#            │ ─┤ ─┤       │ ─┤ ─┤
#            └──┴──┘       └──┴──┘
"""


import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset,DataLoader
import pickle

SEED = 2020
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    torch.cuda.manual_seed(SEED)


LAYER = 6 # 原来是10
ALPHA = 0.6 # alpha参数用于平衡原始节点特征与其邻居特征的重要性。
LAMBDA = 1.3 # 控制着图卷积过程中自连接（self-connection）的权重,基于超参数lamda和层的深度l进行动态调整。
VARIANT = True
DROPOUT = 0.2 # Dropout 比率在 0.2 到 0.5 之间会更加合理
INPUT_DIM = 71 # 71 改成自己的维度
HIDDEN_DIM = 256 # 256改成了512

BATCH_SIZE = 1  # 可以考虑 32、64或128，训练得更快
NUM_CLASSES = 2
NUMBER_EPOCHS = 20
WEIGHT_DECAY = 1e-5  #  当前权重衰减设置为 0,这意味着没有使用 L2 正则化。适当的权重衰减可以防止模型过拟合,提高泛化能力。你可以尝试设置一个较小的权重衰减值(如 1e-5 或 5e-4),看看是否能提升模型性能。
LEARNING_RATE = 1E-3   # 原来是3 可以降低成一半或十分之一，有可能会更加稳定并更容易收敛
POSITIVE_WEIGHT = 3.0  # 正样本的权重，可以根据实际情况调整

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 对图的邻接矩阵进行特定的变换，使得图的传播过程更加平滑，有助于避免梯度爆炸或消失的问题，从而提高模型的训练效率和性能。
def normalize(mx):

    rowsum = np.array(mx.sum(1))
    r_inv = (rowsum ** -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0
    r_mat_inv = np.diag(r_inv)
    result = np.dot(np.dot(r_mat_inv,mx),r_mat_inv)

    return result

def load_graph(sequence_name):

    fpath = './data/scPDB/adjacency_matrix14/' + sequence_name + '.npy'
    adjacency_matrix = np.load(fpath)
    norm_matrix = normalize(adjacency_matrix.astype(np.float32))

    return norm_matrix

def get_node_features(sequence_name):

    fpath = './data/scPDB/node_features/' + sequence_name + '.npy'
    node_features = np.load(fpath)

    return node_features

def cal_edges(sequence_name, radius=15):  # to get the index of the edges
    dist_matrix = np.load('./data/scPDB/adjacency_matrix14/' + sequence_name + '.npy')
    # fpath = './data/adjacency_matrix14/' + sequence_name + '.npy'
    # mask = ((dist_matrix >= 0) * (dist_matrix <= radius))
    adjacency_matrix = dist_matrix.astype(np.int64)
    radius_index_list = np.where(adjacency_matrix == 1)
    radius_index_list = [list(nodes) for nodes in radius_index_list]
    return radius_index_list

# 用于封装图数据的加载和处理。它包括数据的加载、预处理和格式化，使得数据可以直接被模型使用。
class ProDataset(Dataset):

    def __init__(self,dataframe):

        self.names = dataframe['ID'].values
        self.sequences = dataframe['sequence'].values
        self.labels = dataframe['label'].values

    def __getitem__(self,index):

        sequence_name = self.names[index]
        # print(1)
        sequence = self.sequences[index]
        # print(2)
        label = np.array(self.labels[index])
        # print(3)

        try:
            node_features = get_node_features(sequence_name)
            # print(4)
            graph = load_graph(sequence_name)
            # print(5)
        except Exception as e:
            print(f"[ERROR] Loading failed for {sequence_name}: {e}")
            raise e

        return sequence_name,sequence,label,node_features,graph

    def __len__(self):

        return len(self.labels)

class ProDataset_agat(Dataset):
    def __init__(self, dataframe, radius=14, dist=15, psepos_path='./data/scPDB/PCA/PCA_psepos_SC.pkl'):
        self.names = dataframe['ID'].values
        self.sequences = dataframe['sequence'].values
        self.labels = dataframe['label'].values
        self.residue_psepos = pickle.load(open(psepos_path, 'rb'))
        self.radius = radius
        self.dist = dist


    def __getitem__(self, index):
        sequence_name = self.names[index]
        sequence = self.sequences[index]
        label = np.array(self.labels[index])
        pos = self.residue_psepos[sequence_name]
        reference_res_psepos = pos[0]
        pos = pos - reference_res_psepos
        pos = torch.from_numpy(pos)
        edge_index = cal_edges(sequence_name)
        node_features = get_node_features(sequence_name)
        #node_features = node_features.detach().numpy()
        node_features = node_features[np.newaxis, :, :]
        node_features = torch.from_numpy(node_features).type(torch.FloatTensor)
        return sequence_name, sequence, label, node_features, edge_index, pos

    def __len__(self):
        return len(self.labels)

    def cal_edge_attr(self, index_list, pos):
        pdist = nn.PairwiseDistance(p=2,keepdim=True)
        cossim = nn.CosineSimilarity(dim=1)

        distance = (pdist(pos[index_list[0]], pos[index_list[1]]) / self.radius).detach().numpy()
        cos = ((cossim(pos[index_list[0]], pos[index_list[1]]).unsqueeze(-1) + 1) / 2).detach().numpy()
        radius_attr_list = np.array([distance, cos])
        return radius_attr_list

    def add_edges_custom(self, G, radius_index_list, edge_features):
        src, dst = radius_index_list[1], radius_index_list[0]
        if len(src) != len(dst):
            print('source and destination array should have been of the same length: src and dst:', len(src), len(dst))
            raise Exception
        G.add_edges(src, dst)
        G.edata['ex'] = torch.tensor(edge_features)



# 这是图神经网络的基础组件，用于定义图卷积操作。该模型包含权重参数和前向传播的定义，可以处理图结构数据并提取图的空间特征。
class GraphConvolution(nn.Module):

    def __init__(self,in_features,out_features,residual = False,variant = False):
        super(GraphConvolution,self).__init__()
        self.variant = variant
        if self.variant:
            self.in_features = 2 * in_features
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(self.in_features,self.out_features))
        self.reset_parameters()

    def reset_parameters(self):

        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self,input,adj,h0,lamda,alpha,l):

        theta = min(1,math.log(lamda / l + 1))
        hi = torch.spmm(adj,input)

        if self.variant:
            support = torch.cat([hi,h0],1)
            r = (1 - alpha) * hi + alpha * h0
        else:
            support = (1 - alpha) * hi + alpha * h0
            r = support
        output = theta * torch.mm(support,self.weight) + (1- theta) * r

        if self.residual:
            output = output + input

        return output

class deepGCN(nn.Module):

    def __init__(self,nlayers,nfeat,nhidden,nclass,dropout,lamda,alpha,variant):

        super(deepGCN,self).__init__()
        self.convs = nn.ModuleList()

        for i in range(nlayers):
            self.convs.append(GraphConvolution(nhidden,nhidden,variant = variant,residual = True))

        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat,nhidden))
        self.fcs.append(nn.Linear(nhidden,nclass))
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda

    def forward(self,x,adj):

        _layers = []
        x = F.dropout(x,self.dropout,training = self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)

        for i,con in enumerate(self.convs):

            layer_inner = F.dropout(layer_inner,self.dropout,training = self.training)
            layer_inner = self.act_fn(con(layer_inner,adj,_layers[0],self.lamda,self.alpha,i + 1))

        layer_inner = F.dropout(layer_inner,self.dropout,training = self.training)
        layer_inner = self.fcs[-1](layer_inner)

        return layer_inner

class GraphPLBR(nn.Module):

    def __init__(self,nlayers,nfeat,nhidden,nclass,dropout,lamda,alpha,variant):

        super(GraphPLBR,self).__init__()

        self.deep_gcn = deepGCN(
            nlayers = nlayers,
            nfeat = nfeat,
            nhidden = nhidden,
            nclass = nclass,
            dropout = dropout,
            lamda = lamda,
            alpha = alpha,
            variant = variant
        )
        weights = torch.tensor([1.0, POSITIVE_WEIGHT], dtype=torch.float)
        # self.criterion = nn.BCELoss()
        self.criterion = nn.CrossEntropyLoss(weight=weights)
        self.optimizer = torch.optim.Adam(self.parameters(),lr = LEARNING_RATE,weight_decay = WEIGHT_DECAY)

    def forward(self,x,adj):

        x = x.float()
        output = self.deep_gcn(x,adj)

        return output