import warnings
import numpy as np
import collections
import operator
import pickle
import sys
import random
import os
from scipy import sparse
from sklearn import metrics

import cPickle as pickle
repo_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/'
data_dir = '/oak/stanford/groups/rbaltman/swang91/Sheng_repo/data/'
sys.path.append(repo_dir)
os.chdir(repo_dir)

from model import GRep
from util import *
from pikachu.datasets.BioNetwork import BioNetwork
from pikachu.datasets.FunctionAnnotation import FunctionAnnotation
from pikachu.models.random_walk_with_restart.RandomWalkRestart import RandomWalkRestart

if len(sys.argv) <= 2:
    DCA_dim = 100
    dge_dim = 100
    nhidden = 5
    drug_cor_cutoff = 0.5#0.3
    lr = 0.01
    pathway_rst = 0.9
    DCA_rst = 0.5
    dataset = 'ctrp_drug'
    optimize_gene_vec = True
else:
    DCA_dim = int(sys.argv[1])
    dge_dim = str(sys.argv[2])
    nhidden = int(sys.argv[3])
    drug_cor_cutoff = float(sys.argv[4])
    lr = float(sys.argv[5])
    pathway_rst = float(sys.argv[6])
    DCA_rst = float(sys.argv[7])
    dataset = sys.argv[8]
    optimize_gene_vec = sys.argv[9]


#warnings.simplefilter(action='ignore', category=FutureWarning)
net_file_l = []
net_file_l.append(data_dir + 'network/human/string_integrated.txt')
Net_obj = BioNetwork(net_file_l)
network = Net_obj.sparse_network.toarray()
i2g = Net_obj.i2g
g2i = Net_obj.g2i
nnode = len(i2g)

print 'nnode:',nnode,'nedge:',Net_obj.nedge
if dataset == 'gdsc_drug':
    fin = open('data/drug/gdsc/top_corr_genes.txt')
    path2gene = {}
    p2i = {}
    i2p = {}
    nd = 0
    for line in fin:
        w = line.upper().strip().split('\t')
        cor = float(w[2])
        if abs(cor) < drug_cor_cutoff:
            continue
        d = w[1]
        if w[0] not in g2i:
            continue
        if d not in path2gene:
            path2gene[d] = set()
            p2i[d] = nd
            i2p[nd] = d
            nd += 1
        path2gene[d].add(g2i[w[0]])
    fin.close()
    fin = open('data/drug/gdsc/drug_target_mapped.txt')
    path2label = {}
    for line in fin:
        w = line.upper().strip().split('\t')
        if len(w)==1:
            continue
        if w[0] not in p2i:
            continue
        path2label[w[0]] = set()
        gset = w[1].split(';')
        for i in gset:
            if i in g2i:
                path2label[w[0]].add(g2i[i])
    fin.close()
elif dataset == 'ctrp_drug':
    fin = open('data/drug/ctrp/drug_map.txt')
    d2dname = {}
    for line in fin:
        w = line.upper().strip().split('\t')
        d2dname[w[2]] = w[0]
    fin.close()
    fin = open('data/NLP_Dictionary/top_genes_exp_hgnc.txt')
    path2gene = {}
    p2i = {}
    i2p = {}
    nd = 0
    for line in fin:
        w = line.upper().strip().split('\t')
        cor = float(w[2])
        if abs(cor) < drug_cor_cutoff:
            continue
        d = d2dname[w[1]]
        if w[0] not in g2i:
            continue
        if d not in path2gene:
            path2gene[d] = set()
            p2i[d] = nd
            i2p[nd] = d
            nd += 1
        path2gene[d].add(g2i[w[0]])
    fin.close()
    fin = open('data/drug/ctrp/drug_target.txt')
    path2label = {}
    for line in fin:
        w = line.upper().strip().split('\t')
        if len(w)==1:
            continue
        if d2dname[w[0]] not in p2i:
            continue
        path2label[d2dname[w[0]]] = set()
        gset = w[1].split(';')
        for i in gset:
            if i in g2i:
                path2label[d2dname[w[0]]].add(g2i[i])
    fin.close()
else:
    print dataset
    sys.exit('wrong dataset'+dataset)

npath = len(path2gene)


RWR_dump_file = 'data/network/embedding/my_dca/RWR_'+str(DCA_rst)
if os.path.isfile(RWR_dump_file):
    Node_RWR = pickle.load(open(RWR_dump_file, "rb" ))
else:
    Node_RWR = RandomWalkRestart(network, DCA_rst)
    with open(RWR_dump_file, 'wb') as output:
        pickle.dump(Node_RWR, output, pickle.HIGHEST_PROTOCOL)



Pathway_rwr_dump_file = 'data/network/embedding/my_dca/'+dataset+'_RWR_'+str(pathway_rst)+'_'+str(drug_cor_cutoff)
if os.path.isfile(Pathway_rwr_dump_file):
    Path_RWR = pickle.load(open(Pathway_rwr_dump_file, "rb" ))
else:
    Path_RWR = RandomWalkRestart(network, pathway_rst, reset = Path_mat)
    print 'calcualte emb finished'
    with open(Pathway_rwr_dump_file, 'wb') as output:
        pickle.dump(Path_RWR, output, pickle.HIGHEST_PROTOCOL)


alpha = 1./(nnode*nnode)
log_Path_RWR = np.log(Path_RWR +alpha) - np.log(alpha)
log_node_RWR = np.log(Node_RWR +alpha) - np.log(alpha)
auc_d = {}
g2g = GRep(log_Path_RWR, log_node_RWR, path2gene,auc_d, p2i, path2label,
optimize_gene_vec=optimize_gene_vec, alpha = alpha,lr =lr, L=dge_dim, n_hidden = [nhidden],max_iter=3000,seed=0)
Path_our = g2g.train()
