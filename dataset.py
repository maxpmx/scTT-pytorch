from logging import getLogger
import math
import numpy as np
import pandas as pd
import torch
import scanpy as sc
from sklearn.model_selection import train_test_split
from numba import jit
import anndata as ad
from argparse import ArgumentParser
from torch.utils.data import DataLoader
import os
import random
# tok = ['PAD','UNK','SAT','END']

def build_dict(adata_1, adata_2):
    gene_list_1 = adata_1.var_names
    gene_list_2 = adata_2.var_names

    m_list = list(np.load('data/m_list.npy'))
    h_list = list(np.load('data/h_list.npy'))

    adata_1 = adata_1[:,m_list]
    adata_2 = adata_2[:,h_list]

    adata_1.var_names = h_list
    adata_2.var_names = h_list

    all_tissues = list(set(adata_1.obs['tissue']) | set(adata_2.obs['tissue']))
    id2tissue = dict(enumerate(all_tissues))
    tissue2id = {v:u for u,v in id2tissue.items()}

    all_celltype = list(set(adata_1.obs['celltype']) | set(adata_2.obs['celltype']))
    id2celltype = dict(enumerate(all_celltype))
    celltype2id = {v:u for u,v in id2celltype.items()}

    return m_list, adata_1, adata_2, id2tissue,tissue2id, id2celltype, celltype2id

def csr2sentence(csr):
    indices = csr.indices
    indptr = csr.indptr
    data = csr.data
    all_exp = []
    all_val = []
    n = len(indptr)-1
    max_len = 0
    for i in range(n):
        all_exp.append(indices[indptr[i]:indptr[i+1]])
        val = data[indptr[i]:indptr[i+1]]
        all_val.append(val)
        max_len = max(max_len,len(val))
    exps = np.zeros((n,max_len))
    vals = np.zeros((n,max_len))
    for i in range(n):
        l = len(all_exp[i])
        exps[i][:l] = all_exp[i]
        vals[i][:l] = all_val[i]

    return exps, vals

class SingleDataset(torch.utils.data.Dataset):
    """Some Information about SingleDataset"""
    def __init__(self, adata_m,  gene2id, tissue2id=None, celltype2id=None, shuffle=True):
        super(SingleDataset, self).__init__()
        self.adata_m = adata_m

        self.shuffle = shuffle
        self.tissue_m = self.adata_m.obs['tissue']
        self.celltype_m = self.adata_m.obs['celltype']
        self.species_m = self.adata_m.obs['species']


        self.tissue2id = tissue2id
        self.celltype2id = celltype2id

        self.m_indices = adata_m.X.indices
        self.m_indptr = adata_m.X.indptr
        self.m_data = adata_m.X.data
        self.n_mouse = len(self.m_indptr)-1

        lengths_m = self.m_indptr[1:] - self.m_indptr[:self.n_mouse]

        self.lengths_all = lengths_m

        self.indices = np.argsort(self.lengths_all, kind='mergesort')

        self.dic = dict(enumerate(self.indices))
    def __getitem__(self, index):
        idx = index

        exp = self.m_indices[self.m_indptr[idx]:self.m_indptr[idx+1]]
        val = self.m_data[self.m_indptr[idx]:self.m_indptr[idx+1]]

        if self.shuffle:
            exp = torch.Tensor(exp)
            val = torch.Tensor(val)
            rand_id = torch.randperm(exp.shape[0])
            exp = exp[rand_id]
            val = val[rand_id]
        if self.tissue2id == None:
            return exp, val

        tissue = self.tissue2id[self.tissue_m[idx]]
        celltype = self.celltype2id[self.celltype_m[idx]]
        species = self.species_m[idx]

        return exp, val, tissue, celltype, species

    def __len__(self):
        n = len(self.lengths_all)

        return np.floor(0.95*n).astype(int)

class MixDataset(torch.utils.data.Dataset):
    """Some Information about MixDataset"""
    def __init__(self, adata_m, adata_h, gene2id, tissue2id=None, celltype2id=None, shuffle=True):
        super(MixDataset, self).__init__()
        self.adata_m = adata_m
        self.adata_h = adata_h

        self.shuffle = shuffle

        self.tissue_m = self.adata_m.obs['tissue']
        self.celltype_m = self.adata_m.obs['celltype']
        self.species_m = self.adata_m.obs['species']
        self.tissue_h = self.adata_h.obs['tissue']
        self.celltype_h = self.adata_h.obs['celltype']
        self.species_h = self.adata_h.obs['species']

        self.tissue2id = tissue2id
        self.celltype2id = celltype2id

        self.m_indices = adata_m.X.indices
        self.m_indptr = adata_m.X.indptr
        self.m_data = adata_m.X.data
        self.n_mouse = len(self.m_indptr)-1

        self.h_indices = adata_h.X.indices
        self.h_indptr = adata_h.X.indptr
        self.h_data = adata_h.X.data
        self.n_human = len(self.h_indptr)-1

        lengths_m = self.m_indptr[1:] - self.m_indptr[:self.n_mouse]
        lengths_h = self.h_indptr[1:] - self.h_indptr[:self.n_human]

        self.lengths_all = np.concatenate((lengths_m, lengths_h))

        self.indices = np.argsort(self.lengths_all, kind='mergesort')

        self.dic = dict(enumerate(self.indices))
    def __getitem__(self, index):
        # idx = self.dic[index]
        idx = index
        if idx < self.n_mouse:

            exp = self.m_indices[self.m_indptr[idx]:self.m_indptr[idx+1]]
            val = self.m_data[self.m_indptr[idx]:self.m_indptr[idx+1]]

            if self.shuffle:
                exp = torch.Tensor(exp)
                val = torch.Tensor(val)
                rand_id = torch.randperm(exp.shape[0])
                exp = exp[rand_id]
                val = val[rand_id]

            if self.tissue2id == None:
                return exp, val
            tissue = self.tissue2id[self.tissue_m[idx]]
            celltype = self.celltype2id[self.celltype_m[idx]]
            species = self.species_m[idx]
        else:
            idx -= self.n_mouse

            exp = self.h_indices[self.h_indptr[idx]:self.h_indptr[idx+1]]
            val = self.h_data[self.h_indptr[idx]:self.h_indptr[idx+1]]

            if self.shuffle:
                exp = torch.Tensor(exp)
                val = torch.Tensor(val)
                rand_id = torch.randperm(exp.shape[0])
                exp = exp[rand_id]
                val = val[rand_id]

            if self.tissue2id == None:
                return exp, val
            tissue = self.tissue2id[self.tissue_h[idx]]
            celltype = self.celltype2id[self.celltype_h[idx]]
            species = self.species_h[idx]

        return exp, val, tissue, celltype, species

    def __len__(self):
        n = len(self.lengths_all)

        return np.floor(0.6*n).astype(int)


class AdDataset(MixDataset):
    def __len__(self):
        n = (len(self.lengths_all)-1)//2-1
        return np.floor(0.85*n).astype(int)

    def get_item(self, index):
        idx = self.dic[index]
        if idx < self.n_mouse:
            exp = self.m_indices[self.m_indptr[idx]:self.m_indptr[idx+1]]
            val = self.m_data[self.m_indptr[idx]:self.m_indptr[idx+1]]

            if self.shuffle:
                exp = torch.Tensor(exp)
                val = torch.Tensor(val)
                rand_id = torch.randperm(exp.shape[0])
                exp = exp[rand_id]
                val = val[rand_id]

            if self.tissue2id == None:
                return exp, val
            tissue = self.tissue2id[self.tissue_m[idx]]
            celltype = self.celltype2id[self.celltype_m[idx]]
            species = self.species_m[idx]
        else:
            idx -= self.n_mouse
            exp = self.h_indices[self.h_indptr[idx]:self.h_indptr[idx+1]]
            val = self.h_data[self.h_indptr[idx]:self.h_indptr[idx+1]]

            if self.shuffle:
                exp = torch.Tensor(exp)
                val = torch.Tensor(val)
                rand_id = torch.randperm(exp.shape[0])
                exp = exp[rand_id]
                val = val[rand_id]

            if self.tissue2id == None:
                return exp, val
            tissue = self.tissue2id[self.tissue_h[idx]]
            celltype = self.celltype2id[self.celltype_h[idx]]
            species = self.species_h[idx]

        return exp, val, tissue, celltype, species

    def __getitem__(self,index):

        exp1, val1, tissue1, celltype1, species1 = self.get_item(2*index)
        exp2, val2, tissue2, celltype2, species2 = self.get_item(2*index+1)
        l1 = 1 if tissue1==tissue2 else -1
        l2 = 1 if celltype1==celltype2 else -1
        return exp1, val1, species1, exp2, val2,species2, l1, l2, tissue1, celltype1, tissue2, celltype2

def collect_fn(data):
    batch_size = len(data)
    max_len = 0
    for i in range(batch_size):
        length = len(data[i][0])
        max_len = max(max_len,length)

    exp = np.zeros((batch_size, max_len))
    mask = np.ones((batch_size, max_len))
    val = np.zeros((batch_size, max_len))
    tissue = np.zeros((batch_size))
    celltype = np.zeros((batch_size))
    species = np.zeros((batch_size))

    for i in range(batch_size):
        length = len(data[i][0])
        exp[i][:length] = data[i][0]
        mask[i][:length] = 0
        val[i][:length] = data[i][1]
        tissue[i] = data[i][2]
        celltype[i] = data[i][3]
        species[i] = data[i][4]

    exp = torch.LongTensor(exp)
    val = torch.LongTensor(val)
    mask = torch.BoolTensor(mask)
    tissue = torch.LongTensor(tissue)
    celltype = torch.LongTensor(celltype)
    species = torch.LongTensor(species)

    return exp, val, mask, tissue, celltype, species

def collect_ad(data):
    max_len = len(data[-1][3])
    batch_size = len(data)

    exp_1 = np.zeros((batch_size, max_len))
    val_1 = np.zeros((batch_size, max_len))
    mask_1 = np.ones((batch_size, max_len))
    species_1 = np.zeros((batch_size))
    exp_2 = np.zeros((batch_size, max_len))
    val_2 = np.zeros((batch_size, max_len))
    mask_2 = np.ones((batch_size, max_len))
    species_2 = np.zeros((batch_size))
    l_1 = np.zeros((batch_size))
    l_2 = np.zeros((batch_size))
    tissue_1 = np.zeros((batch_size))
    celltype_1 = np.zeros((batch_size))
    tissue_2 = np.zeros((batch_size))
    celltype_2 = np.zeros((batch_size))

    for i in range(batch_size):
        try:
            length = len(data[i][0])
            length_2 = len(data[i][3])
            exp_1[i][:length] = data[i][0]
            val_1[i][:length] = data[i][1]
            mask_1[i][:length] = 0
            species_1[i] = data[i][2]
            exp_2[i][:length_2] = data[i][3]
            val_2[i][:length_2] = data[i][4]
            mask_2[i][:length_2] = 0
            species_2[i] = data[i][5]
            l_1[i] = data[i][6]
            l_2[i] = data[i][7]
            tissue_1[i] = data[i][8]
            celltype_1[i] = data[i][9]
            tissue_2[i] = data[i][10]
            celltype_2[i] = data[i][11]
        except:
            print(i)
            print(data[i][0])
            print(length)
            print(max_len)

    exp_1 = torch.LongTensor(exp_1)
    val_1 = torch.LongTensor(val_1)
    mask_1 = torch.BoolTensor(mask_1)
    species_1 = torch.LongTensor(species_1)
    exp_2 = torch.LongTensor(exp_2)
    val_2 = torch.LongTensor(val_2)
    mask_2 = torch.BoolTensor(mask_2)
    species_2 = torch.LongTensor(species_2)
    l_1 = torch.Tensor(l_1)
    l_2 = torch.Tensor(l_2)
    tissue_1 = torch.LongTensor(tissue_1)
    celltype_1 = torch.LongTensor(celltype_1)
    tissue_2 = torch.LongTensor(tissue_2)
    celltype_2 = torch.LongTensor(celltype_2)

    return exp_1,val_1,mask_1,species_1,exp_2,val_2,mask_2,species_2,l_1,l_2,tissue_1,celltype_1,tissue_2,celltype_2



def load_data(params,dir='/ibex/scratch/pangm0a/data/'):
    if not os.path.exists(dir):
        dir='/home/ubuntu/data/'
    if params.dataset == 'hcl':
        hcl = sc.read(dir+'hcl_csr_norm.h5ad')
        mca = sc.read(dir+'mca_csr_norm.h5ad')
        sc.pp.filter_cells(hcl, min_genes=100)
        sc.pp.filter_cells(mca, min_genes=100)

        mca_ct = mca[mca.obs['celltype']==pd.Series(['T cell']).astype('category')[0]]
        mca_filtered = mca[mca.obs['celltype']!=pd.Series(['T cell']).astype('category')[0]]
        mca_ts = mca_filtered[mca_filtered.obs['tissue']==pd.Series(['AdultPancreas']).astype('category')[0]]
        mca_filtered = mca_filtered[mca_filtered.obs['tissue']!=pd.Series(['AdultPancreas']).astype('category')[0]]

        return mca_filtered, hcl, mca_ts, mca_ct

    elif params.dataset == 'brain':
        mctx = sc.read(dir+'mctx_csr_norm.h5ad')
        hctx = sc.read(dir+'hctx_csr_norm.h5ad')
        mctx.obs['tissue'] = mctx.obs['region']
        hctx.obs['tissue'] = hctx.obs['region']

        hctx_ts = hctx[hctx.obs['celltype']==pd.Series(['Astrocyte']).astype('category')[0]]
        hctx_filtered = hctx[hctx.obs['celltype']!=pd.Series(['Astrocyte']).astype('category')[0]]
        hctx_ct = hctx[hctx.obs['celltype']==pd.Series(['Oligodendrocyte']).astype('category')[0]]
        hctx_filtered = hctx_filtered[hctx_filtered.obs['celltype']!=pd.Series(['Oligodendrocyte']).astype('category')[0]]

        return mctx, hctx_filtered, hctx_ts, hctx_ct



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', default='hcl', type=str)
    params = parser.parse_args()
    adata_m, adata_h, adata_ts, adata_ct = load_data(params)
    if params.dataset == 'hcl':
        adata_m.obs['species'] = 0
        adata_h.obs['species'] = 1
        adata_ct.obs['species'] = 0
        adata_ts.obs['species'] = 0

    elif params.dataset == 'brain':
        adata_m.obs['species'] = 0
        adata_h.obs['species'] = 1
        adata_ct.obs['species'] = 1
        adata_ts.obs['species'] = 1
    gene_list, id2gene, gene2id, id2tissue, tissue2id, id2celltype, celltype2id = build_dict(adata_m, adata_h)
    params.n_genes = len(gene2id)+1
    params.gene2id = gene2id
    params.n_val = 11  # max(adata_m.X.max(), adata_h.X.max()) + 1 = 579
    params.n_tissue = len(tissue2id)
    params.n_celltype = len(celltype2id)
    print('Dictionary builded, gene size: {}, number of tissue: {}, number of celltype: {}'.format(params.n_genes,params.n_tissue,params.n_celltype))

    dataset_train = AdDataset(adata_m, adata_h, gene2id, tissue2id,celltype2id)
    dataset_test = MixDataset(adata_ts, adata_ct, gene2id, tissue2id,celltype2id)

    train_loader = DataLoader(dataset_train, batch_size=2, collate_fn=collect_ad, num_workers=2,drop_last=True)
    test_loader = DataLoader(dataset_test, batch_size=2, collate_fn=collect_fn, num_workers=2,drop_last=True)

    print(dataset_train[1])
    print(dataset_train[10])
    print(dataset_train[10000])
    print(dataset_train[100000])
