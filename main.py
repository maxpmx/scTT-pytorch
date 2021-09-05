from argparse import ArgumentParser
from params import get_parser
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
import json
import random
from dataset import MixDataset, AdDataset, SingleDataset, collect_fn, collect_ad, load_data, build_dict
from modules import TransformerModel
import warnings
from torch.utils.data import DataLoader
import scanpy as sc
from torch.nn.utils import clip_grad_norm_
import time
import numpy as np
import anndata as ad
warnings.filterwarnings('ignore')
sc.set_figure_params(scanpy=True, dpi=300, dpi_save=300)

def cli_main():
    pl.seed_everything(1234)

    # ------------
    # args
    # ------------
    # parser = ArgumentParser()
    parser = get_parser()
    parser.add_argument('--precision', default=16, type=int)
    # parser = pl.Trainer.add_argparse_args(parser)
    # parser = LitClassifier.add_model_specific_args(parser)
    parser.add_argument('--pretrain', default=False, type=bool)
    parser.add_argument('--shuffle', default=True, type=bool)
    params = parser.parse_args()

    # ------------
    # data
    # ------------
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
    gene_list, adata_m, adata_h, id2tissue, tissue2id, id2celltype, celltype2id = build_dict(adata_m, adata_h)
    adata_ts = adata_ts[:,gene_list]
    adata_ct = adata_ct[:,gene_list]
    adata_ts.var_names = gene_list
    adata_ct.var_names = gene_list

    adata_test = ad.concat([adata_ts, adata_ct])

    gene2id = None
    params.n_genes = len(gene_list)+1
    params.gene2id = gene2id
    # params.n_val = 11  # max(adata_m.X.max(), adata_h.X.max()) + 1 = 579
    params.n_tissue = len(tissue2id)
    params.n_celltype = len(celltype2id)
    print('Dictionary builded, gene size: {}, number of tissue: {}, number of celltype: {}'.format(params.n_genes,params.n_tissue,params.n_celltype))

    dataset_train = MixDataset(adata_m, adata_h, gene2id, tissue2id, celltype2id)
    tr_len = int(dataset_train.__len__()*0.9)
    val_len = dataset_train.__len__() - tr_len
    ds_train, ds_val = random_split(dataset_train, [tr_len, val_len])
    dataset_test = MixDataset(adata_ts, adata_ct, gene2id, tissue2id,celltype2id)
    # dataset_test = SingleDataset(adata_test, gene2id, tissue2id,celltype2id)

    train_loader = DataLoader(ds_train, batch_size=params.batch_size, collate_fn=collect_fn, num_workers=params.n_workers,drop_last=True, shuffle=params.shuffle)
    val_loader = DataLoader(ds_val, batch_size=params.batch_size, collate_fn=collect_fn, num_workers=params.n_workers,drop_last=True)
    test_loader = DataLoader(dataset_test, batch_size=params.batch_size, collate_fn=collect_fn, num_workers=params.n_workers,drop_last=True)
    # ------------
    # model
    # ------------
    model = TransformerModel(params)
    print('The number of parameters: {}'.format(count_parameters(model)))
    model = TransformerModel.load_from_checkpoint("model/last_"+str(24)+".ckpt",params=params)

    # ------------
    # training
    # ------------

    # wandb_logger = WandbLogger(project='SCT')

    # # wandb_logger.watch(model)

    checkpoint_callback = ModelCheckpoint(monitor='train_loss')
    trainer = pl.Trainer(gpus=-1,
                            accelerator='dp', precision=params.precision,
                            # logger=wandb_logger,
                            max_epochs=params.n_epochs, gradient_clip_val=0.5, callbacks=[checkpoint_callback])
    trainer.fit(model, train_loader, val_loader)
    trainer.save_checkpoint("model/last_"+params.experiment+".ckpt")
    # # # ------------
    # # # testing
    # # # ------------
    trainer.test(test_dataloaders=test_loader)


if __name__ == '__main__':
    cli_main()
