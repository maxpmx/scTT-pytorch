import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import ArgumentParser
import pytorch_lightning as pl
import wandb
import einops
from torchmetrics.functional import accuracy

class PredLayer(nn.Module):
    """Some Information about PredLayer"""
    def __init__(self, n_genes, n_val, embed_dim):
        super(PredLayer, self).__init__()
        self.n_val = n_val
        self.proj_v = nn.Linear(embed_dim, n_val)

    def forward(self, x, y_v=None):
        scores_v = self.proj_v(x)
        scores_v = F.log_softmax(scores_v, dim=1)
        if y_v == None:
            return scores_v
        else:
            # loss_g = F.cross_entropy(scores_g.view(-1,self.n_genes), y_g, reduction='mean')
            loss_v = F.nll_loss(scores_v.view(-1,self.n_val), y_v)
            return loss_v

class AnnLayer(nn.Module):
    """Some Information about AnnLayer"""
    def __init__(self, embed_dim, n_labels):
        super(AnnLayer, self).__init__()
        self.proj = nn.Linear(embed_dim, n_labels)
        self.bn = nn.BatchNorm1d(n_labels)

    def forward(self, x):
        x = self.proj(x)
        # x = self.bn(x)
        x = F.log_softmax(x, dim=1)
        prob = F.softmax(x, dim=1)
        return x, prob


def discriminate(x,y,label_1,label_2):
    criterion_1 = nn.HingeEmbeddingLoss()
    criterion_2 = nn.HingeEmbeddingLoss()
    x = torch.abs(x-y)
    x = torch.mean(x,1)
    loss_1 = criterion_1(x,label_1)
    loss_2 = criterion_2(x,label_2)

    return loss_1 + loss_2

def max_pool(x):
    return torch.max(x, dim=1)[0]

def sum_pool(x):
    return x.sum(1)

def mean_pool(x):
    return torch.mean(x, dim=1)

class Block(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(Block, self).__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim)
        )

    def forward(self, x, mask):
        x = self.ln_1(x)
        a, _ = self.attn(x, x, x, mask, need_weights=False)
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x


class TransformerModel(pl.LightningModule):
    def __init__(self, params, n_species=2):
        super(TransformerModel, self).__init__()
        self.embed_dim = params.embed_dim
        self.n_heads = params.n_heads
        self.n_layers = params.n_layers
        self.n_species = n_species
        self.n_genes = params.n_genes
        self.n_val = params.n_val
        self.n_tissue = params.n_tissue
        self.n_celltype = params.n_celltype
        self.gene2id = params.gene2id
        self.params = params
        self.pretrain = params.pretrain

        self.token_embeddings = nn.Embedding(self.n_genes, self.embed_dim)
        self.val_embeddings = nn.Embedding(self.n_val, self.embed_dim)
        self.species_embeddings = nn.Embedding(self.n_species, self.embed_dim)

        self.layers = nn.ModuleList()
        for _ in range(params.n_layers):
            self.layers.append(Block(self.embed_dim, self.n_heads))
        self.pred_layer = PredLayer(self.n_genes, self.n_val, self.embed_dim)
        self.ann_celltype = AnnLayer(self.embed_dim, self.n_celltype)
        self.ann_tissue = AnnLayer(self.embed_dim, self.n_tissue)
        self.val_head = nn.Linear(self.embed_dim, self.n_val)
        self.gene_head = nn.Linear(self.embed_dim, self.n_genes)

        if params.pooling == 'max':
            self.pool = max_pool
        elif params.pooling == 'mean':
            self.pool = mean_pool
        elif params.pooling == 'sum':
            self.pool = sum_pool

    def forward(self, batch, batch_idx=0):
        """
        Expect input as shape [sequence len, batch]
        If classify, return classification logits
        """
        exp, val, species, mask = batch
        exp = exp.T
        val = val.T
        length, batch = exp.shape

        h_e = self.token_embeddings(exp) # [sequence len, batch, embed_dim]
        h_v = self.val_embeddings(val)
        token_species = species.expand(length, batch)
        h_s = self.species_embeddings(token_species)
        h = h_e + h_v + h_s
        # transformer
        for layer in self.layers:
            h = layer(h,mask)
        h = h.permute(1,0,2)
        h_pool = self.pool(h)

        return h_pool

    def training_step(self, batch, batch_idx=0):

        exp, val, mask, tissue, celltype, species = batch
        exp = exp.T
        val = val.T

        length, batch_size = exp.shape

        h_e = self.token_embeddings(exp) # [sequence len, batch, embed_dim]
        h_v = self.val_embeddings(val)
        token_species = einops.repeat(species,'b -> l b',l=length)
        # token_species = einops.repeat(species,'b -> b l',l=length)
        h_s = self.species_embeddings(token_species)
        h = h_e + h_v + h_s
        # transformer
        for layer in self.layers:
            h = layer(h,mask)
        h = h.permute(1,0,2)
        h_pool = self.pool(h)

        a_pool = h_pool

        if self.pretrain:
            pred_val = self.gene_head(h)
            loss = F.cross_entropy(pred_val.view(-1,self.n_genes),exp.T.flatten())
        else:
            x_ct, prob_ct = self.ann_celltype(a_pool)
            # loss_ts = F.nll_loss(x_ts, tissue)
            loss_ct = F.nll_loss(x_ct, celltype)
            loss = loss_ct

        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx=0):
        exp, val, mask, tissue, celltype, species = batch
        exp = exp.T
        val = val.T
        length, batch_size = exp.shape
        # batch_size, length = exp.shape

        h_e = self.token_embeddings(exp) # [sequence len, batch, embed_dim]
        h_v = self.val_embeddings(val)
        token_species = einops.repeat(species,'b -> l b',l=length)
        # token_species = einops.repeat(species,'b -> b l',l=length)
        h_s = self.species_embeddings(token_species)
        h = h_e + h_v + h_s
        # transformer
        for layer in self.layers:
            h = layer(h,mask)
        h = h.permute(1,0,2)
        if self.pretrain:
            pred_val = self.gene_head(h)
            loss = F.cross_entropy(pred_val.view(-1,self.n_genes),exp.T.flatten())
            self.log('valid_loss', loss)
        else:
            h_pool = self.pool(h)
            # x_ts, prob_ts = self.ann_tissue(h_pool)
            x_ct, prob_ct = self.ann_celltype(h_pool)
            # loss_ts = F.nll_loss(x_ts, tissue)
            loss_ct = F.nll_loss(x_ct, celltype)

            self.log('valid_loss', loss_ct)
            preds = torch.argmax(x_ct, dim=1)
            acc = accuracy(preds, celltype)
            self.log('valid_acc', loss_ct)
        # self.log('val_loss_ct', loss_ct)

    def test_step(self, batch, batch_idx=0):
        exp, val, mask, tissue, celltype, species = batch
        exp = exp.T
        val = val.T
        # mask = mask.T
        length, batch_size = exp.shape

        h_e = self.token_embeddings(exp) # [sequence len, batch, embed_dim]
        h_v = self.val_embeddings(val)
        token_species = einops.repeat(species,'b -> l b',l=length)
        h_s = self.species_embeddings(token_species)
        h = h_e + h_v + h_s
        # transformer
        for layer in self.layers:
            h = layer(h,mask)
        h = h.permute(1,0,2)
        h_pool = self.pool(h)

        x_ts, prob_ts = self.ann_tissue(h_pool)
        x_ct, prob_ct = self.ann_celltype(h_pool)

        loss_ts = F.nll_loss(x_ts, tissue)
        loss_ct = F.nll_loss(x_ct, celltype)

        # pred_ts = x_ts
        # conf_ts = pred_ts.max(dim=1)[0]
        # pred_ts = pred_ts.argmax(dim=1, keepdim=True)

        # pred_ct = x_ct
        # conf_ct = pred_ct.max(dim=1)[0]
        # pred_ct = pred_ct.argmax(dim=1, keepdim=True)

        # correct_ts = pred_ts.eq(tissue.view_as(pred_ts)).sum().item()
        # correct_ct = pred_ct.eq(celltype.view_as(pred_ct)).sum().item()

        # acc_ts = correct_ts/len(exp)
        # acc_ct = correct_ct/len(exp)

        # self.log('test_acc_ts', acc_ts)
        # self.log('test_acc_ct', acc_ct)

        return x_ts, prob_ts, x_ct, prob_ct, loss_ts, loss_ct, h_pool

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.params.lr)


if __name__ == "__main__":
    from params import get_parser
    parser = get_parser()
    params = parser.parse_args()
    model = TransformerModel(params)
