# ScT-pytorch
<img src="fig1.png" width="800px"></img>
# Example
```python
import torch
from module import ScT
from dataset import SingleDataset
from torch.utils.data import DataLoader

sct = ScT(
    n_genes = 10000,
    n_val = 11,
    n_class = 1000,
    pretrain=False,
    pooling='mean',
    embed_dim=768,n_heads=8,
    n_layers=8,
    lr=1e-4,
    n_species=2
)

dataset = SingleDataset(adata,gene2id)
loader = DataLoader(dataset)

sct.fit(model, loader)
```
# Parameters
- `image_size`: int.  
  Image size. If you have rectangular images, make sure your image size is the maximum of the width and height
- `patch_size`: int.  
  Number of patches. `image_size` must be divisible by `patch_size`.  
  The number of patches is: ` n = (image_size // patch_size) ** 2` and `n` **must be greater than 16**.
- `num_classes`: int.  
  Number of classes to classify.
- `dim`: int.  
  Last dimension of output tensor after linear transformation `nn.Linear(..., dim)`.
- `depth`: int.  
  Number of Transformer blocks.
- `heads`: int.  
  Number of heads in Multi-head Attention layer.
- `mlp_dim`: int.  
  Dimension of the MLP (FeedForward) layer.
- `channels`: int, default `3`.  
  Number of image's channels.
- `dropout`: float between `[0, 1]`, default `0.`.  
  Dropout rate.
- `emb_dropout`: float between `[0, 1]`, default `0`.  
  Embedding dropout rate.
- `pool`: string, either `cls` token pooling or `mean` pooling


## Pretrain


# Citation
