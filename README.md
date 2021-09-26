# ScT-pytorch

# Usage
```python
import torch
from module import ScT

v = ScT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

img = torch.randn(1, 3, 256, 256)

preds = v(img) # (1, 1000)
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
