# This program is referenced from https://github.com/lucidrains/vit-pytorch
import torch
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

#%% Norm
class PreNorm(nn.Module):
    def __init__(self, dim, func):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.func = func
    def forward(self, x, **kwargs):
        return self.func(self.norm(x), **kwargs)
        
#%% FeedForward
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0):
        super().__init__()
        self.net = nn.Sequential(
             nn.Linear(dim, hidden_dim),
             nn.GELU(),
             nn.Dropout(dropout),
             nn.Linear(hidden_dim, dim),
             nn.Dropout(dropout)
            )
    def forward(self, x):
        return self.net(x)
        
#%% Attention
class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head = 64, dropout = 0.1):
        super().__init__()
        inner_dim = dim_head*heads
        project = not (heads == 1 and dim_head == dim)
        
        self.heads = heads    
        self.scale = dim_head ** -0.5
        
        self.soft = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim*3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        ) if project else nn.Identity()
        
    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1) # (b, n, dim*3) ---> 3 * (b, n, dim)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv) #qkv rearrange
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.soft(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
    
#%% Transformer
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
    
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

#%% 
class ViT(nn.Module):
    def __init__(self, *, image_size, patch_height = 3, patch_width = 10, num_classes, dim, depth, heads, mlp_dim, channels=1, dim_head=128, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = 9, 50

        assert  image_height % patch_height ==0 and image_width % patch_width == 0

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches+1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim)) # nn.Parameter() set trainable class token 
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img) # b c (h p1) (w p2) -> b (h w) (p1 p2 c) -> b (h w) dim
        b, n, _ = x.shape # b = batchSize, n = total points in each pateh, _ = dim
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)  # self.cls_token: (1, 1, dim) -> cls_tokens: (batchSize, 1, dim)  
        x = torch.cat((cls_tokens, x), dim=1) # class token concatenate with patch token (b, n+1, dim)
        x += self.pos_embedding[:, :(n+1)] # add postion embadding (b, n+1, dim)
        x = self.dropout(x)

        x = self.transformer(x) # (b, n+1, dim)

        x = x[:, 0] # (b, dim)
        x = self.to_latent(x) # Identity (b, dim)
        # print(x.shape)

        return self.mlp_head(x) # (b, num_classes)
        
#%%
model_vit = ViT(
        image_size = 50,
        patch_height = 3,
        patch_width= 10,
        num_classes = 2,
        dim = 64,
        depth = 1,                                                                                                                           
        heads = 2,
        mlp_dim = 64,
        dropout = 0.1,
        emb_dropout = 0.1
    )

img = torch.randn(64, 1, 9, 50)

preds = model_vit(img) 

print(preds.shape)
