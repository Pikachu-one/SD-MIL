import torch, einops
from torch import nn
import numpy as np
from timm.models.layers import trunc_normal_
from einops import rearrange, reduce
import math
class PositionEmbedding(nn.Module):
    def __init__(self, size, dim=512):
        super().__init__()
        self.size=size
        self.pe = nn.Embedding(size+1, dim, padding_idx=0)
        self.pos_ids = torch.arange(1, size+1, dtype=torch.long).cuda()
        
    def forward(self, emb):
        device = emb.device
        b, n, *_ = emb.shape
        pos_ids = self.pos_ids
        if n > self.size:
            zeros = torch.zeros(n-self.size, dtype=torch.long, device=device)
            pos_ids = torch.cat([pos_ids, zeros])
        pos_ids = einops.repeat(pos_ids, 'n -> b n', b=b)
        pos_emb = self.pe(pos_ids) # [b n pe_dim]
        embeddings = torch.cat([emb, pos_emb], dim=-1)
        return embeddings
        
class PPEG(nn.Module):
    def __init__(self, dim=512,k=7,conv_1d=False,bias=True):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, k, 1, k//2, groups=dim,bias=bias) if not conv_1d else nn.Conv2d(dim, dim, (k,1), 1, (k//2,0), groups=dim,bias=bias)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim,bias=bias) if not conv_1d else nn.Conv2d(dim, dim, (5,1), 1, (5//2,0), groups=dim,bias=bias)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim,bias=bias) if not conv_1d else nn.Conv2d(dim, dim, (3,1), 1, (3//2,0), groups=dim,bias=bias)

    def forward(self, x):
        B, N, C = x.shape
        # padding
        H, W = int(np.ceil(np.sqrt(N))), int(np.ceil(np.sqrt(N)))
        add_length = H * W - N
        # if add_length >0:
        x = torch.cat([x, x[:,:add_length,:]],dim = 1) 
        if H < 7:
            H,W = 7,7
            zero_pad = H * W - (N+add_length)
            x = torch.cat([x, torch.zeros((B,zero_pad,C),device=x.device)],dim = 1)
            add_length += zero_pad
        # H, W = int(N**0.5),int(N**0.5)
        # cls_token, feat_token = x[:, 0], x[:, 1:]
        # feat_token = x
        cnn_feat = x.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        # print(add_length)
        if add_length >0:
            x = x[:,:-add_length]
        # x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x
        
#位置编码生成器（Position Encoding Generator，PEG），用于给输入张量添加位置信息。代码通过卷积操作将位置信息注入输入张量
class PEG(nn.Module):
    def __init__(self, dim=512,k=7,bias=True,conv_1d=False):
        super(PEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, k, 1, k//2, groups=dim,bias=bias) if not conv_1d else nn.Conv2d(dim, dim, (k,1), 1, (k//2,0), groups=dim,bias=bias)
    def forward(self, x):
        B, N, C = x.shape
        # padding
        H, W = int(np.ceil(np.sqrt(N))), int(np.ceil(np.sqrt(N)))
        add_length = H * W - N
        x = torch.cat([x, x[:,:add_length,:]],dim = 1)
        feat_token = x
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat
        x = x.flatten(2).transpose(1, 2)
        if add_length >0:
            x = x[:,:-add_length]
        # x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class SINCOS(nn.Module):
    def __init__(self,embed_dim=512):
        super(SINCOS, self).__init__()
        self.embed_dim = embed_dim
        #self.pos_embed = self.get_2d_sincos_pos_embed(embed_dim, 8)
    #生成一维正弦余弦位置编码。
    def get_1d_sincos_pos_embed_from_grid(self,embed_dim, pos):
        """
        embed_dim: output dimension for each position
        pos: a list of positions to be encoded: size (M,)
        out: (M, D)
        """
        assert embed_dim % 2 == 0
        omega = np.arange(embed_dim // 2, dtype=np.float)
        omega /= embed_dim / 2.
        omega = 1. / 10000**omega  # (D/2,)
        pos = pos.reshape(-1)  # (M,)
        out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product
        emb_sin = np.sin(out) # (M, D/2)
        emb_cos = np.cos(out) # (M, D/2)
        emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
        return emb
    #生成二维正弦余弦位置编码。
    def get_2d_sincos_pos_embed_from_grid(self,embed_dim, grid):
        assert embed_dim % 2 == 0
        # use half of dimensions to encode grid_h
        emb_h = self.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
        emb_w = self.get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
        emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
        return emb
    #生成位置编码矩阵
    def get_2d_sincos_pos_embed(self,embed_dim, grid_size, cls_token=False):
        """
        grid_size: int of the grid height and width
        return:
        pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
        """
        grid_h = np.arange(grid_size, dtype=np.float32)
        grid_w = np.arange(grid_size, dtype=np.float32)
        grid = np.meshgrid(grid_w, grid_h)  # here w goes first
        grid = np.stack(grid, axis=0)
        pos_embed = self.get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
        return pos_embed
    def forward(self, x):
        B, N, C = x.shape
        #B,H,W,C = x.shape
        # # padding
        H, W = int(np.ceil(np.sqrt(N))), int(np.ceil(np.sqrt(N)))
        add_length = H * W - N
        x = torch.cat([x, x[:,:add_length,:]],dim = 1)
        # pos_embed = torch.zeros(1, H * W + 1, self.embed_dim)
        pos_embed = self.get_2d_sincos_pos_embed(self.embed_dim, int(H), cls_token=False)
        pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0).to(x.device)
        #pos_embed = torch.from_numpy(self.pos_embed).float().to(x.device)
        #print(pos_embed.size())
        #print(x.size())
        x = x + pos_embed
        #x = x + pos_embed[:, 1:, :]
        if add_length >0:
             x = x[:,:-add_length]
        #print('x.shape:',x.shape)
        return x
#可训练的绝对位置编码
class APE(nn.Module):
    def __init__(self,embed_dim=512):
        super(APE, self).__init__()
        self.embed_dim=embed_dim
        #使用 trunc_normal_ 函数对其进行截断正态分布初始化。
    def forward(self, x):
        B,N,C = x.shape
        self.absolute_pos_embed = nn.Parameter(torch.zeros(1,N, self.embed_dim))
        trunc_normal_(self.absolute_pos_embed, std=.02)
        return x + self.absolute_pos_embed.to(x.device)

class Coord(nn.Module):
    def __init__(self,embed_dim=512):
        super(Coord, self).__init__()
        self.embed_dim=embed_dim
        self.absolute_pos_embed =nn.Linear(2,self.embed_dim)
    def normalize_coordinates(self,coord):
        # coord shape: (1, N, 2)
        # Split the coordinates into row and column vectors
        coord = coord.squeeze(0)  # Remove the first dimension
        P_r = coord[:, 0]
        P_c = coord[:, 1]
        # Calculate the min and max values for rows and columns
        r_min, r_max = P_r.min().item(), P_r.max().item()
        c_min, c_max = P_c.min().item(), P_c.max().item()
        # Calculate the scale λ of coordinate transformation
        h = r_max - r_min
        w = c_max - c_min
        λ = max(h, w)
        # Normalize the coordinates
        P_r_normalized = (P_r - r_min) / λ
        P_c_normalized = (P_c - c_min) / λ
        # Combine the normalized row and column vectors back into coordinates
        coord_normalized = torch.stack((P_r_normalized, P_c_normalized), dim=1)
        # Add the first dimension back to the coordinates
        coord_normalized = coord_normalized.unsqueeze(0)
        return coord_normalized


    def forward(self, x,coord):
        B,N,C = coord.shape
        coord = coord.float() 
        coord=self.normalize_coordinates(coord)
        coord=self.absolute_pos_embed(coord)
        
        return x + coord.to(x.device)
class RPE(nn.Module):
    def __init__(self,num_heads,region_num):
        super(RPE, self).__init__()
        self.num_heads=num_heads
        self.region_num=region_num
    def padding(self,x):
        B, L, C = x.shape
        H, W = int(np.ceil(np.sqrt(L))), int(np.ceil(np.sqrt(L)))
        _n = -H % self.region_num
        H, W = H+_n, W+_n
        region_size = int(H // self.region_num)
        region_num = self.region_num
        add_length = H * W - L
        if add_length > 0:
            x = torch.cat([x, torch.zeros((B,add_length,C),device=x.device)],dim = 1)
        return x,H,W,add_length,region_num,region_size
    def forward(self, x):
        #num_regions*B, N, C
        nB, L, C = x.shape
        x = x.reshape(1, -1, C)
        x,H,W,add_length,region_num,region_size = self.padding(x)
        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * region_size - 1) * (2 * region_size - 1), self.num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        # get pair-wise relative position index for each token inside the region
        coords_h = torch.arange(region_size)
        coords_w = torch.arange(region_size)
        #meshgrid x轴 y轴 网格
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += region_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += region_size - 1
        relative_coords[:, :, 0] *= 2 * region_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        relative_position_bias = self.relative_position_bias_table[relative_position_index.view(-1)].view(region_size * region_size, region_size * region_size, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        #print(relative_position_bias.shape)
        return relative_position_bias.to(x.device)
@torch.no_grad()
def piecewise_index(relative_position, alpha=1.9, beta=1.9*4, gamma=1.9*6, shift=7, dtype=torch.int32):
    rp_abs = relative_position.abs()
    mask = rp_abs <= alpha*2
    not_mask = ~mask
    rp_out = relative_position[not_mask]
    rp_abs_out = rp_abs[not_mask]
    y_out = (torch.sign(rp_out) * (
                                   torch.log(rp_abs_out / alpha) /
                                   math.log(gamma / alpha) *
                                   (beta - 2*alpha)).round().clip(max=shift)).to(dtype)

    idx = relative_position.clone()
    if idx.dtype in [torch.float32, torch.float64]:
        # round(x) when |x| <= alpha
        idx = idx.round().to(dtype)

    # assign the value when |x| > alpha
    idx[not_mask] = y_out
    idx[mask] = torch.sign(idx[mask])*1
    return idx
class Manhattan_distance(nn.Module):
    def __init__(self,num_heads,region_num):
        super(Manhattan_distance, self).__init__()
        self.num_heads=num_heads
        self.region_num=region_num
    def padding(self,x):
        B, L, C = x.shape
        H, W = int(np.ceil(np.sqrt(L))), int(np.ceil(np.sqrt(L)))
        _n = -H % self.region_num
        H, W = H+_n, W+_n
        region_size = int(H // self.region_num)
        region_num = self.region_num
        add_length = H * W - L
        if add_length > 0:
            x = torch.cat([x, torch.zeros((B,add_length,C),device=x.device)],dim = 1)
        return x,H,W,add_length,region_num,region_size
    def forward(self, x,coords):
        #num_regions*B, N, C
        nB, L, C = x.shape
        x = x.reshape(1, -1, C)
        coords,H,W,add_length,region_num,region_size = self.padding(coords)
        #print('region_num:',region_num)
        #print('region_size:',region_size)
        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * region_size - 1) * (2 * region_size - 1), self.num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        trunc_normal_(self.relative_position_bias_table, std=.02)
        #print('coords0.shape:',coords.shape)
        coords = rearrange(coords, 'b (w ws) c -> b w ws c', ws=region_size)
        #print('coords1.shape:',coords.shape)
        coords = rearrange(coords, 'b w ws c -> (b w) ws c') #[b*num_window, window, C]
        #print('coords2.shape:',coords.shape)

        max_L = coords.shape[1]
        relative_coords = coords.view((-1, max_L, 1, 2))-coords.view((-1, 1, max_L, 2))
        relative_coords = relative_coords.int()
        relative_coords[:, :, :, 0] = piecewise_index(relative_coords[:, :, :, 0], shift=region_size)
        relative_coords[:, :, :, 1] = piecewise_index(relative_coords[:, :, :, 1], shift=region_size)
        relative_coords = relative_coords.abs()
        relative_position_index = relative_coords.sum(-1)  # num_window, Wh*Ww, Wh*Ww
        relative_position_bias = self.relative_position_bias_table[relative_position_index.view(-1)].view(-1, region_size, region_size, self.num_heads)
        relative_position_bias = relative_position_bias.permute(0, 3, 1, 2).contiguous()
        
        #print('relative_position_bias.shape:',relative_position_bias.shape)
        return relative_position_bias.to(x.device)
    



import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def sinusoidal_position_embedding(batch_size, nums_head, max_len, output_dim, device):
    # (max_len, 1)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(-1)
    # (output_dim//2)
    ids = torch.arange(0, output_dim // 2, dtype=torch.float)  # 即公式里的i, i的范围是 [0,d/2]
    theta = torch.pow(10000, -2 * ids / output_dim)
    # (max_len, output_dim//2)
    embeddings = position * theta  # 即公式里的：pos / (10000^(2i/d))
    # (max_len, output_dim//2, 2)
    embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
    # (bs, head, max_len, output_dim//2, 2)
    embeddings = embeddings.repeat((batch_size, nums_head, *([1] * len(embeddings.shape))))  # 在bs维度重复，其他维度都是1不重复
    # (bs, head, max_len, output_dim)
    # reshape后就是：偶数sin, 奇数cos了
    embeddings = torch.reshape(embeddings, (batch_size, nums_head, max_len, output_dim))
    embeddings = embeddings.to(device)
    return embeddings
def RoPE(q, k):
    # q,k: (bs, head, max_len, output_dim)
    batch_size = q.shape[0]
    nums_head = q.shape[1]
    max_len = q.shape[2]
    output_dim = q.shape[-1]
    # (bs, head, max_len, output_dim)
    pos_emb = sinusoidal_position_embedding(batch_size, nums_head, max_len, output_dim, q.device)
    # cos_pos,sin_pos: (bs, head, max_len, output_dim)
    # 看rope公式可知，相邻cos，sin之间是相同的，所以复制一遍。如(1,2,3)变成(1,1,2,2,3,3)
    cos_pos = pos_emb[...,  1::2].repeat_interleave(2, dim=-1)  # 将奇数列信息抽取出来也就是cos 拿出来并复制
    sin_pos = pos_emb[..., ::2].repeat_interleave(2, dim=-1)  # 将偶数列信息抽取出来也就是sin 拿出来并复制
    # q,k: (bs, head, max_len, output_dim)
    q2 = torch.stack([-q[..., 1::2], q[..., ::2]], dim=-1)
    q2 = q2.reshape(q.shape)  # reshape后就是正负交替了
    # 更新qw, *对应位置相乘
    q = q * cos_pos + q2 * sin_pos
    k2 = torch.stack([-k[..., 1::2], k[..., ::2]], dim=-1)
    k2 = k2.reshape(k.shape)
    # 更新kw, *对应位置相乘
    k = k * cos_pos + k2 * sin_pos
    return q, k
def attention(q, k, v, mask=None, dropout=None, use_RoPE=True):
    # q.shape: (bs, head, seq_len, dk)
    # k.shape: (bs, head, seq_len, dk)
    # v.shape: (bs, head, seq_len, dk)
    if use_RoPE:
        q, k = RoPE(q, k)
    d_k = k.size()[-1]
    att_logits = torch.matmul(q, k.transpose(-2, -1))  # (bs, head, seq_len, seq_len)
    att_logits /= math.sqrt(d_k)
    if mask is not None:
        att_logits = att_logits.masked_fill(mask == 0, -1e9)  # mask掉为0的部分，设为无穷大
    #att_scores = F.softmax(att_logits, dim=-1)  # (bs, head, seq_len, seq_len)
    if dropout is not None:
        att_scores = dropout(att_scores)
    # (bs, head, seq_len, seq_len) * (bs, head, seq_len, dk) = (bs, head, seq_len, dk)
    return  att_logits
if __name__ == '__main__':
    # (bs, head, seq_len, dk)
    q = torch.randn((8, 12, 10, 32))
    k = torch.randn((8, 12, 10, 32))
    v = torch.randn((8, 12, 10, 32))
    res, att_scores = attention(q, k, v, mask=None, dropout=None, use_RoPE=True)
    # (bs, head, seq_len, dk),  (bs, head, seq_len, seq_len)
    print(res.shape, att_scores.shape)


