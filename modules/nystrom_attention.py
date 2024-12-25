from math import ceil
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, reduce

# helper functions

def exists(val):
    return val is not None

def moore_penrose_iter_pinv(x, iters = 6):
    device = x.device

    abs_x = torch.abs(x)
    col = abs_x.sum(dim = -1)
    row = abs_x.sum(dim = -2)
    z = rearrange(x, '... i j -> ... j i') / (torch.max(col) * torch.max(row))

    I = torch.eye(x.shape[-1], device = device)
    I = rearrange(I, 'i j -> () i j')

    for _ in range(iters):
        xz = x @ z
        z = 0.25 * z @ (13 * I - (xz @ (15 * I - (xz @ (7 * I - xz)))))

    return z

# main attention class
#NystromAttention 是一种改进的自注意力机制，
#它基于Nystrom方法，该方法通过对较少的标记（landmarks）将复杂度降低到 n×m，其中 m 是标记的数量，通常远小于 n。
class NystromAttention(nn.Module):
    def __init__(self,dim,dim_head = 64,heads = 8,num_landmarks = 256,pinv_iterations = 6,residual = True,residual_conv_kernel = 33,eps = 1e-8,dropout = 0.):
        super().__init__()
        #设置 eps 值以避免除零错误。
        self.eps = eps
        #计算 inner_dim 为每个头的维度与头数量的乘积。
        inner_dim = heads * dim_head
        #初始化标记数量和伪逆迭代次数。
        self.num_landmarks = num_landmarks
        self.pinv_iterations = pinv_iterations
        #设置头的数量和缩放因子
        self.heads = heads
        self.scale = dim_head ** -0.5
        #创建一个线性层，将输入映射到查询（Q）、键（K）和值（V）。
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        #定义输出层和残差卷积层
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim),nn.Dropout(dropout))
        self.residual = residual
        if residual:
            kernel_size = residual_conv_kernel
            padding = residual_conv_kernel // 2
            self.res_conv = nn.Conv2d(heads, heads, (kernel_size, 1), padding = (padding, 0), groups = heads, bias = False)

    def forward(self, x,coord=None, mask = None, return_attn = False):
        #获取输入张量的形状，并解包为变量
        b, n, _, h, m, iters, eps = *x.shape, self.heads, self.num_landmarks, self.pinv_iterations, self.eps
        # 计算输入序列长度和标记数量的余数，如果存在余数，则进行填充。 
        remainder = n % m
        if remainder > 0:
            padding = m - (n % m)
            x = F.pad(x, (0, 0, padding, 0), value = 0)
            #如果存在掩码，也对掩码进行填充。
            if exists(mask):
                mask = F.pad(mask, (padding, 0), value = False)
        # 使用线性层将输入映射到查询、键和值，并在最后一个维度上分割成 Q、K、V 三个张量。
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        #重排列 Q、K、V，使其形状符合多头注意力机制的要求
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        # 如果存在掩码，将掩码应用到 Q、K、V，使得掩码位置对应的值为零。
        if exists(mask):
            mask = rearrange(mask, 'b n -> b () n')
            q, k, v = map(lambda t: t * mask[..., None], (q, k, v))
        #对查询进行缩放。
        q = q * self.scale
        # 计算每个标记对应的长度。
        l = ceil(n / m)
        landmark_einops_eq = '... (n l) d -> ... n d'
        #使用 einops 的 reduce 操作，将 Q、K 分别求和，生成 Q 标记和 K 标记。
        q_landmarks = reduce(q, landmark_einops_eq, 'sum', l = l)
        k_landmarks = reduce(k, landmark_einops_eq, 'sum', l = l)
        # 如果存在掩码，计算标记掩码的和，并根据掩码的和调整除数。
        divisor = l
        if exists(mask):
            mask_landmarks_sum = reduce(mask, '... (n l) -> ... n', 'sum', l = l)
            divisor = mask_landmarks_sum[..., None] + eps
            mask_landmarks = mask_landmarks_sum > 0
        # 对 Q 标记和 K 标记进行均值计算。
        q_landmarks /= divisor
        k_landmarks /= divisor
        # 使用 einsum 计算 Q 与 K 标记、Q 标记与 K 标记、Q 标记与 K 的相似性矩阵。
        einops_eq = '... i d, ... j d -> ... i j'
        attn1 = einsum(einops_eq, q, k_landmarks)
        attn2 = einsum(einops_eq, q_landmarks, k_landmarks)
        attn3 = einsum(einops_eq, q_landmarks, k)
        # 如果存在掩码，将掩码应用到相似性矩阵，并填充掩码位置。
        if exists(mask):
            mask_value = -torch.finfo(q.dtype).max
            sim1.masked_fill_(~(mask[..., None] * mask_landmarks[..., None, :]), mask_value)
            sim2.masked_fill_(~(mask_landmarks[..., None] * mask_landmarks[..., None, :]), mask_value)
            sim3.masked_fill_(~(mask_landmarks[..., None] * mask[..., None, :]), mask_value)
        # 对相似性矩阵进行软最大值操作，并使用 Moore-Penrose 伪逆进行迭代计算，最终聚合值。
        attn1, attn2, attn3 = map(lambda t: t.softmax(dim = -1), (attn1, attn2, attn3))
        attn2 = moore_penrose_iter_pinv(attn2, iters)
        out = (attn1 @ attn2) @ (attn3 @ v)
        # add depth-wise conv residual of values
        if self.residual:
            out += self.res_conv(v)
        # merge and combine heads
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        out = self.to_out(out)
        out = out[:, -n:]
        if return_attn:
            attn1 = attn1[:,:,0].unsqueeze(-2) @ attn2
            attn1 = (attn1 @ attn3)
            return out, attn1[:,:,0,-n+1:]
        return out

# transformer

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

class Nystromformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        num_landmarks = 256,
        pinv_iterations = 6,
        attn_values_residual = True,
        attn_values_residual_conv_kernel = 33,
        attn_dropout = 0.,
        ff_dropout = 0.   
    ):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, NystromAttention(dim = dim, dim_head = dim_head, heads = heads, num_landmarks = num_landmarks, pinv_iterations = pinv_iterations, residual = attn_values_residual, residual_conv_kernel = attn_values_residual_conv_kernel, dropout = attn_dropout)),
                PreNorm(dim, FeedForward(dim = dim, dropout = ff_dropout))
            ]))

    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask) + x
            x = ff(x) + x
        return x