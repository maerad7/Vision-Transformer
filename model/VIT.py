import math
import torch
from torch import nn
from einops import rearrange, repeat

def pair(x):
    """
    스칼라 값이나 튜플을 해당 값의 쌍(pair)으로 변환합니다.
    """
    return x if isinstance(x, tuple) else (x, x)

class DropPath(nn.Module):
    """
        전체 레이어(경로)를 확률적으로 끊어버리는 regularization 기법
        ResNet이나 Transformer에서 Deep network의 과적합을 줄이고, 학습 안정성을 높이기 위해 사용

        1. 깊은 네트워크를 안정적으로 학습
            Stochastic depth 덕분에 gradient vanishing이 줄어듦

        2. Regularization 효과  
            각 sample의 경로가 다르기 때문에 ensemble 효과가 생김

    """
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        # 경로를 살릴 확률 => Keep // 경로를 끊을 확률 => drop_prob
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) +(1,) *(x.ndim - 1) # (batch_size, 1, 1, 1)
        mask  = x.new_empty(shape).bernoulli_(keep).div_(keep)
        return x*mask

# core
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., attn_dropout=0., qkv_bias=True):
       super().__init__()
       # 헤드 개수(heads)와 각 헤드의 차원(dim_head)을 곱하여 multi-head attention 내부 차원(inner dimension)을 정의
       inner = heads * dim_head
       self.heads = heads
       # self.scale은 점곱 연산의 결과를 정규화하여, 수치적으로 안정적인 학습을 돕기 위한 값입니다.
       self.scale = dim_head ** -0.5

       self.norm = nn.LayerNorm(dim)
       self.qkv = nn.Linear(dim, inner * 3, bias=qkv_bias)
       self.attn_drop = nn.Dropout(attn_dropout)
       self.proj = nn.Linear(inner, dim)
       self.drop = nn.Dropout(dropout)

    def forward(self,x):
        x = self.norm(x)  # x: (batch_size, num_tokens, dim)
        qkv = self.qkv(x).chunk(3, dim=-1)  # (batch_size, num_tokens, inner*3) -> 3개 (batch_size, num_tokens, inner)
        # 각각의 쿼리, 키, 밸류를 다중 head 구조로 변환 (b: batch, h: head, n: token,  d: dim per head)
        q, k, v = [rearrange(t, "b n (h d) -> b h n d", h=self.heads) for t in qkv]

        # scaled dot-product attention
        # q: (batch_size, heads, num_tokens, dim_head)
        # k.transpose(-1, -2): (batch_size, heads, dim_head, num_tokens)
        # dots: (batch_size, heads, num_tokens, num_tokens)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = torch.softmax(dots, dim=-1)  # attn: (batch_size, heads, num_tokens, num_tokens)
        attn = self.attn_drop(attn)

        # attn: (batch_size, heads, num_tokens, num_tokens), v: (batch_size, heads, num_tokens, dim_head)
        # out: (batch_size, heads, num_tokens, dim_head)
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")  # out: (batch_size, num_tokens, heads*dim_head)
        return self.proj(out)  # (batch_size, num_tokens, dim)

    
class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, drop=0., attn_drop=0., droppath=0., qkv_bias=True):
        super().__init__()
        self.attn = Attention(dim, heads, dim_head, dropout=drop, attn_dropout=attn_drop, qkv_bias=qkv_bias)
        self.drop_path1 = DropPath(droppath)
        self.ff = FeedForward(dim, mlp_dim, dropout=drop)
        self.drop_path2 = DropPath(droppath)

        def forward(self, x):
            x = x + self.drop_path1(self.attn(x))
            x = x + self.drop_path2(self.ff(x))
            return x


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, drop=0., attn_drop=0., droppath=0., qkv_bias=True):
        super().__init__()
        # dpr은 DropPath 비율을 depth만큼 선형적으로 증가시키는 리스트입니다.
        dpr = torch.linspace(0, droppath, steps=depth).tolist()  # 점점 커지는 DropPath 확률
        self.layers = nn.ModuleList([
            TransformerBlock(dim, heads, dim_head, mlp_dim, drop, attn_drop, dpr[i], qkv_bias)
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        for blk in self.layers:
            x = blk(x)
        return self.norm(x)


class ViT(nn.Module):
    def __init__(
        self, *,
        image_size, patch_size, num_classes, dim, depth, heads,
        mlp_dim, pool='cls', channels=3, dim_head=64,
        drop=0., attn_drop=0., droppath=0., emb_drop=0.,
        qkv_bias=True, use_cls_token=True
    ):
        super().__init__()
        ih, iw = pair(image_size)
        ph, pw = pair(patch_size)
        assert ih % ph == 0 and iw % pw == 0, "이미지 크기는 패치 크기로 나누어 떨어져야 합니다."
        self.grid_h, self.grid_w = ih // ph, iw // pw
        num_patches = self.grid_h * self.grid_w
        self.use_cls = use_cls_token
        assert pool in {"cls", "mean"}, "pool은 'cls' 또는 'mean'이어야 합니다."

        # Conv2d patch embedding (proj to dim)
        self.patch_embed = nn.Sequential(
            nn.Conv2d(channels, dim, kernel_size=(ph, pw), stride=(ph, pw), bias=True),
            Rearrange("b d h w -> b (h w) d"),
        )

        # Positional embedding (learned) for base grid
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + (1 if use_cls_token else 0), dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim)) if use_cls_token else None
        self.emb_drop = nn.Dropout(emb_drop)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim,
                                       drop=drop, attn_drop=attn_drop, droppath=droppath, qkv_bias=qkv_bias)

        self.pool = pool
        self.head = nn.Linear(dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        # some gentle init for the head
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    @staticmethod
    def _interpolate_pos_embed(pe, old_hw, new_hw, has_cls):
        """
        이 함수는 positional embedding(pe)를 입력 이미지 패치 격자(old_hw)에서
        새로운 패치 격자(new_hw) 크기로 보간(interpolate)해주는 역할을 합니다.

        pe: (1, N_old(+1), C)  # (+1은 cls token이 있으면 추가됨)
        old_hw: (H_old, W_old) 기존 패치 격자의 높이/너비(패치 개수 기준)
        new_hw: (H_new, W_new) 새로운 패치 격자의 높이/너비(패치 개수 기준)
        has_cls: cls token 사용 여부

        반환값: (1, N_new(+1), C)로 positional embedding 크기를 맞춤
        """
        # 기존 패치 격자와 새로운 격자가 같으면 보간 필요 없이 그대로 반환
        if old_hw == new_hw:
            return pe

        # cls token이 있으면 분리
        cls, grid = (pe[:, :1], pe[:, 1:]) if has_cls else (None, pe)
        B, N, C = pe.shape  # B: 1, N: 전체 토큰수(cls 포함 가능), C: 차원

        H_old, W_old = old_hw
        H_new, W_new = new_hw

        # 패치 부분 positional embedding을 (1, C, H_old, W_old)로 reshape해서 보간 준비
        grid = grid.reshape(1, H_old, W_old, C).permute(0, 3, 1, 2)  # (1, C, H, W)
        # new size로 bicubic interpolation (align_corners=False로 부드럽게)
        grid = nn.functional.interpolate(grid, size=(H_new, W_new), mode="bicubic", align_corners=False)
        # 다시 (1, H_new, W_new, C) -> (1, N_new, C)로 펼침
        grid = grid.permute(0, 2, 3, 1).reshape(1, H_new * W_new, C)

        if has_cls:
            # cls token이 있으면 앞에 붙임
            pe = torch.cat([cls, grid], dim=1)
        else:
            pe = grid
        return pe

    def forward(self, img):
        # img: (B, C, H, W) 형태의 입력 이미지
        B, C, H, W = img.shape
        x = self.patch_embed(img)  # (B, N, D), 패치 임베딩 적용 결과

        # 토큰 개수 구하기
        tokens = x.shape[1]
        # 입력과 동일한 종횡비(H:W)를 가정하여, 토큰을 패치 격자(grid)에 근사적으로 매핑
        # grid의 높이: 대략 sqrt(tokens * H / W), grid의 너비: tokens // grid의 높이
        # 최대한 원본 종횡비와 가까운 형태로 factorization
        # 패치 토큰 개수(tokens)를 입력 이미지의 높이(H)와 너비(W) 비율에 맞춰 격자(grid) 형태(H*, W*)로 변환합니다.
        # 
        # 1. 우선 원본 이미지의 종횡비(H/W)를 고려해서 grid의 높이(g_h)를 근사적으로 계산합니다.
        #    g_h = round(sqrt(tokens * H / W))
        #    tokens는 전체 패치 개수이고, H/W로 원본 이미지 비율을 반영합니다.
        # 
        # 2. g_h는 1보다 크거나 같고, tokens보다 작거나 같은 값으로 제약합니다.
        #    그리고 g_h가 tokens의 약수가 되도록 최대한 맞춥니다. (tokens를 g_h로 나눠떨어지게)
        #    만약 나눠떨어지지 않으면 g_h 값을 1씩 줄이면서 약수가 될 때까지 찾습니다.
        #
        # 3. grid의 너비(g_w)는 tokens // g_h로 계산합니다.
        #
        # 위 과정은 positional embedding 크기를 유동적으로 보간(interpolate)할 때,
        # 현재 입력 이미지(의 패치 수)와 가장 근접한 격자 행렬차원으로 변환하기 위함입니다.
        """
            토큰 개수(N)를 원래 이미지 비율과 가장 비슷한 2D 격자 형태로 재배치하기 위해 적절한 높이(g_h)와 너비(g_w)를 찾는 과정이다.
        """
        g_h = round(math.sqrt(tokens * H / max(W, 1)))
        g_h = max(1, min(tokens, g_h))
        while tokens % g_h != 0 and g_h > 1:
            g_h -= 1
        g_w = tokens // g_h

        pe = self._interpolate_pos_embed(
            self.pos_embed, (self.grid_h, self.grid_w), (g_h, g_w), self.use_cls
        )  # pe: (1, N, D)  (N: 패치+cls토큰 개수, D: 임베딩 차원)
        
        if self.use_cls:
            cls_tok = repeat(self.cls_token, "1 1 d -> b 1 d", b=B)  # cls_tok: (B, 1, D)
            x = torch.cat([cls_tok, x], dim=1)  # x: (B, N, D)
        x = x + pe  # x: (B, N, D)
        x = self.emb_drop(x)  # x: (B, N, D)

        x = self.transformer(x)  # x: (B, N, D)
        x = x[:, 0] if (self.use_cls and self.pool == "cls") else x.mean(dim=1)  # (B, D)
        return self.head(x)  # (B, num_classes)


