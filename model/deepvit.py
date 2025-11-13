import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# --------------------------------------------------------
# FeedForward (MLP) 블록
#   - DeepViT 논문에서도 transformer block 구조는 그대로 사용
#   - LayerNorm → Linear → GELU → Dropout → Linear → Dropout
#   - SA 뒤에 오는 표준적인 FFN 부분
# --------------------------------------------------------
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),        # 입력 토큰을 정규화 (안정된 학습을 위해)
            nn.Linear(dim, hidden_dim),
            nn.GELU(),               # 비선형 활성 함수
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# --------------------------------------------------------
# Attention + Re-attention
#   - 원래 ViT의 Multi-Head Self-Attention 구조에
#   - DeepViT 논문에서 제안한 "Re-attention" 을 추가한 모듈
#
#   핵심 아이디어:
#     1) 일반적인 MHSA로 head별 attention map A (b, h, i, j)를 계산
#     2) 논문에서 제안한 Θ ∈ R^{H×H} 를 이용해 head 간 attention을 재조합
#        → 서로 다른 head의 정보를 섞어 새로운 attention map을 생성
#     3) 이 과정을 통해 layer 간 attention collapse(비슷한 attention 반복)를 방지
# --------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5    # QK^T / sqrt(d) 의 scaling factor

        # 입력 토큰 정규화 (pre-norm 구조)
        self.norm = nn.LayerNorm(dim)

        # Q, K, V를 한 번에 projection 하기 위한 Linear
        # (dim) → (3 * heads * dim_head)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.dropout = nn.Dropout(dropout)

        # ------------------------------------------------
        # Re-attention 에서 사용하는 Θ (논문에서의 transformation matrix)
        #   - 크기: (heads, heads)
        #   - 각 새 head가 기존 head들의 linear combination으로 구성되도록 학습
        #   - A_reattn = Θ^T · A (head 축에 대해)
        # ------------------------------------------------
        self.reattn_weights = nn.Parameter(torch.randn(heads, heads))

        # ------------------------------------------------
        # 논문에서의 Norm(A') 역할:
        #   - head 간 mixing 이후의 attention map 분산을 안정화
        #   - 여기서는 head 차원에 대해 LayerNorm 적용
        #     A: (b, h, i, j)
        #     → (b, i, j, h) 로 바꾼 뒤 head 축에 LayerNorm
        #     → 다시 (b, h, i, j) 로 복원
        # ------------------------------------------------
        self.reattn_norm = nn.Sequential(
            Rearrange('b h i j -> b i j h'),
            nn.LayerNorm(heads),
            Rearrange('b i j h -> b h i j')
        )

        # 여러 head를 다시 concat 해서 원래 dim으로 되돌리는 projection
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: (b, n, dim), n = 토큰 수 (patch + cls)
        b, n, _, h = *x.shape, self.heads

        # pre-norm
        x = self.norm(x)

        # Q, K, V 계산 후 (b, n, heads*dim_head) → (b, heads, n, dim_head)
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        # ------------------------------------------------
        # 1) 기본 Self-Attention (MHSA)
        #    dots: (b, h, i, j) = Q K^T / sqrt(d)
        # ------------------------------------------------
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)   # (b, h, i, j)

        # ------------------------------------------------
        # 2) Re-attention (논문 Eq. (3) 구현)
        #
        #   A ∈ R^{B×H×T×T}, Θ ∈ R^{H×H}
        #   A' = Θ^T A (head 축에 대한 선형 조합)
        #
        #   einsum('b h i j, h g -> b g i j'):
        #     - 기존 head h를 따라 Θ(h→g)를 곱해 새로운 head g 생성
        # ------------------------------------------------
        attn = einsum('b h i j, h g -> b g i j', attn, self.reattn_weights)

        # head 방향으로 LayerNorm (논문에서의 Norm 연산에 해당)
        attn = self.reattn_norm(attn)

        # ------------------------------------------------
        # 3) 새로운 attention map으로 V를 가중합하여 출력 계산
        #    out: (b, h, i, d)
        # ------------------------------------------------
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # head 차원을 다시 병합 → (b, n, heads*dim_head)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        return out

# --------------------------------------------------------
# Transformer: (Re-Attention + FFN) × depth
#   - 각 레이어: [Re-Attention, FeedForward]
#   - residual 연결: x = x + attn(x), x = x + ff(x)
# --------------------------------------------------------
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        # x: (b, n, dim)
        for attn, ff in self.layers:
            x = attn(x) + x   # SA (Re-attention) + residual
            x = ff(x) + x     # FFN + residual
        return x

# --------------------------------------------------------
# DeepViT 본체
#   - ViT 구조를 따르되 Attention 부분이 Re-Attention으로 대체된 형태
#   - 논문에서 말하는 DeepViT: "Self-Attention → Re-Attention 치환" 버전
# --------------------------------------------------------
class DeepViT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        pool = 'cls',
        channels = 3,
        dim_head = 64,
        dropout = 0.,
        emb_dropout = 0.
    ):
        super().__init__()

        # 이미지 크기가 패치 크기로 나누어 떨어지는지 검사
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # ------------------------------------------------
        # 1) Patch Embedding
        #   - 입력: (b, c, H, W)
        #   - 출력: (b, n_patches, dim)
        #   - 패치를 펼친 뒤 (p1 * p2 * c) → dim 으로 Linear
        #   - LayerNorm을 앞/뒤에 넣어 안정된 임베딩 생성
        # ------------------------------------------------
        self.to_patch_embedding = nn.Sequential(
            # (b, c, h*p1, w*p2) → (b, h*w, p1*p2*c)
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        )

        # ------------------------------------------------
        # 2) 위치 임베딩 + CLS 토큰
        #   - ViT와 동일하게 [CLS] 토큰 + patch 토큰 위치임베딩 사용
        # ------------------------------------------------
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token     = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout       = nn.Dropout(emb_dropout)

        # ------------------------------------------------
        # 3) Re-Attention 기반 Transformer Encoder
        #   - 위에서 정의한 Transformer 사용 (Attention = Re-attention)
        # ------------------------------------------------
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        # pooling 방식: 'cls' 또는 'mean'
        self.pool = pool

        # latent 변환용 placeholder (필요 시 추가 projection 가능)
        self.to_latent = nn.Identity()

        # ------------------------------------------------
        # 4) 최종 분류 헤드
        #   - LayerNorm + Linear(num_classes)
        # ------------------------------------------------
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        # img: (b, c, H, W)
        x = self.to_patch_embedding(img)   # (b, n, dim)
        b, n, _ = x.shape

        # CLS 토큰 복제: (1, 1, dim) → (b, 1, dim)
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)

        # 앞쪽에 CLS 토큰을 붙여서 (b, n+1, dim)
        x = torch.cat((cls_tokens, x), dim=1)

        # 위치 임베딩 추가
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        # Re-Attention 기반 Transformer 인코딩
        x = self.transformer(x)   # (b, n+1, dim)

        # pooling: CLS 토큰 또는 토큰 평균
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
