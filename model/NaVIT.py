from __future__ import annotations

from functools import partial
from typing import List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.utils.rnn import pad_sequence as orig_pad_sequence

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# =========================
# helpers
# =========================

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def always(val):
    return lambda *args: val

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def divisible_by(numer, denom):
    return (numer % denom) == 0

# =========================
# Auto grouping images (Patch n' Pack의 "예제 패킹"을 위한 준비)
# - 서로 다른 해상도의 이미지를 "토큰 수" 관점에서 묶어
#   (시퀀스 길이 상한 max_seq_len 이하가 되도록) 하나의 시퀀스에 패킹
# - 논문에서 말하는 "fixed batch shape"를 유지하면서도
#   각 배치 안에 다양한 해상도의 이미지를 섞어 넣는 핵심 로직
# - token dropping을 고려해 실제 유지될 토큰 수로 길이를 계산
# =========================

def group_images_by_max_seq_len(
    images: List[Tensor],
    patch_size: int,
    calc_token_dropout = None,
    max_seq_len = 2048

) -> List[List[Tensor]]:

    calc_token_dropout = default(calc_token_dropout, always(0.))

    groups = []
    group = []
    seq_len = 0

    if isinstance(calc_token_dropout, (float, int)):
        calc_token_dropout = always(calc_token_dropout)

    for image in images:
        assert isinstance(image, Tensor)

        image_dims = image.shape[-2:]
        ph, pw = map(lambda t: t // patch_size, image_dims)

        # 이미지가 생성할 토큰 길이(= 패치 개수)
        image_seq_len = (ph * pw)

        # (연구 포인트) 이미지별 token dropping 비율을 반영하여
        # 실제 남는 토큰 수로 길이를 계산 -> 동일 max_seq_len 내에서 더 많은 이미지를 패킹 가능
        image_seq_len = int(image_seq_len * (1 - calc_token_dropout(*image_dims)))

        assert image_seq_len <= max_seq_len, f'image with dimensions {image_dims} exceeds maximum sequence length'

        # 현재 그룹에 넣으면 길이 초과면 그룹 종료하고 새 그룹 시작
        if (seq_len + image_seq_len) > max_seq_len:
            groups.append(group)
            group = []
            seq_len = 0

        group.append(image)
        seq_len += image_seq_len

    if len(group) > 0:
        groups.append(group)

    return groups

# =========================
# Normalization
# - 논문 부가 개선: bias 제거 LayerNorm, QK 정규화(RMSNorm 유사)
#   (참고: ViT-22B 등의 개선사항)
# =========================

# PyTorch 기본 LN은 bias 파라미터를 포함하므로, bias 없는 변형 구현
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer('beta', torch.zeros(dim))  # 고정 offset(학습X)

    def forward(self, x):
        # shape[-1] 축에 대해 정규화 (bias는 0으로 고정)
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

# Query-Key 정규화: ViT 22B 계열에서 소개된 QK-norm과 유사.
# - 평균 제거 없이 L2 normalize(+ learned gamma), 스케일링 포함
# - 대규모 ViT에서 학습 안정화 및 성능 개선 보고
class RMSNorm(nn.Module):
    def __init__(self, heads, dim):
        super().__init__()
        self.scale = dim ** 0.5
        # head별로 학습되는 gamma (H, 1, D_head)
        self.gamma = nn.Parameter(torch.ones(heads, 1, dim))

    def forward(self, x):
        normed = F.normalize(x, dim = -1)   # L2 정규화
        return normed * self.scale * self.gamma

# =========================
# FeedForward (MLP 블록)
# - 사전 LayerNorm → Linear → GELU → Dropout → Linear → Dropout
# - 논문에서 언급되는 계산비중: 모델 스케일이 커질수록 MLP 비중이 커지므로
#   패킹으로 attention 오버헤드가 상대적으로 작아지는 현상과 연결
# =========================

def FeedForward(dim, hidden_dim, dropout = 0.):
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, dim),
        nn.Dropout(dropout)
    )

# =========================
# Attention
# - Masked self-attention: "예제 간" 상호참조 차단 (Patch n' Pack의 핵심)
# - key padding mask & attn_mask를 반영하여
#   1) 패딩 토큰 무시
#   2) 서로 다른 이미지 토큰끼리는 attention 불가
# =========================

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.norm = LayerNorm(dim)

        # QK 정규화 (대규모 ViT 안정화)
        self.q_norm = RMSNorm(heads, dim_head)
        self.k_norm = RMSNorm(heads, dim_head)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        # bias 제거 선형 (논문 부가 개선)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias = False),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x,
        context = None,   # cross-attn 스타일의 context 허용 (attn pooling에서 사용)
        mask = None,      # (옵션) key padding mask (여기선 attn_mask와 병용 가능)
        attn_mask = None  # (핵심) 이미지 경계 마스크 + padding 마스크를 포함한 bool mask
    ):
        x = self.norm(x)
        kv_input = default(context, x)

        qkv = (self.to_q(x), *self.to_kv(kv_input).chunk(2, dim = -1))

        # (B, N, H*D) -> (B, H, N, D)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        # Q, K에 대해 head-wise RMSNorm 적용
        q = self.q_norm(q)
        k = self.k_norm(k)

        # 점수 계산
        dots = torch.matmul(q, k.transpose(-1, -2))

        # key padding mask (선택적으로 사용)
        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            dots = dots.masked_fill(~mask, -torch.finfo(dots.dtype).max)

        # (핵심) attn_mask: 서로 다른 이미지 사이의 어텐션 차단 + padding 무시
        if exists(attn_mask):
            dots = dots.masked_fill(~attn_mask, -torch.finfo(dots.dtype).max)

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# =========================
# Transformer
# - 표준 Encoder 형태
# - 각 블록에서 self-attn → FFN (residual)
# - 마지막에 LayerNorm
# =========================

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

        self.norm = LayerNorm(dim)

    def forward(
        self,
        x,
        mask = None,
        attn_mask = None
    ):
        for attn, ff in self.layers:
            # 예제 경계/패딩 마스크를 attn에 전달
            x = attn(x, mask = mask, attn_mask = attn_mask) + x
            x = ff(x) + x

        return self.norm(x)

# =========================
# NaViT (Native Resolution ViT)
# - 핵심: (1) 예제 패킹, (2) 마스크드 self-attention/풀링,
#        (3) factorized 2D absolute positional embedding,
#        (4) 이미지별 token dropping,
#        (5) 변동 해상도 입력 처리
# =========================

class NaViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., token_dropout_prob = None):
        super().__init__()
        image_height, image_width = pair(image_size)

        # -------------------------
        # (논문) 이미지별 token dropping 비율을 허용
        # - float/int: 상수 드랍률
        # - callable: (height, width) -> drop_prob를 가변적으로 결정
        #   (해상도 의존/스케줄 등 구현 가능)
        # -------------------------
        self.calc_token_dropout = None

        if callable(token_dropout_prob):
            self.calc_token_dropout = token_dropout_prob

        elif isinstance(token_dropout_prob, (float, int)):
            assert 0. <= token_dropout_prob < 1.
            token_dropout_prob = float(token_dropout_prob)
            self.calc_token_dropout = lambda height, width: token_dropout_prob

        # -------------------------
        # 패치 관련 계산
        # - "네이티브" 해상도 사용을 가정하지만,
        #   여기서는 image_size가 patch_size로 나누어진다고 가정(간단화)
        #   실제 NaViT는 변동 해상도/종횡비를 허용 → 아래 pos-emb 설계가 핵심
        # -------------------------
        assert divisible_by(image_height, patch_size) and divisible_by(image_width, patch_size), 'Image dimensions must be divisible by the patch size.'

        patch_height_dim, patch_width_dim = (image_height // patch_size), (image_width // patch_size)
        patch_dim = channels * (patch_size ** 2)

        self.channels = channels
        self.patch_size = patch_size

        # -------------------------
        # 패치 임베딩: (C * p * p) -> D
        # - 입력 패치를 LayerNorm 후 Linear로 투사, 다시 LN
        # - 논문 부가개선(Pre-LN 및 bias-free 방향)과 일치
        # -------------------------
        self.to_patch_embedding = nn.Sequential(
            LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            LayerNorm(dim),
        )

        # -------------------------
        # (논문 핵심) Factorized 2D absolute positional embedding
        # - pos_embed_height[h] + pos_embed_width[w] 를 더해 2D 위치를 표현
        # - 2D 임베딩을 (ϕx + ϕy)로 분해 → 다양한 해상도/종횡비에 일반화 용이
        # - 여기선 absolute 버전(direct index) 구현
        # -------------------------
        self.pos_embed_height = nn.Parameter(torch.randn(patch_height_dim, dim))
        self.pos_embed_width = nn.Parameter(torch.randn(patch_width_dim, dim))

        self.dropout = nn.Dropout(emb_dropout)

        # 표준 Transformer 인코더
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        # -------------------------
        # (논문) Attention Pooling
        # - 이미지 단위 representation을 얻기 위해
        #   고정된 "쿼리(이미지 개수만큼)"가 시퀀스(토큰)에 어텐션
        # - 패킹된 시퀀스에서 이미지 경계 마스크를 적용한 pooled rep를 얻음
        # -------------------------
        self.attn_pool_queries = nn.Parameter(torch.randn(dim))
        self.attn_pool = Attention(dim = dim, dim_head = dim_head, heads = heads)

        # 최종 분류 헤드
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            LayerNorm(dim),
            nn.Linear(dim, num_classes, bias = False)
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        batched_images: List[Tensor] | List[List[Tensor]], # (중요) 서로 다른 해상도 이미지들의 리스트 또는 그 그룹
        group_images = False,       # True면 자동 패킹 수행
        group_max_seq_len = 2048    # 패킹 시 최댓 시퀀스 길이(토큰 수)
    ):
        p, c, device, has_token_dropout = self.patch_size, self.channels, self.device, exists(self.calc_token_dropout) and self.training

        arange = partial(torch.arange, device = device)
        pad_sequence = partial(orig_pad_sequence, batch_first = True)

        # -------------------------
        # (옵션) 자동 패킹
        # - 서로 다른 해상도 이미지들을 max_seq_len을 넘지 않도록 묶음
        # - 학습 중이면 token dropping 비율을 고려하여 실제 남는 토큰 수 기준으로 패킹
        # -------------------------
        if group_images:
            batched_images = group_images_by_max_seq_len(
                batched_images,
                patch_size = self.patch_size,
                calc_token_dropout = self.calc_token_dropout if self.training else None,
                max_seq_len = group_max_seq_len
            )

        # 입력이 List[Tensor]라면, 한 배치로 묶어서 List[List[Tensor]]로 통일
        if torch.is_tensor(batched_images[0]):
            batched_images = [batched_images]

        # -------------------------
        # 서로 다른 해상도의 이미지를 "토큰 시퀀스"로 변환
        # - sequences: (토큰 시퀀스)
        # - positions: (각 토큰의 (h,w) 위치 인덱스)
        # - image_ids: 각 토큰이 어느 이미지에 속하는지 표시
        # -------------------------
        num_images = []
        batched_sequences = []
        batched_positions = []
        batched_image_ids = []

        for images in batched_images:
            num_images.append(len(images))

            sequences = []
            positions = []
            image_ids = torch.empty((0,), device = device, dtype = torch.long)

            for image_id, image in enumerate(images):
                assert image.ndim ==3 and image.shape[0] == c
                image_dims = image.shape[-2:]
                assert all([divisible_by(dim, p) for dim in image_dims]), f'height and width {image_dims} of images must be divisible by patch size {p}'

                ph, pw = map(lambda dim: dim // p, image_dims)

                # (h, w) 그리드 좌표 → 각 토큰의 2D 위치 인덱스
                pos = torch.stack(torch.meshgrid((
                    arange(ph),
                    arange(pw)
                ), indexing = 'ij'), dim = -1)
                pos = rearrange(pos, 'h w c -> (h w) c')

                # 이미지를 p x p 패치로 펼친 뒤 (H/p * W/p, C*p*p) 시퀀스화
                seq = rearrange(image, 'c (h p1) (w p2) -> (h w) (c p1 p2)', p1 = p, p2 = p)

                seq_len = seq.shape[-2]

                # (논문) 이미지별 token dropping
                # - 해상도별/스케줄별로 드랍률을 다르게 할 수 있음
                # - 일부 이미지는 완전히 남겨두어 train-inference 괴리 감소
                if has_token_dropout:
                    token_dropout = self.calc_token_dropout(*image_dims)
                    num_keep = max(1, int(seq_len * (1 - token_dropout)))
                    keep_indices = torch.randn((seq_len,), device = device).topk(num_keep, dim = -1).indices

                    seq = seq[keep_indices]
                    pos = pos[keep_indices]

                # image_id를 길이만큼 채워넣어, 마스크 생성에 사용
                image_ids = F.pad(image_ids, (0, seq.shape[-2]), value = image_id)
                sequences.append(seq)
                positions.append(pos)

            batched_image_ids.append(image_ids)
            batched_sequences.append(torch.cat(sequences, dim = 0))
            batched_positions.append(torch.cat(positions, dim = 0))

        # -------------------------
        # Key padding mask 생성 (각 배치 내 시퀀스 길이가 다르므로 pad 후 mask 필요)
        # -------------------------
        lengths = torch.tensor([seq.shape[-2] for seq in batched_sequences], device = device, dtype = torch.long)
        seq_arange = arange(lengths.amax().item())
        key_pad_mask = rearrange(seq_arange, 'n -> 1 n') < rearrange(lengths, 'b -> b 1')  # (B, Nmax)

        # -------------------------
        # (핵심) Attention mask 생성
        # 1) batched_image_ids를 pad하여 (B, Nmax)로 맞춤
        # 2) 서로 다른 image_id끼리는 어텐션 금지
        # 3) padding 토큰도 금지
        # -------------------------
        batched_image_ids = pad_sequence(batched_image_ids)  # (B, Nmax)
        attn_mask = rearrange(batched_image_ids, 'b i -> b 1 i 1') == rearrange(batched_image_ids, 'b j -> b 1 1 j')
        attn_mask = attn_mask & rearrange(key_pad_mask, 'b j -> b 1 1 j')  # padding 위치도 제외

        # -------------------------
        # 패치/포지션을 pad로 정렬 (B, Nmax, *)
        # -------------------------
        patches = pad_sequence(batched_sequences)
        patch_positions = pad_sequence(batched_positions)

        # 각 배치에 포함된 이미지 개수 (attention pooling query 개수에 필요)
        num_images = torch.tensor(num_images, device = device, dtype = torch.long)

        # -------------------------
        # (C*p*p) → D 임베딩
        # -------------------------
        x = self.to_patch_embedding(patches)

        # -------------------------
        # (논문 핵심) Factorized 2D absolute positional embedding
        # - x_pos = ϕx[h] + ϕy[w]
        # - 다양한 종횡비/해상도에 대해 일반화 우수
        # -------------------------
        h_indices, w_indices = patch_positions.unbind(dim = -1)
        h_pos = self.pos_embed_height[h_indices]
        w_pos = self.pos_embed_width[w_indices]
        x = x + h_pos + w_pos

        x = self.dropout(x)

        # -------------------------
        # 마스크드 self-attention (예제 경계/패딩 마스크 적용)
        # -------------------------
        x = self.transformer(x, attn_mask = attn_mask)

        # -------------------------
        # (논문) Attention Pooling
        # - 각 배치에서 "최대 이미지 수"만큼 query를 만들어
        #   시퀀스 토큰에 어텐션 → 이미지별 representation 획득
        # - 주의: pooling 시에도 이미지 경계/패딩 마스크 필요
        # -------------------------
        max_queries = num_images.amax().item()

        # (1,) → (B, max_queries, D)
        queries = repeat(self.attn_pool_queries, 'd -> b n d', n = max_queries, b = x.shape[0])

        # pooling용 마스크:
        #   이미지 id(i) == 토큰의 image_id이면 attend 허용
        image_id_arange = arange(max_queries)
        attn_pool_mask = rearrange(image_id_arange, 'i -> i 1') == rearrange(batched_image_ids, 'b j -> b 1 j')
        attn_pool_mask = attn_pool_mask & rearrange(key_pad_mask, 'b j -> b 1 j')   # padding 제외
        attn_pool_mask = rearrange(attn_pool_mask, 'b i j -> b 1 i j')              # (B, 1, max_queries, Nmax)

        # 쿼리가 컨텍스트 x(토큰들)에 attend
        x = self.attn_pool(queries, context = x, attn_mask = attn_pool_mask) + queries

        # (B, max_queries, D) → (B*max_queries, D)
        x = rearrange(x, 'b n d -> (b n) d')

        # 배치마다 실제 이미지 수가 다르므로, 존재하는 이미지 슬롯만 선택
        is_images = image_id_arange < rearrange(num_images, 'b -> b 1')
        is_images = rearrange(is_images, 'b n -> (b n)')
        x = x[is_images]

        # -------------------------
        # 분류 로짓 출력(이미지 단위)
        # - Patch n' Pack 구조에서 "이미지 단위" 표현을 얻은 뒤
        #   분류 헤드에 투입 (여러 이미지가 한 배치에 패킹되어도 OK)
        # -------------------------
        x = self.to_latent(x)
        return self.mlp_head(x)

# 5 images of different resolutions - List[List[Tensor]]

# for now, you'll have to correctly place images in same batch element as to not exceed maximum allowed sequence length for self-attention w/ masking
v = NaViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1,
    token_dropout_prob = 0.1  # token dropout of 10% (keep 90% of tokens)
)
images = [
    [torch.randn(3, 256, 256), torch.randn(3, 128, 128)],
    [torch.randn(3, 128, 256), torch.randn(3, 256, 128)],
    [torch.randn(3, 64, 256)]
]

preds = v(images)
