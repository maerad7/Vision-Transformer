from __future__ import annotations

from functools import partial
from typing import List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.utils.rnn import pad_sequence as orig_pad_sequence

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

def exists(val):
    """값이 None이 아닌지 확인"""
    return val is not None

def default(val, d):
    """값이 None이면 기본값(d) 반환, 아니면 그대로 반환"""
    return val if exists(val) else d

def always(val):
    """항상 동일한 값(val)을 반환하는 함수"""
    return lambda *args, **kwargs: val

def pair(t):
    """스칼라 또는 튜플 t를 (t, t) 형태의 튜플로 변환"""
    return t if isinstance(t, tuple) else (t, t)

def divisible_by(numer, denom):
    """numer가 denom으로 나누어 떨어지는지 여부 반환"""
    return (numer % denom) == 0

# auto grouping images

def group_images_by_max_weq_len( images, patch_size, ,calc_token_dropout = None, max_seq_len = 2048):
    """
        각 이미지가 생성하는 토큰 수를 계산하여, 토큰 수 제한(max_seq_len)을 초과하지 않는 범위로 이미지를 묶어 그룹(batch)으로 만드는 함수.
    """
    calc_token_dropout = default(calc_token_dropout, always(0.))
    
    groups = []
    group = []
    seq_len = 0

    if isinstance(calc_token_dropout, (float, int)):
        calc_token_dropout = always(calc_token_dropout)

    for image in images:
        assert isinstance(image, Tensor)

        # 이미지 텐서에서 높이와 너비(dimensions)를 추출합니다.
        image_dims = image.shape[-2:]
        ph, pw = map(lambda t:t //patch_size, image_dims)

        image_seq_len = (ph * pw)
        # 의미: 토큰을 드롭아웃하여 실제 시퀀스 길이를 결정한다.
        image_seq_len = int(image_seq_len * (1 - calc_token_dropout(*image_dims)))

        assert image_seq_len <= max_seq_len, f'이미지 크기 {image_dims}가(이) 최대 시퀀스 길이를 초과합니다'

        if (seq_len + image_seq_len) > max_seq_len:
            groups.append(group)
            group = []
            seq_len =0 

        group.append(image)
        seq_len += image_seq_len
    
    if len(group) > 0:
        groups.append(group)

    return groups

class LayerNorm(nn.Module):
    """
    입력 텐서 x의 마지막 차원을 기준으로 평균과 분산을 정규화하고, 
    learnable scale(gamma)과 bias(beta) 파라미터로 조정하는 레이어입니다.
    """
    def __init__(self, dim):
        super().__init__()
        # gamma: 정규화된 값에 곱하는 learnable scale 파라미터 (초기값 1)
        self.gamma = nn.Parameter(torch.ones(dim))
        # beta: 정규화된 값에 더하는 learnable bias 파라미터 (초기값 0)
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        """
        x: (..., dim) shape의 입력 텐서
        출력: 같은 shape의 정규화된 텐서
        """
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm)

    - 입력 x를 RMS 기준으로 정규화
    - learnable scale 파라미터(gamma) 적용
    - 전체 norm 크기를 √dim 만큼 곱해줌
    """
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # L2 normalize
        normed = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
        return normed * self.scale * self.gamma


def FeedForward(dim, hidden_dim, dropout = 0.):
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, dim),
        nn.Dropout(dropout)
    )


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.norm = LayerNorm(dim)

        self.q_norm = RMSNorm(heads, dim_head)
        self.k_norm = RMSNorm(heads, dim_head)

        self.attend = nn.Softmax(dim = - 1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim*2, bias =False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias = False),
            nn.Dropout(dropout)
        )

    def forward(self, x, context = None, mask = None, attn_mask = None):
        
        x = self.norm(x)  # x: (b, n, dim)
        kv_input = default(context, x)  # kv_input: (b, n, dim) or (b, m, dim) if context is given

        qkv = (self.to_q(x), *self.to_kv(kv_input).chunk(2, dim = -1))  # to_q(x): (b, n, heads*dim_head), to_kv(kv_input): (b, n/m, 2*heads*dim_head)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        # q: (b, heads, n, dim_head)
        # k, v: (b, heads, n/m, dim_head)

        q = self.q_norm(q)  # (b, heads, n, dim_head)
        k = self.k_norm(k)  # (b, heads, n/m, dim_head)

        dots = torch.matmul(q, k.transpose(-1, -2))  # (b, heads, n, n/m)

        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')  # (b, 1, 1, n/m)
            dots = dots.masked_fill(~mask, -torch.finfo(dots.dtype).max)

        if exists(attn_mask):
            dots = dots.masked_fill(~attn_mask, -torch.finfo(dots.dtype).max)  # attn_mask: (b, heads, n, n/m) or broadcastable

        attn = self.attend(dots)  # (b, heads, n, n/m)
        attn = self.dropout(attn)  # (b, heads, n, n/m)

        out = torch.matmul(attn, v)  # (b, heads, n, dim_head)
        out = rearrange(out, 'b h n d -> b n (h d)')  # (b, n, heads*dim_head)
        return self.to_out(out)  # (b, n, dim)


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
            x = attn(x, mask = mask, attn_mask = attn_mask) + x
            x = ff(x) + x

        return self.norm(x)


class NaViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., token_dropout_prob = None):
        super().__init__()
        image_height, image_width = pair(image_size)

        # 토큰을 드롭아웃할 확률(percentage)을 설정합니다.
        # 만약 int 또는 float가 주어진다면, 고정된 드롭아웃 확률로 간주합니다.
        # 아니면, 높이(height)와 너비(width)에 따라 드롭아웃 확률을 계산하는 콜백 함수도 허용합니다.

        self.calc_token_dropout = None

        if callable(token_dropout_prob):
            self.calc_token_dropout = token_dropout_prob

        elif isinstance(token_dropout_prob, (float, int)):
            assert 0. <= token_dropout_prob < 1.
            token_dropout_prob = float(token_dropout_prob)
            self.calc_token_dropout = lambda height, width: token_dropout_prob

        # calculate patching related stuff
        # 토큰 드롭아웃 계산 방식을 설정하고, 이미지 크기가 패치 크기로 정확히 나누어 떨어지는지 검증한다
        assert divisible_by(image_height, patch_size) and divisible_by(image_width, patch_size), 'Image dimensions must be divisible by the patch size.'

        patch_height_dim, patch_width_dim = (image_height // patch_size), (image_width // patch_size)
        patch_dim = channels * (patch_size ** 2)

        self.channels = channels
        self.patch_size = patch_size

        self.to_patch_embedding = nn.Sequential(
            LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            LayerNorm(dim),
        )

        self.pos_embed_height = nn.Parameter(torch.randn(patch_height_dim, dim))
        self.pos_embed_width = nn.Parameter(torch.randn(patch_width_dim, dim))

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        # final attention pooling queries

        self.attn_pool_queries = nn.Parameter(torch.randn(dim))
        self.attn_pool = Attention(dim = dim, dim_head = dim_head, heads = heads)

        # output to logits

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
        batched_images: List[Tensor] | List[List[Tensor]],  # (입력) List[Tensor] 또는 List[List[Tensor]].
                                                        # Tensor는 각 이미지: [C, H_i, W_i]
                                                        # List[List[Tensor]]라면 바깥 리스트 길이가 G(그룹 수)
        group_images = False,
        group_max_seq_len = 2048
    ):
        # 기본 파라미터 수집
        p, c, device, has_token_dropout = self.patch_size, self.channels, self.device, exists(self.calc_token_dropout) and self.training

        arange = partial(torch.arange, device = device)
        pad_sequence = partial(orig_pad_sequence, batch_first = True)

        # 1) 필요 시 토큰 예산(max_seq_len) 기준으로 이미지를 자동 그룹핑
        #    입력이 List[Tensor]일 때, 토큰 합이 group_max_seq_len을 넘지 않도록 List[List[Tensor]]로 나눔
        if group_images:
            batched_images = group_images_by_max_seq_len(
                batched_images,
                patch_size = self.patch_size,
                calc_token_dropout = self.calc_token_dropout if self.training else None,
                max_seq_len = group_max_seq_len
            )
            # 결과: List[List[Tensor]], 바깥 리스트 길이 = G

        # 2) 만약 아직 그룹화가 안되어(List[Tensor]) 들어왔다면 한 그룹으로 감싸서 List[List[Tensor]]로 통일
        if torch.is_tensor(batched_images[0]):
            batched_images = [batched_images]  # 이제 항상 List[List[Tensor]]

        # 3) 가변 길이 시퀀스로 변환 준비(토큰/좌표/마스크를 모을 버킷)
        num_images = []            # 각 그룹의 이미지 개수: 길이 G, 원소는 N_b
        batched_sequences = []     # 각 그룹의 토큰 시퀀스: [T_b, C*P*P] (concat 후)
        batched_positions = []     # 각 그룹의 (h,w) 패치 좌표: [T_b, 2]
        batched_image_ids = []     # 각 그룹의 토큰별 소속 이미지 id: [T_b]

        # 4) 그룹 단위로 순회 (images는 한 그룹에 들어있는 이미지 리스트, 길이 = N_b)
        for images in batched_images:
            num_images.append(len(images))  # N_b

            sequences = []                 # 그룹 내 각 이미지의 토큰: [ph*pw, C*P*P] 들을 모아 concat
            positions = []                 # 그룹 내 각 이미지의 좌표: [ph*pw, 2] 들을 모아 concat
            image_ids = torch.empty((0,), device = device, dtype = torch.long)  # 그룹 토큰별 소속 이미지 id 누적: [0] → [T_b]

            # 5) 그룹 안에서 이미지별로 토큰화
            for image_id, image in enumerate(images):
                assert image.ndim == 3 and image.shape[0] == c  # 이미지 Tensor: [C, H_i, W_i]
                image_dims = image.shape[-2:]                   # (H_i, W_i)
                assert all([divisible_by(dim, p) for dim in image_dims]), f'height and width {image_dims} of images must be divisible by patch size {p}'

                ph, pw = map(lambda dim: dim // p, image_dims)  # 패치 그리드 크기: ph=H_i/P, pw=W_i/P

                # (h,w) 패치 좌표 생성: pos: [ph, pw, 2] → reshape → [ph*pw, 2]
                pos = torch.stack(torch.meshgrid((
                    arange(ph),
                    arange(pw)
                ), indexing = 'ij'), dim = -1)
                pos = rearrange(pos, 'h w c -> (h w) c')        # [ph*pw, 2]

                # 이미지 → 패치 펼치기
                # seq: 각 패치가 하나의 토큰: [ph*pw, C*P*P]
                seq = rearrange(image, 'c (h p1) (w p2) -> (h w) (c p1 p2)', p1 = p, p2 = p)

                seq_len = seq.shape[-2]                         # ph*pw

                # 6) (학습 시) 토큰 드롭아웃 적용: keep_indices로 토큰 일부를 샘플링
                if has_token_dropout:
                    token_dropout = self.calc_token_dropout(*image_dims) # 스칼라
                    num_keep = max(1, int(seq_len * (1 - token_dropout)))
                    keep_indices = torch.randn((seq_len,), device = device).topk(num_keep, dim = -1).indices
                    seq = seq[keep_indices]                     # [num_keep, C*P*P]
                    pos = pos[keep_indices]                     # [num_keep, 2]

                # 7) 이 이미지에서 나온 토큰 수만큼 image_id를 채워 붙임
                image_ids = F.pad(image_ids, (0, seq.shape[-2]), value = image_id)  # 최종 [T_b]

                # 8) 그룹 누적 버킷에 push
                sequences.append(seq)                           # seq: [*, C*P*P]
                positions.append(pos)                           # pos: [*, 2]

            # 9) 그룹 단위로 concat → 이 그룹의 총 토큰 길이 T_b가 됨
            batched_image_ids.append(image_ids)                 # [T_b]
            batched_sequences.append(torch.cat(sequences, dim = 0))  # [T_b, C*P*P]
            batched_positions.append(torch.cat(positions, dim = 0))  # [T_b, 2]

        # 10) key padding mask 생성 (가변 길이 시퀀스를 패딩할 준비)  ?
        lengths = torch.tensor([seq.shape[-2] for seq in batched_sequences], device = device, dtype = torch.long)  # [G], 각 원소 T_b
        seq_arange = arange(lengths.amax().item())   # [L], L=max_b T_b
        key_pad_mask = rearrange(seq_arange, 'n -> 1 n') < rearrange(lengths, 'b -> b 1')
        # key_pad_mask: [G, L], True=유효토큰, False=패딩영역

        # 11) 어텐션 마스크(이미지 간 토큰 혼선 방지) 만들기
        #     먼저 토큰별 소속 이미지 id를 패딩 정렬
        batched_image_ids = pad_sequence(batched_image_ids)          # [G, L]
        #     같은 이미지 id 끼리만 attention 허용
        attn_mask = rearrange(batched_image_ids, 'b i -> b 1 i 1') == rearrange(batched_image_ids, 'b j -> b 1 1 j')
        #     패딩 토큰은 attention 차단
        attn_mask = attn_mask & rearrange(key_pad_mask, 'b j -> b 1 1 j')
        # attn_mask: [G, 1, L, L] (브로드캐스트 가능한 형태)

        # 12) 시퀀스/좌표도 패딩 정렬
        patches = pad_sequence(batched_sequences)     # [G, L, C*P*P]
        patch_positions = pad_sequence(batched_positions)  # [G, L, 2]  (각 토큰의 (h,w) 인덱스)

        # 13) 그룹별 이미지 개수 텐서화
        num_images = torch.tensor(num_images, device = device, dtype = torch.long)  # [G], 각 원소 N_b

        # 14) 패치 임베딩으로 투영
        x = self.to_patch_embedding(patches)          # [G, L, D], (Linear: C*P*P -> D)

        # 15) 2D 절대 위치 임베딩(분해형: height + width)
        h_indices, w_indices = patch_positions.unbind(dim = -1)  # 둘 다 [G, L]
        h_pos = self.pos_embed_height[h_indices]                 # [G, L, D]
        w_pos = self.pos_embed_width[w_indices]                  # [G, L, D]
        x = x + h_pos + w_pos                                    # [G, L, D]

        # 16) 드롭아웃
        x = self.dropout(x)                                      # [G, L, D]

        # 17) Transformer 인코더 통과 (마스크 적용)
        x = self.transformer(x, attn_mask = attn_mask)           # [G, L, D]

        # 18) 마지막에 이미지 단위로 attention pooling 수행
        max_queries = num_images.amax().item()                   # Q = max_b N_b

        # 풀링용 쿼리(공유 learnable vector)를 (G,Q,D)로 반복
        queries = repeat(self.attn_pool_queries, 'd -> b n d', n = max_queries, b = x.shape[0])  # [G, Q, D]

        # 19) 풀링 마스크: i번째 쿼리는 image_id==i 인 토큰만 볼 수 있음
        image_id_arange = arange(max_queries)                    # [Q]
        #  - batched_image_ids: [G, L]
        attn_pool_mask = rearrange(image_id_arange, 'i -> i 1') == rearrange(batched_image_ids, 'b j -> b 1 j')
        #    → [G, Q, L]
        attn_pool_mask = attn_pool_mask & rearrange(key_pad_mask, 'b j -> b 1 j')   # 패딩 차단: [G, Q, L]
        attn_pool_mask = rearrange(attn_pool_mask, 'b i j -> b 1 i j')              # [G, 1, Q, L] (모듈 호환)

        # 20) 어텐션 풀링: 각 이미지(쿼리)별로 해당 이미지 토큰 집합(context=x)에서 집계
        x = self.attn_pool(queries, context = x, attn_mask = attn_pool_mask) + queries  # [G, Q, D] (residual 추가)

        # 21) (G,Q,D) → (G*Q, D) 편평화
        x = rearrange(x, 'b n d -> (b n) d')                     # [(G*Q), D]

        # 22) 각 배치 b에서 실제 이미지 개수(N_b)보다 큰 쿼리 인덱스는 제거
        is_images = image_id_arange < rearrange(num_images, 'b -> b 1')  # [G, Q] (True=실제 이미지 쿼리)
        is_images = rearrange(is_images, 'b n -> (b n)')                 # [(G*Q)]
        x = x[is_images]                                                 # [M, D], M=∑_b N_b

        # 23) 분류/회귀 등의 최종 투영
        x = self.to_latent(x)                                            # [M, D] (옵션: LN/Proj 등)
        return self.mlp_head(x)                                          # [M, num_classes] (분류 가정)
