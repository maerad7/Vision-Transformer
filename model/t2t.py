import math
import torch
from torch import nn

from VIT import Transformer  # ⚠️ 사용자가 가진 ViT Transformer 블록(Encoder). MSA+MLP(+LN, Residual)로 구성되어 있다고 가정.

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------

def exists(val):
    """None 체크 유틸 함수"""
    return val is not None

def conv_output_size(image_size, kernel_size, stride, padding):
    """
    논문 Eq.(3) 의 1D 형태 버전과 동일한 개념:
      lo = floor((h + 2p - k) / (k - s) + 1) 와 구조적으로 유사
    여기서는 PyTorch의 unfold 결과에 해당하는 출력 Spatial 크기를 추정.
    - image_size: 현재 (정방형) 이미지 한 변 길이 h 또는 w
    - kernel_size: k
    - stride: 여기서는 (k - s) 에 해당 (즉, s = k - stride → 겹침 = kernel_size - stride)
    - padding: p
    """
    return int(((image_size - kernel_size + (2 * padding)) / stride) + 1)


# ------------------------------------------------------------
# classes
# ------------------------------------------------------------

class RearrangeImage(nn.Module):
    """
    [Re-structurization 단계] 토큰 시퀀스를 (B, H*W, C) → (B, C, H, W) 로 재구성
    논문 Eq.(2): I = Reshape(T')
      - 입력 x: (b, l, c), 여기서 l = h * w 가정
      - 출력: (b, c, h, w)
    """
    def forward(self, x):
        # h*w = seq_len 이라 가정하고 정방형으로 역변환 (h = sqrt(l))
        return rearrange(x, 'b (h w) c -> b c h w', h = int(math.sqrt(x.shape[1])))


class T2TViT(nn.Module):
    """
    T2T-ViT 구현 개요:
      - 전단(T2T module): (Re-structurization + Soft Split) 를 n=2 번 반복하여
        로컬 구조를 토큰에 점진적으로 내재화하고 토큰 길이를 감소.
        * Soft Split 은 nn.Unfold 로 구현 (겹침: overlap = kernel_size - stride)
        * Restructurization 은 RearrangeImage 및 Transformer(=MSA+MLP) 로 구현
        * 논문 기본 설정: P = [7, 3, 3], S = [3, 1, 1] → (k, stride) = (7, 4), (3, 2), (3, 2)
      - 후단(Backbone): ViT 인코더(Transformer)를 깊-좁(Deep-Narrow)로 설계하여
        파라미터/연산을 줄이면서 표현력을 향상.
    """
    def __init__(
        self,
        *,
        image_size,            # 입력 이미지 한 변 (예: 224)
        num_classes,           # 최종 클래스 수
        dim,                   # 임베딩 차원 (백본 입력/출력 dim, 예: 384)
        depth = None,          # 백본 Transformer 레이어 수 (예: 14)
        heads = None,          # 백본 MSA head 수
        mlp_dim = None,        # 백본 MLP hidden 차원 (Deep-Narrow → 512~1536 등)
        pool = 'cls',          # 'cls' 또는 'mean'
        channels = 3,          # 입력 채널 수 (RGB=3)
        dim_head = 64,         # 헤드당 차원
        dropout = 0.,          # 드롭아웃
        emb_dropout = 0.,      # 포지셔널 임베딩 투입 전 드롭아웃
        transformer = None,    # 외부에서 주입 가능한 백본 Transformer (선택)
        t2t_layers = ((7, 4), (3, 2), (3, 2)),  # (kernel, stride) 3단계: P=[7,3,3], S=[3,1,1]
    ):
        super().__init__()
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # ----------------------------------------------------
        # T2T 모듈 구성 (Eq.(1)~(4) 반복)
        #   - layer_dim: 현재 단계 토큰의 채널(C) 크기 (Unfold로 k*k*prev_C 로 증가)
        #   - output_image_size: 각 Soft Split 후의 (정방형) Spatial 크기 추정
        # ----------------------------------------------------
        layers = []
        layer_dim = channels         # 초기 C = 입력 채널 수(3)
        output_image_size = image_size  # 초기 H=W=image_size (예: 224)

        # t2t_layers: [(k1, s1), (k2, s2), (k3, s3)]
        #   stride = (k - overlap) 이므로 overlap = k - stride
        #   여기서는 (7,4) → overlap=3, (3,2) → overlap=1 로 논문 설정과 일치
        for i, (kernel_size, stride) in enumerate(t2t_layers):
            # Unfold 후 채널 증폭: k*k*prev_C (각 패치를 펼쳐 하나의 토큰으로 연결)
            layer_dim *= kernel_size ** 2

            is_first = (i == 0)
            is_last  = (i == (len(t2t_layers) - 1))

            # 출력 Spatial 크기 업데이트 (정방형 가정)
            # padding = stride // 2 로 설정 → 대략적인 경계 보존 & 논문 S=[3,1,1] 에 맞춘 overlap 유도
            output_image_size = conv_output_size(
                output_image_size,
                kernel_size,
                stride,
                stride // 2
            )

            # ---- 한 T2T 단계 구성 ----
            # (1) Re-structurization: 이전 단계 토큰 → 이미지 (B, C, H, W)
            #     첫 단계(i=0)는 입력이 이미지이므로 Reshape 불필요
            layers.extend([
                RearrangeImage() if not is_first else nn.Identity(),

                # (2) Soft Split: nn.Unfold 로 겹치는 패치 추출
                #     - kernel_size=k, stride=(k-s), padding= stride//2
                #     - 출력 shape: (B, k*k*C, N) → rearrange 로 (B, N, k*k*C) 토큰 시퀀스
                nn.Unfold(kernel_size = kernel_size, stride = stride, padding = stride // 2),
                Rearrange('b c n -> b n c'),

                # (3) T' = MLP(MSA(T)) (Eq.(1)) 을 한 번 적용하여 로컬+글로벌 상호작용
                #     마지막 단계 직전까지만 Transformer(1층, head=1)로 변환을 수행
                Transformer(
                    dim = layer_dim,
                    heads = 1,          # T2T 단계의 MSA head는 1로 고정(간단한 상호작용)
                    depth = 1,          # 단계당 1층의 얕은 Transformer (메모리/연산 절약)
                    dim_head = layer_dim,  # head dim 지정 (여기서는 단일 head → 전체 dim)
                    mlp_dim = layer_dim,
                    dropout = dropout
                ) if not is_last else nn.Identity(),
            ])

        # (4) 마지막 T2T 단계 뒤: 토큰 임베딩 차원을 백본 dim 으로 사상 (Deep-Narrow 시작점)
        layers.append(nn.Linear(layer_dim, dim))
        self.to_patch_embedding = nn.Sequential(*layers)

        # ----------------------------------------------------
        # 위치 임베딩 & CLS 토큰
        #   - output_image_size: T2T 모듈 마지막 Soft Split 후의 정방형 길이 (예: 14)
        #   - pos_embedding 길이: (토큰 수 N = output_image_size^2) + CLS(1)
        # ----------------------------------------------------
        self.pos_embedding = nn.Parameter(torch.randn(1, output_image_size ** 2 + 1, dim))
        self.cls_token     = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout       = nn.Dropout(emb_dropout)

        # ----------------------------------------------------
        # 백본 Transformer (Deep-Narrow 권장)
        #   - 외부 주입이 없으면 depth/heads/mlp_dim 필수
        #   - 논문 권장: hidden dim=256~512, mlp_dim=512~1536, depth(레이어수)↑
        # ----------------------------------------------------
        if not exists(transformer):
            assert all([exists(depth), exists(heads), exists(mlp_dim)]), 'depth, heads, and mlp_dim must be supplied'
            self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        else:
            self.transformer = transformer

        # 풀링 방식: CLS 또는 mean
        self.pool = pool
        self.to_latent = nn.Identity()

        # 최종 분류기
        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img):
        """
        순서 요약:
          1) T2T 모듈: (이미지 → [Re-structure → Soft Split → (MSA+MLP)] × n) → 토큰(고정 길이)
          2) CLS 토큰 prepend + 포지셔널 임베딩 추가
          3) 백본 Transformer (Deep-Narrow)로 전역 관계 학습
          4) 풀링(CLS/mean) → FC(head) 로 로짓 출력
        """
        # 1) T2T 모듈: 이미지 → 토큰 임베딩 (B, N, dim)
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        # 2) CLS 토큰 + 포지셔널 임베딩 (논문 Fig.4)
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)  # (1,1,dim) → (B,1,dim)
        x = torch.cat((cls_tokens, x), dim=1)                          # (B, 1+N, dim)
        x += self.pos_embedding[:, :n+1]                               # (B, 1+N, dim)
        x = self.dropout(x)

        # 3) 백본 Transformer 통과 (MSA+MLP 반복, Deep-Narrow 구조)
        x = self.transformer(x)

        # 4) 풀링 및 분류 헤드
        #    - 'cls': CLS 위치만 사용
        #    - 'mean': 모든 토큰 평균
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


v = T2TViT(
    dim = 512,
    image_size = 224,
    depth = 5,
    heads = 8,
    mlp_dim = 512,
    num_classes = 1000,
    t2t_layers = ((7, 4), (3, 2), (3, 2)) # tuples of the kernel size and stride of each consecutive layers of the initial token to token module
)

img = torch.randn(1, 3, 224, 224)

preds = v(img)