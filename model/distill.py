import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F

from VIT import ViT
from t2t import T2TViT
from efficient import ViT as EfficientViT

from einops import rearrange, repeat

# ---------------------------
# helpers
# ---------------------------

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# ---------------------------
# DistillMixin
#  - DeiT의 핵심인 "distillation token" 경로를 추가하기 위한 믹스인
#  - student 모델이 (선택적으로) distill_token을 함께 받아 attention 경로로 흘려보냄
#  - 논문 §4 Figure 2와 동일 개념:
#       [CLS] + [PATCH TOKENS] (+ [DISTILL TOKEN])
# ---------------------------

class DistillMixin:
    def forward(self, img, distill_token = None):
        """
        Args:
            img: (B, C, H, W) 입력 이미지
            distill_token: (1, 1, D) 모양의 학습 가능한 디스틸 토큰(옵션)
                - 주어지면 distillation 경로 활성화
        Returns:
            - distilling=False: class head 로짓 (B, num_classes)
            - distilling=True : (class head 로짓, distill token 마지막 레이어 출력 벡터)
                * distill 토큰은 별도 MLP로 num_classes 로짓으로 변환함 (wrapper에서)
        """
        distilling = exists(distill_token)

        # 1) 패치 임베딩 + [CLS] 토큰 + 위치 임베딩
        x = self.to_patch_embedding(img)            # (B, N, D)
        b, n, _ = x.shape

        # [CLS] 토큰을 배치 크기만큼 복제
        cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b = b)  # (B, 1, D)
        x = torch.cat((cls_tokens, x), dim = 1)                       # (B, 1+N, D)

        # 포지셔널 임베딩 추가(DeiT/ViT 공통)
        x += self.pos_embedding[:, :(n + 1)]                          # (B, 1+N, D)

        # 2) (선택) Distillation token 추가
        #    - 논문 §4: class token과 동일하게 self-attention 경로를 통과
        if distilling:
            distill_tokens = repeat(distill_token, '1 n d -> b n d', b = b)  # (B, 1, D)
            x = torch.cat((x, distill_tokens), dim = 1)                      # (B, 1+N+1, D)

        # 3) Transformer 인코더 통과 (MSA+FFN 반복)
        x = self._attend(x)  # (B, 1+N(+1), D)

        # 4) distillation 토큰 분리 (있다면 마지막 토큰이라고 가정)
        if distilling:
            x, distill_tokens = x[:, :-1], x[:, -1]  # x: (B, 1+N, D), distill_tokens: (B, D)

        # 5) 분류를 위한 대표 토큰 선택
        #    - DeiT/ViT 기본은 [CLS] 사용
        #    - self.pool == 'mean'이면 mean pooling, 아니면 [CLS] (x[:, 0])
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]   # (B, D)

        # 6) class head 로짓
        x = self.to_latent(x)             # (B, D)  (보통 Identity)
        out = self.mlp_head(x)            # (B, num_classes)

        # 7) distillation 경로: wrapper에서 별도의 MLP로 num_classes 로짓으로 변환
        if distilling:
            return out, distill_tokens    # class 로짓, distill 벡터

        return out                        # class 로짓만 반환

# ---------------------------
# Distillable 학생 모델들
#  - DeiT 논문은 ViT 구조 그대로 사용 → 여기서도 vit_pytorch의 ViT/T2T/Efficient 변형
#  - ._attend: 드롭아웃/트랜스포머 호출 지점만 통일
# ---------------------------

class DistillableViT(DistillMixin, ViT):
    def __init__(self, *args, **kwargs):
        super(DistillableViT, self).__init__(*args, **kwargs)
        self.args = args
        self.kwargs = kwargs
        self.dim = kwargs['dim']
        self.num_classes = kwargs['num_classes']

    def to_vit(self):
        # distill 경로 없는 일반 ViT로 state_dict 이식
        v = ViT(*self.args, **self.kwargs)
        v.load_state_dict(self.state_dict())
        return v

    def _attend(self, x):
        x = self.dropout(x)
        x = self.transformer(x)
        return x

class DistillableT2TViT(DistillMixin, T2TViT):
    def __init__(self, *args, **kwargs):
        super(DistillableT2TViT, self).__init__(*args, **kwargs)
        self.args = args
        self.kwargs = kwargs
        self.dim = kwargs['dim']
        self.num_classes = kwargs['num_classes']

    def to_vit(self):
        v = T2TViT(*self.args, **self.kwargs)
        v.load_state_dict(self.state_dict())
        return v

    def _attend(self, x):
        x = self.dropout(x)
        x = self.transformer(x)
        return x

class DistillableEfficientViT(DistillMixin, EfficientViT):
    def __init__(self, *args, **kwargs):
        super(DistillableEfficientViT, self).__init__(*args, **kwargs)
        self.args = args
        self.kwargs = kwargs
        self.dim = kwargs['dim']
        self.num_classes = kwargs['num_classes']

    def to_vit(self):
        v = EfficientViT(*self.args, **self.kwargs)
        v.load_state_dict(self.state_dict())
        return v

    def _attend(self, x):
        return self.transformer(x)

# ---------------------------
# DistillWrapper
#  - teacher/student를 감싸서 Hard/Soft KD를 수행
#  - "distillation token"을 학습 가능한 파라미터로 두고,
#    student forward에 넣어 attention 경로로 teacher 정보를 주입(DeiT 핵심)
#  - 논문 포인트와 매핑:
#      * Hard KD(teacher argmax) vs Soft KD(KL with temperature)
#      * Distillation token → 별도 MLP로 distill head 로짓 산출
#      * 추론 시 "class head"와 "distill head"를 late-fusion(softmax 합)하면 추가 이득 (여기선 주석으로 가이드)
# ---------------------------

class DistillWrapper(Module):
    def __init__(
        self,
        *,
        teacher: nn.Module,
        student: nn.Module,
        temperature: float = 1.,
        alpha: float = 0.5,
        hard: bool = False,
        mlp_layernorm: bool = False
    ):
        """
        Args:
            teacher: 강한 분류기 (ConvNet 권장, RegNetY-16GF 등)
            student: DistillableViT 계열 (ViT/T2T/Efficient 변형)
            temperature: soft KD용 온도 τ (보통 3.0 권장)
            alpha: KD 비중 λ (논문에선 0.1~0.5 범위, 실전 0.5도 흔함)
            hard: True면 Hard KD (teacher argmax), False면 Soft KD (KL)
            mlp_layernorm: distill head 앞에 LayerNorm을 추가할지 여부
        """
        super().__init__()
        assert isinstance(
            student, (DistillableViT, DistillableT2TViT, DistillableEfficientViT)
        ), 'student must be a vision transformer (Distillable*)'

        self.teacher = teacher.eval()           # teacher는 고정(grad X)
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.student = student

        dim = student.dim
        num_classes = student.num_classes
        self.temperature = temperature
        self.alpha = alpha
        self.hard = hard

        # 논문 §4: 학습 가능한 Distillation Token 파라미터
        self.distillation_token = nn.Parameter(torch.randn(1, 1, dim))

        # Distill head: distill token → (LayerNorm옵션) → Linear → num_classes
        self.distill_mlp = nn.Sequential(
            nn.LayerNorm(dim) if mlp_layernorm else nn.Identity(),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img, labels, temperature = None, alpha = None, **kwargs):
        """
        Returns:
            loss: (1 - alpha) * CE(student,class) + alpha * KD(distill, teacher)
        Notes:
            - Soft KD: KL( softmax(distill/T), softmax(teacher/T) ) * T^2
            - Hard KD: CE(distill, argmax(teacher))
            - student는 항상 class-head CE로 진짜 라벨 y에 맞게 학습
            - distill-head는 teacher 신호(y_t 또는 soft targets)에 맞게 학습
        """
        alpha = default(alpha, self.alpha)
        T = default(temperature, self.temperature)

        # 1) Teacher forward (gradient 차단)
        with torch.no_grad():
            teacher_logits = self.teacher(img)    # (B, num_classes)

        # 2) Student forward with distillation token
        #    - student는 (class 로짓, distill token 벡터) 반환
        student_logits, distill_tokens = self.student(
            img,
            distill_token = self.distillation_token,
            **kwargs
        )  # student_logits: (B, num_classes), distill_tokens: (B, D)

        # 3) distill head 로짓으로 변환
        distill_logits = self.distill_mlp(distill_tokens)  # (B, num_classes)

        # 4) class-head 경로: CE(student_logits, y)
        loss_ce = F.cross_entropy(student_logits, labels)

        # 5) distill-head 경로: Hard/Soft KD
        if not self.hard:
            # Soft distillation (논문 §4 식(2)): KL( student/T || teacher/T ) * T^2
            distill_loss = F.kl_div(
                F.log_softmax(distill_logits / T, dim = -1),
                F.softmax(teacher_logits / T, dim = -1).detach(),
                reduction = 'batchmean'
            )
            distill_loss *= T ** 2
        else:
            # Hard distillation (논문 §4 식(3)): teacher argmax를 라벨로 CE
            teacher_labels = teacher_logits.argmax(dim = -1)
            distill_loss = F.cross_entropy(distill_logits, teacher_labels)

        # 6) 최종 손실: (1-α) * CE + α * KD
        return loss_ce * (1 - alpha) + distill_loss * alpha

        # [추론(inference) 팁 - 논문 §5.2]
        # - 실제 테스트 시에는 아래 3가지 전략 중 선택:
        #   (a) class head만 사용
        #   (b) distill head만 사용
        #   (c) late-fusion: softmax(class) + softmax(distill) → argmax
        # - DeiT 논문에서는 (c) late-fusion이 가장 성능이 좋았음.
