from __future__ import annotations

import math

import torch
from torch import nn
from einops import repeat
from timm.models.layers import DropPath

from configs.config import SETUP

NT = SETUP.system.nt
NUM_USERS = SETUP.system.num_users
IN_CHANNELS = SETUP.stage1.in_chans
OUT_CHANNELS = SETUP.stage1.out_chans
PRE_CNN_DEPTH = SETUP.stage1.pre_ant_depth
BEAM_DEPTH = SETUP.stage1.seq_beam_depth
POST_CNN_DEPTH = SETUP.stage1.post_ant_depth
CNN_WIDTH = SETUP.stage1.cnn_width
CNN_KERNEL = SETUP.stage1.cnn_kernel
EMBED_DIM = SETUP.stage1.embed_dim
HIDDEN_DIM = SETUP.stage1.hidden_dim
COND_DIM = SETUP.stage1.cond_dim
DROP_PATH = SETUP.stage1.drop_path
SSM_D_STATE = SETUP.stage1.ssm_d_state
VSS_SCAN_NUMBER = SETUP.stage1.vss_scan_number
VSS_SSM_RATIO = SETUP.stage1.vss_ssm_ratio
VSS_SSM_RANK_RATIO = SETUP.stage1.vss_ssm_rank_ratio
VSS_CONV = SETUP.stage1.vss_conv
FFT_NORM = "ortho"
STAGE1_SETUP_ID = "hybrid_fft_ssm_d2_bssb2"


class DepthwiseSeparableConvBlock(nn.Module):


    def __init__(self) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                CNN_WIDTH,
                CNN_WIDTH,
                kernel_size=CNN_KERNEL,
                padding=CNN_KERNEL // 2,
                groups=CNN_WIDTH,
                bias=False,
            ),
            nn.BatchNorm2d(CNN_WIDTH),
            nn.GELU(),
            nn.Conv2d(CNN_WIDTH, CNN_WIDTH, kernel_size=1, bias=False),
            nn.BatchNorm2d(CNN_WIDTH),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class CNNResidualStage(nn.Module):


    def __init__(self, depth: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(IN_CHANNELS, CNN_WIDTH, kernel_size=1, bias=False),
            nn.BatchNorm2d(CNN_WIDTH),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            *[DepthwiseSeparableConvBlock() for _ in range(depth)],
        )
        self.head = nn.Conv2d(CNN_WIDTH, OUT_CHANNELS, kernel_size=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.stem(observation)))


class HybridFFTSSMEstimator(nn.Module):


    def __init__(self) -> None:
        super().__init__()

        self.gram_encoder = nn.Sequential(
            nn.Conv2d(2 * NUM_USERS, COND_DIM, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(COND_DIM, COND_DIM, kernel_size=1),
            nn.GELU(),
        )
        self.pre_ant_stage = CNNResidualStage(PRE_CNN_DEPTH)
        self.beam_stage = GramFiLMVSSResidualStage(
            depth=BEAM_DEPTH
        )
        self.post_ant_stage = CNNResidualStage(POST_CNN_DEPTH)

    @staticmethod
    def _validate_inputs(observation: torch.Tensor, gram: torch.Tensor) -> None:
        expected_observation = (IN_CHANNELS, NT, NUM_USERS)
        if observation.ndim != 4 or tuple(observation.shape[1:]) != expected_observation:
            raise ValueError(
                f"x must be [B,2,{NT},{NUM_USERS}], got {tuple(observation.shape)}"
            )
        expected_gram = (observation.shape[0], IN_CHANNELS, NUM_USERS, NUM_USERS)
        if gram.ndim != 4 or tuple(gram.shape) != expected_gram:
            raise ValueError(
                f"G must be [B,2,{NUM_USERS},{NUM_USERS}], got {tuple(gram.shape)}"
            )

    def _condition_map(self, gram: torch.Tensor) -> torch.Tensor:
        batch = gram.shape[0]
        gram_columns = gram.reshape(batch, 2 * NUM_USERS, NUM_USERS)
        expanded = gram_columns.unsqueeze(2).expand(
            batch, 2 * NUM_USERS, NT, NUM_USERS
        )
        return self.gram_encoder(expanded.contiguous())

    def forward(
        self,
        observation: torch.Tensor,
        G: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_inputs(observation, G)
        G = G.to(device=observation.device, dtype=observation.dtype)
        condition = self._condition_map(G)

        pre_residual = self.pre_ant_stage(observation)
        pre_refined = observation + pre_residual

        beam = fft_ant_ri(pre_refined, norm=FFT_NORM)
        beam_residual = self.beam_stage(beam, condition)
        refined_beam = beam + beam_residual
        antenna = ifft_ant_ri(refined_beam, norm=FFT_NORM)

        post_residual = self.post_ant_stage(antenna)
        estimate = antenna + post_residual

        return estimate


try:
    SSMODE = "sscore"
    import adaptive_selective_scan_cuda_core
    selective_scan_cuda_core = adaptive_selective_scan_cuda_core
except ImportError:
    try:
        SSMODE = "mamba_ssm"
        import selective_scan_cuda
    except ImportError as exc:
        raise ImportError(
            "Stage-1 requires adaptive_selective_scan_cuda_core or "
            "selective_scan_cuda. Build the no-SNR selective-scan extension "
            "before importing the proposed Stage-1 model."
        ) from exc


class SelectiveScan(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1):
        assert nrows in [1, 2, 3, 4], f"{nrows}"
        assert u.shape[1] % (B.shape[1] * nrows) == 0, f"{nrows}, {u.shape}, {B.shape}"
        ctx.delta_softplus = delta_softplus
        ctx.nrows = nrows

        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if D is not None:
            D = D.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()
        if B.dim() == 3:
            B = B.unsqueeze(dim=1)
            ctx.squeeze_B = True
        if C.dim() == 3:
            C = C.unsqueeze(dim=1)
            ctx.squeeze_C = True

        if SSMODE == "mamba_ssm":
            out, x, *_ = selective_scan_cuda.fwd(
                u, delta, A, B, C, D, None, delta_bias, delta_softplus
            )
        else:
            out, x, *_ = selective_scan_cuda_core.fwd(
                u, delta, A, B, C, D, delta_bias, delta_softplus, nrows
            )
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)

        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors

        if dout.stride(-1) != 1:
            dout = dout.contiguous()

        if SSMODE == "mamba_ssm":
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *_ = selective_scan_cuda.bwd(
                u,
                delta,
                A,
                B,
                C,
                D,
                None,
                delta_bias,
                dout,
                x,
                None,
                None,
                ctx.delta_softplus,
                False,
            )
        else:

            du, ddelta, dA, dB, dC, dD, ddelta_bias, *_ = selective_scan_cuda_core.bwd(
                u,
                delta,
                A,
                B,
                C,
                D,
                delta_bias,
                dout,
                x,
                ctx.delta_softplus,
                1

            )
        dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
        dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC


        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None)


class CrossScan2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        batch, channels, height, width = x.shape
        ctx.shape = (batch, channels, height, width)
        sequences = x.new_empty((batch, 2, channels, height * width))
        sequences[:, 0] = x.flatten(2, 3)
        sequences[:, 1] = torch.flip(sequences[:, 0], dims=[-1])
        return sequences

    @staticmethod
    def backward(ctx, sequences: torch.Tensor):
        batch, channels, height, width = ctx.shape
        length = height * width
        merged = sequences[:, 0].view(batch, 1, channels, length)
        merged = merged + sequences[:, 1].flip(dims=[-1]).view(
            batch, 1, channels, length
        )
        return merged.view(batch, channels, height, width)


class CrossMerge2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sequences: torch.Tensor):
        batch, _, channels, height, width = sequences.shape
        ctx.shape = (height, width)
        sequences = sequences.view(batch, 2, channels, -1)
        return (
            sequences[:, 0].view(batch, 1, channels, -1)
            + sequences[:, 1].flip(dims=[-1]).view(batch, 1, channels, -1)
        ).view(batch, channels, -1)

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        height, width = ctx.shape
        batch, channels, length = x.shape
        sequences = x.new_empty((batch, 2, channels, length))
        sequences[:, 0] = x
        sequences[:, 1] = torch.flip(x, dims=[-1])
        return sequences.view(batch, 2, channels, height, width)


def cross_selective_scan(
    x: torch.Tensor,
    x_proj_weight: torch.Tensor,
    dt_projs_weight: torch.Tensor,
    dt_projs_bias: torch.Tensor,
    A_logs: torch.Tensor,
    Ds: torch.Tensor,
    out_norm: torch.nn.Module,
) -> torch.Tensor:
    input_dtype = x.dtype
    batch, _, height, width = x.shape
    directions, _, rank = dt_projs_weight.shape
    state_dim = A_logs.shape[1]
    length = height * width

    sequences = CrossScan2.apply(x)
    projected = torch.einsum(
        "b k d l, k c d -> b k c l", sequences, x_proj_weight
    )
    deltas, B, C = torch.split(
        projected, [rank, state_dim, state_dim], dim=2
    )
    deltas = torch.einsum(
        "b k r l, k d r -> b k d l", deltas, dt_projs_weight
    )

    sequences = sequences.reshape(batch, -1, length).float()
    deltas = deltas.contiguous().reshape(batch, -1, length).float()
    A = -torch.exp(A_logs.float())
    B = B.contiguous().float()
    C = C.contiguous().float()
    D = Ds.float()
    delta_bias = dt_projs_bias.reshape(-1).float()

    outputs = SelectiveScan.apply(
        sequences,
        deltas,
        A,
        B,
        C,
        D,
        delta_bias,
        True,
        1,
    ).view(batch, directions, -1, height, width)

    merged = CrossMerge2.apply(outputs).transpose(1, 2).contiguous()
    merged = out_norm(merged)
    return merged.view(batch, height, width, -1).to(input_dtype)


class SS2D(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int,
        ssm_ratio: float,
        ssm_rank_ratio: float,
        d_conv: int,
    ) -> None:
        super().__init__()
        expanded_dim = int(ssm_ratio * d_model)
        inner_dim = int(min(ssm_rank_ratio, ssm_ratio) * d_model)
        self.dt_rank = math.ceil(d_model / 16)
        self.d_state = int(d_state)
        self.low_rank = inner_dim < expanded_dim

        self.in_proj = nn.Linear(d_model, 2 * expanded_dim, bias=False)
        self.conv2d = nn.Conv2d(
            expanded_dim,
            expanded_dim,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            groups=expanded_dim,
            bias=True,
        )
        self.act = nn.SiLU()

        if self.low_rank:
            self.in_rank = nn.Conv2d(expanded_dim, inner_dim, kernel_size=1, bias=False)
            self.out_rank = nn.Linear(inner_dim, expanded_dim, bias=False)

        directions = int(VSS_SCAN_NUMBER)
        x_projections = [
            nn.Linear(inner_dim, self.dt_rank + 2 * self.d_state, bias=False)
            for _ in range(directions)
        ]
        self.x_proj_weight = nn.Parameter(
            torch.stack([projection.weight for projection in x_projections], dim=0)
        )

        delta_projections = [
            self._init_delta_projection(self.dt_rank, inner_dim)
            for _ in range(directions)
        ]
        self.dt_projs_weight = nn.Parameter(
            torch.stack([projection.weight for projection in delta_projections], dim=0)
        )
        self.dt_projs_bias = nn.Parameter(
            torch.stack([projection.bias for projection in delta_projections], dim=0)
        )
        self.A_logs = self._init_A_logs(
            self.d_state, inner_dim, copies=directions
        )
        self.Ds = self._init_D(inner_dim, copies=directions)
        self.out_norm = nn.LayerNorm(inner_dim)
        self.out_proj = nn.Linear(expanded_dim, d_model, bias=False)

    @staticmethod
    def _init_delta_projection(dt_rank: int, inner_dim: int) -> nn.Linear:
        projection = nn.Linear(dt_rank, inner_dim, bias=True)
        scale = dt_rank ** -0.5
        nn.init.uniform_(projection.weight, -scale, scale)
        delta = torch.exp(
            torch.rand(inner_dim) * (math.log(0.1) - math.log(0.001))
            + math.log(0.001)
        ).clamp(min=1e-4)
        inverse_softplus = delta + torch.log(-torch.expm1(-delta))
        with torch.no_grad():
            projection.bias.copy_(inverse_softplus)
        return projection

    @staticmethod
    def _init_A_logs(state_dim: int, inner_dim: int, copies: int) -> nn.Parameter:
        A = repeat(
            torch.arange(1, state_dim + 1, dtype=torch.float32),
            "n -> d n",
            d=inner_dim,
        ).contiguous()
        A_logs = repeat(torch.log(A), "d n -> r d n", r=copies).flatten(0, 1)
        parameter = nn.Parameter(A_logs)
        parameter._no_weight_decay = True
        return parameter

    @staticmethod
    def _init_D(inner_dim: int, copies: int) -> nn.Parameter:
        D = repeat(torch.ones(inner_dim), "n -> r n", r=copies).flatten(0, 1)
        parameter = nn.Parameter(D)
        parameter._no_weight_decay = True
        return parameter

    def _selective_scan(self, x: torch.Tensor) -> torch.Tensor:
        if self.low_rank:
            x = self.in_rank(x)
        x = cross_selective_scan(
            x,
            self.x_proj_weight,
            self.dt_projs_weight,
            self.dt_projs_bias,
            self.A_logs,
            self.Ds,
            self.out_norm,
        )
        if self.low_rank:
            x = self.out_rank(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        content, gate = self.in_proj(x).chunk(2, dim=-1)
        gate = self.act(gate)
        content = content.permute(0, 3, 1, 2).contiguous()
        content = self.act(self.conv2d(content))
        scanned = self._selective_scan(content)
        return self.out_proj(scanned * gate)


class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        drop_path: float,
        ssm_d_state: int,
        ssm_ratio: float,
        ssm_rank_ratio: float,
        ssm_conv: int,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.op = SS2D(
            d_model=hidden_dim,
            d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            ssm_rank_ratio=ssm_rank_ratio,
            d_conv=ssm_conv,
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.drop_path(self.op(self.norm(x)))


def ri_to_complex(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 4 or x.shape[1] != 2:
        raise ValueError(f"Expected [B,2,Nt,J], got {tuple(x.shape)}")
    return torch.complex(x[:, 0].contiguous(), x[:, 1].contiguous())


def complex_to_ri(x: torch.Tensor) -> torch.Tensor:
    if not torch.is_complex(x) or x.ndim != 3:
        raise ValueError(f"Expected complex [B,Nt,J], got {tuple(x.shape)}")
    return torch.stack((x.real, x.imag), dim=1).contiguous()


def fft_ant_ri(x: torch.Tensor, norm: str = "ortho") -> torch.Tensor:

    if norm != "ortho":
        raise ValueError(f"FFT normalization is fixed to 'ortho', got {norm!r}")
    transformed = torch.fft.fft(ri_to_complex(x), dim=1, norm="ortho")
    return complex_to_ri(transformed).to(dtype=x.dtype)


def ifft_ant_ri(x: torch.Tensor, norm: str = "ortho") -> torch.Tensor:

    if norm != "ortho":
        raise ValueError(f"FFT normalization is fixed to 'ortho', got {norm!r}")
    transformed = torch.fft.ifft(ri_to_complex(x), dim=1, norm="ortho")
    return complex_to_ri(transformed).to(dtype=x.dtype)


class GramFiLMVSSBlock(nn.Module):


    def __init__(self) -> None:
        super().__init__()
        self.vss = VSSBlock(
            hidden_dim=HIDDEN_DIM,
            drop_path=DROP_PATH,
            ssm_d_state=SSM_D_STATE,
            ssm_ratio=VSS_SSM_RATIO,
            ssm_rank_ratio=VSS_SSM_RANK_RATIO,
            ssm_conv=VSS_CONV,
        )
        self.film = nn.Conv2d(COND_DIM, 2 * HIDDEN_DIM, kernel_size=1)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(
        self,
        feature: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        feature = self.vss(feature)
        gamma_beta = self.film(condition).permute(0, 2, 3, 1).contiguous()
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        return feature * (1.0 + gamma) + beta


class GramFiLMVSSResidualStage(nn.Module):


    def __init__(self, depth: int) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Conv2d(IN_CHANNELS, EMBED_DIM, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(EMBED_DIM, EMBED_DIM, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.channel_expand = nn.Sequential(
            nn.Conv2d(EMBED_DIM, HIDDEN_DIM, kernel_size=1),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [GramFiLMVSSBlock() for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(HIDDEN_DIM)
        self.channel_reduce = nn.Sequential(
            nn.Conv2d(HIDDEN_DIM, EMBED_DIM, kernel_size=1),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Conv2d(EMBED_DIM, EMBED_DIM, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(EMBED_DIM, OUT_CHANNELS, kernel_size=1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        gram_condition: torch.Tensor,
    ) -> torch.Tensor:
        feature = self.input_proj(x)
        feature = self.channel_expand(feature).permute(0, 2, 3, 1).contiguous()
        for block in self.blocks:
            feature = block(feature, gram_condition)
        feature = self.norm(feature).permute(0, 3, 1, 2).contiguous()
        return self.head(self.channel_reduce(feature))
