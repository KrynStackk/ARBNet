from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from configs.config import SETUP

NT = SETUP.system.nt
NUM_USERS = SETUP.system.num_users
TOP_M = SETUP.stage2.top_m
QUANT_BITS = SETUP.stage2.quant_bits
HIDDEN_CHANNELS = 32
NUM_BLOCKS = 6
KERNEL_SIZE = 3
STAGE2_SETUP_ID = "sparse_beam_parb_h32_b6"


def _validate_csi(x: torch.Tensor, name: str) -> None:
    expected = (2, NT, NUM_USERS)
    if x.ndim != 4 or tuple(x.shape[1:]) != expected:
        raise ValueError(f"{name} must be [B,2,{NT},{NUM_USERS}], got {tuple(x.shape)}")


def ri_to_complex(x: torch.Tensor) -> torch.Tensor:

    _validate_csi(x, "CSI")
    return torch.complex(x[:, 0].contiguous(), x[:, 1].contiguous())


def complex_to_ri(x: torch.Tensor) -> torch.Tensor:

    if not torch.is_complex(x) or x.ndim != 3 or tuple(x.shape[1:]) != (NT, NUM_USERS):
        raise ValueError(
            f"Complex CSI must be [B,{NT},{NUM_USERS}], got {tuple(x.shape)}"
        )
    return torch.stack((x.real, x.imag), dim=1).contiguous()


def ant_fft_ri(x: torch.Tensor) -> torch.Tensor:

    return torch.fft.fft(ri_to_complex(x), dim=1, norm="ortho")


def ant_ifft_to_ri(x: torch.Tensor) -> torch.Tensor:

    if not torch.is_complex(x) or x.ndim != 3 or tuple(x.shape[1:]) != (NT, NUM_USERS):
        raise ValueError(
            f"Beam CSI must be complex [B,{NT},{NUM_USERS}], got {tuple(x.shape)}"
        )
    return complex_to_ri(torch.fft.ifft(x, dim=1, norm="ortho"))


def _top_m_mask(beam_csi: torch.Tensor) -> torch.Tensor:
    batch = beam_csi.shape[0]
    indices = beam_csi.abs().flatten(1).topk(
        TOP_M, dim=1, largest=True, sorted=False
    ).indices
    mask = torch.zeros(
        batch, NT * NUM_USERS, device=beam_csi.device, dtype=torch.bool
    )
    mask.scatter_(1, indices, True)
    return mask.reshape(batch, NT, NUM_USERS)


def _quantize_selected(beam_csi: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    sparse = beam_csi * mask
    components = torch.stack((sparse.real, sparse.imag), dim=1)
    scale = components.abs().flatten(1).amax(dim=1).reshape(-1, 1, 1, 1)
    scale = scale.clamp_min(1e-8)
    qmax = float(2 ** (QUANT_BITS - 1) - 1)
    quantized = torch.round(components / scale * qmax).clamp(-qmax, qmax)
    dequantized = quantized / qmax * scale
    return torch.complex(
        dequantized[:, 0].contiguous(), dequantized[:, 1].contiguous()
    ) * mask


def make_sparse_feedback(
    h_hat_ri: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:

    full_beam = ant_fft_ri(h_hat_ri)
    mask = _top_m_mask(full_beam)
    sparse = _quantize_selected(full_beam, mask)
    return complex_to_ri(sparse), mask[:, None].float()


class ResBlock(nn.Module):


    def __init__(self) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, HIDDEN_CHANNELS)
        self.conv1 = nn.Conv2d(
            HIDDEN_CHANNELS, HIDDEN_CHANNELS, KERNEL_SIZE, padding=1
        )
        self.norm2 = nn.GroupNorm(8, HIDDEN_CHANNELS)
        self.conv2 = nn.Conv2d(
            HIDDEN_CHANNELS, HIDDEN_CHANNELS, KERNEL_SIZE, padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.conv1(F.silu(self.norm1(x)))
        residual = self.conv2(F.silu(self.norm2(residual)))
        return x + residual


class SparseBeamRefiner(nn.Module):


    def __init__(self) -> None:
        super().__init__()
        self.in_proj = nn.Conv2d(2, HIDDEN_CHANNELS, KERNEL_SIZE, padding=1)
        self.blocks = nn.Sequential(*[ResBlock() for _ in range(NUM_BLOCKS)])
        self.out_norm = nn.GroupNorm(8, HIDDEN_CHANNELS)
        self.out_proj = nn.Conv2d(HIDDEN_CHANNELS, 2, KERNEL_SIZE, padding=1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        sparse_ri: torch.Tensor,
        known_mask: torch.Tensor,
    ) -> torch.Tensor:
        _validate_csi(sparse_ri, "sparse_ri")
        expected_mask = (sparse_ri.shape[0], 1, NT, NUM_USERS)
        if tuple(known_mask.shape) != expected_mask:
            raise ValueError(
                f"known_mask must be [B,1,{NT},{NUM_USERS}], got {tuple(known_mask.shape)}"
            )


        feature = self.blocks(self.in_proj(sparse_ri))
        correction = self.out_proj(F.silu(self.out_norm(feature)))
        proposal = sparse_ri + correction
        return known_mask * sparse_ri + (1.0 - known_mask) * proposal
