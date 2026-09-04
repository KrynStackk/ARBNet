from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import nn

from configs.config import SETUP

Tensor = torch.Tensor

NUM_APS = SETUP.system.num_aps
NT = SETUP.system.nt
NUM_USERS = SETUP.system.num_users
HIDDEN_DIM = SETUP.stage3.hidden_dim
NUM_MESSAGE_LAYERS = SETUP.stage3.num_layers
PER_AP_POWER = SETUP.system.per_ap_power
RZF_ALPHA = SETUP.stage3.alpha
MASK_INIT = SETUP.stage3.mask_init
P_MIN = SETUP.stage3.p_min
P_MAX = SETUP.stage3.p_max


def _to_complex(x: Tensor) -> Tensor:

    if torch.is_complex(x):
        if x.ndim != 3:
            raise ValueError(f"Complex input must be [B,N,J], got {tuple(x.shape)}")
        return x
    if x.ndim != 4 or x.shape[1] != 2:
        raise ValueError(f"Expected [B,2,N,J], got {tuple(x.shape)}")
    return torch.complex(x[:, 0].contiguous(), x[:, 1].contiguous())


def _to_real_imag(x: Tensor) -> Tensor:

    if not torch.is_complex(x) or x.ndim != 3:
        raise ValueError(f"Expected complex [B,N,J], got {tuple(x.shape)}")
    return torch.stack((x.real, x.imag), dim=1).contiguous()


def _standardize_edges(x: Tensor, eps: float) -> Tensor:

    mean = x.mean(dim=(1, 2), keepdim=True)
    std = x.std(dim=(1, 2), keepdim=True, unbiased=False).clamp_min(eps)
    return (x - mean) / std


def _logit(probability: float) -> float:
    p = min(max(float(probability), 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def _assert_close(name: str, actual: float, expected: float, tol: float = 1e-12) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=tol, abs_tol=tol):
        raise ValueError(f"{name} is fixed to {expected}, got {actual}")


class _EdgeUpdate(nn.Module):

    def __init__(self, hidden_dim: int, include_initial: bool) -> None:
        super().__init__()
        self.include_initial = include_initial
        input_dim = (4 if include_initial else 3) * hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 2 * hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, edge: Tensor, initial_edge: Tensor) -> Tensor:
        batch, num_aps, num_users, hidden = edge.shape
        ap_context = edge.mean(dim=2, keepdim=True).expand(
            batch, num_aps, num_users, hidden
        )
        user_context = edge.mean(dim=1, keepdim=True).expand(
            batch, num_aps, num_users, hidden
        )
        parts = (edge, ap_context, user_context, initial_edge) if self.include_initial else (edge, ap_context, user_context)
        update_input = torch.cat(parts, dim=-1)
        return self.norm(edge + self.mlp(update_input))


class GraphRZFBeamformer(nn.Module):


    def __init__(
        self,
        num_aps: int = NUM_APS,
        nt: int = NT,
        num_users: int = NUM_USERS,
        per_ap_power: float = PER_AP_POWER,
        alpha_init: float = RZF_ALPHA,
        hidden_dim: int = HIDDEN_DIM,
        num_layers: int = NUM_MESSAGE_LAYERS,
        mask_init: float = MASK_INIT,
        p_min: float = P_MIN,
        p_max: float = P_MAX,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self._validate_setup(
            num_aps=num_aps,
            nt=nt,
            num_users=num_users,
            per_ap_power=per_ap_power,
            alpha_init=alpha_init,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            mask_init=mask_init,
            p_min=p_min,
            p_max=p_max,
        )

        self.M = NUM_APS
        self.Nt = NT
        self.J = NUM_USERS
        self.N = self.M * self.Nt
        self.per_ap_power = PER_AP_POWER
        self.p_min = P_MIN
        self.p_max = P_MAX
        self.eps = float(eps)

        edge_input_dim = 4 * self.Nt + 3
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_input_dim, HIDDEN_DIM),
            nn.SiLU(inplace=True),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.SiLU(inplace=True),
        )
        self.layers = nn.ModuleList(
            [
                _EdgeUpdate(HIDDEN_DIM, include_initial=index > 0)
                for index in range(NUM_MESSAGE_LAYERS)
            ]
        )
        self.score_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.SiLU(inplace=True),
            nn.Linear(HIDDEN_DIM, 1),
        )
        self.power_head = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2),
            nn.SiLU(inplace=True),
            nn.Linear(HIDDEN_DIM // 2, 1),
        )


        self.register_buffer("fixed_alpha", torch.tensor(RZF_ALPHA, dtype=torch.float32))

        nn.init.zeros_(self.score_head[-1].weight)
        nn.init.constant_(self.score_head[-1].bias, _logit(MASK_INIT))
        nn.init.zeros_(self.power_head[-1].weight)
        nn.init.constant_(self.power_head[-1].bias, _logit(0.90))

    @staticmethod
    def _validate_setup(
        *,
        num_aps: int,
        nt: int,
        num_users: int,
        per_ap_power: float,
        alpha_init: float,
        hidden_dim: int,
        num_layers: int,
        mask_init: float,
        p_min: float,
        p_max: float,
    ) -> None:
        dimensions = (int(num_aps), int(nt), int(num_users))
        if dimensions != (NUM_APS, NT, NUM_USERS):
            raise ValueError(
                "Graph-RZF dimensions are fixed to "
                f"(M,Nt,J)=({NUM_APS},{NT},{NUM_USERS}), got {dimensions}"
            )
        if int(hidden_dim) != HIDDEN_DIM or int(num_layers) != NUM_MESSAGE_LAYERS:
            raise ValueError(
                f"Graph-RZF is fixed to hidden_dim={HIDDEN_DIM}, "
                f"num_layers={NUM_MESSAGE_LAYERS}"
            )
        _assert_close("per_ap_power", per_ap_power, PER_AP_POWER)
        _assert_close("alpha_init", alpha_init, RZF_ALPHA)
        _assert_close("mask_init", mask_init, MASK_INIT)
        _assert_close("p_min", p_min, P_MIN)
        _assert_close("p_max", p_max, P_MAX)

    def _validate_inputs(self, h_rec: Tensor, steering: Tensor) -> None:
        expected = (2, self.N, self.J)
        if h_rec.ndim != 4 or tuple(h_rec.shape[1:]) != expected:
            raise ValueError(
                f"Hrec must be [B,{expected[0]},{expected[1]},{expected[2]}], "
                f"got {tuple(h_rec.shape)}"
            )
        if steering.shape != h_rec.shape:
            raise ValueError(
                f"A must match Hrec; got A={tuple(steering.shape)}, "
                f"Hrec={tuple(h_rec.shape)}"
            )

    def _edge_features(self, h_rec: Tensor, steering: Tensor) -> Tuple[Tensor, Tensor]:
        batch = h_rec.shape[0]
        h_ap = h_rec.reshape(batch, self.M, self.Nt, self.J)
        a_ap = steering.reshape(batch, self.M, self.Nt, self.J)

        h_power = h_ap.abs().square().sum(dim=2)
        a_power = a_ap.abs().square().sum(dim=2)
        h_unit = h_ap / h_power.sqrt().clamp_min(self.eps)[:, :, None, :]
        a_unit = a_ap / a_power.sqrt().clamp_min(self.eps)[:, :, None, :]
        alignment = (h_unit.conj() * a_unit).sum(dim=2).abs().square().clamp(0.0, 1.0)

        h_vectors = h_unit.permute(0, 1, 3, 2).contiguous()
        a_vectors = a_unit.permute(0, 1, 3, 2).contiguous()
        vector_features = torch.cat(
            (h_vectors.real, h_vectors.imag, a_vectors.real, a_vectors.imag), dim=-1
        )
        scalar_features = torch.stack(
            (
                _standardize_edges(torch.log1p(h_power), self.eps),
                _standardize_edges(torch.log1p(a_power), self.eps),
                alignment,
            ),
            dim=-1,
        )
        return torch.cat((vector_features, scalar_features), dim=-1), h_ap

    def _rzf_direction(self, effective_channel: Tensor) -> Tensor:
        batch = effective_channel.shape[0]
        h_hermitian = effective_channel.conj().transpose(-2, -1)
        gram = h_hermitian @ effective_channel
        eye = torch.eye(
            self.J, dtype=effective_channel.dtype, device=effective_channel.device
        ).expand(batch, self.J, self.J)
        alpha = self.fixed_alpha.to(
            device=effective_channel.device, dtype=effective_channel.real.dtype
        )
        regularized_gram = gram + alpha.to(effective_channel.dtype) * eye
        try:
            factor = torch.linalg.cholesky(regularized_gram)
            solved = torch.cholesky_solve(h_hermitian, factor)
        except RuntimeError:
            solved = torch.linalg.solve(regularized_gram, h_hermitian)
        direction = solved.conj().transpose(-2, -1)
        norm = direction.abs().square().sum(dim=1, keepdim=True).sqrt().clamp_min(self.eps)
        return direction / norm

    def _beamform(self, h_ap: Tensor, mask: Tensor, user_power: Tensor) -> Tensor:
        batch = h_ap.shape[0]
        effective_ap = h_ap * mask[:, :, None, :].to(h_ap.dtype)
        effective_channel = effective_ap.reshape(batch, self.N, self.J)
        direction = self._rzf_direction(effective_channel)
        direction_ap = direction.reshape(batch, self.M, self.Nt, self.J)
        beamformer_ap = direction_ap * mask[:, :, None, :].to(direction.dtype)
        beamformer_ap = beamformer_ap * user_power[:, None, None, :].sqrt().to(direction.dtype)

        ap_power = beamformer_ap.abs().square().sum(dim=(2, 3), keepdim=True)
        scale = math.sqrt(self.per_ap_power) / ap_power.sqrt().clamp_min(self.eps)
        beamformer_ap = beamformer_ap * scale.to(beamformer_ap.dtype)
        return beamformer_ap.reshape(batch, self.N, self.J)

    def forward(
        self,
        h_rec_ri: Tensor,
        steering_ri: Tensor,
    ) -> Tensor:
        self._validate_inputs(h_rec_ri, steering_ri)
        h_rec = _to_complex(h_rec_ri)
        steering = _to_complex(steering_ri)

        edge_features, h_ap = self._edge_features(h_rec, steering)
        initial_edge = self.edge_encoder(edge_features)
        edge = initial_edge
        for layer in self.layers:
            edge = layer(edge, initial_edge)

        logits = self.score_head(edge).squeeze(-1)
        mask = torch.sigmoid(logits)

        user_embedding = edge.mean(dim=1)
        power_logits = self.power_head(user_embedding).squeeze(-1)
        user_power = self.p_min + (self.p_max - self.p_min) * torch.sigmoid(power_logits)

        return _to_real_imag(self._beamform(h_ap, mask, user_power))

