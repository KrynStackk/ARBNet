from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from configs.config import SETUP
from data.stage3 import Stage3Dataset
from torch import nn
from models.stage3 import GraphRZFBeamformer


def save_checkpoint(
    model: torch.nn.Module,
    epoch: int,
    save_path: Path,
    setup_id: str | None = None,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": int(epoch),
        "model": model.state_dict(),
    }
    if setup_id is not None:
        checkpoint["setup_id"] = setup_id
    torch.save(checkpoint, save_path)


def db(x: float, eps: float = 1e-12) -> float:
    return 10.0 * math.log10(max(float(x), eps))


def tensor_float(x) -> float:
    if torch.is_tensor(x):
        return float(x.detach().cpu())
    return float(x)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    model.train()
    for Hrec, A, Htrue in loader:
        Hrec = Hrec.to(device, non_blocking=True)
        A = A.to(device, non_blocking=True)
        Htrue = Htrue.to(device, non_blocking=True)

        F = model(Hrec, A)
        out = stage3_loss(Htrue, A, F, args)
        loss = out["loss"]

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, float]:
    model.eval()
    sums = {"loss": 0.0, "mean_sensing_snr": 0.0, "min_communication_sinr": 0.0}
    n_samples = 0

    for Hrec, A, Htrue in loader:
        Hrec = Hrec.to(device, non_blocking=True)
        A = A.to(device, non_blocking=True)
        Htrue = Htrue.to(device, non_blocking=True)

        F = model(Hrec, A)
        out = stage3_loss(Htrue, A, F, args)

        batch_size = Hrec.size(0)
        for key in sums:
            sums[key] += tensor_float(out[key]) * batch_size
        n_samples += batch_size

    denom = max(n_samples, 1)
    metrics = {key: value / denom for key, value in sums.items()}
    metrics["mean_sensing_snr_db"] = db(metrics["mean_sensing_snr"])
    metrics["min_communication_sinr_db"] = db(metrics["min_communication_sinr"])
    return metrics


def print_eval(metrics: Dict[str, float]) -> None:

    print("\n[Stage 3 | Final Evaluation]")
    print(f"Sensing SNR       : {metrics['mean_sensing_snr_db']:+.2f} dB")
    print(f"Communication SINR: {metrics['min_communication_sinr_db']:+.2f} dB")


def parse_args() -> argparse.Namespace:
    system = SETUP.system
    setup = SETUP.stage3
    ap = argparse.ArgumentParser(description="Train the fixed Stage-3 Graph-RZF beamformer")
    ap.add_argument("--data-dir", type=str, default=SETUP.stage2.stage3_data_dir)
    ap.add_argument("--checkpoint", type=str, default=setup.checkpoint)
    ap.add_argument("--save-dir", type=str, default=setup.save_dir)
    ap.add_argument("--run-id", type=str, default=setup.run_id)
    ap.add_argument("--num-aps", type=int, default=system.num_aps)
    ap.add_argument("--nt", type=int, default=system.nt)
    ap.add_argument("--num-users", type=int, default=system.num_users)
    ap.add_argument("--per-ap-power", type=float, default=system.per_ap_power)
    ap.add_argument("--comm-noise-var", type=float, default=system.comm_noise_var)
    ap.add_argument("--gamma-db", type=float, default=setup.gamma_db)
    ap.add_argument("--eps", type=float, default=1e-9)
    ap.add_argument("--radar-noise-var", type=float, default=setup.radar_noise_var)
    ap.add_argument("--target-rcs-var", type=float, default=setup.target_rcs_var)
    ap.add_argument("--sensing-floor-db", type=float, default=setup.sensing_floor_db)
    ap.add_argument("--sensing-floor-weight", type=float, default=setup.sensing_floor_weight)
    ap.add_argument("--communication-weight", type=float, default=setup.communication_weight)

    ap.add_argument("--lr", type=float, default=setup.learning_rate)
    ap.add_argument("--eta-min", type=float, default=setup.eta_min)
    ap.add_argument("--weight-decay", type=float, default=setup.weight_decay)
    ap.add_argument("--epochs", type=int, default=setup.epochs)
    ap.add_argument("--batch-size", type=int, default=setup.batch_size)
    ap.add_argument("--eval-batch-size", type=int, default=setup.eval_batch_size)
    ap.add_argument("--num-workers", type=int, default=system.num_workers)
    ap.add_argument("--grad-clip", type=float, default=setup.max_grad_norm)
    ap.add_argument("--seed", type=int, default=system.seed)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(args.save_dir) / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    train_ds = Stage3Dataset(args.data_dir, "train")
    test_ds = Stage3Dataset(args.data_dir, "test")
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = GraphRZFBeamformer(
        num_aps=args.num_aps,
        nt=args.nt,
        num_users=args.num_users,
        per_ap_power=args.per_ap_power,
        eps=args.eps,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
        eta_min=args.eta_min,
    )

    best = None
    last_epoch = 0

    for epoch in range(1, args.epochs + 1):
        last_epoch = epoch
        train_one_epoch(model, train_loader, optimizer, device, args)
        eval_m = evaluate(model, test_loader, device, args)
        scheduler.step()

        score = eval_m["loss"]
        if best is None:
            is_best = True
        else:
            is_best = score < best["loss"]
        if is_best:
            best = dict(eval_m)
            best["epoch"] = epoch
            save_checkpoint(model, epoch, Path(args.checkpoint))
    save_checkpoint(model, last_epoch, run_dir / "last.pth")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    final_metrics = evaluate(model, test_loader, device, args)
    print_eval(final_metrics)


def _to_complex(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim != 4 or tensor.size(1) != 2:
        raise ValueError(f"Expected [B,2,N,J], got {tuple(tensor.shape)}")
    return torch.complex(tensor[:, 0].contiguous(), tensor[:, 1].contiguous())


def coordinated_sinr(
    channel_ri: torch.Tensor,
    beamformer_ri: torch.Tensor,
    sigma2: float,
    eps: float = 1e-9,
) -> torch.Tensor:

    channel = _to_complex(channel_ri)
    beamformer = _to_complex(beamformer_ri)
    if channel.shape != beamformer.shape:
        raise ValueError(
            f"Channel/beamformer mismatch: {channel.shape} != {beamformer.shape}"
        )
    coupling = torch.einsum("bnj,bnk->bjk", channel.conj(), beamformer)
    power = coupling.abs().square()
    desired = torch.diagonal(power, dim1=-2, dim2=-1)
    interference = power.sum(dim=-1) - desired
    return desired / (interference + float(sigma2) + eps)


def project_per_ap_power(
    beamformer_ri: torch.Tensor,
    num_aps: int,
    nt: int,
    pmax: float,
    eps: float = 1e-9,
) -> torch.Tensor:

    batch, two, antennas, users = beamformer_ri.shape
    if two != 2 or antennas != num_aps * nt:
        raise ValueError(
            f"Expected [B,2,{num_aps * nt},J], got {tuple(beamformer_ri.shape)}"
        )
    by_ap = beamformer_ri.reshape(batch, 2, num_aps, nt, users)
    power = by_ap.square().sum(dim=(1, 3, 4), keepdim=True)
    limit = torch.as_tensor(
        float(pmax), device=beamformer_ri.device, dtype=beamformer_ri.dtype
    )
    scale = torch.sqrt(limit.clamp_min(eps) / power.clamp_min(eps)).clamp_max(1.0)
    return (by_ap * scale).reshape_as(beamformer_ri).contiguous()


def sensing_alignment_score_per_target(
    steering_ri: torch.Tensor,
    beamformer_ri: torch.Tensor,
    num_aps: int,
    nt: int,
    normalize_steering: bool = True,
    eps: float = 1e-9,
) -> torch.Tensor:

    steering = _to_complex(steering_ri)
    beamformer = _to_complex(beamformer_ri)
    batch, antennas, users = steering.shape
    if antennas != num_aps * nt or steering.shape != beamformer.shape:
        raise ValueError(
            f"Expected matching [B,{num_aps * nt},J] tensors, got "
            f"{steering.shape} and {beamformer.shape}"
        )
    steering = steering.reshape(batch, num_aps, nt, users)
    beamformer = beamformer.reshape(batch, num_aps, nt, users)
    if normalize_steering:
        norm = torch.linalg.vector_norm(steering, dim=2, keepdim=True).clamp_min(eps)
        steering = steering / norm
    response = torch.einsum(
        "bmnj,bmnk->bmjk", steering.conj(), beamformer
    )
    return response.abs().square().sum(dim=(1, 3)) / float(num_aps)


def _db(x: torch.Tensor, eps: float) -> torch.Tensor:
    return 10.0 * torch.log10(x.clamp_min(float(eps)))


def stage3_loss(
    Htrue_ri: torch.Tensor,
    A_ri: torch.Tensor,
    F_ri: torch.Tensor,
    args: argparse.Namespace,
) -> Dict[str, torch.Tensor]:
    num_aps = args.num_aps
    nt = args.nt
    eps = args.eps

    F_proj = project_per_ap_power(
        F_ri,
        num_aps=num_aps,
        nt=nt,
        pmax=args.per_ap_power,
        eps=eps,
    )
    actual_sinr = coordinated_sinr(
        Htrue_ri,
        F_proj,
        sigma2=args.comm_noise_var,
        eps=eps,
    )
    actual_sinr_db = _db(actual_sinr, eps)
    sensing = sensing_alignment_score_per_target(
        A_ri,
        F_proj,
        num_aps=num_aps,
        nt=nt,
        eps=eps,
    )
    sensing_snr = (
        sensing
        * float(num_aps * nt)
        * args.target_rcs_var
        / max(args.radar_noise_var, eps)
    )
    sensing_snr_db = _db(sensing_snr, eps)
    floor_gap = F.relu(args.sensing_floor_db - sensing_snr_db.mean(dim=1))
    sensing_floor_loss = floor_gap.square().mean()
    sinr_gap = F.relu(args.gamma_db - actual_sinr_db)
    communication_loss = sinr_gap.square().mean()

    return {
        "loss": (
            args.sensing_floor_weight * sensing_floor_loss
            + args.communication_weight * communication_loss
        ),
        "mean_sensing_snr": sensing_snr.mean().detach(),
        "min_communication_sinr": actual_sinr.min(dim=1).values.mean().detach(),
    }


if __name__ == "__main__":
    main()
