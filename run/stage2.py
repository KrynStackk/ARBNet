from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader

from configs.config import SETUP
from data.stage2 import Stage2Dataset
from models.stage2 import (
    STAGE2_SETUP_ID,
    SparseBeamRefiner,
    ant_ifft_to_ri,
    make_sparse_feedback,
)


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


def sample_nmse_vec(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    numerator = (pred - target).pow(2).flatten(1).sum(dim=1)
    denominator = target.pow(2).flatten(1).sum(dim=1).clamp_min(eps)
    return numerator / denominator


def db_from_scalar(value: float) -> float:
    return 10.0 * math.log10(max(float(value), 1e-12))


def _beam_to_antenna(beam_ri: torch.Tensor) -> torch.Tensor:
    beam = torch.complex(
        beam_ri[:, 0].contiguous(), beam_ri[:, 1].contiguous()
    )
    return ant_ifft_to_ri(beam)


@torch.no_grad()
def evaluate(
    model: SparseBeamRefiner,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_nmse = 0.0
    n_samples = 0

    for hhat, htrue in loader:
        hhat = hhat.to(device, non_blocking=True)
        htrue = htrue.to(device, non_blocking=True)
        sparse_ri, known_mask = make_sparse_feedback(hhat)
        pred_beam_ri = model(sparse_ri, known_mask)
        hrec = _beam_to_antenna(pred_beam_ri)

        batch_size = hhat.shape[0]
        total_nmse += sample_nmse_vec(hrec, htrue).sum().item()
        n_samples += batch_size

    denominator = max(n_samples, 1)
    nmse = total_nmse / denominator
    return {
        "nmse": nmse,
        "nmse_db": db_from_scalar(nmse),
    }


def train_one_epoch(
    model: SparseBeamRefiner,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
) -> None:
    model.train()

    for hhat, htrue in loader:
        hhat = hhat.to(device, non_blocking=True)
        htrue = htrue.to(device, non_blocking=True)
        sparse_ri, known_mask = make_sparse_feedback(hhat)
        pred_beam_ri = model(sparse_ri, known_mask)
        hrec = _beam_to_antenna(pred_beam_ri)
        loss = stage2_loss(hrec, htrue)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()



def parse_args() -> argparse.Namespace:
    setup = SETUP.stage2
    parser = argparse.ArgumentParser(
        description="Train the fixed final Stage-2 setup"
    )
    parser.add_argument("--data-dir", type=str, default=SETUP.stage1.stage2_data_dir)
    parser.add_argument("--checkpoint", type=str, default=setup.checkpoint)
    parser.add_argument(
        "--save-dir",
        type=str,
        default=setup.save_dir,
    )
    parser.add_argument("--run-id", type=str, default=setup.run_id)
    parser.add_argument("--epochs", type=int, default=setup.epochs)
    parser.add_argument("--batch-size", type=int, default=setup.batch_size)
    parser.add_argument("--eval-batch-size", type=int, default=setup.eval_batch_size)
    parser.add_argument("--lr", type=float, default=setup.learning_rate)
    parser.add_argument("--eta-min", type=float, default=setup.eta_min)
    parser.add_argument("--weight-decay", type=float, default=setup.weight_decay)
    parser.add_argument("--seed", type=int, default=SETUP.system.seed)
    parser.add_argument("--num-workers", type=int, default=SETUP.system.num_workers)
    parser.add_argument("--grad-clip", type=float, default=setup.max_grad_norm)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

    save_dir = Path(args.save_dir) / args.run_id
    save_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = Stage2Dataset(args.data_dir, "train")
    test_ds = Stage2Dataset(args.data_dir, "test")
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    model = SparseBeamRefiner().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.eta_min
    )
    best = None
    last_epoch = 0

    for epoch in range(1, args.epochs + 1):
        last_epoch = epoch
        train_one_epoch(
            model, train_loader, optimizer, device, args.grad_clip
        )
        scheduler.step()
        metrics = evaluate(model, test_loader, device)
        is_best = best is None or metrics["nmse"] < best["nmse"]
        if is_best:
            best = {**metrics, "epoch": epoch}
            save_checkpoint(
                model,
                epoch,
                Path(args.checkpoint),
                STAGE2_SETUP_ID,
            )
    save_checkpoint(
        model,
        last_epoch,
        save_dir / "last.pth",
        STAGE2_SETUP_ID,
    )
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    final_metrics = evaluate(model, test_loader, device)

    print("\n[Stage 2 | Final Evaluation]")
    print(f"Reconstruction NMSE : {final_metrics['nmse_db']:+.2f} dB")


def stage2_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:

    if prediction.ndim != 4 or prediction.size(1) != 2:
        raise ValueError(
            f"prediction must be [B,2,Nt,J], got {tuple(prediction.shape)}"
        )
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction shape {tuple(prediction.shape)} != "
            f"target shape {tuple(target.shape)}"
        )
    numerator = (prediction - target).square().flatten(1).sum(dim=1)
    denominator = target.square().flatten(1).sum(dim=1).clamp_min(eps)
    return (numerator / denominator).mean()




if __name__ == "__main__":
    main()
