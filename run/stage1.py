from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Dict

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from configs.config import SETUP
from data.stage1 import create_channel_dataloader
from models.stage1 import (
    HybridFFTSSMEstimator,
    STAGE1_SETUP_ID,
)

def move_batch_G(batch, device: torch.device):
    G = batch.get("G", None)
    if G is None or not torch.is_tensor(G) or G.numel() == 0:
        return None
    return G.to(device, non_blocking=True)


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


def train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer,
    device: torch.device,
    grad_clip: float,
) -> None:
    model.train()

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        G = move_batch_G(batch, device)

        pred = model(x, G=G)
        loss = stage1_loss(pred, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

@torch.no_grad()
def evaluate(model: torch.nn.Module, loader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_nmse = 0.0
    total_samples = 0
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        G = move_batch_G(batch, device)
        pred = model(x, G=G)
        nmse = nmse_per_sample(pred, y)
        total_nmse += nmse.sum().item()
        total_samples += x.size(0)
    mean_nmse = total_nmse / max(total_samples, 1)
    return {
        "nmse": mean_nmse,
        "nmse_db": nmse_to_db(mean_nmse),
    }


def parse_args() -> argparse.Namespace:
    setup = SETUP.stage1
    parser = argparse.ArgumentParser(description="Train the proposed Hybrid FFT-SSM channel estimator")
    parser.add_argument("--epochs", type=int, default=setup.epochs)
    parser.add_argument("--batch-size", type=int, default=setup.batch_size)
    parser.add_argument("--eval-batch-size", type=int, default=setup.eval_batch_size)
    parser.add_argument("--dataset-root", type=str, default=SETUP.system.dataset_root)
    parser.add_argument("--checkpoint", type=str, default=setup.checkpoint)
    parser.add_argument("--save-dir", type=str, default=setup.save_dir)
    parser.add_argument("--run-id", type=str, default=setup.run_id)
    parser.add_argument("--lr", type=float, default=setup.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=setup.weight_decay)
    parser.add_argument("--eta-min", type=float, default=setup.eta_min)
    parser.add_argument("--num-workers", type=int, default=SETUP.system.num_workers)
    parser.add_argument("--grad-clip", type=float, default=setup.max_grad_norm)
    parser.add_argument("--seed", type=int, default=SETUP.system.seed)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir) / args.run_id
    save_dir.mkdir(parents=True, exist_ok=True)

    train_loader = create_channel_dataloader(
        root_dir=args.dataset_root,
        split="train",
        batch_size=args.eval_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    test_loader = create_channel_dataloader(
        root_dir=args.dataset_root,
        split="test",
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    first_batch = next(iter(train_loader))

    x_shape = tuple(first_batch["x"].shape)
    if (
        len(x_shape) != 4
        or x_shape[2] != SETUP.system.nt
        or x_shape[3] != SETUP.system.num_users
    ):
        raise ValueError(
            "Stage1 dataset/config shape mismatch: "
            f"x={x_shape}, configured Nt={SETUP.system.nt}, "
            f"J={SETUP.system.num_users}"
        )

    model = HybridFFTSSMEstimator().to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs),
        eta_min=args.eta_min,
    )

    best_nmse = float("inf")
    last_epoch = 0

    for epoch in range(1, args.epochs + 1):
        last_epoch = epoch
        train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            grad_clip=args.grad_clip,
        )

        test_log = evaluate(model, test_loader, device)
        is_best = test_log["nmse"] < best_nmse
        if is_best:
            best_nmse = float(test_log["nmse"])

        if is_best:
            save_checkpoint(
                model,
                epoch,
                Path(args.checkpoint),
                STAGE1_SETUP_ID,
            )

        scheduler.step()

    save_checkpoint(
        model,
        last_epoch,
        save_dir / "last.pth",
        STAGE1_SETUP_ID,
    )

    best_path = Path(args.checkpoint)
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    final_log = evaluate(model, test_loader, device)

    print("\n[Stage 1 | Final Evaluation]")
    print(f"NMSE : {final_log['nmse_db']:+.2f} dB")


def nmse_per_sample(
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
    return numerator / denominator


def stage1_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

    return nmse_per_sample(prediction, target).mean()


def nmse_to_db(nmse: float, eps: float = 1e-12) -> float:
    return 10.0 * math.log10(float(nmse) + eps)




if __name__ == "__main__":
    main()
