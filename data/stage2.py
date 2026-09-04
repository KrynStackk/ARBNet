from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset

from configs.config import SETUP
from data.stage1 import create_channel_dataloader


class Stage2Dataset(Dataset):
    def __init__(self, data_dir: str | Path, split: str) -> None:
        root = Path(data_dir)
        self.hhat = torch.load(root / f"{split}_hhat.pt", map_location="cpu").float()
        self.htrue = torch.load(root / f"{split}_htrue.pt", map_location="cpu").float()
        if self.hhat.shape != self.htrue.shape:
            raise ValueError(
                f"Shape mismatch: hhat={self.hhat.shape}, htrue={self.htrue.shape}"
            )
        expected = (2, SETUP.system.nt, SETUP.system.num_users)
        if tuple(self.hhat.shape[1:]) != expected:
            raise ValueError(
                f"Stage-2 data must be [N,{expected[0]},{expected[1]},{expected[2]}], "
                f"got {tuple(self.hhat.shape)}"
            )

    def __len__(self) -> int:
        return int(self.hhat.shape[0])

    def __getitem__(self, index: int):
        return self.hhat[index], self.htrue[index]




def batch_meta_to_records(batch_meta: Dict) -> List[dict]:

    if batch_meta is None:
        return []

    if "name" in batch_meta:
        batch_size = len(batch_meta["name"])
    else:
        first_value = next(iter(batch_meta.values()))
        batch_size = int(first_value.size(0)) if isinstance(first_value, torch.Tensor) else len(first_value)

    def get_value(key: str, i: int):
        value = batch_meta[key]

        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()[i].item()
        else:
            value = value[i]

        if key in {"topoID", "apID", "idxID"}:
            return int(value)
        if key == "pilot_power_dBm":
            return float(value)
        return str(value)

    records = []
    for i in range(batch_size):
        records.append(
            {
                "split": get_value("split", i),
                "topoID": get_value("topoID", i),
                "apID": get_value("apID", i),
                "pilot_power_dBm": get_value("pilot_power_dBm", i),
                "idxID": get_value("idxID", i),
                "name": get_value("name", i),
            }
        )

    return records


def load_stage1_model(args, device: torch.device) -> HybridFFTSSMEstimator:
    from models.stage1 import HybridFFTSSMEstimator, STAGE1_SETUP_ID

    model = HybridFFTSSMEstimator().to(device)

    ckpt = torch.load(args.stage1_checkpoint, map_location="cpu")
    if not isinstance(ckpt, dict) or ckpt.get("setup_id") != STAGE1_SETUP_ID:
        raise ValueError(
            "The checkpoint is not from the final Stage-1 setup "
            f"({STAGE1_SETUP_ID}). Retrain it with `python main.py train-stage1`."
        )
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


@torch.no_grad()
def prepare_split(args, model: HybridFFTSSMEstimator, split: str, device: torch.device) -> None:
    loader = create_channel_dataloader(
        root_dir=args.dataset_root,
        split=split,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    hhat_list, htrue_list = [], []
    meta_list = []

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        G = batch["G"].to(device, non_blocking=True)

        pred = model(x, G)
        hhat_list.append(pred.detach().cpu())
        htrue_list.append(y.detach().cpu())
        if "meta" in batch:
            meta_list.extend(batch_meta_to_records(batch["meta"]))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hhat = torch.cat(hhat_list, dim=0).contiguous()
    htrue = torch.cat(htrue_list, dim=0).contiguous()
    torch.save(hhat, out_dir / f"{split}_hhat.pt")
    torch.save(htrue, out_dir / f"{split}_htrue.pt")

    if meta_list:
        if len(meta_list) != int(hhat.size(0)):
            raise RuntimeError(
                f"Metadata length mismatch for split={split}: "
                f"len(meta_list)={len(meta_list)} but hhat.size(0)={int(hhat.size(0))}"
            )

        with open(out_dir / f"{split}_meta.json", "w") as f:
            json.dump(meta_list, f, indent=2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare Stage-2 tensors from final Stage-1.")
    p.add_argument("--dataset-root", type=str, default=SETUP.system.dataset_root)
    p.add_argument("--stage1-checkpoint", type=str, default=SETUP.stage1.checkpoint)
    p.add_argument("--output-dir", type=str, default=SETUP.stage1.stage2_data_dir)
    p.add_argument("--batch-size", type=int, default=SETUP.stage1.batch_size)
    p.add_argument("--num-workers", type=int, default=SETUP.system.num_workers)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--nt", type=int, default=SETUP.system.nt)
    p.add_argument("--num-users", type=int, default=SETUP.system.num_users)
    p.add_argument("--splits", type=str, nargs="+", default=["train", "test"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    model = load_stage1_model(args, device)
    for split in args.splits:
        prepare_split(args, model, split, device)

if __name__ == "__main__":
    main()
