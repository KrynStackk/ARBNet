from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import scipy.io as sio
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset

from configs.config import SETUP
from models.stage2 import (
    NT,
    NUM_USERS,
    STAGE2_SETUP_ID,
    SparseBeamRefiner,
    ant_ifft_to_ri,
    make_sparse_feedback,
)


class Stage3Dataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
        split: str,
    ) -> None:
        root = Path(data_dir)
        self.hrec = torch.load(root / f"{split}_hrec.pt", map_location="cpu").float()
        self.steering = torch.load(root / f"{split}_A.pt", map_location="cpu").float()
        self.htrue = torch.load(root / f"{split}_htrue.pt", map_location="cpu").float()
        if not (self.hrec.shape == self.steering.shape == self.htrue.shape):
            raise ValueError(
                f"Shape mismatch for {split}: Hrec={tuple(self.hrec.shape)}, "
                f"A={tuple(self.steering.shape)}, Htrue={tuple(self.htrue.shape)}"
            )
        expected = (
            2,
            SETUP.system.num_aps * SETUP.system.nt,
            SETUP.system.num_users,
        )
        if tuple(self.hrec.shape[1:]) != expected:
            raise ValueError(
                f"Stage-3 data must be [N,{expected[0]},{expected[1]},{expected[2]}], "
                f"got {tuple(self.hrec.shape)}"
            )
    def __len__(self) -> int:
        return int(self.hrec.shape[0])

    def __getitem__(self, index: int):
        return self.hrec[index], self.steering[index], self.htrue[index]




GroupKey = Tuple[str, int, float, int]


def load_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def load_stage2_model(args, device: torch.device) -> SparseBeamRefiner:
    model = SparseBeamRefiner().to(device)
    ckpt = torch.load(args.stage2_checkpoint, map_location="cpu")
    checkpoint_setup = ckpt.get("setup_id") if isinstance(ckpt, dict) else None
    if checkpoint_setup != STAGE2_SETUP_ID:
        raise ValueError(
            "The Stage-2 checkpoint does not belong to the final fixed setup. "
            f"Expected setup_id={STAGE2_SETUP_ID!r}, got {checkpoint_setup!r}. "
            "Retrain it with `python main.py train-stage2`."
        )
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()

    return model


@torch.no_grad()
def reconstruct_hrec_apview(
    args,
    model: SparseBeamRefiner,
    hhat: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:

    ds = TensorDataset(hhat.float())
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    out = []
    for (hhat_b,) in loader:
        hhat_b = hhat_b.to(device, non_blocking=True)

        sparse_ri, known_mask = make_sparse_feedback(hhat_b)
        pred_beam_ri = model(sparse_ri, known_mask)
        pred_beam_c = torch.complex(
            pred_beam_ri[:, 0].contiguous(),
            pred_beam_ri[:, 1].contiguous(),
        )
        hrec_b = ant_ifft_to_ri(pred_beam_c)

        out.append(hrec_b.detach().cpu().float())

    return torch.cat(out, dim=0).contiguous()


def build_dataA_path_map(dataset_root: Path, split: str) -> Dict[str, Path]:

    files = sorted((dataset_root / split).rglob("dataA_*.mat"))
    if not files:
        raise FileNotFoundError(f"No dataA_*.mat files found under {dataset_root / split}")

    path_map: Dict[str, Path] = {}
    for p in files:
        if p.name in path_map:
            raise RuntimeError(f"Duplicate dataA filename found: {p.name}")
        path_map[p.name] = p

    return path_map


def dataP_name_to_dataA_name(dataP_name: str) -> str:
    if not dataP_name.startswith("dataP_"):
        raise ValueError(f"Expected dataP filename, got: {dataP_name}")
    return "dataA_" + dataP_name[len("dataP_"):]


def load_dataA_tensor(path: Path, nt: int, num_users: int) -> torch.Tensor:

    mat = sio.loadmat(path)
    if "dataA" not in mat:
        keys = [k for k in mat.keys() if not k.startswith("__")]
        raise KeyError(f"'dataA' key not found in {path}. Available keys: {keys}")

    arr = mat["dataA"]
    if tuple(arr.shape) != (nt, num_users, 2):
        raise ValueError(
            f"Unexpected dataA shape in {path}: {arr.shape}, "
            f"expected {(nt, num_users, 2)}"
        )

    return torch.from_numpy(arr).permute(2, 0, 1).contiguous().float()


def load_all_dataA_apview(
    args,
    split: str,
    meta_list: List[dict],
) -> torch.Tensor:

    dataset_root = Path(args.dataset_root)
    path_map = build_dataA_path_map(dataset_root, split)

    out = []
    for meta in meta_list:
        dataA_name = dataP_name_to_dataA_name(meta["name"])
        path = path_map.get(dataA_name, None)
        if path is None:
            raise FileNotFoundError(
                f"Cannot find dataA file for {meta['name']} -> {dataA_name}"
            )
        out.append(load_dataA_tensor(path, nt=args.nt, num_users=args.num_users))

    return torch.stack(out, dim=0).contiguous()


def make_group_key(meta: dict) -> GroupKey:
    return (
        str(meta["split"]),
        int(meta["topoID"]),
        float(meta["pilot_power_dBm"]),
        int(meta["idxID"]),
    )


def group_apview_to_cpuview(
    args,
    split: str,
    hrec_ap: torch.Tensor,
    A_ap: torch.Tensor,
    htrue_ap: torch.Tensor,
    meta_list: List[dict],
):

    if not (hrec_ap.shape == A_ap.shape == htrue_ap.shape):
        raise ValueError(
            f"AP-view shape mismatch: "
            f"hrec={tuple(hrec_ap.shape)}, A={tuple(A_ap.shape)}, htrue={tuple(htrue_ap.shape)}"
        )

    if len(meta_list) != hrec_ap.shape[0]:
        raise ValueError(
            f"Metadata length mismatch for split={split}: "
            f"len(meta)={len(meta_list)}, N={hrec_ap.shape[0]}"
        )

    groups: Dict[GroupKey, List[int]] = defaultdict(list)
    for i, meta in enumerate(meta_list):
        groups[make_group_key(meta)].append(i)

    hrec_cpu_list = []
    A_cpu_list = []
    htrue_cpu_list = []

    sorted_keys = sorted(groups.keys(), key=lambda x: (x[1], x[2], x[3]))

    expected_apids = list(range(1, args.num_aps + 1))

    for key in sorted_keys:
        indices = groups[key]
        indices = sorted(indices, key=lambda i: int(meta_list[i]["apID"]))

        apids = [int(meta_list[i]["apID"]) for i in indices]
        if apids != expected_apids:
            raise RuntimeError(
                f"Incomplete or unordered AP group for key={key}: "
                f"apIDs={apids}, expected={expected_apids}"
            )

        hrec_cpu = torch.cat([hrec_ap[i] for i in indices], dim=1)
        A_cpu = torch.cat([A_ap[i] for i in indices], dim=1)

        if tuple(hrec_cpu.shape) != (2, args.num_aps * args.nt, args.num_users):
            raise RuntimeError(
                f"Bad Hrec CPU shape for key={key}: {tuple(hrec_cpu.shape)}"
            )

        hrec_cpu_list.append(hrec_cpu)
        A_cpu_list.append(A_cpu)

        htrue_cpu_list.append(torch.cat([htrue_ap[i] for i in indices], dim=1))

    hrec_cpu = torch.stack(hrec_cpu_list, dim=0).contiguous()
    A_cpu = torch.stack(A_cpu_list, dim=0).contiguous()
    htrue_cpu = torch.stack(htrue_cpu_list, dim=0).contiguous()

    return hrec_cpu, A_cpu, htrue_cpu


def process_split(args, model: SparseBeamRefiner, split: str, device: torch.device) -> None:
    stage2_data_dir = Path(args.stage2_data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hhat = torch.load(stage2_data_dir / f"{split}_hhat.pt", map_location="cpu").float()
    htrue = torch.load(stage2_data_dir / f"{split}_htrue.pt", map_location="cpu").float()
    meta_list = load_json(stage2_data_dir / f"{split}_meta.json")

    if tuple(hhat.shape[1:]) != (2, args.nt, args.num_users):
        raise ValueError(
            f"Unexpected hhat shape for split={split}: {tuple(hhat.shape)}, "
            f"expected [N, 2, {args.nt}, {args.num_users}]"
        )

    hrec_ap = reconstruct_hrec_apview(args, model, hhat, device)
    A_ap = load_all_dataA_apview(args, split, meta_list)

    hrec_cpu, A_cpu, htrue_cpu = group_apview_to_cpuview(
        args=args,
        split=split,
        hrec_ap=hrec_ap,
        A_ap=A_ap,
        htrue_ap=htrue,
        meta_list=meta_list,
    )

    torch.save(hrec_cpu, out_dir / f"{split}_hrec.pt")
    torch.save(A_cpu, out_dir / f"{split}_A.pt")
    torch.save(htrue_cpu, out_dir / f"{split}_htrue.pt")


def parse_args():
    p = argparse.ArgumentParser(description="Prepare Stage-3 CPU-view ISAC tensors.")
    p.add_argument("--stage2-data-dir", type=str, default=SETUP.stage1.stage2_data_dir)
    p.add_argument("--dataset-root", type=str, default=SETUP.system.dataset_root)
    p.add_argument("--stage2-checkpoint", type=str, default=SETUP.stage2.checkpoint)
    p.add_argument("--output-dir", type=str, default=SETUP.stage2.stage3_data_dir)

    p.add_argument("--num-aps", type=int, default=SETUP.system.num_aps)
    p.add_argument("--nt", type=int, default=SETUP.system.nt)
    p.add_argument("--num-users", type=int, default=SETUP.system.num_users)

    p.add_argument("--batch-size", type=int, default=SETUP.stage1.batch_size)
    p.add_argument("--num-workers", type=int, default=SETUP.system.num_workers)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--splits", type=str, nargs="+", default=["train", "test"])

    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.nt != NT or args.num_users != NUM_USERS:
        raise ValueError(
            f"Stage 2 is fixed to Nt={NT}, J={NUM_USERS}; "
            f"got Nt={args.nt}, J={args.num_users}"
        )
    device = torch.device(args.device)

    model = load_stage2_model(args, device)

    for split in args.splits:
        process_split(args, model, split, device)

if __name__ == "__main__":
    main()
