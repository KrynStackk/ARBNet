import re
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset, DataLoader

from configs.config import SETUP


class Stage1Dataset(Dataset):


    def __init__(self, root_dir, split="train"):
        self.root_dir = Path(root_dir)
        self.split = split

        self.channel_dir = self.root_dir / split / "channel"
        self.data_dir = self.channel_dir / "data"
        self.label_dir = self.channel_dir / "label"
        self.pilot_path = self.channel_dir / "pilot" / "pilot_info.mat"

        self.data_files = sorted(self.data_dir.glob("dataP_*.mat"))
        if len(self.data_files) == 0:
            raise FileNotFoundError(f"No data files found in {self.data_dir}")

        self.G_realimag = self._load_global_G()

    def _load_global_G(self) -> torch.Tensor:

        if not self.pilot_path.exists():
            return torch.empty(0, dtype=torch.float32)

        pilot_mat = sio.loadmat(self.pilot_path)

        if "G_mix_realimag" in pilot_mat:
            G = pilot_mat["G_mix_realimag"]
            return self._matlab_ri_matrix_to_torch(G, self.pilot_path, "G_mix_realimag")

        if "P_right" in pilot_mat and "P_matrix" in pilot_mat:
            P_right = np.asarray(pilot_mat["P_right"])
            P_matrix = np.asarray(pilot_mat["P_matrix"])
            G_complex = P_right @ P_matrix
            G = np.stack([np.real(G_complex), np.imag(G_complex)], axis=2)
            return self._matlab_ri_matrix_to_torch(G, self.pilot_path, "P_right*P_matrix")

        if "G_realimag" in pilot_mat:
            G = pilot_mat["G_realimag"]
            return self._matlab_ri_matrix_to_torch(G, self.pilot_path, "G_realimag")

        return torch.empty(0, dtype=torch.float32)

    @staticmethod
    def _matlab_ri_matrix_to_torch(arr, file_path: Path, key: str) -> torch.Tensor:

        if arr.ndim != 3 or arr.shape[2] != 2:
            raise ValueError(
                f"{file_path} variable '{key}' must have shape [J, J, 2], got {arr.shape}"
            )
        return torch.tensor(arr, dtype=torch.float32).permute(2, 0, 1).contiguous()

    @staticmethod
    def _parse_db_tag(tag: str) -> float:

        tag = str(tag)
        if tag.startswith("m"):
            return -float(tag[1:])
        if tag.startswith("p"):
            return float(tag[1:])
        return float(tag)

    @staticmethod
    def _parse_ap_view_filename(file_name: str) -> dict:

        power_pattern = (
            r"^dataP_topo_(?P<topoID>\d+)"
            r"_ap_(?P<apID>\d+)"
            r"_ppow_dBm_(?P<pilot_power_dBm>[-+]?\d+(?:\.\d+)?)"
            r"_idx_(?P<idxID>\d+)\.mat$"
        )
        parameterized_power_pattern = (
            r"^dataP_(?P<paramTag>.+?)"
            r"_topo_(?P<topoID>\d+)"
            r"_ap_(?P<apID>\d+)"
            r"_ppow_dBm_(?P<power_tag>[-+]?\d+(?:\.\d+)?|[mp]\d+)"
            r"_idx_(?P<idxID>\d+)\.mat$"
        )
        m = re.match(power_pattern, file_name)
        if m is not None:
            return {
                "topoID": int(m.group("topoID")),
                "apID": int(m.group("apID")),
                "pilot_power_dBm": float(m.group("pilot_power_dBm")),
                "idxID": int(m.group("idxID")),
            }

        m = re.match(parameterized_power_pattern, file_name)
        if m is not None:
            return {
                "topoID": int(m.group("topoID")),
                "apID": int(m.group("apID")),
                "pilot_power_dBm": Stage1Dataset._parse_db_tag(m.group("power_tag")),
                "idxID": int(m.group("idxID")),
            }

        raise ValueError(
            f"Cannot parse AP-view metadata from filename: {file_name}. "
            "Expected a dataP filename containing _ppow_dBm_."
        )

    def __len__(self):
        return len(self.data_files)

    @staticmethod
    def _load_complex_tensor(mat_dict, key: str, file_path: Path) -> torch.Tensor:

        if key not in mat_dict:
            raise KeyError(f"{file_path} does not contain variable '{key}'")

        arr = mat_dict[key]

        if arr.ndim != 3:
            raise ValueError(
                f"{file_path} variable '{key}' must be 3D [Nt, J, 2], got shape {arr.shape}."
            )

        if arr.shape[2] != 2:
            raise ValueError(
                f"{file_path} variable '{key}' last dimension must be 2 for real/imag, got shape {arr.shape}."
            )

        return torch.tensor(arr, dtype=torch.float32).permute(2, 0, 1).contiguous()

    def __getitem__(self, idx):
        data_path = self.data_files[idx]

        label_name = data_path.name.replace("dataP_", "labelH_")
        label_path = self.label_dir / label_name

        if not label_path.exists():
            raise FileNotFoundError(f"Missing label file: {label_path}")

        data_mat = sio.loadmat(data_path)
        label_mat = sio.loadmat(label_path)

        x = self._load_complex_tensor(data_mat, "dataP", data_path)
        y = self._load_complex_tensor(label_mat, "labelH", label_path)

        if x.shape != y.shape:
            raise ValueError(
                f"Input/label shape mismatch for {data_path.name}: x={tuple(x.shape)}, y={tuple(y.shape)}"
            )

        file_meta = self._parse_ap_view_filename(data_path.name)

        meta = {
            "split": self.split,
            "topoID": file_meta["topoID"],
            "apID": file_meta["apID"],
            "pilot_power_dBm": file_meta["pilot_power_dBm"],
            "idxID": file_meta["idxID"],
            "name": data_path.name,
        }

        return {
            "x": x,
            "y": y,
            "G": self.G_realimag,
            "meta": meta,
        }


def create_channel_dataloader(
    root_dir,
    split="train",
    batch_size=SETUP.stage1.batch_size,
    shuffle=True,
    num_workers=SETUP.system.num_workers,
):
    dataset = Stage1Dataset(root_dir=root_dir, split=split)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    return loader
