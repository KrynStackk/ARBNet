from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Tuple, Type, TypeVar

import yaml


@dataclass(frozen=True)
class SystemSetup:
    dataset_root: str = "dataset"
    num_aps: int = 10
    nt: int = 16
    num_users: int = 20
    per_ap_power: float = 1.0
    comm_noise_var: float = 1.0
    seed: int = 1234
    num_workers: int = 2


@dataclass(frozen=True)
class Stage1Setup:
    checkpoint: str
    save_dir: str
    run_id: str
    stage2_data_dir: str
    in_chans: int
    out_chans: int
    pre_ant_depth: int
    seq_beam_depth: int
    post_ant_depth: int
    cnn_width: int
    cnn_kernel: int
    embed_dim: int
    hidden_dim: int
    cond_dim: int
    drop_path: float
    ssm_d_state: int
    vss_scan_number: int
    vss_ssm_ratio: float
    vss_ssm_rank_ratio: float
    vss_conv: int
    batch_size: int
    eval_batch_size: int
    epochs: int
    learning_rate: float
    eta_min: float
    weight_decay: float
    max_grad_norm: float


@dataclass(frozen=True)
class Stage2Setup:
    checkpoint: str
    save_dir: str
    run_id: str
    stage3_data_dir: str
    top_m: int
    quant_bits: int
    batch_size: int
    eval_batch_size: int
    epochs: int
    learning_rate: float
    eta_min: float
    weight_decay: float
    max_grad_norm: float


@dataclass(frozen=True)
class Stage3Setup:
    checkpoint: str
    save_dir: str
    run_id: str
    hidden_dim: int
    num_layers: int
    mask_init: float
    alpha: float
    p_min: float
    p_max: float
    radar_noise_var: float
    target_rcs_var: float
    sensing_floor_db: float
    sensing_floor_weight: float
    gamma_db: float
    communication_weight: float
    batch_size: int
    eval_batch_size: int
    epochs: int
    learning_rate: float
    eta_min: float
    weight_decay: float
    max_grad_norm: float


@dataclass(frozen=True)
class PipelineSetup:
    system: SystemSetup
    stage1: Stage1Setup
    stage2: Stage2Setup
    stage3: Stage3Setup

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def feedback_bits_per_ap(self) -> int:
        raw_coefficients = self.system.nt * self.system.num_users
        index_bits = (raw_coefficients - 1).bit_length()
        return 32 + self.stage2.top_m * (index_bits + 2 * self.stage2.quant_bits)

    @property
    def feedback_ratio(self) -> float:
        raw_bits = 2 * self.system.nt * self.system.num_users * 32
        return self.feedback_bits_per_ap / raw_bits


T = TypeVar("T")
CONFIG_DIR = Path(__file__).resolve().parent


def _load_stage(name: str, schema: Type[T]) -> T:
    path = CONFIG_DIR / f"{name}.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get(name), dict):
        raise ValueError(f"Missing YAML mapping {name!r}: {path}")
    section = raw[name]
    expected = {field.name for field in fields(schema)}
    actual = set(section)
    if actual != expected:
        raise ValueError(
            f"Invalid {name} config: "
            f"missing={sorted(expected - actual)}, unknown={sorted(actual - expected)}"
        )
    return schema(**section)


def load_setup() -> PipelineSetup:
    return PipelineSetup(
        system=SystemSetup(),
        stage1=_load_stage("stage1", Stage1Setup),
        stage2=_load_stage("stage2", Stage2Setup),
        stage3=_load_stage("stage3", Stage3Setup),
    )


SETUP = load_setup()
PIPELINE_STEPS: Tuple[str, ...] = (
    "train-stage1",
    "train-stage2",
    "train-stage3",
    "run-all",
)
