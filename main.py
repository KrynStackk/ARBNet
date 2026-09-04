from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from configs.config import PIPELINE_STEPS, SETUP


EngineCall = Tuple[str, List[str]]
EnginePlan = List[EngineCall]


def _train_stage1() -> EnginePlan:
    s = SETUP.stage1
    return [("run.stage1", [
        "--seed", str(SETUP.system.seed),
        "--dataset-root", SETUP.system.dataset_root,
        "--checkpoint", s.checkpoint,
        "--save-dir", s.save_dir,
        "--run-id", s.run_id,
        "--batch-size", str(s.batch_size),
        "--eval-batch-size", str(s.eval_batch_size),
        "--epochs", str(s.epochs),
        "--lr", str(s.learning_rate),
        "--eta-min", str(s.eta_min),
        "--weight-decay", str(s.weight_decay),
        "--grad-clip", str(s.max_grad_norm),
        "--num-workers", str(SETUP.system.num_workers),
    ])]


def _prepare_stage2_data() -> EngineCall:
    system, s1 = SETUP.system, SETUP.stage1
    return "data.stage2", [
        "--dataset-root", system.dataset_root,
        "--stage1-checkpoint", s1.checkpoint,
        "--output-dir", s1.stage2_data_dir,
        "--batch-size", str(SETUP.stage1.batch_size),
        "--num-workers", str(system.num_workers),
        "--nt", str(system.nt),
        "--num-users", str(system.num_users),
        "--splits", "train", "test",
    ]


def _stage2_data_ready() -> bool:
    root = Path(SETUP.stage1.stage2_data_dir)
    required = ("train_hhat.pt", "train_htrue.pt", "test_hhat.pt", "test_htrue.pt")
    return all((root / name).is_file() for name in required)


def _train_stage2() -> EnginePlan:
    plan: EnginePlan = []
    if not _stage2_data_ready():
        plan.append(_prepare_stage2_data())
    plan.append(_stage2_trainer())
    return plan


def _stage2_trainer() -> EngineCall:
    s = SETUP.stage2
    return "run.stage2", [
        "--data-dir", SETUP.stage1.stage2_data_dir,
        "--checkpoint", s.checkpoint,
        "--save-dir", s.save_dir,
        "--run-id", s.run_id,
        "--epochs", str(s.epochs),
        "--batch-size", str(s.batch_size),
        "--eval-batch-size", str(s.eval_batch_size),
        "--lr", str(s.learning_rate),
        "--eta-min", str(s.eta_min),
        "--weight-decay", str(s.weight_decay),
        "--grad-clip", str(s.max_grad_norm),
        "--seed", str(SETUP.system.seed),
        "--num-workers", str(SETUP.system.num_workers),
    ]


def _prepare_stage3_data() -> EngineCall:
    system, s1, s2 = SETUP.system, SETUP.stage1, SETUP.stage2
    return "data.stage3", [
        "--stage2-data-dir", s1.stage2_data_dir,
        "--dataset-root", system.dataset_root,
        "--stage2-checkpoint", s2.checkpoint,
        "--output-dir", s2.stage3_data_dir,
        "--num-aps", str(system.num_aps),
        "--nt", str(system.nt),
        "--num-users", str(system.num_users),
        "--batch-size", str(SETUP.stage1.batch_size),
        "--num-workers", str(system.num_workers),
        "--splits", "train", "test",
    ]


def _stage3_data_ready() -> bool:
    root = Path(SETUP.stage2.stage3_data_dir)
    required = (
        "train_hrec.pt",
        "train_A.pt",
        "train_htrue.pt",
        "test_hrec.pt",
        "test_A.pt",
        "test_htrue.pt",
    )
    return all((root / name).is_file() for name in required)


def _train_stage3() -> EnginePlan:
    plan: EnginePlan = []
    if not _stage3_data_ready():
        if not _stage2_data_ready():
            plan.append(_prepare_stage2_data())
        plan.append(_prepare_stage3_data())
    plan.append(_stage3_trainer())
    return plan


def _stage3_trainer() -> EngineCall:
    system, s = SETUP.system, SETUP.stage3
    return "run.stage3", [
        "--data-dir", SETUP.stage2.stage3_data_dir,
        "--checkpoint", s.checkpoint,
        "--save-dir", s.save_dir,
        "--run-id", s.run_id,
        "--num-aps", str(system.num_aps),
        "--nt", str(system.nt),
        "--num-users", str(system.num_users),
        "--per-ap-power", str(system.per_ap_power),
        "--comm-noise-var", str(system.comm_noise_var),
        "--radar-noise-var", str(s.radar_noise_var),
        "--target-rcs-var", str(s.target_rcs_var),
        "--gamma-db", str(s.gamma_db),
        "--sensing-floor-db", str(s.sensing_floor_db),
        "--sensing-floor-weight", str(s.sensing_floor_weight),
        "--communication-weight", str(s.communication_weight),
        "--lr", str(s.learning_rate),
        "--eta-min", str(s.eta_min),
        "--weight-decay", str(s.weight_decay),
        "--epochs", str(s.epochs),
        "--batch-size", str(s.batch_size),
        "--eval-batch-size", str(s.eval_batch_size),
        "--num-workers", str(system.num_workers),
        "--grad-clip", str(s.max_grad_norm),
        "--seed", str(system.seed),
    ]


def _run_all() -> EnginePlan:

    return [
        *_train_stage1(),
        _prepare_stage2_data(),
        _stage2_trainer(),
        _prepare_stage3_data(),
        _stage3_trainer(),
    ]


BUILDERS: Dict[str, Callable[[], EnginePlan]] = {
    "train-stage1": _train_stage1,
    "train-stage2": _train_stage2,
    "train-stage3": _train_stage3,
    "run-all": _run_all,
}


def _check_inputs() -> int:
    checks = {
        "dataset": Path(SETUP.system.dataset_root),
        "selected Stage-1 checkpoint": Path(SETUP.stage1.checkpoint),
        "selected Stage-2 checkpoint": Path(SETUP.stage2.checkpoint),
    }
    missing = 0
    for label, path in checks.items():
        ok = path.exists()
        print(f"[{'OK' if ok else 'MISSING'}] {label}: {path}")
        missing += int(not ok)
    return missing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="J05 proposed pipeline with one fixed paper setup"
    )
    parser.add_argument(
        "step",
        choices=("show-setup", "check-inputs", *PIPELINE_STEPS),
        help="Sequential pipeline operation; scientific settings are fixed",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.step == "show-setup":
        print(json.dumps(SETUP.as_dict(), indent=2))
        print(f"feedback_bits_per_ap: {SETUP.feedback_bits_per_ap}")
        print(f"feedback_ratio: 1/{1.0 / SETUP.feedback_ratio:.4f}")
        return
    if args.step == "check-inputs":
        raise SystemExit(1 if _check_inputs() else 0)

    plan = BUILDERS[args.step]()
    for module_name, engine_args in plan:
        module = importlib.import_module(module_name)
        sys.argv = [module_name, *engine_args]
        module.main()


if __name__ == "__main__":
    main()
