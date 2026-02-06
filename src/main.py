import subprocess
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf


def apply_mode(cfg: DictConfig) -> DictConfig:
    OmegaConf.set_struct(cfg, False)
    if cfg.mode == "trial":
        cfg.wandb.mode = "disabled"
        cfg.optuna.n_trials = 0
        cfg.training.epochs = 1
        cfg.training.max_batches = 2
    elif cfg.mode == "full":
        cfg.wandb.mode = "online"
    else:
        raise ValueError(f"Unsupported mode: {cfg.mode}")
    return cfg


@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    cfg = apply_mode(cfg)
    if not hasattr(cfg, "run") or cfg.run is None:
        raise ValueError("run parameter must be provided (run=<run_id>).")
    if not hasattr(cfg, "run_id") or cfg.run_id is None or str(cfg.run_id).strip() == "???":
        raise ValueError("run_id is missing from the loaded run configuration.")

    repo_root = Path(__file__).resolve().parents[1]

    cmd = [
        sys.executable,
        "-m",
        "src.train",
        f"run={cfg.run}",
        f"results_dir={cfg.results_dir}",
        f"mode={cfg.mode}",
    ]
    print("Launching:", " ".join(cmd))
    subprocess.run(cmd, cwd=repo_root, check=True)


if __name__ == "__main__":
    main()
