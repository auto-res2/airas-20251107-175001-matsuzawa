import os
import subprocess
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig


@hydra.main(version_base="1.3", config_path="../config", config_name="config")
def main(cfg: DictConfig):
    if cfg.run is None:
        raise ValueError(
            "Usage: python -m src.main run=<run_id> results_dir=<dir> mode=<trial|full>"
        )

    overrides = [
        f"run={cfg.run}",
        f"results_dir={cfg.results_dir}",
        f"mode={cfg.mode}",
    ]

    cmd = [sys.executable, "-u", "-m", "src.train"] + overrides
    print("🚀 Launching:", " ".join(cmd))
    env = os.environ.copy()
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()