import os
import subprocess
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base="1.3", config_path="../config", config_name="config")
def main(cfg: DictConfig):
    if cfg.run is None:
        raise ValueError(
            "Usage: python -m src.main run=<run_id> results_dir=<dir> mode=<trial|full>"
        )

    # Load run-specific configuration to validate
    from pathlib import Path
    import yaml
    run_config_path = Path(__file__).parent.parent / "config" / "runs" / f"{cfg.run}.yaml"
    if run_config_path.exists():
        with open(run_config_path, 'r') as f:
            run_cfg = OmegaConf.create(yaml.safe_load(f))
        # Disable struct mode to allow adding new keys during merge
        OmegaConf.set_struct(cfg, False)
        cfg = OmegaConf.merge(cfg, run_cfg)
        OmegaConf.set_struct(cfg, True)

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