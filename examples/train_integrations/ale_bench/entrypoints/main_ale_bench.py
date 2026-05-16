from __future__ import annotations

import sys

import ray
from dotenv import load_dotenv
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.entrypoints.main_base import BasePPOExp, validate_cfg
from skyrl.train.utils import initialize_ray
from skyrl_gym.envs import register


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: SkyRLTrainConfig):
    load_dotenv()
    register(
        id="ale_bench",
        entry_point="examples.train_integrations.ale_bench.env:AleBenchEnv",
    )
    exp = BasePPOExp(cfg)
    exp.run()


def main() -> None:
    load_dotenv()
    cfg = SkyRLTrainConfig.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
