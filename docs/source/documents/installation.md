# Installation

```bash
pip install mosaic_multigrid
```

Requires Python 3.9 or later. Dependencies (`gymnasium`, `numpy`, `numba`,
`pygame`, `aenum`) install automatically.

## Optional extras

```bash
pip install "mosaic_multigrid[pettingzoo]"   # PettingZoo Parallel + AEC adapters
pip install "mosaic_multigrid[rllib]"        # RLlib support
pip install "mosaic_multigrid[dev]"          # pytest + pytest-timeout for tests
```

## From source

```bash
git clone https://github.com/Abdulhamid97Mousa/mosaic_multigrid.git
cd mosaic_multigrid
pip install -e ".[dev]"
```

## Verify

```python
import gymnasium as gym
import mosaic_multigrid.envs

count = sum(1 for e in gym.registry if e.startswith("MosaicMultiGrid-"))
print(count)  # 81
```
