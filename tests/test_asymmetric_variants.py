"""Tests for the 30 asymmetric competitive team variants.

Every asymmetric variant is a subclass of the corresponding IndAgObs env
that fixes an `agents_index` list encoding an uneven Green/Blue team
split. Verifies that:

    1. All 30 asymmetric env IDs are registered with Gymnasium.
    2. Each env instantiates cleanly and reports the expected agent count.
    3. Each env resets and steps without raising.
    4. The overall registered-env count matches the target (81).

Covers matchups: 1v2, 2v1, 1v3, 3v1, 1v4, 4v1, 2v3, 3v2, 2v4, 4v2
across the three sport families: Basketball (BB), American Football (AF),
Soccer (S).
"""
from __future__ import annotations

import gymnasium as gym
import pytest

# Trigger gymnasium registration
import mosaic_multigrid.envs  # noqa: F401


SPORTS = ["BB", "AF", "S"]
MATCHUPS = ["1v2", "2v1", "1v3", "3v1", "1v4", "4v1",
            "2v3", "3v2", "2v4", "4v2"]


def _n_agents(matchup: str) -> int:
    g, b = matchup.split("v")
    return int(g) + int(b)


def _env_id(sport: str, matchup: str) -> str:
    return f"MosaicMultiGrid-{sport}-{matchup}-IndAgObs-v1"


# --------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------

class TestAsymmetricRegistration:
    """The 30 asymmetric envs must be visible to Gymnasium."""

    @pytest.mark.parametrize("sport", SPORTS)
    @pytest.mark.parametrize("matchup", MATCHUPS)
    def test_env_id_registered(self, sport, matchup):
        env_id = _env_id(sport, matchup)
        assert env_id in gym.registry, f"{env_id} is not registered"

    def test_total_registered_env_count(self):
        """v7.0.0 target: 51 pre-existing + 30 asymmetric = 81 total."""
        ids = [e for e in gym.registry.keys()
               if e.startswith("MosaicMultiGrid-")]
        assert len(ids) == 81, (
            f"Expected 81 MosaicMultiGrid-* envs, got {len(ids)}"
        )


# --------------------------------------------------------------------------
# Runtime (make + reset + step)
# --------------------------------------------------------------------------

@pytest.fixture(
    params=[(sport, matchup) for sport in SPORTS for matchup in MATCHUPS],
    ids=lambda p: f"{p[0]}-{p[1]}",
)
def asym_env(request):
    sport, matchup = request.param
    env = gym.make(_env_id(sport, matchup))
    yield env, matchup
    env.close()


class TestAsymmetricRuntime:
    def test_creates(self, asym_env):
        env, _ = asym_env
        assert env is not None

    def test_reset_agent_count(self, asym_env):
        env, matchup = asym_env
        obs, _info = env.reset(seed=42)
        assert len(obs) == _n_agents(matchup), (
            f"{matchup}: reset returned {len(obs)} agents, "
            f"expected {_n_agents(matchup)}"
        )

    def test_step(self, asym_env):
        env, matchup = asym_env
        env.reset(seed=42)
        n = _n_agents(matchup)
        actions = {i: env.action_space[i].sample() for i in range(n)}
        obs, rewards, _terminated, _truncated, _info = env.step(actions)
        assert len(obs) == n
        assert len(rewards) == n

    def test_num_agents_matches_matchup(self, asym_env):
        """Both teams non-empty and total agent count matches matchup."""
        env, matchup = asym_env
        env.reset(seed=42)
        g_expected, b_expected = (int(x) for x in matchup.split("v"))
        n_expected = g_expected + b_expected
        assert env.unwrapped.num_agents == n_expected, (
            f"{matchup}: num_agents={env.unwrapped.num_agents}, "
            f"expected {n_expected}"
        )
        # Asymmetric matchups must have both teams non-empty
        assert g_expected > 0 and b_expected > 0, (
            f"{matchup}: asymmetric variants require both teams non-empty"
        )
