"""Regression tests for the v7.0.0 breaking removals.

v7.0.0 fully removed the `TeamObs` observation wrapper, the ~40
`*TeamObsEnv` classes, and all `MosaicMultiGrid-*-TeamObs-v1` Gym
registrations. These tests fail loudly if any of that code is ever
reintroduced by accident.

Also asserts the positive counterpart: the IndAgObs (independent-agent
observation) variants that replaced TeamObs continue to work.
"""
from __future__ import annotations

import gymnasium as gym
import pytest

# Trigger gymnasium registration
import mosaic_multigrid.envs  # noqa: F401


# --------------------------------------------------------------------------
# Negative tests: TeamObs must NOT be present
# --------------------------------------------------------------------------

class TestTeamObsFullyRemoved:
    """The TeamObs surface area removed in v7.0.0 must remain removed."""

    def test_teamobs_wrapper_class_not_importable(self):
        with pytest.raises(ImportError):
            from mosaic_multigrid.wrappers import TeamObsWrapper  # noqa: F401

    def test_no_teamobs_env_ids_registered(self):
        teamobs_ids = [e for e in gym.registry.keys() if "TeamObs" in e]
        assert teamobs_ids == [], (
            "TeamObs env IDs must not exist in v7.0.0+; "
            f"found {len(teamobs_ids)}: {teamobs_ids}"
        )

    @pytest.mark.parametrize("sport,matchup", [
        ("BB", "2v2"), ("BB", "3v3"),
        ("AF", "2v2"), ("AF", "3v3"),
        ("S",  "2v2"), ("S",  "3v3"),
        ("C",  "2v2"),
    ])
    def test_teamobs_make_raises_nameNotFound(self, sport, matchup):
        env_id = f"MosaicMultiGrid-{sport}-{matchup}-TeamObs-v1"
        with pytest.raises(gym.error.NameNotFound):
            gym.make(env_id)


# --------------------------------------------------------------------------
# Positive tests: IndAgObs replacement still works
# --------------------------------------------------------------------------

class TestIndAgObsStillWorks:
    """Envs of the surviving IndAgObs observation model must still create
    and step cleanly."""

    @pytest.mark.parametrize("env_id", [
        "MosaicMultiGrid-BB-2v2-IndAgObs-v1",
        "MosaicMultiGrid-AF-2v2-IndAgObs-v1",
        "MosaicMultiGrid-S-2v2-IndAgObs-v1",
    ])
    def test_indagobs_2v2_env_step(self, env_id):
        env = gym.make(env_id)
        obs, _info = env.reset(seed=42)
        assert len(obs) == 4
        actions = {i: env.action_space[i].sample() for i in range(4)}
        obs, rewards, _term, _trunc, _info = env.step(actions)
        assert len(obs) == 4
        assert len(rewards) == 4
        env.close()

    def test_solo_envs_still_work(self):
        """Solo (G-1v0 / B-0v1) variants are not TeamObs-based and must
        remain unaffected by the removal."""
        for env_id in [
            "MosaicMultiGrid-BB-G-1v0-v1",
            "MosaicMultiGrid-AF-G-1v0-v1",
            "MosaicMultiGrid-S-G-1v0-v1",
            "MosaicMultiGrid-BB-B-0v1-v1",
            "MosaicMultiGrid-AF-B-0v1-v1",
            "MosaicMultiGrid-S-B-0v1-v1",
        ]:
            env = gym.make(env_id)
            obs, _ = env.reset(seed=42)
            assert len(obs) == 1, f"{env_id}: expected 1 agent, got {len(obs)}"
            env.close()
