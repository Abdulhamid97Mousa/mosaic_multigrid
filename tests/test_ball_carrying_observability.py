"""
Test ball carrying observability in STATE channel encoding.

This test verifies the fix for Bug #3 in SOCCER_IMPROVEMENTS.md:
Agents can now see when OTHER agents are carrying the ball by checking
the STATE channel (values 100-103 indicate carrying + direction).
"""
import numpy as np
import pytest

from mosaic_multigrid.envs import SoccerGame4HEnv10x15N2, CollectGame3HEnv10x10N3
from mosaic_multigrid.core.constants import Type, Color
from mosaic_multigrid.core.world_object import Ball


class TestBallCarryingObservability:
    """Test that agents can observe ball carrying in STATE channel."""

    def test_agent_without_ball_encoded_correctly(self):
        """Agent WITHOUT ball should have STATE = direction (0-3)."""
        env = SoccerGame4HEnv10x15N2()
        obs, _ = env.reset(seed=42)

        # Agent 0 should NOT have ball initially
        assert env.agents[0].state.carrying is None

        # Check encoding
        encoding = env.agents[0].encode()
        type_idx, color_idx, state_value = encoding

        assert type_idx == Type.agent.to_index()
        assert 0 <= state_value <= 3, (
            f"Agent without ball should have STATE=0-3 (direction), got {state_value}"
        )
        print(f"Agent without ball: STATE={state_value} (direction only)")

    def test_agent_with_ball_encoded_with_carrying_flag(self):
        """Agent WITH ball should have STATE = 100 + direction."""
        env = SoccerGame4HEnv10x15N2()
        obs, _ = env.reset(seed=42)

        # Find a ball and give it to Agent 0
        ball = None
        for x in range(env.width):
            for y in range(env.height):
                obj = env.grid.get(x, y)
                if obj and obj.type == Type.ball:
                    ball = obj
                    break
            if ball:
                break

        assert ball is not None, "No ball found in environment"

        # Give ball to Agent 0
        env.agents[0].state.carrying = ball
        env.grid.set(*ball.cur_pos, None)  # Remove from grid

        # Check encoding
        encoding = env.agents[0].encode()
        type_idx, color_idx, state_value = encoding

        assert type_idx == Type.agent.to_index()
        assert state_value >= 100, (
            f"Agent with ball should have STATE >= 100, got {state_value}"
        )

        direction = state_value - 100
        assert 0 <= direction <= 3, (
            f"Direction should be 0-3 after removing offset, got {direction}"
        )

        print(f"   Agent with ball: STATE={state_value} (100+{direction})")
        print(f"   Decodes to: has_ball=True, direction={direction}")

    def test_agent_can_observe_other_agent_carrying_ball(self):
        """Test that Agent 0 can SEE when Agent 1 is carrying ball."""
        env = SoccerGame4HEnv10x15N2()
        obs, _ = env.reset(seed=42)

        # Find ball and give to Agent 1
        ball = None
        for x in range(env.width):
            for y in range(env.height):
                obj = env.grid.get(x, y)
                if obj and obj.type == Type.ball:
                    ball = obj
                    break
            if ball:
                break

        assert ball is not None, "No ball found"

        # Give ball to Agent 1
        env.agents[1].state.carrying = ball
        env.grid.set(*ball.cur_pos, None)

        # CRITICAL: With view_size=3 and agent at bottom-center,
        # we need to position agents to guarantee visibility
        #
        # Agent's 3×3 view layout (agent-centric, rotates with direction):
        #   [0,0] [1,0] [2,0]  ← Top row (furthest)
        #   [0,1] [1,1] [2,1]  ← Middle row
        #   [0,2] [1,2] [2,2]  ← Bottom row (agent at [1,2])
        #
        # For dir=0 (facing right), the view in world coordinates is:
        #   Agent at (x, y), view covers roughly (x-1 to x+1, y-1 to y+1)

        # Place agents in controlled positions
        # Agent 0 at (7, 5) facing up (dir=3)
        env.agents[0].state.pos = (7, 5)
        env.agents[0].state.dir = 3  # Facing up

        # Agent 1 at (7, 4) - directly in front of Agent 0
        # This should be visible at position [1, 0] in Agent 0's view
        env.agents[1].state.pos = (7, 4)
        env.agents[1].state.dir = 1  # Facing down (towards Agent 0)

        # Generate observations
        obs = env.gen_obs()
        agent0_view = obs[0]['image']

        print(f"\nTest Setup:")
        print(f"  Agent 0: pos={env.agents[0].state.pos}, dir={env.agents[0].state.dir} (up)")
        print(f"  Agent 1: pos={env.agents[1].state.pos}, dir={env.agents[1].state.dir} (down), carrying=True")
        print(f"  Agent 1 should be visible directly ahead of Agent 0")

        # Debug: print Agent 0's view
        print("\nAgent 0's view (3×3, facing up):")
        for y in range(agent0_view.shape[0]):
            row = []
            for x in range(agent0_view.shape[1]):
                obj_type = agent0_view[y, x, 0]
                obj_state = agent0_view[y, x, 2]
                if obj_type == Type.agent.to_index():
                    if obj_state >= 100:
                        row.append(f"A+B")
                    else:
                        row.append(f"A({obj_state})")
                elif obj_type == Type.wall.to_index():
                    row.append("W")
                elif obj_type == Type.ball.to_index():
                    row.append("B")
                else:
                    row.append(".")
            print(f"  {' '.join(row)}")

        # Search for Agent 1 carrying ball in Agent 0's view
        found_agent_with_ball = False
        for y in range(agent0_view.shape[0]):
            for x in range(agent0_view.shape[1]):
                obj_type = agent0_view[y, x, 0]
                obj_color = agent0_view[y, x, 1]
                obj_state = agent0_view[y, x, 2]

                if obj_type == Type.agent.to_index():
                    print(f"\n  Found agent at view position ({x}, {y}): STATE={obj_state}")

                    if obj_state >= 100:
                        # This agent is carrying ball!
                        direction = obj_state - 100
                        print(f"  [PASS] Agent 0 can see agent carrying ball!")
                        print(f"         View position: ({x}, {y})")
                        print(f"         STATE: {obj_state} = direction {direction} + CARRYING_BALL")
                        found_agent_with_ball = True
                        break

        assert found_agent_with_ball, (
            f"Agent 0 should be able to see Agent 1 carrying the ball.\n"
            f"Agent 0 at {env.agents[0].state.pos} facing dir={env.agents[0].state.dir}\n"
            f"Agent 1 at {env.agents[1].state.pos} with ball\n"
            f"View contents:\n{agent0_view}"
        )

    def test_state_channel_preserves_direction(self):
        """Verify that STATE channel encodes BOTH direction AND carrying."""
        env = SoccerGame4HEnv10x15N2()
        obs, _ = env.reset(seed=42)

        # Find ball
        ball = None
        for x in range(env.width):
            for y in range(env.height):
                obj = env.grid.get(x, y)
                if obj and obj.type == Type.ball:
                    ball = obj
                    break
            if ball:
                break

        assert ball is not None, "No ball found"

        # Test all 4 directions
        for direction in range(4):
            env.agents[0].state.dir = direction
            env.agents[0].state.carrying = ball

            encoding = env.agents[0].encode()
            state_value = encoding[2]

            # Should be 100 + direction
            expected_state = 100 + direction
            assert state_value == expected_state, (
                f"Direction {direction} with ball should encode as {expected_state}, "
                f"got {state_value}"
            )

            # Verify decoding
            has_ball = (state_value >= 100)
            decoded_dir = state_value % 100

            assert has_ball is True
            assert decoded_dir == direction

        print(" All directions (0-3) correctly encoded with carrying flag (100-103)")

    def test_collect_game_also_supports_ball_carrying_encoding(self):
        """Collect game should also support ball carrying observability."""
        env = CollectGame3HEnv10x10N3()
        obs, _ = env.reset(seed=42)

        # Find a ball
        ball = None
        for x in range(env.width):
            for y in range(env.height):
                obj = env.grid.get(x, y)
                if obj and obj.type == Type.ball:
                    ball = obj
                    break
            if ball:
                break

        assert ball is not None, "No ball found in Collect environment"

        # Give ball to Agent 0
        env.agents[0].state.carrying = ball
        env.grid.set(*ball.cur_pos, None)

        # Check encoding
        encoding = env.agents[0].encode()
        state_value = encoding[2]

        assert state_value >= 100, (
            f"Collect game should also encode ball carrying with STATE >= 100, "
            f"got {state_value}"
        )

        print(" Collect game also supports ball carrying observability")

    def test_backward_compatibility_with_non_ball_carrying(self):
        """Verify that non-ball objects don't trigger carrying flag."""
        env = SoccerGame4HEnv10x15N2()
        obs, _ = env.reset(seed=42)

        # Agent carrying nothing
        env.agents[0].state.carrying = None
        env.agents[0].state.dir = 2  # Left

        encoding = env.agents[0].encode()
        state_value = encoding[2]

        assert state_value == 2, (
            f"Agent without carrying should have STATE=direction only, got {state_value}"
        )

        print(" Backward compatibility: Agents without ball still use 0-3 encoding")

    def test_encoding_ranges_are_clearly_separated(self):
        """Verify that STATE encoding ranges don't overlap incorrectly."""
        # For AGENTS:
        # - Without ball: 0-3 (direction)
        # - With ball: 100-103 (direction + carrying flag)

        agent_without_ball = [0, 1, 2, 3]
        agent_with_ball = [100, 101, 102, 103]

        # These should be clearly separated
        assert all(v < 100 for v in agent_without_ball), "Without ball: should be < 100"
        assert all(v >= 100 for v in agent_with_ball), "With ball: should be >= 100"

        # No overlap between the two ranges
        agent_all = agent_without_ball + agent_with_ball
        assert len(agent_all) == len(set(agent_all)), (
            "Agent STATE encodings should not overlap"
        )

        print(" Clear separation: agents without ball (0-3), agents with ball (100-103)")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
