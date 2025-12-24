from __future__ import annotations

import argparse

from ubrain.data.spec import load_challenges
from ubrain.decision.policy import HeuristicPolicy
from ubrain.diffusion.stub import EchoSequenceModel
from ubrain.evaluator.reward import RewardWeights, compute_reward
from ubrain.scheduler.loop import CognitiveLoop


def main():
    parser = argparse.ArgumentParser(description="Run stub cognitive loop on challenges.")
    parser.add_argument(
        "--data",
        type=str,
        default="data/challenges.yaml",
        help="Path to challenges YAML.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=4,
        help="Max steps per episode.",
    )
    args = parser.parse_args()

    challenges = load_challenges(args.data)
    loop = CognitiveLoop(
        model=EchoSequenceModel(),
        policy=HeuristicPolicy(),
        max_steps=args.steps,
        budget_per_step=1.0,
    )
    weights = RewardWeights()

    for ch in challenges:
        result = loop.run_episode(ch)
        correct = ch.ground_truth is not None and result.candidate == ch.ground_truth
        reward = compute_reward(
            result=result,
            weights=weights,
            correct=correct,
            reward_value=ch.reward,
        )
        print(
            f"{ch.id}: decision={result.decision}, candidate={result.candidate}, "
            f"confidence={result.confidence:.2f}, budget_used={result.budget_used:.2f}, "
            f"correct={correct}, reward={reward:.2f}"
        )


if __name__ == "__main__":
    main()
