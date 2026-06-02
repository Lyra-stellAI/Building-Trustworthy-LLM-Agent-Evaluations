"""Part 3 - Pairwise vs. rubric-based scoring (and which to trust).

The COLM 2025 result: a content-neutral distractor (assertiveness, prolixity,
sycophancy) flips a pairwise verdict ~35% of the time but an absolute score only
~9%. Pairwise also refuses to call ties, and any pairwise leaderboard can be
gamed by rewriting answers to sound more confident.
"""

import statistics

from _bootstrap import header, section

from trustworthy_evals.protocols import (
    leaderboard_hacking_demo,
    summarize_distracted,
    tie_recognition_experiment,
)


def main() -> None:
    header("Part 3 - Distracted evaluation: pairwise is the fragile protocol")

    section("Verdict flip rate when a distractor is added to the worse answer")
    results = summarize_distracted()
    print(f"  {'distractor':14s} {'pairwise':>10s} {'absolute':>10s}")
    for d, r in results.items():
        print(f"  {d:14s} {r.pairwise_flip_rate:>9.1%} {r.absolute_flip_rate:>10.1%}")
    pw = statistics.mean(r.pairwise_flip_rate for r in results.values())
    ab = statistics.mean(r.absolute_flip_rate for r in results.values())
    print(f"  {'OVERALL':14s} {pw:>9.1%} {ab:>10.1%}   (paper: ~35% vs ~9%)")

    section("Ties: equal-quality pairs")
    identical, tie = tie_recognition_experiment()
    print(f"  absolute assigns identical scores : {identical:.1%}  (correctly recognizes the tie)")
    print(f"  pairwise calls a tie              : {tie:.1%}  (it manufactures a winner)")

    section("Leaderboard hacking: rewrite the bottom models to be assertive (no facts changed)")
    hack = leaderboard_hacking_demo()
    print(f"  hacked models : {hack.hacked}")
    print(f"  pairwise rank before : {hack.pairwise_before}")
    print(f"  pairwise rank after  : {hack.pairwise_after}")
    print(f"  absolute rank before : {hack.absolute_before}")
    print(f"  absolute rank after  : {hack.absolute_after}")
    for m in hack.hacked:
        print(f"    {m:8s}: pairwise {hack.pairwise_rank_change(m):+d} places, absolute {hack.absolute_rank_change(m):+d}")

    print("\n  Lesson: for verifiable/correctness criteria and near-tie-heavy data, default to")
    print("  ABSOLUTE scoring. Reserve pairwise for open-ended quality with no anchor -- and")
    print("  then run both orders and guard against distracted evaluation.")


if __name__ == "__main__":
    main()
