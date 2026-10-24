import numpy as np
import pandas as pd
import pytest

from eval.bootstrap import paired_bootstrap
from eval.evaluator import scores


def counts(rows):
    """rows: list of (tp, fp, fn), one per video."""
    return pd.DataFrame([(f"v{i}", *r) for i, r in enumerate(rows)],
                        columns=["video_id", "tp", "fp", "fn"])


BASE = [(8, 2, 2), (5, 5, 1), (0, 3, 4), (9, 1, 0), (3, 0, 3)]
BETTER = [(tp + fn, fp, 0) for tp, fp, fn in BASE]     # same FPs, no misses


def test_identical_methods_give_zero_interval():
    out = paired_bootstrap({"A": counts(BASE), "B": counts(BASE)}, [("A", "B")])
    assert out.loc[0, ["delta_f2", "ci_low", "ci_high"]].tolist() == [0, 0, 0]


def test_point_estimate_is_pooled_f2_difference():
    out = paired_bootstrap({"A": counts(BETTER), "B": counts(BASE)}, [("A", "B")])
    pooled = lambda rows: scores(*np.sum(rows, axis=0))[3]
    assert out.delta_f2[0] == pytest.approx(pooled(BETTER) - pooled(BASE))


def test_strictly_better_method_wins_on_every_paired_draw():
    # A beats B on every single video. Videos differ a lot (v2 has F2 0, v3 near 1), so drawing
    # videos independently for A and B would often make A look worse. Paired draws never do.
    out = paired_bootstrap({"A": counts(BETTER), "B": counts(BASE)}, [("A", "B")])
    assert out.ci_low[0] > 0 and out.share_positive[0] == 1.0


def test_fixed_seed_is_reproducible_and_seed_matters():
    c = {"A": counts(BETTER), "B": counts(BASE)}
    a = paired_bootstrap(c, [("A", "B")], seed=1)
    assert a.equals(paired_bootstrap(c, [("A", "B")], seed=1))
    assert not a.equals(paired_bootstrap(c, [("A", "B")], seed=2))


def test_video_order_does_not_matter():
    shuffled = counts(BASE).iloc[::-1]
    a = paired_bootstrap({"A": counts(BETTER), "B": counts(BASE)}, [("A", "B")])
    b = paired_bootstrap({"A": counts(BETTER), "B": shuffled}, [("A", "B")])
    assert a.equals(b)


def test_mismatched_videos_raise():
    with pytest.raises(ValueError, match="same videos"):
        paired_bootstrap({"A": counts(BASE), "B": counts(BASE[:3])}, [("A", "B")])


def test_scores_zero_division_matches_official():
    p, r, f1, f2 = scores(0, 0, 0)
    assert (p, r, f1, f2) == (1.0, 1.0, 1.0, 1.0)
    p, r, f1, f2 = scores(0, 3, 2)
    assert (p, r, f2) == (0.0, 0.0, 0.0)
