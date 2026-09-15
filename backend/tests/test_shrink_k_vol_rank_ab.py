"""The shrink_k/vol_rank A/B's pure helpers.

`matched_width` is the referee both arms are scored by and
`vol_rank_multiplier` is the production normalisation re-stated for the
harness — if either drifts, the paired read is meaningless, so both are pinned
here rather than trusted to review.
"""

import numpy as np
from scripts.archive.shrink_k_vol_rank_ab import (
    matched_width,
    vol_rank_multiplier,
)


class TestMatchedWidth:
    def test_matches_coverage_exactly(self):
        rng = np.random.default_rng(0)
        scale = rng.uniform(0.5, 3.0, size=2000)
        abs_r = rng.uniform(0.0, 10.0, size=2000)
        q, w = matched_width(abs_r, scale)
        assert np.isfinite(q) and np.isfinite(w)
        assert float(np.mean(abs_r <= q * scale)) >= 0.80 - 1e-9
        assert w == q * float(np.mean(scale))

    def test_uniform_rescale_leaves_width_unchanged(self):
        # Width at MATCHED coverage is scale-free: doubling every scale doubles
        # q's denominator and halves q, leaving q*mean(scale) invariant. An arm
        # that merely re-levels must not read as a win.
        rng = np.random.default_rng(1)
        abs_r = rng.uniform(0.0, 10.0, size=2000)
        scale = rng.uniform(0.5, 3.0, size=2000)
        _, w1 = matched_width(abs_r, scale)
        _, w2 = matched_width(abs_r, 2.0 * scale)
        assert w1 == w2 == abs(w1)

    def test_better_shape_reads_narrower(self):
        # An oracle scale (|r| itself) needs q=1.0 everywhere; a pooled constant
        # needs the cross-sectional q80. Shape information must win.
        rng = np.random.default_rng(2)
        abs_r = rng.uniform(0.5, 5.0, size=2000)
        _, w_oracle = matched_width(abs_r, abs_r)
        _, w_pool = matched_width(abs_r, np.full_like(abs_r, abs_r.mean()))
        assert w_oracle < w_pool

    def test_degenerate_fold_votes_nothing(self):
        q, w = matched_width(np.array([]), np.array([]))
        assert np.isnan(q) and np.isnan(w)
        q, w = matched_width(np.array([1.0, 2.0]), np.array([0.0, -1.0]))
        assert np.isnan(q) and np.isnan(w)


class TestVolRankMultiplier:
    def test_mean_one_on_the_fit_split(self):
        # Exact only where no clip binds (the production guarantee is
        # mean-1.0 pre-clip; the [0.25, 4.0] tails move it slightly).
        rng = np.random.default_rng(3)
        raw = rng.uniform(1.0, 3.0, size=1000)
        mult = vol_rank_multiplier(raw, float(np.mean(np.clip(raw, 0.01, None))))
        assert abs(float(np.mean(mult)) - 1.0) < 1e-9
        wide = rng.uniform(0.05, 20.0, size=2000)
        mult_w = vol_rank_multiplier(wide, float(np.mean(np.clip(wide, 0.01, None))))
        assert abs(float(np.mean(mult_w)) - 1.0) < 0.05

    def test_eval_uses_the_fit_mean_not_its_own(self):
        # Normalising on eval would leak the eval volatility level into the
        # scale — the harness must divide by the fit mean even when the eval
        # predictions live on a different level.
        fit_mean = 2.0
        mult = vol_rank_multiplier(np.array([4.0, 4.0, 4.0]), fit_mean)
        assert (mult == 2.0).all()

    def test_clips_match_production(self):
        mult = vol_rank_multiplier(np.array([0.0, 1.0, 100.0]), 1.0)
        assert mult[0] == 0.25  # raw clipped at 0.01 first: 0.01/1 -> floor 0.25
        assert mult[1] == 1.0
        assert mult[2] == 4.0

    def test_unusable_fit_mean_votes_nothing(self):
        out = vol_rank_multiplier(np.array([1.0, 2.0]), 0.0)
        assert np.isnan(out).all()
        out = vol_rank_multiplier(np.array([1.0, 2.0]), float("nan"))
        assert np.isnan(out).all()
