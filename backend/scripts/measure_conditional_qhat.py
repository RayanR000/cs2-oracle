"""Can a conditional `q_hat` fix the served band's coverage?

Pre-registered in `docs/research/2026-08-12-conditional-qhat-preregistration.md`.
Read that first: the bar, the placebo requirement and the void conditions are
fixed there, and this script only reports the numbers they are applied to.

The band over-covers at 87.2/91.8/90.6/89.0% against 80% and runs 58.2-99.2%
per (horizon, date). Five "calibrate on different rows" remedies are refuted
(`changelog/2026-08-12-expanding-window-refuted-for-band-width.md`); the class
left is a `q_hat` that is not a scalar.

WHY THIS IS OFFLINE. The conformal score is `|actual - r_hat| / sigma`, and this
model's predicted `|return|` is median 0.95% / p90 9.01% against half-widths of
10-31%, so `|actual| / sigma` is the score to first order and needs no model.
Validated against the shipped q_hat at ratio 1.020 / 0.958 / 0.887 / 0.785 --
tight at 3d/7d, provisional at 30d. A pass here buys ONE confirm dispatch on
real OOF residuals; it does not buy a ship.

Nothing here reimplements production. Targets come from
`ItemForecaster.prepare_targets` (so the frozen-run, snapshot, collection-shift
and +/-500% rules are production's), sigma from `conformal.sigma_from_columns`,
and every q_hat from `conformal.calibrate` -- including the finite-sample level
`ceil((n+1)(1-alpha))/n`, which is the whole point of routing through it.

Read-only: reads a voted price panel and writes a CSV. No DB, no artifacts.

    venv/bin/python -m scripts.measure_conditional_qhat --horizons 3,7,14,30
"""
from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
from typing import Callable

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import conformal  # noqa: E402
from models.forecaster import ItemForecaster, embargo_days  # noqa: E402

logger = logging.getLogger("conditional_qhat")

# Design constants, all fixed by the pre-registration.
TRAILING_WINDOW_DAYS = 60
TRAILING_MIN_ROWS = 2_000
MIN_HISTORY_DAYS = 120
MIN_ROWS_PER_DATE = 100
MIN_DATES_PER_HORIZON = 100      # void condition
TARGET = conformal.NOMINAL_COVERAGE
MARKET_VOL_WINDOW = 20
PLACEBO_SEED = 20260812

# The >=$1 served cohort. `TRAIN_MIN_MEDIAN_PRICE` filters on the item's MEDIAN
# price before any subsample -- see .claude/rules/training-budget.md.
MIN_MEDIAN_PRICE = 1.0

# `2026-08-11-conformal-centre-follows-serving.md`, for the header only.
SHIPPED_Q_HAT = {3: 94.72, 7: 141.77, 14: 204.34, 30: 312.05}


# --------------------------------------------------------------------------- #
# panel
# --------------------------------------------------------------------------- #

def default_voted_panel() -> str:
    """The local voted cache with the LONGEST date span.

    Not the newest by mtime, which is the obvious default and the wrong one: the
    cache key carries a cutoff date, so a replay anchor leaves behind a recent
    file covering a few hundred dates beside an older one covering twice as many.
    Picking by mtime silently halves the panel, and the only symptom is a smaller
    anchor-date count in a log line.

    The panel is an INPUT, not a fixture: `data/voted_*.parquet` is gitignored
    and the local archive runs behind the durable one. The provenance line in
    `load_panel` is what makes a run citable, so it is printed, not assumed.
    """
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    found = glob.glob(os.path.join(here, "data", "voted_*.parquet"))
    if not found:
        raise SystemExit(
            "no data/voted_*.parquet found. Run a train once to populate the "
            "voted cache, or pass --voted."
        )
    spans = {}
    for f in found:
        d = pd.read_parquet(f, columns=["date"])["date"]
        spans[f] = d.nunique()
    best = max(spans, key=spans.get)
    if len(spans) > 1:
        logger.info("voted caches available: %s", ", ".join(
            f"{os.path.basename(k)}={v}d" for k, v in sorted(
                spans.items(), key=lambda kv: -kv[1])))
    return best


def load_panel(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)[["item_id", "date", "price"]]
    logger.info("panel %s: %s rows, %s dates, %s items, %s -> %s",
                os.path.basename(path), f"{len(df):,}", df["date"].nunique(),
                df["item_id"].nunique(), df["date"].min(), df["date"].max())

    med = df.groupby("item_id")["price"].median()
    df = df[df["item_id"].isin(set(med[med >= MIN_MEDIAN_PRICE].index))].copy()
    df = df.sort_values(["item_id", "date"]).reset_index(drop=True)
    logger.info(">=$%.0f cohort: %s rows, %s items",
                MIN_MEDIAN_PRICE, f"{len(df):,}", df["item_id"].nunique())

    # Exactly engineer_features:1864-1868 -- ROW-based rolling over 60 rows with
    # min_periods=1, grouped by item. NOT a 60-calendar-day window; an item with
    # gaps gets a longer effective span, and that is what production's sigma sees.
    df["price_std_60d"] = (df.groupby("item_id")["price"]
                           .rolling(60, min_periods=1).std().values)
    # State-variable input only: never a label, never compared across arms.
    df["return_1d"] = (df.groupby("item_id")["price"].pct_change() * 100.0)
    return df


def sigma_bounds_for_panel(df: pd.DataFrame) -> tuple[float, float]:
    """Clip bounds from the panel's own cross-section.

    Production derives these on the TRAINING frame and freezes them into the
    artifact so q_hat and every served row share one clip. This panel is the
    training frame's stand-in, so it derives its own -- which is why an absolute
    q_hat from this script is not the artifact's, and only ratios are read.
    """
    raw = (df["price_std_60d"] / df["price"]).replace([np.inf, -np.inf], np.nan)
    return conformal.sigma_bounds(raw.dropna())


def score_frame(fc: ItemForecaster, df: pd.DataFrame, horizon: int,
                floor: float, cap: float) -> pd.DataFrame:
    """One row per surviving item-day: the residual and the sigma it scales by.

    `prepare_targets` is production's, so every label-voiding rule applies here
    identically across arms -- a row voided as a frozen run is absent from all
    of them, not reweighted in one.
    """
    t = fc.prepare_targets(df.copy(), horizon)
    col = f"target_return_{horizon}d"
    t = t[np.isfinite(t[col])].copy()
    t["sigma"] = conformal.sigma_from_columns(
        t["price_std_60d"], t["price"], floor, cap)
    t["resid"] = t[col].to_numpy(dtype=float)
    out = t[["date", "item_id", "resid", "sigma"]].copy()
    out = out[np.isfinite(out["resid"]) & np.isfinite(out["sigma"])
              & (out["sigma"] > 0)]
    return out.sort_values("date").reset_index(drop=True)


def state_variables(df: pd.DataFrame, floor: float, cap: float) -> pd.DataFrame:
    """Per-date state, computable at serve time from date `d` with no forward look.

    V1/V2 read date `d` itself, which serving has; V3 is a trailing window
    ending at `d`. None of them touches a price after `d`.
    """
    sig = pd.Series(
        conformal.sigma_from_columns(df["price_std_60d"], df["price"], floor, cap),
        index=df.index)
    by_date = df.assign(_sigma=sig).groupby("date")

    v1 = by_date["_sigma"].median().rename("V1_xs_med_sigma")
    def _mad(s: pd.Series) -> float:
        # The panel's first date has no return_1d at all, and nanmedian of an
        # empty slice warns and returns NaN. NaN is the right answer; the warning
        # is noise in a log that gets read.
        v = s.to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return float("nan")
        return float(np.median(np.abs(v - np.median(v))))

    v2 = by_date["return_1d"].apply(_mad).rename("V2_xs_mad_ret1d")
    factor = by_date["return_1d"].mean()
    v3 = (factor.rolling(MARKET_VOL_WINDOW, min_periods=5).std()
          .rename("V3_mkt_vol_20d"))

    st = pd.concat([v1, v2, v3], axis=1).reset_index()
    return st.sort_values("date").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# schemes
# --------------------------------------------------------------------------- #

class Panel:
    """Date-indexed views over one horizon's score frame.

    Every scheme forms `q_hat[d]` from rows with anchor <= `d - embargo`, so the
    slicing is by contiguous position in a date-sorted array. `embargo_days` is
    called, never re-derived: it is horizon + 13 and the 13 tracks three other
    constants (see .claude/rules/labels-and-embargo.md).
    """

    def __init__(self, scores: pd.DataFrame, horizon: int,
                 state: pd.DataFrame):
        self.horizon = horizon
        self.embargo = embargo_days(horizon)
        self.resid = scores["resid"].to_numpy(dtype=float)
        self.sigma = scores["sigma"].to_numpy(dtype=float)
        self.dates = pd.to_datetime(scores["date"]).to_numpy()
        # searchsorted needs a sorted key; score_frame guarantees it.
        self.unique_dates = np.unique(self.dates)
        self.state = state.set_index(pd.to_datetime(state["date"]))
        # Precomputed ONCE. Each date's own p80 depends only on its own rows, so
        # there is no leakage in computing the whole series up front -- and a
        # scheme that recomputed it per test date turns this script from seconds
        # into an hour.
        self.date_q = self._per_date_q()

    def upto(self, day: np.datetime64) -> int:
        """Rows with anchor <= day (exclusive upper index)."""
        return int(np.searchsorted(self.dates, day, side="right"))

    def since(self, day: np.datetime64) -> int:
        return int(np.searchsorted(self.dates, day, side="left"))

    def rows_on(self, day: np.datetime64) -> slice:
        return slice(self.since(day), self.upto(day))

    def avail(self, day: np.datetime64) -> np.datetime64:
        """The most recent anchor whose outcome has resolved by `day`."""
        return day - np.timedelta64(self.embargo, "D")

    def _per_date_q(self) -> pd.Series:
        """The p80 each anchor date would need on its own rows."""
        out = {}
        for day in self.unique_dates:
            sl = self.rows_on(day)
            if sl.stop - sl.start < MIN_ROWS_PER_DATE:
                continue
            out[pd.Timestamp(day)] = conformal.calibrate(
                self.resid[sl], self.sigma[sl])
        return pd.Series(out, dtype=float).sort_index()


Scheme = Callable[[Panel, np.datetime64], tuple[float, float] | None]
"""(q_hat, beta) for one test date, or None when the scheme cannot form one."""


def s0_pooled(p: Panel, day: np.datetime64):
    """Production: one scalar over all embargoed history (expanding)."""
    hi = p.upto(p.avail(day))
    if hi < TRAILING_MIN_ROWS:
        return None
    return conformal.calibrate(p.resid[:hi], p.sigma[:hi]), 1.0


def s1_trailing(p: Panel, day: np.datetime64):
    end = p.avail(day)
    start = end - np.timedelta64(TRAILING_WINDOW_DAYS, "D")
    lo, hi = p.since(start), p.upto(end)
    if hi - lo < TRAILING_MIN_ROWS:
        return s0_pooled(p, day)
    return conformal.calibrate(p.resid[lo:hi], p.sigma[lo:hi]), 1.0


def _fit_beta(p: Panel, hi: int) -> float:
    """`d log|resid| / d log sigma`, which split conformal assumes is 1.0.

    Measured at 0.798/0.692 for h=3/7 on the artifact, i.e. sigma OVER-corrects:
    a high-sigma item's residual grows more slowly than its sigma does, so
    dividing by sigma^1 leaves the high-sigma rows over-covered.
    """
    r = np.abs(p.resid[:hi])
    s = p.sigma[:hi]
    ok = (r > 0) & (s > 0) & np.isfinite(r) & np.isfinite(s)
    if ok.sum() < TRAILING_MIN_ROWS:
        return 1.0
    x = np.log(s[ok])
    y = np.log(r[ok])
    xc = x - x.mean()
    denom = float(np.dot(xc, xc))
    if denom <= 0:
        return 1.0
    return float(np.dot(xc, y - y.mean()) / denom)


def s3_sigma_exponent(p: Panel, day: np.datetime64):
    hi = p.upto(p.avail(day))
    if hi < TRAILING_MIN_ROWS:
        return None
    beta = _fit_beta(p, hi)
    return conformal.calibrate(p.resid[:hi], p.sigma[:hi] ** beta), beta


def s4_trailing_exponent(p: Panel, day: np.datetime64):
    end = p.avail(day)
    hi_all = p.upto(end)
    if hi_all < TRAILING_MIN_ROWS:
        return None
    beta = _fit_beta(p, hi_all)
    start = end - np.timedelta64(TRAILING_WINDOW_DAYS, "D")
    lo, hi = p.since(start), hi_all
    if hi - lo < TRAILING_MIN_ROWS:
        lo = 0
    return conformal.calibrate(p.resid[lo:hi], p.sigma[lo:hi] ** beta), beta


def make_s2_state(var: str, shuffle: bool = False) -> Scheme:
    """`log q_hat[d] ~ state[d]`, fitted on embargoed history only.

    Log because q_hat is a positive scale and the fold series moves
    multiplicatively (1.56-2.25x at fixed n_train), not additively.
    """
    rng = np.random.default_rng(PLACEBO_SEED)

    def scheme(p: Panel, day: np.datetime64):
        end = p.avail(day)
        hi = p.upto(end)
        if hi < TRAILING_MIN_ROWS:
            return None
        qd = p.date_q[p.date_q.index <= pd.Timestamp(end)]
        if len(qd) < 30:
            return s0_pooled(p, day)
        st = p.state[var].reindex(qd.index)
        ok = np.isfinite(st.to_numpy()) & np.isfinite(qd.to_numpy())
        if ok.sum() < 30:
            return s0_pooled(p, day)
        x = st.to_numpy()[ok]
        y = np.log(qd.to_numpy()[ok])
        if shuffle:
            # Placebo: same marginal distribution of state, same regression
            # variance, date correspondence destroyed.
            x = rng.permutation(x)
        xc = x - x.mean()
        denom = float(np.dot(xc, xc))
        if denom <= 0:
            return s0_pooled(p, day)
        slope = float(np.dot(xc, y - y.mean()) / denom)
        intercept = float(y.mean() - slope * x.mean())
        now = p.state[var].get(pd.Timestamp(day), np.nan)
        if shuffle:
            now = float(rng.choice(x))
        if not np.isfinite(now):
            return s0_pooled(p, day)
        return float(np.exp(intercept + slope * float(now))), 1.0

    return scheme


def make_s1_placebo() -> Scheme:
    """`S1` with recency destroyed: the same row count drawn from random
    embargoed dates instead of the most recent 60 days.

    This is the control the primary bar needs. A per-date q_hat is noisier than
    a scalar and per-date coverage is measured on ~900 rows, so M2 can move for
    reasons that carry no information at all; the placebo has the trailing
    window's sample size and variance and none of its recency.
    """
    rng = np.random.default_rng(PLACEBO_SEED)

    def scheme(p: Panel, day: np.datetime64):
        end = p.avail(day)
        hi = p.upto(end)
        if hi < TRAILING_MIN_ROWS:
            return None
        start = end - np.timedelta64(TRAILING_WINDOW_DAYS, "D")
        n_real = hi - p.since(start)
        if n_real < TRAILING_MIN_ROWS:
            return s0_pooled(p, day)
        take = rng.choice(hi, size=min(n_real, hi), replace=False)
        return conformal.calibrate(p.resid[take], p.sigma[take]), 1.0

    return scheme


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #

def _cond_err(days: np.ndarray, covered: np.ndarray) -> tuple[float, float, pd.Series]:
    """mean_d |cov[d] - 80%| in pp, plus the fraction of dates inside 10pp."""
    cov = pd.Series(covered).groupby(pd.Series(days)).mean()
    err = float(np.mean(np.abs(cov - TARGET))) * 100.0
    within = float(np.mean(np.abs(cov - TARGET) <= 0.10))
    return err, within, cov


def _sigma_stratum_err(sigma: np.ndarray, covered: np.ndarray,
                       n_strata: int = 10) -> tuple[float, str]:
    """The same statistic over sigma deciles instead of dates.

    Added AFTER the primary read, and it is the dimension `S3` actually
    conditions on: the exponent moves coverage across the sigma range, not
    across the calendar, so scoring it on the per-date statistic alone asks it
    the wrong question. Reported for every arm so the placebo still referees it.
    """
    edges = np.quantile(sigma, np.linspace(0, 1, n_strata + 1)[1:-1])
    idx = np.searchsorted(edges, sigma, side="right")
    cov = pd.Series(covered).groupby(pd.Series(idx)).mean()
    err = float(np.mean(np.abs(cov - TARGET))) * 100.0
    profile = " ".join(f"{v * 100:.0f}" for v in cov.sort_index())
    return err, profile


def evaluate(p: Panel, scheme: Scheme, test_dates) -> dict:
    """Score one scheme, raw and level-matched.

    THE LEVEL-MATCHED COLUMNS ARE THE ONES THAT ANSWER THE QUESTION, and they
    were added after the first read because the pre-registered `M2` could not
    answer it. `M2` falls whenever marginal coverage moves from 83% toward the
    80% target, for any reason at all -- so a scheme that merely lowers the band
    uniformly scores as a conditional-coverage fix. The placebo demonstrated
    exactly that.

    `c` is the single scalar that puts an arm's marginal coverage AT 80%. It is
    fitted on the test rows, which is in-sample for the level -- and that is
    deliberate: it is applied identically to the control and the placebos, so the
    comparison stays paired and the only thing left to distinguish the arms is
    conditional information.
    """
    days, sig, sc, qh, betas = [], [], [], [], []
    for day in test_dates:
        got = scheme(p, day)
        if got is None:
            continue
        q_hat, beta = got
        sl = p.rows_on(day)
        n = sl.stop - sl.start
        if n < MIN_ROWS_PER_DATE:
            continue
        days.append(np.full(n, pd.Timestamp(day).to_datetime64()))
        sig.append(p.sigma[sl])
        sc.append(np.abs(p.resid[sl]) / (p.sigma[sl] ** beta))
        qh.append(np.full(n, q_hat))
        betas.append(beta)
    if not days:
        return {}
    days = np.concatenate(days)
    sig = np.concatenate(sig)
    sc = np.concatenate(sc)
    qh = np.concatenate(qh)

    covered = (sc <= qh)
    err, within, cov = _cond_err(days, covered)

    # The multiplier that lands marginal coverage on 80% exactly. Monotone in c,
    # so bisection is exact to tolerance and needs no derivative.
    lo, hi = 1e-6, 1e6
    for _ in range(60):
        mid = (lo + hi) / 2
        if float(np.mean(sc <= qh * mid)) < TARGET:
            lo = mid
        else:
            hi = mid
    c = (lo + hi) / 2
    cov_lm = (sc <= qh * c)
    err_lm, within_lm, cov_lm_series = _cond_err(days, cov_lm)

    sig_err, sig_profile = _sigma_stratum_err(sig, cov_lm)
    return {
        "n_dates": len(cov),
        "n_rows": int(covered.size),
        "M1_marginal": float(covered.mean()),
        "M2_cond_err_pp": err,
        "M3_frac_within_10pp": within,
        "cov_min": float(cov.min()),
        "cov_max": float(cov.max()),
        "cov_sd_pp": float(cov.std()) * 100.0,
        "beta_mean": float(np.mean(betas)),
        # Level-matched: marginal coverage forced to 80% for every arm.
        "level_match_c": c,
        "M2lm_cond_err_pp": err_lm,
        "M3lm_frac_within_10pp": within_lm,
        "covlm_sd_pp": float(cov_lm_series.std()) * 100.0,
        "M2lm_sigma_err_pp": sig_err,
        "sigma_decile_cov": sig_profile,
    }


def test_dates_for(p: Panel) -> list:
    """Dates with enough embargoed history for every scheme to have a fit.

    The same list for all arms -- the comparison is paired, which is what
    removes the autocorrelation adjacent overlapping anchors introduce.
    """
    first = p.unique_dates.min()
    need = np.timedelta64(MIN_HISTORY_DAYS + p.embargo, "D")
    return [d for d in p.unique_dates if d - first >= need]


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--voted", default=None, help="voted price panel parquet")
    ap.add_argument("--horizons", default="3,7,14,30")
    ap.add_argument("--out", default="conditional_qhat.csv")
    args = ap.parse_args()

    path = args.voted or default_voted_panel()
    df = load_panel(path)
    floor, cap = sigma_bounds_for_panel(df)
    logger.info("sigma clip: floor=%.6f cap=%.6f (this panel's own, not the "
                "artifact's -- read ratios, not levels)", floor, cap)
    state = state_variables(df, floor, cap)
    fc = ItemForecaster(db_session=None)

    schemes: dict[str, Scheme] = {
        "S0_pooled": s0_pooled,
        "S1_trailing60": s1_trailing,
        "S2_V1_xs_med_sigma": make_s2_state("V1_xs_med_sigma"),
        "S2_V2_xs_mad_ret1d": make_s2_state("V2_xs_mad_ret1d"),
        "S2_V3_mkt_vol_20d": make_s2_state("V3_mkt_vol_20d"),
        "S3_sigma_exponent": s3_sigma_exponent,
        "S4_trailing_exponent": s4_trailing_exponent,
        # Placebos, run unconditionally. The pre-registration requires reading
        # them, and running them only after a pass is how a placebo gets skipped.
        "P1_trailing_shuffled": make_s1_placebo(),
        "P2_V3_state_shuffled": make_s2_state("V3_mkt_vol_20d", shuffle=True),
    }

    rows = []
    for h in [int(x) for x in args.horizons.split(",")]:
        sc = score_frame(fc, df, h, floor, cap)
        p = Panel(sc, h, state)
        td = test_dates_for(p)
        logger.info("")
        logger.info("h=%dd: %s scored rows, %d anchor dates, embargo %dd, "
                    "%d test dates", h, f"{len(sc):,}", len(p.unique_dates),
                    p.embargo, len(td))
        if len(td) < MIN_DATES_PER_HORIZON:
            logger.warning("  VOID: %d test dates < %d required",
                           len(td), MIN_DATES_PER_HORIZON)
        for name, scheme in schemes.items():
            m = evaluate(p, scheme, td)
            if not m:
                logger.warning("  %-22s no fit", name)
                continue
            m.update(horizon=h, scheme=name)
            rows.append(m)
            logger.info(
                "  %-22s M1=%.1f%%  M2=%5.2fpp | LEVEL-MATCHED c=%.3f "
                "M2*=%5.2fpp  sigma-strata*=%5.2fpp  sd*=%4.1fpp  beta=%.3f",
                name, m["M1_marginal"] * 100, m["M2_cond_err_pp"],
                m["level_match_c"], m["M2lm_cond_err_pp"],
                m["M2lm_sigma_err_pp"], m["covlm_sd_pp"], m["beta_mean"])
        # The sigma dimension, as a shape rather than a summary. Coverage by
        # sigma decile under production's exponent and under the fitted one --
        # both level-matched, so a tilt here is conditional miscalibration and
        # not the band being too wide overall.
        for name in ("S0_pooled", "S3_sigma_exponent"):
            got = [r for r in rows if r["horizon"] == h and r["scheme"] == name]
            if got:
                logger.info("    sigma-decile cov %-18s %s",
                            name, got[0]["sigma_decile_cov"])

    out = pd.DataFrame(rows)
    if out.empty:
        logger.error("no results")
        return 1
    out.to_csv(args.out, index=False)
    logger.info("")
    logger.info("wrote %s", args.out)

    # The primary bar, applied here so a reader cannot accidentally apply it to
    # M2 alone. Both legs, at >= 3 of 4 horizons, and the placebo must fail.
    base = out[out["scheme"] == "S0_pooled"].set_index("horizon")
    logger.info("")
    logger.info("PRE-REGISTERED PRIMARY BAR — M2 below S0 at >=3 of 4 horizons "
                "AND M1 within 80+/-3pp at >=3 of 4.")
    logger.info("A `P*` row passing this bar VOIDS it: see the pre-registration's "
                "placebo clause.")
    for name in out["scheme"].unique():
        if name == "S0_pooled":
            continue
        arm = out[out["scheme"] == name].set_index("horizon")
        common = arm.index.intersection(base.index)
        better = int((arm.loc[common, "M2_cond_err_pp"]
                      < base.loc[common, "M2_cond_err_pp"]).sum())
        inband = int((
            (arm.loc[common, "M1_marginal"] - TARGET).abs() <= 0.03).sum())
        verdict = "PASS" if (better >= 3 and inband >= 3) else "fail"
        logger.info("  %-22s M2 better at %d/%d, M1 in band at %d/%d -> %s",
                    name, better, len(common), inband, len(common), verdict)

    logger.info("")
    logger.info("LEVEL-MATCHED READ — every arm forced to 80%% marginal first, "
                "so only conditional information can move these.")
    logger.info("  This is NOT the pre-registered statistic. It exists because "
                "M2 above cannot separate a level fix from a conditional one.")
    for name in out["scheme"].unique():
        if name == "S0_pooled":
            continue
        arm = out[out["scheme"] == name].set_index("horizon")
        common = arm.index.intersection(base.index)
        d_date = (arm.loc[common, "M2lm_cond_err_pp"]
                  - base.loc[common, "M2lm_cond_err_pp"])
        d_sig = (arm.loc[common, "M2lm_sigma_err_pp"]
                 - base.loc[common, "M2lm_sigma_err_pp"])
        logger.info("  %-22s dM2*(date) %s | dM2*(sigma) %s",
                    name,
                    " ".join(f"{v:+5.2f}" for v in d_date),
                    " ".join(f"{v:+5.2f}" for v in d_sig))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
