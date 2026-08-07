#!/usr/bin/env python3
"""Probe CSFloat's per-item sale-history endpoint before committing to a collector.

    GET https://csfloat.com/api/v1/history/<market_hash_name>/graph
    -> [{"day": "2026-08-06T00:00:00Z", "avg_price": 7627.725, "count": 80}, ...]

`docs/changelog/2026-08-06-data-acquisition-ranking.md` ranks this Tier 1 on the
strength of TWO items (AK Redline 1,899 days, Kilowatt Case 899 days). This script
turns that into a measurement, and answers the three questions that decide whether
an A/B is worth building:

1. **Depth on the harness cohort.** Does the endpoint cover the 870 deep >=$1 items
   that `ab_test_item_metadata.py` scores, back far enough to populate the harness's
   validation region? The archive puts the 2/3 split at ~2024-07-10 and CSFloat
   starts 2020-04, so the scored region should be fully covered -- verify, don't
   assume.
2. **Serve-time coverage.** The served cohort is the 8,691 slugs in
   `item-metadata.parquet`, of which ~1,423 are >=$1. A feature that wins on 870 deep
   items but is NULL for most of what `predict()` scores is unshippable, so this is
   measured in the same pass rather than after a positive result.
3. **Price basis.** The Steam listing pages quote buyer prices against the archive's
   net -- a constant 1.1607 that would silently have cost 16% error. This computes the
   same statistics for CSFloat: overall ratio, within-item CV, and a breakdown by year
   and price decile. A CONSTANT ratio is absorbed by a log spread; a DRIFTING one means
   the feature carries a fee schedule rather than a market signal, and only the
   breakdown separates those two cases.

## The budget is 500 requests per DAY

`x-ratelimit-limit: 500` with `x-ratelimit-reset` ~86,400s out. The ranking doc's
"3.7 req/s" is bandwidth, not budget, and the two should not be confused: the full
870-item harness cohort takes **two days**, and a full-catalogue backfill of ~26K items
would take **~52 days** at one IP. That bound belongs in the collector decision, not
just in this probe.

So this script **checkpoints after every item and resumes**. Re-run the same command
after the window resets and it picks up where it stopped, skipping completed items. It
never sleeps waiting for a reset.

## Silent failure

Green-while-dead is the recurring shape in this repo (the Steam backfill's stripped
200s, the supply scraper's empty green runs). A **canary** item that must have deep
history is checked before the run and every `--canary-every` requests; if it stops
returning history the run ABORTS rather than recording zeros. A throttled probe that
reports "0% coverage" is worse than no probe at all.

Read-only. Touches no database and writes nothing to `price-archive/`. Archive paths
resolve against the repo root, so it is safe to run from anywhere -- but per `AGENTS.md`
prefer the repo root, because `backend/.env` points at production Supabase.

Usage:
    backend/venv/bin/python backend/scripts/probe_csfloat_history.py \
        --harness-sample 200 --served-sample 200 --out-dir /tmp/csfloat_probe
    # resume the next day for the rest of the harness cohort:
    backend/venv/bin/python backend/scripts/probe_csfloat_history.py \
        --harness-sample 870 --served-sample 200 --out-dir /tmp/csfloat_probe
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import logging
import random
import time
import urllib.parse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("probe_csfloat_history")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ARCHIVE = REPO_ROOT / "price-archive"

API = "https://csfloat.com/api/v1/history/{name}/graph"

# Documented at 1,899 days / 2020-04-02 in
# `2026-08-06-free-bulk-supply-depth-feeds-exist.md`, and re-verified at 1,899 on
# 2026-08-06. Materially less means throttled or changed -- either way, stop.
CANARY_NAME = "AK-47 | Redline (Field-Tested)"
CANARY_MIN_DAYS = 500

# Universe definition, identical to ab_test_item_metadata.py so the probe measures
# the cohort the A/B would actually score.
MIN_MEDIAN_PRICE = 1.0
MIN_ITEM_DAYS = 180
N_UNIVERSE = 870

HEADERS = {
    # Explicit: no `br`. Neither the venv nor macOS curl can decode brotli, and a
    # silently-undecodable body is the Skinport 406 trap in a different costume.
    "Accept-Encoding": "gzip, deflate",
    "Accept": "application/json",
    "User-Agent": "cs2-oracle-probe/1.0 (research; contact via repo)",
}


# --------------------------------------------------------------------------- #
# Cohorts
# --------------------------------------------------------------------------- #

def _archive_union_sql() -> str:
    """One UNION over the archive, restricted to the single continuous series.

    Mirrors `ab_test_item_metadata.py::_archive_union_sql`: 2026 files filtered to
    `aggregator_sync`, pre-2026 files taken whole.
    """
    pre = sorted(glob.glob(str(ARCHIVE / "prices-20[12][0-9].parquet")))
    y26 = sorted(glob.glob(str(ARCHIVE / "prices-2026-*.parquet")))
    if not pre:
        raise SystemExit(f"No pre-2026 price files under {ARCHIVE} -- is the archive present?")
    parts = [
        f"SELECT item_slug, day, mean_price FROM read_parquet('{f}')" for f in pre
    ] + [
        f"SELECT item_slug, day, mean_price FROM read_parquet('{f}') "
        f"WHERE source = 'aggregator_sync'" for f in y26
    ]
    return " UNION ALL ".join(parts)


def load_cohorts(con: duckdb.DuckDBPyConnection) -> tuple[list[str], list[str]]:
    """Return (harness universe, served >=$1 cohort) as market_hash_name lists."""
    union = _archive_union_sql()

    harness = [r[0] for r in con.sql(f"""
        SELECT item_slug
        FROM ({union})
        GROUP BY item_slug
        HAVING COUNT(DISTINCT day) >= {MIN_ITEM_DAYS}
           AND MEDIAN(mean_price) >= {MIN_MEDIAN_PRICE}
           AND MIN(day) < DATE '2026-01-01'
        ORDER BY COUNT(DISTINCT day) DESC, item_slug
        LIMIT {N_UNIVERSE}
    """).fetchall()]
    logger.info("Harness universe: %d deep >=$1 items", len(harness))

    # The served cohort is exactly item-metadata.parquet (8,691 slugs = the 08-05
    # forecast run). Price it off the last 90 archive days across ALL sources, not
    # just aggregator_sync -- these items are served by the live aggregator.
    live = sorted(glob.glob(str(ARCHIVE / "prices-2026-*.parquet")))
    live_union = " UNION ALL ".join(
        f"SELECT item_slug, day, mean_price FROM read_parquet('{f}')" for f in live
    )
    served = [r[0] for r in con.sql(f"""
        WITH recent AS (
            SELECT item_slug, MEDIAN(mean_price) AS p
            FROM ({live_union})
            WHERE day >= (SELECT MAX(day) - INTERVAL 90 DAY FROM ({live_union}))
            GROUP BY item_slug
        )
        SELECT m.item_slug
        FROM read_parquet('{ARCHIVE / "item-metadata.parquet"}') m
        JOIN recent r USING (item_slug)
        WHERE r.p >= {MIN_MEDIAN_PRICE}
        ORDER BY m.item_slug
    """).fetchall()]
    logger.info("Served >=$1 cohort: %d items (of 8,691 forecast)", len(served))

    return harness, served


# --------------------------------------------------------------------------- #
# Fetch
# --------------------------------------------------------------------------- #

def normalize(payload) -> pd.DataFrame:
    """Coerce the response into (day, price, count).

    The live shape is a flat list of {day, avg_price, count}, descending by day.
    Alternative key names are still probed for: the endpoint is undocumented, so a
    rename would otherwise surface as a coverage collapse rather than an error.
    """
    if isinstance(payload, dict):
        for key in ("data", "history", "items", "results"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
    if not isinstance(payload, list) or not payload:
        return pd.DataFrame(columns=["day", "price", "count"])

    sample = payload[0]
    if not isinstance(sample, dict):
        return pd.DataFrame(columns=["day", "price", "count"])
    keys = set(sample)

    def pick(*cands):
        return next((c for c in cands if c in keys), None)

    day_k = pick("day", "date", "timestamp", "time", "created_at")
    price_k = pick("avg_price", "average_price", "price", "mean_price", "avg")
    count_k = pick("count", "quantity", "volume", "sales", "num_sales")
    if day_k is None or price_k is None:
        raise ValueError(f"Unrecognised response shape; keys={sorted(keys)}")

    df = pd.DataFrame(payload)
    out = pd.DataFrame({
        "day": pd.to_datetime(df[day_k], errors="coerce", utc=True)
                 .dt.tz_localize(None).dt.normalize(),
        "price": pd.to_numeric(df[price_k], errors="coerce"),
    })
    out["count"] = pd.to_numeric(df[count_k], errors="coerce") if count_k else np.nan
    return out.dropna(subset=["day", "price"]).sort_values("day").reset_index(drop=True)


class BudgetExhausted(Exception):
    """The daily 500-request window ran out. Not an error -- resume tomorrow."""


class Fetcher:
    def __init__(self, delay: float, timeout: float, max_consecutive_errors: int,
                 budget_floor: int):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.delay = delay
        self.timeout = timeout
        self.max_consecutive_errors = max_consecutive_errors
        self.budget_floor = budget_floor
        self.consecutive_errors = 0
        self.n_requests = 0
        self.rate_remaining: int | None = None
        self.rate_reset: int | None = None
        self.raw_sample = None

    @property
    def reset_at(self) -> str | None:
        if self.rate_reset is None:
            return None
        return dt.datetime.fromtimestamp(self.rate_reset).isoformat(timespec="seconds")

    def _note_rate(self, r: requests.Response) -> None:
        for header, attr in (("x-ratelimit-remaining", "rate_remaining"),
                             ("x-ratelimit-reset", "rate_reset")):
            v = r.headers.get(header)
            if v is not None:
                try:
                    setattr(self, attr, int(v))
                except ValueError:
                    pass

    def get(self, name: str) -> tuple[pd.DataFrame | None, str]:
        """Return (series, status). status is one of ok/empty/http_<code>/error."""
        # Checked before the request so the floor is a floor, not an overshoot.
        if self.rate_remaining is not None and self.rate_remaining <= self.budget_floor:
            raise BudgetExhausted(
                f"{self.rate_remaining} requests left in the window; resets {self.reset_at}")

        url = API.format(name=urllib.parse.quote(name, safe=""))
        for attempt in range(4):
            try:
                r = self.session.get(url, timeout=self.timeout)
            except requests.RequestException as exc:
                logger.warning("  %s: %s", name, exc)
                time.sleep(2 ** attempt)
                continue

            self.n_requests += 1
            self._note_rate(r)

            if r.status_code == 429:
                # The window is daily, so a 429 is not something to sit out.
                raise BudgetExhausted(f"HTTP 429; window resets {self.reset_at}")
            if r.status_code == 404:
                return pd.DataFrame(columns=["day", "price", "count"]), "empty"
            if r.status_code != 200:
                return None, f"http_{r.status_code}"

            try:
                payload = r.json()
            except ValueError:
                return None, "error"
            if self.raw_sample is None:
                self.raw_sample = {"name": name, "payload": payload[:3]
                                   if isinstance(payload, list) else payload}
            try:
                df = normalize(payload)
            except ValueError as exc:
                logger.error("  %s", exc)
                return None, "error"
            return df, ("ok" if len(df) else "empty")

        return None, "error"

    def fetch(self, name: str) -> tuple[pd.DataFrame | None, str]:
        df, status = self.get(name)
        if status.startswith("http_") or status == "error":
            self.consecutive_errors += 1
        else:
            self.consecutive_errors = 0
        if self.consecutive_errors >= self.max_consecutive_errors:
            raise SystemExit(
                f"ABORT: {self.consecutive_errors} consecutive failures. "
                "Treat every result so far as suspect.")
        time.sleep(self.delay)
        return df, status

    def canary(self, label: str) -> None:
        df, status = self.get(CANARY_NAME)
        n = 0 if df is None else len(df)
        if status != "ok" or n < CANARY_MIN_DAYS:
            raise SystemExit(
                f"ABORT ({label}): canary {CANARY_NAME!r} returned status={status} "
                f"days={n} (expected >= {CANARY_MIN_DAYS}). Throttled or the endpoint "
                "changed -- results so far are NOT trustworthy; do not record them as "
                "coverage.")
        logger.info("Canary OK (%s): %d days, %s -> %s | budget %s left, resets %s",
                    label, n, df.day.min().date(), df.day.max().date(),
                    self.rate_remaining, self.reset_at)
        time.sleep(self.delay)


# --------------------------------------------------------------------------- #
# Checkpointing -- the 500/day budget makes multi-session runs the normal case
# --------------------------------------------------------------------------- #

class Checkpoint:
    """Append-only JSONL of per-item results, reloaded on start."""

    def __init__(self, path: Path):
        self.path = path
        self.done: dict[str, dict] = {}
        if path.exists():
            with path.open() as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    self.done[f"{rec['cohort']}\t{rec['item_slug']}"] = rec
            logger.info("Resuming: %d items already probed in %s", len(self.done), path)

    def has(self, cohort: str, name: str) -> bool:
        return f"{cohort}\t{name}" in self.done

    def add(self, rec: dict) -> None:
        self.done[f"{rec['cohort']}\t{rec['item_slug']}"] = rec
        with self.path.open("a") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")

    def frames(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return (per-item summary, long-form series)."""
        if not self.done:
            return (pd.DataFrame(columns=["cohort", "item_slug", "status", "days"]),
                    pd.DataFrame(columns=["day", "price", "count", "item_slug", "cohort"]))
        rows, series = [], []
        for rec in self.done.values():
            pts = rec.pop("series", None) if "series" in rec else None
            rows.append({k: v for k, v in rec.items() if k != "series"})
            if pts:
                df = pd.DataFrame(pts, columns=["day", "price", "count"])
                series.append(df.assign(item_slug=rec["item_slug"], cohort=rec["cohort"]))
            rec["series"] = pts  # restore; frames() may be called more than once
        summary = pd.DataFrame(rows)
        for col in ("first_day", "last_day"):
            summary[col] = pd.to_datetime(summary[col], errors="coerce")
        long = (pd.concat(series, ignore_index=True) if series
                else pd.DataFrame(columns=["day", "price", "count", "item_slug", "cohort"]))
        if not long.empty:
            long["day"] = pd.to_datetime(long["day"], errors="coerce")
        return summary, long


def probe(fetcher: Fetcher, names: list[str], cohort: str, canary_every: int,
          ckpt: Checkpoint) -> bool:
    """Fetch each name into the checkpoint. Returns False if the budget ran out."""
    todo = [n for n in names if not ckpt.has(cohort, n)]
    if not todo:
        logger.info("[%s] all %d items already in the checkpoint", cohort, len(names))
        return True
    logger.info("[%s] %d to fetch (%d already done)", cohort, len(todo), len(names) - len(todo))

    for i, name in enumerate(todo, 1):
        try:
            if canary_every and i > 1 and (i - 1) % canary_every == 0:
                fetcher.canary(f"{cohort} @{i - 1}")
            df, status = fetcher.fetch(name)
        except BudgetExhausted as exc:
            logger.warning("[%s] budget exhausted after %d of %d: %s", cohort, i - 1, len(todo), exc)
            return False

        rec = {"cohort": cohort, "item_slug": name, "status": status, "days": 0,
               "first_day": None, "last_day": None, "median_price": None,
               "total_count": None, "series": None}
        if df is not None and len(df):
            rec.update(
                days=len(df),
                first_day=str(df.day.min().date()),
                last_day=str(df.day.max().date()),
                median_price=float(df.price.median()),
                total_count=float(np.nansum(df["count"].to_numpy(dtype=float))),
                series=[[str(d.date()), float(p), (None if pd.isna(c) else float(c))]
                        for d, p, c in df.itertuples(index=False)],
            )
        ckpt.add(rec)

        if i % 25 == 0 or i == len(todo):
            logger.info("  [%s] %d/%d  budget_remaining=%s",
                        cohort, i, len(todo), fetcher.rate_remaining)
    return True


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #

def coverage_report(summary: pd.DataFrame, cohort: str, split_date: str) -> dict:
    s = summary[summary.cohort == cohort]
    if s.empty:
        return {"probed": 0}
    ok = s[s.status == "ok"]
    if ok.empty:
        return {"probed": int(len(s)), "with_history": 0, "pct_with_history": 0.0,
                "status_counts": s.status.value_counts().to_dict()}
    # The harness scores from the 2/3 split onward; a series that ends before the
    # split, or starts after it, cannot populate the validation region.
    split = pd.Timestamp(split_date)
    covers = ok[(ok.first_day <= split) & (ok.last_day >= split)]
    return {
        "probed": int(len(s)),
        "with_history": int(len(ok)),
        "pct_with_history": round(100 * len(ok) / len(s), 1),
        "status_counts": s.status.value_counts().to_dict(),
        "days_median": float(ok.days.median()),
        "days_p10": float(ok.days.quantile(0.10)),
        "days_p90": float(ok.days.quantile(0.90)),
        "first_day_median": str(ok.first_day.quantile(0.5).date()),
        "earliest": str(ok.first_day.min().date()),
        "latest": str(ok.last_day.max().date()),
        "spans_split_date": int(len(covers)),
        "pct_spans_split_date": round(100 * len(covers) / len(s), 1),
    }


def basis_report(con: duckdb.DuckDBPyConnection, long: pd.DataFrame) -> dict:
    """CSFloat price vs the archive's net price on the same item-day.

    Reports the statistics the Steam 1.1607 divisor was validated with, plus a
    breakdown by year and price decile.
    """
    if long.empty:
        return {"overlapping_rows": 0}

    union = _archive_union_sql()
    cf = long[["item_slug", "day", "price"]].rename(columns={"price": "cf_price"})
    con.register("cf", cf)
    j = con.sql(f"""
        SELECT c.item_slug, c.day, c.cf_price, a.mean_price AS archive_price
        FROM cf c
        JOIN ({union}) a ON a.item_slug = c.item_slug AND a.day = c.day
        WHERE a.mean_price > 0 AND c.cf_price > 0
    """).df()
    con.unregister("cf")
    if j.empty:
        return {"overlapping_rows": 0,
                "note": "No item-day overlap -- check date normalisation before "
                        "concluding anything about the basis."}

    j["ratio"] = j.cf_price / j.archive_price
    per_item = j.groupby("item_slug").ratio.agg(["median", "std", "mean"])
    per_item["cv"] = per_item["std"] / per_item["mean"]

    j["year"] = j.day.dt.year
    by_year = j.groupby("year").ratio.median()
    j["price_decile"] = pd.qcut(j.archive_price, 10, labels=False, duplicates="drop")
    by_tier = j.groupby("price_decile").ratio.median()

    med = float(j.ratio.median())
    # ~100 means avg_price is in cents; ~1.16 would be the Steam buyer/net shape.
    unit = ("cents (divide by 100)" if 50 < med < 200 else
            "dollars" if 0.5 < med < 2.0 else
            f"unclear -- median ratio {med:.3g}")
    return {
        "overlapping_rows": int(len(j)),
        "overlapping_items": int(j.item_slug.nunique()),
        "inferred_unit": unit,
        "ratio_median": round(med, 4),
        "ratio_p10": round(float(j.ratio.quantile(0.10)), 4),
        "ratio_p90": round(float(j.ratio.quantile(0.90)), 4),
        "within_item_cv_median": round(float(per_item.cv.median()), 4),
        "cross_item_ratio_iqr_pct": round(
            100 * float(per_item["median"].quantile(0.75) - per_item["median"].quantile(0.25))
            / med, 1),
        "ratio_by_year": {int(k): round(float(v), 4) for k, v in by_year.items()},
        "year_drift_pct": (round(100 * (by_year.max() - by_year.min()) / by_year.median(), 1)
                           if len(by_year) > 1 else 0.0),
        "ratio_by_price_decile": {int(k): round(float(v), 4) for k, v in by_tier.items()},
        "tier_drift_pct": (round(100 * (by_tier.max() - by_tier.min()) / by_tier.median(), 1)
                           if len(by_tier) > 1 else 0.0),
    }


def verdict(harness: dict, served: dict, basis: dict) -> list[str]:
    """State the kill criteria explicitly, and whether each one fired."""
    out = []
    h_pct = harness.get("pct_spans_split_date", 0.0)
    out.append(
        f"{'PASS' if h_pct >= 50 else 'KILL'}: {h_pct}% of the harness cohort has history "
        f"spanning the 2/3 split (need >=50%, else the treatment arm is mostly NULL "
        f"exactly where it is scored)")
    s_pct = served.get("pct_with_history", 0.0)
    out.append(
        f"{'PASS' if s_pct >= 50 else 'KILL'}: {s_pct}% of the served >=$1 cohort has any "
        f"history (below ~50%, a win on the harness cohort is unshippable)")
    if basis.get("overlapping_rows", 0) == 0:
        out.append("UNKNOWN: no item-day overlap -- the basis question is unanswered")
    else:
        drift = max(basis.get("year_drift_pct", 0.0), basis.get("tier_drift_pct", 0.0))
        cv = basis.get("within_item_cv_median", 1.0)
        stable = drift < 5 and cv < 0.05
        out.append(
            f"{'PASS' if stable else 'CAUTION'}: ratio drift {drift}% across year/tier, "
            f"within-item CV {cv} (Steam's divisor was CV 0.0021; a drifting ratio means "
            f"the spread would carry a fee schedule, not a market signal)")
    return out


# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description="Probe CSFloat sale-history coverage")
    ap.add_argument("--harness-sample", type=int, default=200,
                    help="items from the 870 deep >=$1 universe (0 to skip)")
    ap.add_argument("--served-sample", type=int, default=200,
                    help="items from the served >=$1 cohort (0 to skip)")
    ap.add_argument("--delay", type=float, default=0.35,
                    help="seconds between requests; 0.35 ~ 2.9 req/s, under the 3.7 measured")
    ap.add_argument("--timeout", type=float, default=20.0)
    ap.add_argument("--canary-every", type=int, default=50,
                    help="re-check the canary every N requests (0 to disable)")
    ap.add_argument("--max-consecutive-errors", type=int, default=8)
    ap.add_argument("--budget-floor", type=int, default=5,
                    help="stop with this many requests left in the daily window")
    ap.add_argument("--split-date", default="2024-07-10",
                    help="harness 2/3 split; measured from the archive")
    ap.add_argument("--seed", type=int, default=20260806)
    ap.add_argument("--out-dir", default="/tmp/csfloat_probe")
    ap.add_argument("--report-only", action="store_true",
                    help="re-analyse the existing checkpoint without any requests")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = Checkpoint(out / "checkpoint.jsonl")

    con = duckdb.connect()
    harness_all, served_all = load_cohorts(con)

    # Harness is taken deepest-first (deterministic, matches the A/B's own ordering);
    # served is shuffled, because item-metadata is name-ordered and a prefix would be
    # an alphabetical slice rather than a sample.
    harness = harness_all[:args.harness_sample] if args.harness_sample else []
    served = served_all[:]
    random.Random(args.seed).shuffle(served)
    served = served[:args.served_sample] if args.served_sample else []

    complete = True
    if not args.report_only:
        pending = (sum(not ckpt.has("harness", n) for n in harness)
                   + sum(not ckpt.has("served", n) for n in served))
        logger.info("Probing %d new items at %.2fs delay ~ %.0f min (budget is 500/day)",
                    pending, args.delay, pending * args.delay / 60)

        fetcher = Fetcher(args.delay, args.timeout, args.max_consecutive_errors,
                          args.budget_floor)
        fetcher.canary("pre-run")
        for names, cohort in ((harness, "harness"), (served, "served")):
            if names and complete:
                complete = probe(fetcher, names, cohort, args.canary_every, ckpt)
        if complete:
            fetcher.canary("post-run")
        if fetcher.raw_sample:
            (out / "csfloat_raw_sample.json").write_text(
                json.dumps(fetcher.raw_sample, indent=2, default=str))

    summary, long = ckpt.frames()
    if summary.empty:
        logger.error("Nothing probed. Check connectivity and re-run.")
        return 1

    summary.to_parquet(out / "csfloat_probe_summary.parquet", index=False)
    if not long.empty:
        long.to_parquet(out / "csfloat_probe_series.parquet", index=False)

    h_rep = coverage_report(summary, "harness", args.split_date)
    s_rep = coverage_report(summary, "served", args.split_date)
    b_rep = basis_report(con, long)
    lines = verdict(h_rep, s_rep, b_rep)

    report = {
        "complete": complete,
        "items_probed": int(len(summary)),
        "series_rows": int(len(long)),
        "cohort_sizes": {"harness_universe": len(harness_all), "served_ge1": len(served_all)},
        "requested": {"harness": len(harness), "served": len(served)},
        "harness": h_rep,
        "served": s_rep,
        "basis": b_rep,
        "verdict": lines,
    }
    (out / "csfloat_probe_report.json").write_text(json.dumps(report, indent=2, default=str))

    print("\n" + "=" * 78)
    print(json.dumps(report, indent=2, default=str))
    print("=" * 78)
    for line in lines:
        print(line)
    print(f"\n{len(summary)} item rows, {len(long)} series rows -> {out}")
    if not complete:
        print("INCOMPLETE: the daily budget ran out. Re-run the same command after the "
              "window resets; the checkpoint resumes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
