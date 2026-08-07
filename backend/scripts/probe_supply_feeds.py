#!/usr/bin/env python3
"""Probe four candidate "supply depth" (listing-count) feeds before building a collector.

    1. Skinport      GET api.skinport.com/v1/items?app_id=730&currency=USD   field `quantity`
    2. Waxpeer       GET api.waxpeer.com/v1/prices?game=csgo                 field `count`
    3. market.csgo.com GET market.csgo.com/api/v2/prices/USD.json            field `volume` (a STRING)
    4. Bitskins      GET api.bitskins.com/market/insell/730                  field `quantity`

`docs/changelog/2026-08-06-free-bulk-supply-depth-feeds-exist.md` measured three of these
(not market.csgo.com) from a residential IP and reported coverage of 62.4/63.2/20.1% of the
>=$1 cohort for Skinport/Waxpeer/Bitskins and a 71.9% union. This script re-verifies those
claims and adds the fourth feed, whose field is a landmine: `volume` is the name this repo
uses everywhere else for TRADE volume (completed sales), which is already refuted at
|r| < 0.002 against forward returns. If market.csgo.com's `volume` is trade count rather
than a live listing count, the feed is worthless here and must be dropped -- so this script
treats that question as the load-bearing one, not a footnote:

  - distribution shape (a listing count has a long right tail into the thousands; a daily
    completed-sale count usually does not, except for a handful of extremely liquid cases)
  - cross-feed correlation against Skinport/Waxpeer/Bitskins' listing-count fields (a listing
    count should correlate positively with another marketplace's listing count; a trade count
    need not)
  - a direct comparison against CSFloat's `/api/v1/history/<name>/graph` endpoint, which
    returns genuine daily completed-sale counts, for a handful of canary items spanning very
    liquid to illiquid

## Known trap: Skinport requires `Accept-Encoding: br`

Skinport returns HTTP 406 without a `br` Accept-Encoding, which macOS curl and this venv
cannot decode natively (no `brotli` module). A prior session misdiagnosed this as
"Cloudflare-dead." This script requests `br` and decodes with the stdlib-free `brotli`
package if present, falling back to shelling out to `node -e` (see `_brotli_decode`).

There is a SECOND, different failure mode this script must not confuse with the first: if
the run's egress IP is itself Cloudflare-owned (common for sandboxed/hosted dev
environments -- check `curl -s https://ipinfo.io/json`), Skinport's WAF blocks the request
with an HTTP 403 Cloudflare challenge page, independent of Accept-Encoding. That is a
network/environment limitation, not evidence Skinport is dead, and this script reports it
as a distinct status (`blocked_waf`) rather than silently folding it into "406 trap" or,
worse, a fabricated zero.

## Coverage and overlap

Joins each feed's catalog against `price-archive/prices-2026-08.parquet` on
`item_slug == market_hash_name` (exact string equality -- verified over unmatched samples
that near-misses are not a whitespace/case artifact but real catalog gaps, e.g. Souvenir
weapons and 2026 tournament stickers the archive does not yet track).

Universe = distinct `item_slug` in the file. Cohort = items whose AVG(mean_price) across
all rows for that item_slug in the same file is >= `MIN_SERVED_PRICE_USD` (1.0), matching
`backend/api/serving_policy.py`'s floor. This is a defensible but not the only possible
definition; MEDIAN instead of AVG shifts the cohort by ~4% (checked by hand, not asserted
by this script) -- read the printed cohort size before citing it elsewhere.

Marginal contribution of feed X = |union of all available feeds| - |union of all available
feeds except X|. This is the number that decides whether collecting a feed is worth the
daily request, not its raw match count.

Read-only. Touches no database and writes nothing to `price-archive/`. Safe to run from
anywhere; per `AGENTS.md`, prefer the repo root because `backend/.env` points at
production Supabase (irrelevant here since nothing touches Postgres, but the convention
is cheap to keep).

Usage:
    backend/venv/bin/python backend/scripts/probe_supply_feeds.py --out-dir /tmp/supply_feeds_probe
    # skip the CSFloat cross-check (budget is 500 req/day, shared with other probes):
    backend/venv/bin/python backend/scripts/probe_supply_feeds.py --no-csfloat-check
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import subprocess
import sys
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
logger = logging.getLogger("probe_supply_feeds")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ARCHIVE = REPO_ROOT / "price-archive"
MIN_SERVED_PRICE_USD = 1.0  # backend/api/serving_policy.py

UA = "cs2-oracle-probe/1.0 (research; contact via repo)"

FEEDS = {
    "skinport": {
        "url": "https://api.skinport.com/v1/items?app_id=730&currency=USD",
        "depth_field": "quantity",
        "name_field": "market_hash_name",
        "list_key": None,  # top-level list
        "requires_br": True,
    },
    "waxpeer": {
        "url": "https://api.waxpeer.com/v1/prices?game=csgo",
        "depth_field": "count",
        "name_field": "name",
        "list_key": "items",
        "requires_br": False,
    },
    "market_csgo_com": {
        "url": "https://market.csgo.com/api/v2/prices/USD.json",
        "depth_field": "volume",
        "name_field": "market_hash_name",
        "list_key": "items",
        "requires_br": False,
        "depth_is_string": True,
    },
    "bitskins": {
        "url": "https://api.bitskins.com/market/insell/730",
        "depth_field": "quantity",
        "name_field": "name",
        "list_key": "list",
        "requires_br": False,
    },
}

# Canary items for the trade-volume-vs-listing-count check, spanning very liquid (a case
# that gets opened constantly) to illiquid (a rare contraband skin). CSFloat's
# `/api/v1/history/<name>/graph` returns genuine daily completed-sale `count` -- see
# probe_csfloat_history.py for the endpoint's other properties (500 req/day budget,
# undocumented, no auth).
CSFLOAT_CANARIES = [
    "AK-47 | Redline (Field-Tested)",
    "Kilowatt Case",
    "AWP | Asiimov (Field-Tested)",
    "Glock-18 | Fade (Factory New)",
    "Chroma Case",
    "P250 | Sand Dune (Field-Tested)",
    "Operation Broken Fang Case",
    "M4A4 | Howl (Field-Tested)",
    "Danger Zone Case",
]
CSFLOAT_API = "https://csfloat.com/api/v1/history/{name}/graph"
CSFLOAT_HEADERS = {
    "Accept-Encoding": "gzip, deflate",  # never br -- same undecodable-brotli trap
    "Accept": "application/json",
    "User-Agent": UA,
}


# --------------------------------------------------------------------------- #
# Fetch
# --------------------------------------------------------------------------- #

def _brotli_decode(raw: bytes) -> bytes | None:
    """Decode brotli without assuming the `brotli` package is installed.

    Falls back to `node -e` (see docs/references/data-sources.md's Skinport recipe).
    Returns None -- not raw bytes, not an empty string -- if both paths fail, so a
    caller cannot mistake a failed decode for an empty-but-valid payload.
    """
    try:
        import brotli  # type: ignore
        return brotli.decompress(raw)
    except ImportError:
        pass
    except Exception as exc:  # pragma: no cover - real brotli errors are data, not code
        logger.warning("brotli package present but decompress failed: %s", exc)
        return None

    node = None
    for candidate in ("node",):
        from shutil import which
        node = which(candidate)
        if node:
            break
    if not node:
        logger.error("No `brotli` module and no `node` on PATH -- cannot decode. "
                      "Install one or the other.")
        return None

    tmp_in = Path("/tmp/_probe_supply_feeds_brotli_in.bin")
    tmp_in.write_bytes(raw)
    try:
        out = subprocess.run(
            [node, "-e",
             "process.stdout.write(require('zlib').brotliDecompressSync("
             f"require('fs').readFileSync('{tmp_in}')))"],
            capture_output=True, timeout=30,
        )
    finally:
        tmp_in.unlink(missing_ok=True)
    if out.returncode != 0:
        logger.error("node brotli decode failed: %s", out.stderr.decode(errors="replace"))
        return None
    return out.stdout


def fetch_feed(key: str, spec: dict, timeout: float = 30.0) -> dict:
    """GET one feed. Returns a result dict; never raises on HTTP/network failure.

    status is one of: ok, http_<code>, blocked_waf, decode_failed, parse_failed,
    connection_error.
    """
    headers = {"Accept": "application/json", "User-Agent": UA}
    if spec["requires_br"]:
        headers["Accept-Encoding"] = "br"
    else:
        headers["Accept-Encoding"] = "gzip, deflate"

    t0 = time.time()
    try:
        r = requests.get(spec["url"], headers=headers, timeout=timeout)
    except requests.RequestException as exc:
        return {"feed": key, "status": "connection_error", "error": str(exc),
                "wall_time_s": round(time.time() - t0, 3)}
    wall_time = time.time() - t0
    content_length = len(r.content)

    if r.status_code != 200:
        # Cloudflare WAF blocks render as HTML with server: cloudflare, regardless of
        # Accept-Encoding -- distinct from the documented "406 without br" trap, and
        # must not be reported the same way (one is fixable with a header, the other
        # is an egress-IP problem this script cannot route around).
        is_cf = "cloudflare" in r.headers.get("server", "").lower()
        status = "blocked_waf" if (is_cf and r.status_code in (403, 503)) else f"http_{r.status_code}"
        return {"feed": key, "status": status, "http_status": r.status_code,
                "bytes": content_length, "wall_time_s": round(wall_time, 3),
                "server_header": r.headers.get("server"),
                "content_type": r.headers.get("content-type")}

    body = r.content
    if spec["requires_br"] and r.headers.get("content-encoding") == "br":
        decoded = _brotli_decode(body)
        if decoded is None:
            return {"feed": key, "status": "decode_failed", "bytes": content_length,
                     "wall_time_s": round(wall_time, 3)}
        body = decoded

    try:
        payload = json.loads(body)
    except (ValueError, UnicodeDecodeError) as exc:
        return {"feed": key, "status": "parse_failed", "error": str(exc),
                "bytes": content_length, "wall_time_s": round(wall_time, 3)}

    items = payload[spec["list_key"]] if spec["list_key"] else payload
    if not isinstance(items, list):
        return {"feed": key, "status": "parse_failed",
                "error": f"expected a list at key={spec['list_key']!r}, got {type(items)}",
                "bytes": content_length, "wall_time_s": round(wall_time, 3)}

    return {"feed": key, "status": "ok", "bytes": content_length,
            "wall_time_s": round(wall_time, 3), "item_count": len(items),
            "sample": items[:2], "items": items}


# --------------------------------------------------------------------------- #
# Hygiene: dupes, nulls, zeros, coercion
# --------------------------------------------------------------------------- #

def hygiene_report(key: str, spec: dict, items: list[dict]) -> dict:
    name_f, depth_f = spec["name_field"], spec["depth_field"]
    names = [it.get(name_f) for it in items]
    n = len(items)
    n_names_null = sum(1 for x in names if x is None)
    dup_names = n - len(set(x for x in names if x is not None))

    raw_depths = [it.get(depth_f) for it in items]
    n_depth_null = sum(1 for x in raw_depths if x is None)

    coerced = []
    n_uncoercible = 0
    for x in raw_depths:
        if x is None:
            continue
        try:
            coerced.append(float(x))
        except (TypeError, ValueError):
            n_uncoercible += 1
    n_zero = sum(1 for x in coerced if x == 0)

    report = {
        "feed": key,
        "n_items": n,
        "duplicate_names": dup_names,
        "null_names": n_names_null,
        "null_depth": n_depth_null,
        "depth_field_is_string_in_json": isinstance(
            next((x for x in raw_depths if x is not None), None), str),
        "uncoercible_depth_values": n_uncoercible,
        "zero_depth_values": n_zero,
    }
    if coerced:
        arr = np.array(coerced)
        report["depth_stats"] = {
            "min": float(arr.min()), "median": float(np.median(arr)),
            "p90": float(np.quantile(arr, 0.90)), "p99": float(np.quantile(arr, 0.99)),
            "max": float(arr.max()), "mean": float(arr.mean()),
        }
    return report


# --------------------------------------------------------------------------- #
# Coverage against the archive
# --------------------------------------------------------------------------- #

def build_universe_and_cohort(con: duckdb.DuckDBPyConnection, min_price: float) -> tuple[int, int]:
    path = ARCHIVE / "prices-2026-08.parquet"
    if not path.exists():
        raise SystemExit(f"{path} not found -- is the archive present?")
    con.execute(f"""
        CREATE OR REPLACE TABLE universe AS
        SELECT item_slug, AVG(mean_price) AS recent_mean_price
        FROM read_parquet('{path}')
        GROUP BY item_slug
    """)
    con.execute(f"CREATE OR REPLACE TABLE cohort AS "
                f"SELECT item_slug FROM universe WHERE recent_mean_price >= {min_price}")
    n_universe = con.sql("SELECT COUNT(*) FROM universe").fetchone()[0]
    n_cohort = con.sql("SELECT COUNT(*) FROM cohort").fetchone()[0]
    return n_universe, n_cohort


def coverage_and_overlap(con: duckdb.DuckDBPyConnection, ok_results: dict[str, dict],
                          n_universe: int, n_cohort: int) -> dict:
    for key, res in ok_results.items():
        name_f = FEEDS[key]["name_field"]
        names = sorted({it.get(name_f) for it in res["items"] if it.get(name_f)})
        df = pd.DataFrame({"name": names})
        con.register(f"_names_{key}", df)
        con.execute(f"CREATE OR REPLACE TABLE feed_{key} AS SELECT DISTINCT name FROM _names_{key}")
        con.unregister(f"_names_{key}")

    per_feed = {}
    flag_exprs = []
    for key in ok_results:
        matched_u = con.sql(f"SELECT COUNT(*) FROM universe u "
                             f"JOIN feed_{key} f ON f.name = u.item_slug").fetchone()[0]
        matched_c = con.sql(f"SELECT COUNT(*) FROM cohort c "
                             f"JOIN feed_{key} f ON f.name = c.item_slug").fetchone()[0]
        per_feed[key] = {
            "n_distinct_names": con.sql(f"SELECT COUNT(*) FROM feed_{key}").fetchone()[0],
            "matched_universe": matched_u,
            "pct_universe": round(100 * matched_u / n_universe, 1),
            "matched_cohort": matched_c,
            "pct_cohort": round(100 * matched_c / n_cohort, 1),
        }
        flag_exprs.append(f"EXISTS (SELECT 1 FROM feed_{key} f WHERE f.name = c.item_slug) AS in_{key}")

    if not ok_results:
        return {"per_feed": per_feed, "note": "no feed succeeded -- nothing to join"}

    con.execute("CREATE OR REPLACE TABLE cohort_flags AS "
                f"SELECT c.item_slug, {', '.join(flag_exprs)} FROM cohort c")

    keys = list(ok_results.keys())
    any_expr = " OR ".join(f"in_{k}" for k in keys)
    n_union = con.sql(f"SELECT COUNT(*) FROM cohort_flags WHERE {any_expr}").fetchone()[0]

    marginal = {}
    for key in keys:
        others = [k for k in keys if k != key]
        if others:
            other_expr = " OR ".join(f"in_{k}" for k in others)
            n_without = con.sql(f"SELECT COUNT(*) FROM cohort_flags WHERE {other_expr}").fetchone()[0]
        else:
            n_without = 0
        marginal[key] = {"adds_items": n_union - n_without,
                          "adds_pp_of_cohort": round(100 * (n_union - n_without) / n_cohort, 1)}

    pairwise = {}
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            both = con.sql(f"SELECT COUNT(*) FROM cohort_flags WHERE in_{a} AND in_{b}").fetchone()[0]
            pairwise[f"{a}&{b}"] = both

    return {
        "per_feed": per_feed,
        "union": {"matched_cohort": n_union, "pct_cohort": round(100 * n_union / n_cohort, 1)},
        "marginal_contribution": marginal,
        "pairwise_overlap_cohort": pairwise,
        "missing_feeds": [k for k in FEEDS if k not in ok_results],
    }


# --------------------------------------------------------------------------- #
# Semantics: is market.csgo.com's `volume` a listing count or a trade count?
# --------------------------------------------------------------------------- #

def csfloat_canary_check(ok_results: dict[str, dict], delay: float = 0.4) -> dict:
    """Fetch CSFloat's genuine daily completed-sale count for a handful of items
    spanning very liquid to illiquid, and compare against each feed's depth field.

    A listing count should sit ABOVE the single-platform daily trade count (a listing
    persists for its average dwell time, which is at least a day), and the ratio should
    vary with liquidity rather than being pinned near 1x. A trade-count field would track
    the CSFloat count roughly proportionally (same underlying activity, different market
    share) rather than showing the dwell-time-driven spread this checks for.
    """
    depth_by_name: dict[str, dict] = {}
    for key, res in ok_results.items():
        spec = FEEDS[key]
        name_f, depth_f = spec["name_field"], spec["depth_field"]
        lut = {}
        for it in res["items"]:
            nm = it.get(name_f)
            if nm in CSFLOAT_CANARIES and it.get(depth_f) is not None:
                try:
                    lut[nm] = float(it[depth_f])
                except (TypeError, ValueError):
                    pass
        depth_by_name[key] = lut

    rows = []
    session = requests.Session()
    for name in CSFLOAT_CANARIES:
        url = CSFLOAT_API.format(name=urllib.parse.quote(name, safe=""))
        try:
            r = session.get(url, headers=CSFLOAT_HEADERS, timeout=15)
        except requests.RequestException as exc:
            logger.warning("CSFloat canary %r failed: %s", name, exc)
            time.sleep(delay)
            continue
        if r.status_code != 200:
            logger.warning("CSFloat canary %r -> HTTP %s", name, r.status_code)
            time.sleep(delay)
            continue
        try:
            data = r.json()
        except ValueError:
            time.sleep(delay)
            continue
        if isinstance(data, list) and data:
            recent = data[0]
            row = {"item": name, "csfloat_daily_sales": recent.get("count")}
            for key in ok_results:
                row[f"{key}_depth"] = depth_by_name.get(key, {}).get(name)
            rows.append(row)
        time.sleep(delay)

    return {"rows": rows}


def depth_correlation(ok_results: dict[str, dict]) -> dict:
    """Cross-feed correlation of depth fields on the intersection of catalogs.

    High positive correlation across marketplaces is expected of listing-count fields
    (the same underlying item is more or less liquid everywhere); it is not guaranteed
    for a trade-count field, which depends on that specific platform's activity mix.
    """
    frames = {}
    for key, res in ok_results.items():
        spec = FEEDS[key]
        name_f, depth_f = spec["name_field"], spec["depth_field"]
        rows = []
        for it in res["items"]:
            nm = it.get(name_f)
            dv = it.get(depth_f)
            if nm is None or dv is None:
                continue
            try:
                rows.append((nm, float(dv)))
            except (TypeError, ValueError):
                continue
        frames[key] = pd.DataFrame(rows, columns=["name", key]).drop_duplicates("name")

    if len(frames) < 2:
        return {"note": "fewer than two feeds succeeded -- no correlation possible"}

    merged = None
    for key, df in frames.items():
        merged = df if merged is None else merged.merge(df, on="name", how="inner")
    if merged is None or merged.empty:
        return {"note": "no items in common across the successful feeds"}

    num = merged.drop(columns=["name"])
    pearson = num.corr(method="pearson").round(3).to_dict()
    spearman = num.corr(method="spearman").round(3).to_dict()
    log_pearson = np.log1p(num).corr(method="pearson").round(3).to_dict()
    return {"n_common_items": int(len(merged)), "pearson": pearson,
            "spearman": spearman, "log1p_pearson": log_pearson}


def distribution_shapes(ok_results: dict[str, dict]) -> dict:
    out = {}
    for key, res in ok_results.items():
        spec = FEEDS[key]
        depth_f = spec["depth_field"]
        vals = []
        for it in res["items"]:
            v = it.get(depth_f)
            if v is None:
                continue
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                continue
        if not vals:
            continue
        arr = np.array(vals)
        out[key] = {
            "n": len(arr),
            "deciles": {str(q): round(float(np.quantile(arr, q)), 1)
                        for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)},
        }
    return out


# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description="Probe four candidate supply-depth feeds")
    ap.add_argument("--out-dir", default="/tmp/supply_feeds_probe")
    ap.add_argument("--min-cohort-price", type=float, default=MIN_SERVED_PRICE_USD)
    ap.add_argument("--no-csfloat-check", action="store_true",
                     help="skip the CSFloat trade-volume cross-check (saves ~9 requests "
                          "of the shared 500/day budget)")
    ap.add_argument("--timeout", type=float, default=30.0)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    fetch_results = {}
    for key, spec in FEEDS.items():
        logger.info("Fetching %s: %s", key, spec["url"])
        res = fetch_feed(key, spec, timeout=args.timeout)
        fetch_results[key] = res
        if res["status"] == "ok":
            logger.info("  %s: OK, %d items, %d bytes, %.2fs",
                        key, res["item_count"], res["bytes"], res["wall_time_s"])
        else:
            logger.warning("  %s: FAILED status=%s %s", key, res["status"],
                           {k: v for k, v in res.items() if k not in ("feed", "items")})

    ok_results = {k: v for k, v in fetch_results.items() if v["status"] == "ok"}

    hygiene = {key: hygiene_report(key, FEEDS[key], res["items"])
               for key, res in ok_results.items()}

    con = duckdb.connect()
    n_universe, n_cohort = build_universe_and_cohort(con, args.min_cohort_price)
    logger.info("Archive universe: %d distinct item_slug; >=$%.2f cohort: %d",
                n_universe, args.min_cohort_price, n_cohort)
    coverage = coverage_and_overlap(con, ok_results, n_universe, n_cohort)

    correlation = depth_correlation(ok_results)
    distributions = distribution_shapes(ok_results)

    csfloat_check = None
    if not args.no_csfloat_check:
        logger.info("Running CSFloat canary cross-check (%d items)...", len(CSFLOAT_CANARIES))
        csfloat_check = csfloat_canary_check(ok_results)

    report = {
        "fetch": {k: {kk: vv for kk, vv in v.items() if kk != "items"}
                  for k, v in fetch_results.items()},
        "hygiene": hygiene,
        "archive_universe": n_universe,
        "cohort_min_price": args.min_cohort_price,
        "cohort_size": n_cohort,
        "coverage_and_overlap": coverage,
        "depth_field_correlation": correlation,
        "depth_field_distributions": distributions,
        "csfloat_canary_check": csfloat_check,
    }
    (out / "supply_feeds_report.json").write_text(json.dumps(report, indent=2, default=str))

    for key, res in ok_results.items():
        pd.DataFrame(res["items"]).to_parquet(out / f"{key}_items.parquet", index=False)

    print("\n" + "=" * 78)
    print(json.dumps({k: v for k, v in report.items()}, indent=2, default=str))
    print("=" * 78)
    print(f"\nFull report -> {out / 'supply_feeds_report.json'}")

    n_ok = len(ok_results)
    n_failed = len(FEEDS) - n_ok
    print(f"\n{n_ok}/{len(FEEDS)} feeds fetched OK, {n_failed} failed.")
    if n_failed:
        print("Failed feeds (see 'fetch' in the report for status/reason):",
              [k for k in FEEDS if k not in ok_results])
    return 0 if n_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
