# 2026-10-09 — pg_trgm gets a migration; dead workflow work removed

Performance review 2026-10-08, the `pg_trgm` correctness finding and §5.

## pg_trgm

`api/routes/market.py` calls `similarity()` on every `/market?q=` search: in the `WHERE`
for queries of 3+ characters, and in the `ORDER BY` for all of them. No migration created
pg_trgm. Prod has it (verified read-only on 10-09: pg_trgm 1.6 in `public`,
`similarity('AK-47 Redline','redline')` = 0.571), so search works today, but it was
installed by hand: a database rebuilt from the migrations would error on every search.

- `0029_create_pg_trgm_extension` runs `CREATE EXTENSION IF NOT EXISTS pg_trgm` (Postgres
  only). Prod was at 0028; the Aggregator's `migrate` step applies 0029 on its next run,
  where it is a no-op.
- The schema-drift check now calls `similarity()` on its scratch database after
  `upgrade head`. `compare_metadata` does not see extensions, so the check could not have
  caught this.

## Workflow work removed

- **Drift detection step** (`price-forecast.yml`). It needs `evidently`, which only the
  `mlops` extra installs, so it raised `ModuleNotFoundError` every day behind
  `continue-on-error` (run `37877462648`). Its reports landed on the ephemeral runner, so
  nothing could read them. `monitoring/drift.py` and `scripts/run_drift_check.py` stay for
  local use.
- **Voted-frame Actions cache** (`price-forecast.yml`). `_voted_cache_key` hashes the cutoff
  date and the archive fingerprint, and both change daily. Run `37877462648` restored 53 MB,
  rebuilt both voted frames anyway, and saved nothing back because the `hashFiles` key was
  an exact hit. It could never hit.
- **Backtest weekday cron** (`backtest-accuracy.yml`). A new `gate` job ends a scheduled run
  early when a chained (`workflow_run`) Backtest already succeeded that UTC day. The cron
  still runs as the fallback when the chain broke, and also when the API call fails. A chained run
  after a failed forecast still skips every job and concludes `skipped` (49 of the last 100
  chained runs), so it cannot satisfy the gate. Saves ~25 runner-min on chain-green days.
- **Unused dependencies.** `beautifulsoup4`, `apscheduler`, `joblib`, `huggingface_hub`,
  `onnxruntime` and `transformers` are imported nowhere, since `collectors/social_sentiment.py`
  is gone. Relocking also drops flatbuffers, hf-xet, safetensors, soupsieve, tokenizers and
  tzlocal, which are not imported either. joblib stays in the lock as a transitive dependency.
- **Supply step timeout** (`aggregator-update.yml`): `timeout-minutes: 15` on the supply +
  volume step (the lis-skins ladder takes ~7 min). The step is `continue-on-error`, so a hang
  now costs that day's supply snapshot rather than the whole run.

## Not done

The review's other §5 item, running supply/volume alongside aggregate/append, is not done.
Supply and volume already run in parallel with each other, and the combined step took 74 s on 10-09.
