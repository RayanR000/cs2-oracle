# Price Data Archive

Daily gzipped CSV dumps of live-collected prices, written by the
`aggregator-update.yml` workflow after each collection run.

Layout: `price-archive/YYYY/MM/prices-YYYY-MM-DD.csv.gz`

Snapshot-tier items (no CSMarketAPI historical series) keep only their
latest price row in Supabase, so these files are the only durable record
of their daily price history.
