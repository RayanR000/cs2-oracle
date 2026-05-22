# Historical Data Import Guide

Your database now has infrastructure to import and analyze historical CS:GO and CS2 price data.

## Current Status

✅ **What You Have:**
- 17,752 items (CS:GO + CS2 skins, stickers, cases, etc.)
- 79 events (game updates, majors, operations from Dec 2013 - May 2026)
- Price data from May 21-22, 2026

❌ **What's Missing:**
- Historical prices (pre-May 2026)
- Needed for analyzing price impact of past events

## To Import Historical Prices

### 1. Gather Data
Find historical Steam Community Market prices from:
- **BitSkins API** (recommended - 2015+)
- **Wayback Machine** (free - 2013+)
- **Kaggle** (community datasets)
- **GitHub** (archived project data)

See `backend/data/HISTORICAL_DATA_SOURCES.md` for detailed sources.

### 2. Format Data
Organize data as CSV or JSON using templates:
- `backend/data/historical_prices_template.csv`
- `backend/data/historical_prices_template.json`

### 3. Import
```bash
cd backend
source venv/bin/activate

# CSV import
python scripts/import_historical_prices.py --source csv --file data/your_data.csv

# JSON import
python scripts/import_historical_prices.py --source json --file data/your_data.json
```

## Tools Available

**Data Management:**
- `scripts/import_historical_prices.py` - Import CSV/JSON price data
- `scripts/manage_events.py` - Add/edit/delete events
- `scripts/import_events.py` - Bulk import events

**Data Analysis:**
- `scripts/analyze_events_impact.py` - Correlate events with price movements

## Example Workflow

```bash
# 1. Get BitSkins historical data for 2023
# (download/export to data/prices_2023.csv)

# 2. Import to database
python scripts/import_historical_prices.py --source csv --file data/prices_2023.csv

# 3. Analyze impact
python scripts/analyze_events_impact.py --days 365

# 4. Build your model using full historical context
```

## Next Steps

1. Determine which time period you need (2015+ recommended)
2. Choose data source (BitSkins is best if available)
3. Download/export data in CSV or JSON format
4. Use import script to load into database
5. Verify data with analysis scripts

Once you have historical prices loaded, you'll be able to:
- Correlate major tournaments with price spikes
- Analyze impact of balance updates
- Identify seasonal trends
- Train prediction models on complete market history
