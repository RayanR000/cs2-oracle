# Historical CS:GO/CS2 Price Data Sources

This guide helps you find and import historical Steam Community Market price data.

## Available Data Sources

### 1. **Third-Party APIs with Historical Data**

#### BitSkins Market API
- **Coverage**: 2015-present
- **Items**: All CS:GO/CS2 skins, stickers, cases
- **API**: https://api.bitskins.com
- **Data**: Historical prices, volume, trends
- **Access**: Requires API key
- **Frequency**: Daily snapshots available

```bash
# Example: Get historical prices
curl "https://api.bitskins.com/api/v1/market/price_data?app_id=730&\
  time_period=1m&\
  start_time=1609459200&\
  end_time=1640995200"
```

#### CSGOFloat Market Data
- **Coverage**: 2016-present
- **Items**: Floated skins with quality data
- **Website**: https://csgofloat.com
- **Data**: Individual sales, rarity data
- **Access**: Some historical data available
- **Note**: Requires scraping or API access

#### Steam Community Market API
- **Coverage**: Limited (no official historical API)
- **Items**: All items still on market
- **Note**: Steam doesn't provide historical price API
- **Workaround**: Use Wayback Machine for snapshots

### 2. **Web Scraping & Wayback Machine**

#### Internet Archive (Wayback Machine)
- **Coverage**: 2013-present (depends on snapshots)
- **URL**: https://web.archive.org
- **Data**: Historical snapshots of Steam Market pages
- **Example**: 
  ```
  https://web.archive.org/web/20200101000000*/steamcommunity.com/market/listings/730/AK-47*
  ```

#### CSGOTracker / Alternative Sites
- Some third-party trackers saved historical data
- May have snapshots from 2014-2023
- Access varies by site

### 3. **Public Datasets**

#### Kaggle Datasets
- Search: "CS:GO price history" or "Counter-Strike market"
- May have community-collected data
- Usually CSV format
- Quality varies

#### GitHub Repositories
- Search: "csgofloat" or "csgo-prices" 
- Some projects archived historical data
- Usually available as JSON or CSV

### 4. **Manual Collection Methods**

#### CSGOTracker Historical Export
- If available through their site
- Usually month-by-month exports
- CSV or JSON format

#### Price Tracking Services
- Websites like:
  - FloatDB.net
  - CSGOFloat.com
  - Market trackers (may have exports)

## Data Format Requirements

Your import script expects data in CSV or JSON format.

### CSV Format
```csv
item_name,timestamp,price,volume,source
AK-47 | Phantom Disruptor,2023-01-15T10:00:00,3.50,1200,bitskins
M4A4 | Uncharted,2023-01-15T10:00:00,4.25,850,steam_market
AWP Dragon Lore,2023-01-15T10:00:00,2000.00,5,csgo_float
```

### JSON Format
```json
{
  "prices": [
    {
      "item_name": "AK-47 | Phantom Disruptor",
      "timestamp": "2023-01-15T10:00:00",
      "price": 3.50,
      "volume": 1200,
      "source": "bitskins"
    },
    {
      "item_name": "M4A4 | Uncharted",
      "timestamp": "2023-01-15T10:00:00",
      "price": 4.25,
      "volume": 850,
      "source": "steam_market"
    }
  ]
}
```

## Steps to Import Historical Data

### 1. Obtain Data
- Contact BitSkins for API access
- Scrape Wayback Machine snapshots
- Download from Kaggle or GitHub
- Use alternative price tracking APIs

### 2. Format Data
- Convert to CSV or JSON format above
- Ensure all timestamps are ISO format (YYYY-MM-DDTHH:MM:SS)
- Verify item names match your database

### 3. Validate Item Names
```bash
# Check if items exist in database
cd backend
source venv/bin/activate
python << 'EOF'
from database import SessionLocal, Item

db = SessionLocal()
# Check a sample item
item = db.query(Item).filter(Item.name == "AK-47 | Phantom Disruptor").first()
if item:
    print(f"Found: {item.name}")
else:
    print("Item not found - check spelling")
db.close()
EOF
```

### 4. Import Data
```bash
# CSV import
python scripts/import_historical_prices.py --source csv --file data/historical_prices.csv

# JSON import
python scripts/import_historical_prices.py --source json --file data/prices.json
```

### 5. Verify Import
```bash
# Check records imported
sqlite3 backend/cs2.db "SELECT COUNT(*) FROM price_history"

# Check date range
sqlite3 backend/cs2.db "SELECT MIN(timestamp), MAX(timestamp) FROM price_history"
```

## Example: Getting Data from BitSkins

```python
import requests
from datetime import datetime

API_KEY = "your_bitskins_api_key"

# Get historical data for a specific item
response = requests.get(
    "https://api.bitskins.com/api/v1/market/price_data",
    params={
        "app_id": 730,  # CS:GO/CS2
        "market_hash_name": "AK-47 | Phantom Disruptor",
        "time_period": "7d",  # 7 days
        "api_key": API_KEY
    }
)

data = response.json()
# Process and save to CSV/JSON
```

## Recommended Approach

1. **Start with BitSkins** if you can get API access
   - Most comprehensive historical data
   - 2015-present coverage
   - Structured API

2. **Supplement with Wayback Machine** for gaps
   - Free access
   - Fill gaps in BitSkins data
   - Monthly snapshots work well

3. **Validate with CSGOFloat**
   - Cross-reference prices
   - Fill in souvenir/special editions
   - Verify data quality

## Timeline Recommendations

For meaningful analysis, you want data from:
- **2019+** - Minimum (post-ranked changes)
- **2015+** - Ideal (covers most major events)
- **2013+** - Comprehensive (covers all majors)

Current status:
- Have: May 21-22, 2026 (current only)
- Need: Historical data 2015-2026

## Questions?

If you need help:
1. Check data format examples above
2. Verify item names match database
3. Test with small sample first (100 records)
4. Run validation before full import
