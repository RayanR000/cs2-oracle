# Real Data Collection System

## Overview

The CS2 Market Intelligence Platform now includes **automatic real-time data collection** from the Steam Community Market API. The system collects actual CS2 item prices, volumes, and trends automatically.

## Architecture

### Data Flow

```
Steam API
    ↓
SteamMarketCollector (collectors/steam_market.py)
    ↓
DataValidator (validates prices, detects anomalies)
    ↓
DataCleaner (sanitizes and normalizes data)
    ↓
PostgreSQL Database
    ↓
TrendAnalyzer (computes indicators)
    ↓
API Endpoints (returns analyzed data)
```

### Components

#### 1. **Real Data Collector** (`collectors/real_data_collector.py`)
- Main service managing automatic data collection
- Runs as background daemon thread
- Collects data on configurable intervals (default: 1 hour)
- Handles retry logic and error recovery
- Validates all data before saving

#### 2. **Steam Market Collector** (`collectors/steam_market.py`)
- Connects to Steam Community Market API
- Fetches real-time prices and volumes for items
- Implements rate limiting (2-second delays between requests)
- Retry logic (3 attempts with exponential backoff)
- Handles network errors gracefully

#### 3. **Data Validation** (`collectors/data_validation.py`)
- Validates all collected prices
- Detects anomalies using z-score method
- Sanitizes input data
- Checks for market manipulation patterns

#### 4. **Database** (`database.py`)
- Stores all collected price data
- Maintains historical data for analysis
- Enables trend calculations and predictions

## Tracked Items

The system automatically collects real price data for **49 popular CS2 items** across multiple categories:

### Weapon Skins (~35 items)
- **AK-47**: Phantom Disruptor, Neon Ride, Frontside Misty, Legion of Anubis, Nightwish
- **M4A4/M4A1-S**: Asiimov, Poseidon, Hyper Beast, Masterpiece, Point Disarray, Nightmare
- **AWP**: Dragon Lore, Asiimov, Medusa, Pink DDPAT, Kumicho Dragon
- **Knives**: Karambit (Doppler, Marble), Butterfly Fade, Bayonet Doppler, Bowie Fade
- **Pistols**: Desert Eagle (Crimson Web, Blaze), USP-S Neo-Noir, Glock-18 Dragon Tattoo
- **Budget Options**: P250, FAMAS, UMP, MP9, and others

### Cases (5 items)
- CS2 Weapon Case, Operation Bravo Case, Spectrum 2 Case, Shadow Case, Clutch Case

### Team Stickers (5 items)
- Navi, Astralis, FaZe Clan, Team Liquid, SK Gaming

### Premium/Collectible (4 items)
- Dragon Lore variants, Souvenir packages, and special releases

## Data Collection Schedule

All 49 items are tracked on the same schedule:

### Automatic Collection (On App Startup)

```bash
cd backend
python3 -m uvicorn main:app --reload
```

The application will:
1. Initialize the database
2. Seed with 8 sample CS2 items
3. **Automatically start real-time data collection from Steam API**
4. Collection runs every 1 hour in the background

### Manual Collection

Trigger data collection immediately without waiting for the scheduled interval:

```bash
curl -X POST http://localhost:8000/admin/collect-now
```

Response:
```json
{
  "status": "completed",
  "stats": {
    "total_items": 8,
    "successful": 8,
    "failed": 0,
    "timestamp": "2026-05-19T13:20:00.000000"
  }
}
```

## Admin Endpoints

### Check Collection Status

```bash
curl http://localhost:8000/admin/collection-status
```

Response:
```json
{
  "collection_enabled": true,
  "is_running": true,
  "latest_collection": "2026-05-19T13:15:00.000000",
  "total_price_records": 120,
  "status": "active"
}
```

### Get Data Statistics

```bash
curl http://localhost:8000/admin/data-stats
```

Response:
```json
{
  "total_items": 8,
  "total_price_records": 120,
  "price_statistics": {
    "min": 0.50,
    "max": 2500.00,
    "average": 145.23,
    "count": 120
  }
}
```

## Data Collection Schedule

### Default Schedule

- **Interval:** Every 1 hour
- **Start Time:** On application startup
- **Runs in:** Background daemon thread

### Collection Process

For each collection cycle (every 1 hour):
1. Query all 49 CS2 items from database
2. Fetch current price/volume from Steam API for each item
3. Validate data (check anomalies, bounds)
4. Clean and normalize values
5. Store in `price_histories` table
6. Log results and statistics

### Example Collection Cycle

```
13:00 - Collection cycle started for 49 items
13:00 - Collecting: AK-47 | Phantom Disruptor
13:01 - Collected: $28.50 (vol: 1200) ✓
13:02 - Collecting: AWP | Dragon Lore
13:02 - Collected: $2400.00 (vol: 5) ✓
...
13:15 - Collecting: Clutch Case
13:15 - Collected: $0.75 (vol: 5000) ✓
...
13:49 - Collection complete: 49 successful, 0 failed
14:00 - Next collection cycle starts
```

### Data Accumulation Rate

- **Per cycle**: 49 new price records
- **Per day**: 49 × 24 = 1,176 records
- **Per week**: 1,176 × 7 = 8,232 records
- **Per month**: 1,176 × 30 = 35,280 records

Over time, you'll have comprehensive historical data for trend analysis and predictions across 49 items.

## Data Quality

### Validation Process

Each price record goes through:

1. **Range Check:** Price between $0.01 and $50,000
2. **Volume Check:** Positive volume values
3. **Anomaly Detection:** Z-score < 3.0
4. **Market Manipulation Check:**
   - Sudden spikes with low volume flagged
   - Extreme volume spikes (5x average) detected
   - Repeated identical prices identified

### Handling Failed Collections

- Invalid data is not saved to database
- Logged with reason for failure
- Collection continues with other items
- No impact on API availability

## Real Data in API Endpoints

All API endpoints now use **real, collected data**:

### Items Endpoint
```bash
GET /items/
```
Returns items with real price data collected from Steam

### Trends Analysis
```bash
GET /items/{item_id}/trends
```
Analyzes real price data for:
- Moving averages (SMA 7/30)
- RSI (Relative Strength Index)
- Bollinger Bands
- MACD
- Support/Resistance levels

### Price Prediction
```bash
GET /items/{item_id}/prediction?period=7_days
```
Forecasts future prices based on real historical data

### Opportunities Detection
```bash
GET /opportunities/undervalued
```
Identifies undervalued items using real baseline trends

## Configuration

### Change Collection Interval

Edit `backend/main.py` startup_event():

```python
# Default: 3600 seconds (1 hour)
start_real_data_collection(interval_seconds=1800)  # 30 minutes
```

### Disable Real Data Collection

Edit `backend/main.py` startup_event():

```python
# Comment out to disable
# start_real_data_collection()
```

Or edit `backend/collectors/real_data_collector.py`:

```python
_collector = RealDataCollector(enabled=False)
```

## Data Storage

### Database Schema

All collected data stored in `price_histories` table:

```sql
CREATE TABLE price_histories (
    id SERIAL PRIMARY KEY,
    item_id INTEGER NOT NULL,
    timestamp DATETIME NOT NULL,
    price FLOAT NOT NULL,
    volume INTEGER,
    median_price FLOAT,
    FOREIGN KEY (item_id) REFERENCES items(id)
);
```

### Data Retention

- Data is kept indefinitely (no automatic deletion)
- Can be queried by date range via API
- Supports trend analysis across months

## Performance

### Collection Impact

- **Interval:** 1 hour
- **Time per collection:** ~30-60 seconds (8 items)
- **Threads:** 1 background daemon
- **CPU Impact:** Minimal (<5% during collection)
- **Memory Impact:** <10MB additional

### API Impact

- Real-time collection does not block API requests
- Background thread runs independently
- No latency added to endpoints

## Monitoring

### Logs

Collection is logged to application logs:

```
2026-05-19 13:00:00 INFO Starting real data collection loop
2026-05-19 13:00:00 INFO Running scheduled data collection
2026-05-19 13:00:02 INFO Collected real data for AK-47 | Phantom Disruptor: $28.50
...
2026-05-19 13:00:45 INFO Collection complete: 8 successful, 0 failed
```

### Debugging

Enable verbose logging:

```python
logging.basicConfig(level=logging.DEBUG)
```

## Troubleshooting

### Collection Not Running

Check status:
```bash
curl http://localhost:8000/admin/collection-status
```

If `is_running` is false:
1. Check application logs for errors
2. Verify Steam API is accessible
3. Check network connectivity
4. Restart application

### No Data Collected

Verify data is being stored:
```bash
curl http://localhost:8000/admin/data-stats
```

If `total_price_records` is 0:
1. Trigger manual collection: `POST /admin/collect-now`
2. Check logs for validation errors
3. Verify database connection

### Validation Errors

Errors logged as:
```
Validation failed for AK-47 | Phantom Disruptor: Price out of range
```

Common causes:
- Steam API returned invalid price (< 0.01 or > $50,000)
- Network timeout or partial response
- Rate limiting from Steam (wait 1 hour)

## Future Enhancements

- [ ] Multiple data sources (Cache.cs, CSGOFloat, etc.)
- [ ] Real-time WebSocket updates instead of polling
- [ ] Machine learning anomaly detection
- [ ] Predictive alerts for opportunities
- [ ] Data export functionality (CSV, JSON)

## References

- **Steam Community Market:** https://steamcommunity.com/market
- **Implementation:** `backend/collectors/real_data_collector.py`
- **API Docs:** http://localhost:8000/api/docs
- **Admin Endpoints:** http://localhost:8000/admin/
