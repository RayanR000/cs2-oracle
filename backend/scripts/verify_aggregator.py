import sys
import logging
from pathlib import Path
import requests
import time
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from collectors.csgotrader_aggregator import CSGOTraderAggregator
from collectors.steam_market import SteamMarketCollector

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("verifier")

def verify_prices():
    test_skins = [
        "AK-47 | Slate (Field-Tested)",
        "AWP | Atheris (Minimal Wear)",
        "Glock-18 | Candy Apple (Factory New)",
        "Desert Eagle | Mecha Industries (Field-Tested)",
        "M4A4 | Spider Lily (Minimal Wear)"
    ]

    print("\n" + "="*60)
    print(f"PRICE VERIFICATION: AGGREGATOR VS STEAM LIVE")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60 + "\n")

    aggregator = CSGOTraderAggregator()
    steam = SteamMarketCollector(rate_limit_delay=5.0)

    print("Fetching Aggregator data (Batch)...")
    agg_results = aggregator.collect_batch_items(test_skins)

    print("Fetching Steam Live data (Individual)...")
    
    for skin in test_skins:
        agg_data = agg_results.get(skin)
        agg_price = agg_data[0] if agg_data else 0.0
        
        # Fetch live from Steam
        steam_data = steam.get_price_trend(skin)
        steam_price = steam_data['lowest_price'] if steam_data else 0.0
        
        if agg_price > 0 and steam_price > 0:
            diff = abs(agg_price - steam_price)
            diff_pct = (diff / steam_price) * 100
            status = "✅ EXCELLENT" if diff_pct < 2 else "🟡 GOOD" if diff_pct < 5 else "❌ DISCREPANCY"
            
            print(f"SKIN: {skin}")
            print(f"  → Aggregator: ${agg_price:.2f}")
            print(f"  → Steam Live: ${steam_price:.2f}")
            print(f"  → Difference: ${diff:.2f} ({diff_pct:.2f}%) - {status}")
        else:
            print(f"SKIN: {skin}")
            print(f"  → Error: Could not retrieve data from one of the sources.")
        
        print("-" * 30)
        time.sleep(2) # Extra safety for Steam

if __name__ == "__main__":
    verify_prices()
