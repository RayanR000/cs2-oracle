# GitHub Automation Setup

## Step 1: Create GitHub Secrets

Go to your GitHub repository: https://github.com/RayanR000/cs2-market-analyzer

1. Click **Settings** → **Secrets and variables** → **Actions**
2. Click **New repository secret** and add:

### Required Secrets

**Secret 1: SUPABASE_DATABASE_URL**
- **Value**: `postgresql://postgres.nfymkyqebfvubzsfyfuc:Tc_5q@qwAEM$6z3@aws-1-us-east-1.pooler.supabase.com:5432/postgres`
- This is your Supabase connection string

**Secret 2: STEAM_API_KEY** (if you have one)
- **Value**: Your Steam API key
- Get it from: https://steamcommunity.com/dev/apikey

**Secret 3: CS2SH_API_KEY** (if you have one)
- **Value**: Your CSGOTrader API key
- Get it from: https://csgotrades.net/api

## Step 2: Enable GitHub Actions

1. Go to **Settings** → **Actions** → **General**
2. Ensure **Actions is enabled**
3. Under "Workflow permissions", select **Read and write permissions**

## Step 3: Verify the Workflow

1. Go to **Actions** tab
2. You should see "Daily Market Data Collection"
3. To test it, click **Run workflow** → **Run workflow**

## How It Works

- ✅ Runs automatically every day at 2 AM UTC
- ✅ Collects latest market prices from Steam, Skinport, CSGOTrader, etc.
- ✅ Updates your Supabase database automatically
- ✅ Can be triggered manually anytime via GitHub Actions

## Scheduling

To change the collection time, edit `.github/workflows/daily-collection.yml`:

```yaml
schedule:
  - cron: '0 2 * * *'  # 2 AM UTC daily
  # Alternative examples:
  # '0 10 * * *'  → 10 AM UTC
  # '0 */6 * * *' → Every 6 hours
```

## Monitoring

Check collection status:
1. Go to **Actions** tab
2. Click "Daily Market Data Collection"
3. View logs for each run
