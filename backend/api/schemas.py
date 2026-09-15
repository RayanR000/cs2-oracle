import re
from datetime import datetime

from pydantic import BaseModel, model_validator


class ItemOut(BaseModel):
    id: int
    item_id: str
    name: str
    type: str
    icon_url: str | None = None
    release_date: datetime | None = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class TrendingItemOut(BaseModel):
    id: int
    item_id: str
    name: str
    type: str
    icon_url: str | None = None
    latest_price: float

    class Config:
        from_attributes = True


class PricePointOut(BaseModel):
    timestamp: datetime
    price: float
    volume: int | None = None
    median_price: float | None = None
    sma_7: float | None = None
    sma_30: float | None = None

    class Config:
        from_attributes = True


class TrendAnalysisOut(BaseModel):
    item_id: int
    item_name: str
    current_price: float
    # WITHHELD 2026-09-10: always "neutral". The direction call is
    # anti-informative, not merely null -- see serving_policy.served_direction.
    trend_direction: str
    # `confidence` withdrawn 2026-08-12: the served label is uncalibrated and
    # carries no measurable directional information. See PredictionOut below.
    explanation: str
    rsi: float | None = None
    bollinger_upper: float | None = None
    bollinger_middle: float | None = None
    bollinger_lower: float | None = None
    macd: float | None = None
    macd_signal: float | None = None
    support: float | None = None
    resistance: float | None = None
    factors: list[str] = []
    sma_7: float | None = None
    sma_30: float | None = None

    class Config:
        from_attributes = True


class PredictionOut(BaseModel):
    item_id: int
    item_name: str
    current_price: float
    forecast_low: float
    forecast_mid: float
    forecast_high: float
    forecast_period: str
    # WITHHELD 2026-09-10: always "neutral", at every horizon. Pesaran-
    # Timmermann on the >=$1 `lgbm-v3%` panel is NEGATIVE at all four horizons
    # (-1.62pp p=0.032 / -0.66 / -0.54 / -1.31 at 3/7/14/30d), directional
    # accuracy is below a coin flip everywhere (43.8-48.9%), and a constant
    # "down" call beats the model at every horizon. h=3 (>=$1) and h=7 (>=$20,
    # -1.89pp p=0.037) are significant-negative. The field is kept on the
    # schema because the frontend consumes it and removal is breaking; the
    # `direction` column is still written and still scored. Republish only once
    # a panel shows positive PT. See serving_policy.served_direction.
    trend_direction: str
    # `confidence` is WITHDRAWN, 2026-08-12. It was a bare `>= 0.5` cut on the
    # directional classifier's max class probability, never calibrated against
    # realised hits. Measured within each forecast date on the >=$1 `lgbm-v3%`
    # panel, on the 11 cells with n_high >= 30: the "high" cohort is right
    # 29.0-45.8% of the time — every cell below a coin flip, against a stated
    # target of 80% — and its gap to "low" runs -8.26 to +4.65pp with mixed
    # sign, so the tag carries no ordering either. Consumers weight on it, so
    # an uninformative tag is worse than none. The column is still written and
    # still scored as `conf_gap_pp` — republish only once that is calibrated.
    # See docs/changelog/2026-08-12-served-confidence-withdrawn.md.
    # Did the quote this forecast was built from equal its own local median?
    # The model reaches rank IC +0.13-0.17 at 3/7/14d where it did and ~0 or
    # negative where it did not, so a False here says the forecast is served
    # from a cohort with no measured ordering skill. Those items are kept OFF
    # /opportunities entirely; a lookup by name still gets its forecast, with
    # this flag. None = not recorded (rows predating the 2026-08-11 column).
    anchor_clean: bool | None = None
    anchor_wedge_pct: float | None = None
    # Every item is served a forecast; sub-$1 items are flagged not economically
    # tradeable rather than withheld. Derived from current_price below so every
    # route that builds a PredictionOut is correct by construction.
    # `est_roundtrip_cost_pct` is the round trip at the cheapest venue plus the
    # item's tier spread -- the move a forecast must beat to imply a trade.
    tradeable: bool | None = None
    est_roundtrip_cost_pct: float | None = None
    # Volatility/stability tags (v1). Derived from the served band + exceed_p; no
    # new modelling. `expected_swing_pct` is the calibrated half-band as a
    # fraction of the mid ("+/-X% over the horizon"). `move_odds` is the
    # magnitude signal P(move > round-trip cost), `exceed_p` through the
    # isotonic calibrator (raw head output on artifacts predating it); NULL
    # on artifacts predating the exceedance head. `stability_label` ranks the
    # swing against the within-horizon >=$1 universe tertiles, so it is
    # relative-to-peers, not an absolute cutoff. See api/volatility_tags.py.
    expected_swing_pct: float | None = None
    move_odds: float | None = None
    stability_label: str | None = None
    # P(|return_h| > 2σ_item) from the anomaly head. Served at 3/7/14d only —
    # the 30d head ranks but ties the featureless null on log loss, so it is
    # withheld (None) until calibrated. See ANOMALY_SERVED_HORIZONS.
    anomaly_p: float | None = None

    @model_validator(mode="after")
    def _derive_tradeability(self) -> "PredictionOut":
        from api.serving_policy import tradeability

        t = tradeability(self.current_price)
        self.tradeable = t.tradeable
        self.est_roundtrip_cost_pct = t.est_roundtrip_cost_pct
        return self


class VolatilityRankOut(BaseModel):
    """One row of the /items/volatility ranking. See api/volatility_tags.py."""

    item_id: str
    name: str
    current_price: float
    expected_swing_pct: float
    move_odds: float | None = None
    stability_label: str


class OpportunityOut(BaseModel):
    item_id: int
    item_name: str
    current_price: float
    opportunity_type: str
    opportunity_score: float
    reason: str
    current_trend: str

    class Config:
        from_attributes = True


class SourcePriceOut(BaseModel):
    timestamp: datetime
    price: float
    volume: int | None = None
    median_price: float | None = None

    class Config:
        from_attributes = True


class MultiSourcePricesOut(BaseModel):
    item_id: str
    name: str
    sources: list[str]
    data: dict[str, list[SourcePriceOut]]


class EventOut(BaseModel):
    id: int
    type: str
    timestamp: datetime
    description: str
    created_at: datetime

    class Config:
        from_attributes = True


class QualityVariantOut(BaseModel):
    item_id: str
    name: str
    quality: str
    current_price: float | None = None
    price_change_24h: float | None = None
    volume_24h: int | None = None

    class Config:
        from_attributes = True


class GroupedMarketItemOut(BaseModel):
    base_name: str
    type: str
    icon_url: str | None = None
    price_avg: float | None = None
    price_min: float | None = None
    price_max: float | None = None
    price_change_24h: float | None = None
    volume_24h: int | None = None
    quality_count: int = 1
    qualities: list[QualityVariantOut] = []


def parse_item_name(name: str):
    """Extract base name and quality from a full item name.

    Examples:
        'AK-47 | Redline (Field-Tested)' -> ('AK-47 | Redline', 'Field-Tested')
        'StatTrak™ M4A4 | Desolate (FN)' -> ('StatTrak™ M4A4 | Desolate', 'FN')
        'Sticker | Dragon' -> ('Sticker | Dragon', None)
    """
    match = re.match(r"^(.+?)\s*\(([^)]+)\)\s*$", name)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return name, None


class UserOut(BaseModel):
    id: int
    steam_id: str
    username: str | None = None
    avatar_url: str | None = None

    class Config:
        from_attributes = True


class EventImpactOut(BaseModel):
    event_id: int
    event_type: str
    event_description: str
    event_timestamp: datetime
    price_day_before: float | None = None
    price_day_1: float | None = None
    price_day_3: float | None = None
    price_day_7: float | None = None
    impact_pct_1day: float | None = None
    impact_pct_3day: float | None = None
    impact_pct_7day: float | None = None
    peak_impact_pct: float | None = None
    peak_impact_day: int | None = None
    duration_days: int | None = None
    z_score: float | None = None
    confidence_score: float | None = None

    class Config:
        from_attributes = True


class FeatureImportanceItem(BaseModel):
    feature: str
    importance: float


class FeatureImportanceOut(BaseModel):
    item_id: str
    item_name: str
    horizons: dict[str, list[FeatureImportanceItem]]


class SocialMentionOut(BaseModel):
    post_id: str
    subreddit: str | None = None
    post_title: str | None = None
    post_score: int | None = None
    sentiment_score: float | None = None
    mentioned_at: datetime

    class Config:
        from_attributes = True


class SocialSentimentSummaryOut(BaseModel):
    item_id: str
    item_name: str
    mentions_24h: int
    mentions_7d: int
    mention_velocity: float
    avg_sentiment_7d: float
    avg_score_7d: float
    recent_mentions: list[SocialMentionOut] = []


class HealthOut(BaseModel):
    status: str
    version: str
    environment: str
