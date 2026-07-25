# Quality-Spread / Cross-Wear Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add ~13 cross-variant "quality-spread" features (wear ladder, StatTrak premium, Souvenir premium) behind a default-off flag, so an A/B + permutation experiment can measure whether they improve directional forecast accuracy.

**Architecture:** A new `_add_quality_spread_features` method on `ItemForecaster` computes, for each item, how its price sits relative to its same-date sibling variants, plus the trailing deviation and change of that relationship. Features are computed only when `QUALITY_SPREAD=1`, which also appends the new `quality_spread` group to the feature allowlist (currently `["price_technicals"]`) so the columns actually reach the model. Grouping is done via same-date `groupby` (leak-safe, identical to existing cross-sectional features); deviations/changes use only each item's trailing history.

**Tech Stack:** Python 3.11, pandas, numpy, LightGBM, pytest. All work is in `backend/`.

## Global Constraints

- Class under change is `ItemForecaster` in `backend/models/forecaster.py` (not `PriceForecaster`).
- Item name parsing uses the existing `parse_item_name` from `backend/models/item_parser.py`, which returns `weapon`, `skin_name`, `quality`, `quality_rank`, `is_stattrak`, `is_souvenir`.
- Feature grouping is done by the module-level `_feature_group(name)` function (forecaster.py:78); the allowlist filter at forecaster.py:2039 keeps only groups in `FEATURE_GROUP_ALLOWLIST` (currently `["price_technicals"]`).
- With the flag off, behavior must be byte-identical to current production: no new columns, allowlist unchanged.
- Leak-safety rule: cross-item aggregates use **same-date** sibling prices only; deviations/changes use only trailing per-item history.
- All new features fill to `0.0` where an item has no sibling on that axis; per-axis indicator flags (`has_wear_siblings`, `has_stattrak_pair`, `has_souvenir_pair`) tell the model which rows are real.
- Run `pytest` and `python3 -m py_compile models/forecaster.py` for every change (per AGENTS.md). All commands run from `backend/`.
- The full suite is currently 179 passing; it must stay green with the flag off.
- Model version bump (`lgbm-v3 → lgbm-v4`) and flipping the flag default happen ONLY if the experiment clears the gate — not in this plan.

---

### Task 1: Feature group + flag plumbing

Adds the `quality_spread` group router branch, the `ENABLE_QUALITY_SPREAD` flag, and the effective-allowlist logic that appends the group when the flag is set. No feature computation yet — this is the scaffolding the later tasks slot into.

**Files:**
- Modify: `backend/models/forecaster.py` (function `_feature_group` at :78; class constant near :186; allowlist use at :2039)
- Test: `backend/tests/test_forecaster.py`

**Interfaces:**
- Produces: module fn `_feature_group(name: str) -> str` returns `"quality_spread"` for names prefixed `wear_`, `stattrak_`, `souvenir_`, `has_wear`, `has_stattrak`, `has_souvenir`.
- Produces: `ItemForecaster.ENABLE_QUALITY_SPREAD: bool` class constant (default `False`).
- Produces: method `ItemForecaster._quality_spread_enabled(self) -> bool` — `True` iff the class constant is set OR `os.environ.get("QUALITY_SPREAD") == "1"`.
- Produces: method `ItemForecaster._effective_allowlist(self) -> list[str] | None` — returns `FEATURE_GROUP_ALLOWLIST` unchanged when the flag is off or the allowlist is falsy; otherwise returns the allowlist with `"quality_spread"` appended (no duplicates).

- [ ] **Step 1: Write the failing test**

```python
class TestQualitySpreadPlumbing:
    def test_feature_group_routes_quality_spread(self):
        from models.forecaster import _feature_group
        for name in ["wear_spread_ratio", "wear_spread_ratio_z60",
                     "wear_ladder_dispersion", "stattrak_premium",
                     "stattrak_premium_chg_7d", "souvenir_premium",
                     "has_wear_siblings", "has_stattrak_pair", "has_souvenir_pair"]:
            assert _feature_group(name) == "quality_spread", name
        # existing routing is unaffected
        assert _feature_group("return_7d") == "price_technicals"
        assert _feature_group("is_stattrak") == "item_identity"

    def test_flag_default_off(self, forecaster):
        assert forecaster.ENABLE_QUALITY_SPREAD is False
        assert forecaster._quality_spread_enabled() is False

    def test_effective_allowlist_appends_when_on(self, forecaster, monkeypatch):
        assert forecaster._effective_allowlist() == ["price_technicals"]
        monkeypatch.setenv("QUALITY_SPREAD", "1")
        assert forecaster._effective_allowlist() == ["price_technicals", "quality_spread"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/test_forecaster.py::TestQualitySpreadPlumbing -v`
Expected: FAIL (`_quality_spread_enabled` / `_effective_allowlist` / `ENABLE_QUALITY_SPREAD` not defined; `wear_*` routes to `"other"`).

- [ ] **Step 3: Write minimal implementation**

In `_feature_group` (forecaster.py:78), add this branch BEFORE the final `return "other"` (order does not conflict with existing prefixes):

```python
    if any(name.startswith(p) for p in ("wear_", "stattrak_", "souvenir_",
                                         "has_wear", "has_stattrak", "has_souvenir")):
        return "quality_spread"
```

Add the class constant near the other flags (e.g. after `STACK_RESIDUALS = False` at :186):

```python
    # Quality-spread / cross-wear experiment (2026-07-25). Default OFF.
    # When enabled, _add_quality_spread_features computes cross-variant
    # features AND "quality_spread" is appended to the feature allowlist so
    # the columns reach the model. Env override: QUALITY_SPREAD=1.
    ENABLE_QUALITY_SPREAD = False
```

Add the two methods (anywhere in the class, e.g. just above `_add_quality_spread_features`'s eventual home near the other `_add_*` methods):

```python
    def _quality_spread_enabled(self) -> bool:
        return bool(self.ENABLE_QUALITY_SPREAD) or os.environ.get("QUALITY_SPREAD") == "1"

    def _effective_allowlist(self):
        allow = self.FEATURE_GROUP_ALLOWLIST
        if not allow or not self._quality_spread_enabled():
            return allow
        return list(allow) + (["quality_spread"] if "quality_spread" not in allow else [])
```

Wire the allowlist at forecaster.py:2039 to use the effective list. Change:

```python
        if self.FEATURE_GROUP_ALLOWLIST:
            pre = len(self.feature_cols)
            self.feature_cols = self._apply_feature_allowlist(
                self.feature_cols, self.FEATURE_GROUP_ALLOWLIST)
```

to:

```python
        _allowlist = self._effective_allowlist()
        if _allowlist:
            pre = len(self.feature_cols)
            self.feature_cols = self._apply_feature_allowlist(
                self.feature_cols, _allowlist)
```

(`os` is already imported at the top of forecaster.py.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/test_forecaster.py::TestQualitySpreadPlumbing -v && python3 -m py_compile models/forecaster.py`
Expected: PASS (3 tests), compile clean.

- [ ] **Step 5: Commit**

```bash
git add backend/models/forecaster.py backend/tests/test_forecaster.py
git commit -m "feat: quality_spread feature group + flag plumbing"
```

---

### Task 2: Variant attribute index

A pure helper that turns item ids + a name map into per-item variant attributes and group keys. No price math — just parsing and key construction, so it is trivially unit-testable without a DB.

**Files:**
- Modify: `backend/models/forecaster.py` (new method on `ItemForecaster`, near the other `_fetch_*`/`_add_*` helpers)
- Test: `backend/tests/test_forecaster.py`

**Interfaces:**
- Consumes: `parse_item_name` (already imported in forecaster.py).
- Produces: method `ItemForecaster._build_variant_attributes(self, item_ids, name_map: dict) -> pd.DataFrame` with columns `["item_id", "quality_rank", "is_stattrak", "is_souvenir", "wear_group", "st_group", "sv_group"]`. `name_map` maps `str(item_id) -> name`. For items whose name does not parse into weapon+skin+quality, all three `*_group` columns are `None` and `quality_rank`/`is_stattrak`/`is_souvenir` are `0`. Group keys are `"|"`-joined strings: `wear_group=f"{weapon}|{skin}|{st}|{sv}"`, `st_group=f"{weapon}|{skin}|{quality}|{sv}"`, `sv_group=f"{weapon}|{skin}|{quality}|{st}"`.

- [ ] **Step 1: Write the failing test**

```python
class TestVariantAttributes:
    def test_groups_for_parseable_skins(self, forecaster):
        name_map = {
            "1": "AK-47 | Redline (Field-Tested)",
            "2": "AK-47 | Redline (Factory New)",
            "3": "StatTrak™ AK-47 | Redline (Field-Tested)",
            "4": "Glove Case",  # no wear -> unparseable into a skin
        }
        attrs = forecaster._build_variant_attributes(["1", "2", "3", "4"], name_map)
        by_id = {r["item_id"]: r for _, r in attrs.iterrows()}
        # items 1 and 2 share a wear_group (same weapon+skin+ST+SV, differ by wear)
        assert by_id["1"]["wear_group"] == by_id["2"]["wear_group"]
        assert by_id["1"]["wear_group"] is not None
        # item 3 (StatTrak) is a DIFFERENT wear_group but shares st_group with item 1
        assert by_id["3"]["wear_group"] != by_id["1"]["wear_group"]
        assert by_id["3"]["st_group"] == by_id["1"]["st_group"]
        assert by_id["3"]["is_stattrak"] == 1
        assert by_id["2"]["quality_rank"] == 5  # Factory New
        # unparseable item -> all groups None
        assert by_id["4"]["wear_group"] is None
        assert by_id["4"]["st_group"] is None

    def test_missing_name_yields_none_groups(self, forecaster):
        attrs = forecaster._build_variant_attributes(["9"], {})
        row = attrs.iloc[0]
        assert row["wear_group"] is None
        assert row["is_stattrak"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/test_forecaster.py::TestVariantAttributes -v`
Expected: FAIL with `AttributeError: ... _build_variant_attributes`.

- [ ] **Step 3: Write minimal implementation**

```python
    def _build_variant_attributes(self, item_ids, name_map: dict) -> pd.DataFrame:
        """Parse item names into cross-variant grouping keys.

        Returns one row per item_id. Items whose name does not parse into
        weapon+skin+quality get None group keys (they will receive neutral
        feature values and flag-off indicators downstream).
        """
        recs = []
        for iid in item_ids:
            name = name_map.get(str(iid))
            p = parse_item_name(name) if name else {}
            weapon = p.get("weapon")
            skin = p.get("skin_name")
            quality = p.get("quality")
            st = int(p.get("is_stattrak", False))
            sv = int(p.get("is_souvenir", False))
            if weapon and skin and quality:
                recs.append({
                    "item_id": iid,
                    "quality_rank": int(p.get("quality_rank", 0)),
                    "is_stattrak": st,
                    "is_souvenir": sv,
                    "wear_group": f"{weapon}|{skin}|{st}|{sv}",
                    "st_group": f"{weapon}|{skin}|{quality}|{sv}",
                    "sv_group": f"{weapon}|{skin}|{quality}|{st}",
                })
            else:
                recs.append({
                    "item_id": iid, "quality_rank": 0, "is_stattrak": 0,
                    "is_souvenir": 0, "wear_group": None,
                    "st_group": None, "sv_group": None,
                })
        return pd.DataFrame(recs, columns=[
            "item_id", "quality_rank", "is_stattrak", "is_souvenir",
            "wear_group", "st_group", "sv_group"])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/test_forecaster.py::TestVariantAttributes -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/models/forecaster.py backend/tests/test_forecaster.py
git commit -m "feat: variant attribute index for quality-spread features"
```

---

### Task 3: Wear-ladder features

The core signal. Computes the same-date wear-ladder ratio, its trailing 60d z-score, its 7/14d change, and the ladder dispersion — with neutral handling for items lacking siblings.

**Files:**
- Modify: `backend/models/forecaster.py` (new method `_add_quality_spread_features`)
- Test: `backend/tests/test_forecaster.py`

**Interfaces:**
- Consumes: `_build_variant_attributes` (Task 2); `_fetch_item_metadata()` (returns DataFrame with `item_id`, `name`, `type`).
- Produces: method `ItemForecaster._add_quality_spread_features(self, df: pd.DataFrame) -> pd.DataFrame`. When `_quality_spread_enabled()` is False, returns `df` unchanged. When True, adds wear-axis columns: `wear_spread_ratio`, `wear_spread_ratio_z60`, `wear_spread_ratio_chg_7d`, `wear_spread_ratio_chg_14d`, `wear_ladder_dispersion`, `has_wear_siblings`. Requires `df` to have `item_id`, `date`, `price`. All feature columns are float; no-sibling rows are `0.0` with `has_wear_siblings=0`. (StatTrak/Souvenir columns are added in Task 4 — this task leaves them out.)

- [ ] **Step 1: Write the failing test**

```python
class TestQualitySpreadWear:
    def _ladder_df(self):
        # Two wears of the same skin (item_a=FT, item_b=FN), 80 days.
        # item_a trends up faster so its ratio to the group mean drifts.
        rows = []
        base = date(2026, 1, 1)
        for d in range(80):
            pa = 10.0 + 0.10 * d          # FT
            pb = 20.0 + 0.02 * d          # FN
            rows.append({"item_id": "a", "date": base + timedelta(days=d), "price": round(pa, 2)})
            rows.append({"item_id": "b", "date": base + timedelta(days=d), "price": round(pb, 2)})
        # a lone item with no sibling
        for d in range(80):
            rows.append({"item_id": "z", "date": base + timedelta(days=d), "price": 5.0})
        return pd.DataFrame(rows)

    def _patched(self, forecaster, name_map):
        import pandas as pd
        meta = pd.DataFrame([{"item_id": k, "name": v, "type": "skin"} for k, v in name_map.items()])
        forecaster._fetch_item_metadata = lambda: meta
        return forecaster

    def test_disabled_is_noop(self, forecaster):
        df = self._ladder_df()
        out = forecaster._add_quality_spread_features(df.copy())
        assert "wear_spread_ratio" not in out.columns
        assert list(out.columns) == list(df.columns)

    def test_wear_features_present_and_correct(self, forecaster, monkeypatch):
        monkeypatch.setenv("QUALITY_SPREAD", "1")
        name_map = {
            "a": "AK-47 | Redline (Field-Tested)",
            "b": "AK-47 | Redline (Factory New)",
            "z": "Desert Eagle | Blaze (Factory New)",
        }
        self._patched(forecaster, name_map)
        df = self._ladder_df()
        out = forecaster._add_quality_spread_features(df)
        for c in ["wear_spread_ratio", "wear_spread_ratio_z60",
                  "wear_spread_ratio_chg_7d", "wear_spread_ratio_chg_14d",
                  "wear_ladder_dispersion", "has_wear_siblings"]:
            assert c in out.columns, c
        row = lambda iid, d: out[(out.item_id == iid) &
                                 (out.date == date(2026, 1, 1) + timedelta(days=d))].iloc[0]
        # day 0: mean(10, 20) = 15 -> a ratio = 10/15 = 0.667
        assert row("a", 0)["wear_spread_ratio"] == pytest.approx(10.0 / 15.0, abs=1e-3)
        assert row("a", 0)["has_wear_siblings"] == 1
        # lone item z: neutral everywhere
        assert row("z", 40)["has_wear_siblings"] == 0
        assert row("z", 40)["wear_spread_ratio"] == 0.0
        assert row("z", 40)["wear_spread_ratio_z60"] == 0.0
        # a's ratio rises over time (a outpaces b) -> positive z late in the series
        assert row("a", 79)["wear_spread_ratio_z60"] > 0
        # 7d change is finite and nonzero for a mid-series
        assert abs(row("a", 40)["wear_spread_ratio_chg_7d"]) > 0
        # no NaN/inf leaks into any feature column
        import numpy as np
        for c in ["wear_spread_ratio", "wear_spread_ratio_z60",
                  "wear_spread_ratio_chg_7d", "wear_ladder_dispersion"]:
            assert np.isfinite(out[c]).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/test_forecaster.py::TestQualitySpreadWear -v`
Expected: FAIL with `AttributeError: ... _add_quality_spread_features`.

- [ ] **Step 3: Write minimal implementation**

```python
    def _add_quality_spread_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cross-variant relative-value features (wear ladder; StatTrak/Souvenir
        premiums added in a later step). Gated by _quality_spread_enabled().

        All cross-item aggregates use SAME-DATE sibling prices (leak-safe);
        z-scores and changes use only trailing per-item history. Items without
        a sibling on an axis get 0.0 features and a 0 indicator flag.
        """
        if not self._quality_spread_enabled():
            return df

        logger.info("Adding quality-spread (cross-variant) features...")
        meta = self._fetch_item_metadata()
        name_map = {}
        if meta is not None and not meta.empty:
            name_map = dict(zip(meta["item_id"].astype(str), meta["name"]))
        attrs = self._build_variant_attributes(df["item_id"].unique(), name_map)
        df = df.merge(attrs[["item_id", "wear_group", "st_group", "sv_group"]],
                      on="item_id", how="left")

        df = df.sort_values(["item_id", "date"]).reset_index(drop=True)

        # ── Wear axis ──────────────────────────────────────────────
        gm = df.groupby(["wear_group", "date"])["price"]
        group_mean = gm.transform("mean")
        group_std = gm.transform("std")
        group_n = gm.transform("count")
        has_wear = df["wear_group"].notna() & (group_n >= 2)
        df["has_wear_siblings"] = has_wear.astype(float)

        ratio = df["price"] / group_mean.replace(0, np.nan)
        ratio = ratio.where(has_wear)  # NaN where no sibling (kept out of rolling)
        df["wear_ladder_dispersion"] = (group_std / group_mean.replace(0, np.nan)
                                        ).where(has_wear).fillna(0.0)

        g = df.assign(_r=ratio).groupby("item_id")["_r"]
        roll_mean = g.transform(lambda s: s.rolling(60, min_periods=20).mean())
        roll_std = g.transform(lambda s: s.rolling(60, min_periods=20).std())
        z = (ratio - roll_mean) / roll_std.replace(0, np.nan)
        df["wear_spread_ratio_z60"] = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        df["wear_spread_ratio_chg_7d"] = g.transform(lambda s: s - s.shift(7)).fillna(0.0)
        df["wear_spread_ratio_chg_14d"] = g.transform(lambda s: s - s.shift(14)).fillna(0.0)
        df["wear_spread_ratio"] = ratio.fillna(0.0)

        df = df.drop(columns=["wear_group", "st_group", "sv_group"])
        return df
```

Note: `shift(7)`/`shift(14)` are positional within each date-sorted item series. The archive is continuous post-backfill (2026-07-25 changelog), so positional ≈ calendar for the recent window this experiment cares about; this is an accepted simplification for an experimental feature.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/test_forecaster.py::TestQualitySpreadWear -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/models/forecaster.py backend/tests/test_forecaster.py
git commit -m "feat: wear-ladder quality-spread features"
```

---

### Task 4: StatTrak & Souvenir premium features

Adds the two premium axes to the same method. Each is the same-date ratio of the two variants, broadcast to both rows, with trailing z-score and 7d change.

**Files:**
- Modify: `backend/models/forecaster.py` (extend `_add_quality_spread_features`)
- Test: `backend/tests/test_forecaster.py`

**Interfaces:**
- Produces: `_add_quality_spread_features` additionally adds `stattrak_premium`, `stattrak_premium_z60`, `stattrak_premium_chg_7d`, `has_stattrak_pair`, `souvenir_premium`, `souvenir_premium_z60`, `souvenir_premium_chg_7d`, `has_souvenir_pair`. Premium is defined as `ST_price / non-ST_price` (resp. `Souvenir / normal`), assigned to both members of the pair on each date. No-pair rows are `0.0` with the indicator `0`.

- [ ] **Step 1: Write the failing test**

```python
class TestQualitySpreadPremiums:
    def _pair_df(self):
        rows = []
        base = date(2026, 1, 1)
        for d in range(40):
            rows.append({"item_id": "st", "date": base + timedelta(days=d), "price": 30.0})
            rows.append({"item_id": "plain", "date": base + timedelta(days=d), "price": 10.0})
            rows.append({"item_id": "loner", "date": base + timedelta(days=d), "price": 7.0})
        return pd.DataFrame(rows)

    def test_stattrak_premium(self, forecaster, monkeypatch):
        monkeypatch.setenv("QUALITY_SPREAD", "1")
        name_map = {
            "st": "StatTrak™ AK-47 | Redline (Field-Tested)",
            "plain": "AK-47 | Redline (Field-Tested)",
            "loner": "AWP | Asiimov (Field-Tested)",
        }
        meta = pd.DataFrame([{"item_id": k, "name": v, "type": "skin"} for k, v in name_map.items()])
        forecaster._fetch_item_metadata = lambda: meta
        out = forecaster._add_quality_spread_features(self._pair_df())
        r = lambda iid, d: out[(out.item_id == iid) &
                               (out.date == date(2026, 1, 1) + timedelta(days=d))].iloc[0]
        # premium = 30/10 = 3.0 on BOTH members
        assert r("st", 5)["stattrak_premium"] == pytest.approx(3.0)
        assert r("plain", 5)["stattrak_premium"] == pytest.approx(3.0)
        assert r("st", 5)["has_stattrak_pair"] == 1
        # lone AWP with no StatTrak sibling -> neutral
        assert r("loner", 5)["has_stattrak_pair"] == 0
        assert r("loner", 5)["stattrak_premium"] == 0.0
        # constant premium -> zero change, finite z
        import numpy as np
        assert r("st", 20)["stattrak_premium_chg_7d"] == pytest.approx(0.0)
        assert np.isfinite(out["souvenir_premium"]).all()
        assert np.isfinite(out["stattrak_premium_z60"]).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/test_forecaster.py::TestQualitySpreadPremiums -v`
Expected: FAIL (`stattrak_premium` column missing / KeyError).

- [ ] **Step 3: Write minimal implementation**

Insert, inside `_add_quality_spread_features` BEFORE the final `df = df.drop(columns=[...])`, a reusable premium helper and its two calls:

```python
        # ── Premium axes (StatTrak, Souvenir) ──────────────────────
        def _add_premium(df, group_col, flag_col, prefix, hi_mask):
            # hi_mask marks the "numerator" variant (ST=1 / Souvenir=1).
            hi = df["price"].where(hi_mask)
            lo = df["price"].where(~hi_mask)
            gk = [group_col, "date"]
            hi_p = hi.groupby([df[group_col], df["date"]]).transform("max")
            lo_p = lo.groupby([df[group_col], df["date"]]).transform("max")
            has_pair = df[group_col].notna() & hi_p.notna() & lo_p.notna()
            df[flag_col] = has_pair.astype(float)
            prem = (hi_p / lo_p.replace(0, np.nan)).where(has_pair)
            gg = df.assign(_p=prem).groupby("item_id")["_p"]
            rmean = gg.transform(lambda s: s.rolling(60, min_periods=20).mean())
            rstd = gg.transform(lambda s: s.rolling(60, min_periods=20).std())
            z = (prem - rmean) / rstd.replace(0, np.nan)
            df[f"{prefix}_z60"] = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            df[f"{prefix}_chg_7d"] = gg.transform(lambda s: s - s.shift(7)).fillna(0.0)
            df[prefix] = prem.fillna(0.0)
            return df

        # is_stattrak / is_souvenir come from the attribute index. st_group and
        # sv_group are already on df from the Task-3 merge (drop line removed below).
        _st = df["item_id"].map(dict(zip(attrs["item_id"], attrs["is_stattrak"]))).fillna(0).astype(int)
        _sv = df["item_id"].map(dict(zip(attrs["item_id"], attrs["is_souvenir"]))).fillna(0).astype(int)
        df = _add_premium(df, "st_group", "has_stattrak_pair", "stattrak_premium", _st == 1)
        df = _add_premium(df, "sv_group", "has_souvenir_pair", "souvenir_premium", _sv == 1)
```

Then update the final drop to remove all three group keys (they are consumed above):

```python
        df = df.drop(columns=["wear_group", "st_group", "sv_group"])
        return df
```

(In Task 3, REMOVE the line `df = df.drop(columns=["wear_group", "st_group", "sv_group"])` so `st_group`/`sv_group` survive for the premium helper; this final drop at the end of Task 4 replaces it.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/test_forecaster.py::TestQualitySpreadPremiums tests/test_forecaster.py::TestQualitySpreadWear -v`
Expected: PASS (both classes — confirms Task 3 still green after the drop-line change).

- [ ] **Step 5: Commit**

```bash
git add backend/models/forecaster.py backend/tests/test_forecaster.py
git commit -m "feat: StatTrak/Souvenir premium quality-spread features"
```

---

### Task 5: Wire into pipeline + leakage guard

Calls the new method in the training and predict paths, verifies the flag on/off contract end-to-end, and adds the anti-leakage test.

**Files:**
- Modify: `backend/models/forecaster.py` (`build_training_data` near :2020; predict path near :2984, next to the other `_add_cross_sectional_features` call)
- Test: `backend/tests/test_forecaster.py`

**Interfaces:**
- Consumes: `_add_quality_spread_features` (Tasks 3–4).
- Produces: no new public interface — both pipelines call `df = self._add_quality_spread_features(df)` immediately after `df = self._add_cross_sectional_features(df)` at both sites.

- [ ] **Step 1: Write the failing test**

```python
class TestQualitySpreadPipeline:
    def _mini(self):
        rows = []
        base = date(2026, 1, 1)
        for d in range(30):
            rows.append({"item_id": "a", "date": base + timedelta(days=d), "price": 10.0 + d})
            rows.append({"item_id": "b", "date": base + timedelta(days=d), "price": 20.0 + d})
        return pd.DataFrame(rows)

    def test_pipeline_call_present_at_both_sites(self):
        import inspect
        from models.forecaster import ItemForecaster
        src = inspect.getsource(ItemForecaster)
        assert src.count("_add_quality_spread_features(df)") >= 2

    def test_no_future_leak(self, forecaster, monkeypatch):
        monkeypatch.setenv("QUALITY_SPREAD", "1")
        meta = pd.DataFrame([
            {"item_id": "a", "name": "AK-47 | Redline (Field-Tested)", "type": "skin"},
            {"item_id": "b", "name": "AK-47 | Redline (Factory New)", "type": "skin"},
        ])
        forecaster._fetch_item_metadata = lambda: meta
        base_df = self._mini()
        out1 = forecaster._add_quality_spread_features(base_df.copy())
        # Perturb the LAST day's prices; a feature at day 10 must not change.
        pert = base_df.copy()
        last = pert["date"] == date(2026, 1, 30)
        pert.loc[last, "price"] = pert.loc[last, "price"] * 5
        out2 = forecaster._add_quality_spread_features(pert)
        pick = lambda o: o[(o.item_id == "a") & (o.date == date(2026, 1, 11))].iloc[0]
        for c in ["wear_spread_ratio", "wear_spread_ratio_z60", "wear_spread_ratio_chg_7d"]:
            assert pick(out1)[c] == pytest.approx(pick(out2)[c]), c
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/test_forecaster.py::TestQualitySpreadPipeline -v`
Expected: FAIL on `test_pipeline_call_present_at_both_sites` (call not yet wired; count < 2). `test_no_future_leak` should already pass from Tasks 3–4 (this locks the behavior in).

- [ ] **Step 3: Write minimal implementation**

In `build_training_data`, immediately after the cross-sectional call at forecaster.py:2020:

```python
        df = self._add_cross_sectional_features(df)
        df = self._add_quality_spread_features(df)
```

In the predict path, immediately after the `_add_cross_sectional_features(df)` call near forecaster.py:2984:

```python
            df = self._add_cross_sectional_features(df)
            df = self._add_quality_spread_features(df)
```

(Match the surrounding indentation at each site.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/test_forecaster.py::TestQualitySpreadPipeline -v && python3 -m py_compile models/forecaster.py`
Expected: PASS.

- [ ] **Step 5: Run the full suite (flag off) to confirm no regression**

Run: `cd backend && pytest -q`
Expected: all pass (was 179; now 179 + new tests). If any prior test changed feature counts, confirm the new columns are absent with the flag off (they must be, since default is off).

- [ ] **Step 6: Commit**

```bash
git add backend/models/forecaster.py backend/tests/test_forecaster.py
git commit -m "feat: wire quality-spread features into train + predict paths"
```

---

### Task 6: Experiment runner note + changelog stub

Documents exactly how to run the A/B and where results go, so the experiment is reproducible and the decision is pre-registered. No model code.

**Files:**
- Create: `backend/docs` note — actually create `docs/changelog/2026-07-25-quality-spread-experiment.md` (changelog lives at repo-root `docs/`, per AGENTS.md structure)

**Interfaces:** none (documentation only).

- [ ] **Step 1: Write the changelog/runner note**

Create `docs/changelog/2026-07-25-quality-spread-experiment.md`:

```markdown
# Quality-spread / cross-wear features — experiment

**Date:** 2026-07-25
**Status:** Features implemented behind `QUALITY_SPREAD=1` (default off). A/B pending.
**Spec:** docs/superpowers/specs/2026-07-25-quality-spread-features-design.md

## What shipped (code)
- `_add_quality_spread_features` on `ItemForecaster`: wear-ladder ratio/z60/chg,
  StatTrak & Souvenir premiums, and `has_*` indicator flags (~13 columns).
- New `quality_spread` feature group in `_feature_group`; `QUALITY_SPREAD=1`
  computes the columns AND appends the group to the allowlist.

## How to run the A/B (from backend/)
Baseline (current production):
    DATABASE_URL="sqlite:///./cs2_market.db" python scripts/forecast_prices.py --train-only
    DATABASE_URL="sqlite:///./cs2_market.db" python scripts/backtest_accuracy.py

Treatment:
    QUALITY_SPREAD=1 DATABASE_URL="sqlite:///./cs2_market.db" python scripts/forecast_prices.py --train-only
    DATABASE_URL="sqlite:///./cs2_market.db" python scripts/backtest_accuracy.py

Record per-horizon CV directional accuracy (training log) and the backtest
table for both arms. The permutation gate is reported automatically by
`_validate_feature_groups` — look for the `quality_spread` group's `passed`
flag and `drop_pp` in the training log.

## Decision rule (pre-registered)
SHIP only if: `quality_spread` is permutation-retained on >=1 short horizon
(3d/7d) AND backtest delta >= +0.5pp there AND no horizon regresses beyond
-1.5pp. On ship: bump lgbm-v3 -> lgbm-v4 and set ENABLE_QUALITY_SPREAD=True.
Otherwise SHELVE: leave code, flag off, and add a null-result row to
docs/research/accuracy-opportunities.md's Reality Check table.

## Results
_(fill after the A/B run)_
```

- [ ] **Step 2: Commit**

```bash
git add docs/changelog/2026-07-25-quality-spread-experiment.md
git commit -m "docs: quality-spread experiment runner + pre-registered decision rule"
```

---

## Self-Review

**Spec coverage:**
- Flag + allowlist interaction → Task 1. ✅
- Variant grouping from parsed names → Task 2. ✅
- Wear-ladder features (ratio/z60/chg/dispersion + indicator) → Task 3. ✅
- StatTrak & Souvenir premiums (+ indicators) → Task 4. ✅
- Pipeline wiring (train + predict), neutral handling, leak-safety → Tasks 3–5. ✅
- Permutation gate → automatic via `quality_spread` group (Task 1 routing) + documented in Task 6. ✅
- A/B protocol + pre-registered decision rule → Task 6. ✅
- Full-suite regression with flag off → Task 5 Step 5. ✅

**Placeholder scan:** All code steps contain concrete code; the only "fill after run" is the Results section of the experiment note, which is correct (results don't exist yet). No TBDs in implementation.

**Type consistency:** `_build_variant_attributes` column names (`wear_group`, `st_group`, `sv_group`, `is_stattrak`, `is_souvenir`) are consumed unchanged in Tasks 3–4. `_add_quality_spread_features` feature names match the `_feature_group` prefixes registered in Task 1 (`wear_`, `stattrak_`, `souvenir_`, `has_wear`, `has_stattrak`, `has_souvenir`). `_effective_allowlist` / `_quality_spread_enabled` names are consistent across Tasks 1 and 3–4.

**Note for implementer:** Task 4's edit modifies code introduced in Task 3 (removing the Task-3 drop line so `st_group`/`sv_group` survive). If executing tasks out of order, read both Task 3 and Task 4 before writing the method.
