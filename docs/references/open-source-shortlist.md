# Open-source shortlist

Surveyed 2026-09-23, with licenses and activity checked through the GitHub API. Ranked by
importance. "Add" means take code or a dependency; "consider" means read it or run it as
a reference, with no dependency.

## Tier 1: do these

1. **[aangelopoulos/conformal-time-series](https://github.com/aangelopoulos/conformal-time-series)**
   (MIT). Conformal PID control. **Add** by porting the P+I update from
   `core/methods.py::quantile_integrator_log_scorecaster`, with attribution. Don't import
   it: it uses `np.infty`, which NumPy 2 removed. This is the only candidate that could
   change the served band. Prereg:
   `docs/research/2026-09-23-conformal-pid-served-h3-preregistration.md` (frozen 2026-09-23).
2. **[frazane/scoringrules](https://github.com/frazane/scoringrules)** (Apache-2.0).
   **Add** as a dependency for the interval (Winkler) score: one proper score for the 80%
   band instead of coverage and width reported separately. It needs no retrain and
   doesn't touch serving.
3. **[Nixtla/statsforecast](https://github.com/Nixtla/statsforecast)** (Apache-2.0).
   **Add** as an optional or dev dependency for an outside baseline (naive / ETS +
   conformal at matched 80%) in `PORTFOLIO.md`. It won't improve accuracy, since the
   centre carries no information. The value is credibility.

## Tier 2: consider

4. **[scikit-learn-contrib/MAPIE](https://github.com/scikit-learn-contrib/MAPIE)**
   (BSD-3). Use it as a test oracle only: check that our split-conformal `q_hat` matches
   MAPIE's on the same residuals. Don't replace `conformal.py`.
5. **[HilliamT/scm-price-history](https://github.com/HilliamT/scm-price-history)** (MIT)
   and **[maxokhonko/Steam-Market-Parser](https://github.com/maxokhonko/Steam-Market-Parser)**
   (MIT). Starting points for the Steam history backfill, which is the breadth path in the
   training-breadth verdict. Check first that the `pricehistory` endpoint still works
   without a login cookie.
6. **[Nixtla/utilsforecast](https://github.com/Nixtla/utilsforecast)** (Apache-2.0).
   Evaluation and CV helpers. Only worth it if statsforecast goes in.

## Tier 3: reference reading only

- **[valeman/awesome-conformal-prediction](https://github.com/valeman/awesome-conformal-prediction)**:
  look up prior art before preregistering a new band idea.
- **[ConformalPrediction/crepes](https://github.com/ConformalPrediction/crepes)** (BSD-3):
  conformal predictive systems that give a full distribution rather than one band.
- **[gergelyszabo94/csgo-trader-extension](https://github.com/gergelyszabo94/csgo-trader-extension)**
  (GPL-3.0): the source of the CSGOTrader dumps the Aggregator reads, and how each market
  price is derived. It's GPL, so read it but don't copy code.
- **[EricZhu-42/SteamTradingSiteTracker(-Data)](https://github.com/EricZhu-42/SteamTradingSiteTracker)**
  (MIT): a BUFF/IGXE/C5/Youpin price cross-check. Arbitrage and BUFF basis are already
  refuted.
- **[mzaffran/AdaptiveConformalPredictionsTimeSeries](https://github.com/mzaffran/AdaptiveConformalPredictionsTimeSeries)**,
  **[hamrel-cxu/EnbPI](https://github.com/hamrel-cxu/EnbPI)** (MIT): superseded by #1.

## Skip

- **darts / sktime / mlforecast**: too heavy, and they duplicate the existing pipeline.
  TFT is already refuted.
- **puncc** (no license) and **TorchCP** (LGPL, PyTorch-only): license blocks copying.
- **ByMykel/CSGO-API**: already ingested; the metadata features were refuted on 2026-08-06.
- **Unsafe:** `hanneshill68/cs2-skin-price-forecast-hub`,
  `fainatenor490/cs2-skin-price-forecast-hub` and `hillnathan64/cs2-skin-market-forecaster`.
  They are clones from throwaway accounts with a README and HTML but no code, matching the
  pattern of malware lures. Don't download anything from them.
- **KGaraj, Sapphirine** CS2 predictors: student projects with no license, and behind
  this repo.
