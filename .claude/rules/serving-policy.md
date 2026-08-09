---
paths:
  - "backend/api/**"
---

# Serving policy

**`MIN_SERVED_PRICE_USD = 1.0` is a convention, not a derivation.** `api/serving_policy.py`
sets the floor deliberately equal to the lower bound of `HEADLINE_MIN_TIER` so the
population the product shows is the population the headline accuracy number describes.
`tests/test_serving_policy.py` fails if the two diverge. Sub-$1 items are ~72% of the
forecast universe and one cent there is a 20% move.

Adding or changing a route means updating `frontend/lib/api.ts` in the same change.
