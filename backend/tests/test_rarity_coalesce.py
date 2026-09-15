"""Complete the rarity column by falling back to ByMykel where Steam is empty.

After the key-normalisation fix, 1,938 of 8,691 items still carry no rarity.
They are not a lookup failure — they match `market_catalog.db` exactly and its
`type` column is literally empty for them. They are real, valuable skins
(`AK-47 | Asiimov (Minimal Wear)`, `AUG | Aristocrat (Field-Tested)`), so the
gap is missing upstream data and the only fix is a second source.

`item-metadata-bymykel.parquet` covers **1,933 of the 1,938** (1,147 on the
literal key, 786 more after normalisation). Measured 2026-08-07.

Two constraints on how the two sources combine:

* **Steam wins where it has a value.** The sources agree on 4,389/4,389
  overlapping items, so precedence is not about trusting one more — it is about
  keeping the column reproducible from the primary source, with ByMykel filling
  only what Steam left empty.
* **ByMykel's parquet carries a rank, not a name.** Rank alone cannot restore
  the string, because rank 3 collapses `milspec`, `high_grade` and
  `distinguished`. So the ingest has to emit the token too, which is what
  `rarity_token` covers here.
"""

from __future__ import annotations

import pandas as pd
import pytest
from scripts.archive.backfill_supply_metadata import coalesce_rarity
from scripts.ingest_bymykel_metadata import rarity_rank, rarity_token


class TestRarityToken:
    """The ByMykel rarity name, mapped onto the repo's existing token set."""

    @pytest.mark.parametrize(
        "name,token",
        [
            ("Covert", "covert"),
            ("Classified", "classified"),
            ("Restricted", "restricted"),
            ("Mil-Spec Grade", "milspec"),
            ("Industrial Grade", "industrial"),
            ("Consumer Grade", "consumer"),
            ("Base Grade", "base"),
            ("High Grade", "high_grade"),
            ("Remarkable", "remarkable"),
            ("Exotic", "exotic"),
            ("Extraordinary", "extraordinary"),
            ("Master", "master"),
        ],
    )
    def test_maps_bymykel_names_onto_steam_tokens(self, name, token):
        assert rarity_token({"name": name}) == token

    def test_contraband_gets_its_own_token(self):
        """Absent from Steam's ladder, so RARITY_KEYWORDS never had it."""
        assert rarity_token({"name": "Contraband"}) == "contraband"

    def test_token_and_rank_stay_consistent(self):
        """A token that disagreed with the rank beside it would be worse than
        no token at all — every reader picks one or the other.
        """
        for name in ("Covert", "Mil-Spec Grade", "High Grade", "Contraband"):
            assert (rarity_token({"name": name}) is None) == (rarity_rank({"name": name}) is None)

    @pytest.mark.parametrize("bad", [None, {}, {"name": ""}, {"name": "Nonsense"}, "Covert"])
    def test_unmappable_input_yields_none(self, bad):
        assert rarity_token(bad) is None


class TestCoalesceRarity:
    @staticmethod
    def _steam(rows):
        return pd.DataFrame(rows, columns=["item_slug", "rarity", "rarity_rank", "weapon_type"])

    @staticmethod
    def _bymykel(rows):
        return pd.DataFrame(rows, columns=["item_slug", "rarity_meta", "rarity_meta_rank"])

    def test_fills_an_empty_steam_rarity(self):
        out = coalesce_rarity(
            self._steam([("AK-47 | Asiimov (Minimal Wear)", None, None, "rifle")]),
            self._bymykel([("AK-47 | Asiimov (Minimal Wear)", "covert", 6)]),
        )
        row = out.iloc[0]
        assert row["rarity"] == "covert"
        assert row["rarity_rank"] == 6

    def test_does_not_overwrite_a_populated_steam_rarity(self):
        out = coalesce_rarity(self._steam([("X", "classified", 5, "rifle")]), self._bymykel([("X", "covert", 6)]))
        assert out.iloc[0]["rarity"] == "classified"
        assert out.iloc[0]["rarity_rank"] == 5

    def test_matches_on_the_normalised_key(self):
        """786 of the fills need this — the two files disagree on key format."""
        out = coalesce_rarity(
            self._steam([("sg-553-darkwing-field-tested", None, None, "rifle")]),
            self._bymykel([("SG 553 | Darkwing (Field-Tested)", "restricted", 4)]),
        )
        assert out.iloc[0]["rarity"] == "restricted"

    def test_leaves_a_genuine_miss_null(self):
        out = coalesce_rarity(
            self._steam([("nothing-knows-this", None, None, None)]), self._bymykel([("Other", "covert", 6)])
        )
        assert pd.isna(out.iloc[0]["rarity"])
        assert pd.isna(out.iloc[0]["rarity_rank"])

    def test_base_grade_keeps_rank_zero_through_the_fill(self):
        """Rank 0 is a real value; the fill must not treat it as absent."""
        out = coalesce_rarity(
            self._steam([("Some Capsule", None, None, "case")]), self._bymykel([("Some Capsule", "base", 0)])
        )
        assert out.iloc[0]["rarity"] == "base"
        assert out.iloc[0]["rarity_rank"] == 0

    def test_row_count_and_order_are_preserved(self):
        steam = self._steam([("a", None, None, None), ("b", "covert", 6, "rifle"), ("c", None, None, None)])
        out = coalesce_rarity(steam, self._bymykel([("a", "milspec", 3)]))
        assert list(out["item_slug"]) == ["a", "b", "c"]

    def test_an_empty_bymykel_frame_is_a_no_op(self):
        steam = self._steam([("a", "covert", 6, "rifle")])
        out = coalesce_rarity(steam, self._bymykel([]))
        assert out.iloc[0]["rarity"] == "covert"

    def test_rank_stays_a_nullable_integer(self):
        """Nulls promote a plain int64 column to float64, so covert reads back
        as `6.0` and `items.rarity_rank` (an Integer column) receives a float.
        """
        out = coalesce_rarity(
            self._steam([("a", None, None, None), ("b", "covert", 6, None)]), self._bymykel([("a", "milspec", 3)])
        )
        assert str(out["rarity_rank"].dtype) == "Int64"
        assert out.loc[out["item_slug"] == "b", "rarity_rank"].iloc[0] == 6

    def test_reports_what_it_filled(self):
        """The fill count is the number this whole change is judged on, so it
        has to be observable rather than inferred from a diff.
        """
        out = coalesce_rarity(
            self._steam([("a", None, None, None), ("b", "covert", 6, None)]), self._bymykel([("a", "milspec", 3)])
        )
        assert out.attrs.get("rarity_filled_from_bymykel") == 1
