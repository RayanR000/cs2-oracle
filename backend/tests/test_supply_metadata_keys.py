"""Rarity is NULL on half the catalogue for two unrelated reasons.

Measured 2026-08-07 on the 4,296 NULL-rarity rows of
`price-archive/item-metadata.parquet`:

    2,344 (54.6%)  the slug never matched the catalog's key format
    1,926 (44.8%)  matched fine, but market_catalog.db has an empty `type`
       26 ( 0.6%)  genuinely absent from the catalog

Only the first is a bug here, and it is a pure string-format one.
`build_metadata_df` keys its lookup on the catalog's `hash_name` and
`display_name` -- `Berlin 2019 Legends (Holo/Foil)` -- then looks up the price
archive's `item_slug` -- `berlin-2019-legends-holo-foil` -- with a bare
`if slug in name_to_meta`. No normalisation on either side, so every
slug-formatted key misses. The script even loads a `db_name_map` bridge from
cs2_market.db for exactly this and then never reads it (it would not have
helped: measured, it resolves none of the 2,344).

The second bug is the reported rank. `parse_steam_type` returns
`rarity_rank: 0` for an unknown rarity, and 0 is also the real rank of
`base` and `highlight`. So 4,296 unknowns and 624 genuine base-tier items are
indistinguishable in the column, and anything reading rank instead of the
rarity string silently treats half the catalogue as the lowest tier rather
than as missing. `rarity_ordinal` (`_add_supply_features`) is such a reader.
It is currently allowlisted out of training, so this is latent, not live.

Not covered here: the 1,926 empty-`type` rows. That is missing upstream data,
not a defect -- `item-metadata-bymykel.parquet` is where those come from.
"""

from __future__ import annotations

import pytest
from models.steam_types import parse_steam_type
from scripts.archive.backfill_supply_metadata import _normalise_key, build_name_lookup


class TestNormaliseKey:
    """One key format, reachable from both a display name and a slug."""

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("Berlin 2019 Legends (Holo/Foil)", "berlin-2019-legends-holo-foil"),
            ("AK-47 | Asiimov (Minimal Wear)", "ak-47-asiimov-minimal-wear"),
            ("StatTrak™ AWP | Dragon Lore", "stattrak-awp-dragon-lore"),
            ("★ Karambit | Doppler (Factory New)", "karambit-doppler-factory-new"),
        ],
    )
    def test_display_names_normalise_to_slug_form(self, name, expected):
        assert _normalise_key(name) == expected

    def test_a_slug_is_its_own_normal_form(self):
        """The archive's keys must survive normalisation unchanged, or the
        lookup would need a third format to reconcile.
        """
        slug = "berlin-2019-legends-holo-foil"
        assert _normalise_key(slug) == slug

    def test_the_two_formats_meet(self):
        assert _normalise_key("Berlin 2019 Legends (Holo/Foil)") == _normalise_key("berlin-2019-legends-holo-foil")

    def test_collapses_runs_and_strips_edges(self):
        assert _normalise_key("  Sticker |  Team   (Foil) ") == "sticker-team-foil"


class TestBuildNameLookup:
    ROWS = [
        # (hash_name, display_name, type)
        ("Berlin 2019 Legends (Holo/Foil)", "Berlin 2019 Legends (Holo/Foil)", "High Grade Sticker"),
        ("AK-47 | Redline (Field-Tested)", "AK-47 | Redline (Field-Tested)", "Classified Rifle"),
    ]

    def test_the_slug_form_resolves(self):
        """The regression: 2,344 items missed on this exact lookup."""
        lookup = build_name_lookup(self.ROWS)
        hit = lookup.get("berlin-2019-legends-holo-foil")
        assert hit is not None, "slug-formatted key still misses the catalog"
        assert hit["rarity"] == "high_grade"

    def test_the_exact_display_name_still_resolves(self):
        """4,395 items already matched this way and must not regress."""
        lookup = build_name_lookup(self.ROWS)
        assert lookup["AK-47 | Redline (Field-Tested)"]["rarity"] == "classified"

    def test_a_rarity_bearing_entry_is_not_overwritten_by_an_empty_one(self):
        """Two catalog names can normalise to one key. When they collide, the
        one that actually carries a rarity has to win, or normalising would
        *lose* coverage it previously had.
        """
        rows = [
            ("Foo | Bar (Factory New)", "Foo | Bar (Factory New)", "Covert Rifle"),
            ("foo-bar-factory-new", "foo-bar-factory-new", None),
        ]
        assert build_name_lookup(rows)["foo-bar-factory-new"]["rarity"] == "covert"
        assert build_name_lookup(list(reversed(rows)))["foo-bar-factory-new"]["rarity"] == "covert"

    def test_an_untyped_catalog_row_yields_no_rarity(self):
        """The other 1,926: present, matched, and genuinely rarity-less. The
        lookup must not invent one.
        """
        lookup = build_name_lookup([("X | Y (FN)", "X | Y (FN)", None)])
        assert lookup["x-y-fn"]["rarity"] is None


class TestUnknownRankIsNotZero:
    """Unknown must be distinguishable from base grade, which is really 0."""

    def test_unknown_rarity_reports_no_rank(self):
        assert parse_steam_type(None)["rarity_rank"] is None
        assert parse_steam_type("")["rarity_rank"] is None

    def test_base_grade_still_ranks_zero(self):
        parsed = parse_steam_type("Base Grade Container")
        assert parsed["rarity"] == "base"
        assert parsed["rarity_rank"] == 0

    def test_highlight_still_ranks_zero(self):
        parsed = parse_steam_type("Highlight Base Grade Container")
        assert parsed["rarity"] == "highlight"
        assert parsed["rarity_rank"] == 0

    def test_the_real_ladder_is_untouched(self):
        assert parse_steam_type("Covert Rifle")["rarity_rank"] == 6
        assert parse_steam_type("Mil-Spec Grade Pistol")["rarity_rank"] == 3
