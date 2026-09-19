"""Tests for the ByMykel metadata ingest.

The load-bearing parts are the ones with no schema to protect them: five date
formats in one upstream file, the forward expansion of skin base names into
Steam market hash names, and the earliest-member pick that decides both item age
and crate identity.
"""

import sys
from datetime import date
from pathlib import Path
from typing import ClassVar

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.ingest_bymykel_metadata import (
    CodeBook,
    _earliest,
    build_crate_index,
    build_date_maps,
    build_frame,
    build_records,
    expand_skin_names,
    normalise_date,
    rarity_rank,
)


class TestNormaliseDate:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("2024-01-16", date(2024, 1, 16)),  # ISO
            ("2013/12/17", date(2013, 12, 17)),  # slashes, padded
            ("2014/5/2", date(2014, 5, 2)),  # neither padded
            ("2014/12/5", date(2014, 12, 5)),  # day unpadded
            ("2014-2-19", date(2014, 2, 19)),  # ISO separator, unpadded month
        ],
    )
    def test_accepts_every_upstream_format(self, raw, expected):
        assert normalise_date(raw) == expected

    @pytest.mark.parametrize(
        "raw",
        [
            None,
            "",
            "not a date",
            "2024",
            "2024-13-01",
            "2024-02-30",
            20240116,
        ],
    )
    def test_rejects_rather_than_guesses(self, raw):
        assert normalise_date(raw) is None


class TestRarityRank:
    def test_maps_onto_the_existing_repo_scale(self):
        # These must match models.steam_types.RARITY_RANK, because the model
        # already carries rarity_ordinal on that scale.
        assert rarity_rank({"name": "Consumer Grade"}) == 1
        assert rarity_rank({"name": "Mil-Spec Grade"}) == 3
        assert rarity_rank({"name": "Covert"}) == 6

    def test_covers_the_two_ranks_steam_types_never_saw(self):
        assert rarity_rank({"name": "Contraband"}) == 7
        assert rarity_rank({"name": "Default"}) == 0

    def test_contraband_outranks_covert(self):
        assert rarity_rank({"name": "Contraband"}) > rarity_rank({"name": "Covert"})

    @pytest.mark.parametrize("rarity", [None, {}, {"name": ""}, "Covert"])
    def test_missing_rarity_is_null_not_zero(self, rarity):
        # Zero is Base Grade, a real rank. Unknown must not collapse into it.
        assert rarity_rank(rarity) is None


class TestCodeBook:
    def test_codes_are_dense_and_start_at_zero(self, tmp_path):
        # LightGBM allocates over a categorical's value RANGE, not its
        # cardinality. A sparse encoding OOM-killed the treatment arm.
        book = CodeBook(tmp_path / "codes.json")
        book.assign("crate", ["crate-4288", "crate-1210", "crate-4352"])
        assert sorted(book.codes["crate"].values()) == [0, 1, 2]

    def test_namespaces_are_independent(self, tmp_path):
        book = CodeBook(tmp_path / "codes.json")
        book.assign("crate", ["crate-A"])
        book.assign("collection", ["coll-A"])
        assert book.get("crate", "crate-A") == 0
        assert book.get("collection", "coll-A") == 0

    def test_new_ids_append_and_never_renumber_existing_ones(self, tmp_path):
        # The whole point of persisting: a new crate shipping must not change
        # what an already-trained booster's splits mean.
        book = CodeBook(tmp_path / "codes.json")
        book.assign("crate", ["crate-m", "crate-z"])
        before = dict(book.codes["crate"])
        book.assign("crate", ["crate-a", "crate-m", "crate-z"])
        assert {k: book.codes["crate"][k] for k in before} == before
        assert book.get("crate", "crate-a") == 2

    def test_assignment_survives_a_round_trip_to_disk(self, tmp_path):
        path = tmp_path / "codes.json"
        book = CodeBook(path)
        book.assign("crate", ["crate-A", "crate-B"])
        book.save()
        assert CodeBook(path).codes == book.codes

    def test_unknown_and_missing_ids_are_null(self, tmp_path):
        book = CodeBook(tmp_path / "codes.json")
        book.assign("crate", ["crate-A"])
        assert book.get("crate", "crate-unseen") is None
        assert book.get("crate", None) is None


class TestExpandSkinNames:
    def test_plain_skin_across_wears(self):
        entry = {
            "name": "AK-47 | Redline",
            "stattrak": False,
            "souvenir": False,
            "wears": [{"name": "Factory New"}, {"name": "Field-Tested"}],
        }
        assert expand_skin_names(entry) == [
            ("AK-47 | Redline (Factory New)", False, False),
            ("AK-47 | Redline (Field-Tested)", False, False),
        ]

    def test_stattrak_prefix_goes_after_the_star(self):
        entry = {
            "name": "★ Karambit | Fade",
            "stattrak": True,
            "souvenir": False,
            "wears": [{"name": "Factory New"}],
        }
        names = [n for n, _, _ in expand_skin_names(entry)]
        assert names == [
            "★ Karambit | Fade (Factory New)",
            "★ StatTrak™ Karambit | Fade (Factory New)",
        ]

    def test_souvenir_variant(self):
        entry = {
            "name": "AWP | Safari Mesh",
            "stattrak": False,
            "souvenir": True,
            "wears": [{"name": "Battle-Scarred"}],
        }
        assert ("Souvenir AWP | Safari Mesh (Battle-Scarred)", False, True) in expand_skin_names(entry)

    def test_flags_track_the_emitted_variant_not_the_capability(self):
        entry = {
            "name": "AK-47 | Redline",
            "stattrak": True,
            "souvenir": False,
            "wears": [{"name": "Factory New"}],
        }
        by_name = {n: (st, sv) for n, st, sv in expand_skin_names(entry)}
        # The base variant of a StatTrak-capable skin is NOT itself StatTrak.
        assert by_name["AK-47 | Redline (Factory New)"] == (False, False)
        assert by_name["StatTrak™ AK-47 | Redline (Factory New)"] == (True, False)

    def test_vanilla_knife_has_no_wear_suffix(self):
        entry = {"name": "★ Bayonet", "stattrak": True, "souvenir": False, "wears": []}
        names = [n for n, _, _ in expand_skin_names(entry)]
        assert names == ["★ Bayonet", "★ StatTrak™ Bayonet"]

    def test_unnamed_entry_yields_nothing(self):
        assert expand_skin_names({"wears": [{"name": "Factory New"}]}) == []


class TestEarliest:
    DATES: ClassVar[dict] = {"c1": date(2013, 1, 1), "c2": date(2016, 6, 1), "c3": date(2013, 1, 1)}

    def test_picks_the_earliest_dated_member(self):
        chosen, when, _n = _earliest(["c2", "c1"], self.DATES)
        assert (chosen, when) == ("c1", date(2013, 1, 1))

    def test_counts_distinct_dates_not_members(self):
        # c1 and c3 share a date: one distinct date, so not ambiguous.
        _, _, n = _earliest(["c1", "c3"], self.DATES)
        assert n == 1
        _, _, n = _earliest(["c1", "c2"], self.DATES)
        assert n == 2

    def test_undated_members_keep_their_identity(self):
        chosen, when, n = _earliest(["zzz", "aaa"], {})
        assert (chosen, when, n) == ("aaa", None, 0)

    def test_empty(self):
        assert _earliest([], self.DATES) == (None, None, 0)


@pytest.fixture
def codebook(tmp_path):
    return CodeBook(tmp_path / "codes.json")


@pytest.fixture
def dumps():
    """A miniature twelve-dump set exercising every join path."""
    empty = {
        name: []
        for name in (
            "sticker_slabs",
            "graffiti",
            "collectibles",
            "highlights",
            "music_kits",
            "patches",
            "keychains",
            "agents",
        )
    }
    return {
        "skins": [
            {  # single crate, StatTrak-capable, float caps
                "name": "AK-47 | Redline",
                "rarity": {"name": "Classified"},
                "min_float": 0.1,
                "max_float": 0.7,
                "stattrak": True,
                "souvenir": False,
                "wears": [{"name": "Field-Tested"}],
                "crates": [{"id": "crate-A"}],
                "collections": [{"id": "coll-A"}],
            },
            {  # knife in two crates with different dates -> ambiguous
                "name": "★ Karambit | Fade",
                "rarity": {"name": "Covert"},
                "min_float": 0.0,
                "max_float": 0.08,
                "stattrak": False,
                "souvenir": False,
                "wears": [{"name": "Factory New"}],
                "crates": [{"id": "crate-A"}, {"id": "crate-B"}],
                "collections": [],
            },
            {  # no crate at all -> falls back to the collection release date
                "name": "P250 | Sand Dune",
                "rarity": {"name": "Consumer Grade"},
                "min_float": 0.0,
                "max_float": 1.0,
                "stattrak": False,
                "souvenir": False,
                "wears": [{"name": "Factory New"}],
                "crates": [],
                "collections": [{"id": "coll-A"}],
            },
        ],
        "crates": [
            {
                "id": "crate-A",
                "market_hash_name": "Crate A",
                "first_sale_date": "2013/9/20",
                "rarity": {"name": "Base Grade"},
                "crates": [],
                "collections": [],
            },
            {
                "id": "crate-B",
                "market_hash_name": "Crate B",
                "first_sale_date": "2017-05-01",
                "rarity": {"name": "Base Grade"},
                "crates": [],
                "collections": [],
            },
        ],
        "collections": [
            {"id": "coll-A", "release_date": "2014-2-19"},
        ],
        "stickers": [
            {
                "market_hash_name": "Sticker | Titan (Holo) | Katowice 2014",
                "rarity": {"name": "Exotic"},
                "crates": [{"id": "crate-B"}],
                "collections": [],
            },
        ],
        **empty,
    }


class TestBuildDateMaps:
    def test_direct_dates_win(self, dumps, codebook):
        crate_dates, _ = build_date_maps(dumps)
        assert crate_dates["crate-A"] == date(2013, 9, 20)
        assert crate_dates["crate-B"] == date(2017, 5, 1)

    def test_undated_crate_inherits_its_collection_release_date(self, dumps, codebook):
        dumps["crates"].append(
            {
                "id": "crate-C",
                "market_hash_name": "Crate C",
                "first_sale_date": None,
                "rarity": {"name": "Base Grade"},
                "crates": [],
                "collections": [],
            }
        )
        dumps["collections"][0]["crates"] = [{"id": "crate-C"}]
        crate_dates, _ = build_date_maps(dumps)
        assert crate_dates["crate-C"] == date(2014, 2, 19)

    def test_inheritance_never_overwrites_a_direct_date(self, dumps, codebook):
        dumps["collections"][0]["crates"] = [{"id": "crate-B"}]
        crate_dates, _ = build_date_maps(dumps)
        assert crate_dates["crate-B"] == date(2017, 5, 1)


class TestBuildCrateIndex:
    def test_indexes_both_contains_and_contains_rare(self, dumps, codebook):
        dumps["crates"][0]["contains"] = [{"id": "skin-common"}]
        dumps["crates"][0]["contains_rare"] = [{"id": "skin-rare"}]
        index = build_crate_index(dumps)
        assert index["skin-common"] == ["crate-A"]
        assert index["skin-rare"] == ["crate-A"]

    def test_reverse_index_supplies_a_crate_the_item_does_not_list(self, dumps, codebook):
        # skins[2] (P250 | Sand Dune) has crates: [] of its own.
        dumps["skins"][2]["id"] = "skin-p250"
        dumps["crates"][1]["contains"] = [{"id": "skin-p250"}]
        rec = build_records(dumps, codebook)["P250 | Sand Dune (Factory New)"]
        assert rec["type_meta_crate_id"] == codebook.get("crate", "crate-B")
        assert rec["item_age_first_sale_date"] == date(2017, 5, 1)

    def test_forward_and_reverse_are_unioned_without_double_counting(self, dumps, codebook):
        # crate-A is on the item AND in the reverse index: one distinct date,
        # so the item must not be flagged ambiguous by the duplicate.
        dumps["skins"][0]["id"] = "skin-ak"
        dumps["crates"][0]["contains"] = [{"id": "skin-ak"}]
        rec = build_records(dumps, codebook)["AK-47 | Redline (Field-Tested)"]
        assert rec["item_age_ambiguous"] == 0
        assert rec["type_meta_crate_id"] == codebook.get("crate", "crate-A")


class TestBuildRecords:
    def test_expands_and_joins_every_source(self, dumps, codebook):
        recs = build_records(dumps, codebook)
        assert "AK-47 | Redline (Field-Tested)" in recs
        assert "StatTrak™ AK-47 | Redline (Field-Tested)" in recs
        assert "★ Karambit | Fade (Factory New)" in recs
        assert "Crate A" in recs  # name-keyed dump
        assert "Sticker | Titan (Holo) | Katowice 2014" in recs

    def test_item_age_comes_from_the_earliest_crate(self, dumps, codebook):
        rec = build_records(dumps, codebook)["★ Karambit | Fade (Factory New)"]
        assert rec["item_age_first_sale_date"] == date(2013, 9, 20)

    def test_multiple_crate_dates_flag_ambiguous(self, dumps, codebook):
        recs = build_records(dumps, codebook)
        assert recs["★ Karambit | Fade (Factory New)"]["item_age_ambiguous"] == 1
        assert recs["AK-47 | Redline (Field-Tested)"]["item_age_ambiguous"] == 0

    def test_a_crate_gets_its_own_first_sale_date_as_its_age(self, dumps, codebook):
        # A crate is itself a tradable item; its age is its own sale date, not
        # anything inferred from what it contains.
        assert build_records(dumps, codebook)["Crate A"]["item_age_first_sale_date"] == date(2013, 9, 20)

    def test_own_date_beats_a_date_inferred_from_contents(self, dumps, codebook):
        dumps["crates"][1]["crates"] = [{"id": "crate-A"}]  # older than its own
        assert build_records(dumps, codebook)["Crate B"]["item_age_first_sale_date"] == date(2017, 5, 1)

    def test_collection_release_date_is_the_fallback(self, dumps, codebook):
        rec = build_records(dumps, codebook)["P250 | Sand Dune (Factory New)"]
        assert rec["item_age_first_sale_date"] == date(2014, 2, 19)
        assert rec["item_age_ambiguous"] == 0

    def test_float_caps_survive_a_later_name_keyed_dump(self, dumps, codebook):
        # A dump read after skins.json must not blank out the float caps.
        rec = build_records(dumps, codebook)["AK-47 | Redline (Field-Tested)"]
        assert (rec["float_meta_min"], rec["float_meta_max"]) == (0.1, 0.7)

    def test_crate_and_collection_codes_come_from_the_code_book(self, dumps, codebook):
        rec = build_records(dumps, codebook)["AK-47 | Redline (Field-Tested)"]
        assert rec["type_meta_crate_id"] == codebook.get("crate", "crate-A")
        assert rec["type_meta_collection_id"] == codebook.get("collection", "coll-A")

    def test_no_collection_leaves_the_code_null(self, dumps, codebook):
        rec = build_records(dumps, codebook)["★ Karambit | Fade (Factory New)"]
        assert rec["type_meta_collection_id"] is None


class TestBuildFrame:
    def test_schema_and_key(self, dumps, codebook):
        df = build_frame(dumps, codebook)
        assert list(df.columns) == [
            "item_slug",
            "item_age_first_sale_date",
            "item_age_ambiguous",
            "rarity_meta_rank",
            "rarity_meta",
            "is_meta_stattrak",
            "is_meta_souvenir",
            "float_meta_min",
            "float_meta_max",
            "type_meta_crate_id",
            "type_meta_collection_id",
        ]
        assert df["item_slug"].is_unique

    def test_does_not_store_item_age_days(self, dumps, codebook):
        # Age is (observation date - first sale date) and belongs to a row, not
        # an item. A stored column would freeze the calendar into a constant.
        assert "item_age_days" not in build_frame(dumps, codebook).columns

    def test_unknown_rarity_stays_null_through_the_frame(self, dumps, codebook):
        dumps["skins"][0]["rarity"] = None
        df = build_frame(dumps, codebook).set_index("item_slug")
        assert pd.isna(df.loc["AK-47 | Redline (Field-Tested)", "rarity_meta_rank"])
