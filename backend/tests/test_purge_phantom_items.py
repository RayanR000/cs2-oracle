"""Pairing logic for the phantom-item purge.

The 3,149 phantom `items` rows carry a mangled `item_id` (a slug, or a
`steam_`-prefixed key) while their `name` still holds the real
`market_hash_name`. The aggregator keys its external lookup on `name`
(`collectors/pipeline.py:124-128`), so both the phantom and the correctly-keyed
row receive every day's price — ~9% of recent archive writes.

The purge deletes phantoms. The invariant that makes that safe is that a phantom
is only ever deleted once its *keeper* — the correctly-keyed row holding the same
name — has been positively identified. Anything unpaired is reported, never
deleted, because deleting it would destroy the only copy of that item.
"""
import pytest

from scripts.purge_phantom_items import is_mangled_key, pair_phantoms, purge_archive_frame


class TestPhantomDetection:
    def test_slug_form_is_mangled(self):
        assert is_mangled_key("sealed-graffiti-popdog-battle-green")

    def test_steam_prefixed_form_is_mangled(self):
        """The slug regex does not match this form -- it holds '_' and '|'."""
        assert is_mangled_key("steam_sticker_|_sico_|_rio_2022")

    def test_a_real_market_hash_name_is_not_mangled(self):
        assert not is_mangled_key("AK-47 | Redline (Field-Tested)")
        assert not is_mangled_key("Prisma Case")
        assert not is_mangled_key("★ Karambit | Doppler (Factory New)")


class TestPairing:
    def test_pairs_a_phantom_to_the_real_row_sharing_its_name(self):
        rows = [
            (1, "AK-47 | Redline (Field-Tested)", "AK-47 | Redline (Field-Tested)"),
            (2, "ak-47-redline-field-tested", "AK-47 | Redline (Field-Tested)"),
        ]
        paired, unresolved = pair_phantoms(rows)
        assert unresolved == []
        assert len(paired) == 1
        assert paired[0].id == 2
        assert paired[0].keeper_id == 1
        # slugify(keeper.name) reproducing the phantom key is the cross-check
        # that the pairing is not a name collision.
        assert paired[0].slug_confirms is True

    def test_a_phantom_with_no_real_counterpart_is_never_deletable(self):
        """If nothing else holds this name, the phantom row is the only copy.
        Deleting it would destroy data, so it must land in `unresolved`."""
        rows = [(2, "orphan-item-never-imported", "Orphan Item Never Imported")]
        paired, unresolved = pair_phantoms(rows)
        assert paired == []
        assert len(unresolved) == 1
        assert unresolved[0].id == 2
        assert unresolved[0].reason == "no-keeper"

    def test_a_phantom_whose_name_was_overwritten_is_left_alone(self):
        """init_local_db's mirror writes `name = item_id`. Such a row has lost
        the only link back to its real identity, so it cannot be paired by name
        and must not be guessed at."""
        rows = [
            (1, "Prisma Case", "Prisma Case"),
            (2, "prisma-case", "prisma-case"),
        ]
        paired, unresolved = pair_phantoms(rows)
        assert paired == []
        assert len(unresolved) == 1
        assert unresolved[0].reason == "name-overwritten"

    def test_real_rows_are_never_returned_as_phantoms(self):
        rows = [
            (1, "AK-47 | Redline (Field-Tested)", "AK-47 | Redline (Field-Tested)"),
            (2, "Prisma Case", "Prisma Case"),
        ]
        assert pair_phantoms(rows) == ([], [])

    def test_pairing_flags_a_slug_that_its_keeper_name_does_not_reproduce(self):
        """A shared name whose slug does not regenerate the phantom key means the
        pairing rests on something other than the documented mechanism. Still
        pairable, but flagged so the dry-run surfaces it for review."""
        rows = [
            (1, "Some Item", "Some Item"),
            (2, "totally-unrelated-key", "Some Item"),
        ]
        paired, unresolved = pair_phantoms(rows)
        assert len(paired) == 1
        assert paired[0].slug_confirms is False


class TestArchivePurge:
    def test_drops_mangled_slugs_and_keeps_real_ones(self):
        pd = pytest.importorskip("pandas")
        frame = pd.DataFrame({
            "item_slug": [
                "AK-47 | Redline (Field-Tested)",
                "ak-47-redline-field-tested",
                "steam_sticker_|_sico_|_rio_2022",
                "Prisma Case",
            ],
            "day": ["2026-07-20"] * 4,
            "source": ["aggregator_steam"] * 4,
            "mean_price": [10.0, 10.0, 2.0, 3.0],
        })
        kept, dropped = purge_archive_frame(frame)
        assert dropped == 2
        assert list(kept["item_slug"]) == [
            "AK-47 | Redline (Field-Tested)",
            "Prisma Case",
        ]

    def test_a_clean_frame_is_returned_unchanged(self):
        pd = pytest.importorskip("pandas")
        frame = pd.DataFrame({
            "item_slug": ["Prisma Case"],
            "day": ["2026-07-20"],
            "source": ["aggregator_steam"],
        })
        kept, dropped = purge_archive_frame(frame)
        assert dropped == 0
        assert len(kept) == 1
