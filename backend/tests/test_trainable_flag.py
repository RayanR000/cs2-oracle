from database import Item, backfilled_item_clause


def test_item_has_is_trainable_column():
    assert "is_trainable" in Item.__table__.columns
    assert Item.__table__.columns["is_trainable"].default.arg == 0


def test_backfilled_clause():
    assert str(backfilled_item_clause()) == "items.is_backfilled = :is_backfilled_1"
