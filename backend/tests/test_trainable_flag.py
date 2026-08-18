from database import Item, trainable_item_clause, backfilled_item_clause

def test_item_has_is_trainable_column():
    assert "is_trainable" in Item.__table__.columns
    assert Item.__table__.columns["is_trainable"].default.arg == 0

def test_trainable_clause_distinct_from_backfilled():
    assert str(trainable_item_clause()) == "items.is_trainable = :is_trainable_1"
    assert str(backfilled_item_clause()) == "items.is_backfilled = :is_backfilled_1"
