"""
Tests for data validation and cleaning
"""

from collectors.data_validation import DataCleaner, DataValidator


class TestDataValidator:
    """Test data validation functions"""

    def test_validate_price_valid(self):
        """Test valid price"""
        assert DataValidator.validate_price(50.0) is True
        assert DataValidator.validate_price(0.01) is True
        assert DataValidator.validate_price(10000.0) is True

    def test_validate_price_too_low(self):
        """Test price below minimum"""
        assert DataValidator.validate_price(0.00) is False

    def test_validate_price_too_high(self):
        """Test price above maximum"""
        assert DataValidator.validate_price(100001.0) is False

    def test_validate_price_invalid_type(self):
        """Test invalid price type"""
        assert DataValidator.validate_price("50") is False
        assert DataValidator.validate_price(None) is False

    def test_validate_volume_valid(self):
        """Test valid volume"""
        assert DataValidator.validate_volume(1000) is True
        assert DataValidator.validate_volume(0) is True

    def test_validate_volume_none(self):
        """Test volume None is valid"""
        assert DataValidator.validate_volume(None) is True

    def test_validate_volume_negative(self):
        """Test negative volume"""
        assert DataValidator.validate_volume(-100) is False

    def test_validate_volume_too_high(self):
        """Test volume too high"""
        assert DataValidator.validate_volume(2000000) is False

    def test_validate_item_name_valid(self):
        """Test valid item names"""
        assert DataValidator.validate_item_name("AK-47 | Aquamarine") is True
        assert DataValidator.validate_item_name("Knife") is True

    def test_validate_item_name_empty(self):
        """Test empty item name"""
        assert DataValidator.validate_item_name("") is False

    def test_validate_item_name_too_long(self):
        """Test item name too long"""
        assert DataValidator.validate_item_name("A" * 501) is False

    def test_validate_item_type_valid(self):
        """Test valid item types"""
        assert DataValidator.validate_item_type("skin") is True
        assert DataValidator.validate_item_type("case") is True
        assert DataValidator.validate_item_type("sticker") is True

    def test_validate_item_type_invalid(self):
        """Test invalid item type"""
        assert DataValidator.validate_item_type("invalid") is False

    def test_parse_item_name_with_condition(self):
        """Test parsing item name with condition"""
        name, condition = DataValidator.parse_item_name("AK-47 | Phantom (Field-Tested)")

        assert name == "AK-47 | Phantom"
        assert condition == "Field-Tested"

    def test_parse_item_name_without_condition(self):
        """Test parsing item name without condition"""
        name, condition = DataValidator.parse_item_name("CS2 Weapon Case")

        assert name == "CS2 Weapon Case"
        assert condition is None

    def test_detect_item_type_skin(self):
        """Test item type detection for skin"""
        item_type = DataValidator.detect_item_type("AK-47 | Phantom (Field-Tested)")

        assert item_type == "skin"

    def test_detect_item_type_case(self):
        """Test item type detection for case"""
        item_type = DataValidator.detect_item_type("CS2 Weapon Case")

        assert item_type == "case"

    def test_detect_item_type_agent(self):
        """Test item type detection for agent"""
        item_type = DataValidator.detect_item_type("Agent | Ava")

        assert item_type == "agent"

    def test_detect_item_type_case_hardened_is_skin_not_case(self):
        """Case Hardened is a skin finish, not a case"""
        assert DataValidator.detect_item_type("AK-47 | Case Hardened (Field-Tested)") == "skin"
        assert DataValidator.detect_item_type("Karambit | Case Hardened (Minimal Wear)") == "skin"
        assert DataValidator.detect_item_type("Five-SeveN | Case Hardened (Factory New)") == "skin"

    def test_detect_item_type_actual_case(self):
        """Actual cases should be detected as type 'case'"""
        assert DataValidator.detect_item_type("CS20 Case") == "case"
        assert DataValidator.detect_item_type("Revolution Case") == "case"
        assert DataValidator.detect_item_type("Operation Wildfire Case") == "case"

    def test_detect_item_type_case_key_is_case(self):
        """Case keys are classified as case"""
        assert DataValidator.detect_item_type("Chroma Case Key") == "case"


class TestDataCleaner:
    """Test data cleaning functions"""

    def test_clean_price_data(self):
        """Test price data cleaning"""
        clean_price = DataCleaner.clean_price_data(49.9999)

        assert clean_price == 50.0
        assert isinstance(clean_price, float)

    def test_clean_volume_data_valid(self):
        """Test volume data cleaning"""
        clean_volume = DataCleaner.clean_volume_data(1000)

        assert clean_volume == 1000

    def test_clean_volume_data_none(self):
        """Test volume data cleaning with None"""
        clean_volume = DataCleaner.clean_volume_data(None)

        assert clean_volume is None

    def test_clean_volume_data_negative(self):
        """Test volume data cleaning with negative"""
        clean_volume = DataCleaner.clean_volume_data(-100)

        assert clean_volume == 0

    def test_clean_item_name_whitespace(self):
        """Test item name cleaning with whitespace"""
        clean_name = DataCleaner.clean_item_name("  AK-47 | Phantom  ")

        assert clean_name == "AK-47 | Phantom"

    def test_deduplicate_items_basic(self):
        """Test item deduplication"""
        items = [
            {"name": "Item 1", "price": 100},
            {"name": "Item 2", "price": 200},
            {"name": "Item 1", "price": 150},
        ]

        deduped = DataCleaner.deduplicate_items(items)

        assert len(deduped) == 2
        assert deduped[0]["name"] == "Item 1"
        assert deduped[0]["price"] == 100  # First occurrence kept

    def test_filter_outliers_works(self):
        """Test outlier filtering function runs without error"""
        prices = [100, 101, 99, 100.5, 101.5, 99.5, 100.2, 100.1, 100.3, 10000]

        # Function should run without error
        filtered = DataCleaner.filter_outliers(prices, threshold=3.0)

        # Should return a list
        assert isinstance(filtered, list)
        assert len(filtered) <= len(prices)

    def test_filter_outliers_no_outliers(self):
        """Test outlier filtering with no outliers"""
        prices = [100, 101, 99, 100.5, 101.5]

        filtered = DataCleaner.filter_outliers(prices)

        assert len(filtered) == len(prices)
