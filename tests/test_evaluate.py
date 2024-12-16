import pytest
import math

from fin_qa.evaluate import (
    extract_number, 
    numerical_match_with_units, 
)


def test_extract_number_currency():
    """Test extracting numbers with currency symbols."""
    assert extract_number("Price is $42") == 42
    assert extract_number("$42.50 for this item") == 42.50
    assert extract_number("Value: -$15.75") == -15.75


def test_extract_number_percentage():
    """Test extracting percentage values."""
    assert extract_number("Growth is 42%") == 42
    assert extract_number("Efficiency: 88.5%") == 88.5
    assert extract_number("Decline: -10.25%") == -10.25


@pytest.mark.parametrize("ground_truth,prediction,expected", [
    # Currency format tests
    ("$100", "$100", True),
    ("$100", "$100.0", True),
    ("$100.50", "$100.50", True),
    ("$100.5", "$100.50", True),
    
    # Percentage format tests
    ("42%", "42%", True),
    ("88.5%", "88.5%", True),
    ("88.5%", "88.50%", True),
    
    # Mixed format matches with slight variations
    ("$100", "$99.9", True),
    ("$100", "$100.1", True),
    ("42%", "41.9%", True),
    ("42%", "42.1%", True),
    
    # Unit matching tests
    ("$100", "$100.00", True),
    ("42%", "42.0%", True),
    
    # Cross-format or non-matching cases
    ("$100", "100", False),
    ("$100", "100%", False),
    ("42%", "$42", False),
    
    # Decimal place and precision tests
    ("$100.51", "$100.513", True),
    ("$100.9", "$100.91", True),
    ("88.50%", "88.5%", True),
    
    # Edge cases
    ("$0", "$0.0", True),
    ("-$100", "-$99.9", True),
    ("0%", "0.0%", True),
    ("-42%", "-41.9%", True),
])
def test_numerical_match_with_units(ground_truth, prediction, expected):
    assert numerical_match_with_units(ground_truth, prediction) == expected


def test_numerical_match_with_units_mixed_formats():
    test_cases = [
        ("$100", "100", False),
        ("$100", "100%", False),
        ("42%", "$42", False),
    ]
    for ground_truth, prediction, expected in test_cases:
        assert numerical_match_with_units(ground_truth, prediction) == expected


def test_numerical_match_with_units_whitespace():
    test_cases = [
        (" $100 ", "$100", True),
        ("$100", " $100 ", True),
        (" 42% ", "42%", True),
        ("42%", " 42% ", True),
    ]
    for ground_truth, prediction, expected in test_cases:
        assert numerical_match_with_units(ground_truth, prediction) == expected


def test_numerical_match_with_units_invalid_inputs():
    invalid_cases = [
        ("abc", "def"),
        ("$100", "abc"),
        ("42%", "abc"),
    ]
    for ground_truth, prediction in invalid_cases:
        assert numerical_match_with_units(ground_truth, prediction) is False