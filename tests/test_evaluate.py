import pytest
import math

from fin_qa.evaluate import (
    extract_number, 
    numerical_match, 
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


def test_numerical_match_basic():
    """Test basic number matching within tolerance"""
    assert numerical_match("10.0", "10.2")
    assert numerical_match("100", "100.4")
    assert not numerical_match("10.0", "10.6")

def test_numerical_match_formats():
    """Test different number formats"""
    assert numerical_match("50%", "50")
    assert numerical_match("1,000", "1000")
    assert numerical_match("$123.45", "123.4")
    
def test_numerical_match_edge_cases():
    """Test edge cases and special values"""
    assert numerical_match("0", "0.0")
    assert numerical_match("", "")
    assert numerical_match(None, None)
    assert not numerical_match("invalid", "123")
    
def test_numerical_match_negative():
    """Test handling of negative numbers"""
    assert numerical_match("-10.2", "-10.0")
    assert numerical_match("-50%", "-50")
    assert not numerical_match("-10.0", "10.0")

def test_numerical_match_precision():
    """Test precision handling"""
    assert numerical_match("1.23456", "1.23")
    assert numerical_match("0.999999", "1.0")
    assert not numerical_match("1.234", "1.735")