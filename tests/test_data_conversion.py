import pytest
import json

from fin_qa.data_conversion import convert_to_markdown_table, convert_to_paragraph, fix_invalid_json


def test_convert_to_markdown_table_basic():
    """Test basic markdown table conversion."""
    data = [
        ['Name', 'Age', 'City'],
        ['Alice', 25, 'New York'],
        ['Bob', 30, 'San Francisco']
    ]
    expected = "| Name  | Age | City          |\n|-----|---|-------------|\n| Alice | 25  | New York      |\n| Bob   | 30  | San Francisco |"
    
    assert convert_to_markdown_table(data) == expected


def test_convert_to_markdown_table_mixed_types():
    """Test markdown table conversion with mixed data types."""
    data = [
        ['Product', 'Price', 'In Stock'],
        ['Laptop', 999.99, True],
        ['Phone', 599.5, False]
    ]
    expected = "| Product | Price  | In Stock |\n|-------|------|--------|\n| Laptop  | 999.99 | True     |\n| Phone   | 599.5  | False    |"
    
    assert convert_to_markdown_table(data) == expected


def test_convert_to_markdown_table_empty():
    """Test markdown table conversion with empty list."""
    with pytest.raises(IndexError):
        convert_to_markdown_table([])


def test_convert_to_paragraph_basic():
    """Test basic paragraph conversion."""
    data = ['Hello', 'world', 'how', 'are', 'you']
    expected = "Hello world how are you"
    
    assert convert_to_paragraph(data) == expected


def test_convert_to_paragraph_empty():
    """Test paragraph conversion with empty list."""
    assert convert_to_paragraph([]) == ""


def test_convert_to_paragraph_single_item():
    """Test paragraph conversion with single item."""
    data = ['Singleton']
    expected = "Singleton"
    
    assert convert_to_paragraph(data) == expected


def test_fix_invalid_json_single_quotes():
    """Test JSON string fixing with single quotes."""
    json_str = "{'name': 'John', 'age': 30}"
    expected = '{"name": "John", "age": 30}'
    
    assert fix_invalid_json(json_str) == expected


def test_fix_invalid_json_valid_json():
    """Test fix_invalid_json with already valid JSON."""
    json_str = '{"name": "John", "age": 30}'
    
    assert fix_invalid_json(json_str) == json_str


def test_fix_invalid_json_validate():
    """Ensure the fixed JSON is valid and can be parsed."""
    json_str = "{'name': 'John', 'age': 30}"
    fixed_json = fix_invalid_json(json_str)
    
    parsed_data = json.loads(fixed_json)
    assert parsed_data['name'] == 'John'
    assert parsed_data['age'] == 30