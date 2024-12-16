import json
import os
import tempfile
import pytest
from pathlib import Path

from fin_qa.data_loader import load_financial_data, load_prompt_template, environment


def test_load_financial_data_valid_json():
    test_data = [
        {"id": 1, "name": "Company A", "revenue": 1000000},
        {"id": 2, "name": "Company B", "revenue": 1500000}
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp_file:
        json.dump(test_data, temp_file)
        temp_file_path = temp_file.name
    
    try:
        loaded_data = list(load_financial_data(temp_file_path))
        
        assert len(loaded_data) == 2
        assert loaded_data[0]['id'] == 1
        assert loaded_data[0]['name'] == "Company A"
        assert loaded_data[1]['revenue'] == 1500000
    finally:
        os.unlink(temp_file_path)


def test_load_financial_data_empty_file():
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp_file:
        json.dump([], temp_file)
        temp_file_path = temp_file.name
    
    try:
        loaded_data = list(load_financial_data(temp_file_path))
        
        assert len(loaded_data) == 0
    finally:
        os.unlink(temp_file_path)


def test_load_financial_data_nonexistent_file(capsys):
    nonexistent_file = "/path/to/nonexistent/file.json"
    
    loaded_data = list(load_financial_data(nonexistent_file))
    
    assert len(loaded_data) == 0
    
    captured = capsys.readouterr()
    assert f"Error loading data from {nonexistent_file}" in captured.out


def test_load_financial_data_invalid_json():
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp_file:
        temp_file.write("This is not valid JSON")
        temp_file_path = temp_file.name
    
    try:
        loaded_data = list(load_financial_data(temp_file_path))
        
        assert len(loaded_data) == 0
    finally:
        os.unlink(temp_file_path)


def test_load_prompt_template():
    template_content = "Hello, {{ name }}!"
    template_name = "test"
    
    prompts_dir = Path(__file__).parent.parent / "prompts"
    prompts_dir.mkdir(exist_ok=True)
    
    template_path = prompts_dir / f"{template_name}.j2"
    try:
        with open(template_path, 'w') as f:
            f.write(template_content)
        
        rendered = load_prompt_template(template_name, name="World")
        
        assert rendered == "Hello, World!"
    finally:
        if template_path.exists():
            template_path.unlink()


def test_load_prompt_template_not_found():
    with pytest.raises(Exception):
        load_prompt_template("nonexistent_template")