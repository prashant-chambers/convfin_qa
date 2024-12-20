"""Module for loading and processing financial data."""

import json
from collections.abc import Generator
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

current_dir = Path(__file__).parent.parent
prompt_dir = str(current_dir.parent / "prompts")
environment = Environment(loader=FileSystemLoader(prompt_dir), autoescape=True)


def load_financial_data(file_path: str) -> Generator[dict[str, Any]]:
    """
    Load financial data from a JSON file.

    Args:
        file_path (str): Path to the JSON file containing financial data.

    Yields:
        dict[str, Any]: Individual financial data records.
    """
    try:
        with open(file_path, encoding="utf-8") as file:
            data = json.load(file)
            yield from data
    except (OSError, json.JSONDecodeError) as e:
        print(f"Error loading data from {file_path}: {e}")
        yield from []


def load_prompt_template(template_name: str, **kwargs):
    """Loads and renders a prompt template.

    Args:
        template_name (str): Name of the template file without extension.
        **kwargs: Keyword arguments to pass to the template renderer.

    Returns:
        str: The rendered prompt template content.

    Raises:
        TemplateNotFound: If template file does not exist.
        TemplateError: If there are syntax errors in template.
    """
    template = environment.get_template(f"{template_name}.j2")
    content = template.render(**kwargs)
    return content
