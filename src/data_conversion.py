"""Module for converting financial data to different formats."""


def convert_to_markdown_table(data: list[list[str | int | float]]) -> str:
    """
    Convert a 2D list to a markdown-formatted table.

    Args:
        data (List[List[Union[str, int, float]]]): 2D list of data to convert.

    Returns:
        str: Markdown-formatted table.
    """
    # Find the maximum length of each column
    max_lengths = [max(len(str(item)) for item in col) for col in zip(*data)]

    # Create table rows
    table_rows = []
    for row in data:
        table_row = "|"
        for i, item in enumerate(row):
            table_row += f" {str(item).ljust(max_lengths[i])} |"
        table_rows.append(table_row)

    # Create header separator row
    header_separator = "|" + "|".join(["-" * length for length in max_lengths]) + "|"

    # Combine rows into a complete table
    table = "\n".join([table_rows[0], header_separator] + table_rows[1:])
    return table


def convert_to_paragraph(data: list[str]) -> str:
    """
    Convert a list of strings to a single paragraph.

    Args:
        data (List[str]): List of strings to join.

    Returns:
        str: Joined paragraph.
    """
    return " ".join(data)


def fix_invalid_json(json_string):
    # Replace single quotes with double quotes
    json_string = json_string.replace("'", '"')
    return json_string
