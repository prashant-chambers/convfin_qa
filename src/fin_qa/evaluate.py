import math
import re


def extract_number(string):
    """Extracts the first number from a string, handling currency and comma formatting.

    Args:
        string: Input string containing a number, optionally with currency symbol and commas.

    Returns:
        float: The extracted number value. Returns 0 if no valid number is found.

    Example:
        >>> extract_number("$1,234.56")
        1234.56
        >>> extract_number("Revenue: -123,456")
        -123456
    """
    number_pattern = r"-?(?:\$)?[\d,]+\.?\d*"
    match = re.search(number_pattern, string)
    if match:
        # Remove currency symbol and commas
        number_str = match.group().replace("$", "").replace(",", "")

        # Convert to float or int
        try:
            n = float(number_str) if "." in number_str else int(number_str)
        except ValueError:
            n = 0
        return n


def exact_match(ground_truth: str, prediction: str):
    """Checks if prediction exactly matches the ground truth string.

    Args:
        ground_truth: The expected correct string.
        prediction: The predicted string to compare.

    Returns:
        bool: True if strings match exactly, False otherwise.
    """
    return ground_truth == prediction


def numerical_match(ground_truth: str, prediction: str):
    """Compares numerical values extracted from strings within a tolerance.

    Args:
        ground_truth: String containing the expected correct number.
        prediction: String containing the predicted number.

    Returns:
        bool: True if numbers match within tolerance, False otherwise.

    Note:
        Uses absolute tolerance of 0.5 to allow for minor rounding differences.
    """
    ground_truth = extract_number(str(ground_truth).strip())
    ground_truth_value = float(ground_truth) if ground_truth else 0
    prediction = extract_number(str(prediction).strip())
    prediction_value = float(prediction) if prediction else 0
    return math.isclose(ground_truth_value, prediction_value, rel_tol=0, abs_tol=0.5)
