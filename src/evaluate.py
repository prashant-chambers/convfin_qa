import math
import re
from difflib import SequenceMatcher


def extract_number(string):
    # Regular expression to match a single integer, decimal, or negative number
    number_pattern = r"-?\d+\.?\d*"
    # Search for the first match
    match = re.search(number_pattern, string)
    if match:
        number = match.group()
        # Convert to float or int
        return float(number) if "." in number else int(number)
    else:
        raise ValueError("No number found in the string")


def exact_match(ground_truth: str, prediction: str):
    return ground_truth == prediction


def numerical_match_with_units(ground_truth: str, prediction: str):
    # Strip whitespace and identify the units
    ground_truth = str(ground_truth).strip()
    prediction = str(prediction).strip()

    # Extract numerical values and units
    ground_truth_value = str(extract_number(ground_truth))  # Value without units
    ground_truth_unit = ground_truth.lstrip("0123456789.-").strip()  # Extract unit

    # Fail fast if the ground truth unit is not in prediction
    if ground_truth_unit:
        if not prediction.__contains__(ground_truth_unit):
            return False

    prediction_value = str(extract_number(prediction))  # Value without units

    # Convert to float for numerical comparison
    try:
        ground_truth_float = float(ground_truth_value)
        prediction_float = float(prediction_value)
    except ValueError:
        # If conversion fails, return False (invalid numeric values)
        return False

    # Determine decimal places in the ground truth
    if "." in ground_truth_value:
        decimal_places = len(ground_truth_value.split(".")[1])
        processed_prediction = round(prediction_float, decimal_places)
    else:
        processed_prediction = int(prediction_float)

    # Match the processed prediction against the ground truth
    return math.isclose(ground_truth_float, processed_prediction, rel_tol=1e-1)


def approx_string_match(ground_truth: str, prediction: str, threshold: float = 0.9):
    similarity = SequenceMatcher(None, ground_truth, prediction).ratio()
    return similarity >= threshold
