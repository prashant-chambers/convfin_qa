import math
from difflib import SequenceMatcher


def exact_match(ground_truth: str, prediction: str):
    return ground_truth == prediction


def numerical_match_with_units(ground_truth: str, prediction: str):
    # Strip whitespace and identify the units
    ground_truth = str(ground_truth).strip()
    prediction = str(prediction).strip()

    # Extract numerical values and units
    ground_truth_value = ground_truth.strip("%$")  # Value without units
    ground_truth_unit = ground_truth.lstrip("0123456789.-")  # Extract unit

    prediction_value = prediction.strip("%$")  # Value without units
    prediction_unit = prediction.lstrip("0123456789.-")  # Extract unit

    # Check if the units match
    if ground_truth_unit != prediction_unit:
        return False

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
    return math.isclose(ground_truth_float, processed_prediction, rel_tol=1e-2)


def approx_string_match(ground_truth: str, prediction: str, threshold: float = 0.9):
    similarity = SequenceMatcher(None, ground_truth, prediction).ratio()
    return similarity >= threshold
