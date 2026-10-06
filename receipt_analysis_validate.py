"""Validation function for the sample query receipt_analysis.txt.

Returns True iff the agent's answer matches the expected receipt-analysis result
for the seeded gist data (Blue Moon Cafe / $87.45 / Haifa). Tolerates:
- string case and surrounding whitespace
- float drift up to 0.01 on monetary fields and 0.05 on the percentage
- extra fields in the answer dict
"""


def receipt_analysis_answer(answer: dict) -> bool:
    if not isinstance(answer, dict):
        return False

    required = {
        "merchant",
        "receipt_total",
        "historical_total",
        "percentage_of_historical",
        "top_customer_city",
    }
    if not required.issubset(answer.keys()):
        return False

    if not isinstance(answer["merchant"], str):
        return False
    if answer["merchant"].strip().lower() != "blue moon cafe":
        return False

    try:
        receipt_total = float(answer["receipt_total"])
        historical_total = float(answer["historical_total"])
        pct = float(answer["percentage_of_historical"])
    except (TypeError, ValueError):
        return False

    if abs(receipt_total - 87.45) > 0.01:
        return False
    if abs(historical_total - 1240.30) > 0.01:
        return False
    if abs(pct - 7.05) > 0.05:
        return False

    if not isinstance(answer["top_customer_city"], str):
        return False
    if answer["top_customer_city"].strip().lower() != "haifa":
        return False

    return True
