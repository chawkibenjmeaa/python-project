"""Check that the numbers and percentages in the memo exist in the source data."""
import re

NUMBER = re.compile(r"(?<![\w.])-?\d[\d,]*(?:\.\d+)?%?")


def _numbers(text):
    text = text.replace("10-K", "").replace("10-Q", "")
    return [n.replace(",", "") for n in NUMBER.findall(text)]


def check_memo(memo, source_text):
    """Return verified/unverified numbers and an overall accuracy score (%).

    A memo number matches if a source number rounds to it (e.g. memo "27.6" vs source "27.63").
    Whole numbers below 10 without a % sign are ignored (list numbering, "top 3 risks", ...).
    """
    source_tokens = _numbers(source_text)
    source = [float(n.rstrip("%")) for n in source_tokens]

    verified, unverified = [], []
    for token in dict.fromkeys(_numbers(memo)):  # unique, in order
        is_pct = token.endswith("%")
        clean_token = token.rstrip("%")
        value = float(clean_token)
        decimals = len(clean_token.split(".")[1]) if "." in clean_token else 0

        if not is_pct and decimals == 0 and abs(value) < 10:
            continue

        if any(round(s, decimals) == value for s in source):
            verified.append(token)
        else:
            unverified.append(token)

    total = len(verified) + len(unverified)
    accuracy = round((len(verified) / total) * 100, 1) if total > 0 else 100.0

    return {
        "verified": verified,
        "unverified": unverified,
        "accuracy_pct": accuracy,
    }
