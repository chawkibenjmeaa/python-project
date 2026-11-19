"""Check that the numbers in the memo exist in the source data."""
import re

NUMBER = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")


def _numbers(text):
    text = text.replace("10-K", "").replace("10-Q", "")
    return [n.replace(",", "") for n in NUMBER.findall(text)]


def check_memo(memo, source_text):
    """Return the memo numbers found in the source and those that were not.

    A memo number matches if a source number rounds to it (e.g. memo "27.6" vs source "27.63").
    Whole numbers below 10 are ignored (list numbering, "top 3 risks", ...).
    """
    source = [float(n) for n in _numbers(source_text)]
    verified, unverified = [], []
    for token in dict.fromkeys(_numbers(memo)):  # unique, in order
        value = float(token)
        decimals = len(token.split(".")[1]) if "." in token else 0
        if decimals == 0 and value < 10:
            continue
        if any(round(s, decimals) == value for s in source):
            verified.append(token)
        else:
            unverified.append(token)
    return {"verified": verified, "unverified": unverified}
