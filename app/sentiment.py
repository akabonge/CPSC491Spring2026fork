from typing import Dict, Literal, TypedDict
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()

SentimentLabel = Literal["positive", "neutral", "negative"]

class SentimentScore(TypedDict):
    neg: float
    neu: float
    pos: float
    compound: float
    label: SentimentLabel

def score_sentiment(text: str) -> SentimentScore:
    """
    Deterministic sentiment scoring using VADER.
    Adds a stable label for downstream aggregation/eval.
    Thresholds: compound >= 0.05 => positive, <= -0.05 => negative, else neutral.
    """
    scores: Dict[str, float] = _analyzer.polarity_scores(text or "")
    c = float(scores.get("compound", 0.0))

    if c >= 0.05:
        label: SentimentLabel = "positive"
    elif c <= -0.05:
        label = "negative"
    else:
        label = "neutral"

    return {
        "neg": float(scores["neg"]),
        "neu": float(scores["neu"]),
        "pos": float(scores["pos"]),
        "compound": c,
        "label": label,
    }
