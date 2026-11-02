"""Hazard classification for NOTAM free text (E-field).

Hybrid, explainable design:
  1. If the NOTAM carries a known ICAO Q-code, that is authoritative.
  2. Otherwise a TF-IDF (word + character n-gram) logistic-regression model
     classifies the E-text, and reports the n-grams that drove the decision.
  3. When the model is not confident (out-of-vocabulary phrasing), fall back
     to transparent keyword rules; if those don't match either, the NOTAM is
     flagged REVIEW and ranked near the top rather than silently mis-ranked.

A keyword-rule baseline is included for comparison.
"""
from __future__ import annotations

import pickle
import re
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from .parser import Notam
from .synth import SEVERITY

CONFIDENCE_FLOOR = 0.55

# Expand common NOTAM contractions so that "CLSD" and "CLOSED" look alike.
ABBREV = {
    "CLSD": "CLOSED", "U/S": "UNSERVICEABLE", "OTS": "OUT OF SERVICE", "AVBL": "AVAILABLE",
    "UNAVBL": "UNAVAILABLE", "TWY": "TAXIWAY", "RWY": "RUNWAY", "LGT": "LIGHT", "ACFT": "AIRCRAFT",
    "OBST": "OBSTACLE", "ACT": "ACTIVE", "APCH": "APPROACH", "WI": "WITHIN", "BTN": "BETWEEN",
    "VCY": "VICINITY", "AD": "AERODROME", "WIP": "WORK IN PROGRESS", "SVC": "SERVICE",
    "FREQ": "FREQUENCY", "UNMON": "UNMONITORED", "GP": "GLIDE PATH", "LOC": "LOCALIZER",
    "CONST": "CONSTRUCTION", "LTD": "LIMITED", "TFR": "TEMPORARY FLIGHT RESTRICTION",
    "GNSS": "GPS", "BLW": "BELOW", "TKOF": "TAKEOFF", "LDG": "LANDING", "HGT": "HEIGHT",
}
_TOKEN = re.compile(r"[A-Z]+/[A-Z]+|[A-Z]+")


def normalize(text: str) -> str:
    text = text.upper()
    text = re.sub(r"\d+(\.\d+)?", " # ", text)            # numbers carry no category signal
    return " ".join(ABBREV.get(t, t) for t in _TOKEN.findall(text))


@dataclass
class Prediction:
    category: str
    severity: str
    confidence: float
    source: str            # "qcode" | "model" | "uncertain"
    evidence: list[str]


class HazardClassifier:
    def __init__(self):
        self.pipe = Pipeline([
            ("feats", FeatureUnion([
                ("word", TfidfVectorizer(preprocessor=normalize, ngram_range=(1, 2), sublinear_tf=True)),
                ("char", TfidfVectorizer(preprocessor=normalize, analyzer="char_wb", ngram_range=(3, 5),
                                         sublinear_tf=True, min_df=2)),
            ])),
            ("clf", LogisticRegression(C=8.0, max_iter=3000)),
        ])

    def fit(self, texts, labels):
        self.pipe.fit(texts, labels)
        return self

    def predict_text(self, text: str) -> Prediction:
        proba = self.pipe.predict_proba([text])[0]
        classes = self.pipe.classes_
        k = int(np.argmax(proba))
        cat, conf = classes[k], float(proba[k])
        ev = self._evidence(text, k)
        if conf >= CONFIDENCE_FLOOR:
            return Prediction(cat, SEVERITY[cat], conf, "model", ev)
        rule = rule_baseline(text)
        if rule != "UNKNOWN":
            return Prediction(rule, SEVERITY[rule], conf, "rules", [f"rule:{rule}"])
        return Prediction(cat, "UNKNOWN", conf, "uncertain", ev)

    def predict_model_only(self, text: str) -> str:
        return str(self.pipe.predict([text])[0])

    def predict(self, n: Notam) -> Prediction:
        if n.qcategory:
            return Prediction(n.qcategory, SEVERITY[n.qcategory], 1.0, "qcode", [f"Q-code {n.qcode}"])
        return self.predict_text(n.text)

    def _evidence(self, text: str, k: int, top: int = 3) -> list[str]:
        """Word n-grams in this NOTAM with the largest positive weight for the predicted class."""
        word = self.pipe.named_steps["feats"].transformer_list[0][1]
        n_word = len(word.vocabulary_)
        coef = self.pipe.named_steps["clf"].coef_[k][:n_word]
        x = word.transform([text]).toarray()[0]
        contrib = x * coef
        idx = np.argsort(contrib)[::-1][:top]
        names = word.get_feature_names_out()
        return [names[i] for i in idx if contrib[i] > 0]

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.pipe, f)

    @classmethod
    def load(cls, path):
        obj = cls()
        with open(path, "rb") as f:
            obj.pipe = pickle.load(f)
        return obj


# --- baseline -------------------------------------------------------------
RULES = [
    ("GNSS_INTERFERENCE", r"\bGPS\b|GNSS|JAMM"),
    ("ILS_OUTAGE", r"\bILS\b|\bGP\b|GLIDE|LOC\b|LOCALIZER"),
    ("AIRSPACE_RESTRICTION", r"TFR|RESTRICT|DANGER AREA|MOA|PROHIBITED"),
    ("OBSTACLE", r"OBST|CRANE|TOWER|MAST"),
    ("LIGHTING", r"LGT|REIL|PAPI|ALS\b|LIGHTS"),
    ("NAVAID_OUTAGE", r"VOR|DME|TACAN|NDB"),
    ("TWY_CLOSURE", r"TWY|TAXIWAY"),
    ("RWY_CLOSURE", r"RWY .*CLSD|RWY .*CLOSED"),
    ("FUEL", r"FUEL|AVGAS|JET A"),
    ("FREQ_CHANGE", r"FREQ|ATIS|CTAF"),
    ("WILDLIFE", r"BIRD|WILDLIFE|DEER"),
    ("AD_SERVICE", r"AD CLSD|TWR|CUSTOMS|ARFF|PPR"),
]


def rule_baseline(text: str) -> str:
    for cat, pat in RULES:
        if re.search(pat, text.upper()):
            return cat
    return "UNKNOWN"
