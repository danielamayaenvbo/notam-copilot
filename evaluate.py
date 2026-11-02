"""Reproduce every number in the README.

    python evaluate.py            # writes results.json and prints a summary
"""
import json
import time
from collections import Counter

from sklearn.metrics import accuracy_score, f1_score

from notam_copilot.briefing import build
from notam_copilot.classifier import HazardClassifier, rule_baseline
from notam_copilot.parser import parse
from notam_copilot.synth import SEVERITY, scenario, text_corpus

N_SCENARIOS = 200

# 1. Hazard classification on held-out phrasings ---------------------------------
train, test = text_corpus()
Xtr, ytr = zip(*train)
Xte, yte = zip(*test)
clf = HazardClassifier().fit(Xtr, ytr)
clf.save("model.pkl")

preds = [clf.predict_text(t) for t in Xte]
hybrid = [p.category for p in preds]
ml = [clf.predict_model_only(t) for t in Xte]
rules = [rule_baseline(t) for t in Xte]
sev_true = [SEVERITY[c] for c in yte]
sev_pred = [p.severity for p in preds]
confident = [(p, y) for p, y in zip(preds, yte) if p.source == "model"]
cls = {
    "train_size": len(Xtr), "test_size": len(Xte), "categories": len(set(ytr)),
    "model_accuracy": accuracy_score(yte, ml), "model_macro_f1": f1_score(yte, ml, average="macro"),
    "hybrid_accuracy": accuracy_score(yte, hybrid), "hybrid_macro_f1": f1_score(yte, hybrid, average="macro"),
    "rules_accuracy": accuracy_score(yte, rules), "rules_macro_f1": f1_score(yte, rules, average="macro"),
    "severity_accuracy": accuracy_score(sev_true, sev_pred),
    "flagged_uncertain_rate": sum(p.source == "uncertain" for p in preds) / len(preds),
    "accuracy_when_not_flagged": sum(p.category == y for p, y in zip(preds, yte) if p.source != "uncertain")
                                 / max(1, sum(p.source != "uncertain" for p in preds)),
    "model_confident_rate": len(confident) / len(preds),
    "accuracy_when_model_confident": sum(p.category == y for p, y in confident) / max(1, len(confident)),
    "high_severity_recall": sum(1 for s, p in zip(sev_true, preds) if s == "HIGH" and p.severity in ("HIGH", "UNKNOWN"))
                            / sev_true.count("HIGH"),
}

# 2. Route/time/altitude relevance on constructed scenarios ----------------------
tp = fp_ = fn = tn = 0
total_in = total_kept = 0
errors = Counter()
t0 = time.perf_counter()
for s in range(N_SCENARIOS):
    plan, feed = scenario(seed=1000 + s)
    kept, dropped = build("\n".join(r for r, *_ in feed), plan, clf)
    kept_ids = {i.notam.notam_id for i in kept}
    for raw, cat, rel, kind in feed:
        nid = parse(raw).notam_id
        hit = nid in kept_ids
        if rel and hit: tp += 1
        elif rel: fn += 1; errors[f"missed_{kind}"] += 1
        elif hit: fp_ += 1; errors[f"false_alarm_{kind}"] += 1
        else: tn += 1
    total_in += len(feed); total_kept += len(kept)
elapsed = time.perf_counter() - t0
rel = {
    "scenarios": N_SCENARIOS, "notams_evaluated": total_in,
    "recall": tp / (tp + fn), "precision": tp / (tp + fp_),
    "briefing_reduction": 1 - total_kept / total_in,
    "ms_per_flight_briefing": 1000 * elapsed / N_SCENARIOS,
    "errors": dict(errors),
}

results = {"classification": cls, "relevance": rel}
json.dump(results, open("results.json", "w"), indent=2)
print(json.dumps(results, indent=2))
