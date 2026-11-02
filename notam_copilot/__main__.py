"""Command-line briefing.

    python -m notam_copilot --feed examples/feed.txt --dep KPHX --dest KDEN \
        --alt KABQ --etd 2611051500 --fl 350
    python -m notam_copilot --demo        # synthetic feed for a random flight
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path

from .briefing import build, render
from .classifier import HazardClassifier
from .route import FlightPlan
from .synth import scenario, text_corpus

MODEL = Path(__file__).resolve().parent.parent / "model.pkl"


def load_model() -> HazardClassifier:
    if MODEL.exists():
        return HazardClassifier.load(MODEL)
    train, _ = text_corpus()
    clf = HazardClassifier().fit(*zip(*train))
    clf.save(MODEL)
    return clf


def main():
    ap = argparse.ArgumentParser(description="Route-aware NOTAM briefing")
    ap.add_argument("--feed", type=Path)
    ap.add_argument("--dep"); ap.add_argument("--dest"); ap.add_argument("--alt")
    ap.add_argument("--etd", help="YYMMDDHHMM UTC")
    ap.add_argument("--fl", type=int, default=350)
    ap.add_argument("--gs", type=float, default=450)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--show-dropped", action="store_true")
    a = ap.parse_args()

    clf = load_model()
    if a.demo:
        fp, feed = scenario(a.seed)
        text = "\n".join(r for r, *_ in feed)
    else:
        if not (a.feed and a.dep and a.dest and a.etd):
            ap.error("--feed, --dep, --dest and --etd are required (or use --demo)")
        etd = datetime.strptime(a.etd, "%y%m%d%H%M").replace(tzinfo=timezone.utc)
        fp = FlightPlan(a.dep, a.dest, etd, alternate=a.alt, cruise_ft=a.fl * 100, ground_speed_kt=a.gs)
        text = a.feed.read_text()
    kept, dropped = build(text, fp, clf)
    print(render(kept, dropped, fp))
    if a.show_dropped:
        print("\nFILTERED:")
        for i in dropped:
            print(f"  {i.notam.notam_id}  {i.notam.text[:45]:45}  <- {i.decision.reason}")


if __name__ == "__main__":
    main()
