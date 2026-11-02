"""Turn a raw NOTAM feed + flight plan into a short, ranked pilot briefing."""
from __future__ import annotations

from dataclasses import dataclass

from .classifier import HazardClassifier, Prediction
from .parser import Notam, parse, split_feed
from .relevance import Decision, evaluate
from .route import FlightPlan

SEV_RANK = {"HIGH": 0, "UNKNOWN": 1, "MEDIUM": 2, "LOW": 3}
PHASE_RANK = {"DEPARTURE": 0, "ENROUTE": 1, "ARRIVAL": 2, "ALTERNATE": 3, None: 4}


@dataclass
class BriefingItem:
    notam: Notam
    pred: Prediction
    decision: Decision

    def line(self) -> str:
        n, p, d = self.notam, self.pred, self.decision
        when = f"{n.start:%d%H%MZ}-{n.end:%d%H%MZ}" if n.start and n.end else "PERM"
        tag = p.severity if p.source != "uncertain" else "REVIEW"
        return (f"[{tag:6}] {d.phase or '-':9} {n.notam_id or '?':9} {p.category:21} {when:15} {n.text}\n"
                f"           why: {d.reason}; class via {p.source}"
                + (f" ({', '.join(p.evidence)})" if p.source != "qcode" and p.evidence else ""))


def build(feed: str, fp: FlightPlan, clf: HazardClassifier):
    samples = fp.sample()
    kept, dropped = [], []
    for raw in split_feed(feed):
        n = parse(raw)
        d = evaluate(n, fp, samples)
        item = BriefingItem(n, clf.predict(n), d)
        (kept if d.relevant else dropped).append(item)
    kept.sort(key=lambda i: (SEV_RANK[i.pred.severity], PHASE_RANK[i.decision.phase], i.decision.first_contact))
    return kept, dropped


def render(kept, dropped, fp: FlightPlan) -> str:
    out = [f"NOTAM BRIEFING  {fp.departure} -> {fp.destination}  (ALTN {fp.alternate or '-'})",
           f"ETD {fp.etd:%d%H%MZ}  ETA {fp.eta:%d%H%MZ}  {fp.total_nm():.0f} NM  FL{fp.cruise_ft // 100}",
           f"{len(kept)} relevant of {len(kept) + len(dropped)} NOTAMs "
           f"({len(dropped)} filtered as not affecting this flight)", ""]
    out += [i.line() for i in kept]
    return "\n".join(out)
