"""Decide whether a NOTAM actually affects a given flight.

A NOTAM is relevant when the aircraft will be inside the NOTAM's
3-D volume while it is active (area NOTAMs), or when it concerns the
departure / destination / alternate aerodrome during the time the
aircraft uses it. Every decision returns a human-readable reason so the
pilot (or a reviewer) can see *why* a NOTAM was kept or dropped.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from .parser import Notam
from .route import AIRPORTS, FlightPlan, RoutePoint, haversine_nm

CORRIDOR_NM = 10.0          # lateral buffer around the route centreline
ALT_BUFFER_FT = 1_000.0     # vertical buffer around the NOTAM volume
TIME_BUFFER = timedelta(minutes=30)
AD_WINDOWS = {               # (before, after) the aircraft uses the aerodrome
    "DEPARTURE": (timedelta(hours=1), timedelta(hours=1)),
    "ARRIVAL": (timedelta(hours=1), timedelta(hours=1)),
    "ALTERNATE": (timedelta(hours=1), timedelta(hours=3)),
}


@dataclass
class Decision:
    relevant: bool
    phase: str | None       # DEPARTURE / ENROUTE / ARRIVAL / ALTERNATE
    reason: str
    first_contact: datetime | None = None


def _overlaps(s1, e1, s2, e2) -> bool:
    e1 = e1 or datetime.max.replace(tzinfo=s2.tzinfo)  # PERM
    return s1 <= e2 and s2 <= e1


def _aerodrome_decision(n: Notam, fp: FlightPlan) -> Decision | None:
    roles = {fp.departure: ("DEPARTURE", fp.etd), fp.destination: ("ARRIVAL", fp.eta)}
    if fp.alternate:
        roles[fp.alternate] = ("ALTERNATE", fp.eta)
    for loc in n.locations:
        if loc in roles:
            phase, t = roles[loc]
            before, after = AD_WINDOWS[phase]
            if n.start and _overlaps(n.start, n.end, t - before, t + after):
                return Decision(True, phase, f"{loc} is the {phase.lower()} aerodrome; NOTAM active within "
                                f"-{before.seconds // 3600}h/+{after.seconds // 3600}h of {t:%d%H%MZ}", t)
            return Decision(False, phase, f"{loc} is the {phase.lower()} aerodrome but NOTAM is not active "
                            f"when the aircraft is there")
    return None


def evaluate(n: Notam, fp: FlightPlan, samples: list[RoutePoint] | None = None) -> Decision:
    if n.kind == "C":
        return Decision(False, None, "cancellation NOTAM")

    ad = _aerodrome_decision(n, fp)
    if ad and (ad.relevant or n.lat is None):
        return ad

    if n.lat is None:
        # No geometry: fall back to the aerodrome position if we know it.
        loc = next((l for l in n.locations if l in AIRPORTS), None)
        if loc is None:
            return Decision(False, None, "no location information")
        n.lat, n.lon = AIRPORTS[loc]
        n.radius_nm = n.radius_nm or 5.0

    samples = samples or fp.sample()
    lo = (n.lower_ft or 0) - ALT_BUFFER_FT
    hi = (n.upper_ft if n.upper_ft is not None else 99_900) + ALT_BUFFER_FT
    reach = (n.radius_nm or 0) + CORRIDOR_NM
    lateral = vertical = False
    for p in samples:
        if haversine_nm((p.lat, p.lon), (n.lat, n.lon)) > reach:
            continue
        lateral = True
        if not (lo <= p.alt_ft <= hi):
            continue
        vertical = True
        if n.start and _overlaps(n.start - TIME_BUFFER, (n.end + TIME_BUFFER) if n.end else None, p.time, p.time):
            total = samples[-1].along_nm
            phase = "DEPARTURE" if p.along_nm < 0.15 * total else "ARRIVAL" if p.along_nm > 0.85 * total else "ENROUTE"
            return Decision(True, phase, f"route passes {haversine_nm((p.lat, p.lon), (n.lat, n.lon)):.0f} NM from "
                            f"centre at {p.alt_ft:,.0f} ft, {p.time:%d%H%MZ}, while active", p.time)
    if not lateral:
        return Decision(False, None, f"route never within {reach:.0f} NM of NOTAM area")
    if not vertical:
        return Decision(False, None, f"aircraft altitude outside {n.lower_ft}-{n.upper_ft} ft when nearby")
    return Decision(False, None, "aircraft passes the area only while the NOTAM is inactive")
