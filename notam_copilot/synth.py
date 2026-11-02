"""Synthetic NOTAM corpus + flight scenarios for evaluation.

Real NOTAM feeds aren't redistributable with ground-truth labels, so this
module generates ICAO-format NOTAMs from phrasing templates modelled on
FAA/ICAO conventions. Each category has several templates; the last two
of every category are reserved for the test split so the classifier is
scored on phrasings it never saw during training.

Scenarios place NOTAMs relative to a flight plan *by construction*:
true hits sit on the route, in the altitude band, while active; decoys
violate exactly one of lateral / vertical / time constraints. That gives
exact ground truth for the relevance engine.
"""
from __future__ import annotations

import math
import random
from datetime import datetime, timedelta, timezone

from .route import AIRPORTS, FlightPlan, interpolate

SEVERITY = {
    "RWY_CLOSURE": "HIGH", "AIRSPACE_RESTRICTION": "HIGH", "GNSS_INTERFERENCE": "HIGH",
    "ILS_OUTAGE": "HIGH", "NAVAID_OUTAGE": "MEDIUM", "TWY_CLOSURE": "MEDIUM",
    "OBSTACLE": "MEDIUM", "LIGHTING": "MEDIUM", "WILDLIFE": "MEDIUM",
    "AD_SERVICE": "LOW", "FUEL": "LOW", "FREQ_CHANGE": "LOW",
}
QCODES = {
    "RWY_CLOSURE": "QMRLC", "TWY_CLOSURE": "QMXLC", "ILS_OUTAGE": "QICAS",
    "NAVAID_OUTAGE": "QNVAS", "AIRSPACE_RESTRICTION": "QRTCA", "OBSTACLE": "QOBCE",
    "LIGHTING": "QLRAS", "GNSS_INTERFERENCE": "QGWXX", "WILDLIFE": "QWBLW",
    "AD_SERVICE": "QFAAH", "FUEL": "QFUAU", "FREQ_CHANGE": "QCAXX",
}
AREA = {"AIRSPACE_RESTRICTION", "GNSS_INTERFERENCE", "OBSTACLE", "NAVAID_OUTAGE", "WILDLIFE"}

TEMPLATES = {
    "RWY_CLOSURE": [
        "RWY {rwy} CLSD", "RWY {rwy} CLSD DUE WIP", "RWY {rwy} CLOSED TO ALL ACFT EXC TAXIING",
        "RWY {rwy} CLSD DUE SNOW REMOVAL", "RWY {rwy} CLSD FOR RESURFACING",
        "AD RWY {rwy} NOT AVBL FOR LDG OR TKOF", "RWY {rwy} CLSD DLY {h1}-{h2}",
    ],
    "TWY_CLOSURE": [
        "TWY {twy} CLSD", "TWY {twy} BTN TWY {twy2} AND RWY {rwy} CLSD", "TWY {twy} CLSD DUE CONST",
        "TWY {twy} WIP ADJ, TWY CLSD", "TAXIWAY {twy} CLOSED BETWEEN {twy2} AND {twy3}",
        "TWY {twy} NOT AVBL", "TWY {twy}{n} CLSD TO ACFT WINGSPAN MORE THAN 118FT",
    ],
    "ILS_OUTAGE": [
        "ILS RWY {rwy} U/S", "ILS RWY {rwy} GP U/S", "ILS RWY {rwy} LOC OUT OF SERVICE",
        "RWY {rwy} ILS UNMONITORED", "ILS RWY {rwy} CAT II/III NOT AVBL",
        "GLIDE PATH RWY {rwy} UNSERVICEABLE", "LOCALIZER RWY {rwy} NOT AVAILABLE FOR APCH",
    ],
    "NAVAID_OUTAGE": [
        "{vor} VOR U/S", "{vor} VORTAC OTS", "{vor} VOR/DME UNMON", "{vor} DME U/S",
        "{vor} TACAN AZM UNUSABLE {r1}-{r2} BYD {n}NM", "NDB {ndb} DECOMMISSIONED",
        "VOR {vor} OUT OF SERVICE FOR MAINT",
    ],
    "AIRSPACE_RESTRICTION": [
        "TEMPORARY FLIGHT RESTRICTIONS {n}NM RADIUS SFC-FL{fl} VIP MOVEMENT",
        "AIRSPACE {zone} RESTRICTED AREA ACT SFC-FL{fl}", "TFR WI {n}NM SPACE OPS",
        "DANGER AREA {zone} ACT", "FLIGHT RESTRICTIONS WI AN AREA DEFINED AS {n}NM RADIUS DUE TO FIRE FIGHTING",
        "MOA {zone} ACTIVE MIL TRAINING, NONPARTICIPATING ACFT AVOID",
        "PROHIBITED AREA ESTABLISHED {n}NM RADIUS SPORTING EVENT",
    ],
    "OBSTACLE": [
        "OBST CRANE {n}FT AGL {d}NM {dir} OF AD LGT", "OBST TOWER {n}FT AMSL UNLIT",
        "CRANE ERECTED {d}NM {dir} THR RWY {rwy} HGT {n}FT", "OBST WIND TURBINE {n}FT AGL LGT OUT OF SERVICE",
        "NEW OBST MAST {n}FT AGL", "OBSTACLE LIGHTS ON TOWER {n}FT AGL U/S",
        "TEMPORARY CRANE PENETRATING APCH SFC RWY {rwy}",
    ],
    "LIGHTING": [
        "RWY {rwy} EDGE LGT U/S", "RWY {rwy} REIL OTS", "RWY {rwy} PAPI U/S", "APCH LGT RWY {rwy} U/S",
        "RWY {rwy} CL LGT UNSERVICEABLE", "RWY {rwy} ALS OUT OF SERVICE", "TDZ LIGHTS RWY {rwy} NOT AVBL",
    ],
    "GNSS_INTERFERENCE": [
        "GPS UNRELIABLE WI {n}NM RADIUS FL{fl} AND BLW DUE TESTING", "GNSS SIGNAL MAY BE LOST OR DEGRADED",
        "GPS INTERFERENCE TESTING WI {n}NM", "GNSS JAMMING REPORTED IN AREA",
        "GPS UNRELIABLE AND MAY BE UNAVBL", "RNAV GNSS APPROACHES NOT AVAILABLE DUE TO SIGNAL INTERFERENCE",
        "SATELLITE NAVIGATION SIGNAL DEGRADATION EXPECTED WI {n}NM",
    ],
    "WILDLIFE": [
        "BIRD ACT VCY AD", "INCREASED BIRD ACT IN VCY OF RWY {rwy}", "FLOCKS OF BIRDS REPORTED ON APCH",
        "WILDLIFE HAZARD ON AD", "DEER ACTIVITY NEAR RWY {rwy}",
        "MIGRATORY WATERFOWL SIGHTED UP TO {n}FT", "HEAVY BIRD CONCENTRATION REPORTED NEAR AIRFIELD",
    ],
    "AD_SERVICE": [
        "AD CLSD TO NON-SKED TFC", "ATC TWR HR OF OPS {h1}-{h2}", "CUSTOMS NOT AVBL",
        "ARFF INDEX REDUCED TO CAT {n}", "AD HOURS OF SERVICE CHANGED", "PPR REQUIRED FOR GA",
        "FIRE AND RESCUE SERVICE DOWNGRADED",
    ],
    "FUEL": [
        "FUEL JET A NOT AVBL", "AVGAS 100LL U/S", "FUEL SVC LTD", "JET A1 SUPPLY LIMITED PPR",
        "NO FUEL AVBL", "REFUELING UNAVAILABLE UNTIL FURTHER NOTICE", "FUEL TRUCK OUT OF SERVICE",
    ],
    "FREQ_CHANGE": [
        "GND FREQ {f} U/S USE {f2}", "ATIS FREQ CHANGED TO {f}", "APP FREQ {f} OTS",
        "CTAF CHANGED TO {f}", "CLNC DEL FREQ {f} NOT AVBL", "TOWER FREQUENCY {f} UNSERVICEABLE",
        "COMMUNICATIONS ON {f} NOT AVAILABLE USE ALTERNATE",
    ],
}
N_TEST_TEMPLATES = 2
FIR = "KZXX"
VORS = ["PXR", "TFD", "BXK", "DRK", "ABQ", "GUP", "INW", "SJN", "TCS", "CIE"]


def _fill(tpl: str, rng: random.Random) -> str:
    r = rng.randint(1, 18)
    side = rng.choice(['', 'L', 'R'])
    recip = {'': '', 'L': 'R', 'R': 'L'}[side]
    return tpl.format(
        rwy=f"{r:02d}{side}/{(r + 18) % 36 or 36:02d}{recip}",
        twy=rng.choice("ABCDEFGHJKMN"), twy2=rng.choice("ABCDEFGHJKMN"), twy3=rng.choice("PQRSTUVWY"),
        n=rng.randint(2, 40), d=rng.choice(["0.5", "1.2", "2", "3.4"]), dir=rng.choice(["N", "NE", "E", "SE", "S", "SW", "W", "NW"]),
        fl=rng.choice(["100", "180", "245", "350", "600"]), vor=rng.choice(VORS), ndb=rng.choice(["XY", "BZ", "QR"]),
        r1=rng.randint(0, 180), r2=rng.randint(181, 359), zone=f"R-{rng.randint(2301, 2399)}",
        h1=f"{rng.randint(0, 11):02d}00", h2=f"{rng.randint(12, 23):02d}00",
        f=f"{rng.randint(118, 135)}.{rng.choice(['05', '2', '45', '9', '775'])}",
        f2=f"{rng.randint(118, 135)}.{rng.choice(['1', '35', '6'])}",
    )


def text_corpus(n_per_template: int = 40, seed: int = 7):
    """Return (train, test) lists of (text, category) with disjoint templates."""
    rng = random.Random(seed)
    train, test = [], []
    for cat, tpls in TEMPLATES.items():
        for i, t in enumerate(tpls):
            dest = test if i >= len(tpls) - N_TEST_TEMPLATES else train
            dest.extend((_fill(t, rng), cat) for _ in range(n_per_template))
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def _fmt_coord(lat: float, lon: float) -> str:
    def dm(v, w):
        v = abs(v); d = int(v); m = round((v - d) * 60)
        if m == 60: d, m = d + 1, 0
        return f"{d:0{w}d}{m:02d}"
    return f"{dm(lat, 2)}{'N' if lat >= 0 else 'S'}{dm(lon, 3)}{'E' if lon >= 0 else 'W'}"


def _offset(p: tuple[float, float], dist_nm: float, bearing_deg: float) -> tuple[float, float]:
    lat1, lon1, b = math.radians(p[0]), math.radians(p[1]), math.radians(bearing_deg)
    d = dist_nm / 3440.065
    lat2 = math.asin(math.sin(lat1) * math.cos(d) + math.cos(lat1) * math.sin(d) * math.cos(b))
    lon2 = lon1 + math.atan2(math.sin(b) * math.sin(d) * math.cos(lat1), math.cos(d) - math.sin(lat1) * math.sin(lat2))
    return math.degrees(lat2), math.degrees(lon2)


def render(seq: int, cat: str, text: str, loc: str, start: datetime, end: datetime,
           lat: float, lon: float, radius: int, lower_ft: int, upper_ft: int, scope: str) -> str:
    lo, hi = lower_ft // 100, min(upper_ft // 100, 999)
    f = "SFC" if lower_ft == 0 else f"{lower_ft}FT AMSL" if lower_ft < 18_000 else f"FL{lo:03d}"
    g = "UNL" if upper_ft >= 99_900 else f"{upper_ft}FT AMSL" if upper_ft < 18_000 else f"FL{hi:03d}"
    return (f"A{seq:04d}/26 NOTAMN\n"
            f"Q) {FIR}/{QCODES[cat]}/IV/NBO/{scope}/{lo:03d}/{hi:03d}/{_fmt_coord(lat, lon)}{radius:03d}\n"
            f"A) {loc} B) {start:%y%m%d%H%M} C) {end:%y%m%d%H%M}\n"
            f"E) {text}\nF) {f} G) {g}")


def scenario(seed: int, n_hits: int = 12, n_decoys: int = 48):
    """Build one flight plan plus a NOTAM feed with exact relevance labels.

    Returns (flight_plan, [(raw_notam, category, is_relevant, decoy_type)]).
    """
    rng = random.Random(seed)
    dep, dest, alt = rng.sample([k for k in AIRPORTS], 3)
    while not (300 < FlightPlan(dep, dest, datetime(2026, 1, 1, tzinfo=timezone.utc)).total_nm() < 1800):
        dep, dest, alt = rng.sample([k for k in AIRPORTS], 3)
    etd = datetime(2026, 11, rng.randint(3, 27), rng.randint(0, 23), 0, tzinfo=timezone.utc)
    fp = FlightPlan(dep, dest, etd, alternate=alt, cruise_ft=rng.choice([31_000, 35_000, 39_000]))
    pts = fp.sample(step_nm=2.0)
    total = pts[-1].along_nm
    cats = list(TEMPLATES)
    feed, seq = [], rng.randint(100, 5000)

    def tpl_text(cat):
        return _fill(rng.choice(TEMPLATES[cat]), rng)

    def ad_notam(relevant: bool):
        nonlocal seq
        seq += 1
        cat = rng.choice([c for c in cats if c not in AREA])
        role, t = rng.choice([(dep, fp.etd), (dest, fp.eta), (alt, fp.eta)])
        if relevant:
            start = t - timedelta(hours=rng.uniform(0.5, 6)); end = t + timedelta(hours=rng.uniform(0.5, 6))
            loc, kind = role, "hit"
        elif rng.random() < 0.5:   # right aerodrome, wrong time
            start = t + timedelta(hours=rng.uniform(5, 30)); end = start + timedelta(hours=rng.uniform(1, 8))
            loc, kind = role, "ad_time"
        else:                       # aerodrome not used by this flight
            loc = rng.choice([k for k in AIRPORTS if k not in (dep, dest, alt)])
            start = t - timedelta(hours=2); end = t + timedelta(hours=2); kind = "ad_other"
        lat, lon = AIRPORTS[loc]
        return render(seq, cat, tpl_text(cat), loc, start, end, lat, lon, 5, 0, 3_000, "A"), cat, relevant, kind

    def area_notam(relevant: bool):
        nonlocal seq
        seq += 1
        cat = rng.choice(sorted(AREA))
        # pick an en-route point, away from the terminal areas
        p = rng.choice([q for q in pts if 0.25 * total < q.along_nm < 0.75 * total])
        radius = rng.randint(3, 25)
        start = p.time - timedelta(hours=rng.uniform(0.2, 4)); end = p.time + timedelta(hours=rng.uniform(0.2, 4))
        lower, upper = 0, rng.choice([39_000 + 2_000, 60_000])
        i = pts.index(p)
        nxt = pts[min(i + 1, len(pts) - 1)] if i + 1 < len(pts) else pts[i - 1]
        y = math.sin(math.radians(nxt.lon - p.lon)) * math.cos(math.radians(nxt.lat))
        x = (math.cos(math.radians(p.lat)) * math.sin(math.radians(nxt.lat))
             - math.sin(math.radians(p.lat)) * math.cos(math.radians(nxt.lat)) * math.cos(math.radians(nxt.lon - p.lon)))
        track = math.degrees(math.atan2(y, x))
        bearing = rng.uniform(0, 360)
        dist = rng.uniform(0, radius * 0.8)
        kind = "hit"
        if not relevant:
            kind = rng.choice(["lateral", "vertical", "time"])
            if kind == "lateral":
                dist = radius + 10 + rng.uniform(15, 120)   # clear of corridor
                bearing = track + rng.choice([90, -90])     # perpendicular to the route
            elif kind == "vertical":
                lower, upper = 0, rng.choice([5_000, 10_000, 18_000])  # below cruise
            else:
                start = p.time + timedelta(hours=rng.uniform(3, 30)); end = start + timedelta(hours=rng.uniform(1, 6))
        lat, lon = _offset((p.lat, p.lon), dist, bearing)
        return (render(seq, cat, tpl_text(cat), FIR, start, end, lat, lon, radius, lower, upper, "W"),
                cat, relevant, kind)

    for i in range(n_hits):
        feed.append(ad_notam(True) if i % 2 else area_notam(True))
    for i in range(n_decoys):
        feed.append(ad_notam(False) if i % 2 else area_notam(False))
    rng.shuffle(feed)
    return fp, feed
