from datetime import datetime, timezone

from notam_copilot.parser import parse, split_feed
from notam_copilot.relevance import evaluate
from notam_copilot.route import FlightPlan, haversine_nm, AIRPORTS
from notam_copilot.synth import render

UTC = timezone.utc
FP = FlightPlan("KPHX", "KDEN", datetime(2026, 11, 5, 15, tzinfo=UTC), alternate="KABQ")

RAW = """A0101/26 NOTAMN
Q) KZAB/QMRLC/IV/NBO/A/000/999/3326N11201W005
A) KPHX B) 2611051200 C) 2611051900 EST
E) RWY 08/26 CLSD DUE WIP
F) SFC G) FL180"""


def test_parse_fields():
    n = parse(RAW)
    assert n.notam_id == "A0101/26" and n.kind == "N"
    assert n.qcategory == "RWY_CLOSURE"
    assert n.locations == ["KPHX"] and n.end_estimated
    assert abs(n.lat - 33.433) < 0.01 and abs(n.lon + 112.017) < 0.01
    assert n.lower_ft == 0 and n.upper_ft == 18_000
    assert n.text == "RWY 08/26 CLSD DUE WIP"


def test_split_feed():
    assert len(split_feed(RAW + "\n\n" + RAW.replace("A0101", "A0102"))) == 2


def test_distance_sanity():
    # PHX-DEN great circle is ~520 NM
    assert 500 < haversine_nm(AIRPORTS["KPHX"], AIRPORTS["KDEN"]) < 540


def test_departure_notam_relevant_only_when_active():
    assert evaluate(parse(RAW), FP).relevant
    late = RAW.replace("2611051200", "2611071200").replace("2611051900", "2611071900")
    assert not evaluate(parse(late), FP).relevant


def _area(lat, lon, lower, upper, start, end):
    return parse(render(1, "AIRSPACE_RESTRICTION", "TFR WI 5NM", "KZAB", start, end, lat, lon, 5, lower, upper, "W"))


def test_area_lateral_vertical_time():
    mid = FP.sample()[len(FP.sample()) // 2]
    s, e = datetime(2026, 11, 5, 14, tzinfo=UTC), datetime(2026, 11, 5, 18, tzinfo=UTC)
    assert evaluate(_area(mid.lat, mid.lon, 0, 60_000, s, e), FP).relevant
    assert not evaluate(_area(mid.lat, mid.lon, 0, 10_000, s, e), FP).relevant           # below cruise
    assert not evaluate(_area(mid.lat + 3, mid.lon, 0, 60_000, s, e), FP).relevant       # ~180 NM away
    s2, e2 = datetime(2026, 11, 6, 14, tzinfo=UTC), datetime(2026, 11, 6, 18, tzinfo=UTC)
    assert not evaluate(_area(mid.lat, mid.lon, 0, 60_000, s2, e2), FP).relevant         # inactive


def test_cancellation_ignored():
    assert not evaluate(parse(RAW.replace("NOTAMN", "NOTAMC")), FP).relevant
