"""Parse ICAO-format NOTAMs into structured records.

Handles the standard ICAO layout:

    A1234/26 NOTAMN
    Q) KZAB/QMRLC/IV/NBO/A/000/999/3326N11201W005
    A) KPHX B) 2611031400 C) 2611032200
    E) RWY 08/26 CLSD DUE WIP
    F) SFC G) FL180

Fields that are missing (common in US domestic NOTAMs, which often have
no Q-line) are left as None so downstream code can fall back to the
aerodrome location or to the text classifier.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

# Q-code subject (letters 2-3) -> category. Subset of ICAO Doc 8126 codes.
QCODE_SUBJECT = {
    "MR": "RWY_CLOSURE",  # runway
    "MX": "TWY_CLOSURE",  # taxiway
    "IC": "ILS_OUTAGE", "IL": "ILS_OUTAGE", "IG": "ILS_OUTAGE",
    "NV": "NAVAID_OUTAGE", "ND": "NAVAID_OUTAGE", "NT": "NAVAID_OUTAGE",
    "RT": "AIRSPACE_RESTRICTION", "RR": "AIRSPACE_RESTRICTION",
    "RD": "AIRSPACE_RESTRICTION", "RP": "AIRSPACE_RESTRICTION",
    "OB": "OBSTACLE", "OL": "OBSTACLE",
    "LR": "LIGHTING", "LA": "LIGHTING", "LP": "LIGHTING",
    "GW": "GNSS_INTERFERENCE", "GA": "GNSS_INTERFERENCE",
    "FA": "AD_SERVICE", "FU": "FUEL",
    "CA": "FREQ_CHANGE", "CO": "FREQ_CHANGE",
    "WB": "WILDLIFE",
}

_NOTAM_ID = re.compile(r"^\s*([A-Z]\d{4}/\d{2})\s+NOTAM([NRC])", re.M)
_Q = re.compile(
    r"Q\)\s*(?P<fir>[A-Z]{4})/(?P<code>Q[A-Z]{4})/(?P<traffic>[IVK]+)/"
    r"(?P<purpose>[NBOMK]+)/(?P<scope>[AEWK]+)/(?P<lower>\d{3})/(?P<upper>\d{3})/"
    r"(?P<lat>\d{4})(?P<ns>[NS])(?P<lon>\d{5})(?P<ew>[EW])(?P<radius>\d{3})"
)
_A = re.compile(r"A\)\s*([A-Z0-9 ]+?)\s+(?=B\))")
_B = re.compile(r"B\)\s*(\d{10})")
_C = re.compile(r"C\)\s*(\d{10}|PERM)(\s*EST)?")
_E = re.compile(r"E\)\s*(.+?)(?=\s+[FG]\)|\Z)", re.S)
_F = re.compile(r"F\)\s*(SFC|GND|\d+\s*FT\s*(?:AMSL|AGL)?|FL\d{3})")
_G = re.compile(r"G\)\s*(UNL|\d+\s*FT\s*(?:AMSL|AGL)?|FL\d{3})")


@dataclass
class Notam:
    raw: str
    notam_id: Optional[str] = None
    kind: Optional[str] = None  # N=new, R=replace, C=cancel
    fir: Optional[str] = None
    qcode: Optional[str] = None
    locations: list[str] = field(default_factory=list)
    start: Optional[datetime] = None
    end: Optional[datetime] = None
    end_estimated: bool = False
    text: str = ""
    lower_ft: Optional[int] = None
    upper_ft: Optional[int] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    radius_nm: Optional[float] = None

    @property
    def qcategory(self) -> Optional[str]:
        if self.qcode and len(self.qcode) == 5:
            return QCODE_SUBJECT.get(self.qcode[1:3])
        return None


def _dt(s: str) -> datetime:
    return datetime.strptime(s, "%y%m%d%H%M").replace(tzinfo=timezone.utc)


def _alt_ft(s: str) -> int:
    s = s.strip().upper()
    if s in ("SFC", "GND"):
        return 0
    if s == "UNL":
        return 99_900
    if s.startswith("FL"):
        return int(s[2:]) * 100
    return int(re.match(r"\d+", s).group())


def _coord(deg_min: str, hemi: str, deg_digits: int) -> float:
    deg = int(deg_min[:deg_digits])
    minutes = int(deg_min[deg_digits:])
    v = deg + minutes / 60.0
    return -v if hemi in "SW" else v


def parse(raw: str) -> Notam:
    n = Notam(raw=raw)
    if m := _NOTAM_ID.search(raw):
        n.notam_id, n.kind = m.group(1), m.group(2)
    if m := _Q.search(raw):
        n.fir, n.qcode = m["fir"], m["code"]
        n.lower_ft = int(m["lower"]) * 100
        n.upper_ft = int(m["upper"]) * 100
        n.lat = _coord(m["lat"], m["ns"], 2)
        n.lon = _coord(m["lon"], m["ew"], 3)
        n.radius_nm = float(int(m["radius"]))
    if m := _A.search(raw):
        n.locations = m.group(1).split()
    if m := _B.search(raw):
        n.start = _dt(m.group(1))
    if m := _C.search(raw):
        n.end = None if m.group(1) == "PERM" else _dt(m.group(1))
        n.end_estimated = bool(m.group(2))
    if m := _E.search(raw):
        n.text = " ".join(m.group(1).split())
    # F/G give the precise vertical limits and override the Q-line's coarse FLs.
    if m := _F.search(raw):
        n.lower_ft = _alt_ft(m.group(1))
    if m := _G.search(raw):
        n.upper_ft = _alt_ft(m.group(1))
    return n


def split_feed(feed: str) -> list[str]:
    """Split a feed containing many NOTAMs into individual raw NOTAM strings."""
    starts = [m.start() for m in _NOTAM_ID.finditer(feed)]
    starts.append(len(feed))
    return [feed[a:b].strip() for a, b in zip(starts, starts[1:]) if feed[a:b].strip()]
