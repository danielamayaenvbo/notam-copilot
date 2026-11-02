"""Flight-plan model and 4-D (lat, lon, altitude, time) route sampling.

The aircraft's position along the route is approximated with a simple
climb / cruise / descent profile at constant ground speed. That is enough
to answer "will this aircraft be inside this NOTAM's volume while it is
active?" for pre-flight briefing purposes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta

EARTH_R_NM = 3440.065

# Small aerodrome table (lat, lon). Extend as needed.
AIRPORTS = {
    "KPHX": (33.4343, -112.0116), "KTUS": (32.1161, -110.9410),
    "KLAS": (36.0840, -115.1537), "KLAX": (33.9416, -118.4085),
    "KSAN": (32.7338, -117.1933), "KABQ": (35.0402, -106.6092),
    "KDEN": (39.8561, -104.6737), "KSLC": (40.7899, -111.9791),
    "KELP": (31.8072, -106.3779), "KDFW": (32.8998, -97.0403),
    "KIAH": (29.9902, -95.3368), "KAUS": (30.1945, -97.6699),
    "KORD": (41.9742, -87.9073), "KATL": (33.6407, -84.4277),
    "KPHL": (39.8744, -75.2424), "KJFK": (40.6413, -73.7781),
    "KBOS": (42.3656, -71.0096), "KDCA": (38.8512, -77.0402),
    "KSEA": (47.4502, -122.3088), "KSFO": (37.6213, -122.3790),
    "KMSP": (44.8848, -93.2223), "KSTL": (38.7499, -90.3748),
    "KCLT": (35.2140, -80.9431), "KMIA": (25.7959, -80.2870),
}


def haversine_nm(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_R_NM * math.asin(math.sqrt(h))


def interpolate(a: tuple[float, float], b: tuple[float, float], f: float) -> tuple[float, float]:
    """Great-circle interpolation between a and b at fraction f."""
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    d = 2 * math.asin(math.sqrt(math.sin((lat2 - lat1) / 2) ** 2
                                + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2))
    if d == 0:
        return a
    A = math.sin((1 - f) * d) / math.sin(d)
    B = math.sin(f * d) / math.sin(d)
    x = A * math.cos(lat1) * math.cos(lon1) + B * math.cos(lat2) * math.cos(lon2)
    y = A * math.cos(lat1) * math.sin(lon1) + B * math.cos(lat2) * math.sin(lon2)
    z = A * math.sin(lat1) + B * math.sin(lat2)
    return math.degrees(math.atan2(z, math.hypot(x, y))), math.degrees(math.atan2(y, x))


@dataclass
class RoutePoint:
    lat: float
    lon: float
    alt_ft: float
    time: datetime
    along_nm: float


@dataclass
class FlightPlan:
    departure: str
    destination: str
    etd: datetime
    alternate: str | None = None
    waypoints: list[tuple[float, float]] = field(default_factory=list)  # intermediate fixes
    cruise_ft: int = 35_000
    ground_speed_kt: float = 450.0
    climb_ft_per_nm: float = 300.0
    descent_ft_per_nm: float = 318.0  # ~3 degree path
    field_elev_ft: float = 1_000.0

    def path(self) -> list[tuple[float, float]]:
        return [AIRPORTS[self.departure], *self.waypoints, AIRPORTS[self.destination]]

    def total_nm(self) -> float:
        p = self.path()
        return sum(haversine_nm(a, b) for a, b in zip(p, p[1:]))

    @property
    def eta(self) -> datetime:
        return self.etd + timedelta(hours=self.total_nm() / self.ground_speed_kt)

    def altitude_at(self, along: float, total: float) -> float:
        climb = self.field_elev_ft + along * self.climb_ft_per_nm
        descent = self.field_elev_ft + (total - along) * self.descent_ft_per_nm
        return min(self.cruise_ft, climb, descent)

    def sample(self, step_nm: float = 2.0) -> list[RoutePoint]:
        path, total = self.path(), self.total_nm()
        pts: list[RoutePoint] = []
        done = 0.0
        for a, b in zip(path, path[1:]):
            seg = haversine_nm(a, b)
            n = max(1, int(seg // step_nm))
            for i in range(n + 1):
                f = i / n
                along = done + f * seg
                lat, lon = interpolate(a, b, f)
                t = self.etd + timedelta(hours=along / self.ground_speed_kt)
                pts.append(RoutePoint(lat, lon, self.altitude_at(along, total), t, along))
            done += seg
        return pts
