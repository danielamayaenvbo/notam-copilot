# NOTAM Copilot

Route-aware, explainable NOTAM briefing for pilots.

## The problem

Before every flight, pilots receive a stack of NOTAMs (Notices to Air Missions): runway closures, navaid outages, airspace restrictions, GPS interference tests and so on. Most of them do not affect that particular flight, they change while the aircraft is airborne, and important ones get buried.

Honeywell Aerospace and ASU's School of Computing and Augmented Intelligence named this as a target of their 2026–27 **AI Challenge for Pilot Assist**: AI that summarizes NOTAMs, aligns them with flight data and produces real-time alerts. Honeywell's own writing on AI in aviation sets the bar for how: it has to support the pilot rather than replace them, and its behaviour has to be explainable and verifiable, because a black box that is usually right cannot be certified.

Sources:
- ASU News, "'AI Challenge for Pilot Assist' takes flight in Fulton Schools" (June 2026)
- AZFamily/KOLD, "ASU students will compete to put AI in the cockpit" (Aug 2026)
- Honeywell Aerospace Insights, "The growing use of AI in aviation: Supporting pilots and protecting trust" (Mar 2026)

## What it does

```
raw NOTAM feed ──► parser ──► relevance engine (4-D) ──► hazard classifier ──► ranked briefing
                   ICAO        lat / lon / altitude /      Q-code → ML → rules     severity × phase,
                   fields      time vs. flight profile     → human REVIEW          with a reason per line
```

1. **Parser** (`parser.py`): ICAO NOTAM fields: ID, Q-line (code, vertical limits, centre, radius), A/B/C/E/F/G, `PERM` and `EST` end times, cancellations.
2. **Flight profile** (`route.py`): samples the route every 2 NM along great circles with a climb / cruise / descent profile, giving position, altitude and time at each point.
3. **Relevance engine** (`relevance.py`): keeps a NOTAM only if the aircraft is inside its volume (plus a 10 NM / 1,000 ft / 30 min buffer) while it is active, or if it concerns the departure, destination or alternate aerodrome around the time the aircraft uses it. Every keep/drop decision carries a plain-language reason.
4. **Hazard classifier** (`classifier.py`): 12 hazard categories with HIGH / MEDIUM / LOW severity. A known Q-code is authoritative; otherwise a TF-IDF (word + character n-gram) logistic regression on the expanded E-text; if that is not confident, transparent keyword rules; if those don't match, the NOTAM is shown as **REVIEW** near the top instead of being silently mis-ranked. Model decisions report the n-grams that drove them.
5. **Briefing** (`briefing.py`): ranks by severity, then flight phase, then time of first contact.

## Example

```
$ python -m notam_copilot --feed examples/feed.txt --dep KPHX --dest KDEN --alt KABQ --etd 2611051500 --show-dropped

NOTAM BRIEFING  KPHX -> KDEN  (ALTN KABQ)
ETD 051500Z  ETA 051609Z  523 NM  FL350
5 relevant of 9 NOTAMs (4 filtered as not affecting this flight)

[HIGH  ] DEPARTURE A0101/26  RWY_CLOSURE       RWY 08/26 CLSD DUE WIP
           why: KPHX is the departure aerodrome; NOTAM active within -1h/+1h of 051500Z; class via qcode
[HIGH  ] ENROUTE   A0103/26  GNSS_INTERFERENCE GPS UNRELIABLE WI 60NM RADIUS FL400 AND BLW DUE TESTING
           why: route passes 69 NM from centre at 35,000 ft, 051515Z, while active; class via qcode
[HIGH  ] ARRIVAL   A0105/26  ILS_OUTAGE        ILS RWY 16R/34L GP U/S
           why: KDEN is the arrival aerodrome; ...; class via model (ILS, ILS RUNWAY, RUNWAY GLIDE)
...
FILTERED:
  A0104/26  FLIGHT RESTRICTIONS ... FIRE FIGHTING   <- route never within 20 NM of NOTAM area
  A0106/26  RWY 07/25 CLSD                          <- KDEN is the arrival aerodrome but NOTAM is not active when the aircraft is there
```

`python -m notam_copilot --demo --seed 7` runs on a generated feed for a random flight.

## Results

All numbers come from `python evaluate.py` (written to `results.json`).

**Hazard classification.** 12 categories × 7 phrasing templates. The last 2 templates of every category are held out, so the test set (960 NOTAMs) uses wording the model never saw in training (2,400 NOTAMs).

| Approach | Accuracy | Macro-F1 |
|---|---|---|
| Keyword rules only | 75.0% | 0.73 |
| TF-IDF + logistic regression only | 66.7% | 0.68 |
| **Hybrid (model → rules → REVIEW)** | **83.3%** | **0.83** |

- 12.5% of test NOTAMs are flagged REVIEW; accuracy on the rest is 95.2%.
- 87.5% of HIGH-severity NOTAMs are rated HIGH or sent to REVIEW. The miss is a runway closure worded as "AD RWY … NOT AVBL FOR LDG OR TKOF", which the model confidently called an aerodrome-service notice. Known gap.
- Note: the rules fallback was added after seeing that the model alone breaks down on unseen vocabulary (e.g. "WATERFOWL", "REFUELING"), so the hybrid figure was not set before looking at the test split.

**Relevance engine.** 200 simulated flights between 24 US airports, 60 NOTAMs each (12,000 total). Ground truth is constructed: true hits sit on the route, in the altitude band, while active; decoys violate exactly one constraint (lateral, vertical, time, wrong aerodrome, or right aerodrome at the wrong time).

- Precision 100%, recall 100% on these constructed cases.
- ~33 ms to build a briefing per flight (pure Python, single core).
- The 80% "filtered" rate is set by the scenario mix (12 hits per 60), not a real-world measurement.

## Limitations

- The corpus is synthetic: generated from phrasing templates modelled on FAA/ICAO conventions, not a live NOTAM feed. Real feeds have messier text and many more categories.
- Area NOTAMs are modelled as circles (from the Q-line); polygons in the E-text are not parsed.
- The flight profile is a straight-line climb/cruise/descent at constant ground speed; no SIDs/STARs, winds or holding.
- Relevance results test the engine's logic against known answers; they do not measure performance on real flights.

Next steps: pull real NOTAMs from the FAA NOTAM API, hand-label a sample to test the classifier on real text, parse E-text polygons, and re-evaluate the relevance engine against FAA flight-plan data.

## Run it

```
pip install scikit-learn numpy
python evaluate.py            # trains, evaluates, writes model.pkl + results.json
python -m pytest tests        # unit tests
python -m notam_copilot --demo
```
