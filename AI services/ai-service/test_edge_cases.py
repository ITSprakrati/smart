import requests
import json

BASE = "http://127.0.0.1:8005"

cases = [
    # (name, expected_status_set, payload_dict_or_raw)
    ("valid minimal request", {200}, {"bin_id": "B1", "fill_level": 72, "timestamp": "2026-07-27T10:15:30Z"}),
    ("valid with all optional fields", {200}, {"bin_id": "B1", "fill_level": 72, "battery": 89, "temperature": 32.4,
        "location": {"lat": 23.26, "lng": 77.41}, "timestamp": "2026-07-27T10:15:30Z", "device_id": "D1",
        "zone_id": "Z_01", "prev_fill_level": 60, "prev_timestamp": "2026-07-27T08:15:30Z",
        "last_collection_minutes_ago": 120}),
    ("timestamp with milliseconds", {200}, {"bin_id": "B1", "fill_level": 72, "timestamp": "2026-07-27T10:15:30.123Z"}),
    ("prev_timestamp with milliseconds", {200}, {"bin_id": "B1", "fill_level": 72, "timestamp": "2026-07-27T10:15:30Z",
        "prev_fill_level": 60, "prev_timestamp": "2026-07-27T08:15:30.999Z"}),
    ("timestamp no Z, no offset (naive)", {200, 422}, {"bin_id": "B1", "fill_level": 72, "timestamp": "2026-07-27T10:15:30"}),
    ("timestamp with +offset instead of Z", {200}, {"bin_id": "B1", "fill_level": 72, "timestamp": "2026-07-27T10:15:30+05:30"}),
    ("timestamp with space separator", {200, 422}, {"bin_id": "B1", "fill_level": 72, "timestamp": "2026-07-27 10:15:30Z"}),
    ("garbage timestamp string", {422}, {"bin_id": "B1", "fill_level": 72, "timestamp": "not-a-date"}),
    ("empty string timestamp", {422}, {"bin_id": "B1", "fill_level": 72, "timestamp": ""}),
    ("null timestamp (violates required str)", {422}, {"bin_id": "B1", "fill_level": 72, "timestamp": None}),
    ("missing timestamp entirely", {422}, {"bin_id": "B1", "fill_level": 72}),
    ("missing fill_level entirely", {422}, {"bin_id": "B1", "timestamp": "2026-07-27T10:15:30Z"}),
    ("missing bin_id entirely", {422}, {"fill_level": 72, "timestamp": "2026-07-27T10:15:30Z"}),
    ("fill_level as numeric string '72'", {200, 422}, {"bin_id": "B1", "fill_level": "72", "timestamp": "2026-07-27T10:15:30Z"}),
    ("fill_level as non-numeric string", {422}, {"bin_id": "B1", "fill_level": "abc", "timestamp": "2026-07-27T10:15:30Z"}),
    ("fill_level null", {422}, {"bin_id": "B1", "fill_level": None, "timestamp": "2026-07-27T10:15:30Z"}),
    ("fill_level negative", {422}, {"bin_id": "B1", "fill_level": -5, "timestamp": "2026-07-27T10:15:30Z"}),
    ("fill_level > 100", {422}, {"bin_id": "B1", "fill_level": 250, "timestamp": "2026-07-27T10:15:30Z"}),
    ("fill_level = 0", {200}, {"bin_id": "B1", "fill_level": 0, "timestamp": "2026-07-27T10:15:30Z"}),
    ("fill_level = 100 exactly", {200}, {"bin_id": "B1", "fill_level": 100, "timestamp": "2026-07-27T10:15:30Z"}),
    ("temperature null", {200}, {"bin_id": "B1", "fill_level": 72, "temperature": None, "timestamp": "2026-07-27T10:15:30Z"}),
    ("battery null", {200}, {"bin_id": "B1", "fill_level": 72, "battery": None, "timestamp": "2026-07-27T10:15:30Z"}),
    ("prev_fill_level set, prev_timestamp missing", {200}, {"bin_id": "B1", "fill_level": 72, "prev_fill_level": 60,
        "timestamp": "2026-07-27T10:15:30Z"}),
    ("prev_timestamp set, prev_fill_level missing", {200}, {"bin_id": "B1", "fill_level": 72,
        "prev_timestamp": "2026-07-27T08:15:30Z", "timestamp": "2026-07-27T10:15:30Z"}),
    ("prev_timestamp AFTER current timestamp (negative hours)", {200}, {"bin_id": "B1", "fill_level": 72,
        "prev_fill_level": 60, "prev_timestamp": "2026-07-27T12:15:30Z", "timestamp": "2026-07-27T10:15:30Z"}),
    ("prev_timestamp equals current timestamp (zero hours)", {200}, {"bin_id": "B1", "fill_level": 72,
        "prev_fill_level": 60, "prev_timestamp": "2026-07-27T10:15:30Z", "timestamp": "2026-07-27T10:15:30Z"}),
    ("prev_timestamp garbage string", {422}, {"bin_id": "B1", "fill_level": 72, "prev_fill_level": 60,
        "prev_timestamp": "garbage", "timestamp": "2026-07-27T10:15:30Z"}),
    ("location missing lng (incomplete submodel)", {422}, {"bin_id": "B1", "fill_level": 72,
        "location": {"lat": 23.26}, "timestamp": "2026-07-27T10:15:30Z"}),
    ("location null", {200}, {"bin_id": "B1", "fill_level": 72, "location": None, "timestamp": "2026-07-27T10:15:30Z"}),
    ("zone_id null", {200}, {"bin_id": "B1", "fill_level": 72, "zone_id": None, "timestamp": "2026-07-27T10:15:30Z"}),
    ("last_collection_minutes_ago negative", {200}, {"bin_id": "B1", "fill_level": 72,
        "last_collection_minutes_ago": -50, "timestamp": "2026-07-27T10:15:30Z"}),
    ("extra unexpected field", {200}, {"bin_id": "B1", "fill_level": 72, "timestamp": "2026-07-27T10:15:30Z",
        "some_random_field": "hello"}),
    ("bin_id empty string", {200}, {"bin_id": "", "fill_level": 72, "timestamp": "2026-07-27T10:15:30Z"}),
    ("bin_id as integer (wrong type)", {200, 422}, {"bin_id": 123, "fill_level": 72, "timestamp": "2026-07-27T10:15:30Z"}),
    ("completely empty body", {422}, {}),
]

RAW_CASES = [
    ("malformed JSON body", {422}, '{"bin_id": "B1", "fill_level": 72,'),  # truncated JSON
    ("array instead of object", {422}, '[1,2,3]'),
    ("plain string body", {422}, '"just a string"'),
]

def run(endpoint):
    print(f"\n{'='*70}\nTESTING {endpoint}\n{'='*70}")
    results = []
    for name, expected, payload in cases:
        try:
            r = requests.post(f"{BASE}{endpoint}", json=payload, timeout=10)
            status = r.status_code
            ok = status in expected
            crashed = status >= 500
            marker = "PASS" if ok else ("!!CRASH!!" if crashed else "FAIL")
            results.append((marker, name, status, expected))
            if not ok:
                print(f"[{marker}] {name}: got {status}, expected one of {expected}")
                print(f"       body: {r.text[:200]}")
        except Exception as e:
            results.append(("!!EXCEPTION!!", name, "N/A", expected))
            print(f"[!!EXCEPTION!!] {name}: {e}")

    for name, expected, raw in RAW_CASES:
        try:
            r = requests.post(f"{BASE}{endpoint}", data=raw, headers={"Content-Type": "application/json"}, timeout=10)
            status = r.status_code
            ok = status in expected
            crashed = status >= 500
            marker = "PASS" if ok else ("!!CRASH!!" if crashed else "FAIL")
            results.append((marker, name, status, expected))
            if not ok:
                print(f"[{marker}] {name}: got {status}, expected one of {expected}")
                print(f"       body: {r.text[:200]}")
        except Exception as e:
            results.append(("!!EXCEPTION!!", name, "N/A", expected))
            print(f"[!!EXCEPTION!!] {name}: {e}")

    passed = sum(1 for r in results if r[0] == "PASS")
    total = len(results)
    crashes = [r for r in results if r[0] in ("!!CRASH!!", "!!EXCEPTION!!")]
    print(f"\n{passed}/{total} passed. CRASHES (500s or exceptions): {len(crashes)}")
    if crashes:
        for c in crashes:
            print(f"  -> {c}")
    return crashes

c1 = run("/predict/time-to-full")
c2 = run("/predict/priority-score")

print(f"\n\n{'#'*70}")
print(f"TOTAL CRASHES ACROSS BOTH ENDPOINTS: {len(c1) + len(c2)}")
print(f"{'#'*70}")
