"""Record real responses from a running demo backend for the demo page's offline mode.

    python3 -m cartograph.record_demo --url http://localhost:8012 --out docs/site/recorded.json
Each example is exactly what /v1/bundle returned (only the long "attempts[].text" is kept as-is too).
"""

import argparse
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

FRESH = "2026-10-01T04:00:00+00:00"   # 2 h after the feature watermark
STALE = "2026-10-01T10:00:00+00:00"   # 8 h after: past the 6 h freshness limit
EXAMPLES = [
    ("Quiet keyboard + mouse under $100", "Find a quiet wireless keyboard and mouse bundle under $100. Only include available products and explain the tradeoffs.", FRESH),
    ("Webcam + microphone under $150", "Find a webcam and microphone bundle under $150.", FRESH),
    ("Gaming headphones + microphone under $60", "Find a gaming headphones and microphone bundle under $60.", FRESH),
    ("Keyboard-mouse combo + mouse pad under $50", "Find a wireless keyboard mouse combo and mouse pad bundle under $50.", FRESH),
    ("Monitor + laptop stand under $150", "Find a monitor and laptop stand bundle under $150.", FRESH),
    ("Impossible budget ($5)", "Find a wireless keyboard and mouse bundle under $5.", FRESH),
    ("Stale features (8 h old)", "Find a quiet wireless keyboard and mouse bundle under $100.", STALE),
    ("Vague request (no budget)", "I want something nice for my desk.", FRESH),
]


def call(url, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url + path, data, {"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8012")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--extra", type=Path, nargs="*", default=[], help="saved probe files to include (status, trace)")
    args = parser.parse_args()
    _, info = call(args.url, "/v1/info")
    examples = []
    for label, text, now in EXAMPLES:
        status, trace = call(args.url, "/v1/bundle", {"request": text, "now": now})
        examples.append({"label": label, "status": status, "trace": trace})
        print(status, label, trace.get("decision"))
    for path in args.extra:
        saved = json.loads(path.read_text())
        examples.append({"label": "Redis down (feature store outage)", "status": saved["status"], "trace": saved["trace"],
                         "source": str(path)})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"recorded_at_utc": datetime.now(timezone.utc).isoformat(), "backend": info,
                                    "examples": examples}, indent=1) + "\n")


if __name__ == "__main__":
    main()
