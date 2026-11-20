"""Politely downloads the Acts listed with a verified URL in seed/acts_catalog.json.

    python -m scripts.acts_fetch [--out backend/data_legal/raw]

Rules it follows: reads each host's robots.txt first and skips disallowed URLs; identifies itself with a
descriptive User-Agent; waits 2 seconds between requests; makes one attempt per file; never uses anti-bot
tooling; checks that the response really is a PDF. Acts without a verified URL are only listed: download
the official PDF yourself and save it as <out>/<slug>.pdf (the indexer picks it up).
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.robotparser
from pathlib import Path
from urllib.parse import urlsplit

import requests

BACKEND_DIR = Path(__file__).resolve().parents[1]
CATALOG = BACKEND_DIR / "seed" / "acts_catalog.json"
USER_AGENT = "NyayaCheck-research-prototype/0.1 (academic project; polite, low-volume)"
DELAY_S = 2.0


def allowed_by_robots(url: str) -> bool:
    parts = urlsplit(url)
    parser = urllib.robotparser.RobotFileParser()
    try:
        response = requests.get(f"{parts.scheme}://{parts.netloc}/robots.txt", headers={"User-Agent": USER_AGENT}, timeout=30)
    except requests.RequestException:
        return True  # robots.txt unreachable: no rules to apply (the file request itself still may fail)
    if response.status_code >= 400:
        return True
    parser.parse(response.text.splitlines())
    return parser.can_fetch(USER_AGENT, url)


def fetch(out: Path) -> int:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))["acts"]
    out.mkdir(parents=True, exist_ok=True)
    fetched, manual, failed = [], [], []
    for act in catalog:
        target = out / f"{act['slug']}.pdf"
        if target.exists():
            print(f"have     {act['name']}")
            fetched.append(act["name"])
            continue
        if not act.get("url"):
            manual.append(act)
            continue
        if not allowed_by_robots(act["url"]):
            print(f"robots   {act['name']}: disallowed, skipped")
            failed.append((act["name"], "disallowed by robots.txt"))
            continue
        try:
            response = requests.get(act["url"], headers={"User-Agent": USER_AGENT}, timeout=120)
        except requests.RequestException as exc:
            failed.append((act["name"], type(exc).__name__))
            print(f"failed   {act['name']}: {type(exc).__name__}")
            time.sleep(DELAY_S)
            continue
        if response.status_code == 200 and response.content[:4] == b"%PDF":
            target.write_bytes(response.content)
            print(f"fetched  {act['name']} ({len(response.content) // 1024} KB)")
            fetched.append(act["name"])
        else:
            failed.append((act["name"], f"HTTP {response.status_code} / not a PDF"))
            print(f"failed   {act['name']}: HTTP {response.status_code}")
        time.sleep(DELAY_S)

    print(f"\nObtained {len(fetched)} of {len(catalog)} Acts. Failed: {len(failed)}. Need manual download: {len(manual)}")
    if manual:
        print(f"Save official PDFs under {out} with these file names:")
        for act in manual:
            print(f"  {act['slug']}.pdf   ({act['name']})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=BACKEND_DIR / "data_legal" / "raw")
    return fetch(parser.parse_args().out)


if __name__ == "__main__":
    raise SystemExit(main())
