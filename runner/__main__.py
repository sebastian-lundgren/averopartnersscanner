"""python -m runner --locations data/locations.json --postcode 0101"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from runner import config
from runner.main_loop import run_scan
from runner.result_store import ScanApi

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Street View YOLO scan-runner")
    ap.add_argument("--locations", type=Path, required=True, help="JSON med adresser")
    ap.add_argument("--postcode", type=str, required=True)
    ap.add_argument("--max-addresses", type=int, default=10)
    ap.add_argument("--max-attempts", type=int, default=config.MAX_ATTEMPTS)
    ap.add_argument("--headless", action="store_true", help="Overstyr til headless")
    args = ap.parse_args()
    if args.headless:
        import os

        os.environ["PLAYWRIGHT_HEADLESS"] = "true"

    run_scan(
        locations_file=args.locations,
        postcode=args.postcode,
        max_addresses=args.max_addresses,
        max_attempts=args.max_attempts,
        api=ScanApi(),
    )


if __name__ == "__main__":
    main()
    sys.exit(0)
