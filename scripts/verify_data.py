"""Check the cached CSVs in data/ against data/CHECKSUMS.sha256 (same check as `sha256sum -c`).

Yahoo Finance returns slightly different adjusted prices from one download to the next, so a
fresh download will usually fail this check; results/ were produced from the cached files listed.

Usage: python scripts/verify_data.py
"""
import hashlib
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / 'data'


def main():
    ok = True
    for line in (DATA / 'CHECKSUMS.sha256').read_text().splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        path = DATA / name.lstrip('*')
        if not path.exists():
            print(f"{path.name}: MISSING")
            ok = False
            continue
        match = hashlib.sha256(path.read_bytes()).hexdigest() == digest
        print(f"{path.name}: {'OK' if match else 'FAILED'}")
        ok &= match
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
