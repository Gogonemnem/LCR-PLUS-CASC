"""Download the SeMaEval 2015/2016 ABSA restaurant XMLs required by semeval_reader.py.

The official hosting (catalogue/metashare.ilsp.gr) requires a login to download,
so we mirror the canonical files from the public GitHub repo
`howardhsu/ABSA_preprocessing`, which contains the exact XMLs `semeval_reader.py`
expects under datasets/restaurant/.

Usage:
    python download_semeval.py            # download if missing, validate all
    python download_semeval.py --force    # re-download unconditionally
"""
import argparse
import os
import ssl
import sys
import urllib.request
import xml.etree.ElementTree as ET

REPO = "howardhsu/ABSA_preprocessing"
BASE = f"https://raw.githubusercontent.com/{REPO}/HEAD/dataset/SemEval"

# filename_in_repo -> path within howardhsu/ABSA_preprocessing
FILES = {
    "ABSA16_Restaurants_Train_SB1_v2.xml": "16/rest/ABSA16_Restaurants_Train_SB1_v2.xml",
    "EN_REST_SB1_TEST.xml.gold": "16/rest/EN_REST_SB1_TEST.xml.gold",
    "ABSA-15_Restaurants_Train_Final.xml": "15/rest/ABSA-15_Restaurants_Train_Final.xml",
    "ABSA15_Restaurants_Test.xml": "15/rest/ABSA15_Restaurants_Test.xml",
}

# Minimum reasonable byte sizes (guards against HTML error pages / 404 bodies)
MIN_SIZE = {
    "ABSA16_Restaurants_Train_SB1_v2.xml": 600_000,
    "EN_REST_SB1_TEST.xml.gold": 200_000,
    "ABSA-15_Restaurants_Train_Final.xml": 400_000,
    "ABSA15_Restaurants_Test.xml": 200_000,
}


def _ctx():
    return ssl.create_default_context()


def download(name: str, dest_dir: str, force: bool) -> bool:
    dest = os.path.join(dest_dir, name)
    if os.path.exists(dest) and os.path.getsize(dest) >= MIN_SIZE[name] and not force:
        print(f"  [skip] {name} exists ({os.path.getsize(dest)} bytes)")
        return True

    url = f"{BASE}/{FILES[name]}"
    print(f"  [get ] {name}  <- {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "lcr-plus-casc-download"})
    with urllib.request.urlopen(req, context=_ctx(), timeout=120) as resp:
        data = resp.read()

    if len(data) < MIN_SIZE[name]:
        print(f"  [FAIL] {name}: got only {len(data)} bytes (expected >{MIN_SIZE[name]})")
        return False

    with open(dest, "wb") as f:
        f.write(data)
    print(f"  [ok  ] {name}: {len(data)} bytes")
    return True


def validate(name: str, dest_dir: str) -> bool:
    path = os.path.join(dest_dir, name)
    if not os.path.exists(path):
        return False
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as e:
        print(f"  [FAIL] {name}: XML parse error: {e}")
        return False
    n_sent = len(list(root.iter("sentence")))
    n_op = sum(1 for _ in root.iter("Opinion"))
    if n_op == 0:
        print(f"  [FAIL] {name}: no Opinion elements found")
        return False
    print(f"  [valid] {name}: root='{root.tag}' sentences={n_sent} opinions={n_op}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    ap.add_argument(
        "--dest",
        default=os.path.join(os.path.dirname(__file__), "datasets", "restaurant"),
        help="destination directory (default: <repo>/datasets/restaurant)",
    )
    args = ap.parse_args()

    dest_dir = os.path.abspath(args.dest)
    if not os.path.isdir(dest_dir):
        print(f"  [FAIL] destination dir not found: {dest_dir}")
        return 1
    os.makedirs(dest_dir, exist_ok=True)

    print(f"Downloading into: {dest_dir}")
    ok = all(download(n, dest_dir, args.force) for n in FILES)

    print("\nValidating XML structure:")
    ok &= all(validate(n, dest_dir) for n in FILES)

    if ok:
        print("\nAll 4 SeMaEval XML files present and valid.")
        return 0
    print("\nSome files are missing or invalid.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
