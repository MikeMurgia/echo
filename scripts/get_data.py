#!/usr/bin/env python
"""Download NASA C-MAPSS and set up data/ in one command (standard library only).

    python scripts/get_data.py                  # FD001 + FD002 (default)
    python scripts/get_data.py --subsets FD001 FD002 FD003 FD004
    python scripts/get_data.py --force          # re-download and overwrite
    python scripts/get_data.py --url <zip-url>  # if NASA moves the file

Safe to re-run: files already in data/ are skipped. The download is extracted
into data/ by file name only (zip paths are ignored, so nothing can escape data/).
"""
import argparse
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile

DEFAULT_URL = "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
PAGE = "https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")


def wanted(subsets):
    return {f"{k}_{s}.txt" for s in subsets for k in ("train", "test", "RUL")}


def progress(blocks, bs, total):
    if total > 0:
        pct = min(100, blocks * bs * 100 // total)
        print(f"\r  downloading... {pct}%", end="", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subsets", nargs="+", default=["FD001", "FD002"], choices=["FD001", "FD002", "FD003", "FD004"])
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    os.makedirs(DATA, exist_ok=True)
    need = wanted(a.subsets)
    missing = {f for f in need if a.force or not os.path.exists(os.path.join(DATA, f))}
    if not missing:
        print(f"All {len(need)} files already in {DATA}. Nothing to do (use --force to redo).")
        return 0

    with tempfile.TemporaryDirectory() as tmp:
        zpath = os.path.join(tmp, "cmapss.zip")
        print(f"Downloading C-MAPSS from {a.url}")
        try:
            urllib.request.urlretrieve(a.url, zpath, progress)
            print()
            if not zipfile.is_zipfile(zpath):
                raise ValueError("downloaded file is not a zip")
        except Exception as e:
            print(f"\nDownload failed: {e}\n\nManual fallback:\n  1. Download the 'Turbofan Engine Degradation "
                  f"Simulation' data set from\n     {PAGE}\n  2. Unzip and copy the train_/test_/RUL_ .txt files "
                  f"into {DATA}\n  3. Or rerun with --url <direct zip link>")
            return 1

        found, listing = set(), []

        def scan(zf, prefix=""):
            """Extract wanted files from zf; recurse into zips nested inside it."""
            for info in zf.infolist():
                if info.is_dir():
                    continue
                listing.append(prefix + info.filename)
                name = os.path.basename(info.filename)  # ignore folders inside the zip
                if name in missing and name not in found:
                    with zf.open(info) as src, open(os.path.join(DATA, name + ".part"), "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    os.replace(os.path.join(DATA, name + ".part"), os.path.join(DATA, name))
                    found.add(name)
                    print(f"  extracted {name}")
                elif name.lower().endswith(".zip"):
                    inner = os.path.join(tmp, f"nested_{len(listing)}.zip")
                    with zf.open(info) as src, open(inner, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    if zipfile.is_zipfile(inner):
                        with zipfile.ZipFile(inner) as nz:
                            scan(nz, prefix + info.filename + " > ")

        with zipfile.ZipFile(zpath) as z:
            scan(z)

    left = missing - found
    if left:
        print(f"Warning: not found in the zip: {sorted(left)}")
        print("Files the zip contains (send this list to whoever maintains the script):")
        for n in listing:
            print("  " + n)
        return 1
    print(f"Done. {len(found)} files in {DATA}")
    return 0


if __name__ == "__main__":
    sys.exit(main())