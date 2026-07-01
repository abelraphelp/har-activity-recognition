"""Download and extract the UCI HAR Dataset into ./data.

The raw dataset is not committed to the repository (it is large and belongs on
UCI, not in version control). Run this script once to fetch it locally:

    python src/download_data.py

The UCI download is a zip-inside-a-zip, so this script handles both layers and
cleans up the intermediate archives, leaving a single "UCI HAR Dataset" folder
under ./data.
"""

from pathlib import Path
import urllib.request
import zipfile
import shutil

URL = (
    "https://archive.ics.uci.edu/static/public/240/"
    "human+activity+recognition+using+smartphones.zip"
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FINAL_DIR = DATA_DIR / "UCI HAR Dataset"


def main() -> None:
    if FINAL_DIR.exists():
        print(f"Dataset already present at: {FINAL_DIR}")
        return

    DATA_DIR.mkdir(exist_ok=True)
    outer_zip = DATA_DIR / "uci_har.zip"

    print("Downloading UCI HAR Dataset (~60 MB)...")
    urllib.request.urlretrieve(URL, outer_zip)

    # Layer 1: the outer zip contains an inner "UCI HAR Dataset.zip".
    print("Extracting outer archive...")
    with zipfile.ZipFile(outer_zip) as z:
        z.extractall(DATA_DIR)

    # Layer 2: extract the inner zip that holds the actual data.
    inner_zip = DATA_DIR / "UCI HAR Dataset.zip"
    print("Extracting inner archive...")
    with zipfile.ZipFile(inner_zip) as z:
        z.extractall(DATA_DIR)

    # Clean up intermediate files and Mac/OS junk.
    for junk in [outer_zip, inner_zip, DATA_DIR / "UCI HAR Dataset.names"]:
        junk.unlink(missing_ok=True)
    shutil.rmtree(DATA_DIR / "__MACOSX", ignore_errors=True)
    (FINAL_DIR / ".DS_Store").unlink(missing_ok=True)

    print(f"Done. Dataset ready at: {FINAL_DIR}")


if __name__ == "__main__":
    main()
