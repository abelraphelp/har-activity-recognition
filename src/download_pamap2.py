"""Download and extract the PAMAP2 Physical Activity Monitoring dataset into ./data.

Like the UCI HAR download, the PAMAP2 archive is a zip-inside-a-zip. Run once:

    python src/download_pamap2.py

Leaves a single "PAMAP2_Dataset" folder under ./data (Protocol/ + Optional/ .dat files).
The dataset is large (~1.5 GB extracted) and is not committed to the repository.
"""

from pathlib import Path
import urllib.request
import zipfile
import shutil

URL = (
    "https://archive.ics.uci.edu/static/public/231/"
    "pamap2+physical+activity+monitoring.zip"
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FINAL_DIR = DATA_DIR / "PAMAP2_Dataset"


def main() -> None:
    if FINAL_DIR.exists():
        print(f"Dataset already present at: {FINAL_DIR}")
        return

    DATA_DIR.mkdir(exist_ok=True)
    outer_zip = DATA_DIR / "pamap2.zip"

    print("Downloading PAMAP2 (~250 MB)...")
    urllib.request.urlretrieve(URL, outer_zip)

    print("Extracting outer archive...")
    with zipfile.ZipFile(outer_zip) as z:
        z.extractall(DATA_DIR)

    inner_zip = DATA_DIR / "PAMAP2_Dataset.zip"
    print("Extracting inner archive (large, please wait)...")
    with zipfile.ZipFile(inner_zip) as z:
        z.extractall(DATA_DIR)

    for junk in [outer_zip, inner_zip, DATA_DIR / "readme.pdf"]:
        junk.unlink(missing_ok=True)

    print(f"Done. Dataset ready at: {FINAL_DIR}")


if __name__ == "__main__":
    main()
