from pathlib import Path
import pandas as pd
from ucimlrepo import fetch_ucirepo

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = DATA_DIR / "student-mat.csv"

def main():
    print("Downloading the official UCI Student Performance dataset...")
    dataset = fetch_ucirepo(id=320)
    features = dataset.data.features.copy()
    targets = dataset.data.targets.copy()
    data = pd.concat([features, targets], axis=1)

    # The UCI repository provides both Mathematics and Portuguese records.
    # For this project we use the Mathematics course (395 rows).
    # Fetching via ucimlrepo returns the repository's combined feature table;
    # the official raw Mathematics file is downloaded below for exact course selection.
    import urllib.request, zipfile, io
    url = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=30) as response:
            archive_bytes = response.read()
        outer = zipfile.ZipFile(io.BytesIO(archive_bytes))
        names = outer.namelist()
        if "student-mat.csv" in names:
            with outer.open("student-mat.csv") as f:
                math_df = pd.read_csv(f, sep=";")
        elif "student.zip" in names:
            inner = zipfile.ZipFile(io.BytesIO(outer.read("student.zip")))
            with inner.open("student-mat.csv") as f:
                math_df = pd.read_csv(f, sep=";")
        else:
            raise FileNotFoundError("student-mat.csv was not found in the UCI archive.")
    except Exception as error:
        print("Direct UCI file download was unavailable:", error)
        print("Using the Mathematics records from the UCI package if available.")
        # ucimlrepo normally exposes the combined dataset, so ask user to retry
        # rather than silently training on a different course.
        raise RuntimeError(
            "Could not download the Mathematics CSV. Check internet access and run "
            "'python download_dataset.py' again."
        ) from error

    math_df.to_csv(OUTPUT_FILE, index=False, sep=";")
    print(f"Saved {len(math_df)} Mathematics student records to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
