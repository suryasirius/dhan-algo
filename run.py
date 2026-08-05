
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent

search_locations = [
    ROOT,
    ROOT / "historical_data",
]

csv_files = []

for folder in search_locations:
    if folder.exists():
        csv_files.extend(folder.glob("*.csv"))

# Prefer historical data files
csv_files = sorted(
    csv_files,
    key=lambda x: (
        "historical_data" not in x.name.lower(),
        -x.stat().st_size
    )
)

if not csv_files:
    print("No CSV files found.")
    raise SystemExit

print("\nAvailable CSV files:\n")

for i, f in enumerate(csv_files, 1):
    print(f"{i}. {f.name}")

selected = csv_files[0]

print(f"\nLoading: {selected.name}")

df = pd.read_csv(selected)

print("\nRows:", len(df))
print("Columns:", len(df.columns))

print("\nColumn Names:")

for c in df.columns:
    print(c)