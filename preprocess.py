import pandas as pd
from pathlib import Path

root = Path("data")   # change if needed

for file in sorted(root.rglob("*")):
    if file.suffix.lower() in [".csv", ".xlsx", ".xls"]:
        print("\n" + "="*100)
        print(file)
        try:
            if file.suffix.lower() == ".csv":
                df = pd.read_csv(file, nrows=5)
            else:
                df = pd.read_excel(file, nrows=5)
            print("Columns:")
            print(list(df.columns))
            print("Shape preview:", df.shape)
            print(df.head(2).to_dict(orient="records"))
        except Exception as e:
            print("ERROR:", e)