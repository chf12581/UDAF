"""Mean and sample std across independent SD subjects or LOSO folds."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("metrics", nargs="+")
    args = p.parse_args()
    paths = [Path(x).resolve() for x in args.metrics]
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate result file")
    if len(paths) < 2:
        raise ValueError("At least two folds required for sample standard deviation")
    records = [json.loads(x.read_text()) for x in paths]
    for key in ("accuracy", "macro_f1"):
        values = np.array([r[key] for r in records]) * 100
        print(f"{key}: {values.mean():.2f} +/- {values.std(ddof=1):.2f}% (n={len(values)})")


if __name__ == "__main__":
    main()
