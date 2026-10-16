"""Check the uploaded binary-classification dataset before training."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def check_data(frame: pd.DataFrame) -> dict:
    if "id" not in frame or "label" not in frame:
        raise ValueError("Dataset must contain an id and a label column")
    predictors = [c for c in frame.columns if c not in ("id", "label")]
    if len(predictors) != 512:
        raise ValueError(f"Expected 512 numerical feature columns, found {len(predictors)}")
    if frame.empty or frame["id"].isna().any() or frame["id"].duplicated().any():
        raise ValueError("Record IDs must exist and be unique")
    if frame["label"].isna().any() or set(frame["label"].unique()) != {0, 1}:
        raise ValueError("Labels must contain both classes 0 and 1, with no nulls")
    numeric = frame.loc[:, predictors].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("All 512 features must be finite numerical values")
    return {
        "rows": int(len(frame)),
        "features": 512,
        "class_counts": {str(k): int(v) for k, v in frame["label"].value_counts().sort_index().items()},
    }


def main():
    parser = argparse.ArgumentParser(description="Validate binary classifier CSV structure")
    parser.add_argument("--data", type=Path, default=Path("data.csv"))
    args = parser.parse_args()
    print(json.dumps(check_data(pd.read_csv(args.data)), indent=2))


if __name__ == "__main__":
    main()
