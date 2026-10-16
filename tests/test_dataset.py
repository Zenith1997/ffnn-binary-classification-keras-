import pandas as pd
import pytest

from check_dataset import check_data


def fixture():
    # Synthetic schema-only test records; not the real dataset.
    rows = [{"id": i, "label": i % 2, **{f"f{j}": float(j + i) for j in range(512)}} for i in range(4)]
    return pd.DataFrame(rows)


def test_expected_numeric_schema():
    assert check_data(fixture())["features"] == 512
    assert check_data(fixture())["class_counts"] == {"0": 2, "1": 2}


def test_duplicate_ids_rejected():
    frame = fixture()
    frame.loc[1, "id"] = frame.loc[0, "id"]
    with pytest.raises(ValueError, match="unique"):
        check_data(frame)


def test_bad_feature_rejected():
    frame = fixture()
    frame["f1"] = frame["f1"].astype(object)
    frame.loc[0, "f1"] = "invalid"
    with pytest.raises(ValueError, match="numerical"):
        check_data(frame)


def test_missing_feature_rejected():
    with pytest.raises(ValueError, match="512"):
        check_data(fixture().drop(columns="f0"))
