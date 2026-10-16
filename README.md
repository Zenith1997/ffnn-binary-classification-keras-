# Binary Classification with a Feed-Forward Neural Network

A small **Keras/TensorFlow binary classifier** using 512 numerical inputs per example. This is a tabular-data neural network project, not a computer-vision model.

## What I built

The notebook trains a three-layer feed-forward network using **4,500 labelled records** (512 numerical inputs plus a binary target). The input data is split **85% training / 15% held-out test** using stratification and `random_state=17`.

| Network / setting | Value in executed notebook |
|---|---|
| Hidden layers | Dense(12, ReLU) → Dense(8, ReLU) |
| Output | Dense(1, sigmoid) |
| Optimiser | Adam |
| Loss | Binary cross-entropy |
| Epochs | 200 |
| Batch size | 64 |

### Saved test results

These figures are from the notebook's **stored execution output**. They were not independently reproduced when this README was updated.

| Metric | Recorded result |
|---|---:|
| Precision | **0.9930** |
| Recall | **1.0000** |
| F1-score | **0.9965** |
| Accuracy | **0.9956** |

A high held-out score does not automatically prove that the model will generalise to a different dataset. The notebook does not show an independent validation split or repeated runs. It also does not adequately document the origin or meaning of the 512 input features; that needs addressing before real-world use.

## Files

- [`binary_classification_clean.ipynb`](binary_classification_clean.ipynb): cleaned presentation copy with the original training/evaluation cells and saved results
- [`1.3C.ipynb`](1.3C.ipynb): original notebook, preserved unchanged
- [`data.csv`](data.csv): dataset as currently published; **provenance and permission to redistribute still need verification**
- [`check_dataset.py`](check_dataset.py): schema and basic quality checks
- [`tests/`](tests/): tests for input-validation code

## Run

Python 3.10–3.12 is recommended for an easier TensorFlow installation.

```bash
python -m venv .venv
# Activate the environment, then:
python -m pip install -r requirements.txt
python check_dataset.py --data data.csv
jupyter notebook binary_classification_clean.ipynb
```

To run only the data-schema tests (without installing TensorFlow), use:

```bash
python -m pip install -r requirements-test.txt
python -m pytest -q
```

## Limitations

The notebook trains on the available features and tests once on the held-out partition. It does **not** establish performance on newly collected or externally validated data. The 200-epoch experiment may overfit and does not include early stopping. Verify the public dataset's provenance and redistribution rights before relying on it as portfolio evidence.

**Author:** Zenith Anthony Dabarera · [GitHub](https://github.com/Zenith1997)
