# Machine Learning

Learning notebooks for PyTorch: tensors, building and training small neural
networks from scratch, and an LSTM that predicts stock prices. The notebooks
carry stored outputs from previous runs.

## Notebooks

### `3-pytorch.ipynb` — PyTorch and Machine Learning Guide (32 cells)

A guided tutorial, written with explanations between code cells:

1. **Data preprocessing** — tensors from lists, loading MNIST with
   `torchvision.datasets`, train/test split with `DataLoader`.
2. **Model, loss, optimizer** — builds up from a one-parameter network to a
   two-layer network with ReLU, then MSE loss and SGD.
3. **Training** — an `MNISTNetwork` (flatten → linear → ReLU → linear,
   28×28 → 128 → 10) trained with cross-entropy loss for 10 epochs,
   tracking train/val loss and accuracy.
4. **Evaluation** — predictions on sample digits, visualized next to the
   true labels, with a note that the simple model does not always get them
   right.

### `script.ipynb` — PyTorch from scratch (9 cells)

The same MNIST ground, without the tutorial text: loads and visualizes the
dataset, an 80/20 train/test split, then three networks built up step by step
(`LinearNetwork`, `NetworkWithReLU`, `MNISTNetwork`) and a 1000-epoch
training loop tracking loss and accuracy. A scratch pad for the ideas in
`3-pytorch.ipynb`.

### `stockscript.ipynb` — LSTM stock-price prediction (13 cells)

Predicts the next-day closing price from 30-day windows of S&P 500 data:

- `StockDataset` reads `data/S&P500/all_stocks_5yr.csv`, normalizes the five
  numeric features (open, high, low, close, volume), and embeds the ticker
  symbol (505 tickers, PAD/OOV ids).
- `LSTMReg` (1 layer, hidden size 64, SmoothL1Loss, Adam, lr 0.001) trains
  for up to 20 epochs with early stopping (patience 5) on the train split.
- The best weights are saved to `model.pt`, reloaded into a fresh model, and
  evaluated on `data/TESTDATA/top_10_stock_data.csv`. The final cell prints
  the first 30 predictions next to their actual closes.

**Fixed 2026-11-09:** the final cell used to error with `ValueError: not
enough values to unpack (expected 2, got 1)`. `display_results` iterated its
argument as `(prediction, expected)` pairs, but `run_epoch` returned only a
predictions tensor. `run_epoch` now collects and returns the targets as well
(the commented-out code in the function shows that was the original intent),
`display_results` takes them as an argument and zips the two, and the two
call sites unpack the third return value. The same change also fixed a
pandas 3 incompatibility in `StockDataset` (`fillna(method=...)` was
removed; it is now `.ffill().bfill()`). Verified by executing cells 0–7
and 10–12 with torch 2.15.0+cpu / pandas 3.1.1: test loss 0.002599
(matching the stored output 0.002599) and 30 prediction/expected pairs
printed. The 20-epoch training cell was not re-run; `model.pt` is the
previously saved weights.

## Data

```
data/
├── MNIST/raw/              MNIST train/test images and labels (idx-ubyte, + .gz)
├── S&P500/all_stocks_5yr.csv   619,040 rows, 505 tickers, 2013-02-08 → 2018-02-07
│                               columns: date, open, high, low, close, volume, ticker
└── TESTDATA/top_10_stock_data.csv   25,486 rows, same OHLCV columns plus adj close
```

`model.pt` (156 KB) is the saved `LSTMReg` state dict from `stockscript.ipynb`.

## Run them

```bash
# once: create an environment with PyTorch (CPU is fine), plus pandas/numpy/matplotlib
pip install torch torchvision pandas numpy matplotlib jupyter

jupyter notebook   # or: jupyter lab
```

Run cells top to bottom. MNIST downloads on first run if `data/MNIST` is
missing (the raw files are already committed here). The notebooks were
written against a Conda setup; `3-pytorch.ipynb` refers to a `1-conda.md`
tutorial that is not in this repo, so install PyTorch from
<https://pytorch.org/get-started/locally/> if you do not have it yet.
