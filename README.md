# HFsyncNN: Nuclear-Norm Synchronization for High-Frequency Data (MATLAB)

This MATLAB toolbox implements the nuclear-norm-based synchronization
framework developed in the paper:

> "Data Synchronization at High Frequencies" : 
   Authors: Xinbing Kong, Cheng Liu, Bin Wu

The method recasts the data-synchronization problem for asynchronous
high-frequency log-prices as a constrained matrix-completion problem.
We recover the potential increment matrix by minimizing its nuclear
norm under a large system of linear constraints derived from observed
asynchronous price changes, and we solve the resulting problem using a
scaled ADMM algorithm.

---

## 1. Features

- **Asynchronous high-frequency panel data**  
  Handles log-price matrices with missing entries (encoded as zeros),
  where different assets are observed at irregular times.

- **Nuclear-norm minimization under linear constraints**  
  Recovers a low-rank potential increment matrix `Δ` subject to
  `A * vec(Δ') = b`, where `A` encodes durations and `b` stacks the
  observed increments over durations.

- **Scalable ADMM solver**  
  Uses a scaled ADMM algorithm with closed-form updates based on
  a Woodbury-type identity, suitable for large T and N.

- **PCA-based or user-defined initialization**  
  By default, uses PCA to construct a low-rank initialization of the
  signal component. Alternatively, you can provide your own initial
  low-rank signal.

- **Optional reconstruction of synchronized log-prices**  
  Returns either synchronized log-returns only, or also reconstructs
  a synchronized log-price matrix using `fillLogPriceFromReturn`.

---

## 2. Installation

1. Clone or download this repository:

   ```bash
   git clone https://github.com/BinWu58/HFsyncNN
