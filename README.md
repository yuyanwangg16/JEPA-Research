# JEPA Research: Self-Supervised Learning for Jet Tagging

Self-supervised learning (SSL) on particle physics jet data. This repository
implements JEPA-style pretraining on quark/gluon (q/g) jets and systematically
ablates **LeJEPA** (Balestriero & LeCun, 2025) to understand why its SIGReg
objective underperforms on jet representations — and which of its components
can be salvaged.

🚧 **Work in progress** — ablations are ongoing; notebooks and findings are
updated as experiments complete.

## Overview

- **Data**: particle clouds of shape `(B, 30, 4)` = [η, φ, pT, valid].
  q/g jets are used for SSL pretraining; W, Z, and top-quark jets are
  test-only anomaly classes.
- **Model**: 4-layer Transformer encoder (d=128, 8 heads), trained with
  particle-level pT-ratio target masking (jBOT-style).
- **Evaluation** (two-metric framework, frozen encoder):
  1. q vs g linear probe accuracy
  2. kNN anomaly detection AUC for W/Z/top vs QCD background
- **Framework**: TensorFlow / Keras.

## Notebooks

### `jjepa_reproduction.ipynb` — J-JEPA Reproduction & Representation Diagnostics

JEPA pretraining on q/g jets with extensive representation-quality diagnostics.

- SSL pretraining on q/g only (330k train / 10k val); test set drawn from the
  full 5-class pool (100k jets)
- jBOT-style pT-ratio target masking (masked pT fraction 40–60%)
- EMA teacher (m=0.996), cross-attention predictor, MSE loss on
  masked-particle representations; 100 epochs, AdamW + cosine LR
- Diagnostics: 5-class multinomial linear probe with three sanity baselines
  (random encoder, shuffled labels, zero input), confusion matrix, per-class
  predicted-probability distributions, corner plots, t-SNE, and a pooling
  comparison (masked mean vs. pT-weighted mean vs. raw-feature baseline)

**Note:** this notebook predates the strict 80/20 pretrain/held-out split
used in the notebooks below, so test q/g jets may overlap with pretraining
data. Use `train_qg.ipynb` for leakage-controlled probe numbers.

### `train_qg.ipynb` — JEPA Baseline (particle-level masking, MSE)

End-to-end JEPA pretraining and evaluation with a leakage-controlled split.

- q/g jets capped at 340k, split 80/20 into pretrain / held-out **before**
  any training (fixed seeds, identical across notebooks for fair comparison);
  includes a hash-fingerprint leakage check
- Same architecture and masking as above; loss is MSE between predicted and
  EMA-teacher representations on masked particles
- Evaluation on the frozen teacher (masked mean-pooling): q vs g linear probe
  vs. a random-encoder baseline; kNN anomaly detection sweep (bank size M,
  euclidean/cosine, k) with ROC curves; covariance spectrum and effective rank

**Key result:** healthy, non-collapsed representations — q vs g accuracy
~0.775, 5-class kNN AUC ~0.85. This is the working baseline against which
the LeJEPA ablations are compared.

### `trainqg_jepa_ce.ipynb` — JEPA with Cross-Entropy Objective (DINO-style)

Same architecture and data pipeline as the MSE baseline, but the prediction
loss is cross-entropy through a shared prototype head (D → K=2048), with
DINO-style teacher centering (momentum 0.9) and teacher temperature 0.07.
Serves as the CE-objective counterpart for comparison against both the MSE
baseline and the LeJEPA variants.

### `trainqg_lejepa_ce_symmetric.ipynb` — LeJEPA / SIGReg Ablation 🚧

LeJEPA counterpart to the JEPA notebooks: teacher encoder, EMA, and
stop-gradient are removed (single shared encoder), and collapse prevention is
delegated to **SIGReg** — an Epps–Pulley characteristic-function test over
random 1D projections (512 slices, 17 quadrature points, T ∈ [−5, 5],
paper-recommended settings).

- Encoder with **CLS token** + learnable mask token; the CLS embedding is
  used for both SIGReg and downstream evaluation
- Objective: symmetric DINO-style CE (stop-gradient only on the target
  softmax, not the encoder) + λ · SIGReg; single hyperparameter λ
- Includes SIGReg sanity tests (λ=1 → loss converges to ~0.0005, matching a
  true N(0,1); per-dim std → 1.0 across all 128 dimensions) and collapse
  diagnostics

**Status: work in progress.** Remaining ablations (λ=0, mean-pool SIGReg)
are ongoing.

## Key Findings (so far)

1. **The JEPA baseline works on jet data.** EMA teacher + predictor +
   stop-gradient produces healthy representations (q vs g ~0.775,
   5-class kNN AUC ~0.85, stable entropy, no dead dimensions).
2. **SIGReg is an anti-collapse regularizer, not a source of task signal.**
   At λ=1 (pure SIGReg) the loss converges to the true-Gaussian value and
   every dimension reaches unit variance — yet the q vs g probe drops to
   0.582, below the random-encoder baseline (0.756). The implementation is
   correct; the objective simply carries no discriminative information.
3. **LeJEPA's failure on jets decomposes into two independent causes:**
   (a) SIGReg's isotropic Gaussian target is geometrically mismatched to jet
   representations, which are intrinsically anisotropic along physically
   meaningful axes (mass, pT, multiplicity); and
   (b) without EMA, masked views admit trivial CLS→constant collapse
   solutions that satisfy the prediction loss exactly — SIGReg alone at
   paper-recommended λ cannot prevent them.
4. **EMA does double duty** in JEPA: it is both an anti-collapse mechanism
   and a stable, dynamic learning target. SIGReg can replace only the first
   role.
5. **Evaluate SIGReg in inference mode.** Dropout noise can mask collapse by
   inflating apparent distributional spread (`training=False` is required
   for diagnostics).


## Dataset

The `datasets/` directory contains five HDF5 files — `q.hdf5`, `g.hdf5`
(QCD background), and `w.hdf5`, `z.hdf5`, `t.hdf5` (signal) — each with a
`particle_features` array of shape `(N, 30, 4)` = [η, φ, pT, valid].


## Setup

```bash
pip install tensorflow h5py numpy scikit-learn matplotlib seaborn corner
```

Notebooks assume the data lives in `datasets/` relative to the repo root.
Checkpoint save paths inside the notebooks may need to be adjusted to your
environment.

## References

- **LeJEPA** — Balestriero & LeCun (2025), *LeJEPA: Provable and Scalable
  Self-Supervised Learning Without the Heuristics*, arXiv:2511.08544
- **J-JEPA** — Katel et al. (2024), *Learning Symmetry-Independent Jet
  Representations via Jet-Based Joint Embedding Predictive Architecture*,
  arXiv:2412.05333
- **jBOT** — Tsoi & Rankin (2026), *jBOT: Semantic Jet Representation Clustering Emerges from
  Self-Distillation*, arXiv:2601.11719
- **I-JEPA** — Assran et al. (2023), *Self-Supervised Learning from Images
  with a Joint-Embedding Predictive Architecture*, arXiv:2301.08243
- **DINO** — Caron et al. (2021), *Emerging Properties in Self-Supervised
  Vision Transformers*, arXiv:2104.14294