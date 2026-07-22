# scirs2-transform TODO

## Status: v0.6.2 (current, 2026-07-22) — reassessed Stable → Partial

- [x] **Removed unused direct `rand` dependency** — `scirs2-transform`'s `Cargo.toml` carried a direct `rand` crate dependency (previously banned; all random-number generation in this workspace goes through `scirs2_core::random`) with zero actual `rand::`/`use rand` call sites anywhere in `src/`. Removed as pure cleanup; no behavior change. (All apparent `rand`-prefixed hits in a naive grep, e.g. in `performance.rs`, are `scirs2_core::random::{Rng, RngExt}` usage and identifiers like `random_matrix`/`use_randomized` — substring matches on "rand" inside "random", not the banned crate.)
- 0 `todo!()`/`unimplemented!()` markers in `src/`. GPU-accelerated dimensionality reduction (`gpu.rs`: matmul, SVD, eigendecomposition, t-SNE) still returns explicit "not yet implemented, use CPU" errors — this is an **honest** gap (clear `Err`, not a silent no-op) and is already tracked below under "GPU-Accelerated Dimensionality Reduction" (unchecked roadmap items); it does not by itself justify a status downgrade.
- **Status downgraded from Stable to Partial**: a targeted sweep for the *silent*-stub pattern (code that compiles, looks real, and returns a plausible-looking value without actually computing it — the same pattern `scirs2-ndimage`/`scirs2-integrate` are marked "Partial" for) found a confirmed, crate-root-reachable instance:
  - **`distributed::check_node_health`** (`src/distributed.rs`, module re-exported at the crate root via `pub use distributed::{...}` in `lib.rs`) — the function's own comment reads "Simulate health check - in real implementation, this would make HTTP requests"; it `sleep`s 10ms and then returns `NodeHealth { status: NodeStatus::Healthy, cpu_utilization: rng.random_range(0.1..0.9), memory_utilization: rng.random_range(0.2..0.8), network_latency_ms: rng.random_range(1.0..50.0), error_rate: rng.random_range(0.0..0.05), ... }` — every metric is a random number, not a measurement, and `status` is hardcoded `Healthy` regardless of the random values generated. Any caller relying on this for real cluster health monitoring (via the `distributed` feature) is silently getting fabricated data. Not implemented as part of this documentation pass (out of scope for a README/TODO accuracy sweep); recorded here for the next implementation pass.
  - `quantum_optimization.rs` also uses "simulate"-labeled language (`static_evaluate_pipeline_performance` inside a black-box hyperparameter-search objective), but on inspection it appears to call through to real evaluation logic rather than returning a hardcoded number — not counted as a confirmed silent stub, but worth a closer look in a future pass.
  - This sweep was targeted, not exhaustive — `src/` has 554+ public items across ~70 files; `neuromorphic_adaptation.rs`'s "Simulate network dynamics" was not individually triaged in depth. A dedicated follow-up stub-check pass would be worthwhile.

Fresh test counts (2026-07-15, `cargo nextest run -p scirs2-transform` / `--all-features`): **493
passed, 3 skipped** (default features) / **532 passed, 3 skipped** (all-features) — the all-features
run adds 39 tests gated behind `gpu`/`distributed`/`monitoring`/`auto-feature-engineering`.

## Status: v0.4.3 Released (May 3, 2026)

All v0.4.3 features are complete and production-ready. Quality gate (cargo check + clippy) clean.

## Status: v0.3.4 Released (March 18, 2026)

## v0.3.3 Completed

### Normalization and Scaling
- Min-Max, Z-score, Robust (IQR), Max-absolute, L1/L2, Quantile
- Reusable `Normalizer` with fit/transform/inverse_transform
- Custom range normalization
- SIMD-accelerated normalize path (`normalize_simd.rs`)

### Feature Engineering
- Polynomial features with degree and interaction-only options
- Box-Cox and Yeo-Johnson power transformations with optimal lambda
- Equal-width and equal-frequency discretization (binning)
- Binarization with configurable thresholds
- Log transformations with epsilon
- SIMD-accelerated feature ops (`features_simd.rs`)

### Dimensionality Reduction
- PCA with centering/scaling, explained variance ratio, inverse transform
- Truncated SVD for sparse and large matrices
- Linear Discriminant Analysis (LDA) with SVD solver
- Barnes-Hut t-SNE (O(n log n), multicore, trustworthiness metric)
- UMAP (uniform manifold approximation and projection) (`umap.rs`)
- Isomap (geodesic distance manifold learning)
- Locally Linear Embedding (LLE)
- Kernel PCA (RBF, polynomial, sigmoid)
- Probabilistic PCA (`reduction/ppca.rs`)
- Factor analysis (`reduction/fastica.rs`)

### Independent Component Analysis
- FastICA (fixed-point iteration, `reduction/fastica.rs`)
- Spatial ICA (`spatial_ica.rs`)

### NMF Variants
- Standard NMF with multiplicative update rules
- Sparse NMF, Convex NMF, Semi-NMF
- Online NMF for streaming
- NMF variants module (`nmf_variants.rs`)

### Sparse PCA and Dictionary Learning
- Sparse PCA via LASSO encoding (`sparse_pca.rs`)
- Sparse coding transform (`sparse_coding_transform/`)

### Metric Learning
- Mahalanobis distance learning
- LMNN (Large Margin Nearest Neighbor)
- NCA (Neighborhood Components Analysis)
- Advanced metric learning extensions (`metric_learning_ext/`)

### Kernel Methods
- Kernel PCA
- Deep kernel learning (`deep_kernel.rs`)
- Random Fourier Features (RFF) and Orthogonal RF (`random_features.rs`)

### Optimal Transport
- Wasserstein distance
- Sinkhorn-Knopp regularized OT
- Sliced Wasserstein distance
- OT-based domain adaptation (`optimal_transport.rs`)

### Topological Data Analysis (TDA)
- Vietoris-Rips complex
- Persistent homology: Betti numbers, persistence diagrams
- Persistence landscape feature vectors
- Topological feature vectorization (`tda.rs`, `tda_ext.rs`)
- Persistent diagram analysis (`persistent_diagram.rs`)
- TopoMap / topological layout (`topomap.rs`)

### Archetypal Analysis
- Archetypal analysis via simplex vertex finding (`archetypal.rs`)

### Autoencoder-Based Reduction
- Linear autoencoder (`linear_ae.rs`)
- Nonlinear autoencoder for reduction (`autoencoder_reduction.rs`)

### Encoder Models
- Configurable encoder model wrapper (`encoder_models.rs`)

### Categorical Encoding
- One-hot encoding (sparse and dense), drop-first option
- Ordinal encoding
- Target encoding with regularization
- Binary encoding for high-cardinality
- Unknown category handling

### Missing Value Imputation
- Simple imputation: mean, median, mode, constant
- KNN imputation with multiple metrics
- Iterative imputation / MICE
- Missing indicator

### Feature Selection
- Variance threshold
- Recursive Feature Elimination (RFE)
- Mutual information-based selection (`feature_selection/`)

### Pipeline API
- Sequential transformation chains (`pipeline.rs`)
- ColumnTransformer for column-wise transforms
- Consistent fit/transform/inverse_transform API

### Signal Transforms
- DWT 1D and 2D: Haar, Daubechies, Symlet, Coiflet; multi-level decompose/reconstruct
- CWT: Morlet, Mexican Hat, Gaussian; scalogram
- Wavelet Packet Transform (WPT) with best-basis selection (Shannon entropy)
- STFT with multiple window functions; inverse STFT with perfect reconstruction
- Spectrograms: power, magnitude, dB scaling
- MFCC: mel filterbank, DCT-II, liftering, delta/delta-delta
- Constant-Q Transform (CQT)
- Chromagram (12-bin pitch class profiles)

### Multi-View Learning
- Multi-view PCA and CCA (`multiview/`)

### Online / Incremental Learning
- Incremental PCA (`online/`)
- Online NMF
- Streaming normalizer with partial-fit (`streaming.rs`)

### Out-of-Core Processing
- Chunked array reader/writer (`out_of_core.rs`)
- Bulk buffer read/write with pre-allocated pools

### Structure Learning
- Covariance structure estimation (`structure_learning.rs`)

### Latent Variable Models
- Latent variable model framework (`latent/`)

### Nonlinear Methods
- Nonlinear reduction wrappers (`nonlinear/`)

### Projection Methods
- Random projection, Johnson-Lindenstrauss (`projection/`)

### Factorization
- Tensor factorization utilities (`factorization/`)

## v0.4.0 Roadmap

### GPU-Accelerated Dimensionality Reduction
- GPU-accelerated UMAP (cuML-compatible path)
- GPU PCA via batched SVD on GPU
- GPU t-SNE attraction/repulsion kernels

### Online PCA and Incremental SVD
- Truly streaming (O(1) memory) online PCA
- Incremental SVD with rank-1 updates

### Multimodal Alignment
- Cross-modal embedding alignment (e.g. vision-language)
- Procrustes alignment for embedding spaces

### Advanced Optimal Transport
- Unbalanced OT (for distributions of different total mass)
- Gromov-Wasserstein distance
- Multi-marginal OT

### Improved TDA
- Alpha complex (faster than Vietoris-Rips for low-d)
- Cubical complex for image data
- Zigzag persistence

### Production Monitoring
- [x] Drift detection: KS test, PSI, Wasserstein, MMD (implemented — see src/drift/mod.rs and src/monitoring/drift_detection.rs)
- Data quality alerts and degradation tracking

## Known Issues / Technical Debt

- t-SNE convergence can be slow for very large n; KD-tree acceleration has a known approximation error that may affect cluster separation in some cases
- UMAP random seed behavior is not fully deterministic across platforms due to floating-point ordering
- Some signal transform modules currently alias `scirs2-fft` operations directly; a cleaner abstraction boundary is planned
- `topomap.rs` TDA layout is approximate; exact persistent homology needs optimization for high-dimensional inputs
- `distributed::check_node_health` returns simulated random health metrics rather than performing real network probes (see "Status downgraded from Stable to Partial" note above for full detail) — do not rely on it for real cluster health monitoring today
- GPU-accelerated dimensionality reduction (`gpu.rs`: matmul, SVD, eigendecomposition, t-SNE) is not yet implemented and returns a clear "use CPU instead" error (see "GPU-Accelerated Dimensionality Reduction" roadmap section)
