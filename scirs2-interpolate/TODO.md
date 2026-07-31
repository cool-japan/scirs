# scirs2-interpolate TODO

## Status: v0.6.5 (released, 2026-07-31)

Untouched by this release (no interpolate-specific changes shipped in 0.6.5); the v0.6.2 status
below remains accurate since the crate source is unchanged.

scirs2-interpolate's own test suite (freshly re-run 2026-07-15): 1143 tests pass, 13 skipped, 0 failed with default features; 1173 tests pass, 13 skipped, 0 failed with `--all-features`. See "Online / streaming interpolation" entry further down for one verified discrepancy (Kriging streaming was never implemented despite being marked done).

## v0.3.3 Completed

### 1D Interpolation
- [x] Linear and nearest-neighbor interpolation with boundary handling
- [x] Natural cubic spline with natural, not-a-knot, clamped, periodic BCs
- [x] Akima spline: outlier-robust local construction
- [x] PCHIP (Piecewise Cubic Hermite Interpolating Polynomial): shape- and monotonicity-preserving
- [x] B-splines: arbitrary-order, de Boor evaluation, knot insertion and removal
- [x] NURBS: Non-Uniform Rational B-Splines for exact conics and free-form curves
- [x] Bezier curves: de Casteljau; rational and polynomial
- [x] Tension splines: splines under tension
- [x] Penalized splines (P-splines): regularized B-spline fitting for noisy data
- [x] Monotone constrained splines
- [x] Hermite splines with user-specified derivatives
- [x] Floater-Hormann barycentric rational interpolation (arbitrary blending order d)

### Scattered Data Interpolation
- [x] RBF interpolation: multiquadric, thin-plate spline, Gaussian, inverse multiquadric, linear
- [x] RBF parameter optimization: cross-validation and LOOCV error
- [x] Compactly supported RBF kernels (Wendland, Wu)
- [x] Ordinary Kriging with variogram fitting and prediction variance
- [x] Universal Kriging with polynomial trend
- [x] Indicator Kriging for binary/categorical data
- [x] Bayesian Kriging with uncertainty quantification
- [x] Fast Kriging: local O(k^3), fixed-rank, sparse tapering, HODLR
- [x] Moving Least Squares (MLS): weighted polynomial fitting
- [x] Natural Neighbor (Sibson): Voronoi-based, C1 continuity
- [x] Thin-Plate Spline (TPS): global scattered data interpolant
- [x] Shepard's method: inverse distance weighting; modified Shepard variant
- [x] Scattered 2D interpolation via Delaunay triangulation (linear and cubic)

### Spherical and Parametric
- [x] Spherical harmonic interpolation on the sphere (real harmonics, arbitrary l,m)
- [x] Parametric arc-length curve interpolation of 2D/3D point sequences
- [x] Barycentric coordinates on arbitrary triangulated manifolds

### Multidimensional Grid Interpolation
- [x] Regular grid N-D interpolation (`RegularGridInterpolator`): linear and cubic
- [x] Tensor product interpolation on separable grids
- [x] Bivariate splines: smoothing and interpolating on 2D rectangular grids
- [x] B-spline surface fitting of 3D point clouds

### Adaptive Interpolation
- [x] Error-controlled adaptive refinement: subdivide until local tolerance met
- [x] Hierarchical multi-level sparse-grid construction
- [x] Meshless methods: partition-of-unity and reproducing-kernel particle method

### Performance
- [x] SIMD-accelerated de Boor B-spline evaluation
- [x] SIMD-accelerated pairwise distance computation for RBF
- [x] Parallel batch evaluation using Rayon
- [x] K-d tree for O(log n) nearest-neighbor queries
- [x] Ball tree for metric-space nearest-neighbor queries
- [x] Cache-aware memory access patterns in hot paths

### Bug Fixes (v0.3.1)
- [x] PCHIP extrapolation: switched to linear extension at endpoints to avoid polynomial blow-up (issue #96)
- [x] Bicubic Hermite: corrected 4x4 Hermite matrix transpose
- [x] CubicSpline boundary condition: fixed not-a-knot third-derivative condition

## v0.4.0 Roadmap

### GPU-Accelerated Scattered Data
- [x] GPU-accelerated RBF solve: CPU-simulated GPU dispatch for dense system assembly and direct solve — Implemented in v0.4.2 (`gpu_rbf.rs`)
- [x] GPU batch evaluation: block-chunked parallel evaluation for RBF — Implemented in v0.4.2 (`gpu_rbf.rs`)
- [x] GPU-accelerated k-d tree queries for large scattered datasets — Implemented in v0.4.3 (`gpu_kdtree/` module: `GpuKdTree`, `KdTreeConfig`, `knn_auto_dispatch`; CPU Rayon batch path + gpu_kdtree feature stub for wgpu linear-scan dispatch)

### Machine Learning Enhanced Interpolation
- [x] Physics-Informed interpolation: enforce PDE residuals as constraints — Implemented in v0.4.2 (`physics_interp.rs`)
- [x] Neural-network-enhanced interpolation: learned correction terms on top of RBF (planned 2026-04-17)
  - **Goal:** `ResidualMlpRbf` — fit RBF, compute residuals rᵢ = yᵢ - rbf(xᵢ), train 2-3 layer MLP on (xᵢ, rᵢ), predict = rbf(x) + mlp(x).
  - **Design:** `neural_enhanced/residual_mlp_rbf.rs`. `ResidualMlpRbfConfig { hidden_sizes, activation, epochs, lr, batch_size, l2, seed }`. Pure-ndarray analytic backprop (same pattern as Wave 46 SimCSE). `Activation::{Tanh, Relu, GeluApprox}`.
  - **Files:** `scirs2-interpolate/src/neural_enhanced/{mod.rs,residual_mlp_rbf.rs,tiny_mlp.rs}` (new), `scirs2-interpolate/src/lib.rs`, `scirs2-interpolate/tests/neural_enhanced_tests.rs` (new).
  - **Tests:** `residual_rbf_reduces_error_on_noisy_sin`, `residual_rbf_matches_pure_rbf_when_epochs_zero`, `residual_rbf_deterministic_with_seed`, `tiny_mlp_forward_backward_gradient_check` (finite-diff).
  - **Risk:** Backprop correctness — finite-diff gradient check catches silent bugs.
- [x] Gaussian Process surrogate with automatic kernel structure discovery — Implemented in v0.4.3 (`auto_kernel_gp/`)
- [x] Deep Kriging: deep neural feature maps combined with Kriging — Implemented in v0.4.2 (`deep_kriging/mlp_kriging.rs`)
- [x] Active learning for adaptive sampling: minimize interpolation error with fewest evaluations — Implemented in v0.4.2 (`active_learning.rs`)

### New Interpolation Methods
- [x] Hermite-Birkhoff interpolation: arbitrary derivative data at arbitrary points — Implemented in v0.4.0
- [x] Polyharmonic splines: higher-order thin-plate generalizations — Implemented in v0.4.0
- [x] Subdivision surfaces: Loop and Catmull-Clark subdivision for smooth surfaces — Implemented in v0.4.0
- [x] Kernel interpolation on Lie groups and homogeneous spaces — Implemented in v0.4.3 (`lie_group/` module: sphere/so3/se3, Heat/Matérn/SphericalHarmonic kernels, 34 tests)

### High-Dimensional and Tensor Methods
- [x] Sparse grid interpolation via Smolyak construction — Implemented in v0.4.0 (`sparse_grid/smolyak.rs`)
- [x] Tensor-train / TT-cross interpolation for very high-dimensional grids — Implemented in v0.4.0 (`tensor_train/` module)
- [x] ANOVA decomposition for variance-based adaptive sparse grids — Implemented in v0.4.3 (`sparse_grid/anova.rs`)
  - **Goal:** Classical ANOVA f(x) = f₀ + Σᵢ fᵢ(xᵢ) + Σᵢⱼ fᵢⱼ(xᵢ,xⱼ) + ... Returns `AnovaDecomposition { mean, main_effects, interactions, sobol_indices }`. Sobol indices Sᵢ = Var(fᵢ)/Var(f).
  - **Design:** `sparse_grid/anova.rs`. Gauss-Legendre integration for marginal integrals. Recursive subtraction for higher-order terms (truncated at `max_order`). `AnovaConfig { max_order, n_quad_points }`. Ishigami function as golden benchmark (S₁≈0.31, S₂≈0.44, S₁₃≈0.24).
  - **Files:** `scirs2-interpolate/src/sparse_grid/anova.rs` (new), `scirs2-interpolate/src/sparse_grid/mod.rs`, `scirs2-interpolate/tests/anova_tests.rs` (new).
  - **Tests:** `anova_constant_function_has_only_mean`, `anova_sum_of_main_effects_exact_for_separable`, `anova_sobol_indices_sum_to_one`, `anova_ishigami_function_recovers_known_sobol_values`, `anova_max_order_truncation_stable`.
  - **Risk:** Ishigami golden values used as reference — standard ANOVA benchmark.
- [x] Anchored-ANOVA: exploit low effective dimensionality — Implemented in v0.4.3 (`sparse_grid/anchored_anova.rs`)
  - **Goal:** Anchored decomposition around anchor c: fᵢ(xᵢ) = f(xᵢ, c_{-i}) - f(c). No integration needed. `AnchoredAnovaDecomposition` + `adaptive_anchored_anova_refinement` (add terms until marginal variance < tol).
  - **Design:** `sparse_grid/anchored_anova.rs`. Default anchor c = 0.5·1. Recursive subtraction at order 1, 2, ... Kuo-Sloan-Wasilkowski (2010) approach.
  - **Files:** `scirs2-interpolate/src/sparse_grid/anchored_anova.rs` (new), `scirs2-interpolate/src/sparse_grid/mod.rs`, `scirs2-interpolate/tests/anchored_anova_tests.rs` (new).
  - **Tests:** `anchored_anova_reproduces_separable_function_exactly`, `anchored_anova_exact_for_polynomial_of_bounded_degree`, `anchored_anova_truncation_converges_for_low_effective_dim`, `anchored_anova_anchor_choice_affects_higher_order_residual`.
  - **Risk:** Terms not orthogonal (unlike classical ANOVA) — document clearly.

### Approximate and Streaming Interpolation
- [x] Random feature approximation of RBF (Rahimi-Recht) (planned 2026-04-17)
  - **Goal:** Implement Rahimi-Recht random Fourier features: for shift-invariant RBF kernels, approximate K(x,y) ≈ ⟨z(x),z(y)⟩ with z(x) = sqrt(2/D)·cos(ωᵀx+b), ω ~ p_K(ω). Provide RFF-based RBF regressor with linear-in-D prediction.
  - **Design:** `random_features/mod.rs` → full implementation. Support Gaussian (normal ω), Laplacian (Cauchy ω), Matern-3/2 & Matern-5/2 (student-t-like). Feature map `FourierFeatureMap::new(kernel, d_in, d_out, rng)` and `RandomFeaturesRegressor` using linear ridge regression via `scirs2-linalg`. Provide orthogonal random features (ORF) as an optional higher-accuracy variant.
  - **Files:** `scirs2-interpolate/src/random_features/mod.rs`, `scirs2-interpolate/src/random_features/{feature_map.rs,regressor.rs,orthogonal.rs}` (new), `scirs2-interpolate/tests/random_features_tests.rs` (new), `scirs2-interpolate/TODO.md`.
  - **Prerequisites:** none.
  - **Tests:** `rff_kernel_approximation_error_decays`, `rff_regressor_matches_rbf_on_known_function`, `rff_orthogonal_lower_variance_than_gaussian`, `rff_multiple_kernel_types`.
  - **Risk:** variance of the kernel approximation can be high at low D; ORF mitigates this — ship it in the same PR.
- [x] Nystrom approximation for large Kriging systems (planned 2026-04-17)
  - **Goal:** Production Nyström approximator that picks m ≪ n landmarks, approximates the kernel matrix as K̃ = K_{n,m} K_{m,m}^{-1} K_{m,n}, and exposes a drop-in replacement for `kriging` with O(n·m²) prediction cost.
  - **Design:** `nystrom/mod.rs` grows three landmark-selection strategies — uniform random, k-means centres, leverage-score sampling (Drineas-Mahoney, with approximate-leverage via randomized SVD) — and a `NystromKriging<Kernel>` struct with `fit` / `predict` / `predict_variance`. Use `scirs2-core::random` for RNG, `scirs2-linalg` Cholesky + randomized SVD for K_{m,m}^{-1} and leverage-score paths. Shared covariance-function trait with the full-kriging path; no duplication.
  - **Files:** `scirs2-interpolate/src/nystrom/mod.rs`, `scirs2-interpolate/src/nystrom/{landmarks.rs,predictor.rs}` (new), `scirs2-interpolate/src/lib.rs` (re-export), `scirs2-interpolate/tests/nystrom_tests.rs` (new), `scirs2-interpolate/TODO.md`.
  - **Prerequisites:** none.
  - **Tests:** `nystrom_converges_to_full_kriging_as_m_grows`, `nystrom_uniform_vs_kmeans_landmark_accuracy`, `nystrom_leverage_score_accuracy`, `nystrom_prediction_variance_positive`, `nystrom_large_n_memory_footprint`.
  - **Risk:** leverage-score computation can itself be expensive — provide both exact (small n) and approximate (randomized) paths.
- [x] Online / streaming interpolation for RBF and splines; **kriging streaming was not delivered** (verified 2026-07-15) (planned 2026-04-17)
  - **Goal:** Sherman-Morrison rank-1 update path for RBF/kriging so that appending a new (x_i,y_i) pair costs O(n²) instead of O(n³). Bounded-memory sliding-window variant. For splines, incremental knot insertion + local re-spline.
  - **Actual delivered scope:** `src/streaming_online/mod.rs` (single consolidated file, 673 lines — no separate `rbf.rs`/`kriging.rs`/`spline.rs`) implements `OnlineRbfInterpolator` + `OnlineConfig` + `UpdateStrategy` with genuine Sherman-Morrison updates, plus a spline knot-insertion path. Confirmed via `tests/streaming_online_tests.rs`: `streaming_rbf_matches_full_resolve_per_step`, `streaming_rbf_sliding_window_bounded_memory`, `streaming_spline_insert_knot` all exist and pass. There is **no Kriging streaming/online type anywhere in the module or its tests** — grep for "kriging" (case-insensitive) in `src/streaming_online/` and `tests/streaming_online_tests.rs` returns zero hits, and the planned `streaming_kriging_numerical_stability_over_1000_steps` test does not exist. Treat streaming Kriging as a genuinely open item, not done.
  - **Design:** Extend `streaming_online/mod.rs`. Add `StreamingRbf` holding factorised K^{-1} (or R from QR/Cholesky) and applying the Sherman-Morrison-Woodbury identity on `add_sample`. Sliding-window: combine add + delete (reverse Sherman-Morrison or refactorise when numerical drift crosses a threshold). For splines: `StreamingCubicSpline::insert_knot(x_i, y_i)` with local de-Boor update. Numerically stable regularization to keep K positive definite.
  - **Files:** `scirs2-interpolate/src/streaming_online/mod.rs`, `scirs2-interpolate/tests/streaming_online_tests.rs`, `scirs2-interpolate/TODO.md`.
  - **Prerequisites:** none.
  - **Tests:** `streaming_rbf_matches_full_resolve_per_step`, `streaming_rbf_sliding_window_bounded_memory`, `streaming_spline_insert_knot` (done); `streaming_kriging_numerical_stability_over_1000_steps` (not done — no Kriging streaming implementation exists).
  - **Risk:** Sherman-Morrison drifts under ill-conditioning; refactorise on ESS-style trigger.
- [x] Out-of-core interpolation: disk-backed coefficient storage for huge datasets (2026-04-17)
  - **Goal:** `OutOfCoreRbf` + `OutOfCoreKriging` that handle n ≥ 10^6 via (a) chunked sample loading, (b) disk-backed memmap'd coefficient storage, (c) block-Cholesky panel factorization.
  - **Design:** `outofcore/mod.rs`. `OutOfCoreConfig { chunk_size, cache_size_mb, scratch_dir }`. Coefficients as column-major f64 memmap. For Kriging: store K matrix blocks + L factor blocks. Use `tempfile` (check workspace; add if missing).
  - **Files:** `scirs2-interpolate/src/outofcore/mod.rs` (new), `scirs2-interpolate/src/outofcore/{rbf.rs,kriging.rs,storage.rs}` (new), `scirs2-interpolate/src/lib.rs`, `scirs2-interpolate/tests/outofcore_tests.rs` (new).
  - **Tests:** `ooc_rbf_matches_in_memory_for_n_10k`, `ooc_kriging_chunked_cholesky_matches_in_memory`, `ooc_handles_scratch_dir_cleanup`, `ooc_resumes_from_partial_write`.
  - **Risk:** Peak-memory assertion may need `#[ignore]` on constrained CI; document.

### Usability and Tooling
- [x] Automatic method selection: given data size, dimension, and smoothness estimate, recommend best method (planned 2026-04-17)
  - **Goal:** `auto_select.rs` returns a concrete `Recommendation { method, parameters, rationale }` for any `(n, d, is_gridded, smoothness_estimate, noise_level)` input. Covers linear, cubic spline, RBF (thin-plate/gaussian/multiquadric), kriging, Nyström-kriging, random-features, B-spline, natural-neighbour.
  - **Design:** Rule-based decision tree documented with citations (Fasshauer, Wendland, Rasmussen). Smoothness estimated by finite-difference variance on uniform scattered sample (O(n) pass). Noise detection via residual of local linear fit. For very large n + moderate d, recommend Nyström or random-features. Grid data → tensor spline/B-spline. Small d + smooth → thin-plate RBF. Ship heuristics as a table of cut-offs drawn from the published literature.
  - **Files:** `scirs2-interpolate/src/auto_select.rs`, `scirs2-interpolate/tests/auto_select_tests.rs` (new), `scirs2-interpolate/TODO.md`.
  - **Prerequisites:** none.
  - **Tests:** `auto_select_small_smooth_1d_picks_cubic_spline`, `auto_select_large_scattered_picks_nystrom_or_rff`, `auto_select_noisy_picks_kriging_or_ridge`, `auto_select_high_dim_picks_sparse_grid_or_rff`, `auto_select_grid_input_picks_tensor_spline`.
  - **Risk:** benchmarking ground truth — rely on published-heuristics argument and defensive unit tests.
- [x] Extrapolation modes: nearest, linear, polynomial, reflection, periodic (planned 2026-04-17)
  - **Goal:** All five modes implemented, exposed via `ExtrapolationMode` enum, and wired into the main 1D/ND interpolators (linear, cubic, B-spline, RBF).
  - **Design:** Extend `extrapolation.rs` + `extrapolation_modules/`. Enum variants: `Nearest`, `Linear`, `Polynomial(degree: usize)`, `Reflection`, `Periodic`. Provide a trait `Extrapolate` with `extrapolate(&self, x: &ArrayView1<f64>) -> f64`. `Polynomial` uses boundary-point Neville / divided-difference extrapolation at configurable degree. `Reflection` and `Periodic` wrap the query coordinate before delegating to the interior interpolator.
  - **Files:** `scirs2-interpolate/src/extrapolation.rs`, `scirs2-interpolate/src/extrapolation_modules/{nearest.rs,linear.rs,polynomial.rs,reflection.rs,periodic.rs}` (new or extend), `scirs2-interpolate/tests/extrapolation_tests.rs` (new), `scirs2-interpolate/TODO.md`.
  - **Prerequisites:** none.
  - **Tests:** `extrap_nearest_returns_boundary_value`, `extrap_linear_matches_analytic_line`, `extrap_polynomial_degree_3_matches_cubic`, `extrap_reflection_symmetry`, `extrap_periodic_wraps_correctly_for_sin`.
  - **Risk:** Polynomial extrapolation can explode; clamp degree and document the hazard.
- [x] Grid resampling utilities: resample scattered data onto regular or irregular grids (planned 2026-04-17)
  - **Goal:** Top-level API `resample_scattered_to_grid` and `resample_grid_to_grid`. Three strategies: (a) interpolate (use any `scirs2-interpolate` interpolator), (b) rasterize (mean/median/max/min/count within grid cells), (c) conservative (area-weighted average with source-cell overlap). Handle masked / missing cells.
  - **Design:** Extend `resampling/mod.rs`. `GridSpec` struct describing target grid (axes, origin, spacing, N-D support). `ResampleStrategy::Interpolate(Box<dyn Interpolator>) | Rasterize(Aggregator) | Conservative`. Use scirs2-spatial KD-tree for scattered-neighbour lookup in rasterize; implement conservative resampling via inclusion-exclusion for axis-aligned cells.
  - **Files:** `scirs2-interpolate/src/resampling/mod.rs`, `scirs2-interpolate/src/resampling/{grid_spec.rs,interpolate_strategy.rs,rasterize.rs,conservative.rs}` (new), `scirs2-interpolate/tests/resampling_tests.rs` (new), `scirs2-interpolate/TODO.md`.
  - **Prerequisites:** none.
  - **Tests:** `resample_known_function_roundtrip_small_error`, `resample_rasterize_mean_matches_analytic`, `resample_conservative_preserves_integral`, `resample_handles_masked_cells`, `resample_2d_and_3d`.
  - **Risk:** Conservative resampling on non-axis-aligned grids is expensive; scope to axis-aligned for v0.4.3 and note limitation.
- [x] Symbolic derivative evaluation: analytically differentiate spline representations (planned 2026-04-17)
  - **Goal:** `trait Differentiable { type Derivative; fn derivative(&self, order: usize) -> Self::Derivative; }` implemented for `BSpline1D`, `CubicSpline`, `AkimaSpline`, `PchipSpline`.
  - **Design:** `symbolic_derivative/mod.rs`. B-spline derivative: degree p-1, knot vector trimmed, c'ᵢ = p·(cᵢ₊₁-cᵢ)/(tᵢ₊ₚ₊₁-tᵢ₊₁). Piecewise cubic: c₁ + 2c₂x + 3c₃x² per interval.
  - **Files:** `scirs2-interpolate/src/symbolic_derivative/mod.rs` (new), `{bspline_deriv,cubic_deriv,pchip_deriv,akima_deriv}.rs` (new), `scirs2-interpolate/src/lib.rs`, `scirs2-interpolate/tests/symbolic_derivative_tests.rs` (new).
  - **Tests:** `bspline_derivative_matches_finite_difference_at_interior`, `cubic_spline_derivative_matches_analytic_for_x_cubed`, `pchip_derivative_preserves_monotonicity`, `akima_derivative_continuous_at_knots`, `second_derivative_of_constant_spline_is_zero`.
  - **Risk:** Private field access — add `pub(crate)` accessors if needed.

## Known Issues

- Kriging with very large nugget values (> 0.5 * signal variance) may produce numerically unstable Cholesky factorizations; increasing nugget regularization is the current workaround.
- Natural Neighbor interpolation does not extrapolate beyond the convex hull of the data; it returns an error for out-of-hull queries.
- NURBS surface fitting is implemented for structured (grid-like) point clouds; unstructured point cloud fitting requires the scattered-2D module.
- Meshless partition-of-unity methods require a minimum patch overlap ratio of 1.5x; smaller overlaps cause oscillation in the unity partition.
- B-spline surface fitting performance degrades for large grids (> 200x200 control points) due to dense system assembly; sparse assembly is planned.
- `KdTree` leaf nodes holding more than one point (`src/spatial/kdtree.rs::build_subtree`) store only `indices[0]`, silently dropping the remaining points in that partition once input size exceeds the default `leaf_size` of 10; `BallTree` is not affected.
