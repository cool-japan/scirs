# scirs2-metrics TODO

## Status: v0.3.4 Released (March 18, 2026) — v0.4.3 in progress (2026-05-03)

## v0.3.3 Completed

### Classification Metrics
- [x] `accuracy_score` with sample weighting
- [x] `precision_score`, `recall_score`, `f1_score`, `fbeta_score`
- [x] `precision_recall_fscore_support` (all in one)
- [x] Matthews correlation coefficient (MCC)
- [x] Balanced accuracy, Cohen's kappa
- [x] `roc_curve`, `roc_auc_score`
- [x] `average_precision_score`, `precision_recall_curve`
- [x] `confusion_matrix`, `classification_report`
- [x] `log_loss`, `brier_score_loss`
- [x] `hinge_loss`, `hamming_loss`, `jaccard_score`
- [x] Multi-class: micro/macro/weighted/samples averaging
- [x] `cohen_kappa_score`, `matthews_corrcoef`
- [x] Optimal threshold: `g_means_score`, `find_optimal_threshold`
- [x] Label binarization utilities

### Regression Metrics
- [x] `mean_squared_error` (MSE), `root_mean_squared_error` (RMSE)
- [x] `mean_absolute_error` (MAE), `median_absolute_error`
- [x] `mean_absolute_percentage_error` (MAPE), symmetric MAPE
- [x] `r2_score`, `explained_variance_score`, adjusted R²
- [x] `mean_squared_log_error` (MSLE)
- [x] `mean_tweedie_deviance`, Poisson/Gamma deviance
- [x] Huber loss, quantile (pinball) loss
- [x] Max error, relative absolute/squared error
- [x] `regression_advanced`: interval score, coverage probability, Winkler score

### Clustering Metrics
- [x] `silhouette_score`, `silhouette_samples`
- [x] `calinski_harabasz_score` (variance ratio)
- [x] `davies_bouldin_score`
- [x] Dunn index, gap statistic
- [x] `adjusted_rand_index` (ARI)
- [x] `normalized_mutual_info_score`, `adjusted_mutual_info_score`
- [x] `homogeneity_completeness_v_measure`
- [x] `fowlkes_mallows_score`
- [x] Contingency matrix, pair confusion matrix
- [x] Cluster stability, consensus scoring

### Ranking and Information Retrieval
- [x] `ndcg_score` at k, DCG
- [x] `mean_average_precision` (MAP), MAP@k
- [x] `mrr_score` (MRR)
- [x] `precision_at_k`, `recall_at_k`
- [x] Kendall's tau, Spearman's rank correlation
- [x] Label ranking average precision (LRAP)
- [x] `ir_metrics` module: comprehensive IR evaluation

### Object Detection Metrics (New in v0.3.1)
- [x] `iou_score` for axis-aligned bounding boxes
- [x] `average_precision` at IoU threshold
- [x] `compute_map` (mAP) with configurable IoU thresholds
- [x] Non-Maximum Suppression (NMS) utilities
- [x] PASCAL VOC-style evaluation
- [x] COCO-style mAP@[0.5:0.95] evaluation
- [x] Per-class AP breakdown

### Generative Model Evaluation (New in v0.3.1)
- [x] Fréchet Inception Distance (FID)
- [x] Inception Score (IS)
- [x] Precision and Recall for generative models
- [x] Maximum Mean Discrepancy (MMD)
- [x] Kernel-based evaluation metrics

### Segmentation Metrics (New in v0.3.1)
- [x] Pixel accuracy, mean pixel accuracy
- [x] Per-class IoU, mean IoU (mIoU)
- [x] Dice coefficient, Jaccard index
- [x] Boundary F-measure
- [x] Panoptic Quality (PQ)

### Fairness and Bias Detection
- [x] `demographic_parity` (difference and ratio)
- [x] `equalized_odds` difference
- [x] `equal_opportunity` difference
- [x] `disparate_impact` ratio
- [x] Consistency score across subgroups
- [x] Slice analysis for subgroup performance
- [x] Intersectional fairness measures
- [x] Robustness testing: performance invariance, sensitivity

### Streaming Metrics
- [x] Memory-efficient online evaluation
- [x] `streaming/optimization/patterns/batching.rs` - batch accumulator
- [x] `streaming/optimization/patterns/buffering.rs` - ring-buffer streaming
- [x] `streaming/optimization/patterns/partitioning.rs` - keyed/group metrics
- [x] `streaming/optimization/patterns/windowing.rs` - sliding/tumbling windows
- [x] Online Welford variance for streaming statistics

### Evaluation Framework
- [x] K-fold cross-validation, stratified K-fold, leave-one-out
- [x] Time series cross-validation
- [x] `cross_val_score`, `cross_validate`
- [x] Learning curve, validation curve generation
- [x] `grid_search_cv`, `randomized_search_cv`
- [x] Bootstrap confidence intervals
- [x] McNemar's test, Friedman test, Wilcoxon signed-rank test

### Bayesian Evaluation
- [x] Bayes factor computation
- [x] BIC, AIC, WAIC, LOO-CV, DIC
- [x] Posterior predictive checks
- [x] Bayesian model averaging
- [x] Credible intervals and HPD intervals

### Hardware Acceleration
- [x] SIMD-accelerated computations (SSE2, AVX2, AVX-512)
- [x] Automatic hardware capability detection
- [x] Configurable acceleration settings
- [x] Parallel Rayon-based batch evaluation

### Visualization
- [x] ROC curve, precision-recall curve, calibration curve
- [x] Confusion matrix heatmap (normalized and unnormalized)
- [x] Learning and validation curves
- [x] Histogram, scatter, bar chart, heatmap
- [x] Plotters backend (PNG/SVG)
- [x] Plotly backend (interactive HTML)
- [x] Dashboard server (HTTP, Chart.js, RESTful API)
- [x] Export: JSON, CSV, HTML

### Neural Integration (`neural_common` feature)
- [x] `NeuralMetricAdapter` for `scirs2-neural` trainer callbacks
- [x] `MetricsCallback` for per-epoch metric collection
- [x] Training history visualization

### Optimization Integration (`optim_integration` feature)
- [x] `MetricScheduler` (reduce-on-plateau)
- [x] `HyperParameterTuner` with random and grid search
- [x] `MetricBasedReduceOnPlateau` optimizer wrapper

## v0.4.0 Roadmap

### Distributional Metrics
- [x] Wasserstein distance (Earth Mover's distance) — exact and approximate — Implemented in v0.4.0 (`distribution/mod.rs`)
- [x] Sinkhorn divergence for regularized optimal transport — Implemented in v0.4.0 (`distribution/mod.rs`)
- [x] Energy distance between empirical distributions — Implemented in v0.4.0 (`distribution/mod.rs`)
- [x] Kernel Stein Discrepancy (KSD) for goodness-of-fit — Implemented in v0.4.0 (`distribution/mod.rs`)
- [x] Total variation distance — Implemented in v0.4.0 (`distribution/mod.rs`)

### Uncertainty Calibration Metrics
- [x] Expected Calibration Error (ECE) — Implemented in v0.4.0 (`calibration/metrics.rs`)
- [x] Maximum Calibration Error (MCE) — Implemented in v0.4.0 (`calibration/metrics.rs`)
- [x] Reliability diagram generation — Implemented in v0.4.0 (`calibration/reliability.rs`)
- [x] Temperature scaling calibration diagnostic — Implemented in v0.4.0 (`calibration/advanced.rs`)
- [x] Adaptive Calibration Error (ACE) — Implemented in v0.4.0 (`calibration/metrics.rs`)
- [x] Conformal prediction coverage metrics — Implemented in v0.4.0 (`calibration/advanced.rs`)

### Active Learning Utility
- [x] Margin sampling score — Implemented in v0.4.0 (`active_learning/mod.rs` `margin_sampling_score`)
- [x] Entropy-based uncertainty score — Implemented in v0.4.0 (`active_learning/mod.rs` `EntropySampling`)
- [x] Query-by-committee disagreement — Implemented in v0.4.0 (`active_learning/mod.rs` `query_by_committee`)
- [x] Core-set selection metrics — Implemented in v0.4.0 (`active_learning/mod.rs` `CoreSet` / `greedy_k_center`)
- [x] Expected model change — Implemented in v0.4.0 (`active_learning/mod.rs` `expected_model_change`)

### Expanded Detection Metrics
- [x] 3D IoU for point cloud bounding boxes — Implemented in v0.4.0 (`detection_3d/iou_3d.rs`)
- [x] Rotated bounding box IoU — Implemented in v0.4.2 (`detection/rotated_iou.rs`)
- [x] Tracking metrics: MOTA, MOTP, IDF1 — Implemented in v0.4.0 (`tracking/` module)
- [x] Keypoint metrics: OKS (Object Keypoint Similarity), PCK — Implemented in v0.4.0 (`keypoint/mod.rs`)

### Time Series Metrics
- [x] Dynamic Time Warping (DTW) distance as metric — Implemented in v0.4.0 (`temporal/mod.rs`)
- [x] Forecast skill scores (Brier skill score, CRPS) — Implemented in v0.4.0 (`temporal/mod.rs`)
- [x] Directional accuracy (hit rate) for forecasts — Implemented in v0.4.0 (`temporal/mod.rs`)
- [x] Diebold-Mariano test for forecast comparison — Implemented in v0.4.0 (`temporal/mod.rs`)

### Documentation and Examples
- [x] Comprehensive cookbook with domain-specific metric selection guides (planned 2026-04-17)
  - **Goal:** 6 runnable Rust examples under `scirs2-metrics/examples/cookbook_*.rs` — one per domain: classification, regression, ranking, clustering, distribution, calibration. No new markdown docs.
  - **Design:** Each example: (1) synthetic data via `scirs2_core::random`, (2) compute metrics, (3) print tabulated comparison with inline commentary on when each metric is appropriate.
  - **Files:** `scirs2-metrics/examples/cookbook_{classification,regression,ranking,clustering,distribution,calibration}.rs` (all new), `scirs2-metrics/TODO.md`.
  - **Tests:** Compile-check via `cargo check --examples -p scirs2-metrics`. Add `examples_compile_check` test.
  - **Risk:** Low — examples validated via `cargo check --examples`.
- [x] Integration examples with popular ML frameworks (planned 2026-04-17)
  - **Goal:** 4 runnable examples showing SciRS2 ML module outputs → `scirs2-metrics` evaluation: neural, optimize, cluster, transform.
  - **Design:** `scirs2-metrics/examples/integration_{neural,optimize,cluster,transform}.rs`. Each generates synthetic data, runs sister-crate algorithm, computes metrics. Add sister crates as dev-deps if not present.
  - **Files:** `scirs2-metrics/examples/integration_{neural,optimize,cluster,transform}.rs` (all new), `scirs2-metrics/Cargo.toml`, `scirs2-metrics/TODO.md`.
  - **Tests:** Compile-check all 4. `framework_integration_compile_check` test.
  - **Risk:** Circular dev-deps possible — check; if circular, move to sister crate.
- [x] Benchmark report vs `sklearn.metrics` (planned 2026-04-17)
  - **Goal:** Criterion benchmark suite `scirs2-metrics/benches/sklearn_comparison.rs` measuring per-metric throughput (n=10³,10⁴,10⁵). Reference sklearn numbers from published benchmarks (hard-coded, cited). `cargo run --example sklearn_comparison_summary` prints speedup table.
  - **Design:** Criterion benches for classification (accuracy/F1/ROC-AUC), regression (MSE/MAE/R²), ranking (MAP/NDCG), clustering (silhouette). sklearn reference numbers cited from Buitinck et al. 2013.
  - **Files:** `scirs2-metrics/benches/sklearn_comparison.rs` (new), `scirs2-metrics/examples/sklearn_comparison_summary.rs` (new), `scirs2-metrics/Cargo.toml`, `scirs2-metrics/TODO.md`.
  - **Tests:** `cargo bench --bench sklearn_comparison -p scirs2-metrics --no-run` compiles. `sklearn_comparison_bench_compiles` test.
  - **Risk:** No runtime Python needed — reference numbers are static/cited.

## Known Issues

- `plotly_backend` feature generates HTML that requires a network connection to load Chart.js from CDN; add offline/bundled mode.
- `dashboard_server` requires `tokio` runtime; document how to integrate with existing async applications.
- SIMD acceleration is only available on x86/x86_64 targets; ARM NEON support is planned for v0.4.0.
- The `generative` module (FID, IS) requires pre-computed feature vectors; it does not include the Inception network — document this clearly.
