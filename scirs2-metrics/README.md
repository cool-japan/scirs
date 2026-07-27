# SciRS2 Metrics

[![crates.io](https://img.shields.io/crates/v/scirs2-metrics.svg)](https://crates.io/crates/scirs2-metrics)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../LICENSE)
[![Documentation](https://img.shields.io/docsrs/scirs2-metrics)](https://docs.rs/scirs2-metrics)
[![Version](https://img.shields.io/badge/version-0.6.3-green)]()
[![Status](https://img.shields.io/badge/status-stable-brightgreen)]()

Comprehensive machine learning evaluation metrics for the SciRS2 scientific computing ecosystem. Covers classification, regression, clustering, ranking, object detection, information retrieval, generative model evaluation, fairness, segmentation, and streaming/online metrics — with SIMD acceleration and parallel processing throughout.

## Features

### Classification Metrics
- Accuracy, precision, recall, F1-score, F-beta score
- Matthews correlation coefficient (MCC), Cohen's kappa
- Balanced accuracy, specificity, sensitivity
- ROC curve, AUC, average precision score
- Precision-recall curve and average precision (AP)
- Confusion matrix and classification report
- Log loss (cross-entropy), Brier score
- Hinge loss, Hamming loss, Jaccard score
- Multi-class and multi-label support (micro/macro/weighted averaging)
- Optimal threshold finding (G-means, custom criteria)

### Regression Metrics
- MSE, RMSE, MAE, median absolute error, max error
- R² score, explained variance, adjusted R²
- MAPE (mean absolute percentage error), SMAPE (symmetric MAPE)
- MSLE (mean squared log error), Huber loss
- Quantile loss (pinball loss), Tweedie deviance
- Relative absolute error, relative squared error
- Normalized RMSE

### Clustering Metrics
- **Internal** (no ground truth): Silhouette score/samples, Calinski-Harabasz index, Davies-Bouldin index, Dunn index
- **External** (with ground truth): Adjusted Rand Index (ARI), Normalized MI, Adjusted MI, V-measure, Fowlkes-Mallows score
- Homogeneity, completeness
- Cluster stability, consensus scoring

### Ranking and Information Retrieval
- NDCG (normalized discounted cumulative gain), DCG
- Mean Average Precision (MAP), MAP@k
- Mean Reciprocal Rank (MRR)
- Precision@k, Recall@k
- Kendall's tau, Spearman's rank correlation
- Label ranking average precision (LRAP)

### Object Detection Metrics
- Intersection over Union (IoU) for bounding boxes
- Average Precision (AP), mean AP (mAP) at IoU thresholds
- Non-Maximum Suppression (NMS) utilities
- PASCAL VOC and COCO-style evaluation protocols
- Per-class AP breakdown

### Generative Model Evaluation (`domains::generative_ai::gan_evaluation`)
- Fréchet Inception Distance (FID) — `GANEvaluationMetrics::frechet_inception_distance`
- Inception Score (IS) — `GANEvaluationMetrics::inception_score`
- Kernel Inception Distance (KID) — `GANEvaluationMetrics::kernel_inception_distance`
- General-purpose Maximum Mean Discrepancy (MMD) two-sample test — `anomaly::maximum_mean_discrepancy`

### Segmentation Metrics (via `detection` module)
- Pixel accuracy — `detection::pixel_accuracy`
- Mean IoU for segmentation masks — `detection::mean_iou_segmentation`
- Dice coefficient — `detection::dice_coefficient`

### Fairness and Bias Detection
- Demographic parity difference and ratio
- Equalized odds difference
- Equal opportunity difference
- Disparate impact ratio
- Consistency score across groups
- Slice analysis for subgroup performance
- Intersectional fairness measures
- Bias detection and robustness testing

### Streaming Metrics (Online Estimation, `streaming::advanced`)
- `AdaptiveStreamingMetrics`: memory-efficient online evaluation with adaptive window sizing
- Concept drift detection — `AdwinDetector`, `DdmDetector`, `PageHinkleyDetector`
- Streaming anomaly detection and alerting — `AnomalyDetector`, `AlertsManager`
- Performance monitoring and degradation tracking — `PerformanceMonitor`
- Metric ensembling (`MetricEnsemble`) and history buffering (`HistoryBuffer`)

### Statistical Testing and Validation
- McNemar's test for classifier comparison
- Cochran's Q test for multiple classifiers
- Friedman test (non-parametric)
- Wilcoxon signed-rank test
- Bootstrap confidence intervals
- Cross-validation utilities (K-fold, stratified, time series)

### Bayesian Evaluation
- Bayes factor model comparison
- BIC, AIC, WAIC, LOO-CV information criteria
- Posterior predictive checks
- Bayesian model averaging

### Visualization
- ROC curve, precision-recall curve, calibration curve
- Confusion matrix heatmap
- Learning and validation curves
- Histogram and scatter plots
- Dashboard server (HTTP, real-time, with Chart.js)
- Plotters and Plotly backends

## Installation

```toml
[dependencies]
scirs2-metrics = "0.6.3"
```

Selective features:

```toml
[dependencies]
scirs2-metrics = { version = "0.6.3", features = ["neural_common", "plotters_backend"] }
```

Available features:
- `plotly_backend` (default) — interactive web visualizations
- `optim_integration` (default) — integration with `scirs2-optimize`
- `neural_common` — integration with `scirs2-neural`
- `plotters_backend` — static PNG/SVG via Plotters
- `dashboard_server` — HTTP dashboard server (requires tokio)

## Quick Start

### Classification

```rust
use scirs2_metrics::classification::{accuracy_score, precision_score, recall_score, f1_score, roc_auc_score};
use scirs2_core::ndarray::array;

let y_true   = array![0, 1, 0, 1, 0, 1, 0, 1];
let y_pred   = array![0, 1, 1, 1, 0, 0, 0, 1];
let y_scores = array![0.1, 0.9, 0.8, 0.7, 0.2, 0.3, 0.4, 0.8];

let accuracy  = accuracy_score(&y_true, &y_pred)?;
let precision = precision_score(&y_true, &y_pred, 1)?;
let recall    = recall_score(&y_true, &y_pred, 1)?;
let f1        = f1_score(&y_true, &y_pred, 1)?;
let auc       = roc_auc_score(&y_true, &y_scores)?;

println!("Acc={:.3} P={:.3} R={:.3} F1={:.3} AUC={:.3}", accuracy, precision, recall, f1, auc);
```

### Regression

```rust
use scirs2_metrics::regression::{mean_squared_error, mean_absolute_error, r2_score};
use scirs2_core::ndarray::array;

let y_true = array![3.0, -0.5, 2.0, 7.0, 2.0];
let y_pred = array![2.5,  0.0, 2.1, 7.8, 1.8];

let mse = mean_squared_error(&y_true, &y_pred)?;
let mae = mean_absolute_error(&y_true, &y_pred)?;
let r2  = r2_score(&y_true, &y_pred)?;
println!("MSE={:.4} MAE={:.4} R2={:.4}", mse, mae, r2);
```

### Clustering

```rust
use scirs2_metrics::clustering::{silhouette_score, adjusted_rand_index, davies_bouldin_score};
use scirs2_core::ndarray::{array, arr2};

let data   = arr2(&[[1.0,2.0],[1.5,1.8],[5.0,8.0],[8.0,8.0],[1.0,0.6],[9.0,11.0]]);
let pred   = array![0, 0, 1, 1, 0, 1];
let truth  = array![0, 0, 1, 1, 0, 2];

let silhouette = silhouette_score(&data, &pred, "euclidean")?;
let db         = davies_bouldin_score(&data, &pred)?;
let ari        = adjusted_rand_index(&truth, &pred)?;
println!("Silhouette={:.3} DB={:.3} ARI={:.3}", silhouette, db, ari);
```

### Object Detection

```rust
use scirs2_metrics::detection::{iou, mean_average_precision_detection};

// Compute IoU between predicted and ground-truth bounding boxes
// Boxes in (x1, y1, x2, y2) format
let pred_box  = (10.0_f64, 20.0, 50.0, 60.0);
let true_box  = (12.0_f64, 22.0, 48.0, 58.0);
let iou_val   = iou(pred_box, true_box)?;

// mAP@0.5 over multiple images (each prediction box carries a confidence score)
let map50 = mean_average_precision_detection(&predictions, &ground_truths, 0.5)?;
println!("IoU={:.3} mAP@0.5={:.3}", iou_val, map50);
```

### Information Retrieval

```rust
use scirs2_metrics::ranking::{ndcg_score, mean_average_precision, mean_reciprocal_rank};
use scirs2_core::ndarray::array;

// NDCG@5 for a single query (queries are passed as one array per query)
let relevance = array![3.0, 2.0, 3.0, 0.0, 1.0, 2.0];
let scores    = array![0.9, 0.8, 0.7, 0.6, 0.5, 0.4];
let ndcg      = ndcg_score(&[relevance], &[scores], Some(5))?;

// MAP over a query set
let map = mean_average_precision(&queries_relevance, &queries_scores, None)?;
println!("NDCG@5={:.4} MAP={:.4}", ndcg, map);
```

### Fairness Metrics

```rust
use scirs2_metrics::fairness::{demographic_parity_difference, equalized_odds_difference, disparate_impact};
use scirs2_core::ndarray::array;

let y_true   = array![1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0];
let y_pred   = array![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0];
let groups   = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];   // 0 = group A, 1 = group B

let dp_diff  = demographic_parity_difference(&y_pred, &groups)?;
let eo_diff  = equalized_odds_difference(&y_true, &y_pred, &groups)?;
let di_ratio = disparate_impact(&y_pred, &groups)?;

println!("DP diff={:.4} EO diff={:.4} DI ratio={:.4}", dp_diff, eo_diff, di_ratio);
```

### Streaming Metrics

```rust
use scirs2_metrics::streaming::advanced::{AdaptiveStreamingMetrics, StreamingConfig};

// Adaptive online evaluation with drift/anomaly detection enabled by default
let config = StreamingConfig::default();
let mut metrics = AdaptiveStreamingMetrics::<f64>::new(config)?;

// Feed (truth, prediction) pairs from a live stream
for (truth, pred) in prediction_stream {
    let update = metrics.update(truth, pred, None)?;
    if update.drift_detected {
        println!("Concept drift detected!");
    }
}

// Current windowed performance snapshot
let performance = metrics.get_current_performance();
println!("{:?}", performance);
```

### Visualization

```rust
use scirs2_metrics::{
    classification::roc_curve,
    visualization::helpers,
    visualization::VisualizationOptions,
};

let (fpr, tpr, _thresholds) = roc_curve(&y_true, &y_scores)?;
let roc_viz = helpers::visualize_roc_curve(fpr.view(), tpr.view(), None, Some(auc));

// Inspect visualization metadata (title, plot type, axis labels, ...)
let metadata = roc_viz.get_metadata();
println!("{}: {:?}", metadata.title, metadata.plot_type);

// Configure rendering options (used by the Plotters/Plotly backends)
let _options = VisualizationOptions::new()
    .with_width(800)
    .with_height(600)
    .with_grid(true)
    .with_legend(true);
```

### Interactive Dashboard

```rust
use scirs2_metrics::dashboard::{InteractiveDashboard, DashboardConfig};

let mut config = DashboardConfig::default();
config.title = "Training Dashboard".to_string();
config.refresh_interval = 2;

let dashboard = InteractiveDashboard::new(config);
dashboard.add_metric("accuracy", 0.95)?;
dashboard.add_metric("loss", 0.12)?;

// Start HTTP server on port 8080 (requires `dashboard_server` feature)
#[cfg(feature = "dashboard_server")]
scirs2_metrics::dashboard::server::start_http_server(dashboard.clone())?;

// Export results
let json = dashboard.export_to_json()?;
let html = dashboard.generate_html()?;
```

## Integration with SciRS2 Ecosystem

### Neural Networks (`neural_common` feature)

```rust
use scirs2_metrics::integration::neural::{NeuralMetricAdapter, MetricsCallback};

let accuracy = NeuralMetricAdapter::<f32>::accuracy();
let f1       = NeuralMetricAdapter::<f32>::f1_score();
let callback = MetricsCallback::new(vec![accuracy, f1], true);
// Pass callback to scirs2-neural trainer
```

### Optimization (`optim_integration` feature)

```rust
use scirs2_metrics::integration::optim::{MetricLRScheduler, HyperParameterTuner, HyperParameter};

let mut scheduler = MetricLRScheduler::new(0.1, 0.5, 2, 0.001, "val_loss", false);
let new_lr = scheduler.step_with_metric(val_loss);

let params = vec![
    HyperParameter::new("learning_rate", 0.01, 0.001, 0.1),
    HyperParameter::new("hidden_size", 5.0, 2.0, 20.0),
];
let mut tuner = HyperParameterTuner::new(params, "accuracy", true, 20)?;
let result = tuner.random_search(|p| train_and_evaluate(p))?;
```

## API Summary

| Module | Key Functions |
|--------|--------------|
| `classification` | `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `roc_auc_score`, `confusion_matrix`, `average_precision_score` |
| `regression` | `mean_squared_error`, `mean_absolute_error`, `r2_score`, `mean_absolute_percentage_error`, `explained_variance_score` |
| `clustering` | `silhouette_score`, `calinski_harabasz_score`, `davies_bouldin_score`, `adjusted_rand_index`, `normalized_mutual_info_score` |
| `ranking` | `ndcg_score`, `mean_average_precision`, `mean_reciprocal_rank`, `precision_at_k` |
| `detection` | `iou`, `average_precision_bbox`, `mean_average_precision_detection`, `nms`, `pixel_accuracy`, `dice_coefficient` |
| `fairness` | `demographic_parity_difference`, `equalized_odds_difference`, `disparate_impact` |
| `domains::generative_ai::gan_evaluation` | `GANEvaluationMetrics::{inception_score, frechet_inception_distance, kernel_inception_distance}` |
| `streaming::advanced` | `AdaptiveStreamingMetrics`, `AdwinDetector`, `DdmDetector`, `PageHinkleyDetector`, `PerformanceMonitor` |
| `evaluation` | `k_fold_cross_validation`, `train_test_split`, `learning_curve`, `nested_cross_validation` |
| `visualization` | `visualize_roc_curve`, `visualize_confusion_matrix`, `visualize_metric` |

## Performance

- **SIMD acceleration** with automatic hardware detection (SSE2, AVX2, AVX-512)
- **Parallel processing** via Rayon for batch metric computation
- **Memory-efficient streaming** algorithms for large-scale evaluation
- **1,032 tests passing** (default features), **1,043 passing** with `--all-features` (4 skipped in each run) — numerical precision validated via `cargo nextest`, 2026-07-22
- **Zero-warning** builds

## License

Licensed under the Apache License 2.0. See [LICENSE](../LICENSE) for details.

## Authors

COOLJAPAN OU (Team KitaSan)
