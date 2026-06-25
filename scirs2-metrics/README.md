# SciRS2 Metrics

[![crates.io](https://img.shields.io/crates/v/scirs2-metrics.svg)](https://crates.io/crates/scirs2-metrics)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](../LICENSE)
[![Documentation](https://img.shields.io/docsrs/scirs2-metrics)](https://docs.rs/scirs2-metrics)
[![Version](https://img.shields.io/badge/version-0.5.1-green)]()

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
- Homogeneity, completeness, contingency matrix, pair confusion matrix
- Cluster stability, consensus scoring, gap statistic

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

### Generative Model Evaluation
- Fréchet Inception Distance (FID)
- Inception Score (IS)
- Precision and Recall for generative models
- Maximum Mean Discrepancy (MMD)
- Kernel-based evaluation metrics

### Segmentation Metrics
- Pixel accuracy, mean pixel accuracy
- Intersection over Union (IoU) per-class and mean IoU
- Dice coefficient, Jaccard index
- Boundary F-measure
- Panoptic Quality (PQ)

### Fairness and Bias Detection
- Demographic parity difference and ratio
- Equalized odds difference
- Equal opportunity difference
- Disparate impact ratio
- Consistency score across groups
- Slice analysis for subgroup performance
- Intersectional fairness measures
- Bias detection and robustness testing

### Advanced Regression Metrics (`regression_advanced`)
- Pinball (quantile) loss for quantile regression
- Interval score for prediction interval evaluation
- Coverage probability, interval width
- Winkler score

### Streaming Metrics (Online Estimation)
- Memory-efficient online evaluation for large-scale and real-time applications
- **Optimization patterns** (`streaming/optimization/patterns/`):
  - Batching: group evaluations into batches for efficiency
  - Buffering: ring-buffer based streaming metric windows
  - Partitioning: shard metrics by key/group
  - Windowing: sliding and tumbling window metrics

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
scirs2-metrics = "0.5.1"
```

Selective features:

```toml
[dependencies]
scirs2-metrics = { version = "0.5.1", features = ["neural_common", "plotters_backend"] }
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
let precision = precision_score(&y_true, &y_pred, None, None, None)?;
let recall    = recall_score(&y_true, &y_pred, None, None, None)?;
let f1        = f1_score(&y_true, &y_pred, None, None, None)?;
let auc       = roc_auc_score(&y_true, &y_scores)?;

println!("Acc={:.3} P={:.3} R={:.3} F1={:.3} AUC={:.3}", accuracy, precision, recall, f1, auc);
```

### Regression

```rust
use scirs2_metrics::regression::{mean_squared_error, mean_absolute_error, r2_score};
use scirs2_core::ndarray::array;

let y_true = array![3.0, -0.5, 2.0, 7.0, 2.0];
let y_pred = array![2.5,  0.0, 2.1, 7.8, 1.8];

let mse = mean_squared_error(&y_true, &y_pred, None)?;
let mae = mean_absolute_error(&y_true, &y_pred, None)?;
let r2  = r2_score(&y_true, &y_pred, None)?;
println!("MSE={:.4} MAE={:.4} R2={:.4}", mse, mae, r2);
```

### Clustering

```rust
use scirs2_metrics::clustering::{silhouette_score, adjusted_rand_index, davies_bouldin_score};
use scirs2_core::ndarray::{array, arr2};

let data   = arr2(&[[1.0,2.0],[1.5,1.8],[5.0,8.0],[8.0,8.0],[1.0,0.6],[9.0,11.0]]);
let pred   = array![0, 0, 1, 1, 0, 1];
let truth  = array![0, 0, 1, 1, 0, 2];

let silhouette = silhouette_score(&data, &pred, None, None)?;
let db         = davies_bouldin_score(&data, &pred)?;
let ari        = adjusted_rand_index(&truth, &pred)?;
println!("Silhouette={:.3} DB={:.3} ARI={:.3}", silhouette, db, ari);
```

### Object Detection

```rust
use scirs2_metrics::detection::{iou_score, average_precision, compute_map};

// Compute IoU between predicted and ground-truth bounding boxes
// Boxes in [x1, y1, x2, y2] format
let pred_box  = [10.0_f64, 20.0, 50.0, 60.0];
let true_box  = [12.0_f64, 22.0, 48.0, 58.0];
let iou       = iou_score(&pred_box, &true_box);

// mAP@0.5 over multiple classes
let map50 = compute_map(&predictions, &ground_truths, 0.5)?;
println!("IoU={:.3} mAP@0.5={:.3}", iou, map50);
```

### Information Retrieval

```rust
use scirs2_metrics::ranking::ir_metrics::{ndcg_score, mean_average_precision, mrr_score};
use scirs2_core::ndarray::array;

// NDCG@5 for a single query
let relevance = array![3.0, 2.0, 3.0, 0.0, 1.0, 2.0];
let scores    = array![0.9, 0.8, 0.7, 0.6, 0.5, 0.4];
let ndcg      = ndcg_score(&relevance, &scores, Some(5))?;

// MAP over a query set
let map = mean_average_precision(&queries_relevance, &queries_scores, None)?;
println!("NDCG@5={:.4} MAP={:.4}", ndcg, map);
```

### Fairness Metrics

```rust
use scirs2_metrics::fairness::advanced::{demographic_parity, equalized_odds, disparate_impact};

let y_true   = array![1, 0, 1, 1, 0, 0, 1, 0];
let y_pred   = array![1, 0, 1, 0, 0, 1, 1, 0];
let groups   = array![0, 0, 0, 0, 1, 1, 1, 1];   // 0 = group A, 1 = group B

let dp_diff  = demographic_parity(&y_pred, &groups)?;
let eo_diff  = equalized_odds(&y_true, &y_pred, &groups)?;
let di_ratio = disparate_impact(&y_pred, &groups)?;

println!("DP diff={:.4} EO diff={:.4} DI ratio={:.4}", dp_diff, eo_diff, di_ratio);
```

### Streaming Metrics

```rust
use scirs2_metrics::streaming::optimization::patterns::{
    batching::BatchingAccumulator,
    windowing::SlidingWindowMetric,
};

// Sliding window accuracy over last 1000 predictions
let mut window = SlidingWindowMetric::new(1000);
for (pred, truth) in prediction_stream {
    window.push(pred, truth);
    println!("Window accuracy: {:.4}", window.accuracy());
}

// Batch evaluations for efficiency
let mut batcher = BatchingAccumulator::new(64);
for batch in dataset.batches(64) {
    batcher.add_batch(&batch.predictions, &batch.labels);
}
let final_metrics = batcher.finalize()?;
```

### Visualization

```rust
use scirs2_metrics::{
    classification::roc_curve,
    visualization::helpers,
    visualization::VisualizationOptions,
};

let (fpr, tpr, _) = roc_curve(&y_true, &y_scores, None, None)?;
let viz = helpers::visualize_roc_curve(fpr.view(), tpr.view(), None, Some(auc));

let options = VisualizationOptions::new()
    .with_width(800)
    .with_height(600)
    .with_grid(true)
    .with_legend(true);

viz.save_to_file("roc_curve.png", Some(options))?;
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
start_http_server(dashboard.clone())?;

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
use scirs2_metrics::integration::optim::{MetricScheduler, HyperParameterTuner, HyperParameter};

let mut scheduler = MetricScheduler::new(0.1, 0.5, 2, 0.001, "val_loss", false);
let new_lr = scheduler.step_with_metric(val_loss);

let params = vec![
    HyperParameter::new("learning_rate", 0.01, 0.001, 0.1),
    HyperParameter::new("hidden_size", 5.0, 2.0, 20.0),
];
let mut tuner = HyperParameterTuner::new(params, "accuracy", true, 20);
let result = tuner.random_search(|p| train_and_evaluate(p))?;
```

## API Summary

| Module | Key Functions |
|--------|--------------|
| `classification` | `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `roc_auc_score`, `confusion_matrix`, `average_precision_score` |
| `regression` | `mean_squared_error`, `mean_absolute_error`, `r2_score`, `mape`, `explained_variance_score` |
| `clustering` | `silhouette_score`, `calinski_harabasz_score`, `davies_bouldin_score`, `adjusted_rand_index`, `normalized_mutual_info_score` |
| `ranking` | `ndcg_score`, `mean_average_precision`, `mrr_score`, `precision_at_k` |
| `detection` | `iou_score`, `average_precision`, `compute_map`, `nms` |
| `fairness.advanced` | `demographic_parity`, `equalized_odds`, `disparate_impact` |
| `segmentation` | `pixel_accuracy`, `mean_iou`, `dice_coefficient` |
| `generative` | `frechet_inception_distance`, `inception_score`, `mmd` |
| `streaming.optimization.patterns` | `BatchingAccumulator`, `SlidingWindowMetric`, `BufferingAccumulator`, `PartitionedMetric` |
| `evaluation` | `cross_val_score`, `train_test_split`, `learning_curve`, `grid_search_cv` |
| `visualization` | `visualize_roc_curve`, `visualize_confusion_matrix`, `visualize_metric` |

## Performance

- **SIMD acceleration** with automatic hardware detection (SSE2, AVX2, AVX-512)
- **Parallel processing** via Rayon for batch metric computation
- **Memory-efficient streaming** algorithms for large-scale evaluation
- **142+ comprehensive tests** with numerical precision validation
- **Zero-warning** builds

## License

Licensed under the Apache License 2.0. See [LICENSE](../LICENSE) for details.

## Authors

COOLJAPAN OU (Team KitaSan)
