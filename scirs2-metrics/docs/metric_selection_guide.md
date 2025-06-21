# Metric Selection Guide by Task Type

This comprehensive guide helps you choose the most appropriate metrics for different machine learning tasks and scenarios. Each section provides specific recommendations based on problem characteristics, data properties, and business requirements.

## Quick Reference Table

| Task Type | Primary Metrics | Secondary Metrics | Avoid When |
|-----------|----------------|-------------------|------------|
| Balanced Binary Classification | Accuracy, F1-Score, AUC-ROC | Precision, Recall, MCC | - |
| Imbalanced Binary Classification | F1-Score, AUC-PR, MCC | Balanced Accuracy, Precision, Recall | Accuracy |
| Multi-class Classification | Macro F1, Accuracy, Confusion Matrix | Micro F1, Weighted F1, Per-class metrics | - |
| Multi-label Classification | Micro F1, Hamming Loss, Label Ranking | Macro F1, Coverage Error | Accuracy |
| Continuous Regression | RMSE, MAE, R² | MAPE, Explained Variance | - |
| Count/Rate Regression | Poisson Deviance, MAE | RMSE, MAPE | R² (if non-linear) |
| Bounded Regression | MAPE, SMAPE, MAE | RMSE, R² | - |
| Unsupervised Clustering | Silhouette Score, CH Index | DB Index, Gap Statistic | ARI (no ground truth) |
| Semi-supervised Clustering | ARI, NMI, Silhouette | V-measure, Fowlkes-Mallows | - |
| Anomaly Detection | AUC-PR, F1-Score | AUC-ROC, Precision@K | Accuracy |

## Binary Classification

### Balanced Data (≈50/50 class distribution)

**Primary Metrics:**
- **Accuracy**: Simple, interpretable baseline
- **F1-Score**: Balanced precision and recall
- **AUC-ROC**: Threshold-independent performance

```rust
use scirs2_metrics::classification::{accuracy_score, f1_score};
use scirs2_metrics::classification::curves::roc_auc_score;

fn evaluate_balanced_binary(y_true: &Array1<f64>, y_pred: &Array1<f64>, 
                           y_scores: &Array1<f64>) -> Result<()> {
    let accuracy = accuracy_score(y_true, y_pred)?;
    let f1 = f1_score(y_true, y_pred, 1.0)?;
    let auc_roc = roc_auc_score(y_true, y_scores)?;
    
    println!("Accuracy: {:.3}", accuracy);
    println!("F1-Score: {:.3}", f1);
    println!("AUC-ROC: {:.3}", auc_roc);
    
    Ok(())
}
```

**When to Use Each:**
- **Accuracy**: General performance, easy stakeholder communication
- **F1-Score**: When precision and recall are equally important
- **AUC-ROC**: Comparing models across different thresholds

### Moderately Imbalanced Data (70/30 to 90/10)

**Primary Metrics:**
- **F1-Score**: Focuses on minority class performance
- **AUC-PR**: Better than AUC-ROC for imbalanced data
- **Balanced Accuracy**: Accounts for class imbalance

```rust
use scirs2_metrics::classification::advanced::balanced_accuracy_score;
use scirs2_metrics::classification::curves::{precision_recall_curve, average_precision_score};

fn evaluate_moderate_imbalance(y_true: &Array1<f64>, y_pred: &Array1<f64>, 
                              y_scores: &Array1<f64>) -> Result<()> {
    let f1 = f1_score(y_true, y_pred, 1.0)?;
    let balanced_acc = balanced_accuracy_score(y_true, y_pred)?;
    let auc_pr = average_precision_score(y_true, y_scores)?;
    
    println!("F1-Score: {:.3}", f1);
    println!("Balanced Accuracy: {:.3}", balanced_acc);
    println!("AUC-PR: {:.3}", auc_pr);
    
    Ok(())
}
```

### Severely Imbalanced Data (>95/5)

**Primary Metrics:**
- **AUC-PR**: Focus on positive class detection
- **Matthews Correlation Coefficient**: Robust to extreme imbalance
- **Precision and Recall**: Separate evaluation of each

```rust
use scirs2_metrics::classification::advanced::matthews_corrcoef;
use scirs2_metrics::classification::{precision_score, recall_score};

fn evaluate_severe_imbalance(y_true: &Array1<f64>, y_pred: &Array1<f64>, 
                            y_scores: &Array1<f64>) -> Result<()> {
    let mcc = matthews_corrcoef(y_true, y_pred)?;
    let precision = precision_score(y_true, y_pred, 1.0)?;
    let recall = recall_score(y_true, y_pred, 1.0)?;
    let auc_pr = average_precision_score(y_true, y_scores)?;
    
    println!("Matthews Correlation: {:.3}", mcc);
    println!("Precision: {:.3}", precision);
    println!("Recall: {:.3}", recall);
    println!("AUC-PR: {:.3}", auc_pr);
    
    Ok(())
}
```

### Cost-Sensitive Applications

For applications where different errors have different costs:

```rust
use scirs2_metrics::classification::cost_sensitive::{cost_sensitive_accuracy, expected_cost};

fn evaluate_cost_sensitive(y_true: &Array1<f64>, y_pred: &Array1<f64>, 
                          cost_matrix: &Array2<f64>) -> Result<()> {
    // Define cost matrix: [TN_cost, FP_cost; FN_cost, TP_cost]
    let costs = array![[0.0, 1.0], [10.0, 0.0]]; // FN 10x more costly than FP
    
    let cost_accuracy = cost_sensitive_accuracy(y_true, y_pred, &costs)?;
    let expected_cost = expected_cost(y_true, y_pred, &costs)?;
    
    // Also evaluate component metrics
    let precision = precision_score(y_true, y_pred, 1.0)?;
    let recall = recall_score(y_true, y_pred, 1.0)?;
    
    println!("Cost-Sensitive Accuracy: {:.3}", cost_accuracy);
    println!("Expected Cost: {:.3}", expected_cost);
    println!("Precision (FP control): {:.3}", precision);
    println!("Recall (FN control): {:.3}", recall);
    
    Ok(())
}
```

## Multi-class Classification

### Balanced Multi-class

**Primary Metrics:**
- **Accuracy**: Overall performance
- **Macro F1**: Equal importance to all classes
- **Confusion Matrix**: Detailed per-class analysis

```rust
use scirs2_metrics::classification::{confusion_matrix, f1_score, accuracy_score};

fn evaluate_multiclass_balanced(y_true: &Array1<usize>, y_pred: &Array1<usize>) -> Result<()> {
    let accuracy = accuracy_score(y_true, y_pred)?;
    let macro_f1 = f1_score(y_true, y_pred, None, Some("macro"), None)?;
    let (cm, labels) = confusion_matrix(y_true, y_pred, None)?;
    
    println!("Accuracy: {:.3}", accuracy);
    println!("Macro F1: {:.3}", macro_f1);
    println!("Confusion Matrix:\n{:?}", cm);
    
    // Per-class analysis
    for (i, &label) in labels.iter().enumerate() {
        let class_precision = cm[[i, i]] as f64 / cm.row(i).sum() as f64;
        let class_recall = cm[[i, i]] as f64 / cm.column(i).sum() as f64;
        println!("Class {}: Precision={:.3}, Recall={:.3}", label, class_precision, class_recall);
    }
    
    Ok(())
}
```

### Imbalanced Multi-class

**Primary Metrics:**
- **Weighted F1**: Accounts for class frequency
- **Macro F1**: Equal treatment of all classes
- **Per-class Precision/Recall**: Detailed minority class analysis

```rust
fn evaluate_multiclass_imbalanced(y_true: &Array1<usize>, y_pred: &Array1<usize>) -> Result<()> {
    let weighted_f1 = f1_score(y_true, y_pred, None, Some("weighted"), None)?;
    let macro_f1 = f1_score(y_true, y_pred, None, Some("macro"), None)?;
    let micro_f1 = f1_score(y_true, y_pred, None, Some("micro"), None)?;
    
    println!("Weighted F1: {:.3}", weighted_f1);
    println!("Macro F1: {:.3}", macro_f1);
    println!("Micro F1: {:.3}", micro_f1);
    
    // Focus on minority classes
    let (cm, labels) = confusion_matrix(y_true, y_pred, None)?;
    let class_counts: Vec<usize> = labels.iter()
        .map(|&label| y_true.iter().filter(|&&x| x == label).count())
        .collect();
    
    // Report metrics for smallest classes
    let mut class_info: Vec<(usize, usize, f64, f64)> = Vec::new();
    for (i, &label) in labels.iter().enumerate() {
        let count = class_counts[i];
        let precision = cm[[i, i]] as f64 / cm.row(i).sum() as f64;
        let recall = cm[[i, i]] as f64 / cm.column(i).sum() as f64;
        class_info.push((label, count, precision, recall));
    }
    
    class_info.sort_by_key(|(_, count, _, _)| *count);
    
    println!("\nMinority classes performance:");
    for (label, count, precision, recall) in class_info.iter().take(3) {
        println!("Class {} (n={}): P={:.3}, R={:.3}", label, count, precision, recall);
    }
    
    Ok(())
}
```

## Regression Tasks

### Continuous Regression (Unbounded Targets)

**Primary Metrics:**
- **RMSE**: Penalizes large errors, same units as target
- **MAE**: Robust to outliers, interpretable
- **R²**: Proportion of variance explained

```rust
use scirs2_metrics::regression::{
    root_mean_squared_error, mean_absolute_error, r2_score
};

fn evaluate_continuous_regression(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<()> {
    let rmse = root_mean_squared_error(y_true, y_pred)?;
    let mae = mean_absolute_error(y_true, y_pred)?;
    let r2 = r2_score(y_true, y_pred)?;
    
    println!("RMSE: {:.3}", rmse);
    println!("MAE: {:.3}", mae);
    println!("R²: {:.3}", r2);
    
    // Additional context
    let mean_target = y_true.mean().unwrap();
    let rmse_normalized = rmse / mean_target;
    
    println!("RMSE (% of mean): {:.1}%", rmse_normalized * 100.0);
    
    Ok(())
}
```

### Count/Rate Regression (Non-negative Integer Targets)

**Primary Metrics:**
- **Mean Absolute Error**: Interpretable for count data
- **Poisson Deviance**: Appropriate for count distributions
- **Mean Absolute Percentage Error**: For rate comparisons

```rust
use scirs2_metrics::regression::{
    mean_absolute_error, mean_absolute_percentage_error, poisson_deviance
};

fn evaluate_count_regression(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<()> {
    let mae = mean_absolute_error(y_true, y_pred)?;
    let mape = mean_absolute_percentage_error(y_true, y_pred)?;
    let poisson_dev = poisson_deviance(y_true, y_pred)?;
    
    println!("MAE: {:.3}", mae);
    println!("MAPE: {:.1}%", mape);
    println!("Poisson Deviance: {:.3}", poisson_dev);
    
    // Check for zero-inflation
    let zero_true = y_true.iter().filter(|&&x| x == 0.0).count();
    let zero_pred = y_pred.iter().filter(|&&x| x == 0.0).count();
    
    println!("Zero counts - True: {}, Predicted: {}", zero_true, zero_pred);
    
    Ok(())
}
```

### Bounded Regression (0-1 or percentage targets)

**Primary Metrics:**
- **Mean Absolute Error**: Direct interpretation
- **Mean Absolute Percentage Error**: Relative performance
- **Root Mean Squared Error**: If large errors are critical

```rust
fn evaluate_bounded_regression(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<()> {
    let mae = mean_absolute_error(y_true, y_pred)?;
    let mape = mean_absolute_percentage_error(y_true, y_pred)?;
    let rmse = root_mean_squared_error(y_true, y_pred)?;
    
    println!("MAE: {:.3}", mae);
    println!("MAPE: {:.1}%", mape);
    println!("RMSE: {:.3}", rmse);
    
    // Check boundary violations
    let below_zero = y_pred.iter().filter(|&&x| x < 0.0).count();
    let above_one = y_pred.iter().filter(|&&x| x > 1.0).count();
    
    if below_zero > 0 || above_one > 0 {
        println!("Warning: {} predictions below 0, {} above 1", below_zero, above_one);
    }
    
    Ok(())
}
```

### Time Series Regression

**Primary Metrics:**
- **MAE**: Robust to outliers in time series
- **MAPE**: For interpretable percentage errors
- **Directional Accuracy**: For trend prediction

```rust
use scirs2_metrics::regression::time_series::{directional_accuracy, mean_absolute_scaled_error};

fn evaluate_time_series(y_true: &Array1<f64>, y_pred: &Array1<f64>, 
                       seasonal_period: usize) -> Result<()> {
    let mae = mean_absolute_error(y_true, y_pred)?;
    let mape = mean_absolute_percentage_error(y_true, y_pred)?;
    let directional_acc = directional_accuracy(y_true, y_pred)?;
    let mase = mean_absolute_scaled_error(y_true, y_pred, seasonal_period)?;
    
    println!("MAE: {:.3}", mae);
    println!("MAPE: {:.1}%", mape);
    println!("Directional Accuracy: {:.3}", directional_acc);
    println!("MASE: {:.3}", mase);
    
    Ok(())
}
```

## Clustering Tasks

### K-means Style Clustering (Spherical Clusters)

**Primary Metrics:**
- **Silhouette Score**: Cluster separation and cohesion
- **Calinski-Harabasz Index**: Variance-based quality
- **Inertia/Within-cluster Sum of Squares**: Direct K-means objective

```rust
use scirs2_metrics::clustering::{
    silhouette_score, calinski_harabasz_score, within_cluster_sum_of_squares
};

fn evaluate_spherical_clustering(X: &Array2<f64>, labels: &Array1<usize>) -> Result<()> {
    let silhouette = silhouette_score(X, labels, "euclidean")?;
    let ch_score = calinski_harabasz_score(X, labels)?;
    let wcss = within_cluster_sum_of_squares(X, labels)?;
    
    println!("Silhouette Score: {:.3}", silhouette);
    println!("Calinski-Harabasz Index: {:.3}", ch_score);
    println!("WCSS: {:.3}", wcss);
    
    // Cluster size analysis
    let n_clusters = labels.iter().max().unwrap() + 1;
    for cluster_id in 0..n_clusters {
        let cluster_size = labels.iter().filter(|&&x| x == cluster_id).count();
        println!("Cluster {}: {} points", cluster_id, cluster_size);
    }
    
    Ok(())
}
```

### Density-Based Clustering (Arbitrary Shapes)

**Primary Metrics:**
- **Adjusted Rand Index**: If ground truth available
- **Davies-Bouldin Index**: Cluster separation
- **Custom density metrics**: Domain-specific

```rust
use scirs2_metrics::clustering::{davies_bouldin_score, adjusted_rand_index};

fn evaluate_density_clustering(X: &Array2<f64>, labels_pred: &Array1<usize>, 
                              labels_true: Option<&Array1<usize>>) -> Result<()> {
    let db_score = davies_bouldin_score(X, labels_pred)?;
    println!("Davies-Bouldin Index: {:.3} (lower is better)", db_score);
    
    if let Some(labels_true) = labels_true {
        let ari = adjusted_rand_index(labels_true, labels_pred)?;
        println!("Adjusted Rand Index: {:.3}", ari);
    }
    
    // Handle noise points (typically labeled as -1 in DBSCAN)
    let noise_points = labels_pred.iter().filter(|&&x| x == usize::MAX).count();
    let n_clusters = labels_pred.iter()
        .filter(|&&x| x != usize::MAX)
        .max().map(|x| x + 1).unwrap_or(0);
    
    println!("Clusters found: {}", n_clusters);
    println!("Noise points: {}", noise_points);
    
    Ok(())
}
```

### Hierarchical Clustering

**Primary Metrics:**
- **Cophenetic Correlation**: Dendrogram quality
- **Silhouette Score**: At chosen cut level
- **Gap Statistic**: Optimal number of clusters

```rust
use scirs2_metrics::clustering::{cophenetic_correlation, gap_statistic};

fn evaluate_hierarchical_clustering(X: &Array2<f64>, linkage_matrix: &Array2<f64>, 
                                   labels: &Array1<usize>) -> Result<()> {
    let cophenetic_corr = cophenetic_correlation(X, linkage_matrix)?;
    let silhouette = silhouette_score(X, labels, "euclidean")?;
    let (gap_scores, k_optimal) = gap_statistic(X, 1, 10, 10)?;
    
    println!("Cophenetic Correlation: {:.3}", cophenetic_corr);
    println!("Silhouette Score: {:.3}", silhouette);
    println!("Optimal clusters (Gap): {}", k_optimal);
    
    Ok(())
}
```

## Anomaly Detection

### Binary Anomaly Detection

**Primary Metrics:**
- **AUC-PR**: Focus on anomaly detection performance
- **F1-Score**: Balance of precision and recall for anomalies
- **Precision@K**: Top-K anomalies accuracy

```rust
use scirs2_metrics::anomaly::{precision_at_k, average_precision_score};

fn evaluate_anomaly_detection(y_true: &Array1<f64>, y_scores: &Array1<f64>) -> Result<()> {
    let auc_pr = average_precision_score(y_true, y_scores)?;
    
    // Convert scores to binary predictions (threshold can be optimized)
    let threshold = y_scores.quantile(0.95)?; // Top 5% as anomalies
    let y_pred: Array1<f64> = y_scores.mapv(|x| if x > threshold { 1.0 } else { 0.0 });
    
    let f1 = f1_score(y_true, &y_pred, 1.0)?;
    let precision_at_10 = precision_at_k(y_true, y_scores, 10)?;
    
    println!("AUC-PR: {:.3}", auc_pr);
    println!("F1-Score: {:.3}", f1);
    println!("Precision@10: {:.3}", precision_at_10);
    
    Ok(())
}
```

### Novelty Detection (Clean Training Data)

**Primary Metrics:**
- **AUC-ROC**: Separation of normal vs novel
- **False Positive Rate**: At operational threshold
- **Coverage**: Percentage of normal data accepted

```rust
use scirs2_metrics::anomaly::{false_positive_rate, coverage_score};

fn evaluate_novelty_detection(y_true: &Array1<f64>, y_scores: &Array1<f64>, 
                             threshold: f64) -> Result<()> {
    let y_pred: Array1<f64> = y_scores.mapv(|x| if x > threshold { 1.0 } else { 0.0 });
    
    let auc_roc = roc_auc_score(y_true, y_scores)?;
    let fpr = false_positive_rate(y_true, &y_pred)?;
    let coverage = coverage_score(y_true, &y_pred)?;
    
    println!("AUC-ROC: {:.3}", auc_roc);
    println!("False Positive Rate: {:.3}", fpr);
    println!("Coverage (Normal Data): {:.3}", coverage);
    
    Ok(())
}
```

## Special Considerations

### Multi-label Classification

```rust
use scirs2_metrics::classification::multilabel::{
    hamming_loss, jaccard_score, coverage_error, label_ranking_loss
};

fn evaluate_multilabel(y_true: &Array2<f64>, y_pred: &Array2<f64>) -> Result<()> {
    let hamming = hamming_loss(y_true, y_pred)?;
    let jaccard = jaccard_score(y_true, y_pred, "samples")?;
    let coverage = coverage_error(y_true, y_pred)?;
    let ranking_loss = label_ranking_loss(y_true, y_pred)?;
    
    println!("Hamming Loss: {:.3} (lower is better)", hamming);
    println!("Jaccard Score: {:.3}", jaccard);
    println!("Coverage Error: {:.3}", coverage);
    println!("Label Ranking Loss: {:.3}", ranking_loss);
    
    Ok(())
}
```

### Ordinal Classification

```rust
use scirs2_metrics::classification::ordinal::{mean_absolute_error, quadratic_weighted_kappa};

fn evaluate_ordinal(y_true: &Array1<usize>, y_pred: &Array1<usize>) -> Result<()> {
    let mae = mean_absolute_error(y_true, y_pred)?;
    let qwk = quadratic_weighted_kappa(y_true, y_pred)?;
    
    // Regular classification metrics
    let accuracy = accuracy_score(y_true, y_pred)?;
    
    println!("MAE (ordinal distance): {:.3}", mae);
    println!("Quadratic Weighted Kappa: {:.3}", qwk);
    println!("Accuracy: {:.3}", accuracy);
    
    Ok(())
}
```

## Decision Framework

### 1. Identify Your Problem Type
- **Supervised vs Unsupervised**
- **Classification vs Regression** 
- **Binary vs Multi-class vs Multi-label**

### 2. Assess Data Characteristics
- **Sample size**: Small (<1K), Medium (1K-100K), Large (>100K)
- **Class balance**: Balanced, Moderately imbalanced, Severely imbalanced
- **Noise level**: Clean, Moderate noise, High noise
- **Feature dimensionality**: Low (<100), Medium (100-1K), High (>1K)

### 3. Define Business Requirements
- **Cost sensitivity**: Equal costs vs Asymmetric costs
- **Interpretability**: Black box OK vs Must be interpretable
- **Threshold flexibility**: Fixed threshold vs Adjustable threshold
- **Real-time constraints**: Batch evaluation vs Online evaluation

### 4. Select Metric Suite
- **Primary metric**: Main evaluation criterion
- **Secondary metrics**: Additional perspectives
- **Diagnostic metrics**: Debugging and analysis

### 5. Validation Strategy
- **Cross-validation**: Appropriate for sample size and data type
- **Hold-out testing**: Independent final evaluation
- **Confidence intervals**: Uncertainty quantification

Remember: The best metric suite provides multiple perspectives on model performance while aligning with your specific problem requirements and business objectives.