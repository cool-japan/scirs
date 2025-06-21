# Best Practices for Model Evaluation

This guide provides comprehensive best practices for evaluating machine learning models using the SciRS2 Metrics library. Following these practices will help ensure reliable, meaningful, and reproducible evaluation results.

## General Evaluation Principles

### 1. Understand Your Task Type

Before selecting metrics, clearly understand your problem type:

- **Binary Classification**: Two classes (e.g., spam/not spam)
- **Multi-class Classification**: Multiple mutually exclusive classes  
- **Multi-label Classification**: Multiple non-exclusive labels
- **Regression**: Continuous target values
- **Clustering**: Grouping without labeled data
- **Anomaly Detection**: Identifying outliers

### 2. Consider Data Characteristics

#### Class Imbalance
For imbalanced datasets:
- **Avoid accuracy** as the primary metric
- Use **precision, recall, F1-score**, or **AUC**
- Consider **balanced accuracy** or **weighted metrics**
- Use **stratified sampling** in cross-validation

```rust
use scirs2_metrics::classification::{precision_score, recall_score, f1_score};
use scirs2_metrics::evaluation::StratifiedKFold;

// For imbalanced data, use stratified CV and appropriate metrics
let cv = StratifiedKFold::new(5, true, Some(42));
let f1_scores = cross_val_score(&model, &X, &y, cv, f1_score)?;
```

#### Data Size
- **Small datasets**: Use cross-validation, bootstrap confidence intervals
- **Large datasets**: Consider held-out validation sets
- **Very large datasets**: Use sampling for expensive metrics

#### Target Distribution
- **Skewed targets**: Consider log-transformed metrics, quantile-based metrics
- **Heavy tails**: Use robust metrics (median-based)
- **Bounded targets**: Consider percentage-based errors

### 3. Use Multiple Metrics

Never rely on a single metric. Use a comprehensive evaluation suite:

```rust
use scirs2_metrics::classification::{accuracy_score, precision_score, recall_score, f1_score};
use scirs2_metrics::classification::advanced::{matthews_corrcoef, balanced_accuracy_score};

fn comprehensive_classification_eval(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<()> {
    // Basic metrics
    let accuracy = accuracy_score(y_true, y_pred)?;
    let precision = precision_score(y_true, y_pred, 1.0)?;
    let recall = recall_score(y_true, y_pred, 1.0)?;
    let f1 = f1_score(y_true, y_pred, 1.0)?;
    
    // Advanced metrics
    let mcc = matthews_corrcoef(y_true, y_pred)?;
    let bal_acc = balanced_accuracy_score(y_true, y_pred)?;
    
    println!("Accuracy: {:.3}", accuracy);
    println!("Precision: {:.3}", precision);
    println!("Recall: {:.3}", recall);
    println!("F1-Score: {:.3}", f1);
    println!("Matthews Correlation: {:.3}", mcc);
    println!("Balanced Accuracy: {:.3}", bal_acc);
    
    Ok(())
}
```

## Cross-Validation Best Practices

### 1. Choose Appropriate CV Strategy

```rust
use scirs2_metrics::evaluation::{KFold, StratifiedKFold, TimeSeriesSplit};

// Standard classification
let cv = StratifiedKFold::new(5, true, Some(42));

// Time series data
let ts_cv = TimeSeriesSplit::new(5);

// Regression with continuous targets
let reg_cv = KFold::new(5, true, Some(42));
```

### 2. Ensure Proper Data Splitting

- **Never** use test data for model selection
- Use **train/validation/test** splits or **nested cross-validation**
- Ensure **temporal ordering** for time series data
- Consider **group-based splitting** for grouped data

### 3. Report Confidence Intervals

```rust
use scirs2_metrics::evaluation::bootstrap_confidence_interval;

let scores = cross_val_score(&model, &X, &y, cv, accuracy_score)?;
let (mean, std) = (scores.mean().unwrap(), scores.std(1.0));
let (ci_low, ci_high) = bootstrap_confidence_interval(&scores, 0.95, 1000)?;

println!("Accuracy: {:.3} ± {:.3}", mean, std);
println!("95% CI: [{:.3}, {:.3}]", ci_low, ci_high);
```

## Task-Specific Guidelines

### Binary Classification

#### Primary Metrics
1. **AUC-ROC**: Overall discriminative ability
2. **AUC-PR**: Performance on positive class (especially for imbalanced data)
3. **F1-Score**: Balance of precision and recall

#### Secondary Metrics
- **Precision**: When false positives are costly
- **Recall**: When false negatives are costly
- **Specificity**: When true negative rate matters

```rust
use scirs2_metrics::classification::curves::{roc_curve, roc_auc_score, precision_recall_curve};

fn binary_classification_eval(y_true: &Array1<f64>, y_scores: &Array1<f64>) -> Result<()> {
    // ROC analysis
    let (fpr, tpr, _) = roc_curve(y_true, y_scores)?;
    let auc_roc = roc_auc_score(y_true, y_scores)?;
    
    // Precision-Recall analysis
    let (precision, recall, _) = precision_recall_curve(y_true, y_scores)?;
    let auc_pr = average_precision_score(y_true, y_scores)?;
    
    println!("AUC-ROC: {:.3}", auc_roc);
    println!("AUC-PR: {:.3}", auc_pr);
    
    Ok(())
}
```

### Multi-class Classification

#### Averaging Strategies
- **Macro-average**: Equal weight to all classes
- **Micro-average**: Equal weight to all samples
- **Weighted-average**: Weight by class frequency

```rust
use scirs2_metrics::classification::{precision_score, recall_score, f1_score};

fn multiclass_eval(y_true: &Array1<usize>, y_pred: &Array1<usize>) -> Result<()> {
    // Macro-averaged metrics (equal class importance)
    let precision_macro = precision_score(y_true, y_pred, None, Some("macro"), None)?;
    let recall_macro = recall_score(y_true, y_pred, None, Some("macro"), None)?;
    let f1_macro = f1_score(y_true, y_pred, None, Some("macro"), None)?;
    
    // Micro-averaged metrics (equal sample importance)
    let precision_micro = precision_score(y_true, y_pred, None, Some("micro"), None)?;
    let recall_micro = recall_score(y_true, y_pred, None, Some("micro"), None)?;
    let f1_micro = f1_score(y_true, y_pred, None, Some("micro"), None)?;
    
    println!("Macro - Precision: {:.3}, Recall: {:.3}, F1: {:.3}", 
             precision_macro, recall_macro, f1_macro);
    println!("Micro - Precision: {:.3}, Recall: {:.3}, F1: {:.3}", 
             precision_micro, recall_micro, f1_micro);
    
    Ok(())
}
```

### Regression

#### Primary Metrics
1. **RMSE**: Interpretable, penalizes large errors
2. **MAE**: Robust to outliers
3. **R²**: Proportion of variance explained

#### Choosing the Right Metric
- **RMSE** when large errors are particularly problematic
- **MAE** when you want equal weight to all errors
- **MAPE** when relative errors matter more than absolute
- **Huber Loss** for robust regression with some outliers

```rust
use scirs2_metrics::regression::{
    mean_squared_error, root_mean_squared_error, mean_absolute_error, 
    r2_score, mean_absolute_percentage_error
};

fn regression_eval(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<()> {
    let mse = mean_squared_error(y_true, y_pred)?;
    let rmse = root_mean_squared_error(y_true, y_pred)?;
    let mae = mean_absolute_error(y_true, y_pred)?;
    let r2 = r2_score(y_true, y_pred)?;
    let mape = mean_absolute_percentage_error(y_true, y_pred)?;
    
    println!("RMSE: {:.3}", rmse);
    println!("MAE: {:.3}", mae);
    println!("R²: {:.3}", r2);
    println!("MAPE: {:.3}%", mape);
    
    // Check for overfitting
    if r2 > 0.99 {
        println!("Warning: Suspiciously high R² - check for data leakage");
    }
    
    Ok(())
}
```

### Clustering

#### Internal Metrics (No Ground Truth)
- **Silhouette Score**: General cluster quality
- **Davies-Bouldin Index**: Lower is better
- **Calinski-Harabasz Index**: Higher is better

#### External Metrics (With Ground Truth)
- **Adjusted Rand Index**: Corrects for chance
- **Normalized Mutual Information**: Information-theoretic measure

```rust
use scirs2_metrics::clustering::{
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_index, normalized_mutual_info_score
};

fn clustering_eval(X: &Array2<f64>, labels_pred: &Array1<usize>, 
                   labels_true: Option<&Array1<usize>>) -> Result<()> {
    // Internal metrics (always available)
    let silhouette = silhouette_score(X, labels_pred, "euclidean")?;
    let db_score = davies_bouldin_score(X, labels_pred)?;
    let ch_score = calinski_harabasz_score(X, labels_pred)?;
    
    println!("Silhouette Score: {:.3}", silhouette);
    println!("Davies-Bouldin Index: {:.3} (lower is better)", db_score);
    println!("Calinski-Harabasz Index: {:.3} (higher is better)", ch_score);
    
    // External metrics (if ground truth available)
    if let Some(labels_true) = labels_true {
        let ari = adjusted_rand_index(labels_true, labels_pred)?;
        let nmi = normalized_mutual_info_score(labels_true, labels_pred, "arithmetic")?;
        
        println!("Adjusted Rand Index: {:.3}", ari);
        println!("Normalized Mutual Information: {:.3}", nmi);
    }
    
    Ok(())
}
```

## Statistical Significance Testing

### 1. Compare Models Properly

```rust
use scirs2_metrics::evaluation::statistical_tests::{mcnemar_test, wilcoxon_signed_rank_test};

fn compare_models(y_true: &Array1<usize>, y_pred1: &Array1<usize>, 
                  y_pred2: &Array1<usize>) -> Result<()> {
    // McNemar's test for classification
    let (statistic, p_value) = mcnemar_test(y_true, y_pred1, y_pred2)?;
    
    if p_value < 0.05 {
        println!("Models are significantly different (p = {:.4})", p_value);
    } else {
        println!("No significant difference between models (p = {:.4})", p_value);
    }
    
    Ok(())
}
```

### 2. Use Appropriate Statistical Tests

- **McNemar's Test**: For comparing two classification models
- **Wilcoxon Signed-Rank Test**: For comparing two regression models
- **Friedman Test**: For comparing multiple models across datasets

## Avoiding Common Pitfalls

### 1. Data Leakage

```rust
// ❌ Wrong: Using test data in any way during model development
let all_scores = model.predict(&X_test);
let threshold = find_optimal_threshold(&y_test, &all_scores); // WRONG!

// ✅ Correct: Use only training/validation data for decisions
let val_scores = model.predict(&X_val);
let threshold = find_optimal_threshold(&y_val, &val_scores);
let final_predictions = model.predict(&X_test) > threshold;
```

### 2. Inappropriate Metric Choice

```rust
// ❌ Wrong: Using accuracy for highly imbalanced data (99% negative class)
let accuracy = accuracy_score(&y_true, &y_pred)?;

// ✅ Correct: Use appropriate metrics for imbalanced data
let f1 = f1_score(&y_true, &y_pred, 1.0)?; // Positive class F1
let auc = roc_auc_score(&y_true, &y_scores)?;
```

### 3. Ignoring Confidence Intervals

```rust
// ❌ Wrong: Reporting only point estimates
println!("Model accuracy: {:.3}", accuracy);

// ✅ Correct: Include uncertainty estimates
let scores = cross_val_score(&model, &X, &y, cv, accuracy_score)?;
let mean = scores.mean().unwrap();
let std = scores.std(1.0);
println!("Model accuracy: {:.3} ± {:.3} (std)", mean, std);
```

## Performance Considerations

### 1. Large Datasets

For very large datasets, consider:

```rust
use scirs2_metrics::optimization::sampling::{stratified_sample, random_sample};

// Sample for expensive metrics
let (X_sample, y_sample) = stratified_sample(&X, &y, 10000, Some(42))?;
let silhouette = silhouette_score(&X_sample, &labels_sample, "euclidean")?;
```

### 2. High-Dimensional Data

```rust
use scirs2_metrics::optimization::dimensionality::pca_reduce;

// Reduce dimensionality for distance-based metrics
let X_reduced = pca_reduce(&X, 50)?; // Reduce to 50 dimensions
let silhouette = silhouette_score(&X_reduced, &labels, "euclidean")?;
```

### 3. Memory Efficiency

```rust
use scirs2_metrics::optimization::memory::chunked_computation;

// Process large arrays in chunks
let mse = chunked_computation::mean_squared_error(&y_true, &y_pred, 10000)?;
```

## Reproducibility

### 1. Set Random Seeds

```rust
use scirs2_metrics::evaluation::set_random_seed;

// Ensure reproducible results
set_random_seed(42);
let cv_scores = cross_val_score(&model, &X, &y, cv, accuracy_score)?;
```

### 2. Document Evaluation Setup

Always document:
- Cross-validation strategy and parameters
- Metrics used and their configurations
- Any preprocessing steps
- Hardware and software versions

### 3. Save Evaluation Results

```rust
use scirs2_metrics::evaluation::EvaluationReport;

let report = EvaluationReport::new()
    .add_metric("accuracy", accuracy)
    .add_metric("f1_score", f1)
    .add_config("cv_folds", 5)
    .add_config("random_seed", 42);

report.save_to_file("evaluation_report.json")?;
```

## Continuous Monitoring

### 1. Track Metrics Over Time

```rust
use scirs2_metrics::monitoring::{MetricTracker, AlertSystem};

let mut tracker = MetricTracker::new();
tracker.log_metric("accuracy", accuracy, timestamp)?;
tracker.log_metric("f1_score", f1, timestamp)?;

// Set up alerts for metric degradation
let alert_system = AlertSystem::new()
    .add_threshold_alert("accuracy", 0.85, "below")
    .add_trend_alert("f1_score", -0.05, "7d");
```

### 2. A/B Testing for Model Updates

```rust
use scirs2_metrics::evaluation::ab_testing::{ABTest, StatisticalPower};

let ab_test = ABTest::new("model_v2_test")
    .with_metric("conversion_rate")
    .with_minimum_effect_size(0.02)
    .with_statistical_power(0.8)
    .with_significance_level(0.05);

let result = ab_test.analyze(&control_results, &treatment_results)?;
```

## Summary

Following these best practices will help ensure that your model evaluation is:

1. **Reliable**: Using appropriate metrics and statistical methods
2. **Comprehensive**: Evaluating multiple aspects of model performance  
3. **Reproducible**: Documenting setup and using consistent procedures
4. **Meaningful**: Choosing metrics that align with business objectives
5. **Robust**: Accounting for uncertainty and avoiding common pitfalls

Remember that evaluation is an iterative process - continuously refine your approach based on the specific requirements of your problem domain.