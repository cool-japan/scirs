# SciRS2 Metrics

[![crates.io](https://img.shields.io/crates/v/scirs2-metrics.svg)](https://crates.io/crates/scirs2-metrics)
[![License](https://img.shields.io/crates/l/scirs2-metrics.svg)](../LICENSE)
[![Documentation](https://img.shields.io/docsrs/scirs2-metrics)](https://docs.rs/scirs2-metrics)

Evaluation metrics module for the SciRS2 scientific computing library. This module provides functions to evaluate prediction performance for classification, regression, and clustering tasks.

## Features

- **Classification Metrics**: Accuracy, precision, recall, F1-score, ROC curves, AUC, etc.
- **Regression Metrics**: MSE, MAE, R2 score, explained variance, etc.
- **Clustering Metrics**: Silhouette score, Calinski-Harabasz index, Davies-Bouldin index, etc.
- **General Evaluation**: Cross-validation, learning curves, confusion matrices

## Installation

Add the following to your `Cargo.toml`:

```toml
[dependencies]
scirs2-metrics = "0.1.0-alpha.1"
```

To enable optimizations through the core module, add feature flags:

```toml
[dependencies]
scirs2-metrics = { version = "0.1.0-alpha.1", features = ["parallel"] }
```

## Usage

Basic usage examples:

```rust
use scirs2_metrics::{classification, regression, clustering};
use ndarray::array;
use scirs2_core::error::CoreResult;

// Classification metrics example
fn classification_metrics_example() -> CoreResult<()> {
    // True labels
    let y_true = array![0, 1, 0, 1, 0, 1, 0, 1];
    
    // Predicted labels
    let y_pred = array![0, 1, 1, 1, 0, 0, 0, 1];
    
    // Calculate classification metrics
    let accuracy = classification::accuracy_score(&y_true, &y_pred)?;
    let precision = classification::precision_score(&y_true, &y_pred, None, None, None)?;
    let recall = classification::recall_score(&y_true, &y_pred, None, None, None)?;
    let f1 = classification::f1_score(&y_true, &y_pred, None, None, None)?;
    
    println!("Accuracy: {}", accuracy);
    println!("Precision: {}", precision);
    println!("Recall: {}", recall);
    println!("F1 Score: {}", f1);
    
    // Predicted probabilities for ROC curve
    let y_scores = array![0.1, 0.9, 0.8, 0.7, 0.2, 0.3, 0.4, 0.8];
    
    // Calculate ROC curve
    let (fpr, tpr, thresholds) = classification::roc_curve(&y_true, &y_scores, None, None)?;
    
    // Calculate Area Under the ROC Curve (AUC)
    let auc = classification::roc_auc_score(&y_true, &y_scores)?;
    println!("AUC: {}", auc);
    
    Ok(())
}

// Regression metrics example
fn regression_metrics_example() -> CoreResult<()> {
    // True values
    let y_true = array![3.0, -0.5, 2.0, 7.0, 2.0];
    
    // Predicted values
    let y_pred = array![2.5, 0.0, 2.1, 7.8, 1.8];
    
    // Calculate regression metrics
    let mse = regression::mean_squared_error(&y_true, &y_pred, None)?;
    let mae = regression::mean_absolute_error(&y_true, &y_pred, None)?;
    let r2 = regression::r2_score(&y_true, &y_pred, None)?;
    let explained_variance = regression::explained_variance_score(&y_true, &y_pred, None)?;
    
    println!("Mean Squared Error: {}", mse);
    println!("Mean Absolute Error: {}", mae);
    println!("R² Score: {}", r2);
    println!("Explained Variance: {}", explained_variance);
    
    Ok(())
}

// Clustering metrics example
fn clustering_metrics_example() -> CoreResult<()> {
    // Sample data points
    let data = array![
        [1.0, 2.0],
        [1.5, 1.8],
        [5.0, 8.0],
        [8.0, 8.0],
        [1.0, 0.6],
        [9.0, 11.0]
    ];
    
    // Cluster labels
    let labels = array![0, 0, 1, 1, 0, 1];
    
    // Calculate clustering metrics
    let silhouette = clustering::silhouette_score(&data, &labels, None, None)?;
    let calinski_harabasz = clustering::calinski_harabasz_score(&data, &labels)?;
    let davies_bouldin = clustering::davies_bouldin_score(&data, &labels)?;
    
    println!("Silhouette Score: {}", silhouette);
    println!("Calinski-Harabasz Index: {}", calinski_harabasz);
    println!("Davies-Bouldin Index: {}", davies_bouldin);
    
    Ok(())
}
```

## Components

### Classification Metrics

Functions for classification evaluation:

```rust
use scirs2_metrics::classification::{
    // Basic Metrics
    accuracy_score,         // Calculate accuracy score
    precision_score,        // Calculate precision score
    recall_score,           // Calculate recall score
    f1_score,               // Calculate F1 score
    fbeta_score,            // Calculate F-beta score
    precision_recall_fscore_support, // Calculate precision, recall, F-score, and support
    
    // Multi-class and Multi-label Metrics
    jaccard_score,          // Calculate Jaccard similarity coefficient
    hamming_loss,           // Calculate Hamming loss
    
    // Probability-based Metrics
    log_loss,               // Calculate logarithmic loss
    brier_score_loss,       // Calculate Brier score loss
    
    // ROC and AUC
    roc_curve,              // Calculate Receiver Operating Characteristic (ROC) curve
    roc_auc_score,          // Calculate Area Under the ROC Curve (AUC)
    average_precision_score, // Calculate average precision score
    precision_recall_curve, // Calculate precision-recall curve
    
    // Confusion Matrix and Derived Metrics
    confusion_matrix,       // Calculate confusion matrix
    classification_report,  // Generate text report of classification metrics
    
    // Probabilities to Labels
    binarize,               // Transform probabilities to binary labels
    label_binarize,         // Transform multi-class labels to binary labels
    
    // Other Metrics
    cohen_kappa_score,      // Calculate Cohen's kappa
    matthews_corrcoef,      // Calculate Matthews correlation coefficient
    hinge_loss,             // Calculate hinge loss
};
```

### Regression Metrics

Functions for regression evaluation:

```rust
use scirs2_metrics::regression::{
    // Error Metrics
    mean_squared_error,     // Calculate mean squared error
    mean_absolute_error,    // Calculate mean absolute error
    mean_absolute_percentage_error, // Calculate mean absolute percentage error
    median_absolute_error,  // Calculate median absolute error
    max_error,              // Calculate maximum error
    
    // Goodness of Fit
    r2_score,               // Calculate R² score (coefficient of determination)
    explained_variance_score, // Calculate explained variance score
    
    // Other Metrics
    mean_squared_log_error, // Calculate mean squared logarithmic error
    mean_poisson_deviance,  // Calculate mean Poisson deviance
    mean_gamma_deviance,    // Calculate mean Gamma deviance
    mean_tweedie_deviance,  // Calculate mean Tweedie deviance
};
```

### Clustering Metrics

Functions for clustering evaluation:

```rust
use scirs2_metrics::clustering::{
    // Internal Metrics (no ground truth)
    silhouette_score,       // Calculate Silhouette Coefficient
    calinski_harabasz_score, // Calculate Calinski-Harabasz Index
    davies_bouldin_score,   // Calculate Davies-Bouldin Index
    
    // External Metrics (with ground truth)
    adjusted_rand_score,    // Calculate Adjusted Rand Index
    normalized_mutual_info_score, // Calculate normalized mutual information
    adjusted_mutual_info_score, // Calculate adjusted mutual information
    fowlkes_mallows_score,  // Calculate Fowlkes-Mallows Index
    
    // Contingency Matrix
    contingency_matrix,     // Calculate contingency matrix
    pair_confusion_matrix,  // Calculate pair confusion matrix
};
```

### Evaluation Functions

General evaluation tools:

```rust
use scirs2_metrics::evaluation::{
    // Cross-validation
    cross_val_score,        // Evaluate a score by cross-validation
    cross_validate,         // Evaluate metrics by cross-validation
    
    // Train/Test Splitting
    train_test_split,       // Split arrays into random train and test subsets
    
    // Learning Curves
    learning_curve,         // Generate learning curve
    validation_curve,       // Generate validation curve
    
    // Hyperparameter Optimization
    grid_search_cv,         // Exhaustive search over parameter grid
    randomized_search_cv,   // Random search over parameters
};
```

## Contributing

See the [CONTRIBUTING.md](../CONTRIBUTING.md) file for contribution guidelines.

## License

This project is licensed under the Apache License, Version 2.0 - see the [LICENSE](../LICENSE) file for details.