# Limitations and Considerations for Metrics

Understanding the limitations and appropriate use cases for each metric is crucial for proper model evaluation. This document outlines key considerations, edge cases, and potential pitfalls for each metric category.

## Classification Metrics

### Accuracy Score

#### Limitations
- **Class Imbalance**: Misleading for imbalanced datasets
- **Cost Insensitivity**: Treats all errors equally
- **Multi-class Aggregation**: May hide poor performance on minority classes

#### When NOT to Use
- Datasets with significant class imbalance (>80% in one class)
- Applications where false positives and false negatives have different costs
- Medical diagnosis, fraud detection, or rare event detection

#### Edge Cases
```rust
// Example: Misleading accuracy on imbalanced data
let y_true = array![0, 0, 0, 0, 0, 0, 0, 0, 0, 1]; // 90% class 0
let y_pred = array![0, 0, 0, 0, 0, 0, 0, 0, 0, 0]; // Predicts all class 0

let accuracy = accuracy_score(&y_true, &y_pred)?; // 0.9 (90%)
// High accuracy but completely misses the minority class!
```

#### Considerations
- Use with balanced datasets or as a secondary metric
- Consider balanced accuracy for imbalanced data
- Always examine per-class performance

### Precision and Recall

#### Precision Limitations
- **Undefined for Zero Predictions**: When no positive predictions are made
- **Sensitivity to Threshold**: Highly dependent on classification threshold
- **Ignores True Negatives**: Doesn't account for correctly rejected negatives

#### Recall Limitations  
- **Undefined for Zero Positives**: When no positive examples exist in ground truth
- **Can Be Trivially High**: By predicting all samples as positive
- **Ignores False Positives**: Doesn't penalize incorrect positive predictions

#### Edge Cases
```rust
// Precision undefined when no positive predictions
let y_true = array![0, 1, 0, 1, 0];
let y_pred = array![0, 0, 0, 0, 0]; // No positive predictions
// precision_score() would return error or 0.0 depending on implementation

// Recall undefined when no positive ground truth
let y_true = array![0, 0, 0, 0, 0]; // No positive examples
let y_pred = array![0, 1, 0, 1, 0];
// recall_score() would return error or undefined
```

#### Considerations
- Always use precision and recall together, never in isolation
- Consider F1-score or F-beta score for balanced evaluation
- Handle edge cases explicitly in your evaluation pipeline

### F1-Score

#### Limitations
- **Equal Weight Assumption**: Assumes precision and recall are equally important
- **Harmonic Mean Sensitivity**: Dominated by the lower of precision or recall
- **Binary Focus**: Standard F1 is designed for binary classification

#### Micro vs Macro vs Weighted F1
- **Micro F1**: Biased toward frequent classes
- **Macro F1**: Treats all classes equally (may not reflect real-world importance)  
- **Weighted F1**: Accounts for class frequency but may mask minority class issues

#### Edge Cases
```rust
// F1-score is sensitive to extreme precision/recall imbalance
let precision = 0.01; // Very low precision
let recall = 0.99;    // Very high recall
// F1 = 2 * (0.01 * 0.99) / (0.01 + 0.99) = 0.0198 (~0.02)
// F1 is dominated by the lower value
```

### ROC AUC

#### Limitations
- **Class Imbalance Insensitivity**: Can be overly optimistic on imbalanced datasets
- **Calibration Independence**: Good AUC doesn't guarantee well-calibrated probabilities
- **Threshold Independence**: Doesn't help with threshold selection

#### When to Avoid
- Highly imbalanced datasets (use PR AUC instead)
- When probability calibration matters
- Cost-sensitive applications

#### Edge Cases
```rust
// ROC AUC can be misleading for imbalanced data
// Consider: 95% negative, 5% positive class
// A model that's slightly better than random on positives
// but good on negatives can achieve high AUC
// while having poor precision
```

#### Considerations
- Use PR AUC for imbalanced datasets
- Examine the actual ROC curve, not just the AUC value
- Consider cost curves for cost-sensitive applications

### Matthews Correlation Coefficient (MCC)

#### Limitations
- **Less Intuitive**: Harder to interpret than other metrics
- **Sensitivity to True Negatives**: Heavily influenced by TN count
- **Binary Focus**: Extension to multi-class is complex

#### Considerations
- Excellent for imbalanced binary classification
- More robust than F1-score for highly imbalanced data
- Consider alongside other metrics for complete picture

## Regression Metrics

### Mean Squared Error (MSE) / Root Mean Squared Error (RMSE)

#### Limitations
- **Outlier Sensitivity**: Heavily penalizes large errors due to squaring
- **Scale Dependence**: Values depend on target variable scale
- **Interpretability**: MSE units are squared, making interpretation difficult

#### Edge Cases
```rust
// MSE is heavily influenced by outliers
let y_true = array![1.0, 2.0, 3.0, 4.0, 100.0]; // One outlier
let y_pred = array![1.1, 2.1, 3.1, 4.1, 4.0];   // Good predictions except outlier

let mse = mean_squared_error(&y_true, &y_pred)?;
// MSE dominated by the outlier: (100-4)² = 9216
// Other errors: (0.1)² + (0.1)² + (0.1)² + (0.1)² = 0.04
// Total MSE ≈ 1843.2, completely dominated by outlier
```

#### Considerations
- Use MAE for outlier-robust evaluation
- Consider Huber loss for mixed scenarios
- Normalize or use relative metrics for scale-independent comparison

### Mean Absolute Error (MAE)

#### Limitations
- **Equal Error Weight**: All errors weighted equally (may not reflect importance)
- **Less Differentiable**: Absolute value is not differentiable at zero
- **Underpenalizes Large Errors**: May be too forgiving of significant mistakes

#### Considerations
- More robust to outliers than MSE
- Better for scenarios where all errors are equally important
- Consider alongside RMSE for complete picture

### R² Score (Coefficient of Determination)

#### Limitations
- **Can Be Negative**: When model performs worse than predicting the mean
- **Not Necessarily Causal**: High R² doesn't imply causation
- **Feature Inflation**: Always increases with more features (use adjusted R²)

#### Edge Cases
```rust
// R² can be negative for very poor models
let y_true = array![1.0, 2.0, 3.0, 4.0, 5.0];
let y_pred = array![5.0, 1.0, 4.0, 2.0, 3.0]; // Worse than predicting mean

let r2 = r2_score(&y_true, &y_pred)?;
// R² can be negative when SS_res > SS_tot
```

#### Interpretation Issues
- R² = 0.7 means 70% of variance explained, not 70% accuracy
- Non-linear relationships may have low R² but strong associations
- Correlation vs. causation confusion

#### Considerations
- Use adjusted R² for model comparison with different feature counts
- Examine residual plots alongside R²
- Consider domain-specific interpretation guidelines

### Mean Absolute Percentage Error (MAPE)

#### Limitations
- **Division by Zero**: Undefined when true values are zero
- **Asymmetric Penalization**: Penalizes overforecasts more than underforecasts
- **Scale Issues**: Problematic for values near zero

#### Edge Cases
```rust
// MAPE undefined for zero values
let y_true = array![0.0, 1.0, 2.0, 3.0]; // Contains zero
let y_pred = array![0.1, 1.1, 2.1, 3.1];
// MAPE calculation fails due to division by zero
```

#### Asymmetry Example
```rust
// MAPE is asymmetric
let true_val = 100.0;
let over_pred = 150.0;  // 50% overforecast
let under_pred = 50.0;  // 50% underforecast

let mape_over = (over_pred - true_val).abs() / true_val * 100.0; // 50%
let mape_under = (under_pred - true_val).abs() / true_val * 100.0; // 50%
// Same MAPE but different implications for business
```

#### Considerations
- Use SMAPE (Symmetric MAPE) for more balanced evaluation
- Handle zero values explicitly
- Consider log-based metrics for multiplicative errors

## Clustering Metrics

### Silhouette Score

#### Limitations
- **Convex Cluster Assumption**: Assumes spherical, convex clusters
- **Density Sensitivity**: Poor for clusters of varying densities
- **Computational Complexity**: O(n²) time complexity

#### Edge Cases
```rust
// Silhouette score poor for non-convex clusters
// Consider two interleaving crescents - low silhouette but valid clusters
// Density-based clustering (DBSCAN) would find them correctly
// but have low silhouette scores
```

#### Considerations
- Best for spherical, well-separated clusters
- Use multiple internal metrics for robustness
- Consider domain-specific validation

### Davies-Bouldin Index

#### Limitations
- **Spherical Assumption**: Assumes clusters are spherical and compact
- **Centroid Dependency**: Based on cluster centroids (sensitive to outliers)
- **Scale Sensitivity**: Affected by feature scaling

#### Considerations
- Lower values indicate better clustering
- Use with proper feature scaling
- Combine with other metrics for comprehensive evaluation

### Calinski-Harabasz Index

#### Limitations
- **Variance-Based**: Assumes clusters have similar sizes and shapes
- **K-Means Bias**: Tends to favor solutions found by K-means
- **Outlier Sensitivity**: Affected by outliers in cluster centers

#### Considerations
- Higher values indicate better clustering
- Good for comparing different numbers of clusters
- Less reliable for clusters of very different sizes

## Cross-Cutting Considerations

### Sample Size Effects

#### Small Samples (n < 100)
- High variance in metric estimates
- Bootstrap confidence intervals recommended
- Stratification crucial for classification

#### Large Samples (n > 100,000)
- Computational efficiency becomes important
- Sampling strategies for expensive metrics
- Parallel computation benefits

### Missing Data

#### Impact on Metrics
```rust
// Missing data can bias metric calculations
// Always document missing data handling strategy
enum MissingDataStrategy {
    CompleteCase,    // Remove samples with missing values
    Imputation,      // Fill missing values
    PartialCredit,   // Adjust denominator for available data
}
```

### Temporal Considerations

#### Time Series Data
- Avoid random sampling (use temporal splits)
- Consider concept drift in evaluation
- Use appropriate cross-validation strategies

#### Model Decay
- Metrics may degrade over time
- Implement monitoring and retraining strategies
- Account for seasonal patterns

### Computational Complexity

#### Time Complexity by Metric
- **O(n)**: Accuracy, Precision, Recall, MSE, MAE
- **O(n log n)**: Median-based metrics (due to sorting)
- **O(n²)**: Silhouette score, some distance-based metrics
- **O(n³)**: Some clustering validity indices

#### Memory Considerations
```rust
// For large datasets, consider streaming computation
use scirs2_metrics::optimization::streaming::StreamingMetrics;

let mut streaming_mse = StreamingMetrics::mse();
for chunk in data_chunks {
    streaming_mse.update(&chunk.y_true, &chunk.y_pred)?;
}
let final_mse = streaming_mse.finalize()?;
```

## Metric Selection Guidelines

### Choose Metrics Based On:
1. **Problem Type**: Classification, regression, clustering
2. **Data Characteristics**: Size, balance, noise level
3. **Business Requirements**: Cost sensitivity, interpretability needs
4. **Computational Constraints**: Time, memory limitations
5. **Stakeholder Expectations**: Familiarity, regulatory requirements

### Avoid Common Anti-Patterns:
- Using accuracy for imbalanced classification
- Relying on single metrics
- Ignoring confidence intervals
- Comparing metrics across different scales
- Using inappropriate averaging strategies
- Neglecting edge case handling

## Debugging Poor Metric Performance

### Systematic Approach:
1. **Verify Data Quality**: Check for leakage, errors, inconsistencies
2. **Examine Class Distributions**: Look for imbalance, missing classes
3. **Visualize Predictions**: Use confusion matrices, residual plots
4. **Check Edge Cases**: Handle missing values, extreme values
5. **Validate Metric Choice**: Ensure metric aligns with objectives
6. **Compare Baselines**: Evaluate against simple baselines

Remember: Metrics are tools for understanding model performance, not ends in themselves. Always interpret them in the context of your specific problem domain and business requirements.