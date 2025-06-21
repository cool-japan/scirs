# Interpretation Examples and Common Pitfalls

This guide provides concrete examples of how to interpret metrics correctly and highlights common mistakes that can lead to wrong conclusions. Understanding these examples will help you avoid pitfalls and make better decisions based on evaluation results.

## Classification Metrics Interpretation

### Accuracy Score Examples

#### Example 1: Misleading High Accuracy
```rust
// Email spam detection dataset
let y_true = array![0, 0, 0, 0, 0, 0, 0, 0, 0, 1]; // 90% non-spam, 10% spam
let y_pred = array![0, 0, 0, 0, 0, 0, 0, 0, 0, 0]; // Predicts all non-spam

let accuracy = accuracy_score(&y_true, &y_pred)?; // 0.9 (90%)
```

**Interpretation:** 
- ✅ **Correct**: "The model correctly classifies 90% of emails"
- ❌ **Wrong**: "This is a good spam detector"
- 🚨 **Pitfall**: High accuracy masks complete failure on minority class

**Why it's misleading:** The model never detects spam (0% recall), making it useless for the intended purpose despite high accuracy.

#### Example 2: Context-Dependent Accuracy
```rust
// Medical diagnosis: Cancer screening (1% prevalence)
let y_true = array![0; 99].chain(array![1; 1]).collect(); // 99% healthy, 1% cancer
let y_pred = array![0; 100].collect(); // Predicts all healthy

let accuracy = accuracy_score(&y_true, &y_pred)?; // 0.99 (99%)
```

**Interpretation:**
- ✅ **Correct**: "99% of diagnoses are correct"
- ❌ **Wrong**: "This diagnostic tool is 99% effective"
- 🚨 **Pitfall**: Missing all cancer cases is catastrophic regardless of accuracy

### Precision and Recall Trade-offs

#### Example 3: High Precision, Low Recall
```rust
// Fraud detection system
let y_true = array![0, 0, 0, 0, 1, 1, 1, 1, 1, 1]; // 40% fraud cases
let y_pred = array![0, 0, 0, 0, 0, 0, 0, 1, 1, 1]; // Conservative predictions

let precision = precision_score(&y_true, &y_pred, 1.0)?; // 1.0 (100%)
let recall = recall_score(&y_true, &y_pred, 1.0)?;       // 0.5 (50%)
let f1 = f1_score(&y_true, &y_pred, 1.0)?;               // 0.67
```

**Interpretation:**
- ✅ **Correct**: "When the model flags fraud, it's always right (100% precision), but it only catches half of actual fraud cases (50% recall)"
- ❌ **Wrong**: "The model is 100% accurate at fraud detection"
- 📊 **Business Impact**: Low false alarms but missing many fraud cases

#### Example 4: High Recall, Low Precision
```rust
// Email spam detection - aggressive filtering
let y_true = array![0, 0, 0, 0, 0, 0, 1, 1, 1, 1]; // 40% spam
let y_pred = array![1, 1, 1, 1, 0, 0, 1, 1, 1, 1]; // Aggressive filtering

let precision = precision_score(&y_true, &y_pred, 1.0)?; // 0.5 (50%)
let recall = recall_score(&y_true, &y_pred, 1.0)?;       // 1.0 (100%)
let f1 = f1_score(&y_true, &y_pred, 1.0)?;               // 0.67
```

**Interpretation:**
- ✅ **Correct**: "The filter catches all spam (100% recall) but half of flagged emails are actually legitimate (50% precision)"
- ❌ **Wrong**: "The spam filter is 100% effective"
- 📊 **Business Impact**: No spam in inbox but many legitimate emails filtered

### ROC AUC Interpretation

#### Example 5: AUC on Imbalanced Data
```rust
// Rare disease detection (5% prevalence)
let y_true = array![0; 95].chain(array![1; 5]).collect();
let y_scores = array![0.1; 95].chain(array![0.6; 5]).collect(); // Modest separation

let auc_roc = roc_auc_score(&y_true, &y_scores)?; // ~0.85
let auc_pr = average_precision_score(&y_true, &y_scores)?; // ~0.3
```

**Interpretation:**
- ✅ **Correct**: "The model has good discriminative ability (AUC-ROC=0.85) but poor precision for positive predictions (AUC-PR=0.3)"
- ❌ **Wrong**: "85% AUC means the model is 85% accurate"
- 🚨 **Pitfall**: ROC AUC can be misleadingly optimistic for imbalanced data

### Matthews Correlation Coefficient

#### Example 6: MCC vs F1-Score Comparison
```rust
// Scenario A: Balanced performance
let y_true_a = array![0, 0, 1, 1, 0, 0, 1, 1];
let y_pred_a = array![0, 1, 0, 1, 0, 1, 0, 1]; // Poor but balanced errors

// Scenario B: Biased performance  
let y_true_b = array![0, 0, 1, 1, 0, 0, 1, 1];
let y_pred_b = array![0, 0, 0, 1, 0, 0, 0, 1]; // Biased toward negative

let mcc_a = matthews_corrcoef(&y_true_a, &y_pred_a)?; // 0.0
let f1_a = f1_score(&y_true_a, &y_pred_a, 1.0)?;      // 0.5

let mcc_b = matthews_corrcoef(&y_true_b, &y_pred_b)?; // 0.33
let f1_b = f1_score(&y_true_b, &y_pred_b, 1.0)?;      // 0.5
```

**Interpretation:**
- ✅ **Correct**: "Both models have the same F1-score (0.5), but model B shows better overall correlation (MCC=0.33 vs 0.0)"
- 📊 **Insight**: MCC accounts for true negatives, providing a more balanced view

## Regression Metrics Interpretation

### R² Score Examples

#### Example 7: Negative R² Score
```rust
// House price prediction
let y_true = array![100.0, 200.0, 300.0, 400.0, 500.0]; // $100K to $500K
let y_pred = array![500.0, 100.0, 400.0, 200.0, 300.0]; // Random predictions

let r2 = r2_score(&y_true, &y_pred)?; // -1.5 (negative!)
let mae = mean_absolute_error(&y_true, &y_pred)?; // $160K average error
```

**Interpretation:**
- ✅ **Correct**: "The model performs worse than simply predicting the mean price (R²=-1.5)"
- ❌ **Wrong**: "R² should always be between 0 and 1"
- 📊 **Insight**: Negative R² indicates the model is worse than a constant predictor

#### Example 8: High R² with Poor Practical Performance
```rust
// Stock price prediction
let y_true = array![100.0, 101.0, 102.0, 103.0, 104.0]; // Gradual increase
let y_pred = array![100.5, 101.5, 102.5, 103.5, 104.5]; // Consistent +0.5 bias

let r2 = r2_score(&y_true, &y_pred)?;   // 0.995 (very high!)
let mae = mean_absolute_error(&y_true, &y_pred)?; // 0.5
```

**Interpretation:**
- ✅ **Correct**: "The model explains 99.5% of price variance but has a consistent bias"
- ❌ **Wrong**: "99.5% R² means excellent predictions"
- 📊 **Insight**: High R² doesn't guarantee practical utility if bias matters

### Mean Absolute Percentage Error (MAPE)

#### Example 9: MAPE Asymmetry
```rust
// Sales forecasting
let true_sales = array![100.0, 100.0]; 
let over_forecast = array![150.0, 100.0];  // 50% overestimate
let under_forecast = array![50.0, 100.0];  // 50% underestimate

let mape_over = mean_absolute_percentage_error(&true_sales, &over_forecast)?; // 25%
let mape_under = mean_absolute_percentage_error(&true_sales, &under_forecast)?; // 25%
```

**Interpretation:**
- ✅ **Correct**: "Both forecasts have 25% MAPE, but overforecasting by 50% has different business impact than underforecasting by 50%"
- 🚨 **Pitfall**: MAPE treats over- and under-forecasting asymmetrically
- 💡 **Solution**: Use SMAPE or separate over/under error analysis

### Outlier Effects

#### Example 10: Outlier Impact on Different Metrics
```rust
// Website traffic prediction (daily visitors)
let y_true = array![1000.0, 1100.0, 1050.0, 1200.0, 50000.0]; // One viral day
let y_pred = array![1050.0, 1150.0, 1100.0, 1250.0, 1500.0];  // Missed viral day

let mse = mean_squared_error(&y_true, &y_pred)?;    // ~470M (dominated by outlier)
let mae = mean_absolute_error(&y_true, &y_pred)?;   // ~9,730
let rmse = root_mean_squared_error(&y_true, &y_pred)?; // ~21,679
```

**Interpretation:**
- ✅ **Correct**: "The model performs well on typical days (MAE ≈ 50-70 for first 4 days) but fails catastrophically on outliers"
- 📊 **Insight**: MSE/RMSE heavily penalize the viral day miss, MAE provides clearer typical performance

## Clustering Metrics Interpretation

### Silhouette Score Examples

#### Example 11: Misleading Silhouette Scores
```rust
// Customer segmentation data
// Scenario A: Well-separated spherical clusters
let silhouette_a = 0.8; // High score

// Scenario B: Complex non-convex clusters (e.g., concentric circles)
let silhouette_b = 0.2; // Low score despite valid structure
```

**Interpretation:**
- ✅ **Correct**: "Scenario A has well-separated, compact clusters. Scenario B may have valid but complex cluster shapes"
- ❌ **Wrong**: "Scenario A clustering is always better than Scenario B"
- 📊 **Insight**: Silhouette score assumes convex clusters; low scores don't always mean bad clustering

### Internal vs External Metrics

#### Example 12: Conflicting Clustering Evaluations
```rust
// Document clustering results
let internal_metrics = ClusteringMetrics {
    silhouette: 0.3,     // Low internal quality
    davies_bouldin: 1.8, // Poor separation
    calinski_harabasz: 45.2
};

let external_metrics = ClusteringMetrics {
    adjusted_rand_index: 0.75,  // Good agreement with ground truth
    normalized_mutual_info: 0.68 // Good information recovery
};
```

**Interpretation:**
- ✅ **Correct**: "The clustering recovers meaningful structure (high ARI/NMI) but doesn't form compact, well-separated clusters (low internal metrics)"
- 📊 **Insight**: Internal and external metrics can disagree; prioritize based on whether ground truth is available and reliable

## Common Pitfalls and How to Avoid Them

### Pitfall 1: Single Metric Obsession

❌ **Wrong Approach:**
```rust
let accuracy = accuracy_score(&y_true, &y_pred)?;
if accuracy > 0.95 {
    println!("Excellent model!");
}
```

✅ **Correct Approach:**
```rust
let accuracy = accuracy_score(&y_true, &y_pred)?;
let f1 = f1_score(&y_true, &y_pred, 1.0)?;
let precision = precision_score(&y_true, &y_pred, 1.0)?;
let recall = recall_score(&y_true, &y_pred, 1.0)?;

println!("Model Performance:");
println!("Accuracy: {:.3}", accuracy);
println!("F1-Score: {:.3}", f1);
println!("Precision: {:.3}", precision);
println!("Recall: {:.3}", recall);

// Check for red flags
if accuracy > 0.95 && (precision < 0.5 || recall < 0.5) {
    println!("Warning: High accuracy but poor precision/recall suggests class imbalance issues");
}
```

### Pitfall 2: Ignoring Base Rates

❌ **Wrong Interpretation:**
```rust
// Cancer screening test
let sensitivity = 0.95; // 95% of cancers detected
let specificity = 0.95; // 95% of healthy correctly identified
// "This test is 95% accurate!"
```

✅ **Correct Analysis:**
```rust
// Consider base rate (cancer prevalence = 1%)
let prevalence = 0.01;
let sensitivity = 0.95;
let specificity = 0.95;

// Calculate positive predictive value (precision)
let ppv = (sensitivity * prevalence) / 
          (sensitivity * prevalence + (1.0 - specificity) * (1.0 - prevalence));

println!("Positive Predictive Value: {:.1}%", ppv * 100.0); // Only ~16%!
println!("Out of 100 positive tests, only ~16 actually have cancer");
```

### Pitfall 3: Data Leakage in Cross-Validation

❌ **Wrong Approach:**
```rust
// Normalize entire dataset first
let normalized_X = normalize(&X);
let cv_scores = cross_val_score(&model, &normalized_X, &y, cv, accuracy_score)?;
```

✅ **Correct Approach:**
```rust
// Normalize within each CV fold
let cv_scores = cross_validate_with_preprocessing(&model, &X, &y, cv, 
    |X_train, X_test| {
        let scaler = StandardScaler::fit(&X_train)?;
        let X_train_norm = scaler.transform(&X_train)?;
        let X_test_norm = scaler.transform(&X_test)?;
        Ok((X_train_norm, X_test_norm))
    },
    accuracy_score
)?;
```

### Pitfall 4: Threshold Dependency

❌ **Wrong Approach:**
```rust
// Using default threshold without justification
let y_pred: Array1<f64> = y_scores.mapv(|x| if x > 0.5 { 1.0 } else { 0.0 });
let f1 = f1_score(&y_true, &y_pred, 1.0)?;
```

✅ **Correct Approach:**
```rust
use scirs2_metrics::classification::threshold::find_optimal_threshold;

// Find optimal threshold for F1-score
let optimal_threshold = find_optimal_threshold(&y_true, &y_scores, f1_score)?;
let y_pred_optimal: Array1<f64> = y_scores.mapv(|x| if x > optimal_threshold { 1.0 } else { 0.0 });

// Report both threshold-independent and threshold-dependent metrics
let auc = roc_auc_score(&y_true, &y_scores)?;
let f1_default = f1_score(&y_true, &y_pred_default, 1.0)?;
let f1_optimal = f1_score(&y_true, &y_pred_optimal, 1.0)?;

println!("AUC (threshold-independent): {:.3}", auc);
println!("F1 at default threshold (0.5): {:.3}", f1_default);
println!("F1 at optimal threshold ({:.3}): {:.3}", optimal_threshold, f1_optimal);
```

### Pitfall 5: Overfitting to Validation Metrics

❌ **Wrong Approach:**
```rust
// Repeatedly tuning on validation set
for threshold in (0.1..0.9).step_by(0.01) {
    let y_pred: Array1<f64> = y_scores.mapv(|x| if x > threshold { 1.0 } else { 0.0 });
    let f1 = f1_score(&y_val_true, &y_pred, 1.0)?;
    if f1 > best_f1 {
        best_threshold = threshold;
        best_f1 = f1;
    }
}
// Using best_threshold for final evaluation
```

✅ **Correct Approach:**
```rust
// Proper train/val/test split
let (X_train, X_temp, y_train, y_temp) = train_test_split(&X, &y, 0.6, Some(42))?;
let (X_val, X_test, y_val, y_test) = train_test_split(&X_temp, &y_temp, 0.5, Some(42))?;

// Tune on validation set
let optimal_threshold = find_optimal_threshold(&y_val, &y_val_scores, f1_score)?;

// Final evaluation on test set (only once!)
let y_test_pred: Array1<f64> = y_test_scores.mapv(|x| if x > optimal_threshold { 1.0 } else { 0.0 });
let final_f1 = f1_score(&y_test, &y_test_pred, 1.0)?;

println!("Final F1 score (unbiased): {:.3}", final_f1);
```

### Pitfall 6: Inappropriate Averaging for Multi-class

❌ **Wrong Interpretation:**
```rust
// Class distribution: Class 0: 80%, Class 1: 15%, Class 2: 5%
let macro_f1 = 0.6;  // Equal weight to all classes
let micro_f1 = 0.75; // Weighted by frequency
println!("Model performance: {:.1}%", macro_f1 * 100.0); // Focuses on rare classes
```

✅ **Correct Interpretation:**
```rust
let macro_f1 = 0.6;   // Performance on rare classes
let micro_f1 = 0.75;  // Overall sample-weighted performance
let weighted_f1 = 0.72; // Frequency-weighted performance

println!("Performance Summary:");
println!("- Macro F1: {:.3} (equal importance to all classes)", macro_f1);
println!("- Micro F1: {:.3} (overall sample performance)", micro_f1);
println!("- Weighted F1: {:.3} (frequency-weighted performance)", weighted_f1);

if macro_f1 < micro_f1 {
    println!("Note: Model performs worse on minority classes");
}
```

## Interpretation Guidelines

### 1. Always Provide Context
```rust
fn report_with_context(metric_value: f64, metric_name: &str, 
                       dataset_characteristics: &str) {
    println!("{}: {:.3}", metric_name, metric_value);
    println!("Context: {}", dataset_characteristics);
    println!("Baseline comparison needed");
}

// Example usage
report_with_context(0.85, "F1-Score", 
    "Imbalanced dataset (5% positive class), medical diagnosis task");
```

### 2. Compare Against Meaningful Baselines
```rust
// Always include baseline comparisons
let dummy_classifier_accuracy = majority_class_percentage; // e.g., 0.95 for 95% majority
let random_classifier_accuracy = 0.5; // For balanced binary
let model_accuracy = accuracy_score(&y_true, &y_pred)?;

println!("Model accuracy: {:.3}", model_accuracy);
println!("Dummy classifier: {:.3}", dummy_classifier_accuracy);
println!("Random classifier: {:.3}", random_classifier_accuracy);

if model_accuracy <= dummy_classifier_accuracy {
    println!("Warning: Model doesn't beat dummy classifier!");
}
```

### 3. Include Confidence Intervals
```rust
use scirs2_metrics::evaluation::bootstrap_confidence_interval;

let cv_scores = cross_val_score(&model, &X, &y, cv, f1_score)?;
let mean_score = cv_scores.mean().unwrap();
let (ci_lower, ci_upper) = bootstrap_confidence_interval(&cv_scores, 0.95, 1000)?;

println!("F1-Score: {:.3} (95% CI: [{:.3}, {:.3}])", mean_score, ci_lower, ci_upper);
```

### 4. Document Assumptions and Limitations
```rust
fn comprehensive_evaluation_report(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<()> {
    println!("=== Model Evaluation Report ===");
    
    // Basic metrics
    let accuracy = accuracy_score(y_true, y_pred)?;
    let f1 = f1_score(y_true, y_pred, 1.0)?;
    
    println!("Metrics:");
    println!("- Accuracy: {:.3}", accuracy);
    println!("- F1-Score: {:.3}", f1);
    
    // Assumptions and limitations
    println!("\nAssumptions:");
    println!("- Threshold set to 0.5");
    println!("- Equal misclassification costs assumed");
    println!("- Test set representative of deployment data");
    
    println!("\nLimitations:");
    println!("- Single test set evaluation");
    println!("- No confidence intervals provided");
    println!("- Temporal stability not assessed");
    
    Ok(())
}
```

## Summary

Proper metric interpretation requires:

1. **Understanding what each metric measures and doesn't measure**
2. **Considering data characteristics (balance, size, noise)**
3. **Providing appropriate context and baselines**
4. **Using multiple complementary metrics**
5. **Accounting for uncertainty and confidence intervals**
6. **Avoiding common pitfalls and anti-patterns**
7. **Documenting assumptions and limitations clearly**

Remember: Metrics are tools for understanding model performance, not absolute measures of model quality. Always interpret them in the context of your specific problem domain and business requirements.