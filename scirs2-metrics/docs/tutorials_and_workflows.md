# Interactive Tutorials and Workflow Examples

This document provides comprehensive, step-by-step tutorials and workflows for common evaluation scenarios. Each tutorial includes complete code examples, expected outputs, and decision points to guide you through real-world evaluation tasks.

## Tutorial 1: Complete Binary Classification Evaluation

### Scenario
You're building a fraud detection system for credit card transactions. You have a dataset with 10,000 transactions where 5% are fraudulent.

### Step 1: Data Setup and Initial Assessment

```rust
use ndarray::{Array1, Array2, array};
use scirs2_metrics::classification::{accuracy_score, precision_score, recall_score, f1_score};
use scirs2_metrics::classification::curves::{roc_curve, roc_auc_score, average_precision_score};
use scirs2_metrics::classification::advanced::{matthews_corrcoef, balanced_accuracy_score};
use scirs2_metrics::evaluation::{train_test_split, StratifiedKFold};

// Simulate fraud detection data
fn generate_fraud_data() -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    // y_true: 0 = legitimate, 1 = fraud (5% fraud rate)
    let mut y_true = vec![0.0; 9500];
    y_true.extend(vec![1.0; 500]);
    let y_true = Array1::from_vec(y_true);
    
    // Simulate model scores (higher = more likely fraud)
    let mut y_scores = Vec::new();
    for &label in y_true.iter() {
        if label == 1.0 {
            // Fraud cases: higher scores with some noise
            y_scores.push(0.7 + 0.2 * random_normal() + 0.1 * random_uniform());
        } else {
            // Legitimate cases: lower scores with some noise
            y_scores.push(0.3 + 0.2 * random_normal() + 0.1 * random_uniform());
        }
    }
    let y_scores = Array1::from_vec(y_scores);
    
    // Convert scores to binary predictions using 0.5 threshold
    let y_pred = y_scores.mapv(|x| if x > 0.5 { 1.0 } else { 0.0 });
    
    (y_true, y_pred, y_scores)
}

// Step 1: Initial data assessment
fn step1_data_assessment(y_true: &Array1<f64>) -> Result<()> {
    println!("=== Step 1: Data Assessment ===");
    
    let total_samples = y_true.len();
    let fraud_count = y_true.iter().filter(|&&x| x == 1.0).count();
    let fraud_rate = fraud_count as f64 / total_samples as f64;
    
    println!("Total samples: {}", total_samples);
    println!("Fraud cases: {} ({:.1}%)", fraud_count, fraud_rate * 100.0);
    println!("Legitimate cases: {} ({:.1}%)", 
             total_samples - fraud_count, (1.0 - fraud_rate) * 100.0);
    
    // Decision point: Is this imbalanced?
    if fraud_rate < 0.1 || fraud_rate > 0.9 {
        println!("⚠️  IMBALANCED DATASET DETECTED");
        println!("   → Accuracy will be misleading");
        println!("   → Focus on Precision, Recall, F1, and AUC-PR");
        println!("   → Consider cost-sensitive evaluation");
    }
    
    Ok(())
}
```

### Step 2: Baseline Evaluation

```rust
fn step2_baseline_evaluation(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<()> {
    println!("\n=== Step 2: Baseline Evaluation ===");
    
    // Calculate basic metrics
    let accuracy = accuracy_score(y_true, y_pred)?;
    let precision = precision_score(y_true, y_pred, 1.0)?;
    let recall = recall_score(y_true, y_pred, 1.0)?;
    let f1 = f1_score(y_true, y_pred, 1.0)?;
    
    println!("Basic Metrics:");
    println!("  Accuracy:  {:.3}", accuracy);
    println!("  Precision: {:.3}", precision);
    println!("  Recall:    {:.3}", recall);
    println!("  F1-Score:  {:.3}", f1);
    
    // Baseline comparisons
    let majority_class_rate = y_true.iter().filter(|&&x| x == 0.0).count() as f64 / y_true.len() as f64;
    println!("\nBaseline Comparisons:");
    println!("  Always predict majority class: {:.3}", majority_class_rate);
    println!("  Random classifier: ~0.500");
    
    // Interpretation guidance
    if accuracy > majority_class_rate {
        println!("✓ Model beats majority class baseline");
    } else {
        println!("❌ Model doesn't beat majority class baseline!");
    }
    
    if f1 > 0.1 {
        println!("✓ Reasonable F1-score for imbalanced data");
    } else {
        println!("❌ Very low F1-score - check model and features");
    }
    
    Ok(())
}
```

### Step 3: Threshold-Independent Analysis

```rust
fn step3_threshold_independent_analysis(y_true: &Array1<f64>, y_scores: &Array1<f64>) -> Result<()> {
    println!("\n=== Step 3: Threshold-Independent Analysis ===");
    
    // ROC Analysis
    let (fpr, tpr, thresholds) = roc_curve(y_true, y_scores)?;
    let auc_roc = roc_auc_score(y_true, y_scores)?;
    
    // Precision-Recall Analysis
    let (precision_curve, recall_curve, pr_thresholds) = precision_recall_curve(y_true, y_scores)?;
    let auc_pr = average_precision_score(y_true, y_scores)?;
    
    println!("Threshold-Independent Metrics:");
    println!("  AUC-ROC: {:.3}", auc_roc);
    println!("  AUC-PR:  {:.3}", auc_pr);
    
    // Interpretation for imbalanced data
    println!("\nInterpretation for Fraud Detection:");
    if auc_roc > 0.9 {
        println!("  ✓ Excellent discriminative ability (AUC-ROC > 0.9)");
    } else if auc_roc > 0.8 {
        println!("  ✓ Good discriminative ability (AUC-ROC > 0.8)");
    } else if auc_roc > 0.7 {
        println!("  ⚠️ Fair discriminative ability (AUC-ROC > 0.7)");
    } else {
        println!("  ❌ Poor discriminative ability (AUC-ROC ≤ 0.7)");
    }
    
    if auc_pr > 0.5 {
        println!("  ✓ Good precision-recall trade-off (AUC-PR > 0.5)");
    } else if auc_pr > 0.2 {
        println!("  ⚠️ Moderate precision-recall trade-off (AUC-PR > 0.2)");
    } else {
        println!("  ❌ Poor precision-recall trade-off (AUC-PR ≤ 0.2)");
    }
    
    // For imbalanced data, AUC-PR is more informative
    if auc_roc - auc_pr > 0.3 {
        println!("  ⚠️ Large gap between AUC-ROC and AUC-PR suggests optimistic ROC due to imbalance");
    }
    
    Ok(())
}
```

### Step 4: Threshold Optimization

```rust
use scirs2_metrics::classification::threshold::{find_optimal_threshold, precision_recall_curve};

fn step4_threshold_optimization(y_true: &Array1<f64>, y_scores: &Array1<f64>) -> Result<()> {
    println!("\n=== Step 4: Threshold Optimization ===");
    
    // Find optimal thresholds for different objectives
    let f1_threshold = find_optimal_threshold(y_true, y_scores, 
        |yt, yp| f1_score(yt, yp, 1.0))?;
    
    // Custom precision threshold (targeting 80% precision)
    let precision_80_threshold = find_threshold_for_precision(y_true, y_scores, 0.8)?;
    
    // Custom recall threshold (targeting 90% recall)
    let recall_90_threshold = find_threshold_for_recall(y_true, y_scores, 0.9)?;
    
    println!("Optimal Thresholds:");
    println!("  F1-Score maximizing: {:.3}", f1_threshold);
    println!("  80% Precision:       {:.3}", precision_80_threshold);
    println!("  90% Recall:          {:.3}", recall_90_threshold);
    
    // Evaluate at different thresholds
    let thresholds = vec![0.3, 0.5, f1_threshold, 0.7, 0.9];
    
    println!("\nPerformance at Different Thresholds:");
    println!("Threshold | Precision | Recall | F1-Score | Fraud Detected");
    println!("----------|-----------|--------|----------|---------------");
    
    for &threshold in &thresholds {
        let y_pred_thresh = y_scores.mapv(|x| if x > threshold { 1.0 } else { 0.0 });
        let precision = precision_score(y_true, &y_pred_thresh, 1.0)?;
        let recall = recall_score(y_true, &y_pred_thresh, 1.0)?;
        let f1 = f1_score(y_true, &y_pred_thresh, 1.0)?;
        let detected = y_pred_thresh.sum() as usize;
        
        println!("  {:.3}   |   {:.3}   |  {:.3} |  {:.3}  |     {}",
                 threshold, precision, recall, f1, detected);
    }
    
    // Business guidance
    println!("\nBusiness Considerations:");
    println!("  • Higher threshold → Higher precision, lower recall");
    println!("  • Lower threshold → Lower precision, higher recall");
    println!("  • Consider cost of false positives vs false negatives");
    
    Ok(())
}

// Helper functions for custom thresholds
fn find_threshold_for_precision(y_true: &Array1<f64>, y_scores: &Array1<f64>, 
                               target_precision: f64) -> Result<f64> {
    let (precision_vals, _, thresholds) = precision_recall_curve(y_true, y_scores)?;
    
    for (i, &precision) in precision_vals.iter().enumerate() {
        if precision >= target_precision {
            return Ok(thresholds[i]);
        }
    }
    
    Ok(thresholds[thresholds.len() - 1]) // Return highest threshold if target not met
}

fn find_threshold_for_recall(y_true: &Array1<f64>, y_scores: &Array1<f64>, 
                            target_recall: f64) -> Result<f64> {
    let (_, recall_vals, thresholds) = precision_recall_curve(y_true, y_scores)?;
    
    for (i, &recall) in recall_vals.iter().enumerate() {
        if recall >= target_recall {
            return Ok(thresholds[i]);
        }
    }
    
    Ok(thresholds[0]) // Return lowest threshold if target not met
}
```

### Step 5: Advanced Metrics and Robustness

```rust
fn step5_advanced_metrics(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<()> {
    println!("\n=== Step 5: Advanced Metrics ===");
    
    // Advanced single metrics
    let mcc = matthews_corrcoef(y_true, y_pred)?;
    let balanced_acc = balanced_accuracy_score(y_true, y_pred)?;
    
    println!("Advanced Metrics:");
    println!("  Matthews Correlation Coefficient: {:.3}", mcc);
    println!("  Balanced Accuracy:               {:.3}", balanced_acc);
    
    // Confusion matrix analysis
    let (cm, _) = confusion_matrix(y_true, y_pred, None)?;
    let tn = cm[[0, 0]] as f64;
    let fp = cm[[0, 1]] as f64;
    let fn_val = cm[[1, 0]] as f64;
    let tp = cm[[1, 1]] as f64;
    
    println!("\nConfusion Matrix Analysis:");
    println!("                 Predicted");
    println!("              Legit  Fraud");
    println!("Actual Legit  {:5}  {:5}", tn as usize, fp as usize);
    println!("       Fraud  {:5}  {:5}", fn_val as usize, tp as usize);
    
    // Business impact metrics
    let false_positive_rate = fp / (fp + tn);
    let false_negative_rate = fn_val / (fn_val + tp);
    
    println!("\nBusiness Impact:");
    println!("  False Positive Rate: {:.3} ({:.1}% legit flagged as fraud)", 
             false_positive_rate, false_positive_rate * 100.0);
    println!("  False Negative Rate: {:.3} ({:.1}% fraud missed)", 
             false_negative_rate, false_negative_rate * 100.0);
    
    // Cost analysis (example costs)
    let cost_fp = 10.0;   // Cost of investigating false positive
    let cost_fn = 100.0;  // Cost of missing fraud
    let total_cost = fp * cost_fp + fn_val * cost_fn;
    
    println!("\nCost Analysis (example):");
    println!("  Cost per false positive: ${:.0}", cost_fp);
    println!("  Cost per false negative: ${:.0}", cost_fn);
    println!("  Total cost: ${:.0}", total_cost);
    
    Ok(())
}
```

### Step 6: Cross-Validation and Confidence Intervals

```rust
use scirs2_metrics::evaluation::{cross_val_score, bootstrap_confidence_interval};

fn step6_robust_evaluation(X: &Array2<f64>, y: &Array1<f64>) -> Result<()> {
    println!("\n=== Step 6: Robust Evaluation ===");
    
    // Stratified cross-validation
    let cv = StratifiedKFold::new(5, true, Some(42));
    
    // Evaluate multiple metrics with CV
    let f1_scores = cross_val_score(&model, X, y, cv, 
        |yt, yp| f1_score(yt, yp, 1.0))?;
    let precision_scores = cross_val_score(&model, X, y, cv, 
        |yt, yp| precision_score(yt, yp, 1.0))?;
    let recall_scores = cross_val_score(&model, X, y, cv, 
        |yt, yp| recall_score(yt, yp, 1.0))?;
    
    // Calculate confidence intervals
    let (f1_ci_low, f1_ci_high) = bootstrap_confidence_interval(&f1_scores, 0.95, 1000)?;
    let (prec_ci_low, prec_ci_high) = bootstrap_confidence_interval(&precision_scores, 0.95, 1000)?;
    let (rec_ci_low, rec_ci_high) = bootstrap_confidence_interval(&recall_scores, 0.95, 1000)?;
    
    println!("Cross-Validation Results (5-fold, stratified):");
    println!("Metric     | Mean  | Std   | 95% CI");
    println!("-----------|-------|-------|------------------");
    println!("F1-Score   | {:.3} | {:.3} | [{:.3}, {:.3}]", 
             f1_scores.mean().unwrap(), f1_scores.std(1.0), f1_ci_low, f1_ci_high);
    println!("Precision  | {:.3} | {:.3} | [{:.3}, {:.3}]", 
             precision_scores.mean().unwrap(), precision_scores.std(1.0), prec_ci_low, prec_ci_high);
    println!("Recall     | {:.3} | {:.3} | [{:.3}, {:.3}]", 
             recall_scores.mean().unwrap(), recall_scores.std(1.0), rec_ci_low, rec_ci_high);
    
    // Stability analysis
    let f1_cv = f1_scores.std(1.0) / f1_scores.mean().unwrap();
    if f1_cv < 0.1 {
        println!("✓ Stable performance across folds (CV < 10%)");
    } else if f1_cv < 0.2 {
        println!("⚠️ Moderate variance across folds (CV 10-20%)");
    } else {
        println!("❌ High variance across folds (CV > 20%)");
    }
    
    Ok(())
}
```

### Step 7: Final Recommendations

```rust
fn step7_final_recommendations(evaluation_results: &EvaluationResults) -> Result<()> {
    println!("\n=== Step 7: Final Recommendations ===");
    
    println!("Model Assessment:");
    
    // Overall performance
    if evaluation_results.auc_pr > 0.5 && evaluation_results.f1_mean > 0.3 {
        println!("✓ Model shows promise for fraud detection");
    } else {
        println!("❌ Model performance insufficient for production");
    }
    
    // Stability
    if evaluation_results.f1_cv < 0.15 {
        println!("✓ Model performance is stable across validation folds");
    } else {
        println!("⚠️ Model performance varies significantly - investigate further");
    }
    
    // Business readiness
    println!("\nProduction Readiness Checklist:");
    println!("□ Performance metrics meet business requirements");
    println!("□ Model stable across cross-validation folds");
    println!("□ Optimal threshold determined based on business costs");
    println!("□ Monitoring plan for model degradation");
    println!("□ Interpretability requirements addressed");
    println!("□ Bias and fairness evaluation completed");
    
    println!("\nNext Steps:");
    println!("1. Conduct bias and fairness analysis");
    println!("2. Set up model monitoring with key metrics");
    println!("3. Design A/B test for production deployment");
    println!("4. Create model documentation and decision boundaries");
    
    Ok(())
}

struct EvaluationResults {
    auc_pr: f64,
    f1_mean: f64,
    f1_cv: f64,
}
```

### Complete Tutorial Runner

```rust
pub fn tutorial_1_fraud_detection() -> Result<()> {
    println!("🎯 Tutorial 1: Complete Binary Classification Evaluation");
    println!("📊 Scenario: Credit Card Fraud Detection\n");
    
    // Generate example data
    let (y_true, y_pred, y_scores) = generate_fraud_data();
    
    // Run all steps
    step1_data_assessment(&y_true)?;
    step2_baseline_evaluation(&y_true, &y_pred)?;
    step3_threshold_independent_analysis(&y_true, &y_scores)?;
    step4_threshold_optimization(&y_true, &y_scores)?;
    step5_advanced_metrics(&y_true, &y_pred)?;
    
    // For steps requiring features, you'd need actual feature data
    // step6_robust_evaluation(&X, &y_true)?;
    
    let results = EvaluationResults {
        auc_pr: 0.65, // Example values - would be calculated from actual evaluation
        f1_mean: 0.45,
        f1_cv: 0.12,
    };
    step7_final_recommendations(&results)?;
    
    println!("\n🎉 Tutorial completed! You now have a comprehensive evaluation framework.");
    
    Ok(())
}
```

## Tutorial 2: Multi-class Classification Workflow

### Scenario
Building an image classification system for medical scans with 4 classes: Normal, Pneumonia, COVID-19, and Other.

```rust
pub fn tutorial_2_multiclass_medical() -> Result<()> {
    println!("🎯 Tutorial 2: Multi-class Medical Image Classification");
    println!("📊 Scenario: Medical Scan Classification (4 classes)\n");
    
    // Step 1: Class distribution analysis
    let class_names = vec!["Normal", "Pneumonia", "COVID-19", "Other"];
    let class_counts = vec![800, 600, 150, 200]; // Imbalanced
    let total = class_counts.iter().sum::<usize>();
    
    println!("=== Class Distribution Analysis ===");
    for (i, (name, count)) in class_names.iter().zip(class_counts.iter()).enumerate() {
        let percentage = *count as f64 / total as f64 * 100.0;
        println!("Class {}: {} - {} samples ({:.1}%)", i, name, count, percentage);
    }
    
    // Check for imbalance
    let max_count = *class_counts.iter().max().unwrap();
    let min_count = *class_counts.iter().min().unwrap();
    let imbalance_ratio = max_count as f64 / min_count as f64;
    
    if imbalance_ratio > 3.0 {
        println!("⚠️ Significant class imbalance detected (ratio: {:.1}:1)", imbalance_ratio);
        println!("   → Use stratified sampling");
        println!("   → Focus on per-class metrics");
        println!("   → Consider weighted averaging");
    }
    
    // Step 2: Generate and evaluate predictions
    let (y_true, y_pred) = generate_multiclass_predictions(total, &class_counts);
    
    // Step 3: Comprehensive multi-class evaluation
    multiclass_evaluation_workflow(&y_true, &y_pred, &class_names)?;
    
    Ok(())
}

fn multiclass_evaluation_workflow(y_true: &Array1<usize>, y_pred: &Array1<usize>, 
                                 class_names: &[&str]) -> Result<()> {
    println!("\n=== Multi-class Evaluation Workflow ===");
    
    // Overall metrics with different averaging
    let accuracy = accuracy_score(y_true, y_pred)?;
    let macro_f1 = f1_score(y_true, y_pred, None, Some("macro"), None)?;
    let micro_f1 = f1_score(y_true, y_pred, None, Some("micro"), None)?;
    let weighted_f1 = f1_score(y_true, y_pred, None, Some("weighted"), None)?;
    
    println!("Overall Performance:");
    println!("  Accuracy:    {:.3}", accuracy);
    println!("  Macro F1:    {:.3} (equal class importance)", macro_f1);
    println!("  Micro F1:    {:.3} (equal sample importance)", micro_f1);
    println!("  Weighted F1: {:.3} (frequency weighted)", weighted_f1);
    
    // Per-class analysis
    println!("\nPer-Class Performance:");
    let (cm, labels) = confusion_matrix(y_true, y_pred, None)?;
    
    println!("Class       | Precision | Recall | F1-Score | Support");
    println!("------------|-----------|--------|----------|--------");
    
    for (i, &class_id) in labels.iter().enumerate() {
        let tp = cm[[i, i]] as f64;
        let fp: f64 = cm.column(i).iter().enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, &val)| val as f64)
            .sum();
        let fn_val: f64 = cm.row(i).iter().enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, &val)| val as f64)
            .sum();
        
        let precision = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
        let recall = if tp + fn_val > 0.0 { tp / (tp + fn_val) } else { 0.0 };
        let f1 = if precision + recall > 0.0 { 
            2.0 * precision * recall / (precision + recall) 
        } else { 0.0 };
        let support = cm.row(i).sum();
        
        println!("{:<11} |   {:.3}   |  {:.3} |  {:.3}  |  {}",
                 class_names[class_id], precision, recall, f1, support);
    }
    
    // Confusion matrix visualization
    println!("\nConfusion Matrix:");
    println!("Actual\\Predicted  Normal  Pneumonia  COVID-19  Other");
    for (i, actual_class) in class_names.iter().enumerate() {
        print!("{:<15}", actual_class);
        for j in 0..class_names.len() {
            print!("  {:>6}", cm[[i, j]]);
        }
        println!();
    }
    
    // Error analysis
    println!("\nError Analysis:");
    let total_errors = cm.iter().enumerate()
        .filter(|(idx, _)| idx / class_names.len() != idx % class_names.len())
        .map(|(_, &val)| val as usize)
        .sum::<usize>();
    
    println!("Total misclassifications: {}", total_errors);
    
    // Most common errors
    let mut errors = Vec::new();
    for i in 0..class_names.len() {
        for j in 0..class_names.len() {
            if i != j && cm[[i, j]] > 0 {
                errors.push((i, j, cm[[i, j]]));
            }
        }
    }
    errors.sort_by_key(|(_, _, count)| std::cmp::Reverse(*count));
    
    println!("Most common confusions:");
    for (actual, predicted, count) in errors.iter().take(3) {
        println!("  {} → {}: {} cases", 
                 class_names[*actual], class_names[*predicted], count);
    }
    
    Ok(())
}
```

## Tutorial 3: Regression Model Evaluation

### Scenario
Predicting house prices with various regression metrics and residual analysis.

```rust
pub fn tutorial_3_regression_analysis() -> Result<()> {
    println!("🎯 Tutorial 3: Comprehensive Regression Evaluation");
    println!("📊 Scenario: House Price Prediction\n");
    
    // Generate realistic house price data
    let (y_true, y_pred) = generate_house_price_data();
    
    // Step 1: Basic regression metrics
    regression_basic_metrics(&y_true, &y_pred)?;
    
    // Step 2: Residual analysis
    regression_residual_analysis(&y_true, &y_pred)?;
    
    // Step 3: Distribution analysis
    regression_distribution_analysis(&y_true, &y_pred)?;
    
    // Step 4: Error characteristics
    regression_error_characteristics(&y_true, &y_pred)?;
    
    Ok(())
}

fn regression_basic_metrics(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<()> {
    println!("=== Basic Regression Metrics ===");
    
    let mae = mean_absolute_error(y_true, y_pred)?;
    let mse = mean_squared_error(y_true, y_pred)?;
    let rmse = root_mean_squared_error(y_true, y_pred)?;
    let r2 = r2_score(y_true, y_pred)?;
    let mape = mean_absolute_percentage_error(y_true, y_pred)?;
    
    // Calculate additional context metrics
    let mean_price = y_true.mean().unwrap();
    let std_price = y_true.std(1.0);
    let rmse_normalized = rmse / mean_price;
    
    println!("Core Metrics:");
    println!("  MAE:  ${:>10,.0} (Mean Absolute Error)", mae);
    println!("  RMSE: ${:>10,.0} (Root Mean Squared Error)", rmse);
    println!("  R²:   {:>10.3} (Coefficient of Determination)", r2);
    println!("  MAPE: {:>10.1}% (Mean Absolute Percentage Error)", mape);
    
    println!("\nContext:");
    println!("  Mean house price: ${:>10,.0}", mean_price);
    println!("  Price std dev:    ${:>10,.0}", std_price);
    println!("  RMSE as % of mean: {:>9.1}%", rmse_normalized * 100.0);
    
    // Interpretation guidance
    println!("\nInterpretation:");
    if r2 > 0.8 {
        println!("  ✓ Strong predictive power (R² > 0.8)");
    } else if r2 > 0.6 {
        println!("  ✓ Good predictive power (R² > 0.6)");
    } else if r2 > 0.4 {
        println!("  ⚠️ Moderate predictive power (R² > 0.4)");
    } else {
        println!("  ❌ Weak predictive power (R² ≤ 0.4)");
    }
    
    if rmse_normalized < 0.1 {
        println!("  ✓ Low prediction error (RMSE < 10% of mean)");
    } else if rmse_normalized < 0.2 {
        println!("  ⚠️ Moderate prediction error (RMSE 10-20% of mean)");
    } else {
        println!("  ❌ High prediction error (RMSE > 20% of mean)");
    }
    
    Ok(())
}

fn regression_residual_analysis(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Result<()> {
    println!("\n=== Residual Analysis ===");
    
    // Calculate residuals
    let residuals: Array1<f64> = y_true - y_pred;
    
    // Residual statistics
    let residual_mean = residuals.mean().unwrap();
    let residual_std = residuals.std(1.0);
    let residual_min = residuals.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let residual_max = residuals.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    
    println!("Residual Statistics:");
    println!("  Mean:     ${:>10,.0} (should be ~0)", residual_mean);
    println!("  Std Dev:  ${:>10,.0}", residual_std);
    println!("  Min:      ${:>10,.0}", residual_min);
    println!("  Max:      ${:>10,.0}", residual_max);
    
    // Check for bias
    let bias_threshold = residual_std * 0.1;
    if residual_mean.abs() > bias_threshold {
        println!("  ⚠️ Systematic bias detected (mean residual != 0)");
    } else {
        println!("  ✓ No systematic bias (mean residual ≈ 0)");
    }
    
    // Outlier analysis
    let outlier_threshold = 2.0 * residual_std;
    let outliers = residuals.iter()
        .filter(|&&r| r.abs() > outlier_threshold)
        .count();
    let outlier_percentage = outliers as f64 / residuals.len() as f64 * 100.0;
    
    println!("\nOutlier Analysis:");
    println!("  Outliers (>2σ): {} ({:.1}%)", outliers, outlier_percentage);
    
    if outlier_percentage > 5.0 {
        println!("  ⚠️ High outlier rate - investigate data quality or model assumptions");
    } else {
        println!("  ✓ Normal outlier rate (≤5%)");
    }
    
    // Residual distribution analysis
    let residuals_sorted = {
        let mut r = residuals.to_vec();
        r.sort_by(|a, b| a.partial_cmp(b).unwrap());
        r
    };
    
    let q25_idx = residuals_sorted.len() / 4;
    let q75_idx = 3 * residuals_sorted.len() / 4;
    let q25 = residuals_sorted[q25_idx];
    let q75 = residuals_sorted[q75_idx];
    let iqr = q75 - q25;
    
    println!("  Q25:      ${:>10,.0}", q25);
    println!("  Q75:      ${:>10,.0}", q75);
    println!("  IQR:      ${:>10,.0}", iqr);
    
    Ok(())
}
```

## Tutorial 4: Clustering Evaluation Workflow

### Scenario
Customer segmentation analysis with multiple clustering algorithms and evaluation strategies.

```rust
pub fn tutorial_4_clustering_evaluation() -> Result<()> {
    println!("🎯 Tutorial 4: Comprehensive Clustering Evaluation");
    println!("📊 Scenario: Customer Segmentation Analysis\n");
    
    // Generate customer data
    let (X, true_labels, pred_labels_kmeans, pred_labels_dbscan) = generate_customer_data();
    
    // Step 1: Internal evaluation (no ground truth)
    internal_clustering_evaluation(&X, &pred_labels_kmeans, "K-Means")?;
    internal_clustering_evaluation(&X, &pred_labels_dbscan, "DBSCAN")?;
    
    // Step 2: External evaluation (with ground truth)
    external_clustering_evaluation(&true_labels, &pred_labels_kmeans, "K-Means")?;
    external_clustering_evaluation(&true_labels, &pred_labels_dbscan, "DBSCAN")?;
    
    // Step 3: Comparative analysis
    comparative_clustering_analysis(&X, &true_labels, 
                                  &pred_labels_kmeans, &pred_labels_dbscan)?;
    
    Ok(())
}

fn internal_clustering_evaluation(X: &Array2<f64>, labels: &Array1<usize>, 
                                 algorithm: &str) -> Result<()> {
    println!("=== Internal Evaluation: {} ===", algorithm);
    
    let silhouette = silhouette_score(X, labels, "euclidean")?;
    let davies_bouldin = davies_bouldin_score(X, labels)?;
    let calinski_harabasz = calinski_harabasz_score(X, labels)?;
    
    println!("Internal Metrics:");
    println!("  Silhouette Score:       {:.3} (higher is better, [-1, 1])", silhouette);
    println!("  Davies-Bouldin Index:   {:.3} (lower is better)", davies_bouldin);
    println!("  Calinski-Harabasz:      {:.1} (higher is better)", calinski_harabasz);
    
    // Interpretation
    if silhouette > 0.5 {
        println!("  ✓ Well-defined clusters (Silhouette > 0.5)");
    } else if silhouette > 0.25 {
        println!("  ⚠️ Reasonable cluster structure (Silhouette > 0.25)");
    } else {
        println!("  ❌ Poor cluster definition (Silhouette ≤ 0.25)");
    }
    
    // Cluster statistics
    let n_clusters = labels.iter().max().unwrap() + 1;
    let total_points = labels.len();
    
    println!("\nCluster Statistics:");
    println!("  Number of clusters: {}", n_clusters);
    println!("  Total data points:  {}", total_points);
    
    for cluster_id in 0..n_clusters {
        let cluster_size = labels.iter().filter(|&&x| x == cluster_id).count();
        let percentage = cluster_size as f64 / total_points as f64 * 100.0;
        println!("  Cluster {}: {} points ({:.1}%)", cluster_id, cluster_size, percentage);
    }
    
    Ok(())
}

fn external_clustering_evaluation(true_labels: &Array1<usize>, pred_labels: &Array1<usize>, 
                                 algorithm: &str) -> Result<()> {
    println!("\n=== External Evaluation: {} ===", algorithm);
    
    let ari = adjusted_rand_index(true_labels, pred_labels)?;
    let nmi = normalized_mutual_info_score(true_labels, pred_labels, "arithmetic")?;
    let (homogeneity, completeness, v_measure) = 
        homogeneity_completeness_v_measure(true_labels, pred_labels, 1.0)?;
    
    println!("External Metrics:");
    println!("  Adjusted Rand Index:    {:.3} (higher is better, [-1, 1])", ari);
    println!("  Normalized Mutual Info: {:.3} (higher is better, [0, 1])", nmi);
    println!("  Homogeneity:           {:.3} (clusters contain single class)", homogeneity);
    println!("  Completeness:          {:.3} (class members in same cluster)", completeness);
    println!("  V-measure:             {:.3} (harmonic mean of H&C)", v_measure);
    
    // Interpretation
    if ari > 0.75 {
        println!("  ✓ Excellent agreement with ground truth (ARI > 0.75)");
    } else if ari > 0.5 {
        println!("  ✓ Good agreement with ground truth (ARI > 0.5)");
    } else if ari > 0.25 {
        println!("  ⚠️ Moderate agreement with ground truth (ARI > 0.25)");
    } else {
        println!("  ❌ Poor agreement with ground truth (ARI ≤ 0.25)");
    }
    
    Ok(())
}
```

## Tutorial 5: Model Comparison and Selection

This tutorial demonstrates how to systematically compare multiple models using proper statistical methods.

```rust
pub fn tutorial_5_model_comparison() -> Result<()> {
    println!("🎯 Tutorial 5: Model Comparison and Selection");
    println!("📊 Scenario: Comparing 3 Classification Models\n");
    
    // Simulate results from 3 different models
    let model_names = vec!["Logistic Regression", "Random Forest", "Neural Network"];
    let (y_true, model_predictions, model_scores) = generate_model_comparison_data();
    
    // Step 1: Individual model evaluation
    for (i, name) in model_names.iter().enumerate() {
        println!("=== Model {}: {} ===", i + 1, name);
        individual_model_evaluation(&y_true, &model_predictions[i], &model_scores[i])?;
    }
    
    // Step 2: Cross-validation comparison
    cross_validation_comparison(&model_names)?;
    
    // Step 3: Statistical significance testing
    statistical_comparison(&y_true, &model_predictions, &model_names)?;
    
    // Step 4: Final recommendation
    model_selection_recommendation(&model_names)?;
    
    Ok(())
}

fn statistical_comparison(y_true: &Array1<f64>, predictions: &[Array1<f64>], 
                         model_names: &[&str]) -> Result<()> {
    println!("\n=== Statistical Significance Testing ===");
    
    // McNemar's test for comparing classification models
    for i in 0..predictions.len() {
        for j in (i + 1)..predictions.len() {
            let (statistic, p_value) = mcnemar_test(y_true, &predictions[i], &predictions[j])?;
            
            println!("{} vs {}:", model_names[i], model_names[j]);
            println!("  McNemar statistic: {:.3}", statistic);
            println!("  p-value: {:.4}", p_value);
            
            if p_value < 0.001 {
                println!("  *** Highly significant difference (p < 0.001)");
            } else if p_value < 0.01 {
                println!("  ** Significant difference (p < 0.01)");
            } else if p_value < 0.05 {
                println!("  * Marginally significant difference (p < 0.05)");
            } else {
                println!("  No significant difference (p ≥ 0.05)");
            }
            println!();
        }
    }
    
    Ok(())
}
```

## Running All Tutorials

```rust
pub fn run_all_tutorials() -> Result<()> {
    println!("🚀 SciRS2 Metrics: Interactive Tutorials");
    println!("=========================================\n");
    
    // Run each tutorial
    tutorial_1_fraud_detection()?;
    println!("\n" + &"=".repeat(60) + "\n");
    
    tutorial_2_multiclass_medical()?;
    println!("\n" + &"=".repeat(60) + "\n");
    
    tutorial_3_regression_analysis()?;
    println!("\n" + &"=".repeat(60) + "\n");
    
    tutorial_4_clustering_evaluation()?;
    println!("\n" + &"=".repeat(60) + "\n");
    
    tutorial_5_model_comparison()?;
    
    println!("\n🎉 All tutorials completed!");
    println!("You now have comprehensive evaluation frameworks for all major ML tasks.");
    
    Ok(())
}
```

## Usage Instructions

To run these tutorials:

1. **Add to your Cargo.toml:**
```toml
[dependencies]
scirs2-metrics = { version = "0.1.0-alpha.5", features = ["tutorials"] }
```

2. **Run individual tutorials:**
```rust
use scirs2_metrics::tutorials::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Run specific tutorial
    tutorial_1_fraud_detection()?;
    
    // Or run all tutorials
    run_all_tutorials()?;
    
    Ok(())
}
```

3. **Adapt to your data:**
Replace the synthetic data generation functions with your actual data loading and preprocessing code.

Each tutorial provides:
- ✅ **Step-by-step guidance** through evaluation workflows
- 📊 **Concrete examples** with realistic data scenarios  
- 🔍 **Interpretation help** for understanding results
- ⚠️ **Common pitfall warnings** to avoid mistakes
- 💡 **Best practice recommendations** for production use

These tutorials serve as templates that you can customize for your specific use cases and domain requirements.