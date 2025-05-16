# Neural Network Evaluation and Visualization Guide

This guide covers the evaluation and visualization tools available in the scirs2-neural crate for assessing and understanding model performance.

## Overview

The evaluation module provides:

1. Confusion matrices for classification evaluation
2. Feature importance visualization
3. ROC curves for binary classification models
4. Learning curves to diagnose underfitting and overfitting
5. Comprehensive metric calculation and visualization
6. Colored terminal output for better readability (optional)

## Confusion Matrix

Confusion matrices are essential for evaluating classification models. They show the counts of true positives, false positives, true negatives, and false negatives.

```rust
use scirs2_neural::utils::ConfusionMatrix;
use ndarray::Array;

// Create true labels and predictions
let y_true = Array::from_shape_fn(100, |_| rand::random::<usize>() % 3);
let y_pred = Array::from_shape_fn(100, |i| {
    if rand::random::<f32>() < 0.7 { y_true[i] } else { rand::random::<usize>() % 3 }
});

// Create confusion matrix with class labels
let class_labels = vec!["Cat".to_string(), "Dog".to_string(), "Bird".to_string()];
let cm = ConfusionMatrix::<f32>::new(
    &y_true.view(),
    &y_pred.view(),
    Some(3),
    Some(class_labels),
)?;

// Print raw confusion matrix
println!("{}", cm.to_ascii(Some("Pet Classification Results"), false));

// Print normalized confusion matrix (showing proportions)
println!("{}", cm.to_ascii(Some("Normalized Results"), true));

// Create heatmap visualization for better visual analysis
println!("{}", cm.to_heatmap(Some("Confusion Matrix Heatmap"), true));

// Calculate metrics
let accuracy = cm.accuracy();
let precision = cm.precision();
let recall = cm.recall();
let f1 = cm.f1_score();
let macro_f1 = cm.macro_f1();

println!("Overall accuracy: {:.3}", accuracy);
```

### Sample Output

```
Pet Classification Results

Pred→   | Cat     | Dog     | Bird    | Recall
--------|---------|---------|---------|--------
Cat     | 23      | 3       | 4       | 0.767
Dog     | 5       | 24      | 1       | 0.800
Bird    | 3       | 2       | 35      | 0.875
--------|---------|---------|---------|--------
Precision| 0.742  | 0.828   | 0.875   | 0.820
F1-score | 0.754  | 0.814   | 0.875   | 0.814
```

### Heatmap Visualization

The library also provides a heatmap visualization for confusion matrices, making it easier to spot patterns and problematic classes:

```rust
// Create a confusion matrix with the heatmap visualization
let cm = ConfusionMatrix::<f32>::new(&y_true.view(), &y_pred.view(), None, Some(class_labels))?;

// Display using heatmap with color gradient 
// (normalized to show proportions rather than raw counts)
println!("{}", cm.to_heatmap(Some("Classification Heatmap"), true));

// Customize colors with options
let color_options = ColorOptions {
    enabled: true,
    use_bright: true,
    use_background: false,
};

println!("{}", cm.to_heatmap_with_options(
    Some("Detailed Heatmap"), 
    true, 
    &color_options
));
```

The heatmap uses a detailed color gradient:
- Blue: Very low values (0.0-0.2)
- Cyan: Low values (0.2-0.4)
- Yellow: Medium values (0.4-0.6)
- Red: High values (0.6-0.8)
- Magenta: Very high values (0.8-1.0)

This visualization makes it much easier to identify class confusion patterns compared to the standard visualization.

### Error Pattern Heatmap

For an even more focused analysis of misclassification patterns, the library provides a specialized error pattern heatmap that deliberately highlights where the model makes mistakes:

```rust
// Create a confusion matrix
let cm = ConfusionMatrix::<f32>::new(&y_true.view(), &y_pred.view(), None, Some(class_labels))?;

// Generate an error-focused heatmap that highlights misclassification patterns
println!("{}", cm.error_heatmap(Some("Misclassification Analysis")));
```

The error pattern heatmap:
- De-emphasizes diagonal elements (correct classifications) with dim styling
- Uses a specialized color gradient to highlight off-diagonal elements (errors)
- Normalizes error values relative to the maximum off-diagonal value
- Provides an error-specific legend that explains the intensity levels
- Helps quickly identify which classes the model frequently confuses

This visualization is particularly useful for:
- Identifying systematic error patterns in multiclass classification
- Spotting asymmetric confusion (where Class A is mistaken for Class B, but not vice versa)
- Finding classes that are frequently confused with many other classes
- Focusing remediation efforts on the most problematic class pairs

## Feature Importance

Visualize the relative importance of features in your model to understand which inputs contribute most to predictions.

```rust
use scirs2_neural::utils::FeatureImportance;
use ndarray::Array1;

// Feature names and importance scores
let feature_names = vec![
    "Age".to_string(),
    "Income".to_string(),
    "Education".to_string(),
    "Location".to_string(),
    "Family Size".to_string(),
];

let importance = Array1::from_vec(vec![0.35, 0.25, 0.15, 0.18, 0.07]);

// Create feature importance visualization
let fi = FeatureImportance::new(feature_names, importance)?;

// Display all features
println!("{}", fi.to_ascii(Some("Customer Churn Predictors"), 50, None));

// Display top 3 features
println!("{}", fi.to_ascii(Some("Top 3 Predictors"), 50, Some(3)));
```

### Sample Output

```
Customer Churn Predictors

Age          | 0.350 |████████████████████████████████████|
Income       | 0.250 |██████████████████████████|
Location     | 0.180 |██████████████████|
Education    | 0.150 |███████████████|
Family Size  | 0.070 |███████|
```

## ROC Curve

ROC (Receiver Operating Characteristic) curves visualize the performance of binary classification models across different thresholds.

```rust
use scirs2_neural::utils::ROCCurve;
use ndarray::Array1;

// Binary classification data (0 or 1)
let y_true = Array1::from_vec(vec![0, 1, 1, 0, 1, 0, 1, 0, 0, 1]);

// Predicted probabilities for the positive class
let y_scores = Array1::from_vec(vec![0.1, 0.9, 0.8, 0.3, 0.7, 0.2, 0.6, 0.4, 0.3, 0.8]);

// Create ROC curve
let roc = ROCCurve::new(&y_true.view(), &y_scores.view())?;

// Display ROC curve
println!("ROC AUC: {:.3}", roc.auc);
println!("{}", roc.to_ascii(Some("Binary Classification ROC"), 60, 25));

// With color support
let color_options = ColorOptions {
    enabled: true,
    use_bright: true,
    use_background: false,
};
println!("{}", roc.to_ascii_with_options(Some("Colored ROC Curve"), 60, 25, &color_options));
```

### Sample Output

```
Binary Classification ROC (AUC = 0.860)

1.0 |          ●●●●●                                  
    |        ●●                                       
    |       ●                                         
    |      ●                                         
    |      ●                                         
0.5 |     ●                                         
    |    ●                                         
    |    ●                                         
    |   ●                                         
    |  ●                                         
0.0 |●.................................................
     0.0                                         1.0
     False Positive Rate (FPR)
```

## Learning Curve

Learning curves help diagnose overfitting, underfitting, and determine if more training data would help improve model performance.

```rust
use scirs2_neural::utils::LearningCurve;
use ndarray::{Array1, Array2};

// Training set sizes
let train_sizes = Array1::from_vec(vec![100, 200, 300, 400, 500, 600, 700, 800]);

// Multiple cross-validation runs (rows=sizes, cols=CV folds)
let train_scores = Array2::from_shape_vec((8, 3), vec![
    0.99, 0.98, 0.97,  // 100 samples
    0.97, 0.96, 0.95,  // 200 samples
    0.94, 0.93, 0.94,  // 300 samples
    0.93, 0.92, 0.91,  // 400 samples
    0.91, 0.90, 0.91,  // 500 samples
    0.89, 0.88, 0.89,  // 600 samples
    0.88, 0.87, 0.86,  // 700 samples
    0.87, 0.86, 0.85,  // 800 samples
])?;

let val_scores = Array2::from_shape_vec((8, 3), vec![
    0.70, 0.69, 0.68,  // 100 samples
    0.75, 0.74, 0.73,  // 200 samples
    0.78, 0.77, 0.76,  // 300 samples
    0.80, 0.79, 0.78,  // 400 samples
    0.82, 0.81, 0.80,  // 500 samples
    0.83, 0.82, 0.83,  // 600 samples
    0.84, 0.83, 0.84,  // 700 samples
    0.85, 0.84, 0.85,  // 800 samples
])?;

// Create learning curve
let lc = LearningCurve::new(train_sizes, train_scores, val_scores)?;

// Display learning curve
println!("{}", lc.to_ascii(Some("Model Learning Curve"), 60, 20, "Accuracy"));

// With color support
let color_options = ColorOptions {
    enabled: true,
    use_bright: true,
    use_background: false,
};
println!("{}", lc.to_ascii_with_options(
    Some("Colored Learning Curve"), 
    60, 
    20, 
    "Accuracy", 
    &color_options
));
```

### Sample Output

```
Model Learning Curve (Accuracy)

0.99 |●                                                
     |  ●                                              
     |    ●                                            
     |      ●                                          
0.90 |         ●                                      
     |           ●                                     
     |             ●                                   
     |               ●                                 
0.80 |                                 ○              
     |                             ○                   
     |                         ○                       
     |                     ○                           
0.70 |○                                               
     +--------------------------------------------------
      100       300       500       700       800
      Training Set Size

      ● Training score   ○ Validation score
```

## Integration with Training Loop

These visualizations can be integrated with your training loop to monitor progress and diagnose issues:

```rust
// Within your training loop:
let mut history = HashMap::new();
history.insert("train_loss".to_string(), vec![]);
history.insert("val_loss".to_string(), vec![]);
history.insert("accuracy".to_string(), vec![]);

// After each epoch:
for epoch in 0..num_epochs {
    // Train and evaluate model
    let train_loss = /* ... */;
    let val_loss = /* ... */;
    let accuracy = /* ... */;
    
    // Update history
    history.get_mut("train_loss").unwrap().push(train_loss);
    history.get_mut("val_loss").unwrap().push(val_loss);
    history.get_mut("accuracy").unwrap().push(accuracy);
    
    // Every N epochs, visualize current learning curve
    if epoch > 0 && epoch % 5 == 0 {
        let train_sizes = Array1::from_iter(0..epoch+1);
        let train_scores = Array2::from_shape_vec((epoch+1, 1), 
            history.get("train_loss").unwrap().clone())?;
        let val_scores = Array2::from_shape_vec((epoch+1, 1), 
            history.get("val_loss").unwrap().clone())?;
        
        let lc = LearningCurve::new(train_sizes, train_scores, val_scores)?;
        println!("{}", lc.to_ascii(Some("Current Learning Curve"), 60, 20, "Loss"));
    }
}

// After training completes, generate confusion matrix
let y_true = /* Get true labels */;
let y_pred = /* Get model predictions */;
let cm = ConfusionMatrix::<f32>::new(&y_true.view(), &y_pred.view(), None, None)?;
println!("{}", cm.to_ascii(Some("Final Model Performance"), false));
```

## Using Colored Visualizations

The visualization tools support colored output for better readability in terminal environments. You can enable colors and customize their appearance:

```rust
use scirs2_neural::utils::{
    ConfusionMatrix, ColorOptions, Color, Style, colorize, stylize
};

// Configure color options
let color_options = ColorOptions {
    enabled: true,           // Enable colors
    use_background: false,   // Use foreground colors only
    use_bright: true,        // Use bright colors
};

// Create confusion matrix
let cm = ConfusionMatrix::new(&y_true.view(), &y_pred.view(), None, None)?;

// Display with colors
println!("{}", cm.to_ascii_with_options(
    Some("Confusion Matrix"), 
    true,  // Normalize values 
    &color_options
));

// Create styled text
println!("{}", stylize("Important metrics:", Style::Bold));

// Apply colors to text
println!("{}", colorize("High accuracy: 0.95", Color::BrightGreen));
println!("{}", colorize("Low recall: 0.45", Color::BrightRed));
```

The color system automatically:
- Highlights diagonal cells in the confusion matrix
- Colors values based on their magnitude (red for low, yellow for medium, green for high)
- Emphasizes important headers and labels
- Highlights problematic values in red

### Auto-detecting Color Support

By default, the `ColorOptions::default()` will auto-detect terminal color support based on environment variables and platform. You can override this by setting `enabled` to `true` or `false`.

## Complete Examples

The examples directory contains:

- `model_visualization_example.rs` - Basic visualization features without colors
- `colored_eval_visualization.rs` - Enhanced confusion matrix and feature importance with colors
- `colored_curve_visualization.rs` - ROC curves and learning curves with color support
- `confusion_matrix_heatmap.rs` - Demonstrates the new heatmap visualization for confusion matrices
- `error_pattern_heatmap.rs` - Shows how to highlight and analyze misclassification patterns with the error heatmap

## Best Practices

1. **Visualize Early and Often**: Generate visualizations throughout training to catch issues early.

2. **Compare Models**: Use these tools to compare different models or hyperparameter settings.

3. **Understand Your Data**: Use feature importance to better understand what drives your model's decisions.

4. **Diagnose Learning Problems**: Use learning curves to determine if you need more data, more model capacity, or more regularization.

5. **Use Colors for Clarity**: Enable colors when running in a terminal to make visualization more readable and highlight important metrics.

6. **Disable Colors for Logs**: Disable colors when saving output to log files to avoid ANSI code artifacts.

7. **Present Results Clearly**: The ASCII visualizations provide an effective way to share results directly in terminal output or documentation.