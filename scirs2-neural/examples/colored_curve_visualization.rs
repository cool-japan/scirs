use ndarray::{Array1, Array2};
use rand::prelude::*;
use rand::rngs::SmallRng;
use rand_distr::StandardNormal;
use scirs2_neural::utils::colors::ColorOptions;
use scirs2_neural::utils::evaluation::{LearningCurve, ROCCurve};

fn main() {
    // Create a reproducible random number generator
    let mut rng = SmallRng::seed_from_u64(42);

    // Example 1: ROC Curve with color
    println!("Example 1: ROC Curve Visualization (with color)\n");

    // Generate synthetic binary classification data
    let n_samples = 200;

    // Generate true labels: 0 or 1
    let y_true: Vec<usize> = (0..n_samples)
        .map(|_| if rng.random::<f64>() > 0.5 { 1 } else { 0 })
        .collect();

    // Generate scores with some separability
    let y_score: Vec<f64> = y_true
        .iter()
        .map(|&label| {
            if label == 1 {
                0.7 + 0.3 * rng.sample::<f64, _>(StandardNormal)
            } else {
                0.3 + 0.3 * rng.sample::<f64, _>(StandardNormal)
            }
        })
        .collect();

    // Convert to ndarray views
    let y_true_array = Array1::from(y_true.clone());
    let y_score_array = Array1::from(y_score.clone());

    // Create ROC curve
    let roc = ROCCurve::new(&y_true_array.view(), &y_score_array.view()).unwrap();

    // Enable color options
    let color_options = ColorOptions {
        enabled: true,
        use_bright: true,
        use_background: false,
    };

    // Plot ROC curve with color
    let roc_plot = roc.to_ascii_with_options(
        Some("Binary Classification ROC Curve"),
        60,
        20,
        &color_options,
    );
    println!("{}", roc_plot);

    // Example 2: Learning Curve with color
    println!("\nExample 2: Learning Curve Visualization (with color)\n");

    // Simulate learning curves for different training set sizes
    let train_sizes = Array1::from(vec![100, 200, 300, 400, 500]);

    // Simulated training scores for each size (5 sizes, 3 CV folds)
    let train_scores = Array2::from_shape_fn((5, 3), |(i, _j)| {
        let base = 0.5 + 0.4 * (i as f64 / 4.0);
        let noise = 0.05 * rng.sample::<f64, _>(StandardNormal);
        base + noise
    });

    // Simulated validation scores (typically lower than training)
    let val_scores = Array2::from_shape_fn((5, 3), |(i, _j)| {
        let base = 0.4 + 0.3 * (i as f64 / 4.0);
        let noise = 0.07 * rng.sample::<f64, _>(StandardNormal);
        base + noise
    });

    // Create learning curve
    let learning_curve = LearningCurve::new(train_sizes, train_scores, val_scores).unwrap();

    // Plot learning curve with color
    let learning_plot = learning_curve.to_ascii_with_options(
        Some("Neural Network Training"),
        70,
        20,
        "Accuracy",
        &color_options,
    );
    println!("{}", learning_plot);
}
