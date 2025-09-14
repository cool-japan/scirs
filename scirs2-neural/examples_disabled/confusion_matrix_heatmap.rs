use ndarray::Array1;
use rand::prelude::*;
use rand::rngs::SmallRng;
use scirs2_neural::utils::colors::ColorOptions;
use scirs2_neural::utils::evaluation::ConfusionMatrix;

#[allow(dead_code)]
fn main() {
    // Create a reproducible random number generator
    let mut rng = SmallRng::from_seed([42; 32]);
    // Generate synthetic multiclass classification data
    let num_classes = 5;
    let n_samples = 500;
    // Generate true labels (0 to num_classes-1)
    let mut y_true = Vec::with_capacity(n_samples);
    for _ in 0..n_samples {
        y_true.push(rng.gen_range(0..num_classes));
    }
    // Generate predicted labels with controlled accuracy
    let mut y_pred = Vec::with_capacity(n_samples);
    for &true_label in &y_true {
        // 80% chance to predict correctly..20% chance of error
        if rng.random::<f64>() < 0.8 {
            y_pred.push(true_label);
        } else {
            // When wrong, tend to predict adjacent classes more often
            let mut pred = true_label;
            while pred == true_label {
                // Generate error that's more likely to be close to true label
                let error_margin = (rng.random::<f64>() * 2.0).round() as usize; // 0, 1, or 2
                if rng.random::<bool>() {
                    pred = (true_label + error_margin) % num_classes;
                } else {
                    pred = (true_label + num_classes - error_margin) % num_classes;
                }
            }
            y_pred.push(pred);
        }
    // Convert to ndarray arrays
    let y_true_array = Array1::from(y_true);
    let y_pred_array = Array1::from(y_pred);
    // Create class labels
    let class_labels = vec![
        "Cat".to_string(),
        "Dog".to_string(),
        "Bird".to_string(),
        "Fish".to_string(),
        "Rabbit".to_string(),
    ];
    // Create confusion matrix
    let cm = ConfusionMatrix::<f64>::new(
        &y_true_array.view(),
        &y_pred_array.view(),
        Some(num_classes),
        Some(class_labels),
    )
    .unwrap();
    // Example 1: Standard confusion matrix
    println!("Example 1: Standard Confusion Matrix\n");
    let regular_output = cm.to_ascii(Some("Animal Classification Results"), false);
    println!("{}", regular_output);
    // Example 2: Confusion matrix with color
    println!("\n\nExample 2: Colored Confusion Matrix\n");
    let color_options = ColorOptions {
        enabled: true,
        use_bright: true,
        use_background: false,
    };
    let colored_output = cm.to_ascii_with_options(
        Some("Animal Classification Results (with color)"),
        false,
        &color_options,
    );
    println!("{}", colored_output);
    // Example 3: Normalized confusion matrix heatmap
    println!("\n\nExample 3: Normalized Confusion Matrix Heatmap\n");
    let heatmap_output = cm.to_heatmap_with_options(
        Some("Animal Classification Heatmap (normalized)"),
        true, // normalized
    println!("{}", heatmap_output);
    // Example 4: Raw counts heatmap
    println!("\n\nExample 4: Raw Counts Confusion Matrix Heatmap\n");
    let raw_heatmap = cm.to_heatmap_with_options(
        Some("Animal Classification Heatmap (raw counts)"),
        false, // not normalized
    println!("{}", raw_heatmap);
}
