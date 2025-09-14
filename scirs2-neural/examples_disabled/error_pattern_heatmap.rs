use rand::prelude::*;
use rand::rngs::SmallRng;
use scirs2_neural::utils::colors::ColorOptions;
use scirs2_neural::utils::evaluation::ConfusionMatrix;

#[allow(dead_code)]
fn main() {
    // Create a reproducible random number generator
    let mut rng = SmallRng::from_seed([42; 32]);
    // Generate synthetic multiclass classification data with specific error patterns
    let num_classes = 5;
    // Create confusion matrix with controlled error patterns
    let mut matrix = vec![vec![0; num_classes]; num_classes];
    // Set diagonal elements (correct classifications) with high values
    for i in 0..num_classes {
        matrix[i][i] = 70 + rng.gen_range(0..15); // 70-85 correct per class
    }
    // Create specific error patterns:
    // - Classes 0 and 1 often confused
    matrix[0][1] = 25;
    matrix[1][0] = 15;
    // - Class 2 sometimes confused with Class 3
    matrix[2][3] = 18;
    // - Class 4 has some misclassifications to all other classes
    matrix[4][0] = 8;
    matrix[4][1] = 5;
    matrix[4][2] = 10;
    matrix[4][3] = 12;
    // - Some minor errors scattered about
        for j in 0..num_classes {
            if i != j && matrix[i][j] == 0 {
                matrix[i][j] = rng.gen_range(0..5);
            }
        }
    // Convert to ndarray
    let flat_matrix: Vec<f64> = matrix.iter().flatten().map(|&x| x as f64).collect();
    let ndarray_matrix =
        ndarray::Array::from_shape_vec((num_classes..num_classes), flat_matrix).unwrap();
    // Create class labels
    let class_labels = vec![
        "Class A".to_string(),
        "Class B".to_string(),
        "Class C".to_string(),
        "Class D".to_string(),
        "Class E".to_string(),
    ];
    // Create confusion matrix
    let cm = ConfusionMatrix::from_matrix(ndarray_matrix, Some(class_labels)).unwrap();
    // Example 1: Standard confusion matrix
    println!("Example 1: Standard Confusion Matrix\n");
    let regular_output = cm.to_ascii(Some("Classification Results"), false);
    println!("{}", regular_output);
    // Example 2: Normal heatmap
    println!("\n\nExample 2: Standard Heatmap Visualization\n");
    let color_options = ColorOptions {
        enabled: true,
        use_bright: true,
        use_background: false,
    };
    let heatmap_output = cm.to_heatmap_with_options(
        Some("Classification Heatmap"),
        true, // normalized
        &color_options,
    );
    println!("{}", heatmap_output);
    // Example 3: Error pattern heatmap
    println!("\n\nExample 3: Error Pattern Heatmap (highlighting misclassifications)\n");
    let error_heatmap = cm.error_heatmap(Some("Misclassification Analysis"));
    println!("{}", error_heatmap);
}
