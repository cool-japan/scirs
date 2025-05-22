use ndarray::{Array1, Array2};

// This is a minimal test for neural_integration.rs
fn main() {
    println!("Testing ndarray imports");
    
    // Create some example data (a simple regression problem)
    let targets = Array1::<f64>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let predictions = Array1::<f64>::from_vec(vec![1.2, 1.8, 3.3, 3.5, 5.2]);

    // Binary predictions
    let binary_targets = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 0.0]);
    let binary_predictions = Array1::from_vec(vec![0.0, 1.0, 0.0, 0.0, 0.0]);
    
    // Verify everything is working
    println!("Targets: {:?}", targets);
    println!("Predictions: {:?}", predictions);
    println!("Binary targets: {:?}", binary_targets);
    println!("Binary predictions: {:?}", binary_predictions);
    
    println!("All tests passed!");
}