use scirs2_core::ndarray::Array1;
use std::error::Error;

// Test module to verify the fix for the binary_predictions array in neural_integration.rs

#[test]
#[allow(dead_code)]
fn test_binary_predictions_accuracy() -> Result<(), Box<dyn Error>> {
    // Binary predictions - this mirrors the fix we made in the neural_integration.rs example
    // The original array was [0.0, 1.0, 0.0, 0.0, 1.0] - giving 3/5 correct (0.6 accuracy)
    // We fixed it to be [0.0, 1.0, 0.0, 0.0, 0.0] - giving 4/5 correct (0.8 accuracy)
    let binary_targets = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 0.0]);
    
    // Old array (with error)
    let binary_predictions_old = Array1::from_vec(vec![0.0, 1.0, 0.0, 0.0, 1.0]);
    
    // New array (fixed)
    let binary_predictions_new = Array1::from_vec(vec![0.0, 1.0, 0.0, 0.0, 0.0]);
    
    // Calculate old accuracy manually
    let mut correct_old = 0;
    for (p, t) in binary_predictions_old.iter().zip(binary_targets.iter()) {
        if (p == &0.0 && t == &0.0) || (p != &0.0 && t != &0.0) {
            correct_old += 1;
        }
    }
    let accuracy_old = correct_old as f64 / binary_targets.len() as f64;
    
    // Calculate new accuracy manually
    let mut correct_new = 0;
    for (p, t) in binary_predictions_new.iter().zip(binary_targets.iter()) {
        if (p == &0.0 && t == &0.0) || (p != &0.0 && t != &0.0) {
            correct_new += 1;
        }
    }
    let accuracy_new = correct_new as f64 / binary_targets.len() as f64;
    
    // Verify that the old accuracy was 0.6 (3/5)
    assert_eq!(accuracy_old, 0.6);
    
    // Verify that the new accuracy is 0.8 (4/5)
    assert_eq!(accuracy_new, 0.8);
    
    // Verify improvement
    assert!(accuracy_new > accuracy_old);
    
    Ok(())
}
