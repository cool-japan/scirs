use scirs2_core::ndarray::{array, Array2};
use scirs2_stats::{stepwise_regression, StepwiseCriterion, StepwiseDirection};

#[allow(dead_code)]
fn main() {
    println!("Stepwise Regression Example\n");

    // Create a design matrix with 5 features
    let x = Array2::from_shape_vec(
        (10, 5),
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 2.0, 1.0, 3.0, 4.0, 6.0, 3.0, 2.0, 1.0, 4.0, 7.0, 4.0, 3.0,
            2.0, 1.0, 8.0, 5.0, 4.0, 3.0, 2.0, 1.0, 6.0, 5.0, 4.0, 3.0, 2.0, 7.0, 6.0, 5.0, 4.0,
            3.0, 8.0, 7.0, 6.0, 5.0, 4.0, 9.0, 8.0, 7.0, 6.0, 5.0, 10.0, 9.0, 8.0, 7.0, 6.0,
        ],
    )
    .unwrap();

    // Let's create a synthetic response variable
    // y ≈ 2 + 3*x1 - 0*x2 + 0*x3 - 2*x4 + 0*x5 + noise
    let y = array![10.1, 11.2, 12.8, 14.3, 9.5, 8.1, 7.2, 6.6, 5.3, 4.9];

    // Feature names (for display purposes only)
    let feature_names = [
        "Feature A".to_string(),
        "Feature B".to_string(),
        "Feature C".to_string(),
        "Feature D".to_string(),
        "Feature E".to_string(),
    ];

    println!("--- Forward Stepwise Regression (AIC) ---");

    // Perform forward stepwise regression with AIC criterion
    let forward_results = stepwise_regression(
        &x.view(),
        &y.view(),
        StepwiseDirection::Forward,
        StepwiseCriterion::AIC,
        Some(0.05), // p_enter
        Some(0.1),  // p_remove
        None,       // max_steps (use default)
        true,       // include_intercept
    )
    .unwrap();

    // Print the results
    println!(
        "Selected feature indices: {:?}",
        forward_results.selected_indices
    );
    println!("Selection sequence: {:?}", forward_results.sequence);
    println!(
        "Final model R² = {:.4}",
        forward_results.final_model.r_squared
    );

    // Print selected feature names
    println!("Selected features:");
    for &idx in &forward_results.selected_indices {
        println!("  {}: {}", idx, feature_names[idx]);
    }
    println!();

    println!("--- Backward Stepwise Regression (BIC) ---");

    // Perform backward stepwise regression with BIC criterion
    let backward_results = stepwise_regression(
        &x.view(),
        &y.view(),
        StepwiseDirection::Backward,
        StepwiseCriterion::BIC,
        Some(0.05), // p_enter
        Some(0.1),  // p_remove
        None,       // max_steps (use default)
        true,       // include_intercept
    )
    .unwrap();

    // Print the results
    println!(
        "Selected feature indices: {:?}",
        backward_results.selected_indices
    );
    println!("Selection sequence: {:?}", backward_results.sequence);
    println!(
        "Final model R² = {:.4}",
        backward_results.final_model.r_squared
    );

    // Print selected feature names
    println!("Selected features:");
    for &idx in &backward_results.selected_indices {
        println!("  {}: {}", idx, feature_names[idx]);
    }
    println!();

    println!("--- Bidirectional Stepwise Regression (Adjusted R²) ---");

    // Perform bidirectional stepwise regression with Adjusted R² criterion
    let bidirectional_results = stepwise_regression(
        &x.view(),
        &y.view(),
        StepwiseDirection::Both, // Both = Bidirectional
        StepwiseCriterion::AdjR2,
        Some(0.01), // p_enter - threshold for improvement
        Some(0.05), // p_remove - threshold for removal
        None,       // max_steps (use default)
        true,       // include_intercept
    )
    .unwrap();

    // Print the results
    println!(
        "Selected feature indices: {:?}",
        bidirectional_results.selected_indices
    );
    println!("Selection sequence: {:?}", bidirectional_results.sequence);
    println!(
        "Final model R² = {:.4}",
        bidirectional_results.final_model.r_squared
    );
    println!(
        "Final model Adjusted R² = {:.4}",
        bidirectional_results.final_model.adj_r_squared
    );

    // Print selected feature names
    println!("Selected features:");
    for &idx in &bidirectional_results.selected_indices {
        println!("  {}: {}", idx, feature_names[idx]);
    }
    println!();

    // Print detailed summary of the final model
    println!("Detailed Summary of Bidirectional Stepwise Regression:");
    println!("{}", bidirectional_results.summary());
}
