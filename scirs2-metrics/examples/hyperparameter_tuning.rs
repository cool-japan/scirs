//! Example of hyperparameter tuning with metrics
//!
//! This example shows how to use hyperparameter tuning with metrics.
//! To run this example, enable the 'optim_integration' feature:
//!
//! ```bash
//! cargo run --example hyperparameter_tuning --features optim_integration
//! ```

use scirs2_core::ndarray::{array, Array1, Array2};
#[cfg(feature = "optim_integration")]
use scirs2_metrics::integration::optim::{HyperParameter, HyperParameterTuner};
use std::error::Error;

#[cfg(feature = "optim_integration")]
use std::collections::HashMap;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn Error>> {
    #[cfg(not(feature = "optim_integration"))]
    {
        println!("This example requires the 'optim_integration' feature to be enabled.");
        println!(
            "Run with: cargo run --example hyperparameter_tuning --features optim_integration"
        );
        return Ok(());
    }

    #[cfg(feature = "optim_integration")]
    {
        println!("Hyperparameter Tuning Example");
        println!("----------------------------");

        // Create a simple dataset (XOR problem)
        let x = Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])?;
        let y = array![0.0, 1.0, 1.0, 0.0];

        // Define hyperparameters to tune
        let params = vec![
            HyperParameter::new("learning_rate", 0.01, 0.001, 0.1),
            HyperParameter::new("hidden_size", 5.0, 2.0, 20.0),
            HyperParameter::discrete("num_epochs", 100.0, 50.0, 500.0, 50.0),
        ];

        // Create hyperparameter tuner
        let mut tuner = HyperParameterTuner::new(params, "accuracy", true, 20)?;

        // Define evaluation function
        let eval_fn = |params: &HashMap<String, f64>| -> scirs2_metrics::error::Result<f64> {
            // Extract parameters
            let learning_rate = params["learning_rate"];
            let hidden_size = params["hidden_size"] as usize;
            let num_epochs = params["num_epochs"] as usize;

            // Train a simple model with these hyperparameters
            // This is a mock implementation - in a real world scenario, you would train a model here
            let accuracy = simulate_model_training(&x, &y, learning_rate, hidden_size, num_epochs)
                .map_err(|e| scirs2_metrics::error::MetricsError::Other(e.to_string()))?;

            println!(
                "Evaluated: lr={learning_rate:.4}, hidden={hidden_size}, epochs={num_epochs} -> accuracy={accuracy:.4}"
            );

            Ok(accuracy)
        };

        // Run hyperparameter search
        println!("\nRunning random search with 20 evaluations...");
        let result = tuner.random_search(eval_fn)?;

        // Print results
        println!("\nBest hyperparameters found:");
        for (name, value) in result.best_params() {
            match name.as_str() {
                "hidden_size" | "num_epochs" => println!("  {}: {}", name, *value as usize),
                _ => println!("  {name}: {value:.6}"),
            }
        }
        println!("Best accuracy: {:.6}", result.best_metric());

        // Get best hyperparameters
        let best_params = result.best_params();
        let best_learning_rate = best_params["learning_rate"];
        let best_hidden_size = best_params["hidden_size"] as usize;
        let best_epochs = best_params["num_epochs"] as usize;

        // Final evaluation with best parameters
        println!("\nFinal evaluation with best parameters:");
        let final_accuracy =
            simulate_model_training(&x, &y, best_learning_rate, best_hidden_size, best_epochs)?;
        println!("Final accuracy with best parameters: {final_accuracy:.6}");

        Ok(())
    }
}

/// Simulate training a simple neural network model for the XOR problem
#[allow(dead_code)]
fn simulate_model_training(
    _x: &Array2<f64>,
    _y: &Array1<f64>,
    learning_rate: f64,
    hidden_size: usize,
    num_epochs: usize,
) -> Result<f64, Box<dyn Error>> {
    // This is a simplified simulation of a model's performance based on hyperparameters
    // In a real-world scenario, you would train an actual model here

    // Simulate how well the model will perform based on hyperparameters
    let base_accuracy = 0.6;

    // Learning _rate factor: too small or too large reduces accuracy
    let lr_factor = if learning_rate < 0.005 {
        learning_rate / 0.005 // Too small
    } else if learning_rate > 0.05 {
        1.0 - ((learning_rate - 0.05) / 0.05) * 0.5 // Too large
    } else {
        1.0 // Good range
    };

    // Hidden _size factor: XOR needs at least 2 neurons, more is fine but less efficient
    let hidden_factor = if hidden_size < 3 {
        0.5 // Too small
    } else if hidden_size < 8 {
        1.0 // Good range
    } else {
        0.9 // Larger than needed
    };

    // Epochs factor: more _epochs generally better up to a point
    let epoch_factor = if num_epochs < 100 {
        0.7 + (num_epochs as f64 / 100.0) * 0.3 // Too few _epochs
    } else if num_epochs < 300 {
        1.0 // Good range
    } else {
        0.95 // More than needed
    };

    // Add some randomness to simulate training stochasticity
    let mut rng = scirs2_core::random::rng();
    let random_factor = 0.95 + scirs2_core::random::Rng::random_range(&mut rng, 0.0..0.1);

    // Calculate simulated accuracy
    let mut accuracy = base_accuracy * lr_factor * hidden_factor * epoch_factor * random_factor;

    // Cap at 1.0
    accuracy = accuracy.min(1.0);

    // For XOR with the right parameters, we should be able to get perfect accuracy
    if (0.005..=0.05).contains(&learning_rate)
        && (3..=8).contains(&hidden_size)
        && num_epochs >= 100
        && random_factor > 0.98
    {
        accuracy = 1.0;
    }

    Ok(accuracy)
}
