// Simple demo of in-place optimizers
use ndarray::Array1;
use scirs2_optim::memory_efficient::{clip_inplace, normalize_inplace, scale_inplace};
use scirs2_optim::memory_efficient::{InPlaceAdam, InPlaceOptimizer, InPlaceSGD};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    println!("Memory-Efficient Optimizer Demo");
    println!("==============================");

    // Create parameters and gradients
    let params = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    let gradients = Array1::from_vec(vec![0.1, 0.2, 0.3]);

    // Original parameters
    println!("Original parameters: {:?}", params);

    // In-place SGD
    {
        // Clone to preserve original
        let mut sgd_params = params.clone();
        let mut optimizer = InPlaceSGD::new(0.1);

        println!("\nSGD Update (learning rate = 0.1):");
        println!("Before: {:?}", sgd_params);

        optimizer.step_inplace(&mut sgd_params, &gradients)?;

        println!("After:  {:?}", sgd_params);
        println!(
            "Memory address unchanged: {}",
            params.as_ptr() != sgd_params.as_ptr()
        );
    }

    // In-place Adam
    {
        // Clone to preserve original
        let mut adam_params = params.clone();
        let mut optimizer = InPlaceAdam::new(0.1);

        println!("\nAdam Update (learning rate = 0.1):");
        println!("Before: {:?}", adam_params);

        optimizer.step_inplace(&mut adam_params, &gradients)?;

        println!("After:  {:?}", adam_params);
        println!(
            "Memory address unchanged: {}",
            params.as_ptr() != adam_params.as_ptr()
        );

        // Multiple updates to show momentum effects
        println!("\nAdam after 5 updates:");
        for _ in 0..4 {
            optimizer.step_inplace(&mut adam_params, &gradients)?;
        }
        println!("After 5 updates: {:?}", adam_params);
    }

    // Demonstrate utility functions
    println!("\nUtility Functions:");

    // Scale in-place
    let mut array = Array1::from_vec(vec![1.0, 2.0, 3.0]);
    println!("Original: {:?}", array);
    scale_inplace(&mut array, 0.5);
    println!("After scaling by 0.5: {:?}", array);

    // Clip in-place
    let mut array = Array1::from_vec(vec![0.5, 1.5, 2.5]);
    println!("Original: {:?}", array);
    clip_inplace(&mut array, 1.0, 2.0);
    println!("After clipping to [1.0, 2.0]: {:?}", array);

    // Normalize in-place
    let mut array = Array1::from_vec(vec![3.0, 4.0, 0.0]);
    println!("Original: {:?}", array);
    normalize_inplace(&mut array);
    println!("After normalizing: {:?}", array);

    Ok(())
}
