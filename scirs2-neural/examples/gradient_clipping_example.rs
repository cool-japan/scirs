use ndarray::ScalarOperand;
use ndarray::{Array, ArrayBase, IxDyn, OwnedRepr};
use num_traits::Float;
use scirs2_neural::error::Result;
use std::fmt::Debug;

// Simplified gradient clipping demo
fn main() -> Result<()> {
    println!("Gradient Clipping Example");
    println!("------------------------");

    // For a full implementation, please see the gradient_clipping.rs file
    // This example demonstrates the basic concept of gradient clipping

    println!("\nBasic Gradient Clipping by Norm:");

    // Create some sample gradients
    let mut gradients = create_sample_gradients();

    // Compute the global norm
    let global_norm = compute_global_norm(&gradients);
    println!("Original global norm: {:.4}", global_norm);

    // Set maximum norm
    let max_norm = 1.0;
    println!("Maximum allowed norm: {:.4}", max_norm);

    // Clip if necessary
    if global_norm > max_norm {
        let scale = max_norm / global_norm;
        println!("Applying clipping with scale factor: {:.4}", scale);

        // Scale all gradients
        for grad in &mut gradients {
            // Convert to f32 to match the gradient type
            let scale_f32 = scale as f32;
            *grad = grad.clone() * scale_f32;
        }

        // Compute new global norm
        let new_norm = compute_global_norm(&gradients);
        println!("New global norm after clipping: {:.4}", new_norm);
    } else {
        println!("No clipping needed, norm is below threshold");
    }

    println!("\nGradient Clipping by Value:");

    // Create some sample gradients
    let mut gradients = create_sample_gradients();

    // Set maximum value
    let max_value = 0.5;
    println!("Maximum allowed value: {:.4}", max_value);

    // Find maximum absolute value in gradients
    let mut max_abs = 0.0;
    for grad in &gradients {
        for &val in grad.iter() {
            let abs_val = val.abs() as f64;
            if abs_val > max_abs {
                max_abs = abs_val;
            }
        }
    }
    println!("Maximum absolute value before clipping: {:.4}", max_abs);

    // Clip if necessary
    if max_abs > max_value as f64 {
        println!("Applying value clipping");

        // Clip all gradients
        for grad in &mut gradients {
            for val in grad.iter_mut() {
                if *val > max_value {
                    *val = max_value;
                } else if *val < -max_value {
                    *val = -max_value;
                }
            }
        }

        // Find new maximum absolute value
        let mut new_max_abs = 0.0;
        for grad in &gradients {
            for &val in grad.iter() {
                let abs_val = val.abs() as f64;
                if abs_val > new_max_abs {
                    new_max_abs = abs_val;
                }
            }
        }
        println!("Maximum absolute value after clipping: {:.4}", new_max_abs);
    } else {
        println!("No clipping needed, all values are below threshold");
    }

    println!("\nFor production use, the GradientClipping callback provides:");
    println!("- Integration with the training loop through callbacks");
    println!("- Automatic application of clipping before optimization step");
    println!("- Tracking of clipping statistics");
    println!("- Support for both global norm and value clipping methods");

    Ok(())
}

// Create sample gradients with some large values
fn create_sample_gradients() -> Vec<Array<f32, IxDyn>> {
    let mut gradients = Vec::new();

    // First layer gradients (weight)
    let grad1 =
        Array::from_shape_vec(IxDyn(&[3, 2]), vec![0.1, 0.2, 1.5, -0.8, 0.7, -1.2]).unwrap();
    gradients.push(grad1);

    // First layer gradients (bias)
    let grad2 = Array::from_shape_vec(IxDyn(&[3]), vec![0.5, -1.8, 0.9]).unwrap();
    gradients.push(grad2);

    // Second layer gradients (weight)
    let grad3 =
        Array::from_shape_vec(IxDyn(&[2, 3]), vec![0.3, -0.4, 1.2, -1.5, 0.8, 0.6]).unwrap();
    gradients.push(grad3);

    // Second layer gradients (bias)
    let grad4 = Array::from_shape_vec(IxDyn(&[2]), vec![0.2, -0.9]).unwrap();
    gradients.push(grad4);

    gradients
}

// Compute global norm of gradients
fn compute_global_norm<A>(gradients: &[ArrayBase<OwnedRepr<A>, IxDyn>]) -> f64
where
    A: Float + ScalarOperand + Debug,
{
    let mut global_norm_sq = 0.0;

    for grad in gradients {
        for &val in grad.iter() {
            let val_f64 = val.to_f64().unwrap_or(0.0);
            global_norm_sq += val_f64 * val_f64;
        }
    }

    global_norm_sq.sqrt()
}
