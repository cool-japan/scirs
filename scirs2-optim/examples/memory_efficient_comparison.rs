// Example demonstrating memory-efficient optimizers
use ndarray::Array2;
use scirs2_optim::memory_efficient::{InPlaceAdam, InPlaceOptimizer, InPlaceSGD};
use scirs2_optim::optimizers::{Adam, SGD};
use scirs2_optim::Optimizer;
use std::error::Error;
use std::time::Instant;

// Function to measure memory usage (approximation)
fn estimate_memory_usage<T>(arrays: &[&T]) -> usize {
    std::mem::size_of_val(arrays)
}

// Simple quadratic loss function: f(x) = (x - target)^2
fn compute_loss_and_gradient(params: &Array2<f64>, target: &Array2<f64>) -> (f64, Array2<f64>) {
    let diff = params - target;
    let loss = diff.mapv(|x| x * x).sum() / (params.nrows() * params.ncols()) as f64;
    let grad = diff * 2.0 / (params.nrows() * params.ncols()) as f64;
    (loss, grad)
}

fn main() -> Result<(), Box<dyn Error>> {
    // Create large parameter array for memory testing
    let size = 1000;
    let mut params = Array2::from_elem((size, size), 0.5);
    let mut params_inplace = params.clone();
    let target = Array2::from_elem((size, size), 1.0);

    // Initialize optimizers
    let mut adam = Adam::new(0.01);
    let mut inplace_adam = InPlaceAdam::new(0.01);
    let mut sgd = SGD::new(0.1);
    let mut inplace_sgd = InPlaceSGD::new(0.1);

    // Track memory and performance
    let mut regular_adam_memory = vec![];
    let mut inplace_adam_memory = vec![];
    let mut regular_sgd_memory = vec![];
    let mut inplace_sgd_memory = vec![];

    println!("Memory-Efficient Optimizer Comparison");
    println!("====================================");
    println!("Parameter size: {}x{}", size, size);
    println!();

    // Test regular vs in-place Adam
    println!("Adam Optimizer Comparison:");
    println!("-------------------------");

    let start = Instant::now();
    for step in 0..10 {
        let (loss, grad) = compute_loss_and_gradient(&params, &target);

        // Regular Adam update
        params = adam.step(&params, &grad)?;

        // Estimate memory usage (this is approximate)
        regular_adam_memory.push(estimate_memory_usage(&[&params, &grad]));

        if step % 3 == 0 {
            println!(
                "  Step {}: Loss = {:.6}, Approx Memory = {} bytes",
                step,
                loss,
                regular_adam_memory.last().unwrap()
            );
        }
    }
    let regular_adam_time = start.elapsed();

    let start = Instant::now();
    for step in 0..10 {
        let (loss, grad) = compute_loss_and_gradient(&params_inplace, &target);

        // In-place Adam update
        inplace_adam.step_inplace(&mut params_inplace, &grad)?;

        // Estimate memory usage
        inplace_adam_memory.push(estimate_memory_usage(&[&params_inplace, &grad]));

        if step % 3 == 0 {
            println!(
                "  Step {} (in-place): Loss = {:.6}, Approx Memory = {} bytes",
                step,
                loss,
                inplace_adam_memory.last().unwrap()
            );
        }
    }
    let inplace_adam_time = start.elapsed();

    println!("\nAdam Performance:");
    println!("  Regular: {:?}", regular_adam_time);
    println!("  In-place: {:?}", inplace_adam_time);
    println!(
        "  Memory efficiency: ~{}% reduction",
        ((regular_adam_memory[0] - inplace_adam_memory[0]) * 100) / regular_adam_memory[0]
    );

    // Reset parameters for SGD test
    params = Array2::from_elem((size, size), 0.5);
    params_inplace = params.clone();

    println!("\nSGD Optimizer Comparison:");
    println!("------------------------");

    let start = Instant::now();
    for step in 0..10 {
        let (loss, grad) = compute_loss_and_gradient(&params, &target);

        // Regular SGD update
        params = sgd.step(&params, &grad)?;

        regular_sgd_memory.push(estimate_memory_usage(&[&params, &grad]));

        if step % 3 == 0 {
            println!(
                "  Step {}: Loss = {:.6}, Approx Memory = {} bytes",
                step,
                loss,
                regular_sgd_memory.last().unwrap()
            );
        }
    }
    let regular_sgd_time = start.elapsed();

    let start = Instant::now();
    for step in 0..10 {
        let (loss, grad) = compute_loss_and_gradient(&params_inplace, &target);

        // In-place SGD update
        inplace_sgd.step_inplace(&mut params_inplace, &grad)?;

        inplace_sgd_memory.push(estimate_memory_usage(&[&params_inplace, &grad]));

        if step % 3 == 0 {
            println!(
                "  Step {} (in-place): Loss = {:.6}, Approx Memory = {} bytes",
                step,
                loss,
                inplace_sgd_memory.last().unwrap()
            );
        }
    }
    let inplace_sgd_time = start.elapsed();

    println!("\nSGD Performance:");
    println!("  Regular: {:?}", regular_sgd_time);
    println!("  In-place: {:?}", inplace_sgd_time);
    println!(
        "  Memory efficiency: ~{}% reduction",
        ((regular_sgd_memory[0] - inplace_sgd_memory[0]) * 100) / regular_sgd_memory[0]
    );

    // Demonstrate utility functions
    println!("\nUtility Functions Demo:");
    println!("----------------------");

    use scirs2_optim::memory_efficient::{clip_inplace, normalize_inplace, scale_inplace};

    let mut test_array = Array2::from_elem((3, 3), 2.0);
    println!("Original array:\n{}", test_array);

    scale_inplace(&mut test_array, 0.5);
    println!("After scaling by 0.5:\n{}", test_array);

    clip_inplace(&mut test_array, -0.5, 0.5);
    println!("After clipping to [-0.5, 0.5]:\n{}", test_array);

    normalize_inplace(&mut test_array);
    println!("After normalizing:\n{}", test_array);

    Ok(())
}
