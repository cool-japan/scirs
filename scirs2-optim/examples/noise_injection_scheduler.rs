//! Example demonstrating the noise injection scheduler
//!
//! This example shows how adding noise to the learning rate can help
//! escape local minima and improve exploration during training.

use ndarray::Array1;
use rand::prelude::*;
use scirs2_optim::{
    optimizers::{Optimizer, SGD},
    schedulers::{
        ConstantScheduler, ExponentialDecay, LearningRateScheduler, NoiseDistribution,
        NoiseInjectionScheduler,
    },
};
use std::error::Error;

/// A simple 2D function with multiple local minima
/// f(x, y) = x^2 + 5*sin(y) + (x*y)^2
fn multimodal_function(point: &Array1<f64>) -> f64 {
    let x = point[0];
    let y = point[1];

    x.powi(2) + 5.0 * y.sin() + (x * y).powi(2)
}

/// Compute the gradient of the multimodal function
fn multimodal_gradient(point: &Array1<f64>) -> Array1<f64> {
    let x = point[0];
    let y = point[1];

    // Gradient with respect to x: 2x + 2xy^2
    let dx = 2.0 * x + 2.0 * x * y.powi(2);

    // Gradient with respect to y: 5*cos(y) + 2x^2y
    let dy = 5.0 * y.cos() + 2.0 * x.powi(2) * y;

    Array1::from_vec(vec![dx, dy])
}

/// Run optimization with the given scheduler and return the optimization trajectory
fn optimize<S: LearningRateScheduler<f64>>(
    initial_point: &Array1<f64>,
    scheduler: S,
    iterations: usize,
) -> Result<(Array1<f64>, Vec<(f64, f64, f64)>), Box<dyn Error>> {
    let mut optimizer = SGD::<f64>::new(scheduler.get_learning_rate());
    let mut point = initial_point.clone();
    let mut trajectory = Vec::with_capacity(iterations);
    let mut scheduler = scheduler;

    // Record initial point
    let initial_loss = multimodal_function(&point);
    trajectory.push((point[0], point[1], initial_loss));

    for _ in 0..iterations {
        // Compute loss and gradient
        let loss = multimodal_function(&point);
        let gradient = multimodal_gradient(&point);

        // Update learning rate
        let lr = scheduler.step();
        <SGD<f64> as Optimizer<f64, ndarray::Ix1>>::set_learning_rate(&mut optimizer, lr);

        // Update parameters
        point = optimizer.step(&point, &gradient)?;

        // Record trajectory
        trajectory.push((point[0], point[1], loss));
    }

    Ok((point, trajectory))
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("Noise Injection Scheduler Example");
    println!("=================================");

    // Set number of iterations
    let iterations = 100;

    // Create multiple initial points to test different optimizers
    let mut rng = rand::rng();
    let initial_points: Vec<Array1<f64>> = (0..5)
        .map(|_| {
            Array1::from_vec(vec![
                rng.random_range(-2.0..2.0),
                rng.random_range(-2.0..2.0),
            ])
        })
        .collect();

    // Test different schedulers
    for (i, initial_point) in initial_points.into_iter().enumerate() {
        println!("\nOptimization Run {}", i + 1);
        println!("-------------------");
        println!(
            "Initial point: ({:.4}, {:.4})",
            initial_point[0], initial_point[1]
        );
        println!("Initial loss: {:.4}", multimodal_function(&initial_point));

        // 1. Constant learning rate
        let constant_scheduler = ConstantScheduler::new(0.01);
        let (final_point_constant, _) = optimize(&initial_point, constant_scheduler, iterations)?;
        let final_loss_constant = multimodal_function(&final_point_constant);

        // 2. Exponential decay
        let exp_scheduler = ExponentialDecay::new(0.01, 0.99, 10); // Decay every 10 steps
        let (final_point_exp, _) = optimize(&initial_point, exp_scheduler, iterations)?;
        let final_loss_exp = multimodal_function(&final_point_exp);

        // 3. Uniform noise injection
        let uniform_noise_scheduler = NoiseInjectionScheduler::new(
            ConstantScheduler::new(0.01),
            NoiseDistribution::Uniform {
                min: -0.005,
                max: 0.005,
            },
            0.001,
        );
        let (final_point_uniform, _) =
            optimize(&initial_point, uniform_noise_scheduler, iterations)?;
        let final_loss_uniform = multimodal_function(&final_point_uniform);

        // 4. Gaussian noise injection
        let gaussian_noise_scheduler = NoiseInjectionScheduler::new(
            ConstantScheduler::new(0.01),
            NoiseDistribution::Gaussian {
                mean: 0.0,
                std_dev: 0.003,
            },
            0.001,
        );
        let (final_point_gaussian, _) =
            optimize(&initial_point, gaussian_noise_scheduler, iterations)?;
        let final_loss_gaussian = multimodal_function(&final_point_gaussian);

        // 5. Decaying noise injection
        let decaying_noise_scheduler = NoiseInjectionScheduler::new(
            ConstantScheduler::new(0.01),
            NoiseDistribution::Decaying {
                initial_scale: 0.01,
                final_scale: 0.0001,
                decay_steps: iterations,
            },
            0.001,
        );
        let (final_point_decaying, _) =
            optimize(&initial_point, decaying_noise_scheduler, iterations)?;
        let final_loss_decaying = multimodal_function(&final_point_decaying);

        // Print results
        println!("\nResults after {} iterations:", iterations);
        println!(
            "  Constant LR:      Loss = {:.6}, Point = ({:.4}, {:.4})",
            final_loss_constant, final_point_constant[0], final_point_constant[1]
        );
        println!(
            "  Exponential Decay: Loss = {:.6}, Point = ({:.4}, {:.4})",
            final_loss_exp, final_point_exp[0], final_point_exp[1]
        );
        println!(
            "  Uniform Noise:     Loss = {:.6}, Point = ({:.4}, {:.4})",
            final_loss_uniform, final_point_uniform[0], final_point_uniform[1]
        );
        println!(
            "  Gaussian Noise:    Loss = {:.6}, Point = ({:.4}, {:.4})",
            final_loss_gaussian, final_point_gaussian[0], final_point_gaussian[1]
        );
        println!(
            "  Decaying Noise:    Loss = {:.6}, Point = ({:.4}, {:.4})",
            final_loss_decaying, final_point_decaying[0], final_point_decaying[1]
        );

        // Identify the best optimizer for this run
        let losses = [
            final_loss_constant,
            final_loss_exp,
            final_loss_uniform,
            final_loss_gaussian,
            final_loss_decaying,
        ];

        let optimizer_names = [
            "Constant LR",
            "Exponential Decay",
            "Uniform Noise",
            "Gaussian Noise",
            "Decaying Noise",
        ];

        let best_idx = losses
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        println!(
            "\n  Best optimizer: {} (Loss = {:.6})",
            optimizer_names[best_idx], losses[best_idx]
        );
    }

    println!("\nConclusion:");
    println!("Adding noise to the learning rate can help escape local minima,");
    println!("especially in functions with complex landscapes. The decaying");
    println!("noise strategy often performs well because it allows for more");
    println!("exploration early in training and more exploitation later.");

    Ok(())
}
