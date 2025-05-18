//! Example demonstrating one-cycle learning rate policy

use ndarray::Array1;
use scirs2_optim::{
    optimizers::{Optimizer, SGD},
    schedulers::{AnnealStrategy, LearningRateScheduler, OneCycle},
};

/// Simple quadratic loss function for demonstration
fn quadratic_loss(x: &Array1<f64>) -> f64 {
    x.iter().map(|&xi| xi * xi).sum()
}

/// Compute gradient of quadratic loss
fn quadratic_gradient(x: &Array1<f64>) -> Array1<f64> {
    x * 2.0
}

/// Visualize learning rate and momentum over time
fn plot_schedule(initial_lr: f64, max_lr: f64, total_steps: usize, warmup_frac: f64, name: &str) {
    println!("\n{} Schedule:", name);
    println!("Step | Learning Rate | Progress");
    println!("-----|---------------|----------");

    let mut scheduler = OneCycle::new(initial_lr, max_lr, total_steps, warmup_frac);
    for step in 0..=total_steps {
        if step % (total_steps / 20) == 0 {
            let lr = scheduler.get_learning_rate();
            let progress = scheduler.get_percentage_complete();

            println!("{:4} | {:13.6} | {:8.1}%", step, lr, progress * 100.0);
        }
        if step < total_steps {
            scheduler.step();
        }
    }
}

fn main() {
    // Initialize parameters
    let initial_params = Array1::from_vec(vec![5.0, -3.0, 2.0, -4.0]);

    // Total training steps
    let total_steps = 200;

    // Basic one-cycle
    println!("\n{}", "=".repeat(60));
    println!("Running: Basic One-Cycle");
    println!("{}", "=".repeat(60));
    plot_schedule(0.001, 0.01, total_steps, 0.3, "Basic One-Cycle");
    let scheduler = OneCycle::new(0.001, 0.01, total_steps, 0.3);
    run_one_cycle_optimization(initial_params.clone(), scheduler, total_steps);

    // One-cycle with linear anneal
    println!("\n{}", "=".repeat(60));
    println!("Running: One-Cycle with Linear Anneal");
    println!("{}", "=".repeat(60));
    plot_schedule(
        0.001,
        0.01,
        total_steps,
        0.3,
        "One-Cycle with Linear Anneal",
    );
    let scheduler =
        OneCycle::new(0.001, 0.01, total_steps, 0.3).with_anneal_strategy(AnnealStrategy::Linear);
    run_one_cycle_optimization(initial_params.clone(), scheduler, total_steps);

    // One-cycle with momentum
    println!("\n{}", "=".repeat(60));
    println!("Running: One-Cycle with Momentum");
    println!("{}", "=".repeat(60));
    plot_schedule(0.001, 0.01, total_steps, 0.3, "One-Cycle with Momentum");
    let scheduler = OneCycle::new(0.001, 0.01, total_steps, 0.3).with_momentum(0.85, 0.95, 0.9);
    run_one_cycle_optimization(initial_params.clone(), scheduler, total_steps);

    // One-cycle with custom final LR
    println!("\n{}", "=".repeat(60));
    println!("Running: One-Cycle with Custom Final LR");
    println!("{}", "=".repeat(60));
    plot_schedule(
        0.001,
        0.01,
        total_steps,
        0.3,
        "One-Cycle with Custom Final LR",
    );
    let scheduler = OneCycle::new(0.001, 0.01, total_steps, 0.3)
        .with_final_lr(0.0001)
        .with_momentum(0.85, 0.95, 0.9);
    run_one_cycle_optimization(initial_params.clone(), scheduler, total_steps);

    // Demonstrate combining with Adam optimizer
    println!("\n{}", "=".repeat(60));
    println!("One-Cycle with Adam Optimizer");
    println!("{}", "=".repeat(60));

    use scirs2_optim::optimizers::Adam;

    let mut adam = Adam::new(0.001);
    let mut scheduler = OneCycle::new(0.001, 0.01, total_steps, 0.3).with_momentum(0.85, 0.95, 0.9);

    let mut params = initial_params.clone();
    let mut losses = Vec::new();

    for step in 0..total_steps {
        let gradient = quadratic_gradient(&params);

        // Apply scheduler to optimizer
        scheduler.apply_to::<ndarray::Ix1, _>(&mut adam);

        // Update parameters
        params = adam.step(&params, &gradient).unwrap();

        // Calculate loss
        let loss = quadratic_loss(&params);

        if step % 10 == 0 {
            losses.push(loss);
            println!(
                "Step {:3}: Loss = {:.6}, LR = {:.6}",
                step,
                loss,
                scheduler.get_learning_rate()
            );
        }

        scheduler.step();
    }

    println!("\nFinal loss: {:.6}", quadratic_loss(&params));
    println!(
        "Improvement: {:.2}%",
        (losses[0] - losses.last().unwrap()) / losses[0] * 100.0
    );
}

fn run_one_cycle_optimization(
    initial_params: Array1<f64>,
    mut scheduler: OneCycle<f64>,
    total_steps: usize,
) {
    let mut params = initial_params.clone();
    let mut optimizer = SGD::new(0.001);

    // Track metrics
    let mut losses = Vec::new();
    let initial_loss = quadratic_loss(&params);

    println!("\nOptimization Progress:");
    println!("Initial loss: {:.6}", initial_loss);

    for step in 0..total_steps {
        // Compute gradient
        let gradient = quadratic_gradient(&params);

        // Apply learning rate from scheduler
        scheduler.apply_to::<ndarray::Ix1, _>(&mut optimizer);

        // Apply momentum if available
        if let Some(momentum) = scheduler.get_momentum() {
            optimizer.set_momentum(momentum);
        }

        // Update parameters
        params = optimizer.step(&params, &gradient).unwrap();

        // Calculate loss
        let loss = quadratic_loss(&params);

        if step % 20 == 0 {
            losses.push(loss);
            println!(
                "Step {:3}: Loss = {:.6}, LR = {:.6}",
                step,
                loss,
                scheduler.get_learning_rate()
            );
        }

        // Step scheduler
        scheduler.step();
    }

    let final_loss = quadratic_loss(&params);
    losses.push(final_loss);

    println!("Final loss: {:.6}", final_loss);
    println!(
        "Improvement: {:.2}%",
        (initial_loss - final_loss) / initial_loss * 100.0
    );

    // Show parameter convergence
    println!("Final parameters: {:?}", params);
}
