//! Example demonstrating cyclic learning rate scheduling

use ndarray::Array1;
use scirs2_optim::{
    optimizers::{Optimizer, SGD},
    schedulers::{CyclicLR, LearningRateScheduler},
};

fn simulate_loss(x: &Array1<f64>) -> f64 {
    // Rosenbrock function
    let a = 1.0;
    let b = 100.0;
    let n = x.len();
    let mut loss = 0.0;

    for i in 0..n - 1 {
        let term1 = b * (x[i + 1] - x[i] * x[i]).powi(2);
        let term2 = (a - x[i]).powi(2);
        loss += term1 + term2;
    }

    loss
}

fn compute_gradient(x: &Array1<f64>) -> Array1<f64> {
    // Numerical gradient computation
    let mut grad = Array1::zeros(x.len());
    let epsilon = 1e-8;

    for i in 0..x.len() {
        let mut x_plus = x.clone();
        let mut x_minus = x.clone();

        x_plus[i] += epsilon;
        x_minus[i] -= epsilon;

        grad[i] = (simulate_loss(&x_plus) - simulate_loss(&x_minus)) / (2.0 * epsilon);
    }

    grad
}

fn main() {
    // Initialize parameters
    let params = Array1::from_vec(vec![-1.0, 1.5, 2.0, -2.5]);

    // Create optimizer with initial learning rate
    let initial_lr = 0.0001;
    let mut optimizer = SGD::new(initial_lr);

    // Create cyclic learning rate schedulers with different modes
    println!("Demonstrating different cyclic learning rate modes:\n");

    // 1. Triangular mode
    println!("Triangular Mode:");
    let mut triangular_scheduler = CyclicLR::triangular(0.0001, 0.001, 50);
    run_optimization(
        params.clone(),
        &mut optimizer,
        &mut triangular_scheduler,
        200,
    );

    // 2. Triangular2 mode (halved amplitude each cycle)
    println!("\nTriangular2 Mode:");
    let mut triangular2_scheduler = CyclicLR::triangular2(0.0001, 0.001, 50);
    run_optimization(
        params.clone(),
        &mut optimizer,
        &mut triangular2_scheduler,
        200,
    );

    // 3. Exponential range mode
    println!("\nExponential Range Mode:");
    let mut exp_scheduler = CyclicLR::exp_range(0.0001, 0.001, 50, 0.999);
    run_optimization(params.clone(), &mut optimizer, &mut exp_scheduler, 200);

    // 4. Custom scale function
    println!("\nCustom Scale Function (Sine Wave):");
    let mut custom_scheduler =
        CyclicLR::triangular(0.0001, 0.001, 50).with_scale_fn(|step, half_cycle, _, _| {
            let position = (step % (2 * half_cycle)) as f64 / (2.0 * half_cycle as f64);
            (position * std::f64::consts::PI * 2.0).sin().abs()
        });
    run_optimization(params.clone(), &mut optimizer, &mut custom_scheduler, 200);
}

fn run_optimization<LR: LearningRateScheduler<f64>>(
    mut params: Array1<f64>,
    optimizer: &mut SGD<f64>,
    scheduler: &mut LR,
    steps: usize,
) {
    let mut loss_history = Vec::new();
    let mut lr_history = Vec::new();

    for step in 0..steps {
        // Compute gradient
        let gradient = compute_gradient(&params);

        // Get current learning rate from scheduler
        let lr = scheduler.get_learning_rate();
        scheduler.apply_to::<ndarray::Ix1, _>(optimizer);

        // Update parameters
        params = optimizer.step(&params, &gradient).unwrap();

        // Compute loss
        let loss = simulate_loss(&params);

        // Record data every 10 steps
        if step % 10 == 0 {
            loss_history.push(loss);
            lr_history.push(lr);

            if step % 50 == 0 {
                println!("Step {:3}: Loss = {:.6}, LR = {:.6}", step, loss, lr);
            }
        }

        // Step scheduler
        scheduler.step();
    }

    // Print summary
    let final_loss = simulate_loss(&params);
    let improvement = (loss_history[0] - final_loss) / loss_history[0] * 100.0;
    println!(
        "Final: Loss = {:.6}, Improvement = {:.2}%",
        final_loss, improvement
    );

    // Print learning rate range
    let min_lr = lr_history.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_lr = lr_history.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    println!("LR range: {:.6} - {:.6}", min_lr, max_lr);
}
