//! Example demonstrating the custom scheduler framework
use ndarray::Array1;
use scirs2_optim::{
    optimizers::{Optimizer, SGD},
    schedulers::{CombinedScheduler, CustomScheduler, LearningRateScheduler, SchedulerBuilder},
};
use std::error::Error;

/// Simple quadratic loss function for demonstration
fn quadratic_loss(x: &Array1<f64>) -> f64 {
    x.iter().map(|&xi| xi * xi).sum()
}

/// Compute gradient of quadratic loss
fn quadratic_gradient(x: &Array1<f64>) -> Array1<f64> {
    x * 2.0
}

/// Test a scheduler and print its learning rate schedule
fn test_scheduler<S: LearningRateScheduler<f64>>(
    name: &str,
    mut scheduler: S,
    steps: usize,
    initial_params: &Array1<f64>,
) -> Result<Array1<f64>, Box<dyn Error>> {
    println!("\n{}", "=".repeat(50));
    println!("Testing scheduler: {}", name);
    println!("{}", "=".repeat(50));

    // Print the learning rate schedule
    println!("Learning rate schedule:");
    println!("Step | Learning Rate");
    println!("-----|-------------");

    let mut params = initial_params.clone();
    let mut optimizer = SGD::<f64>::new(scheduler.get_learning_rate());

    for step in 0..steps {
        // Print every few steps
        if step % (steps / 10).max(1) == 0 || step == steps - 1 {
            println!("{:4} | {:13.8}", step, scheduler.get_learning_rate());
        }

        // Update optimizer with current learning rate
        <SGD<f64> as Optimizer<f64, ndarray::Ix1>>::set_learning_rate(
            &mut optimizer,
            scheduler.get_learning_rate(),
        );

        // Compute gradient and update parameters
        let gradient = quadratic_gradient(&params);
        params = optimizer.step(&params, &gradient)?;

        // Step the scheduler
        if step < steps - 1 {
            scheduler.step();
        }
    }

    println!("\nFinal loss: {:.8}", quadratic_loss(&params));
    println!("Final parameters: {:?}", params);

    Ok(params)
}

fn main() -> Result<(), Box<dyn Error>> {
    // Initial parameters
    let initial_params = Array1::from_vec(vec![5.0, -3.0, 2.0, -4.0]);
    let initial_loss = quadratic_loss(&initial_params);
    println!("Initial loss: {:.8}", initial_loss);
    println!("Initial parameters: {:?}", initial_params);

    // The total number of training steps
    let steps = 100;

    // 1. Using a custom scheduler with a simple function
    let custom_func_scheduler = CustomScheduler::new(0.1, |step| {
        // A custom function that scales learning rate based on step
        let step = step as f64;
        let warmup_steps = 10.0;
        let decay_rate: f64 = 0.9;

        if step < warmup_steps {
            // Linear warmup
            0.01 + (0.1 - 0.01) * (step / warmup_steps)
        } else {
            // Exponential decay after warmup
            0.1 * decay_rate.powf((step - warmup_steps) / 10.0)
        }
    });

    let params1 = test_scheduler(
        "Custom Function Scheduler",
        custom_func_scheduler,
        steps,
        &initial_params,
    )?;

    // 2. Using the scheduler builder
    let step_scheduler = SchedulerBuilder::new(0.1).step_decay(20, 0.5);
    let params2 = test_scheduler(
        "Step Decay Scheduler (from builder)",
        step_scheduler,
        steps,
        &initial_params,
    )?;

    // 3. Using a combined scheduler
    let exp_scheduler = CustomScheduler::new(0.1, |step| 0.1 * 0.95f64.powi((step / 5) as i32));

    let cosine_scheduler = SchedulerBuilder::new(0.1).cosine_annealing(steps, 0.001);

    let combined_scheduler = CombinedScheduler::new(
        exp_scheduler,
        cosine_scheduler,
        // Take the minimum of both schedulers
        |lr1, lr2| lr1.min(lr2),
    );

    let params3 = test_scheduler(
        "Combined Scheduler (Exp + Cosine)",
        combined_scheduler,
        steps,
        &initial_params,
    )?;

    // 4. Create a custom "staircase" scheduler
    let staircase_scheduler = CustomScheduler::new(0.1, move |step| {
        let stair_size = 20;
        let num_stairs = (step / stair_size) as i32;

        // Learning rate decreases by half every stair
        0.1 * 0.5f64.powi(num_stairs)
    });

    let params4 = test_scheduler(
        "Staircase Scheduler",
        staircase_scheduler,
        steps,
        &initial_params,
    )?;

    // 5. Create a noisy scheduler
    let mut rng = rand::rng();
    let noisy_scheduler = CustomScheduler::new(0.1, move |step| {
        use rand::Rng;

        // Exponential decay with noise
        let base_lr = 0.1 * 0.95f64.powi((step / 10) as i32);
        let noise = rng.random_range(-0.01..0.01); // Add noise in range [-0.01, 0.01]
        (base_lr + noise).max(0.001) // Ensure LR doesn't go below 0.001
    });

    let params5 = test_scheduler("Noisy Scheduler", noisy_scheduler, steps, &initial_params)?;

    // Compare final losses
    println!("\n{}", "=".repeat(50));
    println!("Final Loss Comparison");
    println!("{}", "=".repeat(50));
    println!("Initial loss: {:.8}", initial_loss);
    println!("Custom Function Scheduler: {:.8}", quadratic_loss(&params1));
    println!("Step Decay Scheduler: {:.8}", quadratic_loss(&params2));
    println!("Combined Scheduler: {:.8}", quadratic_loss(&params3));
    println!("Staircase Scheduler: {:.8}", quadratic_loss(&params4));
    println!("Noisy Scheduler: {:.8}", quadratic_loss(&params5));

    Ok(())
}
