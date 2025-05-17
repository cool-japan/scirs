//! Example of learning rate scheduling with metrics
//!
//! This example shows how to use learning rate scheduling based on metrics.
//! To run this example, enable the 'optim_integration' feature:
//!
//! ```bash
//! cargo run --example lr_scheduling --features optim_integration
//! ```

use std::error::Error;

#[cfg(feature = "optim_integration")]
use scirs2_metrics::integration::optim::MetricScheduler;

fn main() -> Result<(), Box<dyn Error>> {
    #[cfg(not(feature = "optim_integration"))]
    {
        println!("This example requires the 'optim_integration' feature to be enabled.");
        println!("Run with: cargo run --example lr_scheduling --features optim_integration");
        Ok(())
    }

    #[cfg(feature = "optim_integration")]
    {
        println!("Learning Rate Scheduling Example");
        println!("-------------------------------");

        // Create a metric scheduler
        let mut scheduler = MetricScheduler::new(
            0.1,        // Initial learning rate
            0.5,        // Factor to reduce learning rate (0.5 = halve it)
            2,          // Patience - number of epochs with no improvement before reducing
            0.001,      // Minimum learning rate
            "val_loss", // Metric name
            false,      // Maximize? No, we want to minimize loss
        );

        // Simulate training for 20 epochs
        println!("Simulating training loop with learning rate scheduling...");
        println!("Epoch | Validation Loss | Learning Rate");
        println!("--------------------------------------");

        // Initial validation loss
        let mut val_loss = 1.0;

        for epoch in 0..20 {
            // Simulate training for this epoch
            let (train_loss, new_val_loss) =
                simulate_epoch_training(val_loss, scheduler.get_learning_rate());

            // Update validation loss
            val_loss = new_val_loss;

            // Update learning rate scheduler
            let new_lr = scheduler.step_with_metric(val_loss);

            // Print status
            println!("{:5} | {:14.6} | {:12.6}", epoch + 1, val_loss, new_lr);
        }

        // Print history
        println!("\nLearning rate history:");
        for (i, &lr) in scheduler.history().iter().enumerate() {
            println!("Change {}: {:.6}", i, lr);
        }

        // Best metric value
        println!(
            "\nBest validation loss: {:.6}",
            scheduler.best_metric().unwrap_or(f64::INFINITY)
        );

        // Using the scheduler adapter with scirs2-optim
        #[cfg(feature = "optim_integration")]
        {
            use scirs2_metrics::integration::optim::ReduceOnPlateauAdapter;
            use scirs2_optim::optimizers::{Optimizer, SGD};

            println!("\nUsing scheduler with scirs2-optim SGD optimizer:");

            // Create a scheduler adapter
            let mut scheduler_adapter = ReduceOnPlateauAdapter::new(
                0.1,        // Initial learning rate
                0.5,        // Factor
                2,          // Patience
                0.001,      // Minimum learning rate
                "val_loss", // Metric name
                false,      // Minimize
            );

            // Create an SGD optimizer
            let mut optimizer = SGD::new(0.1);

            // Simulate training
            let mut val_loss = 1.0;

            println!("Epoch | Validation Loss | Learning Rate");
            println!("--------------------------------------");

            for epoch in 0..10 {
                // Apply scheduler to optimizer
                scheduler_adapter.apply_to(&mut optimizer);

                // Get current learning rate
                let current_lr = optimizer.get_learning_rate();

                // Simulate training
                let (train_loss, new_val_loss) = simulate_epoch_training(val_loss, current_lr);
                val_loss = new_val_loss;

                // Update scheduler
                let new_lr = scheduler_adapter.step_with_metric(val_loss);

                // Print status
                println!("{:5} | {:14.6} | {:12.6}", epoch + 1, val_loss, new_lr);
            }
        }

        Ok(())
    }
}

/// Simulate training for one epoch
#[allow(dead_code)]
fn simulate_epoch_training(current_val_loss: f64, learning_rate: f64) -> (f64, f64) {
    // Simulate training loss
    let train_loss = current_val_loss * (0.8 + rand::random::<f64>() * 0.2);

    // Calculate new validation loss
    let base_improvement = if current_val_loss > 0.2 {
        0.05 + rand::random::<f64>() * 0.05
    } else {
        0.01 + rand::random::<f64>() * 0.01
    };

    // Learning rate effect:
    // - Very small learning rates improve slowly
    // - Mid-range learning rates improve well
    // - Very large learning rates can cause instability
    let lr_factor = if learning_rate < 0.01 {
        learning_rate / 0.01 // Slower improvement for very small LRs
    } else if learning_rate > 0.2 {
        1.0 - ((learning_rate - 0.2) / 0.8) // Worse improvement for very large LRs
    } else {
        1.0 // Good improvement for mid-range LRs
    };

    // Calculate improvement
    let improvement = base_improvement * lr_factor;

    // Calculate new validation loss
    let new_val_loss = (current_val_loss - improvement).max(0.01);

    (train_loss, new_val_loss)
}
