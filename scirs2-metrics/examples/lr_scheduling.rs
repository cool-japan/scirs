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
use scirs2_metrics::integration::optim::MetricLRScheduler;

#[allow(dead_code)]
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
        let mut scheduler = MetricLRScheduler::new(
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
            let (_train_loss, new_val_loss) =
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
            println!("Change {i}: {lr:.6}");
        }

        // Best metric value
        println!(
            "\nBest validation loss: {:.6}",
            scheduler.best_metric().unwrap_or(f64::INFINITY)
        );

        // Demonstrate scheduler configuration for external optimizers
        #[cfg(feature = "optim_integration")]
        {
            use scirs2_metrics::integration::optim::{MetricOptimizer, SchedulerConfig};

            println!("\nDemonstrating external optimizer integration:");

            // Create a metric optimizer for tracking accuracy
            let mut metric_optimizer = MetricOptimizer::new("accuracy", true);

            // Add some metric values
            metric_optimizer.add_value(0.85);
            metric_optimizer.add_value(0.87);
            metric_optimizer.add_value(0.89);

            // Create scheduler configuration for external use
            let scheduler_config: SchedulerConfig<f64> = metric_optimizer.create_scheduler_config(
                0.01, // Initial learning rate
                0.8,  // Factor to reduce by
                3,    // Patience
                1e-6, // Minimum learning rate
            );

            println!("Scheduler configuration created:");
            println!("  Initial LR: {}", scheduler_config.initial_lr);
            println!("  Factor: {}", scheduler_config.factor);
            println!("  Patience: {}", scheduler_config.patience);
            println!("  Min LR: {}", scheduler_config.min_lr);
            println!("  Mode: {}", scheduler_config.mode);
            println!("  Metric: {}", scheduler_config.metric_name);

            // Show how to extract configuration values
            let (initial_lr, factor, patience, min_lr, mode) = scheduler_config.as_tuple();
            println!("\nConfiguration as tuple:");
            println!("  ({initial_lr}, {factor}, {patience}, {min_lr}, {mode})");

            println!("\nThis configuration can be used to create external schedulers");
            println!("from scirs2-optim or other optimization libraries.");
        }

        Ok(())
    }
}

/// Simulate training for one epoch
#[allow(dead_code)]
fn simulate_epoch_training(current_val_loss: f64, learningrate: f64) -> (f64, f64) {
    // Simulate training loss
    let train_loss = current_val_loss * (0.8 + scirs2_core::random::random::<f64>() * 0.2);

    // Calculate new validation loss
    let base_improvement = if current_val_loss > 0.2 {
        0.05 + scirs2_core::random::random::<f64>() * 0.05
    } else {
        0.01 + scirs2_core::random::random::<f64>() * 0.01
    };

    // Learning rate effect:
    // - Very small learning rates improve slowly
    // - Mid-range learning rates improve well
    // - Very large learning rates can cause instability
    let lr_factor = if learningrate < 0.01 {
        learningrate / 0.01 // Slower improvement for very small LRs
    } else if learningrate > 0.2 {
        1.0 - ((learningrate - 0.2) / 0.8) // Worse improvement for very large LRs
    } else {
        1.0 // Good improvement for mid-range LRs
    };

    // Calculate improvement
    let improvement = base_improvement * lr_factor;

    // Calculate new validation loss
    let new_val_loss = (current_val_loss - improvement).max(0.01);

    (train_loss, new_val_loss)
}
