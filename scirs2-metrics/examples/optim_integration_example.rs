//! Example: Integration with scirs2-optim for metric-based optimization
//!
//! This example demonstrates how scirs2-metrics can be integrated with
//! external optimizers and schedulers without circular dependencies.

use ndarray::array;
use scirs2_metrics::classification::{accuracy_score, f1_score, precision_score, recall_score};
use scirs2_metrics::integration::optim::{MetricOptimizer, SchedulerConfig};
use scirs2_metrics::regression::{mean_squared_error, r2_score};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== SciRS2 Metrics + Optimization Integration Example ===\n");

    // Example 1: Classification metric optimization
    classification_optimization_example()?;

    // Example 2: Regression metric optimization
    regression_optimization_example()?;

    // Example 3: Multi-metric tracking
    multi_metric_tracking_example()?;

    // Example 4: External scheduler integration pattern
    external_scheduler_pattern_example()?;

    Ok(())
}

fn classification_optimization_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Example 1: Classification Metric Optimization");
    println!("================================================");

    // Ground truth labels
    let y_true = array![1, 0, 1, 1, 0, 1, 0, 0, 1, 1];

    // Set up metric optimizer for F1-score (maximize)
    let mut f1_optimizer = MetricOptimizer::<f64>::new("f1_score", true);

    // Simulate training epochs with different predictions
    let training_epochs = vec![
        array![1, 0, 0, 1, 0, 1, 1, 0, 1, 0], // Epoch 1
        array![1, 0, 1, 1, 0, 1, 0, 0, 1, 0], // Epoch 2
        array![1, 0, 1, 1, 0, 1, 0, 0, 1, 1], // Epoch 3
        array![1, 0, 1, 1, 1, 1, 0, 0, 1, 1], // Epoch 4
        array![1, 0, 1, 1, 0, 1, 0, 0, 1, 1], // Epoch 5 (back to best)
    ];

    println!("Training progress:");
    for (epoch, y_pred) in training_epochs.iter().enumerate() {
        let f1 = f1_score(&y_true, y_pred, 1)?;
        let accuracy = accuracy_score(&y_true, y_pred)?;
        let precision = precision_score(&y_true, y_pred, 1)?;
        let recall = recall_score(&y_true, y_pred, 1)?;

        let is_improvement = f1_optimizer.is_improvement(f1);
        f1_optimizer.add_value(f1);

        // Track additional metrics
        f1_optimizer.add_additional_value("accuracy", accuracy);
        f1_optimizer.add_additional_value("precision", precision);
        f1_optimizer.add_additional_value("recall", recall);

        println!(
            "  Epoch {}: F1={:.3}, Acc={:.3}, P={:.3}, R={:.3} {}",
            epoch + 1,
            f1,
            accuracy,
            precision,
            recall,
            if is_improvement { "✓ IMPROVED" } else { "" }
        );
    }

    println!("\nOptimization Summary:");
    println!("  Best F1-score: {:.3}", f1_optimizer.best_value().unwrap());
    println!("  Total epochs: {}", f1_optimizer.history().len());

    // Create scheduler configuration for external optimizer
    let scheduler_config = f1_optimizer.create_scheduler_config(
        0.01, // initial_lr
        0.5,  // factor
        3,    // patience
        1e-6, // min_lr
    );

    println!(
        "  Scheduler config: LR={}, Factor={}, Patience={}, Mode={}",
        scheduler_config.initial_lr,
        scheduler_config.factor,
        scheduler_config.patience,
        scheduler_config.mode
    );

    println!();
    Ok(())
}

fn regression_optimization_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("📊 Example 2: Regression Metric Optimization");
    println!("=============================================");

    // Ground truth values
    let y_true = array![1.0, 2.0, 3.0, 4.0, 5.0];

    // Set up metric optimizer for MSE (minimize)
    let mut mse_optimizer = MetricOptimizer::<f64>::new("mse", false);

    // Simulate training with improving predictions
    let training_predictions = vec![
        array![1.5, 2.8, 2.2, 4.5, 4.8], // Epoch 1: Large errors
        array![1.2, 2.3, 2.8, 4.2, 4.9], // Epoch 2: Better
        array![1.1, 2.1, 2.9, 4.1, 5.0], // Epoch 3: Even better
        array![1.0, 2.0, 3.0, 4.0, 5.1], // Epoch 4: Best
        array![1.1, 2.0, 3.1, 4.0, 5.0], // Epoch 5: Slight regression
    ];

    println!("Training progress:");
    for (epoch, y_pred) in training_predictions.iter().enumerate() {
        let mse = mean_squared_error(&y_true, y_pred)?;
        let r2 = r2_score(&y_true, y_pred)?;

        let is_improvement = mse_optimizer.is_improvement(mse);
        mse_optimizer.add_value(mse);
        mse_optimizer.add_additional_value("r2", r2);

        println!(
            "  Epoch {}: MSE={:.3}, R²={:.3} {}",
            epoch + 1,
            mse,
            r2,
            if is_improvement { "✓ IMPROVED" } else { "" }
        );
    }

    println!("\nOptimization Summary:");
    println!("  Best MSE: {:.3}", mse_optimizer.best_value().unwrap());

    // Show R² progression
    let r2_history = mse_optimizer.additional_metric_history("r2").unwrap();
    println!(
        "  R² progression: {:?}",
        r2_history
            .iter()
            .map(|&x| format!("{:.3}", x))
            .collect::<Vec<_>>()
    );

    println!();
    Ok(())
}

fn multi_metric_tracking_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("📈 Example 3: Multi-Metric Tracking");
    println!("===================================");

    let y_true = array![1, 0, 1, 1, 0, 1, 0, 0, 1, 1];
    let y_pred = array![1, 0, 1, 1, 0, 1, 0, 0, 1, 1]; // Perfect predictions

    // Primary optimizer tracks F1-score
    let mut primary_optimizer = MetricOptimizer::<f64>::new("f1_score", true);

    // Calculate all metrics
    let f1 = f1_score(&y_true, &y_pred, 1)?;
    let accuracy = accuracy_score(&y_true, &y_pred)?;
    let precision = precision_score(&y_true, &y_pred, 1)?;
    let recall = recall_score(&y_true, &y_pred, 1)?;

    // Add primary metric
    primary_optimizer.add_value(f1);

    // Track additional metrics
    primary_optimizer.add_additional_value("accuracy", accuracy);
    primary_optimizer.add_additional_value("precision", precision);
    primary_optimizer.add_additional_value("recall", recall);

    println!("Metrics tracked:");
    println!("  Primary (F1-score): {:.3}", f1);
    println!("  Additional metrics:");

    for metric_name in ["accuracy", "precision", "recall"] {
        if let Some(history) = primary_optimizer.additional_metric_history(metric_name) {
            println!("    {}: {:.3}", metric_name, history[0]);
        }
    }

    // Create multiple scheduler configs for different optimization strategies
    let aggressive_config = primary_optimizer.create_scheduler_config(0.1, 0.1, 1, 1e-5);
    let conservative_config = primary_optimizer.create_scheduler_config(0.01, 0.8, 10, 1e-7);

    println!("\nScheduler strategies:");
    println!(
        "  Aggressive: LR={}, Factor={}, Patience={}",
        aggressive_config.initial_lr, aggressive_config.factor, aggressive_config.patience
    );
    println!(
        "  Conservative: LR={}, Factor={}, Patience={}",
        conservative_config.initial_lr, conservative_config.factor, conservative_config.patience
    );

    println!();
    Ok(())
}

fn external_scheduler_pattern_example() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔗 Example 4: External Scheduler Integration Pattern");
    println!("====================================================");

    // This demonstrates how scirs2-metrics would integrate with scirs2-optim
    // The pattern avoids circular dependencies by using configuration objects

    let mut metric_optimizer = MetricOptimizer::<f64>::new("validation_loss", false);

    // Simulate validation losses over epochs
    let validation_losses = [0.8, 0.6, 0.5, 0.5, 0.51, 0.52, 0.48, 0.47];

    println!("Validation loss progression:");
    let mut lr = 0.01; // Starting learning rate
    let mut patience_counter = 0;
    let patience = 3;
    let lr_factor = 0.5;

    for (epoch, &loss) in validation_losses.iter().enumerate() {
        let was_improvement = metric_optimizer.is_improvement(loss);
        metric_optimizer.add_value(loss);

        if !was_improvement {
            patience_counter += 1;
        } else {
            patience_counter = 0;
        }

        // Simulate learning rate scheduling
        let lr_action = if patience_counter >= patience {
            lr *= lr_factor;
            patience_counter = 0;
            " [LR REDUCED]"
        } else {
            ""
        };

        println!(
            "  Epoch {}: Loss={:.3}, LR={:.4}, Patience={}/{}{}",
            epoch + 1,
            loss,
            lr,
            patience_counter,
            patience,
            if was_improvement { " ✓" } else { lr_action }
        );
    }

    // Show how to create external scheduler config
    let final_config = metric_optimizer.create_scheduler_config(lr, lr_factor, patience, 1e-6);

    println!("\nFinal scheduler configuration:");
    println!("  Current LR: {:.4}", final_config.initial_lr);
    println!("  Reduction factor: {}", final_config.factor);
    println!("  Patience: {}", final_config.patience);
    println!("  Mode: {} (lower loss is better)", final_config.mode);
    println!(
        "  Best loss achieved: {:.3}",
        metric_optimizer.best_value().unwrap()
    );

    println!("\n💡 Integration Note:");
    println!("   This configuration can be passed to scirs2-optim schedulers:");
    println!("   ```rust");
    println!("   let (lr, factor, patience, min_lr, mode) = config.as_tuple();");
    println!("   let scheduler = ReduceOnPlateau::new(lr, factor, patience, min_lr);");
    println!("   scheduler.set_mode(mode);");
    println!("   ```");

    Ok(())
}

/// Example of how external libraries would implement the integration
/// This shows what scirs2-optim would do to integrate with scirs2-metrics
mod external_integration_example {
    use super::*;
    use scirs2_metrics::integration::optim::{MetricSchedulerTrait, OptimizationMode};

    /// Mock external scheduler that implements the MetricSchedulerTrait
    pub struct ExternalReduceOnPlateau {
        current_lr: f64,
        factor: f64,
        patience: usize,
        min_lr: f64,
        mode: OptimizationMode,
        best_metric: Option<f64>,
        patience_counter: usize,
        threshold: f64,
    }

    impl ExternalReduceOnPlateau {
        pub fn from_config(config: &SchedulerConfig<f64>) -> Self {
            Self {
                current_lr: config.initial_lr,
                factor: config.factor,
                patience: config.patience,
                min_lr: config.min_lr,
                mode: config.mode,
                best_metric: None,
                patience_counter: 0,
                threshold: 1e-4,
            }
        }
    }

    impl MetricSchedulerTrait<f64> for ExternalReduceOnPlateau {
        fn step_with_metric(&mut self, metric: f64) -> f64 {
            let is_improvement = match self.best_metric {
                None => true,
                Some(best) => match self.mode {
                    OptimizationMode::Minimize => metric < best * (1.0 - self.threshold),
                    OptimizationMode::Maximize => metric > best * (1.0 + self.threshold),
                },
            };

            if is_improvement {
                self.best_metric = Some(metric);
                self.patience_counter = 0;
            } else {
                self.patience_counter += 1;

                if self.patience_counter >= self.patience {
                    self.current_lr = (self.current_lr * self.factor).max(self.min_lr);
                    self.patience_counter = 0;
                }
            }

            self.current_lr
        }

        fn get_learning_rate(&self) -> f64 {
            self.current_lr
        }

        fn reset(&mut self) {
            self.best_metric = None;
            self.patience_counter = 0;
        }

        fn set_mode(&mut self, mode: OptimizationMode) {
            self.mode = mode;
        }
    }

    pub fn demonstrate_external_integration() {
        println!("\n🔧 External Integration Demo:");
        println!("=============================");

        // Create metric optimizer
        let metric_optimizer = MetricOptimizer::<f64>::new("accuracy", true);

        // Create scheduler config
        let config = metric_optimizer.create_scheduler_config(0.01, 0.5, 2, 1e-6);

        // External library creates scheduler from config
        let mut external_scheduler = ExternalReduceOnPlateau::from_config(&config);

        println!("External scheduler created with:");
        println!(
            "  Initial LR: {:.3}",
            external_scheduler.get_learning_rate()
        );
        println!("  Mode: {}", config.mode);

        // Simulate training
        let metrics = [0.7, 0.8, 0.82, 0.81, 0.80]; // accuracy values

        for (step, &metric) in metrics.iter().enumerate() {
            let new_lr = external_scheduler.step_with_metric(metric);
            println!(
                "  Step {}: Accuracy={:.3}, LR={:.4}",
                step + 1,
                metric,
                new_lr
            );
        }
    }
}

/// Run the external integration example
#[allow(dead_code)]
fn external_integration_demo() {
    external_integration_example::demonstrate_external_integration();
}
