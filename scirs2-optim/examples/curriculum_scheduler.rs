//! Example demonstrating curriculum learning with the CurriculumScheduler
//!
//! This example shows how curriculum learning can be used to train a model
//! by gradually increasing task difficulty and adjusting the learning rate.

use ndarray::{Array1, Array2};
use rand::prelude::*;
use scirs2_optim::{
    optimizers::{Optimizer, SGD},
    schedulers::{CurriculumScheduler, CurriculumStage, LearningRateScheduler, TransitionStrategy},
};
use std::error::Error;

/// Generates a synthetic regression dataset with varying difficulty levels
fn generate_dataset(
    difficulty: usize,
    n_samples: usize,
    n_features: usize,
) -> (Array2<f64>, Array1<f64>) {
    let mut rng = rand::rng();

    // Create features matrix (X)
    let mut x = Array2::<f64>::zeros((n_samples, n_features));
    for i in 0..n_samples {
        for j in 0..n_features {
            x[[i, j]] = rng.random_range(-1.0..1.0);
        }
    }

    // Create true weights with varying complexity based on difficulty
    let mut true_weights = Array1::<f64>::zeros(n_features);
    for j in 0..n_features {
        // Higher difficulty = more non-zero weights = more complex model
        if j < difficulty {
            true_weights[j] = rng.random_range(-1.0..1.0);
        }
    }

    // Create target values (y = X * weights + noise)
    let mut y = Array1::<f64>::zeros(n_samples);
    for i in 0..n_samples {
        for j in 0..n_features {
            y[i] += x[[i, j]] * true_weights[j];
        }
        // Add noise (more noise for harder problems)
        let noise_scale = 0.01 * (1.0 + (difficulty as f64) / 5.0);
        y[i] += rng.random_range(-noise_scale..noise_scale);
    }

    (x, y)
}

/// Compute mean squared error for regression
fn compute_mse(predictions: &Array1<f64>, targets: &Array1<f64>) -> f64 {
    let errors = predictions - targets;
    let squared_errors = errors.mapv(|x| x * x);
    squared_errors.mean().unwrap()
}

/// Trains a model on a curriculum of tasks with increasing difficulty
fn train_with_curriculum(
    curriculum_scheduler: &mut CurriculumScheduler<f64>,
    epochs_per_stage: usize,
) -> Result<Vec<(usize, f64, f64)>, Box<dyn Error>> {
    let mut rng = rand::rng();
    let mut training_log = Vec::new();

    // Initialize optimizer with initial learning rate
    let mut optimizer = SGD::<f64>::new(curriculum_scheduler.get_learning_rate());

    // Track current stage
    let mut current_stage = 0;

    while !curriculum_scheduler.completed() {
        // Get the current stage information before any mutations
        let current_lr = curriculum_scheduler.get_learning_rate();
        let stage_description = curriculum_scheduler.current_stage().description.clone();
        let difficulty = current_stage + 1; // Difficulty increases with each stage

        println!(
            "Training on stage {} (difficulty: {})",
            current_stage + 1,
            difficulty
        );
        if let Some(desc) = &stage_description {
            println!("Description: {}", desc);
        }
        println!("Learning rate: {:.6}", current_lr);

        // Generate a dataset appropriate for this difficulty level
        let n_features = 20;
        let n_samples = 100;
        let (x_train, y_train) = generate_dataset(difficulty, n_samples, n_features);

        // Initialize weights to random values
        let mut weights = Array1::<f64>::zeros(n_features);
        for j in 0..n_features {
            weights[j] = rng.random_range(-0.1..0.1);
        }

        // Train for this stage
        for epoch in 0..epochs_per_stage {
            // Compute predictions
            let mut predictions = Array1::<f64>::zeros(n_samples);
            for i in 0..n_samples {
                for j in 0..n_features {
                    predictions[i] += x_train[[i, j]] * weights[j];
                }
            }

            // Compute loss and gradients
            let loss = compute_mse(&predictions, &y_train);

            // Compute gradients for linear regression
            let mut gradients = Array1::<f64>::zeros(n_features);
            for j in 0..n_features {
                for i in 0..n_samples {
                    gradients[j] +=
                        2.0 * (predictions[i] - y_train[i]) * x_train[[i, j]] / (n_samples as f64);
                }
            }

            // Update learning rate
            let lr = curriculum_scheduler.step();
            <SGD<f64> as Optimizer<f64, ndarray::Ix1>>::set_learning_rate(&mut optimizer, lr);

            // Update weights
            weights = optimizer.step(&weights, &gradients)?;

            // Log results every few epochs
            if epoch % 10 == 0 || epoch == epochs_per_stage - 1 {
                let step = current_stage * epochs_per_stage + epoch;
                let progress = curriculum_scheduler.overall_progress();
                training_log.push((step, loss, lr));

                println!(
                    "  Epoch {}/{}: loss = {:.6}, lr = {:.6}, progress = {:.1}%",
                    epoch + 1,
                    epochs_per_stage,
                    loss,
                    lr,
                    progress * 100.0
                );
            }
        }

        // Check if we've moved to a new stage after the training
        if curriculum_scheduler.get_learning_rate() != current_lr {
            current_stage += 1;
        } else if curriculum_scheduler.transition_strategy() == TransitionStrategy::Manual {
            // Manually advance if using manual transitions
            println!("Manually advancing to next stage");
            curriculum_scheduler.advance_stage();
            current_stage += 1;
        }
    }

    println!("Curriculum training completed!");
    Ok(training_log)
}

// Import necessary traits

fn main() -> Result<(), Box<dyn Error>> {
    println!("Curriculum Learning Scheduler Example");
    println!("====================================");

    // Define a curriculum with three stages of increasing difficulty
    let stages = vec![
        CurriculumStage {
            learning_rate: 0.1,
            duration: 100,
            description: Some("Easy tasks (low dimensionality)".to_string()),
        },
        CurriculumStage {
            learning_rate: 0.05,
            duration: 100,
            description: Some("Medium difficulty tasks".to_string()),
        },
        CurriculumStage {
            learning_rate: 0.01,
            duration: 100,
            description: Some("Hard tasks (high dimensionality)".to_string()),
        },
        CurriculumStage {
            learning_rate: 0.001,
            duration: 100,
            description: Some("Very complex tasks with fine-tuning".to_string()),
        },
    ];

    // Epochs to train per stage
    let epochs_per_stage = 50;

    // Train with immediate transitions
    println!("\n1. Training with immediate transitions between stages");
    let mut immediate_scheduler =
        CurriculumScheduler::new(stages.clone(), TransitionStrategy::Immediate, 0.0001);
    let immediate_log = train_with_curriculum(&mut immediate_scheduler, epochs_per_stage)?;
    println!(
        "Final loss with immediate transitions: {:.6}",
        immediate_log.last().unwrap().1
    );

    // Train with smooth transitions
    println!("\n2. Training with smooth transitions between stages");
    let mut smooth_scheduler = CurriculumScheduler::new(
        stages.clone(),
        TransitionStrategy::Smooth { blend_steps: 20 },
        0.0001,
    );
    let smooth_log = train_with_curriculum(&mut smooth_scheduler, epochs_per_stage)?;
    println!(
        "Final loss with smooth transitions: {:.6}",
        smooth_log.last().unwrap().1
    );

    // Train with manual transitions
    println!("\n3. Training with manual transitions between stages");
    let mut manual_scheduler =
        CurriculumScheduler::new(stages.clone(), TransitionStrategy::Manual, 0.0001);
    let manual_log = train_with_curriculum(&mut manual_scheduler, epochs_per_stage)?;
    println!(
        "Final loss with manual transitions: {:.6}",
        manual_log.last().unwrap().1
    );

    // Compare results
    println!("\nComparison of Different Transition Strategies:");
    println!("---------------------------------------------");
    println!(
        "Immediate transitions: final loss = {:.6}",
        immediate_log.last().unwrap().1
    );
    println!(
        "Smooth transitions:    final loss = {:.6}",
        smooth_log.last().unwrap().1
    );
    println!(
        "Manual transitions:    final loss = {:.6}",
        manual_log.last().unwrap().1
    );

    println!("\nConclusion:");
    println!("Curriculum learning allows training to progress from simpler to");
    println!("more complex tasks, which can lead to better performance and faster");
    println!("convergence. The transition strategy between curriculum stages can");
    println!("have an impact on the final model quality.");

    Ok(())
}
