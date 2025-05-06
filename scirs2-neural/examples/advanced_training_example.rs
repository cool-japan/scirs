use ndarray::{Array, IxDyn, ScalarOperand};
use scirs2_core::error::Result;
use scirs2_neural::{
    callbacks::{
        EarlyStopping, GradientClipping, GradientClippingMethod, LearningRateScheduler,
        ModelCheckpoint,
    },
    data::{DataLoader, Dataset, InMemoryDataset},
    layers::{Dense, Dropout, Layer, Sequential},
    losses::{Loss, MSELoss},
    optimizers::{Adam, Optimizer, SGD},
    prelude::*,
    training::{
        GradientAccumulationConfig, GradientAccumulator, MixedPrecisionConfig, Trainer,
        TrainingConfig, ValidationSettings,
    },
};

use num_traits::Float;
use rand::Rng;
use std::fmt::Debug;

// Simple sequential model for regression
fn create_regression_model<F: Float + Debug + ScalarOperand>(
    input_dim: usize,
    hidden_dim: usize,
    output_dim: usize,
) -> Result<Sequential<F>> {
    let mut model = Sequential::new();

    model.add(Dense::<F>::new(
        input_dim,
        hidden_dim,
        Some(Box::new(ReLU::new())),
        None,
        Some("dense1"),
    )?);

    model.add(Dropout::<F>::new(0.2, Some("dropout1"))?);

    model.add(Dense::<F>::new(
        hidden_dim,
        hidden_dim / 2,
        Some(Box::new(ReLU::new())),
        None,
        Some("dense2"),
    )?);

    model.add(Dropout::<F>::new(0.2, Some("dropout2"))?);

    model.add(Dense::<F>::new(
        hidden_dim / 2,
        output_dim,
        None,
        None,
        Some("dense3"),
    )?);

    Ok(model)
}

// Generate synthetic regression dataset
fn generate_regression_dataset<F: Float + Debug + ScalarOperand>(
    n_samples: usize,
    input_dim: usize,
    output_dim: usize,
) -> InMemoryDataset<F> {
    let mut rng = rand::rng();

    // Generate random inputs
    let mut inputs = Vec::with_capacity(n_samples);
    for _ in 0..n_samples {
        let mut input = Vec::with_capacity(input_dim);
        for _ in 0..input_dim {
            input.push(F::from(rng.random_range(0.0..1.0)).unwrap());
        }
        inputs.push(
            Array::<F, _>::from_shape_vec([1, input_dim], input)
                .unwrap()
                .into_dyn(),
        );
    }

    // Generate targets (simple linear relationship plus noise)
    let mut targets = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let input = &inputs[i];
        let mut target = Vec::with_capacity(output_dim);

        for o in 0..output_dim {
            let mut val = F::zero();
            for j in 0..input_dim {
                let weight = F::from(((j + o) % input_dim) as f64 / input_dim as f64).unwrap();
                val = val + input[[0, j]] * weight;
            }

            // Add noise
            let noise = F::from(rng.random_range(-0.1..0.1)).unwrap();
            val = val + noise;

            target.push(val);
        }

        targets.push(
            Array::<F, _>::from_shape_vec([1, output_dim], target)
                .unwrap()
                .into_dyn(),
        );
    }

    InMemoryDataset::new(inputs, targets)
}

// Cosine annealing learning rate scheduler
struct CosineAnnealingScheduler<F: Float + Debug + ScalarOperand> {
    initial_lr: F,
    min_lr: F,
}

impl<F: Float + Debug + ScalarOperand> CosineAnnealingScheduler<F> {
    fn new(initial_lr: F, min_lr: F) -> Self {
        Self { initial_lr, min_lr }
    }
}

impl<F: Float + Debug + ScalarOperand> LearningRateScheduler<F> for CosineAnnealingScheduler<F> {
    fn get_learning_rate(&mut self, progress: f64) -> Result<F> {
        let cosine = (1.0 + (std::f64::consts::PI * progress).cos()) / 2.0;
        let lr = self.min_lr + (self.initial_lr - self.min_lr) * F::from(cosine).unwrap();
        Ok(lr)
    }
}

fn main() -> Result<()> {
    println!("Advanced Training Examples");
    println!("-------------------------");

    // 1. Gradient Accumulation
    println!("\n1. Training with Gradient Accumulation:");

    // Generate synthetic dataset
    let dataset = generate_regression_dataset::<f32>(1000, 10, 2);
    let val_dataset = generate_regression_dataset::<f32>(200, 10, 2);

    // Create model, optimizer, and loss function
    let model = create_regression_model::<f32>(10, 64, 2)?;
    let optimizer = Adam::new(0.001);
    let loss_fn = MSELoss::new();

    // Create gradient accumulation config
    let ga_config = GradientAccumulationConfig {
        accumulation_steps: 4,
        average_gradients: true,
        zero_gradients_after_update: true,
        clip_gradients: true,
        max_gradient_norm: Some(1.0),
        log_gradient_stats: true,
    };

    // Create training config
    let training_config = TrainingConfig {
        batch_size: 32,
        shuffle: true,
        num_workers: 0,
        learning_rate: 0.001,
        epochs: 5,
        verbose: 1,
        validation: Some(ValidationSettings {
            enabled: true,
            validation_split: 0.0, // Use separate validation dataset
            batch_size: 32,
            num_workers: 0,
        }),
        gradient_accumulation: Some(ga_config),
        mixed_precision: None,
    };

    // Create trainer
    let mut trainer = Trainer::new(model, optimizer, loss_fn, training_config);

    // Add callbacks
    trainer.add_callback(Box::new(EarlyStopping::new(
        "val_loss".to_string(),
        0.001,
        5,
        true,
        crate::callbacks::EarlyStoppingMode::Min,
    )));

    trainer.add_callback(Box::new(ModelCheckpoint::new(
        "model_checkpoint".to_string(),
        "val_loss".to_string(),
        crate::callbacks::CheckpointMode::Min,
        true,
    )));

    // Create learning rate scheduler
    let lr_scheduler = CosineAnnealingScheduler::new(0.001_f32, 0.0001_f32);
    trainer.add_callback(Box::new(lr_scheduler));

    // Train model
    println!("\nTraining model with gradient accumulation...");
    let session = trainer.train(&dataset, Some(&val_dataset))?;

    println!("\nTraining completed in {} epochs", session.epochs_trained);
    println!(
        "Final loss: {:.4}",
        session.get_metric("loss").unwrap().last().unwrap()
    );
    println!(
        "Final validation loss: {:.4}",
        session.get_metric("val_loss").unwrap().last().unwrap()
    );

    // 2. Manual Gradient Accumulation
    println!("\n2. Manual Gradient Accumulation:");

    // Create model, optimizer, and loss function
    let mut model = create_regression_model::<f32>(10, 64, 2)?;
    let mut optimizer = Adam::new(0.001);
    let loss_fn = MSELoss::new();

    // Create gradient accumulator
    let mut accumulator = GradientAccumulator::new(GradientAccumulationConfig {
        accumulation_steps: 4,
        average_gradients: true,
        zero_gradients_after_update: true,
        clip_gradients: false,
        max_gradient_norm: None,
        log_gradient_stats: false,
    });

    // Initialize accumulator
    accumulator.initialize(&model)?;

    // Create data loader
    let data_loader = DataLoader::new(&dataset, 32, true, 0);

    println!("\nTraining for 1 epoch with manual gradient accumulation...");

    let mut total_loss = 0.0_f32;
    let mut batch_count = 0;

    // Train for one epoch
    for (batch_idx, (inputs, targets)) in data_loader.enumerate() {
        // Accumulate gradients
        let loss = accumulator.accumulate_gradients(&mut model, &inputs, &targets, &loss_fn)?;

        total_loss += loss;
        batch_count += 1;

        // Print gradient stats if available
        if let Some(stats) = accumulator.get_gradient_stats() {
            println!(
                "Batch {} - Gradient stats: min={:.4}, max={:.4}, mean={:.4}, norm={:.4}",
                batch_idx + 1,
                stats.min,
                stats.max,
                stats.mean,
                stats.norm
            );
        }

        // Update if needed
        if accumulator.should_update() || batch_idx == data_loader.len() - 1 {
            println!(
                "Applying accumulated gradients after {} batches",
                accumulator.get_current_step()
            );
            accumulator.apply_gradients(&mut model, &mut optimizer)?;
        }

        // Early stopping for example
        if batch_idx >= 10 {
            break;
        }
    }

    if batch_count > 0 {
        println!("Average loss: {:.4}", total_loss / batch_count as f32);
    }

    // 3. Mixed Precision (not fully implemented, pseudocode)
    println!("\n3. Mixed Precision Training (Pseudocode):");

    println!(
        "// Create mixed precision config
let mp_config = MixedPrecisionConfig {
    dynamic_loss_scaling: true,
    initial_loss_scale: 65536.0,
    scale_factor: 2.0,
    scale_window: 2000,
    min_loss_scale: 1.0,
    max_loss_scale: 2_f64.powi(24),
    verbose: true,
};

// Create high precision and low precision models
let high_precision_model = create_regression_model::<f32>(10, 64, 2)?;
let low_precision_model = create_regression_model::<f16>(10, 64, 2)?;

// Create mixed precision model
let mut mixed_model = MixedPrecisionModel::new(
    high_precision_model,
    low_precision_model,
    mp_config,
)?;

// Create optimizer and loss function
let mut optimizer = Adam::new(0.001);
let loss_fn = MSELoss::new();

// Train for one epoch
mixed_model.train_epoch(
    &mut optimizer,
    &dataset,
    &loss_fn,
    32,
    true,
)?;"
    );

    // 4. Gradient Clipping
    println!("\n4. Gradient Clipping:");

    // Create model, optimizer, and loss function
    let model = create_regression_model::<f32>(10, 64, 2)?;
    let optimizer = Adam::new(0.001);
    let loss_fn = MSELoss::new();

    // Create training config
    let training_config = TrainingConfig {
        batch_size: 32,
        shuffle: true,
        num_workers: 0,
        learning_rate: 0.001,
        epochs: 5,
        verbose: 1,
        validation: Some(ValidationSettings {
            enabled: true,
            validation_split: 0.0, // Use separate validation dataset
            batch_size: 32,
            num_workers: 0,
        }),
        gradient_accumulation: None,
        mixed_precision: None,
    };

    // Create trainer
    let mut trainer = Trainer::new(model, optimizer, loss_fn, training_config);

    // Add gradient clipping callback with global norm method
    trainer.add_callback(Box::new(GradientClipping::by_global_norm(
        1.0_f32, // Max norm
        true,    // Log stats
    )));

    println!("\nTraining model with gradient clipping by global norm...");

    // Train model for a few epochs
    let dataset_small = generate_regression_dataset::<f32>(500, 10, 2);
    let val_dataset_small = generate_regression_dataset::<f32>(100, 10, 2);
    let session = trainer.train(&dataset_small, Some(&val_dataset_small))?;

    println!("\nTraining completed in {} epochs", session.epochs_trained);
    println!(
        "Final loss: {:.4}",
        session.get_metric("loss").unwrap().last().unwrap()
    );
    println!(
        "Final validation loss: {:.4}",
        session.get_metric("val_loss").unwrap().last().unwrap()
    );

    // Example with value clipping
    println!("\nExample with gradient clipping by value:");

    // Create model and trainer with value clipping
    let model = create_regression_model::<f32>(10, 64, 2)?;
    let optimizer = Adam::new(0.001);
    let mut trainer = Trainer::new(model, optimizer, loss_fn, training_config);

    // Add gradient clipping callback with value method
    trainer.add_callback(Box::new(GradientClipping::by_value(
        0.5_f32, // Max value
        true,    // Log stats
    )));

    println!("\nDemonstration of how to set up gradient clipping by value:");
    println!("trainer.add_callback(Box::new(GradientClipping::by_value(");
    println!("    0.5_f32, // Max value");
    println!("    true,    // Log stats");
    println!(")));");

    // Demonstrate the training utilities
    println!("\nAdvanced Training Examples Completed Successfully!");

    Ok(())
}
