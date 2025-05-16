use ndarray::Array2;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use scirs2_neural::callbacks::{
    Callback, CallbackContext, CallbackTiming, ScheduleMethod, StepDecay, VisualizationCallback,
};
use scirs2_neural::error::Result;
use scirs2_neural::layers::Dense;
use scirs2_neural::losses::{Loss, MeanSquaredError};
use scirs2_neural::models::{sequential::Sequential, Model};
use scirs2_neural::optimizers::{Adam, Optimizer};
use scirs2_neural::utils::visualization::{
    analyze_training_history, export_history_to_csv, PlotOptions,
};
use std::collections::HashMap;
use std::f32::consts::PI;

// Helper function to generate random noise
fn generate_noise(rng: &mut SmallRng) -> f32 {
    rng.random_range(-0.05..0.05)
}

// Generate a simple nonlinear dataset (sine wave with noise)
fn generate_nonlinear_data(num_samples: usize, rng: &mut SmallRng) -> (Array2<f32>, Array2<f32>) {
    // Generate x values evenly spaced in [0, 4π]
    let x_values: Vec<f32> = (0..num_samples)
        .map(|i| 4.0 * PI * (i as f32) / (num_samples as f32 - 1.0))
        .collect();

    // Apply sine function and add noise
    let y_values: Vec<f32> = x_values
        .iter()
        .map(|&x| {
            let y = f32::sin(x);
            // Add some mild noise
            let noise: f32 = generate_noise(rng);
            y + noise
        })
        .collect();

    // Convert to 2D arrays
    let x = Array2::from_shape_vec((num_samples, 1), x_values).unwrap();
    let y = Array2::from_shape_vec((num_samples, 1), y_values).unwrap();

    (x, y)
}

// Split data into training and validation sets
fn train_val_split(
    x: &Array2<f32>,
    y: &Array2<f32>,
    train_ratio: f32,
) -> (Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>) {
    let num_samples = x.shape()[0];
    let num_train = (num_samples as f32 * train_ratio) as usize;

    let x_train = x.slice(ndarray::s![0..num_train, ..]).to_owned();
    let y_train = y.slice(ndarray::s![0..num_train, ..]).to_owned();
    let x_val = x.slice(ndarray::s![num_train.., ..]).to_owned();
    let y_val = y.slice(ndarray::s![num_train.., ..]).to_owned();

    (x_train, y_train, x_val, y_val)
}

// Create a simple regression model
fn create_regression_model(input_dim: usize, rng: &mut SmallRng) -> Result<Sequential<f32>> {
    let mut model = Sequential::new();

    // Hidden layer with 16 neurons and ReLU activation
    let dense1 = Dense::new(input_dim, 16, Some("relu"), rng)?;
    model.add_layer(dense1);

    // Hidden layer with 8 neurons and ReLU activation
    let dense2 = Dense::new(16, 8, Some("relu"), rng)?;
    model.add_layer(dense2);

    // Output layer with 1 neuron and linear activation
    let dense3 = Dense::new(8, 1, None, rng)?;
    model.add_layer(dense3);

    Ok(model)
}

fn main() -> Result<()> {
    println!("Training Visualization Example");
    println!("==============================\n");

    // Initialize RNG with a fixed seed for reproducibility
    let mut rng = SmallRng::seed_from_u64(42);

    // Generate synthetic data
    let num_samples = 200;
    let (x, y) = generate_nonlinear_data(num_samples, &mut rng);
    println!("Generated synthetic dataset with {} samples", num_samples);

    // Split data into training and validation sets
    let (x_train, y_train, x_val, y_val) = train_val_split(&x, &y, 0.8);
    println!(
        "Split data: {} training samples, {} validation samples",
        x_train.shape()[0],
        x_val.shape()[0]
    );

    // Create model
    let mut model = create_regression_model(1, &mut rng)?;
    println!("Created model with {} layers", model.num_layers());

    // Setup loss function and optimizer
    let loss_fn = MeanSquaredError::new();
    let mut optimizer = Adam::new(0.01, 0.9, 0.999, 1e-8);

    // Training parameters
    let epochs = 100;

    // Configure learning rate scheduler
    let mut scheduler = StepDecay::new(
        0.01.into(), // Initial learning rate
        0.5.into(),  // Decay factor
        30,          // Step size
        ScheduleMethod::Epoch,
        1e-4.into(), // Min learning rate
    );

    // Add visualization callback
    let mut visualization_cb =
        VisualizationCallback::new(5) // Show every 5 epochs
            .with_tracked_metrics(vec!["train_loss".to_string(), "val_loss".to_string()])
            .with_plot_options(PlotOptions {
                width: 80,
                height: 15,
                max_x_ticks: 10,
                max_y_ticks: 5,
                line_char: '─',
                point_char: '●',
                background_char: ' ',
                show_grid: true,
                show_legend: true,
            })
            .with_save_path("training_plot.txt");

    // Train the model manually
    let mut epoch_history = HashMap::new();
    epoch_history.insert("train_loss".to_string(), Vec::new());
    epoch_history.insert("val_loss".to_string(), Vec::new());

    // Convert data to dynamic arrays and ensure they are owned (not views)
    let x_train_dyn = x_train.clone().into_dyn();
    let y_train_dyn = y_train.clone().into_dyn();
    let x_val_dyn = x_val.clone().into_dyn();
    let y_val_dyn = y_val.clone().into_dyn();

    println!("\nStarting training with visualization...");

    // Manual training loop
    println!("\nStarting training loop...");
    for epoch in 0..epochs {
        // Update learning rate with scheduler
        let current_lr = scheduler.get_lr();
        optimizer.set_learning_rate(current_lr);

        // Train for one epoch (batch size = full dataset in this example)
        let train_loss = model.train_batch(&x_train_dyn, &y_train_dyn, &loss_fn, &mut optimizer)?;

        // Compute validation loss
        let predictions = model.forward(&x_val_dyn)?;
        let val_loss = loss_fn.forward(&predictions, &y_val_dyn)?;

        // Store metrics
        epoch_history
            .get_mut("train_loss")
            .unwrap()
            .push(train_loss);
        epoch_history.get_mut("val_loss").unwrap().push(val_loss);

        // Update the scheduler
        scheduler.update_lr(epoch);

        // Print progress
        if epoch % 10 == 0 || epoch == epochs - 1 {
            println!(
                "Epoch {}/{}: train_loss = {:.6}, val_loss = {:.6}, lr = {:.6}",
                epoch + 1,
                epochs,
                train_loss,
                val_loss,
                current_lr
            );
        }

        // Visualize progress (manually calling the visualization callback)
        if epoch % 5 == 0 || epoch == epochs - 1 {
            let mut context = CallbackContext {
                epoch,
                total_epochs: epochs,
                batch: 0,
                total_batches: 1,
                batch_loss: None,
                epoch_loss: Some(train_loss),
                val_loss: Some(val_loss),
                metrics: vec![],
                history: &epoch_history,
                stop_training: false,
            };

            visualization_cb.on_event(CallbackTiming::AfterEpoch, &mut context)?;
        }
    }

    // Final visualization
    let mut context = CallbackContext {
        epoch: epochs - 1,
        total_epochs: epochs,
        batch: 0,
        total_batches: 1,
        batch_loss: None,
        epoch_loss: Some(*epoch_history.get("train_loss").unwrap().last().unwrap()),
        val_loss: Some(*epoch_history.get("val_loss").unwrap().last().unwrap()),
        metrics: vec![],
        history: &epoch_history,
        stop_training: false,
    };

    visualization_cb.on_event(CallbackTiming::AfterTraining, &mut context)?;

    println!("\nTraining complete!");

    // Export history to CSV
    export_history_to_csv(&epoch_history, "training_history.csv")?;
    println!("Exported training history to training_history.csv");

    // Analyze training results
    let analysis = analyze_training_history(&epoch_history);
    println!("\nTraining Analysis:");
    println!("------------------");
    for issue in analysis {
        println!("{}", issue);
    }

    // Make predictions on validation data
    println!("\nMaking predictions on validation data...");
    let predictions = model.forward(&x_val_dyn)?;

    // Calculate and display final metrics
    let mse = loss_fn.forward(&predictions, &y_val_dyn)?;
    println!("Final validation MSE: {:.6}", mse);

    // Display a few sample predictions
    println!("\nSample predictions:");
    println!("------------------");
    println!("  X  |  True Y  | Predicted Y ");
    println!("---------------------------");

    // Show first 5 predictions
    let num_samples_to_show = std::cmp::min(5, x_val.shape()[0]);
    for i in 0..num_samples_to_show {
        println!(
            "{:.4} | {:.4}   | {:.4}",
            x_val[[i, 0]],
            y_val[[i, 0]],
            predictions[[i, 0]]
        );
    }

    println!("\nVisualization demonstration complete!");
    Ok(())
}
