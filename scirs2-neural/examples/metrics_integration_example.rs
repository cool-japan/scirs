//! Example of integrating scirs2-metrics with scirs2-neural
//!
//! This example shows how to use metrics from scirs2-metrics with
//! neural network training in scirs2-neural.
//!
//! To run this example, enable the 'metrics_integration' feature:
//!
//! ```bash
//! cargo run --example metrics_integration_example --features metrics_integration
//! ```

#[cfg(feature = "metrics_integration")]
use ndarray::Array2;

// This example requires the metrics_integration feature
#[cfg(not(feature = "metrics_integration"))]
fn main() {
    println!("This example requires the 'metrics_integration' feature.");
    println!("Run it with: cargo run --example metrics_integration_example --features metrics_integration");
}

#[cfg(feature = "metrics_integration")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use scirs2_metrics::integration::neural::NeuralMetricAdapter;
    use scirs2_neural::activations::relu::ReLU;
    use scirs2_neural::activations::sigmoid::Sigmoid;
    use scirs2_neural::callbacks::{
        Callback, CallbackContext, CallbackTiming, ScirsMetricsCallback,
    };
    use scirs2_neural::evaluation::MetricType;
    use scirs2_neural::layers::{Dense, Layer};
    use scirs2_neural::losses::mse::MeanSquaredError;
    use scirs2_neural::models::sequential::Sequential;
    use scirs2_neural::optimizers::sgd::SGD;

    println!("Neural Network with Metrics Integration Example");
    println!("---------------------------------------------");

    // Create a simple synthetic dataset for binary classification
    let n_samples = 500;
    let (x_train, y_train) = generate_binary_classification_data(n_samples, 2, 42)?;

    // Create a simple neural network
    let mut model = Sequential::new();
    model.add(Dense::new(2, 10, Some(ReLU)));
    model.add(Dense::new(10, 1, Some(Sigmoid)));

    println!("Model architecture:");
    println!("  Input: 2 features");
    println!("  Hidden layer: 10 neurons with ReLU activation");
    println!("  Output layer: 1 neuron with Sigmoid activation");

    // Create optimizer
    let optimizer = SGD::new(0.1);

    // Create loss function
    let loss = MeanSquaredError::new();

    // Create metrics callback using scirs2-metrics
    let metrics = vec![
        NeuralMetricAdapter::<f64>::accuracy(),
        NeuralMetricAdapter::<f64>::precision(),
        NeuralMetricAdapter::<f64>::recall(),
        NeuralMetricAdapter::<f64>::f1_score(),
    ];

    let metrics_callback = ScirsMetricsCallback::new(metrics).map(|cb| cb.with_verbose(true));

    // Prepare for training
    let num_epochs = 20;
    let batch_size = 32;

    // Create a dummy initial context for our callback
    let mut context = CallbackContext {
        epoch: 0,
        total_epochs: num_epochs,
        batch: 0,
        total_batches: n_samples / batch_size,
        batch_loss: None,
        epoch_loss: None,
        val_loss: None,
        metrics: Vec::new(),
        history: &HashMap::new(),
        stop_training: false,
    };

    println!(
        "\nStart training for {} epochs with batch size {}:",
        num_epochs, batch_size
    );

    // Train model (simplified training loop - not using actual training method)
    for epoch in 0..num_epochs {
        context.epoch = epoch;

        // Shuffle data
        let indices: Vec<usize> = (0..n_samples).collect();
        let shuffled_indices = shuffle_indices(indices);

        // Initialize epoch loss
        let mut epoch_loss = 0.0;

        // Process each batch
        for batch_idx in (0..n_samples).step_by(batch_size) {
            let end_idx = (batch_idx + batch_size).min(n_samples);
            let batch_indices = &shuffled_indices[batch_idx..end_idx];

            // Create batch data
            let x_batch = create_batch(&x_train, batch_indices);
            let y_batch = create_batch(&y_train, batch_indices);

            // Forward pass
            let y_pred = model.forward(&x_batch)?;

            // Compute loss
            let batch_loss = loss.forward(&y_pred, &y_batch)?;
            epoch_loss += batch_loss;

            // Backward pass and weights update would be here in a real training loop

            // Update context
            context.batch = batch_idx / batch_size;
            context.batch_loss = Some(batch_loss);
        }

        // Compute epoch metrics
        epoch_loss /= (n_samples as f64) / (batch_size as f64);
        context.epoch_loss = Some(epoch_loss);

        // Make predictions on entire dataset for metrics
        let y_pred = model.forward(&x_train)?;

        // In a real implementation, the batch predictions and targets would be
        // accessible through the callback context. Here we're simulating that
        // by manually setting the predictions and targets in our ScirsMetricsCallback.
        if let Some(cb) = metrics_callback.as_mut() {
            // Set current predictions and targets
            cb.current_predictions = Some(y_pred.clone());
            cb.current_targets = Some(y_train.clone().into_dyn());

            // Call callback
            cb.on_event(CallbackTiming::AfterEpoch, &mut context)?;

            // Get results for display
            println!(
                "Epoch {}/{}: loss = {:.4}",
                epoch + 1,
                num_epochs,
                epoch_loss
            );
            for (name, value) in cb.epoch_results() {
                println!("  {}: {:.4}", name, value);
            }
        }
    }

    println!("\nTraining completed successfully!");

    Ok(())
}

#[cfg(feature = "metrics_integration")]
fn generate_binary_classification_data(
    n_samples: usize,
    n_features: usize,
    seed: u64,
) -> Result<(Array2<f64>, Array2<f64>), Box<dyn std::error::Error>> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    // Create features matrix
    let mut x = Array2::zeros((n_samples, n_features));

    // Create target vector
    let mut y = Array2::zeros((n_samples, 1));

    for i in 0..n_samples {
        // Generate random features
        for j in 0..n_features {
            x[[i, j]] = rng.gen_range(-1.0..1.0);
        }

        // Simple decision boundary: x1 + x2 > 0
        let sum = x[[i, 0]] + x[[i, 1]];
        y[[i, 0]] = if sum > 0.0 { 1.0 } else { 0.0 };
    }

    Ok((x, y))
}

#[cfg(feature = "metrics_integration")]
fn shuffle_indices(mut indices: Vec<usize>) -> Vec<usize> {
    let mut rng = rand::thread_rng();
    rand::seq::SliceRandom::shuffle(&mut indices, &mut rng);
    indices
}

#[cfg(feature = "metrics_integration")]
fn create_batch(data: &Array2<f64>, indices: &[usize]) -> Array2<f64> {
    let n_samples = indices.len();
    let n_features = data.shape()[1];

    let mut batch = Array2::zeros((n_samples, n_features));

    for (i, &idx) in indices.iter().enumerate() {
        for j in 0..n_features {
            batch[[i, j]] = data[[idx, j]];
        }
    }

    batch
}

#[cfg(feature = "metrics_integration")]
use std::collections::HashMap;
