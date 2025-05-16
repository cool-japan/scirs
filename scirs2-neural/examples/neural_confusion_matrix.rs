use ndarray::{Array1, Array2};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use scirs2_neural::callbacks::{Callback, CallbackContext, CallbackTiming, VisualizationCallback};
use scirs2_neural::error::Result;
use scirs2_neural::layers::Dense;
use scirs2_neural::losses::MeanSquaredError;
use scirs2_neural::models::{sequential::Sequential, Model};
use scirs2_neural::optimizers::Adam;
use scirs2_neural::utils::evaluation::ConfusionMatrix;
use std::collections::HashMap;
use std::f32::consts::PI;

// Generate a spiral dataset for multi-class classification
fn generate_spiral_dataset(
    n_samples: usize,
    n_classes: usize,
    noise: f32,
    rng: &mut SmallRng,
) -> (Array2<f32>, Array1<usize>) {
    let mut x = Array2::<f32>::zeros((n_samples * n_classes, 2));
    let mut y = Array1::<usize>::zeros(n_samples * n_classes);

    for j in 0..n_classes {
        // Angular separation between spirals
        let r = (j as f32) * 2.0 * PI / (n_classes as f32);

        for i in 0..n_samples {
            // Generate points along a spiral
            let t = 1.0 * (i as f32) / (n_samples as f32);
            let radius = 2.0 * t;

            // Angle
            let theta = 1.5 * t * 2.0 * PI + r;

            // Point coordinates
            let x1 = radius * f32::cos(theta) + noise * rng.random_range(-1.0..1.0);
            let x2 = radius * f32::sin(theta) + noise * rng.random_range(-1.0..1.0);

            // Store the point and label
            let idx = j * n_samples + i;
            x[[idx, 0]] = x1;
            x[[idx, 1]] = x2;
            y[idx] = j;
        }
    }

    (x, y)
}

// Create a simple classification model
fn create_classification_model(
    input_dim: usize,
    hidden_dim: usize,
    output_dim: usize,
    rng: &mut SmallRng,
) -> Result<Sequential<f32>> {
    let mut model = Sequential::new();

    // First hidden layer
    let dense1 = Dense::new(input_dim, hidden_dim, Some("relu"), rng)?;
    model.add_layer(dense1);

    // Second hidden layer
    let dense2 = Dense::new(hidden_dim, hidden_dim / 2, Some("relu"), rng)?;
    model.add_layer(dense2);

    // Output layer
    let dense3 = Dense::new(hidden_dim / 2, output_dim, Some("sigmoid"), rng)?;
    model.add_layer(dense3);

    Ok(model)
}

// Convert one-hot encoded predictions to class indices
fn predictions_to_classes(
    predictions: &ndarray::ArrayBase<ndarray::OwnedRepr<f32>, ndarray::Dim<ndarray::IxDynImpl>>,
) -> Array1<usize> {
    let shape = predictions.shape();
    let n_samples = shape[0];
    let n_classes = shape[1];
    let mut classes = Array1::zeros(n_samples);

    for i in 0..n_samples {
        // Create a view of the i-th row
        let mut max_val = predictions[[i, 0]];
        let mut max_idx = 0;

        // Find the index of the highest value
        for j in 1..n_classes {
            let val = predictions[[i, j]];
            if val > max_val {
                max_val = val;
                max_idx = j;
            }
        }

        classes[i] = max_idx;
    }

    classes
}

// Helper to convert class indices to one-hot encoded vectors
fn one_hot_encode(y: &Array1<usize>, n_classes: usize) -> Array2<f32> {
    let n_samples = y.len();
    let mut one_hot = Array2::zeros((n_samples, n_classes));

    for i in 0..n_samples {
        let class_idx = y[i];
        if class_idx < n_classes {
            one_hot[[i, class_idx]] = 1.0;
        }
    }

    one_hot
}

fn main() -> Result<()> {
    println!("Neural Network Confusion Matrix Visualization");
    println!("==============================================\n");

    // Initialize RNG with a fixed seed for reproducibility
    let mut rng = SmallRng::seed_from_u64(42);

    // Generate spiral dataset for 3-class classification
    let n_classes = 3;
    let n_samples_per_class = 100;
    let noise = 0.15;

    let (x, y) = generate_spiral_dataset(n_samples_per_class, n_classes, noise, &mut rng);
    println!(
        "Generated spiral dataset with {} classes, {} samples per class",
        n_classes, n_samples_per_class
    );

    // Split data into training and test sets (80/20 split)
    let n_samples = x.shape()[0];
    let n_train = (n_samples as f32 * 0.8) as usize;
    let n_test = n_samples - n_train;

    let x_train = x.slice(ndarray::s![0..n_train, ..]).to_owned();
    let y_train = y.slice(ndarray::s![0..n_train]).to_owned();
    let x_test = x.slice(ndarray::s![n_train.., ..]).to_owned();
    let y_test = y.slice(ndarray::s![n_train..]).to_owned();

    println!(
        "Split data into {} training and {} test samples",
        n_train, n_test
    );

    // Create a classification model
    let input_dim = 2; // 2D input (x, y coordinates)
    let hidden_dim = 32; // Hidden layer size
    let output_dim = n_classes; // One output per class

    let mut model = create_classification_model(input_dim, hidden_dim, output_dim, &mut rng)?;
    println!("Created model with {} layers", model.num_layers());

    // Setup loss function and optimizer
    let loss_fn = MeanSquaredError::new();
    let mut optimizer = Adam::new(0.01, 0.9, 0.999, 1e-8);

    // Train the model
    let epochs = 100;
    let x_train_dyn = x_train.clone().into_dyn();
    let y_train_onehot = one_hot_encode(&y_train, n_classes);
    let y_train_onehot_dyn = y_train_onehot.into_dyn();

    // Create visualization callback for training metrics
    let mut visualization_cb = VisualizationCallback::new(10) // Show every 10 epochs
        .with_tracked_metrics(vec![
            "train_loss".to_string(),
            "val_accuracy".to_string(),
        ]);

    // Define class labels for confusion matrix
    let class_labels = vec![
        "Class A".to_string(),
        "Class B".to_string(),
        "Class C".to_string(),
    ];

    // Train the model (simple manual training loop)
    println!("\nTraining model...");

    // Initialize history for tracking metrics
    let mut epoch_history = HashMap::new();
    epoch_history.insert("train_loss".to_string(), Vec::new());
    epoch_history.insert("val_accuracy".to_string(), Vec::new());

    // Training loop
    for epoch in 0..epochs {
        // Train for one epoch
        let train_loss =
            model.train_batch(&x_train_dyn, &y_train_onehot_dyn, &loss_fn, &mut optimizer)?;

        // Compute validation accuracy
        let x_test_dyn = x_test.clone().into_dyn();
        let predictions = model.forward(&x_test_dyn)?;
        let predicted_classes = predictions_to_classes(&predictions);

        // Calculate validation accuracy
        let mut correct = 0;
        for i in 0..n_test {
            if predicted_classes[i] == y_test[i] {
                correct += 1;
            }
        }
        let val_accuracy = correct as f32 / n_test as f32;

        // Store metrics
        epoch_history
            .get_mut("train_loss")
            .unwrap()
            .push(train_loss);
        epoch_history
            .get_mut("val_accuracy")
            .unwrap()
            .push(val_accuracy);

        // Print progress
        if (epoch + 1) % 10 == 0 || epoch == 0 {
            println!(
                "Epoch {}/{}: loss = {:.6}, val_accuracy = {:.4}",
                epoch + 1,
                epochs,
                train_loss,
                val_accuracy
            );
        }

        // Update visualization callback
        let mut context = CallbackContext {
            epoch,
            total_epochs: epochs,
            batch: 0,
            total_batches: 1,
            batch_loss: None,
            epoch_loss: Some(train_loss),
            val_loss: None,
            metrics: vec![val_accuracy],
            history: &epoch_history,
            stop_training: false,
        };

        // Visualize progress with metrics chart
        if epoch % 10 == 0 || epoch == epochs - 1 {
            visualization_cb.on_event(CallbackTiming::AfterEpoch, &mut context)?;
        }

        // Calculate and show confusion matrix during training
        if epoch % 20 == 0 || epoch == epochs - 1 {
            // Create confusion matrix
            let cm = ConfusionMatrix::<f32>::new(
                &y_test.view(),
                &predicted_classes.view(),
                Some(n_classes),
                Some(class_labels.clone()),
            )?;

            // Show heatmap visualization
            println!("\nConfusion Matrix at Epoch {}:", epoch + 1);
            println!(
                "{}",
                cm.to_heatmap(
                    Some(&format!("Confusion Matrix - Epoch {}", epoch + 1)),
                    true
                )
            );
        }
    }

    // Final evaluation
    println!("\nFinal model evaluation:");

    // Make predictions on test set
    let x_test_dyn = x_test.clone().into_dyn();
    let predictions = model.forward(&x_test_dyn)?;
    let predicted_classes = predictions_to_classes(&predictions);

    // Create confusion matrix
    let cm = ConfusionMatrix::<f32>::new(
        &y_test.view(),
        &predicted_classes.view(),
        Some(n_classes),
        Some(class_labels.clone()),
    )?;

    // Calculate and show metrics
    let accuracy = cm.accuracy();
    let precision = cm.precision();
    let recall = cm.recall();
    let f1 = cm.f1_score();

    println!("\nFinal Classification Metrics:");
    println!("Overall Accuracy: {:.4}", accuracy);

    println!("\nPer-Class Metrics:");
    println!("Class | Precision | Recall | F1-Score");
    println!("-----------------------------------");

    for i in 0..n_classes {
        println!(
            "{}    | {:.4}     | {:.4}  | {:.4}",
            class_labels[i], precision[i], recall[i], f1[i]
        );
    }

    println!("\nMacro F1-Score: {:.4}", cm.macro_f1());

    // Show different confusion matrix visualizations
    println!("\nFinal Confusion Matrix Visualizations:");

    // 1. Standard confusion matrix
    println!("\n1. Standard Confusion Matrix:");
    println!("{}", cm.to_ascii(Some("Final Confusion Matrix"), false));

    // 2. Normalized confusion matrix
    println!("\n2. Normalized Confusion Matrix:");
    println!(
        "{}",
        cm.to_ascii(Some("Final Normalized Confusion Matrix"), true)
    );

    // 3. Confusion matrix heatmap
    println!("\n3. Confusion Matrix Heatmap:");
    println!(
        "{}",
        cm.to_heatmap(Some("Final Confusion Matrix Heatmap"), true)
    );

    // 4. Error pattern analysis
    println!("\n4. Error Pattern Analysis:");
    println!("{}", cm.error_heatmap(Some("Final Error Pattern Analysis")));

    println!("\nNeural Network Confusion Matrix Visualization Complete!");
    Ok(())
}
