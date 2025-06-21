use ndarray::{Array, IxDyn};
use scirs2_neural::callbacks::{
    CallbackContext, CallbackTiming, EarlyStopping, ReduceOnPlateau, TensorBoardLogger,
    VisualizationCallback,
};
use scirs2_neural::data::{
    DataLoader, InMemoryDataset, OneHotEncoder, StandardScaler, TransformedDataset,
};
use scirs2_neural::error::Result;
use scirs2_neural::layers::{Dense, Layer};
use scirs2_neural::losses::CrossEntropyLoss;
use scirs2_neural::optimizers::Adam;
use scirs2_neural::utils::{
    analyze_training_history, ascii_plot, export_history_to_csv, LearningRateSchedule, PlotOptions,
};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

fn main() -> Result<()> {
    println!("Training loop example with visualization");

    // Create dummy data
    let n_samples = 1000;
    let n_features = 10;
    let n_classes = 3;

    println!(
        "Generating dummy data with {} samples, {} features, {} classes",
        n_samples, n_features, n_classes
    );

    // Generate random features
    let features = Array::from_shape_fn(IxDyn(&[n_samples, n_features]), |_| {
        rand::random::<f32>() * 2.0 - 1.0
    });

    // Generate random labels (integers 0 to n_classes-1)
    let labels = Array::from_shape_fn(IxDyn(&[n_samples, 1]), |_| {
        (rand::random::<f32>() * n_classes as f32).floor()
    });

    // Create dataset
    let dataset = InMemoryDataset::new(features, labels)?;

    // Split into training and validation sets
    let (train_dataset, val_dataset) = dataset.train_test_split(0.2)?;

    println!(
        "Split data into {} training samples and {} validation samples",
        train_dataset.features.shape()[0],
        val_dataset.features.shape()[0]
    );

    // Create transformations
    let feature_scaler = StandardScaler::new(false);
    let label_encoder = OneHotEncoder::new(n_classes);

    // Apply transformations
    let train_dataset = TransformedDataset::new(train_dataset)
        .with_feature_transform(feature_scaler)
        .with_label_transform(label_encoder);

    let val_dataset = TransformedDataset::new(val_dataset)
        .with_feature_transform(StandardScaler::new(false))
        .with_label_transform(OneHotEncoder::new(n_classes));

    // Create data loaders
    let batch_size = 32;
    let train_loader = DataLoader::new(train_dataset.clone(), batch_size, true, false);
    let val_loader = DataLoader::new(val_dataset.clone(), batch_size, false, false);

    println!(
        "Created data loaders with batch size {}. Training: {} batches, Validation: {} batches",
        batch_size,
        train_loader.num_batches(),
        val_loader.num_batches()
    );

    // Create model, loss, and optimizer
    let _model = create_model(n_features, n_classes)?;
    let _loss_fn = CrossEntropyLoss::new(1e-10);

    // Create learning rate schedule
    let lr_schedule = LearningRateSchedule::StepDecay {
        initial_lr: 0.001,
        decay_factor: 0.5,
        step_size: 3,
    };

    // Generate learning rates for all epochs
    let num_epochs = 10;
    let learning_rates = lr_schedule.generate_schedule(num_epochs);
    println!("Learning rate schedule:");
    for (i, &lr) in learning_rates.iter().enumerate() {
        println!("  Epoch {}: {:.6}", i + 1, lr);
    }

    let _optimizer = Adam::new(0.001, 0.9, 0.999, 1e-8);

    // Create callbacks
    let _checkpoint_dir = PathBuf::from("./checkpoints");
    let tensorboard_dir = PathBuf::from("./logs");

    // Create output directories if they don't exist
    create_dir_if_not_exists("./checkpoints")?;
    create_dir_if_not_exists("./logs")?;
    create_dir_if_not_exists("./outputs")?;

    // For this example, we'll just remove the ModelCheckpoint
    let mut callbacks: Vec<Box<dyn scirs2_neural::callbacks::Callback<f32>>> = vec![
        Box::new(EarlyStopping::new(5, 0.001, true)),
        // ModelCheckpoint removed for simplicity as it requires special handling
        Box::new(ReduceOnPlateau::new(0.001, 0.5, 3, 0.001, 0.0001)),
        Box::new(TensorBoardLogger::new(tensorboard_dir, true, 10)),
        // Add our visualization callback
        Box::new(
            VisualizationCallback::new(1)
                .with_save_path("./outputs/training_plot.txt")
                .with_tracked_metrics(vec![
                    "train_loss".to_string(),
                    "val_loss".to_string(),
                    "accuracy".to_string(),
                    "learning_rate".to_string(),
                ]),
        ),
    ];

    // Training loop
    let mut history = HashMap::<String, Vec<f32>>::new();
    history.insert("train_loss".to_string(), Vec::new());
    history.insert("val_loss".to_string(), Vec::new());
    history.insert("learning_rate".to_string(), Vec::new());
    history.insert("accuracy".to_string(), Vec::new());

    println!("Starting training for {} epochs", num_epochs);

    // Run callbacks before training
    // Create a copy of history for the context
    let mut context_history = HashMap::<String, Vec<f32>>::new();
    context_history.insert("train_loss".to_string(), Vec::new());
    context_history.insert("val_loss".to_string(), Vec::new());
    context_history.insert("learning_rate".to_string(), Vec::new());
    context_history.insert("accuracy".to_string(), Vec::new());

    // For this example, we adapt to use Vec<F> for metrics
    // which is simpler than using Vec<(String, Option<F>)>
    // In a real implementation, use the proper context format
    let mut context = CallbackContext {
        epoch: 0,
        total_epochs: num_epochs,
        batch: 0,
        total_batches: train_loader.num_batches(),
        batch_loss: None,
        epoch_loss: None,
        val_loss: None,
        metrics: Vec::new(),
        history: &context_history,
        stop_training: false,
        model: None,
    };

    for callback in &mut callbacks {
        callback.on_event(CallbackTiming::BeforeTraining, &mut context)?;
    }

    // Training loop
    for epoch in 0..num_epochs {
        println!("Epoch {}/{}", epoch + 1, num_epochs);

        // Get learning rate for this epoch
        let learning_rate = learning_rates[epoch];
        history
            .get_mut("learning_rate")
            .unwrap()
            .push(learning_rate);

        // Reset data loader
        let mut train_loader = DataLoader::new(train_dataset.clone(), batch_size, true, false);
        train_loader.reset();

        // Update context
        context.epoch = epoch;
        context.epoch_loss = None;
        context.val_loss = None;

        // Run callbacks before epoch
        for callback in &mut callbacks {
            callback.on_event(CallbackTiming::BeforeEpoch, &mut context)?;
        }

        // Train on batches
        let mut epoch_loss = 0.0;
        let mut batch_count = 0;

        for (batch, batch_result) in train_loader.enumerate() {
            let (_batch_x, _batch_y) = batch_result?;

            // Update context
            context.batch = batch;
            context.batch_loss = None;

            // Run callbacks before batch
            for callback in &mut callbacks {
                callback.on_event(CallbackTiming::BeforeBatch, &mut context)?;
            }

            // In a real implementation, we'd train the model here
            // For now, just compute a random loss
            let batch_loss = rand::random::<f32>() * (1.0 / (epoch as f32 + 1.0));

            // Update batch loss
            context.batch_loss = Some(batch_loss);

            // Run callbacks after batch
            for callback in &mut callbacks {
                callback.on_event(CallbackTiming::AfterBatch, &mut context)?;
            }

            epoch_loss += batch_loss;
            batch_count += 1;
        }

        // Compute epoch loss
        epoch_loss /= batch_count as f32;
        history.get_mut("train_loss").unwrap().push(epoch_loss);
        context.epoch_loss = Some(epoch_loss);

        println!("Train loss: {:.6}", epoch_loss);

        // Evaluate on validation set
        let mut val_loss = 0.0;
        let mut val_batch_count = 0;

        let mut val_loader = DataLoader::new(val_dataset.clone(), batch_size, false, false);
        val_loader.reset();

        for batch_result in val_loader {
            let (_batch_x, _batch_y) = batch_result?;

            // In a real implementation, we'd evaluate the model here
            // For now, just compute a random loss
            let batch_loss = rand::random::<f32>() * (1.0 / (epoch as f32 + 1.0)) * 1.1;

            val_loss += batch_loss;
            val_batch_count += 1;
        }

        // Compute validation loss
        val_loss /= val_batch_count as f32;
        history.get_mut("val_loss").unwrap().push(val_loss);
        context.val_loss = Some(val_loss);

        // Simulate accuracy metric
        let accuracy =
            0.5 + 0.4 * (epoch as f32 / num_epochs as f32) + rand::random::<f32>() * 0.05;
        history.get_mut("accuracy").unwrap().push(accuracy);

        // Add accuracy to metrics
        context.metrics = vec![accuracy];

        println!("Validation loss: {:.6}", val_loss);
        println!("Accuracy: {:.2}%", accuracy * 100.0);

        // Run callbacks after epoch
        for callback in &mut callbacks {
            callback.on_event(CallbackTiming::AfterEpoch, &mut context)?;
        }

        // Check if training should be stopped
        if context.stop_training {
            println!("Early stopping triggered, terminating training");
            break;
        }

        // Visualize after each epoch
        if epoch > 0 {
            // Plot training and validation loss
            let loss_plot = ascii_plot(
                &history,
                Some("Training and Validation Loss"),
                Some(PlotOptions {
                    width: 80,
                    height: 20,
                    max_x_ticks: 10,
                    max_y_ticks: 5,
                    line_char: '─',
                    point_char: '●',
                    background_char: ' ',
                    show_grid: true,
                    show_legend: true,
                }),
            )?;
            println!("\n{}", loss_plot);
        }
    }

    // Run callbacks after training
    for callback in &mut callbacks {
        callback.on_event(CallbackTiming::AfterTraining, &mut context)?;
    }

    println!("Training complete!");

    // Export metrics to CSV
    let csv_path = "./outputs/training_history.csv";
    export_history_to_csv(&history, csv_path)?;
    println!("Training history exported to {}", csv_path);

    // Analyze training history
    let analysis = analyze_training_history(&history);
    println!("\nTraining Analysis:");
    for issue in analysis {
        println!("  {}", issue);
    }

    // Final visualization of metrics
    println!("\nFinal Training Metrics:\n");

    // Prepare subset of metrics for separate accuracy plot
    let mut accuracy_data = HashMap::new();
    accuracy_data.insert(
        "accuracy".to_string(),
        history.get("accuracy").unwrap().clone(),
    );

    // Plot accuracy
    let accuracy_plot = ascii_plot(
        &accuracy_data,
        Some("Model Accuracy"),
        Some(PlotOptions {
            width: 80,
            height: 15,
            max_x_ticks: 10,
            max_y_ticks: 5,
            line_char: '─',
            point_char: '●',
            background_char: ' ',
            show_grid: true,
            show_legend: true,
        }),
    )?;
    println!("{}", accuracy_plot);

    // Prepare subset of metrics for learning rate plot
    let mut lr_data = HashMap::new();
    lr_data.insert(
        "learning_rate".to_string(),
        history.get("learning_rate").unwrap().clone(),
    );

    // Plot learning rate
    let lr_plot = ascii_plot(
        &lr_data,
        Some("Learning Rate Schedule"),
        Some(PlotOptions {
            width: 80,
            height: 15,
            max_x_ticks: 10,
            max_y_ticks: 5,
            line_char: '─',
            point_char: '■',
            background_char: ' ',
            show_grid: true,
            show_legend: true,
        }),
    )?;
    println!("{}", lr_plot);

    // Visualize both train and validation losses in a single plot
    let mut loss_data = HashMap::new();
    loss_data.insert(
        "train_loss".to_string(),
        history.get("train_loss").unwrap().clone(),
    );
    loss_data.insert(
        "val_loss".to_string(),
        history.get("val_loss").unwrap().clone(),
    );

    let loss_plot = ascii_plot(
        &loss_data,
        Some("Training and Validation Loss"),
        Some(PlotOptions {
            width: 80,
            height: 20,
            max_x_ticks: 10,
            max_y_ticks: 5,
            line_char: '─',
            point_char: '●',
            background_char: ' ',
            show_grid: true,
            show_legend: true,
        }),
    )?;
    println!("{}", loss_plot);

    Ok(())
}

// Create a simple model for classification
fn create_model(input_size: usize, num_classes: usize) -> Result<impl Layer<f32>> {
    // Create a mutable RNG for initialization
    let mut rng = rand::rng();

    // Create a simple neural network model with two hidden layers
    let hidden_size1 = 64;
    let hidden_size2 = 32;

    // In a real implementation, we'd connect more layers here
    // For demo purposes, we're just returning the first layer
    println!("Creating model with architecture:");
    println!("  Input size: {}", input_size);
    println!("  Hidden layer 1: {}", hidden_size1);
    println!("  Hidden layer 2: {}", hidden_size2);
    println!("  Output size: {}", num_classes);

    // Create a dense layer with ReLU activation
    let model = Dense::<f32>::new(input_size, hidden_size1, Some("relu"), &mut rng)?;

    Ok(model)
}

// Helper function to create a directory if it doesn't exist
fn create_dir_if_not_exists(path: impl AsRef<Path>) -> Result<()> {
    let path = path.as_ref();
    if !path.exists() {
        std::fs::create_dir_all(path).map_err(|e| {
            scirs2_neural::error::NeuralError::IOError(format!(
                "Failed to create directory {}: {}",
                path.display(),
                e
            ))
        })?;
    }
    Ok(())
}
