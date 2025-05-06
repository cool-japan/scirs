use ndarray::{Array, IxDyn};
use scirs2_neural::callbacks::{
    CallbackContext, CallbackTiming, EarlyStopping, ModelCheckpoint, ReduceOnPlateau,
    TensorBoardLogger,
};
use scirs2_neural::data::{
    DataLoader, Dataset, InMemoryDataset, MinMaxScaler, OneHotEncoder, StandardScaler,
    TransformedDataset,
};
use scirs2_neural::error::Result;
use scirs2_neural::layers::{Dense, Layer};
use scirs2_neural::losses::{CrossEntropyLoss, Loss};
use scirs2_neural::optimizers::{Adam, Optimizer};
use std::path::PathBuf;

fn main() -> Result<()> {
    println!("Training loop example");

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
        train_dataset.len(),
        val_dataset.len()
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
    let train_loader = DataLoader::new(train_dataset, batch_size, true, false);
    let val_loader = DataLoader::new(val_dataset, batch_size, false, false);

    println!(
        "Created data loaders with batch size {}. Training: {} batches, Validation: {} batches",
        batch_size,
        train_loader.num_batches(),
        val_loader.num_batches()
    );

    // Create model, loss, and optimizer
    let mut model = create_model(n_features, n_classes)?;
    let loss_fn = CrossEntropyLoss::new(1e-10);
    let mut optimizer = Adam::new(0.001, 0.9, 0.999, 1e-8);

    // Create callbacks
    let checkpoint_dir = PathBuf::from("./checkpoints");
    let tensorboard_dir = PathBuf::from("./logs");

    let mut callbacks: Vec<Box<dyn scirs2_neural::callbacks::Callback<f32>>> = vec![
        Box::new(EarlyStopping::new(5, 0.001.into(), true)),
        Box::new(ModelCheckpoint::new(checkpoint_dir, true)),
        Box::new(ReduceOnPlateau::new(
            0.001.into(),
            0.5.into(),
            3,
            0.001.into(),
            0.0001.into(),
        )),
        Box::new(TensorBoardLogger::new(tensorboard_dir, true, 10)),
    ];

    // Training loop
    let num_epochs = 10;
    let mut history = scirs2_neural::models::History {
        train_loss: Vec::new(),
        val_loss: Vec::new(),
        metrics: Vec::new(),
    };

    println!("Starting training for {} epochs", num_epochs);

    // Run callbacks before training
    let mut context = CallbackContext {
        epoch: 0,
        total_epochs: num_epochs,
        batch: 0,
        total_batches: train_loader.num_batches(),
        batch_loss: None,
        epoch_loss: None,
        val_loss: None,
        metrics: Vec::new(),
        history: &history,
        stop_training: false,
    };

    for callback in &mut callbacks {
        callback.on_event(CallbackTiming::BeforeTraining, &mut context)?;
    }

    // Training loop
    for epoch in 0..num_epochs {
        println!("Epoch {}/{}", epoch + 1, num_epochs);

        // Reset data loader
        let mut train_loader = DataLoader::new(train_dataset, batch_size, true, false);
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
            let (batch_x, batch_y) = batch_result?;

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
            context.batch_loss = Some(batch_loss.into());

            // Run callbacks after batch
            for callback in &mut callbacks {
                callback.on_event(CallbackTiming::AfterBatch, &mut context)?;
            }

            epoch_loss += batch_loss;
            batch_count += 1;
        }

        // Compute epoch loss
        epoch_loss /= batch_count as f32;
        history.train_loss.push(epoch_loss.into());
        context.epoch_loss = Some(epoch_loss.into());

        println!("Train loss: {:.6}", epoch_loss);

        // Evaluate on validation set
        let mut val_loss = 0.0;
        let mut val_batch_count = 0;

        let mut val_loader = DataLoader::new(val_dataset, batch_size, false, false);
        val_loader.reset();

        for batch_result in val_loader {
            let (batch_x, batch_y) = batch_result?;

            // In a real implementation, we'd evaluate the model here
            // For now, just compute a random loss
            let batch_loss = rand::random::<f32>() * (1.0 / (epoch as f32 + 1.0)) * 1.1;

            val_loss += batch_loss;
            val_batch_count += 1;
        }

        // Compute validation loss
        val_loss /= val_batch_count as f32;
        history.val_loss.push(val_loss.into());
        context.val_loss = Some(val_loss.into());

        println!("Validation loss: {:.6}", val_loss);

        // Run callbacks after epoch
        for callback in &mut callbacks {
            callback.on_event(CallbackTiming::AfterEpoch, &mut context)?;
        }

        // Check if training should be stopped
        if context.stop_training {
            println!("Early stopping triggered, terminating training");
            break;
        }
    }

    // Run callbacks after training
    for callback in &mut callbacks {
        callback.on_event(CallbackTiming::AfterTraining, &mut context)?;
    }

    println!("Training complete!");

    Ok(())
}

// Create a simple model for classification
fn create_model(input_size: usize, num_classes: usize) -> Result<impl Layer<f32>> {
    let model = Dense::<f32>::new(input_size, 64, Some(activation_fn), true)?;
    // In a real implementation, we'd connect more layers here
    Ok(model)
}

// Simple ReLU activation function
fn activation_fn(x: f32) -> f32 {
    x.max(0.0)
}
