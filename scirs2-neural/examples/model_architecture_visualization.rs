use rand::rngs::SmallRng;
use rand::SeedableRng;
use scirs2_neural::error::Result;
use scirs2_neural::layers::{BatchNorm, Conv2D, Dense, Dropout, PaddingMode};
use scirs2_neural::models::sequential::Sequential;
use scirs2_neural::utils::colors::ColorOptions;
use scirs2_neural::utils::{sequential_model_dataflow, sequential_model_summary, ModelVizOptions};

fn main() -> Result<()> {
    println!("Model Architecture Visualization Example");
    println!("=======================================\n");

    // Initialize RNG with a fixed seed for reproducibility
    let mut rng = SmallRng::seed_from_u64(42);

    // Example 1: MLP (Multilayer Perceptron) Architecture
    println!("\n--- Example 1: MLP Architecture ---\n");
    let mlp = create_mlp_model(&mut rng)?;

    // Display model summary
    let mlp_summary = sequential_model_summary(
        &mlp,
        Some(vec![32, 784]), // Input shape (batch_size, input_features)
        Some("MLP Neural Network"),
        Some(ModelVizOptions {
            width: 80,
            show_params: true,
            show_shapes: true,
            show_properties: true,
            color_options: ColorOptions::default(),
        }),
    )?;
    println!("{}", mlp_summary);

    // Display model dataflow
    let mlp_dataflow = sequential_model_dataflow(
        &mlp,
        vec![32, 784], // Input shape
        Some("MLP Data Flow"),
        None, // Use default options
    )?;
    println!("\n{}", mlp_dataflow);

    // Example 2: CNN (Convolutional Neural Network) Architecture
    println!("\n--- Example 2: CNN Architecture ---\n");
    let cnn = create_cnn_model(&mut rng)?;

    // Display model summary with colored output
    let mut color_options = ColorOptions::default();
    color_options.enabled = true; // Force enable colors
    color_options.use_bright = true;

    let cnn_summary = sequential_model_summary(
        &cnn,
        Some(vec![32, 28, 28, 1]), // Input shape (batch_size, height, width, channels)
        Some("CNN Neural Network"),
        Some(ModelVizOptions {
            width: 80,
            show_params: true,
            show_shapes: true,
            show_properties: true,
            color_options,
        }),
    )?;
    println!("{}", cnn_summary);

    // Display model dataflow
    let cnn_dataflow = sequential_model_dataflow(
        &cnn,
        vec![32, 28, 28, 1], // Input shape
        Some("CNN Data Flow"),
        Some(ModelVizOptions {
            width: 80,
            show_params: true,
            show_shapes: true,
            show_properties: false,
            color_options,
        }),
    )?;
    println!("\n{}", cnn_dataflow);

    // Example 3: RNN (Recurrent Neural Network) Architecture
    println!("\n--- Example 3: RNN (LSTM) Architecture ---\n");
    println!("Skipping RNN example due to threading constraints with LSTM implementation.");

    println!("\nModel Architecture Visualization Complete!");
    Ok(())
}

// Create a simple MLP model
fn create_mlp_model(rng: &mut SmallRng) -> Result<Sequential<f32>> {
    let mut model = Sequential::new();

    // Hidden layers with decreasing sizes
    let dense1 = Dense::new(784, 512, Some("relu"), rng)?;
    model.add_layer(dense1);

    let dropout1 = Dropout::new(0.2, rng)?;
    model.add_layer(dropout1);

    let dense2 = Dense::new(512, 256, Some("relu"), rng)?;
    model.add_layer(dense2);

    let dense3 = Dense::new(256, 128, Some("relu"), rng)?;
    model.add_layer(dense3);

    let dropout2 = Dropout::new(0.3, rng)?;
    model.add_layer(dropout2);

    // Output layer
    let dense4 = Dense::new(128, 10, Some("softmax"), rng)?;
    model.add_layer(dense4);

    Ok(model)
}

// Create a simple CNN model for MNIST
fn create_cnn_model(rng: &mut SmallRng) -> Result<Sequential<f32>> {
    let mut model = Sequential::new();

    // First convolutional block
    let conv1 = Conv2D::new(
        1,                      // input channels
        32,                     // output channels
        (3, 3),                 // kernel size
        (1, 1),                 // stride
        PaddingMode::Custom(1), // padding mode
        rng,
    )?;
    model.add_layer(conv1);

    let batch_norm1 = BatchNorm::new(32, 0.99, 1e-5, rng)?;
    model.add_layer(batch_norm1);

    // Second convolutional block
    let conv2 = Conv2D::new(
        32,                     // input channels
        64,                     // output channels
        (3, 3),                 // kernel size
        (2, 2),                 // stride (downsampling)
        PaddingMode::Custom(1), // padding mode
        rng,
    )?;
    model.add_layer(conv2);

    let batch_norm2 = BatchNorm::new(64, 0.99, 1e-5, rng)?;
    model.add_layer(batch_norm2);

    let dropout1 = Dropout::new(0.25, rng)?;
    model.add_layer(dropout1);

    // Flatten for fully connected layers - commented out as Flatten is not available
    // let flatten = Flatten::new()?;
    // model.add_layer(flatten);

    // Dense layers
    let dense1 = Dense::new(64 * 14 * 14, 128, Some("relu"), rng)?;
    model.add_layer(dense1);

    let dropout2 = Dropout::new(0.5, rng)?;
    model.add_layer(dropout2);

    // Output layer
    let dense2 = Dense::new(128, 10, Some("softmax"), rng)?;
    model.add_layer(dense2);

    Ok(model)
}

// Create a simple RNN (LSTM) model - Currently disabled due to thread safety constraints
/*
fn create_rnn_model(rng: &mut SmallRng) -> Result<Sequential<f32>> {
    let mut model = Sequential::new();

    // LSTM layers
    let lstm1 = LSTM::new(
        128,   // input size
        256,   // hidden size
        rng,
    )?;
    model.add_layer(lstm1);

    let dropout1 = Dropout::new(0.2, rng)?;
    model.add_layer(dropout1);

    let lstm2 = LSTM::new(
        256,   // input size
        128,   // hidden size
        rng,
    )?;
    model.add_layer(lstm2);

    let dropout2 = Dropout::new(0.2, rng)?;
    model.add_layer(dropout2);

    // Output layer
    let dense = Dense::new(128, 10, Some("softmax"), rng)?;
    model.add_layer(dense);

    Ok(model)
}
*/
