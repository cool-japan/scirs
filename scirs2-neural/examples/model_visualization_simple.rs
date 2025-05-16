use rand::rngs::SmallRng;
use rand::SeedableRng;
use scirs2_neural::error::Result;
use scirs2_neural::layers::Dense;
use scirs2_neural::models::sequential::Sequential;
use scirs2_neural::utils::colors::ColorOptions;
use scirs2_neural::utils::{sequential_model_dataflow, sequential_model_summary, ModelVizOptions};

fn main() -> Result<()> {
    // Initialize random number generator
    let mut rng = SmallRng::seed_from_u64(42);

    // Create a simple MLP model
    let model = create_mlp_model(&mut rng)?;

    // Display model summary
    let summary = sequential_model_summary(
        &model,
        Some(vec![32, 784]), // Input shape (batch_size, input_features)
        Some("Simple MLP Neural Network"),
        Some(ModelVizOptions {
            width: 80,
            show_params: true,
            show_shapes: true,
            show_properties: true,
            color_options: ColorOptions {
                enabled: true,
                ..Default::default()
            },
        }),
    )?;
    println!("{}", summary);

    // Display model dataflow
    let dataflow = sequential_model_dataflow(
        &model,
        vec![32, 784], // Input shape
        Some("MLP Data Flow Diagram"),
        None, // Use default options
    )?;
    println!("\n{}", dataflow);

    Ok(())
}

// Create a simple MLP model
fn create_mlp_model<R: rand::Rng>(rng: &mut R) -> Result<Sequential<f64>> {
    let mut model = Sequential::new();

    // Input layer is implicitly defined by the first layer
    // Hidden layers
    model.add_layer(Dense::new(784, 128, Some("relu"), rng)?);
    model.add_layer(Dense::new(128, 64, Some("relu"), rng)?);

    // Output layer
    model.add_layer(Dense::new(64, 10, Some("softmax"), rng)?);

    Ok(model)
}
