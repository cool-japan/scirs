use rand::rngs::SmallRng;
use rand::SeedableRng;
use scirs2_neural::error::Result;
use scirs2_neural::layers::{Conv2D, Dense, Dropout, MaxPool2D, PaddingMode};
use scirs2_neural::models::sequential::Sequential;
use scirs2_neural::utils::colors::ColorOptions;
use scirs2_neural::utils::{sequential_model_dataflow, sequential_model_summary, ModelVizOptions};

fn main() -> Result<()> {
    // Initialize random number generator
    let mut rng = SmallRng::seed_from_u64(42);

    // Create a CNN model
    let model = create_cnn_model(&mut rng)?;

    // Display model summary
    let summary = sequential_model_summary(
        &model,
        Some(vec![32, 3, 224, 224]), // Input shape (batch_size, channels, height, width)
        Some("CNN Architecture"),
        Some(ModelVizOptions {
            width: 100,
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
        vec![32, 3, 224, 224], // Input shape
        Some("CNN Data Flow Diagram"),
        Some(ModelVizOptions {
            width: 80,
            show_params: true,
            show_shapes: true,
            show_properties: false,
            color_options: ColorOptions {
                enabled: true,
                ..Default::default()
            },
        }),
    )?;
    println!("\n{}", dataflow);

    Ok(())
}

// Create a simple CNN model (VGG-like)
fn create_cnn_model<R: rand::Rng + Clone + Send + Sync + 'static>(
    rng: &mut R,
) -> Result<Sequential<f64>> {
    let mut model = Sequential::new();

    // Block 1
    model.add_layer(Conv2D::new(3, 64, (3, 3), (1, 1), PaddingMode::Same, rng)?);
    model.add_layer(Conv2D::new(64, 64, (3, 3), (1, 1), PaddingMode::Same, rng)?);
    model.add_layer(MaxPool2D::new((2, 2), (2, 2), None)?);

    // Block 2
    model.add_layer(Conv2D::new(
        64,
        128,
        (3, 3),
        (1, 1),
        PaddingMode::Same,
        rng,
    )?);
    model.add_layer(Conv2D::new(
        128,
        128,
        (3, 3),
        (1, 1),
        PaddingMode::Same,
        rng,
    )?);
    model.add_layer(MaxPool2D::new((2, 2), (2, 2), None)?);

    // Block 3
    model.add_layer(Conv2D::new(
        128,
        256,
        (3, 3),
        (1, 1),
        PaddingMode::Same,
        rng,
    )?);
    model.add_layer(Conv2D::new(
        256,
        256,
        (3, 3),
        (1, 1),
        PaddingMode::Same,
        rng,
    )?);
    model.add_layer(MaxPool2D::new((2, 2), (2, 2), None)?);

    // Fully connected layers
    // In a real implementation we would need to add a Flatten layer here
    model.add_layer(Dense::new(256 * 28 * 28, 512, Some("relu"), rng)?);
    model.add_layer(Dropout::new(0.5, rng)?);
    model.add_layer(Dense::new(512, 10, Some("softmax"), rng)?);

    Ok(model)
}
