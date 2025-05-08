use ndarray::{Array, IxDyn};
use scirs2_neural::layers::Layer;
use scirs2_neural::models::{EfficientNet, EfficientNetConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("EfficientNet Example");

    // Create EfficientNet-B0 model for image classification
    let input_channels = 3; // RGB images
    let num_classes = 1000; // ImageNet classes

    println!(
        "Creating EfficientNet-B0 model with {} input channels and {} output classes",
        input_channels, num_classes
    );

    // Create model
    let model = EfficientNet::<f32>::efficientnet_b0(input_channels, num_classes)?;

    // Create dummy input (batch_size=1, channels=3, height=224, width=224)
    let input = Array::from_shape_fn(IxDyn(&[1, input_channels, 224, 224]), |_| {
        rand::random::<f32>()
    });

    println!("Input shape: {:?}", input.shape());

    // Forward pass
    let output = model.forward(&input)?;

    println!("Output shape: {:?}", output.shape());
    println!("Output contains logits for {} classes", output.shape()[1]);

    // Create EfficientNet-B3 model (larger model)
    println!("\nCreating EfficientNet-B3 model...");

    let model_b3 = EfficientNet::<f32>::efficientnet_b3(input_channels, num_classes)?;

    // Create dummy input with higher resolution for B3 (300x300)
    let input_b3 = Array::from_shape_fn(IxDyn(&[1, input_channels, 300, 300]), |_| {
        rand::random::<f32>()
    });

    println!("Input shape for B3: {:?}", input_b3.shape());

    // Forward pass
    let output_b3 = model_b3.forward(&input_b3)?;

    println!("Output shape for B3: {:?}", output_b3.shape());

    // Create a custom EfficientNet model for smaller images
    println!("\nCreating a custom EfficientNet model for smaller images...");

    // Create simplified config with fewer stages
    let mut custom_config = EfficientNetConfig::efficientnet_b0(input_channels, 10); // 10 classes

    // Simplify by keeping only first 4 stages
    custom_config.stages.truncate(4);

    // Scale down the model
    custom_config.width_coefficient = 0.5;
    custom_config.depth_coefficient = 0.5;
    custom_config.resolution = 32; // For CIFAR-10 size images

    let custom_model = EfficientNet::<f32>::new(custom_config)?;

    // Create dummy input for small images (32x32)
    let small_input = Array::from_shape_fn(IxDyn(&[1, input_channels, 32, 32]), |_| {
        rand::random::<f32>()
    });

    println!("Custom input shape: {:?}", small_input.shape());

    // Forward pass
    let custom_output = custom_model.forward(&small_input)?;

    println!("Custom output shape: {:?}", custom_output.shape());
    println!(
        "Custom model produces logits for {} classes",
        custom_output.shape()[1]
    );

    println!("\nEfficientNet example completed successfully!");

    Ok(())
}
