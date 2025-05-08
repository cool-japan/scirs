use ndarray::{Array, IxDyn};
use scirs2_neural::layers::Layer;
use scirs2_neural::models::{ResNet, ResNetConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("ResNet Example");

    // Create a simple ResNet-18 model for image classification
    let input_channels = 3; // RGB images
    let num_classes = 1000; // ImageNet classes

    println!(
        "Creating ResNet-18 model with {} input channels and {} output classes",
        input_channels, num_classes
    );

    // Create model
    let model = ResNet::<f32>::resnet18(input_channels, num_classes)?;

    // Create dummy input (batch_size=1, channels=3, height=224, width=224)
    let input = Array::from_shape_fn(IxDyn(&[1, input_channels, 224, 224]), |_| {
        rand::random::<f32>()
    });

    println!("Input shape: {:?}", input.shape());

    // Forward pass
    let output = model.forward(&input)?;

    println!("Output shape: {:?}", output.shape());
    println!("Output contains logits for {} classes", output.shape()[1]);

    // Create a custom ResNet configuration
    println!("\nCreating a custom ResNet model...");

    let custom_config = ResNetConfig {
        block: scirs2_neural::models::ResNetBlock::Basic,
        layers: vec![
            scirs2_neural::models::ResNetLayer {
                blocks: 2,
                channels: 32,
                stride: 1,
            },
            scirs2_neural::models::ResNetLayer {
                blocks: 2,
                channels: 64,
                stride: 2,
            },
            scirs2_neural::models::ResNetLayer {
                blocks: 2,
                channels: 128,
                stride: 2,
            },
        ],
        input_channels: 1, // Grayscale images
        num_classes: 10,   // MNIST classes
        dropout_rate: 0.5, // Add dropout for regularization
    };

    let custom_model = ResNet::<f32>::new(custom_config)?;

    // Create dummy grayscale input (batch_size=1, channels=1, height=28, width=28)
    let grayscale_input = Array::from_shape_fn(IxDyn(&[1, 1, 28, 28]), |_| rand::random::<f32>());

    println!("Custom input shape: {:?}", grayscale_input.shape());

    // Forward pass
    let custom_output = custom_model.forward(&grayscale_input)?;

    println!("Custom output shape: {:?}", custom_output.shape());
    println!(
        "Custom model produces logits for {} classes",
        custom_output.shape()[1]
    );

    println!("\nResNet example completed successfully!");

    Ok(())
}
