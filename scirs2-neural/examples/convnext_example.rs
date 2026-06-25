//! ConvNeXt example
//!
//! Demonstrates building a ConvNeXt model and running a forward pass on a small
//! dummy image batch. A small custom configuration and input are used so the
//! example runs quickly.

use scirs2_core::ndarray::{Array, IxDyn};
use scirs2_neural::error::Result;
use scirs2_neural::layers::Layer;
use scirs2_neural::models::architectures::{ConvNeXt, ConvNeXtConfig, ConvNeXtVariant};

fn main() -> Result<()> {
    println!("ConvNeXt Example");
    println!("----------------");

    // Small custom ConvNeXt configuration for a lightweight forward pass.
    let config = ConvNeXtConfig {
        variant: ConvNeXtVariant::Tiny,
        input_channels: 3,
        depths: vec![1, 1, 1, 1],
        dims: vec![8, 16, 32, 64],
        num_classes: 10,
        dropout_rate: Some(0.1),
        layer_scale_init_value: 1e-6,
        include_top: true,
    };

    println!("Creating a small custom ConvNeXt model...");
    let model = ConvNeXt::<f32>::new(config)?;

    // Small dummy input (batch_size=1, channels=3, height=16, width=16) so the
    // naive convolution kernels run quickly.
    let input = Array::from_shape_fn(IxDyn(&[1, 3, 16, 16]), |_| {
        scirs2_core::random::random::<f32>()
    });
    println!("Input shape: {:?}", input.shape());

    let output = model.forward(&input)?;
    println!("Output shape: {:?}", output.shape());

    // Find the top predicted class.
    let mut max_val = f32::MIN;
    let mut max_idx = 0;
    for (i, &val) in output.iter().enumerate() {
        if val > max_val {
            max_val = val;
            max_idx = i;
        }
    }
    println!("Predicted class: {} with score: {:.4}", max_idx, max_val);

    // Demonstrate the `convnext_tiny` convenience constructor. The full Tiny
    // configuration ([3, 3, 9, 3] depths up to 768 channels) is heavy for the
    // naive convolution kernels, so we construct it and report its configuration
    // rather than running a full forward pass, keeping the example fast. (The
    // forward-pass demonstration above uses the small custom model.)
    println!("\nCreating ConvNeXt-Tiny (feature extractor)...");
    let tiny = ConvNeXt::<f32>::convnext_tiny(10, false)?;
    println!("ConvNeXt-Tiny model created successfully.");
    println!("  - stage depths: {:?}", tiny.config.depths);
    println!("  - stage dims: {:?}", tiny.config.dims);

    println!("\nConvNeXt example completed successfully!");
    Ok(())
}
