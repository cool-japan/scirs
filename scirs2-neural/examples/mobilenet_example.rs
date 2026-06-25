//! MobileNet example
//!
//! Demonstrates building MobileNet variants and running a forward pass on a
//! small dummy image batch. A small input is used so the example runs quickly.

use scirs2_core::ndarray::{Array, IxDyn};
use scirs2_neural::error::Result;
use scirs2_neural::layers::Layer;
use scirs2_neural::models::architectures::{MobileNet, MobileNetConfig, MobileNetVersion};

fn main() -> Result<()> {
    println!("MobileNet Example");
    println!("-----------------");

    let input_channels = 3;
    let num_classes = 10;

    // Small dummy input (batch_size=1, channels=3, height=16, width=16) so the
    // naive convolution kernels run quickly.
    let input = Array::from_shape_fn(IxDyn(&[1, input_channels, 16, 16]), |_| {
        scirs2_core::random::random::<f32>()
    });
    println!("Input shape: {:?}", input.shape());

    // MobileNetV2 via the convenience constructor (runs a full forward pass).
    println!("\nMobileNetV2:");
    let mobilenet_v2 = MobileNet::<f32>::mobilenet_v2(input_channels, num_classes)?;
    let output_v2 = mobilenet_v2.forward(&input)?;
    println!("Output shape: {:?}", output_v2.shape());

    // Custom MobileNetV2 configuration with a reduced width multiplier.
    println!("\nCustom MobileNetV2 (width multiplier 0.5):");
    let mut config = MobileNetConfig::mobilenet_v2(input_channels, num_classes);
    config.version = MobileNetVersion::V2;
    config.width_multiplier = 0.5;
    let custom = MobileNet::<f32>::new(config)?;
    let output_custom = custom.forward(&input)?;
    println!("Output shape: {:?}", output_custom.shape());

    // Demonstrate the MobileNetV3-Small convenience constructor (construction
    // only, to keep the example fast).
    println!("\nMobileNetV3-Small:");
    let _mobilenet_v3 = MobileNet::<f32>::mobilenet_v3_small(input_channels, num_classes)?;
    println!("MobileNetV3-Small model created successfully.");

    println!("\nMobileNet example completed successfully!");
    Ok(())
}
