// use scirs2_core::ndarray::Array; // Unused import
use scirs2_neural::{
    error::Result,
    // models::architectures::{ConvNeXt, ConvNeXtConfig, ConvNeXtVariant}, // TODO: Implement models module
    // prelude::*, // Unused import
};

#[allow(dead_code)]
fn main() -> Result<()> {
    println!("ConvNeXt Example");
    println!("----------------");
    println!("Note: ConvNeXt models are not yet implemented.");
    println!("This example is a placeholder for future ConvNeXt functionality.");

    // TODO: Implement ConvNeXt models module and uncomment the following code
    /*
    // Create a random input tensor (batch_size=1, channels=3, height=224, width=224)
    let inputshape = [1, 3, 224, 224];
    let mut input = Array::<f32>::zeros(inputshape).into_dyn();
    // Fill with random values between 0 and 1
    for elem in input.iter_mut() {
        *elem = scirs2_core::random::random::<f32>();
    }
    // Create ConvNeXt-Tiny with default configuration
    println!("\nConvNeXt-Tiny:");
    let convnext_tiny = ConvNeXt::convnext_tiny(1000, true)?;
    let output_tiny = convnext_tiny.forward(&input)?;
    println!("Output shape: {:?}", output_tiny.shape());
    // Create ConvNeXt-Small with default configuration
    println!("\nConvNeXt-Small:");
    let convnext_small = ConvNeXt::convnext_small(1000, true)?;
    let output_small = convnext_small.forward(&input)?;
    println!("Output shape: {:?}", output_small.shape());
    // Create ConvNeXt-Base with default configuration
    println!("\nConvNeXt-Base:");
    let convnext_base = ConvNeXt::convnext_base(1000, true)?;
    let output_base = convnext_base.forward(&input)?;
    println!("Output shape: {:?}", output_base.shape());
    // Create ConvNeXt-Large with default configuration
    println!("\nConvNeXt-Large:");
    let convnext_large = ConvNeXt::convnext_large(1000, true)?;
    let output_large = convnext_large.forward(&input)?;
    println!("Output shape: {:?}", output_large.shape());
    // Custom ConvNeXt with specific configuration
    println!("\nCustom ConvNeXt:");
    let custom_config = ConvNeXtConfig {
        variant: ConvNeXtVariant::Tiny,
        input_channels: 3,
        depths: vec![3, 3, 9, 3],
        dims: vec![96, 192, 384, 768],
        num_classes: 10,
        dropout_rate: Some(0.2),
        layer_scale_init_value: 1e-6,
        include_top: true,
    };
    let custom_convnext = ConvNeXt::new(custom_config)?;
    let output_custom = custom_convnext.forward(&input)?;
    println!("Output shape: {:?}", output_custom.shape());
    // Example of inference
    println!("\nInference example with ConvNeXt-Tiny:");
    let inference_input = Array::<f32>::zeros(inputshape).into_dyn();
    let inference_output = convnext_tiny.forward(&inference_input)?;
    // Get top prediction (normally you'd have class labels)
    let mut max_val = f32::MIN;
    let mut max_idx = 0;
    for (i, &val) in inference_output.iter().enumerate() {
        if val > max_val {
            max_val = val;
            max_idx = i;
        }
    }
    println!(
        "Predicted class: {} with confidence: {:.4}",
        max_idx, max_val
    );
    */
    Ok(())
}
