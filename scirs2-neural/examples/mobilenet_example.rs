use ndarray::Array;
use scirs2_neural::error::Result;
use scirs2_neural::layers::Layer;
use scirs2_neural::models::architectures::{MobileNet, MobileNetConfig};

fn main() -> Result<()> {
    println!("MobileNet Example");
    println!("----------------");

    // Create a random input tensor (batch_size=1, channels=3, height=224, width=224)
    let input_shape = [1, 3, 224, 224];
    let mut input = Array::<f32, _>::zeros(input_shape).into_dyn();

    // Fill with random values between 0 and 1
    for elem in input.iter_mut() {
        *elem = rand::random::<f32>();
    }

    // Create MobileNetV1 with default configuration
    println!("\nMobileNetV1:");
    let mobilenet_v1 = MobileNet::mobilenet_v1(3, 1000)?;
    let output_v1 = mobilenet_v1.forward(&input)?;
    println!("Output shape: {:?}", output_v1.shape());

    // Create MobileNetV2 with default configuration
    println!("\nMobileNetV2:");
    let mobilenet_v2 = MobileNet::mobilenet_v2(3, 1000)?;
    let output_v2 = mobilenet_v2.forward(&input)?;
    println!("Output shape: {:?}", output_v2.shape());

    // Create MobileNetV3-Small with default configuration
    println!("\nMobileNetV3-Small:");
    let mobilenet_v3_small = MobileNet::mobilenet_v3_small(3, 1000)?;
    let output_v3_small = mobilenet_v3_small.forward(&input)?;
    println!("Output shape: {:?}", output_v3_small.shape());

    // Create MobileNetV3-Large with default configuration
    println!("\nMobileNetV3-Large:");
    let mobilenet_v3_large = MobileNet::mobilenet_v3_large(3, 1000)?;
    let output_v3_large = mobilenet_v3_large.forward(&input)?;
    println!("Output shape: {:?}", output_v3_large.shape());

    // Custom MobileNet with specific configuration
    println!("\nCustom MobileNet:");

    // Create a custom MobileNetV2 configuration
    let mut custom_config = MobileNetConfig::mobilenet_v2(3, 10);
    custom_config.width_multiplier = 0.75;
    custom_config.dropout_rate = 0.2;

    let custom_mobilenet = MobileNet::new(custom_config)?;
    let output_custom = custom_mobilenet.forward(&input)?;
    println!("Output shape: {:?}", output_custom.shape());

    // Example of inference with MobileNetV2
    println!("\nInference example with MobileNetV2:");
    let inference_input = Array::<f32, _>::zeros(input_shape).into_dyn();
    let inference_output = mobilenet_v2.forward(&inference_input)?;

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

    Ok(())
}
