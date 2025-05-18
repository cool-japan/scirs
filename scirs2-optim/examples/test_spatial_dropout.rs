use ndarray::{Array, Axis};
use scirs2_optim::regularizers::SpatialDropout;

fn main() {
    let sd = SpatialDropout::new(0.0).unwrap(); // No dropout to test

    // Create a simple 4D tensor (batch, channels, height, width)
    let features = Array::from_shape_fn((1, 2, 2, 2), |(b, c, h, w)| {
        println!("Creating element at ({}, {}, {}, {})", b, c, h, w);
        c as f64 + h as f64 + w as f64
    });

    println!("Original features shape: {:?}", features.shape());
    println!("Original features:\n{:?}", features);

    // Apply with no dropout
    let result = sd.apply(&features, true);
    println!("Result shape: {:?}", result.shape());
    println!("Result:\n{:?}", result);

    // Test with dropout = 0.5
    let sd_drop = SpatialDropout::new(0.5).unwrap();
    let result_drop = sd_drop.apply(&features, true);
    println!("Result with dropout:\n{:?}", result_drop);

    // Check channel axis
    for c in 0..2 {
        let channel = result_drop.index_axis(Axis(1), c);
        println!("Channel {}: {:?}", c, channel);
    }

    // Let's check how it works with slicing
    println!("\nExamining the tensor structure:");
    let batch_0 = features.index_axis(Axis(0), 0);
    println!("Batch 0 shape: {:?}", batch_0.shape());
    println!("Batch 0:\n{:?}", batch_0);

    let channel_0 = batch_0.index_axis(Axis(0), 0);
    println!("Channel 0 in batch 0 shape: {:?}", channel_0.shape());
    println!("Channel 0 in batch 0:\n{:?}", channel_0);
}
