//! Example demonstrating gradient clipping techniques

use ndarray::Array1;
use scirs2_optim::{
    gradient_processing::GradientProcessor,
    optimizers::{Adam, Optimizer},
};

fn main() {
    // Create example parameters and large gradients (simulating gradient explosion)
    let params = Array1::zeros(5);
    let mut large_gradients = Array1::from_vec(vec![100.0, -200.0, 300.0, -150.0, 250.0]);

    println!("Original gradients: {:?}", large_gradients);

    // Create gradient processor with L2 norm clipping
    let mut processor = GradientProcessor::new();

    // Configure L2 norm clipping
    processor.set_max_norm(10.0);

    // Clip gradients
    processor.process(&mut large_gradients).unwrap();
    println!("Clipped gradients (L2 norm): {:?}", large_gradients);
    println!(
        "L2 norm of clipped gradients: {:.4}",
        large_gradients.iter().map(|x| x * x).sum::<f64>().sqrt()
    );

    // Use value clipping
    let mut value_gradients = Array1::from_vec(vec![100.0, -200.0, 300.0, -150.0, 250.0]);
    let mut value_processor = GradientProcessor::new();
    value_processor.set_max_value(50.0).set_min_value(-50.0);

    value_processor.process(&mut value_gradients).unwrap();
    println!("\nValue clipped gradients (±50): {:?}", value_gradients);

    // Use L1 norm clipping
    let mut l1_gradients = Array1::from_vec(vec![100.0, -200.0, 300.0, -150.0, 250.0]);
    let mut l1_processor = GradientProcessor::new();
    l1_processor.set_max_l1_norm(100.0);

    l1_processor.process(&mut l1_gradients).unwrap();
    println!("\nL1 norm clipped gradients: {:?}", l1_gradients);

    // Apply gradient clipping in optimization loop
    let mut optimizer = Adam::new(0.001);
    let mut current_params = params.clone();
    let mut processor = GradientProcessor::new();
    processor.set_max_norm(10.0);

    for epoch in 0..5 {
        let mut gradients = Array1::from_vec(vec![
            10.0 * (epoch as f64 + 1.0),
            -20.0 * (epoch as f64 + 1.0),
            15.0 * (epoch as f64 + 1.0),
            -5.0 * (epoch as f64 + 1.0),
            12.0 * (epoch as f64 + 1.0),
        ]);

        // Clip gradients before optimization
        processor.process(&mut gradients).unwrap();

        // Update parameters with clipped gradients
        current_params = optimizer.step(&current_params, &gradients).unwrap();

        println!(
            "\nEpoch {}: param_norm={:.4}",
            epoch,
            current_params.iter().map(|x| x * x).sum::<f64>().sqrt()
        );
    }

    // Demonstrate gradient zeroing for small gradients
    let mut small_gradients = Array1::from_vec(vec![0.0001, -0.0002, 0.0003, 0.00001, -0.00005]);
    println!("\nSmall gradients: {:?}", small_gradients);

    // Zero gradients smaller than threshold
    let mut zeroing_processor = GradientProcessor::new();
    zeroing_processor.set_zero_threshold(0.0002);

    zeroing_processor.process(&mut small_gradients).unwrap();
    println!(
        "Gradients after zeroing small values: {:?}",
        small_gradients
    );

    // Demonstrate gradient centralization
    let mut grad_for_centralization = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    println!(
        "\nGradients before centralization: {:?}",
        grad_for_centralization
    );

    let mut central_processor = GradientProcessor::new();
    central_processor.set_centralization(true);

    central_processor
        .process(&mut grad_for_centralization)
        .unwrap();
    println!(
        "Gradients after centralization: {:?}",
        grad_for_centralization
    );
    println!(
        "Mean after centralization: {:.6}",
        grad_for_centralization.mean().unwrap()
    );
}
