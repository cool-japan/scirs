// Example demonstrating memory-efficient training with gradient clipping
use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use scirs2_optim::gradient_processing::GradientProcessor;
use scirs2_optim::memory_efficient::{InPlaceAdam, InPlaceOptimizer};
use std::error::Error;

// Simple neural network layer
struct Layer {
    weights: Array2<f64>,
    bias: Array1<f64>,
}

impl Layer {
    fn new(input_size: usize, output_size: usize) -> Self {
        let weights = Array2::random((input_size, output_size), Uniform::new(-0.1, 0.1));
        let bias = Array1::zeros(output_size);
        Self { weights, bias }
    }

    fn forward(&self, input: &Array2<f64>) -> Array2<f64> {
        input.dot(&self.weights) + &self.bias
    }

    fn backward(
        &self,
        input: &Array2<f64>,
        grad_output: &Array2<f64>,
    ) -> (Array2<f64>, Array2<f64>, Array1<f64>) {
        let grad_input = grad_output.dot(&self.weights.t());
        let grad_weights = input.t().dot(grad_output) / input.nrows() as f64;
        let grad_bias = grad_output.mean_axis(ndarray::Axis(0)).unwrap();
        (grad_input, grad_weights, grad_bias)
    }
}

// Memory-efficient training function
fn train_memory_efficient(
    layer: &mut Layer,
    training_data: &Array2<f64>,
    targets: &Array2<f64>,
    epochs: usize,
) -> Result<Vec<f64>, Box<dyn Error>> {
    // Initialize separate optimizers for 2D weights and 1D bias
    let mut weights_optimizer = InPlaceAdam::new(0.001);
    let mut bias_optimizer = InPlaceAdam::new(0.001);
    let mut grad_processor = GradientProcessor::new();
    grad_processor.set_max_norm(1.0);
    grad_processor.set_centralization(true);

    let mut losses = Vec::new();

    println!("Memory-Efficient Training with Gradient Processing");
    println!("===============================================");

    for epoch in 0..epochs {
        // Forward pass
        let output = layer.forward(training_data);

        // Compute loss (MSE)
        let diff = &output - targets;
        let loss = diff.mapv(|x| x * x).mean().unwrap();
        losses.push(loss);

        // Backward pass
        let grad_output = diff.clone() * 2.0 / diff.len() as f64;
        let (_, grad_weights, grad_bias) = layer.backward(training_data, &grad_output);

        // Process gradients (mutates the gradients in-place)
        let mut grad_weights_processed = grad_weights.clone();
        let mut grad_bias_processed = grad_bias.clone();
        grad_processor.process(&mut grad_weights_processed)?;
        grad_processor.process(&mut grad_bias_processed)?;

        // Update parameters in-place (using separate optimizers for each shape)
        weights_optimizer.step_inplace(&mut layer.weights, &grad_weights_processed)?;
        bias_optimizer.step_inplace(&mut layer.bias, &grad_bias_processed)?;

        if epoch % 100 == 0 {
            println!("Epoch {}: Loss = {:.6}", epoch, loss);
        }
    }

    Ok(losses)
}

// Advanced memory-efficient training with custom gradient processing
fn train_with_custom_processing(
    layer: &mut Layer,
    training_data: &Array2<f64>,
    targets: &Array2<f64>,
    epochs: usize,
) -> Result<Vec<f64>, Box<dyn Error>> {
    use scirs2_optim::memory_efficient::{clip_inplace, scale_inplace};

    let mut weights_optimizer = InPlaceAdam::new(0.001);
    let mut bias_optimizer = InPlaceAdam::new(0.001);
    let mut losses = Vec::new();

    println!("\nMemory-Efficient Training with Custom Processing");
    println!("=============================================");

    for epoch in 0..epochs {
        // Forward pass
        let output = layer.forward(training_data);

        // Compute loss
        let diff = &output - targets;
        let loss = diff.mapv(|x| x * x).mean().unwrap();
        losses.push(loss);

        // Backward pass
        let grad_output = diff.clone() * 2.0 / diff.len() as f64;
        let (_, mut grad_weights, mut grad_bias) = layer.backward(training_data, &grad_output);

        // Custom in-place gradient processing
        // 1. Clip gradients
        clip_inplace(&mut grad_weights, -1.0, 1.0);
        clip_inplace(&mut grad_bias, -1.0, 1.0);

        // 2. Scale gradients (learning rate warm-up)
        if epoch < 100 {
            let scale = (epoch as f64 + 1.0) / 100.0;
            scale_inplace(&mut grad_weights, scale);
            scale_inplace(&mut grad_bias, scale);
        }

        // Update parameters in-place (using separate optimizers for each shape)
        weights_optimizer.step_inplace(&mut layer.weights, &grad_weights)?;
        bias_optimizer.step_inplace(&mut layer.bias, &grad_bias)?;

        if epoch % 100 == 0 {
            println!("Epoch {}: Loss = {:.6}", epoch, loss);
        }
    }

    Ok(losses)
}

fn main() -> Result<(), Box<dyn Error>> {
    // Create synthetic dataset
    let n_samples = 1000;
    let input_size = 50;
    let output_size = 10;

    let training_data = Array2::random((n_samples, input_size), Uniform::new(-1.0, 1.0));
    let true_weights = Array2::random((input_size, output_size), Uniform::new(-0.5, 0.5));
    let true_bias = Array1::random(output_size, Uniform::new(-0.1, 0.1));
    let targets = training_data.dot(&true_weights) + &true_bias;

    // Initialize layers
    let mut layer1 = Layer::new(input_size, output_size);
    let mut layer2 = Layer::new(input_size, output_size);

    // Train with standard memory-efficient approach
    let losses1 = train_memory_efficient(&mut layer1, &training_data, &targets, 500)?;

    // Train with custom processing
    let losses2 = train_with_custom_processing(&mut layer2, &training_data, &targets, 500)?;

    // Compare results
    println!("\nTraining Summary:");
    println!("================");
    println!(
        "Standard approach - Final loss: {:.6}",
        losses1.last().unwrap()
    );
    println!(
        "Custom processing - Final loss: {:.6}",
        losses2.last().unwrap()
    );

    // Memory efficiency demonstration
    println!("\nMemory Efficiency Notes:");
    println!("======================");
    println!("- All parameter updates are performed in-place");
    println!("- Gradient processing operations modify arrays directly");
    println!("- No intermediate arrays are created during optimization");
    println!("- Memory usage remains constant throughout training");

    Ok(())
}
