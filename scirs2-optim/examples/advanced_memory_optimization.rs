// Advanced memory optimization features demonstration
use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use scirs2_optim::memory_efficient::{
    adaptive::{get_memory_usage_ratio, MemoryAwareBatchSizer},
    fused::{fused_adam_update, fused_apply_constraints, fused_gradient_clip_normalize},
    mixed_precision::LossScaler,
    InPlaceAdam, InPlaceOptimizer,
};
use std::error::Error;
use std::time::Instant;

// Comprehensive training example with all memory optimizations
struct AdvancedTrainer {
    // Model parameters
    weights: Array2<f64>,
    bias: Array1<f64>,

    // Optimizer states
    weights_m: Array2<f64>,
    weights_v: Array2<f64>,
    bias_m: Array1<f64>,
    bias_v: Array1<f64>,

    // Optimization components
    loss_scaler: LossScaler,
    batch_sizer: MemoryAwareBatchSizer,
    step_count: i32,

    // Hyperparameters
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    weight_decay: f64,
}

impl AdvancedTrainer {
    fn new(input_size: usize, output_size: usize, initial_batch_size: usize) -> Self {
        let weights = Array2::random((input_size, output_size), Uniform::new(-0.1, 0.1));
        let bias = Array1::zeros(output_size);

        Self {
            weights_m: Array2::zeros((input_size, output_size)),
            weights_v: Array2::zeros((input_size, output_size)),
            bias_m: Array1::zeros(output_size),
            bias_v: Array1::zeros(output_size),
            weights,
            bias,
            loss_scaler: LossScaler::new(65536.0),
            batch_sizer: MemoryAwareBatchSizer::new(initial_batch_size)
                .with_memory_threshold(0.85)
                .with_adaptation_factor(1.3),
            step_count: 0,
            learning_rate: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.01,
        }
    }

    fn forward(&self, input: &Array2<f64>) -> Array2<f64> {
        input.dot(&self.weights) + &self.bias
    }

    fn backward(
        &self,
        input: &Array2<f64>,
        grad_output: &Array2<f64>,
    ) -> (Array2<f64>, Array1<f64>) {
        let grad_weights = input.t().dot(grad_output) / input.nrows() as f64;
        let grad_bias = grad_output.mean_axis(ndarray::Axis(0)).unwrap();
        (grad_weights, grad_bias)
    }

    fn step_with_fused_operations(
        &mut self,
        input: &Array2<f64>,
        targets: &Array2<f64>,
    ) -> Result<f64, Box<dyn Error>> {
        self.step_count += 1;

        // Forward pass
        let output = self.forward(input);

        // Compute loss (MSE)
        let diff = &output - targets;
        let loss = diff.mapv(|x| x * x).mean().unwrap();

        // Scale loss for mixed precision
        let _scaled_loss = self.loss_scaler.scale_loss(loss as f32) as f64;

        // Backward pass
        let grad_output = &diff * (2.0 / diff.len() as f64);
        let (mut grad_weights, mut grad_bias) = self.backward(input, &grad_output);

        // Unscale gradients
        self.loss_scaler.unscale_gradients(&mut grad_weights);
        self.loss_scaler.unscale_gradients(&mut grad_bias);

        // Check for gradient overflow
        let weights_inf = self.loss_scaler.check_gradients(&grad_weights);
        let bias_inf = self.loss_scaler.check_gradients(&grad_bias);
        let found_inf = weights_inf || bias_inf;

        // Update loss scaler
        self.loss_scaler.update(found_inf);

        if !found_inf {
            // Fused gradient processing (clipping + normalization)
            fused_gradient_clip_normalize(&mut grad_weights, Some(1.0), Some(5.0));
            fused_gradient_clip_normalize(&mut grad_bias, Some(1.0), Some(5.0));

            // Compute bias correction terms
            let bias1 = 1.0 - self.beta1.powi(self.step_count);
            let bias2 = 1.0 - self.beta2.powi(self.step_count);

            // Fused Adam updates
            fused_adam_update(
                &mut self.weights,
                &grad_weights,
                &mut self.weights_m,
                &mut self.weights_v,
                self.learning_rate,
                self.beta1,
                self.beta2,
                self.epsilon,
                bias1,
                bias2,
                Some(self.weight_decay),
            );

            fused_adam_update(
                &mut self.bias,
                &grad_bias,
                &mut self.bias_m,
                &mut self.bias_v,
                self.learning_rate,
                self.beta1,
                self.beta2,
                self.epsilon,
                bias1,
                bias2,
                None, // No weight decay for bias
            );

            // Apply parameter constraints
            fused_apply_constraints(
                &mut self.weights,
                Some(10.0),        // L2 constraint
                Some((-5.0, 5.0)), // Value bounds
            );
        }

        Ok(loss)
    }

    fn adapt_batch_size(&mut self) {
        // Get current memory usage (simplified)
        let memory_ratio = get_memory_usage_ratio();

        // Estimate memory usage of current parameters
        let weights_memory = self.weights.len() * std::mem::size_of::<f64>();
        let bias_memory = self.bias.len() * std::mem::size_of::<f64>();
        let param_memory = weights_memory + bias_memory;

        println!(
            "  Memory usage ratio: {:.2}, Parameter memory: {} bytes",
            memory_ratio, param_memory
        );

        // Adapt batch size based on memory pressure
        self.batch_sizer.adapt(memory_ratio);
    }

    fn get_current_batch_size(&self) -> usize {
        self.batch_sizer.current_batch_size()
    }
}

fn train_with_memory_optimization(
    trainer: &mut AdvancedTrainer,
    data: &Array2<f64>,
    targets: &Array2<f64>,
    epochs: usize,
) -> Result<Vec<f64>, Box<dyn Error>> {
    let mut losses = Vec::new();
    let total_samples = data.nrows();

    println!("Advanced Memory-Optimized Training");
    println!("=================================");
    println!("Total samples: {}", total_samples);
    println!("Initial batch size: {}", trainer.get_current_batch_size());
    println!();

    for epoch in 0..epochs {
        let mut epoch_loss = 0.0;
        let mut num_batches = 0;

        // Adaptive batch size
        trainer.adapt_batch_size();
        let batch_size = trainer.get_current_batch_size();

        // Process data in batches with current batch size
        for batch_start in (0..total_samples).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(total_samples);

            let batch_data = data
                .slice(ndarray::s![batch_start..batch_end, ..])
                .to_owned();
            let batch_targets = targets
                .slice(ndarray::s![batch_start..batch_end, ..])
                .to_owned();

            // Training step with all optimizations
            let batch_loss = trainer.step_with_fused_operations(&batch_data, &batch_targets)?;

            epoch_loss += batch_loss;
            num_batches += 1;
        }

        let avg_loss = epoch_loss / num_batches as f64;
        losses.push(avg_loss);

        if epoch % 50 == 0 {
            println!(
                "Epoch {}: Loss = {:.6}, Batch size = {}, Loss scale = {:.0}",
                epoch,
                avg_loss,
                batch_size,
                trainer.loss_scaler.get_scale()
            );
        }
    }

    Ok(losses)
}

fn benchmark_memory_efficiency() -> Result<(), Box<dyn Error>> {
    println!("\nMemory Efficiency Benchmark");
    println!("===========================");

    let size = 1000;
    let mut params = Array2::from_elem((size, size), 1.0);
    let gradients = Array2::from_elem((size, size), 0.001);

    // Standard Adam implementation (for comparison)
    let mut standard_optimizer = InPlaceAdam::new(0.01);

    // Benchmark standard approach
    let start = Instant::now();
    for _ in 0..100 {
        standard_optimizer.step_inplace(&mut params, &gradients)?;
    }
    let standard_time = start.elapsed();

    // Reset parameters for fused benchmark
    params = Array2::from_elem((size, size), 1.0);
    let mut m = Array2::zeros((size, size));
    let mut v = Array2::zeros((size, size));

    // Benchmark fused operations
    let start = Instant::now();
    for step in 0..100 {
        let bias1 = 1.0 - 0.9_f64.powi(step + 1);
        let bias2 = 1.0 - 0.999_f64.powi(step + 1);

        fused_adam_update(
            &mut params,
            &gradients,
            &mut m,
            &mut v,
            0.01,  // lr
            0.9,   // beta1
            0.999, // beta2
            1e-8,  // epsilon
            bias1,
            bias2,
            Some(0.01), // weight_decay
        );
    }
    let fused_time = start.elapsed();

    println!("Standard in-place Adam: {:?}", standard_time);
    println!("Fused Adam operations: {:?}", fused_time);
    println!(
        "Speedup: {:.2}x",
        standard_time.as_secs_f64() / fused_time.as_secs_f64()
    );

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    // Create synthetic dataset
    let n_samples = 5000;
    let input_size = 100;
    let output_size = 20;
    let initial_batch_size = 64;

    println!("Creating synthetic dataset...");
    let training_data = Array2::random((n_samples, input_size), Uniform::new(-1.0, 1.0));
    let true_weights = Array2::random((input_size, output_size), Uniform::new(-0.5, 0.5));
    let true_bias = Array1::random(output_size, Uniform::new(-0.1, 0.1));
    let targets = training_data.dot(&true_weights) + &true_bias;

    // Initialize trainer with all optimizations
    let mut trainer = AdvancedTrainer::new(input_size, output_size, initial_batch_size);

    // Train with advanced memory optimizations
    let losses = train_with_memory_optimization(&mut trainer, &training_data, &targets, 200)?;

    println!("\nTraining Summary:");
    println!("================");
    println!("Initial loss: {:.6}", losses[0]);
    println!("Final loss: {:.6}", losses.last().unwrap());
    println!("Final batch size: {}", trainer.get_current_batch_size());
    println!("Final loss scale: {:.0}", trainer.loss_scaler.get_scale());

    // Run memory efficiency benchmark
    benchmark_memory_efficiency()?;

    // Demonstrate mixed-precision features
    println!("\nMixed-Precision Features:");
    println!("========================");

    let mut scaler = LossScaler::new(32768.0);
    let mut test_gradients = Array1::from_vec(vec![1.0, 2.0, f64::INFINITY, 4.0]);

    println!(
        "Original gradients: {:?}",
        test_gradients.as_slice().unwrap()
    );

    let has_inf = scaler.check_gradients(&test_gradients);
    println!("Contains infinite values: {}", has_inf);

    scaler.update(has_inf);
    println!("Loss scale after overflow: {:.0}", scaler.get_scale());

    // Test with finite gradients
    test_gradients = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
    scaler.unscale_gradients(&mut test_gradients);
    println!(
        "Unscaled gradients: {:?}",
        test_gradients.as_slice().unwrap()
    );

    // Demonstrate memory-aware batch sizing
    println!("\nMemory-Aware Batch Sizing:");
    println!("==========================");

    let mut batch_sizer = MemoryAwareBatchSizer::new(32)
        .with_memory_threshold(0.8)
        .with_adaptation_factor(1.5);

    println!("Initial batch size: {}", batch_sizer.current_batch_size());

    // Simulate high memory usage
    batch_sizer.adapt(0.95);
    println!(
        "After high memory usage (95%): {}",
        batch_sizer.current_batch_size()
    );

    // Simulate low memory usage
    batch_sizer.adapt(0.3);
    println!(
        "After low memory usage (30%): {}",
        batch_sizer.current_batch_size()
    );

    println!("\nAll memory optimization features demonstrated successfully!");

    Ok(())
}
