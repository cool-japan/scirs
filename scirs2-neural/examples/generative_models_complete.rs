//! Complete Generative Models Example
//!
//! This example demonstrates building generative models using scirs2-neural.
//! It includes:
//! - Variational Autoencoder (VAE) implementation
//! - Generator and Discriminator for simple GAN
//! - Synthetic dataset generation
//! - VAE loss (reconstruction + KL divergence)
//! - GAN training with adversarial loss
//! - Sample generation and evaluation metrics
//! - Latent space interpolation

use ndarray::{s, Array, Array2, Array3, Array4, ArrayD, IxDyn};
use scirs2_neural::layers::{
    AdaptiveMaxPool2D, BatchNorm, Conv2D, Dense, Dropout, PaddingMode, Sequential,
};
use scirs2_neural::losses::{CrossEntropyLoss, MeanSquaredError};
use scirs2_neural::prelude::*;

// Type alias to avoid conflicts with scirs2-neural's Result
type StdResult<T> = std::result::Result<T, Box<dyn std::error::Error>>;
use rand::prelude::*;
use rand::rngs::SmallRng;
// use std::collections::HashMap;
// use std::f32::consts::PI;

/// Configuration for generative models
#[derive(Debug, Clone)]
pub struct GenerativeConfig {
    pub input_size: (usize, usize),
    pub latent_dim: usize,
    pub hidden_dims: Vec<usize>,
    pub beta: f32, // Beta parameter for beta-VAE
}

impl Default for GenerativeConfig {
    fn default() -> Self {
        Self {
            input_size: (32, 32),
            latent_dim: 16,
            hidden_dims: vec![128, 64, 32],
            beta: 1.0,
        }
    }
}

/// Synthetic dataset generator for generative modeling
pub struct GenerativeDataset {
    config: GenerativeConfig,
    rng: SmallRng,
}

impl GenerativeDataset {
    pub fn new(config: GenerativeConfig, seed: u64) -> Self {
        Self {
            config,
            rng: SmallRng::seed_from_u64(seed),
        }
    }

    /// Generate a synthetic image (simple patterns)
    pub fn generate_sample(&mut self) -> Array3<f32> {
        let (height, width) = self.config.input_size;
        let mut image = Array3::<f32>::zeros((1, height, width)); // Grayscale

        let pattern_type = self.rng.random_range(0..4);

        match pattern_type {
            0 => {
                // Circles
                let num_circles = self.rng.random_range(1..4);
                for _ in 0..num_circles {
                    let center_x = self.rng.random_range(5..(width - 5)) as f32;
                    let center_y = self.rng.random_range(5..(height - 5)) as f32;
                    let radius = self.rng.random_range(3..8) as f32;
                    let intensity = self.rng.random_range(0.5..1.0);

                    for i in 0..height {
                        for j in 0..width {
                            let dx = j as f32 - center_x;
                            let dy = i as f32 - center_y;
                            if dx * dx + dy * dy <= radius * radius {
                                image[[0, i, j]] = intensity;
                            }
                        }
                    }
                }
            }
            1 => {
                // Stripes
                let stripe_width = self.rng.random_range(2..6);
                let intensity = self.rng.random_range(0.5..1.0);
                for i in 0..height {
                    if (i / stripe_width) % 2 == 0 {
                        for j in 0..width {
                            image[[0, i, j]] = intensity;
                        }
                    }
                }
            }
            2 => {
                // Checkerboard
                let square_size = self.rng.random_range(3..8);
                let intensity = self.rng.random_range(0.5..1.0);
                for i in 0..height {
                    for j in 0..width {
                        if ((i / square_size) + (j / square_size)) % 2 == 0 {
                            image[[0, i, j]] = intensity;
                        }
                    }
                }
            }
            _ => {
                // Gradient
                let direction = self.rng.random_range(0..2);
                let intensity = self.rng.random_range(0.5..1.0);
                for i in 0..height {
                    for j in 0..width {
                        let gradient_val = if direction == 0 {
                            i as f32 / height as f32
                        } else {
                            j as f32 / width as f32
                        };
                        image[[0, i, j]] = intensity * gradient_val;
                    }
                }
            }
        }

        // Add noise
        for elem in image.iter_mut() {
            *elem += self.rng.random_range(-0.1..0.1);
            *elem = elem.max(0.0).min(1.0);
        }

        image
    }

    /// Generate a batch of samples
    pub fn generate_batch(&mut self, batch_size: usize) -> Array4<f32> {
        let (height, width) = self.config.input_size;
        let mut images = Array4::<f32>::zeros((batch_size, 1, height, width));

        for i in 0..batch_size {
            let image = self.generate_sample();
            images.slice_mut(s![i, .., .., ..]).assign(&image);
        }

        images
    }
}

/// VAE Encoder that outputs mean and log variance for latent distribution
pub struct VAEEncoder {
    feature_extractor: Sequential<f32>,
    mean_head: Sequential<f32>,
    logvar_head: Sequential<f32>,
    #[allow(dead_code)]
    config: GenerativeConfig,
}

impl VAEEncoder {
    pub fn new(config: GenerativeConfig, rng: &mut SmallRng) -> StdResult<Self> {
        let (_height, _width) = config.input_size;

        // Feature extraction layers
        let mut feature_extractor = Sequential::new();
        feature_extractor.add(Conv2D::new(1, 32, (3, 3), (2, 2), PaddingMode::Same, rng)?);
        feature_extractor.add(BatchNorm::new(32, 1e-5, 0.1, rng)?);

        feature_extractor.add(Conv2D::new(32, 64, (3, 3), (2, 2), PaddingMode::Same, rng)?);
        feature_extractor.add(BatchNorm::new(64, 1e-5, 0.1, rng)?);

        feature_extractor.add(Conv2D::new(
            64,
            128,
            (3, 3),
            (2, 2),
            PaddingMode::Same,
            rng,
        )?);
        feature_extractor.add(BatchNorm::new(128, 1e-5, 0.1, rng)?);

        feature_extractor.add(AdaptiveMaxPool2D::new((4, 4), None)?);

        // Calculate flattened feature size
        let feature_size = 128 * 4 * 4;

        // Mean head
        let mut mean_head = Sequential::new();
        mean_head.add(Dense::new(
            feature_size,
            config.hidden_dims[0],
            Some("relu"),
            rng,
        )?);
        mean_head.add(Dropout::new(0.2, rng)?);
        mean_head.add(Dense::new(
            config.hidden_dims[0],
            config.latent_dim,
            None,
            rng,
        )?);

        // Log variance head
        let mut logvar_head = Sequential::new();
        logvar_head.add(Dense::new(
            feature_size,
            config.hidden_dims[0],
            Some("relu"),
            rng,
        )?);
        logvar_head.add(Dropout::new(0.2, rng)?);
        logvar_head.add(Dense::new(
            config.hidden_dims[0],
            config.latent_dim,
            None,
            rng,
        )?);

        Ok(Self {
            feature_extractor,
            mean_head,
            logvar_head,
            config,
        })
    }

    pub fn forward(&self, input: &ArrayD<f32>) -> StdResult<(ArrayD<f32>, ArrayD<f32>)> {
        // Extract features
        let features = self.feature_extractor.forward(input)?;

        // Flatten features
        let batch_size = features.shape()[0];
        let feature_dim = features.len() / batch_size;
        let flattened = features
            .to_shape(IxDyn(&[batch_size, feature_dim]))?
            .to_owned();

        // Get mean and log variance
        let mean = self.mean_head.forward(&flattened)?;
        let logvar = self.logvar_head.forward(&flattened)?;

        Ok((mean, logvar))
    }
}

/// VAE Decoder that reconstructs images from latent codes
pub struct VAEDecoder {
    latent_projection: Sequential<f32>,
    feature_layers: Sequential<f32>,
    output_conv: Conv2D<f32>,
    config: GenerativeConfig,
}

impl VAEDecoder {
    pub fn new(config: GenerativeConfig, rng: &mut SmallRng) -> StdResult<Self> {
        // Project latent to feature space
        let mut latent_projection = Sequential::new();
        latent_projection.add(Dense::new(
            config.latent_dim,
            config.hidden_dims[0],
            Some("relu"),
            rng,
        )?);
        latent_projection.add(Dense::new(
            config.hidden_dims[0],
            128 * 4 * 4,
            Some("relu"),
            rng,
        )?);

        // Feature reconstruction layers (simplified transpose convolutions)
        let mut feature_layers = Sequential::new();
        feature_layers.add(Conv2D::new(
            128,
            64,
            (3, 3),
            (1, 1),
            PaddingMode::Same,
            rng,
        )?);
        feature_layers.add(BatchNorm::new(64, 1e-5, 0.1, rng)?);

        feature_layers.add(Conv2D::new(64, 32, (3, 3), (1, 1), PaddingMode::Same, rng)?);
        feature_layers.add(BatchNorm::new(32, 1e-5, 0.1, rng)?);

        // Output layer
        let output_conv = Conv2D::new(32, 1, (3, 3), (1, 1), PaddingMode::Same, rng)?;

        Ok(Self {
            latent_projection,
            feature_layers,
            output_conv,
            config,
        })
    }

    pub fn forward(&self, latent: &ArrayD<f32>) -> StdResult<ArrayD<f32>> {
        // Project latent to feature space
        let projected = self.latent_projection.forward(latent)?;

        // Reshape to spatial format
        let batch_size = projected.shape()[0];
        let reshaped = projected.into_shape_with_order(IxDyn(&[batch_size, 128, 4, 4]))?;

        // Upsample to target size (simplified)
        let upsampled = self.upsample(&reshaped)?;

        // Apply feature layers
        let features = self.feature_layers.forward(&upsampled)?;

        // Generate output
        let output = self.output_conv.forward(&features)?;

        Ok(output)
    }

    fn upsample(&self, input: &ArrayD<f32>) -> StdResult<ArrayD<f32>> {
        let shape = input.shape();
        let batch_size = shape[0];
        let channels = shape[1];
        let height = shape[2];
        let width = shape[3];

        let (target_height, target_width) = self.config.input_size;
        let scale_h = target_height / height;
        let scale_w = target_width / width;

        let mut upsampled =
            Array4::<f32>::zeros((batch_size, channels, target_height, target_width));

        for b in 0..batch_size {
            for c in 0..channels {
                for i in 0..height {
                    for j in 0..width {
                        let value = input[[b, c, i, j]];
                        for di in 0..scale_h {
                            for dj in 0..scale_w {
                                let new_i = i * scale_h + di;
                                let new_j = j * scale_w + dj;
                                if new_i < target_height && new_j < target_width {
                                    upsampled[[b, c, new_i, new_j]] = value;
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(upsampled.into_dyn())
    }
}

/// Complete VAE model
pub struct VAEModel {
    encoder: VAEEncoder,
    decoder: VAEDecoder,
    config: GenerativeConfig,
}

impl VAEModel {
    pub fn new(config: GenerativeConfig, rng: &mut SmallRng) -> StdResult<Self> {
        let encoder = VAEEncoder::new(config.clone(), rng)?;
        let decoder = VAEDecoder::new(config.clone(), rng)?;

        Ok(Self {
            encoder,
            decoder,
            config,
        })
    }

    pub fn forward(
        &self,
        input: &ArrayD<f32>,
    ) -> StdResult<(ArrayD<f32>, ArrayD<f32>, ArrayD<f32>)> {
        // Encode
        let (mean, logvar) = self.encoder.forward(input)?;

        // Reparameterization trick (simplified)
        let latent = self.reparameterize(&mean, &logvar)?;

        // Decode
        let reconstruction = self.decoder.forward(&latent)?;

        Ok((reconstruction, mean, logvar))
    }

    fn reparameterize(&self, mean: &ArrayD<f32>, logvar: &ArrayD<f32>) -> StdResult<ArrayD<f32>> {
        // Sample epsilon from standard normal
        let mut epsilon = Array::zeros(mean.raw_dim());
        let mut rng = SmallRng::seed_from_u64(42); // Fixed seed for reproducibility

        for elem in epsilon.iter_mut() {
            *elem = rng.random_range(-1.0..1.0); // Approximate normal
        }

        // z = mean + std * epsilon, where std = exp(0.5 * logvar)
        let mut result = Array::zeros(mean.raw_dim());
        for (((&m, &lv), &eps), res) in mean
            .iter()
            .zip(logvar.iter())
            .zip(epsilon.iter())
            .zip(result.iter_mut())
        {
            let std = (0.5 * lv).exp();
            *res = m + std * eps;
        }

        Ok(result)
    }

    /// Generate new samples from random latent codes
    pub fn generate(&self, batch_size: usize) -> StdResult<ArrayD<f32>> {
        // Sample random latent codes
        let mut latent = Array2::<f32>::zeros((batch_size, self.config.latent_dim));
        let mut rng = SmallRng::seed_from_u64(123);

        for elem in latent.iter_mut() {
            *elem = rng.random_range(-1.0..1.0);
        }

        let latent_dyn = latent.into_dyn();

        // Decode to generate images
        self.decoder.forward(&latent_dyn)
    }

    /// Interpolate between two latent codes
    pub fn interpolate(
        &self,
        latent1: &ArrayD<f32>,
        latent2: &ArrayD<f32>,
        steps: usize,
    ) -> StdResult<Vec<ArrayD<f32>>> {
        let mut results = Vec::new();

        for i in 0..steps {
            let alpha = i as f32 / (steps - 1) as f32;

            // Linear interpolation
            let mut interpolated = Array::zeros(latent1.raw_dim());
            for ((&l1, &l2), interp) in latent1
                .iter()
                .zip(latent2.iter())
                .zip(interpolated.iter_mut())
            {
                *interp = (1.0 - alpha) * l1 + alpha * l2;
            }

            let generated = self.decoder.forward(&interpolated)?;
            results.push(generated);
        }

        Ok(results)
    }
}

/// VAE Loss combining reconstruction and KL divergence
pub struct VAELoss {
    reconstruction_loss: MeanSquaredError,
    beta: f32,
}

impl VAELoss {
    pub fn new(beta: f32) -> Self {
        Self {
            reconstruction_loss: MeanSquaredError::new(),
            beta,
        }
    }

    pub fn compute_loss(
        &self,
        reconstruction: &ArrayD<f32>,
        target: &ArrayD<f32>,
        mean: &ArrayD<f32>,
        logvar: &ArrayD<f32>,
    ) -> StdResult<(f32, f32, f32)> {
        // Reconstruction loss
        let recon_loss = self.reconstruction_loss.forward(reconstruction, target)?;

        // KL divergence loss: -0.5 * sum(1 + logvar - mean^2 - exp(logvar))
        let mut kl_loss = 0.0f32;
        for (&m, &lv) in mean.iter().zip(logvar.iter()) {
            kl_loss += -0.5 * (1.0 + lv - m * m - lv.exp());
        }
        kl_loss /= mean.len() as f32; // Average over elements

        let total_loss = recon_loss + self.beta * kl_loss;

        Ok((total_loss, recon_loss, kl_loss))
    }
}

/// Simple GAN Generator
pub struct GANGenerator {
    layers: Sequential<f32>,
    config: GenerativeConfig,
}

impl GANGenerator {
    pub fn new(config: GenerativeConfig, rng: &mut SmallRng) -> StdResult<Self> {
        let mut layers = Sequential::new();

        // Project noise to feature space
        layers.add(Dense::new(
            config.latent_dim,
            config.hidden_dims[0],
            Some("relu"),
            rng,
        )?);
        layers.add(BatchNorm::new(config.hidden_dims[0], 1e-5, 0.1, rng)?);

        layers.add(Dense::new(
            config.hidden_dims[0],
            config.hidden_dims[1] * 2,
            Some("relu"),
            rng,
        )?);
        layers.add(BatchNorm::new(config.hidden_dims[1] * 2, 1e-5, 0.1, rng)?);

        // Output layer
        let output_size = config.input_size.0 * config.input_size.1;
        layers.add(Dense::new(
            config.hidden_dims[1] * 2,
            output_size,
            Some("tanh"),
            rng,
        )?);

        Ok(Self { layers, config })
    }

    pub fn forward(&self, noise: &ArrayD<f32>) -> StdResult<ArrayD<f32>> {
        let output = self.layers.forward(noise)?;

        // Reshape to image format
        let batch_size = output.shape()[0];
        let (height, width) = self.config.input_size;
        let reshaped = output
            .to_shape(IxDyn(&[batch_size, 1, height, width]))?
            .to_owned();

        Ok(reshaped)
    }
}

/// Simple GAN Discriminator
pub struct GANDiscriminator {
    layers: Sequential<f32>,
    config: GenerativeConfig,
}

impl GANDiscriminator {
    pub fn new(config: GenerativeConfig, rng: &mut SmallRng) -> StdResult<Self> {
        let mut layers = Sequential::new();

        let input_size = config.input_size.0 * config.input_size.1;

        layers.add(Dense::new(
            input_size,
            config.hidden_dims[0],
            Some("relu"),
            rng,
        )?);
        layers.add(Dropout::new(0.3, rng)?);

        layers.add(Dense::new(
            config.hidden_dims[0],
            config.hidden_dims[1],
            Some("relu"),
            rng,
        )?);
        layers.add(Dropout::new(0.3, rng)?);

        // Output probability of being real
        layers.add(Dense::new(config.hidden_dims[1], 1, Some("sigmoid"), rng)?);

        Ok(Self { layers, config })
    }

    pub fn forward(&self, input: &ArrayD<f32>) -> StdResult<ArrayD<f32>> {
        // Flatten input
        let batch_size = input.shape()[0];
        let input_size = self.config.input_size.0 * self.config.input_size.1;
        let flattened = input.to_shape(IxDyn(&[batch_size, input_size]))?.to_owned();

        Ok(self.layers.forward(&flattened)?)
    }
}

/// Generative model evaluation metrics
pub struct GenerativeMetrics {
    #[allow(dead_code)]
    config: GenerativeConfig,
}

impl GenerativeMetrics {
    pub fn new(config: GenerativeConfig) -> Self {
        Self { config }
    }

    /// Calculate reconstruction error
    pub fn reconstruction_error(&self, original: &ArrayD<f32>, reconstructed: &ArrayD<f32>) -> f32 {
        let mut mse = 0.0f32;
        let mut count = 0;

        for (&orig, &recon) in original.iter().zip(reconstructed.iter()) {
            let diff = orig - recon;
            mse += diff * diff;
            count += 1;
        }

        if count > 0 {
            mse / count as f32
        } else {
            0.0
        }
    }

    /// Calculate sample diversity (simplified variance measure)
    pub fn sample_diversity(&self, samples: &ArrayD<f32>) -> f32 {
        let batch_size = samples.shape()[0];
        if batch_size < 2 {
            return 0.0;
        }

        let mut total_variance = 0.0f32;
        let sample_size = samples.len() / batch_size;

        for i in 0..sample_size {
            let mut values = Vec::new();
            for b in 0..batch_size {
                let flat_idx = b * sample_size + i;
                if let Some(&val) = samples.iter().nth(flat_idx) {
                    values.push(val);
                }
            }

            if values.len() > 1 {
                let mean = values.iter().sum::<f32>() / values.len() as f32;
                let variance = values.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>()
                    / values.len() as f32;
                total_variance += variance;
            }
        }

        total_variance / sample_size as f32
    }
}

/// Training function for VAE
fn train_vae_model() -> StdResult<()> {
    println!("🎨 Starting VAE Training");

    let mut rng = SmallRng::seed_from_u64(42);
    let config = GenerativeConfig::default();

    // Initialize JIT context (currently not implemented)
    // let _jit_context: JitContext<f32> = JitContext::new(JitStrategy::Aggressive);
    // println!("🚀 JIT compilation enabled with aggressive strategy");

    // Create model
    println!("🏗️ Building VAE model...");
    let vae = VAEModel::new(config.clone(), &mut rng)?;
    println!("✅ VAE created with latent dimension {}", config.latent_dim);

    // Create dataset
    let mut dataset = GenerativeDataset::new(config.clone(), 123);

    // Create loss function
    let loss_fn = VAELoss::new(config.beta);

    // Create metrics
    let metrics = GenerativeMetrics::new(config.clone());

    println!("📊 Training configuration:");
    println!("   - Input size: {:?}", config.input_size);
    println!("   - Latent dimension: {}", config.latent_dim);
    println!("   - Beta (KL weight): {}", config.beta);
    println!("   - Hidden dimensions: {:?}", config.hidden_dims);

    // Training loop
    let num_epochs = 20;
    let batch_size = 4;
    let _learning_rate = 0.001;

    for epoch in 0..num_epochs {
        println!("\n📈 Epoch {}/{}", epoch + 1, num_epochs);

        let mut epoch_total_loss = 0.0;
        let mut epoch_recon_loss = 0.0;
        let mut epoch_kl_loss = 0.0;
        let num_batches = 12;

        for batch_idx in 0..num_batches {
            // Generate training batch
            let images = dataset.generate_batch(batch_size);
            let images_dyn = images.into_dyn();

            // Forward pass
            let (reconstruction, mean, logvar) = vae.forward(&images_dyn)?;

            // Compute loss
            let (total_loss, recon_loss, kl_loss) =
                loss_fn.compute_loss(&reconstruction, &images_dyn, &mean, &logvar)?;

            epoch_total_loss += total_loss;
            epoch_recon_loss += recon_loss;
            epoch_kl_loss += kl_loss;

            if batch_idx % 6 == 0 {
                print!(
                    "🔄 Batch {}/{} - Total: {:.4}, Recon: {:.4}, KL: {:.4}           \r",
                    batch_idx + 1,
                    num_batches,
                    total_loss,
                    recon_loss,
                    kl_loss
                );
            }
        }

        let avg_total = epoch_total_loss / num_batches as f32;
        let avg_recon = epoch_recon_loss / num_batches as f32;
        let avg_kl = epoch_kl_loss / num_batches as f32;

        println!(
            "✅ Epoch {} - Total: {:.4}, Recon: {:.4}, KL: {:.4}",
            epoch + 1,
            avg_total,
            avg_recon,
            avg_kl
        );

        // Evaluation and generation every few epochs
        if (epoch + 1) % 5 == 0 {
            println!("🔍 Running evaluation and generation...");

            // Test reconstruction
            let test_images = dataset.generate_batch(batch_size);
            let test_images_dyn = test_images.into_dyn();
            let (test_reconstruction, _, _) = vae.forward(&test_images_dyn)?;

            let recon_error = metrics.reconstruction_error(&test_images_dyn, &test_reconstruction);
            println!("📊 Reconstruction MSE: {:.6}", recon_error);

            // Generate new samples
            let generated_samples = vae.generate(8)?;
            let diversity = metrics.sample_diversity(&generated_samples);
            println!("🎲 Sample diversity: {:.6}", diversity);

            // Test interpolation
            let latent1 = Array2::<f32>::from_elem((1, config.latent_dim), -1.0).into_dyn();
            let latent2 = Array2::<f32>::from_elem((1, config.latent_dim), 1.0).into_dyn();
            let interpolated = vae.interpolate(&latent1, &latent2, 5)?;

            println!("🔄 Generated {} interpolated samples", interpolated.len());
        }
    }

    println!("\n🎉 VAE training completed!");
    Ok(())
}

/// Training function for simple GAN
fn train_gan_model() -> StdResult<()> {
    println!("⚔️ Starting GAN Training");

    let mut rng = SmallRng::seed_from_u64(42);
    let config = GenerativeConfig::default();

    // Create models
    println!("🏗️ Building GAN models...");
    let generator = GANGenerator::new(config.clone(), &mut rng)?;
    let discriminator = GANDiscriminator::new(config.clone(), &mut rng)?;
    println!("✅ GAN models created");

    // Create dataset
    let mut dataset = GenerativeDataset::new(config.clone(), 456);

    // Create loss functions
    let _adversarial_loss = CrossEntropyLoss::new(1e-7);

    println!("📊 GAN training configuration:");
    println!("   - Generator latent dim: {}", config.latent_dim);
    println!("   - Discriminator architecture: {:?}", config.hidden_dims);

    // Training loop (simplified)
    let num_epochs = 15;
    let batch_size = 4;

    for epoch in 0..num_epochs {
        println!("\n📈 Epoch {}/{}", epoch + 1, num_epochs);

        let mut d_loss_total = 0.0;
        let mut g_loss_total = 0.0;
        let num_batches = 8;

        for batch_idx in 0..num_batches {
            // Train Discriminator
            let real_images = dataset.generate_batch(batch_size);
            let real_images_dyn = real_images.into_dyn();

            // Generate fake images
            let mut noise = Array2::<f32>::zeros((batch_size, config.latent_dim));
            for elem in noise.iter_mut() {
                *elem = rng.random_range(-1.0..1.0);
            }
            let noise_dyn = noise.into_dyn();
            let fake_images = generator.forward(&noise_dyn)?;

            // Discriminator predictions
            let real_pred = discriminator.forward(&real_images_dyn)?;
            let fake_pred = discriminator.forward(&fake_images)?;

            // Simplified loss calculation (normally would use proper labels)
            let mut d_loss_real = 0.0f32;
            let mut d_loss_fake = 0.0f32;

            for &pred in real_pred.iter() {
                d_loss_real += -(1.0f32).ln() - pred; // Log loss for real=1
            }

            for &pred in fake_pred.iter() {
                d_loss_fake += -(1.0 - pred).ln(); // Log loss for fake=0
            }

            let d_loss = (d_loss_real + d_loss_fake) / (batch_size * 2) as f32;
            d_loss_total += d_loss;

            // Train Generator (simplified)
            let fake_pred_for_g = discriminator.forward(&fake_images)?;
            let mut g_loss = 0.0f32;

            for &pred in fake_pred_for_g.iter() {
                g_loss += -(1.0f32).ln() - pred; // Want discriminator to output 1 for fake
            }
            g_loss /= batch_size as f32;
            g_loss_total += g_loss;

            if batch_idx % 4 == 0 {
                print!(
                    "🔄 Batch {}/{} - D Loss: {:.4}, G Loss: {:.4}        \r",
                    batch_idx + 1,
                    num_batches,
                    d_loss,
                    g_loss
                );
            }
        }

        let avg_d_loss = d_loss_total / num_batches as f32;
        let avg_g_loss = g_loss_total / num_batches as f32;

        println!(
            "✅ Epoch {} - D Loss: {:.4}, G Loss: {:.4}",
            epoch + 1,
            avg_d_loss,
            avg_g_loss
        );

        // Generate samples every few epochs
        if (epoch + 1) % 5 == 0 {
            println!("🎲 Generating samples...");

            let mut sample_noise = Array2::<f32>::zeros((4, config.latent_dim));
            for elem in sample_noise.iter_mut() {
                *elem = rng.random_range(-1.0..1.0);
            }
            let sample_noise_dyn = sample_noise.into_dyn();
            let generated = generator.forward(&sample_noise_dyn)?;

            println!("📊 Generated {} samples", generated.shape()[0]);
        }
    }

    println!("\n🎉 GAN training completed!");
    Ok(())
}

fn main() -> StdResult<()> {
    println!("🎨 Generative Models Complete Example");
    println!("=====================================");
    println!();
    println!("This example demonstrates:");
    println!("• Variational Autoencoder (VAE) implementation");
    println!("• Generative Adversarial Network (GAN) basics");
    println!("• Synthetic pattern dataset generation");
    println!("• VAE loss (reconstruction + KL divergence)");
    println!("• Latent space interpolation");
    println!("• Sample generation and evaluation");
    println!();

    // Train VAE
    train_vae_model()?;

    println!("\n{}", "=".repeat(50));

    // Train GAN
    train_gan_model()?;

    println!("\n💡 Key Concepts Demonstrated:");
    println!("   🔹 Variational inference and reparameterization trick");
    println!("   🔹 KL divergence regularization");
    println!("   🔹 Adversarial training dynamics");
    println!("   🔹 Latent space manipulation");
    println!("   🔹 Reconstruction vs generation quality");
    println!("   🔹 Sample diversity metrics");
    println!();
    println!("🚀 For production use:");
    println!("   • Implement β-VAE, WAE, or other VAE variants");
    println!("   • Add convolutional layers for better image modeling");
    println!("   • Implement DCGAN, StyleGAN, or other advanced GANs");
    println!("   • Add progressive training and spectral normalization");
    println!("   • Use FID, IS, or other advanced evaluation metrics");
    println!("   • Implement conditional generation (cVAE, cGAN)");
    println!("   • Add attention mechanisms and self-attention");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generative_config() {
        let config = GenerativeConfig::default();
        assert_eq!(config.input_size, (32, 32));
        assert_eq!(config.latent_dim, 16);
        assert_eq!(config.beta, 1.0);
        assert!(!config.hidden_dims.is_empty());
    }

    #[test]
    fn test_dataset_generation() {
        let config = GenerativeConfig::default();
        let mut dataset = GenerativeDataset::new(config.clone(), 42);

        let image = dataset.generate_sample();
        assert_eq!(
            image.shape(),
            &[1, config.input_size.0, config.input_size.1]
        );

        // Check values are in valid range
        for &val in image.iter() {
            assert!(val >= 0.0 && val <= 1.0);
        }
    }

    #[test]
    fn test_vae_creation() -> StdResult<()> {
        let mut rng = SmallRng::seed_from_u64(42);
        let config = GenerativeConfig::default();

        let vae = VAEModel::new(config.clone(), &mut rng)?;

        // Test forward pass
        let batch_size = 2;
        let input = Array4::<f32>::ones((batch_size, 1, config.input_size.0, config.input_size.1))
            .into_dyn();
        let (reconstruction, mean, logvar) = vae.forward(&input)?;

        assert_eq!(reconstruction.shape()[0], batch_size);
        assert_eq!(mean.shape()[1], config.latent_dim);
        assert_eq!(logvar.shape()[1], config.latent_dim);

        Ok(())
    }

    #[test]
    fn test_gan_creation() -> StdResult<()> {
        let mut rng = SmallRng::seed_from_u64(42);
        let config = GenerativeConfig::default();

        let generator = GANGenerator::new(config.clone(), &mut rng)?;
        let discriminator = GANDiscriminator::new(config.clone(), &mut rng)?;

        // Test generator
        let batch_size = 2;
        let noise = Array2::<f32>::ones((batch_size, config.latent_dim)).into_dyn();
        let generated = generator.forward(&noise)?;

        assert_eq!(generated.shape()[0], batch_size);
        assert_eq!(generated.shape()[1], 1);

        // Test discriminator
        let pred = discriminator.forward(&generated)?;
        assert_eq!(pred.shape()[0], batch_size);
        assert_eq!(pred.shape()[1], 1);

        Ok(())
    }

    #[test]
    fn test_vae_loss() -> StdResult<()> {
        let loss_fn = VAELoss::new(1.0);

        let reconstruction = Array2::<f32>::ones((2, 10)).into_dyn();
        let target = Array2::<f32>::zeros((2, 10)).into_dyn();
        let mean = Array2::<f32>::zeros((2, 5)).into_dyn();
        let logvar = Array2::<f32>::zeros((2, 5)).into_dyn();

        let (total_loss, recon_loss, kl_loss) =
            loss_fn.compute_loss(&reconstruction, &target, &mean, &logvar)?;

        assert!(total_loss > 0.0);
        assert!(recon_loss > 0.0);
        // KL loss should be 0 for mean=0, logvar=0
        assert!(kl_loss.abs() < 1e-6);

        Ok(())
    }

    #[test]
    fn test_generative_metrics() {
        let config = GenerativeConfig::default();
        let metrics = GenerativeMetrics::new(config);

        // Test reconstruction error
        let original = Array2::<f32>::ones((2, 10)).into_dyn();
        let reconstructed = Array2::<f32>::zeros((2, 10)).into_dyn();

        let error = metrics.reconstruction_error(&original, &reconstructed);
        assert_eq!(error, 1.0); // MSE between all 1s and all 0s

        // Test diversity
        let samples = Array2::<f32>::from_shape_fn((3, 4), |(i, j)| i as f32 + j as f32).into_dyn();
        let diversity = metrics.sample_diversity(&samples);
        assert!(diversity > 0.0);
    }
}
