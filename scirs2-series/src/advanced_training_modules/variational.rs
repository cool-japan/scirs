//! Variational Autoencoder for Time Series
//!
//! This module provides implementations for variational autoencoders specifically designed
//! for time series data, including uncertainty quantification and probabilistic modeling.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::Debug;

use crate::error::Result;

/// Type alias for complex encoder weights return type
type EncoderWeights<F> = (
    Array2<F>,
    Array1<F>,
    Array2<F>,
    Array1<F>,
    Array2<F>,
    Array1<F>,
);

/// Variational Autoencoder for Time Series with Uncertainty Quantification
#[derive(Debug)]
pub struct TimeSeriesVAE<F: Float + Debug + scirs2_core::ndarray::ScalarOperand> {
    /// Encoder parameters
    encoder_params: Array2<F>,
    /// Decoder parameters
    decoder_params: Array2<F>,
    /// Latent dimension
    latent_dim: usize,
    /// Input sequence length
    seq_len: usize,
    /// Feature dimension
    feature_dim: usize,
    /// Hidden dimensions
    encoder_hidden: usize,
    decoder_hidden: usize,
}

impl<F: Float + Debug + Clone + FromPrimitive + scirs2_core::ndarray::ScalarOperand>
    TimeSeriesVAE<F>
{
    /// Create new Time Series VAE
    pub fn new(
        seq_len: usize,
        feature_dim: usize,
        latent_dim: usize,
        encoder_hidden: usize,
        decoder_hidden: usize,
    ) -> Self {
        let input_size = seq_len * feature_dim;

        // Initialize encoder parameters (input -> _hidden -> latent_mean, latent_logvar)
        let encoder_param_count = input_size * encoder_hidden
            + encoder_hidden
            + encoder_hidden * latent_dim * 2
            + latent_dim * 2;
        let mut encoder_params = Array2::zeros((1, encoder_param_count));

        // Initialize decoder parameters (latent -> _hidden -> output)
        let decoder_param_count =
            latent_dim * decoder_hidden + decoder_hidden + decoder_hidden * input_size + input_size;
        let mut decoder_params = Array2::zeros((1, decoder_param_count));

        // Xavier initialization
        let encoder_scale = F::from(2.0).unwrap() / F::from(input_size + latent_dim).unwrap();
        let decoder_scale = F::from(2.0).unwrap() / F::from(latent_dim + input_size).unwrap();

        for i in 0..encoder_param_count {
            let val = ((i * 19) % 1000) as f64 / 1000.0 - 0.5;
            encoder_params[[0, i]] = F::from(val).unwrap() * encoder_scale.sqrt();
        }

        for i in 0..decoder_param_count {
            let val = ((i * 31) % 1000) as f64 / 1000.0 - 0.5;
            decoder_params[[0, i]] = F::from(val).unwrap() * decoder_scale.sqrt();
        }

        Self {
            encoder_params,
            decoder_params,
            latent_dim,
            seq_len,
            feature_dim,
            encoder_hidden,
            decoder_hidden,
        }
    }

    /// Encode time series to latent distribution
    pub fn encode(&self, input: &Array2<F>) -> Result<(Array1<F>, Array1<F>)> {
        // Flatten input
        let input_flat = self.flatten_input(input);

        // Extract encoder weights
        let (w1, b1, w_mean, b_mean, w_logvar, b_logvar) = self.extract_encoder_weights();

        // Forward through encoder
        let mut hidden = Array1::zeros(self.encoder_hidden);
        for i in 0..self.encoder_hidden {
            let mut sum = b1[i];
            for j in 0..input_flat.len() {
                sum = sum + w1[[i, j]] * input_flat[j];
            }
            hidden[i] = self.relu(sum);
        }

        // Compute latent mean and log variance
        let mut latent_mean = Array1::zeros(self.latent_dim);
        let mut latent_logvar = Array1::zeros(self.latent_dim);

        for i in 0..self.latent_dim {
            let mut mean_sum = b_mean[i];
            let mut logvar_sum = b_logvar[i];

            for j in 0..self.encoder_hidden {
                mean_sum = mean_sum + w_mean[[i, j]] * hidden[j];
                logvar_sum = logvar_sum + w_logvar[[i, j]] * hidden[j];
            }

            latent_mean[i] = mean_sum;
            latent_logvar[i] = logvar_sum;
        }

        Ok((latent_mean, latent_logvar))
    }

    /// Sample from latent distribution using reparameterization trick
    pub fn reparameterize(&self, mean: &Array1<F>, logvar: &Array1<F>) -> Array1<F> {
        let mut sample = Array1::zeros(self.latent_dim);

        for i in 0..self.latent_dim {
            // Sample from standard normal (simplified)
            let eps = F::from(((i * 47) % 1000) as f64 / 1000.0 - 0.5).unwrap();
            let std = (logvar[i] / F::from(2.0).unwrap()).exp();
            sample[i] = mean[i] + std * eps;
        }

        sample
    }

    /// Decode latent representation to time series
    pub fn decode(&self, latent: &Array1<F>) -> Result<Array2<F>> {
        // Extract decoder weights
        let (w1, b1, w2, b2) = self.extract_decoder_weights();

        // Forward through decoder
        let mut hidden = Array1::zeros(self.decoder_hidden);
        for i in 0..self.decoder_hidden {
            let mut sum = b1[i];
            for j in 0..self.latent_dim {
                sum = sum + w1[[i, j]] * latent[j];
            }
            hidden[i] = self.relu(sum);
        }

        // Generate output
        let output_size = self.seq_len * self.feature_dim;
        let mut output_flat = Array1::zeros(output_size);

        for i in 0..output_size {
            let mut sum = b2[i];
            for j in 0..self.decoder_hidden {
                sum = sum + w2[[i, j]] * hidden[j];
            }
            output_flat[i] = sum;
        }

        // Reshape to time series format
        self.unflatten_output(&output_flat)
    }

    /// Full forward pass with reconstruction and KL divergence
    pub fn forward(&self, input: &Array2<F>) -> Result<VAEOutput<F>> {
        let (latent_mean, latent_logvar) = self.encode(input)?;
        let latent_sample = self.reparameterize(&latent_mean, &latent_logvar);
        let reconstruction = self.decode(&latent_sample)?;

        // Compute KL divergence
        let mut kl_div = F::zero();
        for i in 0..self.latent_dim {
            let mean_sq = latent_mean[i] * latent_mean[i];
            let var = latent_logvar[i].exp();
            kl_div = kl_div + mean_sq + var - latent_logvar[i] - F::one();
        }
        kl_div = kl_div / F::from(2.0).unwrap();

        // Compute reconstruction loss
        let mut recon_loss = F::zero();
        let (seq_len, feature_dim) = input.dim();

        for i in 0..seq_len {
            for j in 0..feature_dim {
                let diff = reconstruction[[i, j]] - input[[i, j]];
                recon_loss = recon_loss + diff * diff;
            }
        }
        recon_loss = recon_loss / F::from(seq_len * feature_dim).unwrap();

        Ok(VAEOutput {
            reconstruction,
            latent_mean,
            latent_logvar,
            latent_sample,
            reconstruction_loss: recon_loss,
            kl_divergence: kl_div,
        })
    }

    /// Generate new time series by sampling from latent space
    pub fn generate(&self, numsamples: usize) -> Result<Vec<Array2<F>>> {
        let mut _samples = Vec::new();

        for i in 0..numsamples {
            // Sample from prior distribution (standard normal)
            let mut latent = Array1::zeros(self.latent_dim);
            for j in 0..self.latent_dim {
                let val = ((i * 53 + j * 29) % 1000) as f64 / 1000.0 - 0.5;
                latent[j] = F::from(val).unwrap();
            }

            let generated = self.decode(&latent)?;
            _samples.push(generated);
        }

        Ok(_samples)
    }

    /// Estimate uncertainty by sampling multiple reconstructions
    pub fn estimate_uncertainty(
        &self,
        input: &Array2<F>,
        num_samples: usize,
    ) -> Result<(Array2<F>, Array2<F>)> {
        let (latent_mean, latent_logvar) = self.encode(input)?;
        let mut reconstructions = Vec::new();

        // Generate multiple _samples
        for _ in 0..num_samples {
            let latent_sample = self.reparameterize(&latent_mean, &latent_logvar);
            let reconstruction = self.decode(&latent_sample)?;
            reconstructions.push(reconstruction);
        }

        // Compute mean and standard deviation
        let (seq_len, feature_dim) = input.dim();
        let mut mean_recon = Array2::zeros((seq_len, feature_dim));
        let mut std_recon = Array2::zeros((seq_len, feature_dim));

        // Compute mean
        for recon in &reconstructions {
            for i in 0..seq_len {
                for j in 0..feature_dim {
                    mean_recon[[i, j]] = mean_recon[[i, j]] + recon[[i, j]];
                }
            }
        }

        let num_samples_f = F::from(num_samples).unwrap();
        for i in 0..seq_len {
            for j in 0..feature_dim {
                mean_recon[[i, j]] = mean_recon[[i, j]] / num_samples_f;
            }
        }

        // Compute standard deviation
        for recon in &reconstructions {
            for i in 0..seq_len {
                for j in 0..feature_dim {
                    let diff = recon[[i, j]] - mean_recon[[i, j]];
                    std_recon[[i, j]] = std_recon[[i, j]] + diff * diff;
                }
            }
        }

        for i in 0..seq_len {
            for j in 0..feature_dim {
                let val: F = std_recon[[i, j]] / num_samples_f;
                std_recon[[i, j]] = val.sqrt();
            }
        }

        Ok((mean_recon, std_recon))
    }

    // Helper methods
    fn flatten_input(&self, input: &Array2<F>) -> Array1<F> {
        let (seq_len, feature_dim) = input.dim();
        let mut flat = Array1::zeros(seq_len * feature_dim);

        for i in 0..seq_len {
            for j in 0..feature_dim {
                flat[i * feature_dim + j] = input[[i, j]];
            }
        }

        flat
    }

    fn unflatten_output(&self, output: &Array1<F>) -> Result<Array2<F>> {
        let mut result = Array2::zeros((self.seq_len, self.feature_dim));

        for i in 0..self.seq_len {
            for j in 0..self.feature_dim {
                let idx = i * self.feature_dim + j;
                if idx < output.len() {
                    result[[i, j]] = output[idx];
                }
            }
        }

        Ok(result)
    }

    fn extract_encoder_weights(&self) -> EncoderWeights<F> {
        let param_vec = self.encoder_params.row(0);
        let input_size = self.seq_len * self.feature_dim;
        let mut idx = 0;

        // W1: input_size x encoder_hidden
        let mut w1 = Array2::zeros((self.encoder_hidden, input_size));
        for i in 0..self.encoder_hidden {
            for j in 0..input_size {
                w1[[i, j]] = param_vec[idx];
                idx += 1;
            }
        }

        // b1: encoder_hidden
        let mut b1 = Array1::zeros(self.encoder_hidden);
        for i in 0..self.encoder_hidden {
            b1[i] = param_vec[idx];
            idx += 1;
        }

        // W_mean: encoder_hidden x latent_dim
        let mut w_mean = Array2::zeros((self.latent_dim, self.encoder_hidden));
        for i in 0..self.latent_dim {
            for j in 0..self.encoder_hidden {
                w_mean[[i, j]] = param_vec[idx];
                idx += 1;
            }
        }

        // b_mean: latent_dim
        let mut b_mean = Array1::zeros(self.latent_dim);
        for i in 0..self.latent_dim {
            b_mean[i] = param_vec[idx];
            idx += 1;
        }

        // W_logvar: encoder_hidden x latent_dim
        let mut w_logvar = Array2::zeros((self.latent_dim, self.encoder_hidden));
        for i in 0..self.latent_dim {
            for j in 0..self.encoder_hidden {
                w_logvar[[i, j]] = param_vec[idx];
                idx += 1;
            }
        }

        // b_logvar: latent_dim
        let mut b_logvar = Array1::zeros(self.latent_dim);
        for i in 0..self.latent_dim {
            b_logvar[i] = param_vec[idx];
            idx += 1;
        }

        (w1, b1, w_mean, b_mean, w_logvar, b_logvar)
    }

    fn extract_decoder_weights(&self) -> (Array2<F>, Array1<F>, Array2<F>, Array1<F>) {
        let param_vec = self.decoder_params.row(0);
        let output_size = self.seq_len * self.feature_dim;
        let mut idx = 0;

        // W1: latent_dim x decoder_hidden
        let mut w1 = Array2::zeros((self.decoder_hidden, self.latent_dim));
        for i in 0..self.decoder_hidden {
            for j in 0..self.latent_dim {
                w1[[i, j]] = param_vec[idx];
                idx += 1;
            }
        }

        // b1: decoder_hidden
        let mut b1 = Array1::zeros(self.decoder_hidden);
        for i in 0..self.decoder_hidden {
            b1[i] = param_vec[idx];
            idx += 1;
        }

        // W2: decoder_hidden x output_size
        let mut w2 = Array2::zeros((output_size, self.decoder_hidden));
        for i in 0..output_size {
            for j in 0..self.decoder_hidden {
                w2[[i, j]] = param_vec[idx];
                idx += 1;
            }
        }

        // b2: output_size
        let mut b2 = Array1::zeros(output_size);
        for i in 0..output_size {
            b2[i] = param_vec[idx];
            idx += 1;
        }

        (w1, b1, w2, b2)
    }

    fn relu(&self, x: F) -> F {
        x.max(F::zero())
    }
}

/// VAE output structure
#[derive(Debug, Clone)]
pub struct VAEOutput<F: Float + Debug> {
    /// Reconstructed time series
    pub reconstruction: Array2<F>,
    /// Latent mean
    pub latent_mean: Array1<F>,
    /// Latent log variance
    pub latent_logvar: Array1<F>,
    /// Latent sample
    pub latent_sample: Array1<F>,
    /// Reconstruction loss
    pub reconstruction_loss: F,
    /// KL divergence
    pub kl_divergence: F,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_vae_creation() {
        let vae = TimeSeriesVAE::<f64>::new(10, 3, 5, 16, 16);
        assert_eq!(vae.seq_len, 10);
        assert_eq!(vae.feature_dim, 3);
        assert_eq!(vae.latent_dim, 5);
        assert_eq!(vae.encoder_hidden, 16);
        assert_eq!(vae.decoder_hidden, 16);
    }

    #[test]
    fn test_vae_encode_decode() {
        let vae = TimeSeriesVAE::<f64>::new(5, 2, 3, 8, 8);
        let input =
            Array2::from_shape_vec((5, 2), (0..10).map(|i| i as f64 * 0.1).collect()).unwrap();

        let (mean, logvar) = vae.encode(&input).unwrap();
        assert_eq!(mean.len(), 3);
        assert_eq!(logvar.len(), 3);

        let sample = vae.reparameterize(&mean, &logvar);
        assert_eq!(sample.len(), 3);

        let decoded = vae.decode(&sample).unwrap();
        assert_eq!(decoded.dim(), (5, 2));
    }

    #[test]
    fn test_vae_forward() {
        let vae = TimeSeriesVAE::<f64>::new(4, 2, 3, 8, 8);
        let input =
            Array2::from_shape_vec((4, 2), (0..8).map(|i| i as f64 * 0.1).collect()).unwrap();

        let output = vae.forward(&input).unwrap();
        assert_eq!(output.reconstruction.dim(), (4, 2));
        assert_eq!(output.latent_mean.len(), 3);
        assert_eq!(output.latent_logvar.len(), 3);
        assert_eq!(output.latent_sample.len(), 3);
        assert!(output.reconstruction_loss >= 0.0);
        assert!(output.kl_divergence >= 0.0);
    }

    #[test]
    fn test_vae_uncertainty_estimation() {
        let vae = TimeSeriesVAE::<f64>::new(3, 2, 2, 6, 6);
        let input =
            Array2::from_shape_vec((3, 2), (0..6).map(|i| i as f64 * 0.2).collect()).unwrap();

        let (mean_recon, std_recon) = vae.estimate_uncertainty(&input, 5).unwrap();
        assert_eq!(mean_recon.dim(), (3, 2));
        assert_eq!(std_recon.dim(), (3, 2));

        // Check that standard deviations are non-negative
        for &val in std_recon.iter() {
            assert!(val >= 0.0);
        }
    }

    #[test]
    fn test_vae_generation() {
        let vae = TimeSeriesVAE::<f64>::new(4, 2, 3, 8, 8);
        let samples = vae.generate(3).unwrap();

        assert_eq!(samples.len(), 3);
        for sample in samples {
            assert_eq!(sample.dim(), (4, 2));
        }
    }
}
