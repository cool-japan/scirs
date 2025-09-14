//! Noise injection scheduler
//!
//! This module provides a learning rate scheduler that adds noise to the learning rate
//! to help escape local minima and improve exploration during training.

use ndarray::ScalarOperand;
use num_traits::{Float, NumCast};
use rand::Rng;
use std::fmt::Debug;

use super::LearningRateScheduler;

/// Noise distribution types for learning rate
#[derive(Debug, Clone, Copy)]
pub enum NoiseDistribution<A: Float> {
    /// Uniform noise in the range [min, max]
    Uniform {
        /// Minimum noise value
        min: A,
        /// Maximum noise value
        max: A,
    },
    /// Gaussian noise with specified mean and standard deviation
    Gaussian {
        /// Mean value of noise distribution
        mean: A,
        /// Standard deviation of noise distribution
        std_dev: A,
    },
    /// Cyclical noise that oscillates according to a sine wave with specified amplitude
    Cyclical {
        /// Maximum amplitude of oscillation
        amplitude: A,
        /// Number of steps to complete one full cycle
        period: usize,
    },
    /// Decaying noise that decreases over time to a minimum value
    Decaying {
        /// Initial scale factor for noise at step 0
        initial_scale: A,
        /// Final scale factor for noise after decay_steps
        final_scale: A,
        /// Number of steps over which to decay from initial to final scale
        decay_steps: usize,
    },
}

/// A learning rate scheduler that injects noise into the base learning rate
pub struct NoiseInjectionScheduler<A, S>
where
    A: Float + Debug + ScalarOperand,
    S: LearningRateScheduler<A>,
{
    /// The base scheduler to add noise to
    base_scheduler: S,
    /// The noise distribution
    noise_dist: NoiseDistribution<A>,
    /// Current step number
    step_count: usize,
    /// Random number generator
    rng: scirs2_core::random::Random,
    /// Minimum learning rate to ensure training stability
    min_lr: A,
}

impl<A, S> NoiseInjectionScheduler<A, S>
where
    A: Float + Debug + ScalarOperand,
    S: LearningRateScheduler<A>,
{
    /// Create a new noise injection scheduler
    ///
    /// # Arguments
    ///
    /// * `base_scheduler` - The base scheduler to add noise to
    /// * `noise_dist` - The noise distribution to use
    /// * `min_lr` - The minimum learning rate allowed (to ensure stability)
    ///
    /// # Example
    ///
    /// ```
    /// use scirs2_optim::schedulers::{
    ///     ExponentialDecay, NoiseDistribution, NoiseInjectionScheduler, LearningRateScheduler
    /// };
    ///
    /// // Create a base scheduler
    /// let base_scheduler = ExponentialDecay::new(0.1, 0.9, 10);
    ///
    /// // Create a noise injection scheduler with uniform noise
    /// let mut scheduler = NoiseInjectionScheduler::new(
    ///     base_scheduler,
    ///     NoiseDistribution::Uniform { min: -0.01, max: 0.01 },
    ///     0.001, // Minimum learning rate
    /// );
    ///
    /// // Get the learning rate (will be 0.1 plus some noise)
    /// let lr = scheduler.get_learning_rate();
    /// assert!(lr >= 0.001); // Learning rate should be at least min_lr
    /// ```
    pub fn new(base_scheduler: S, noise_dist: NoiseDistribution<A>, min_lr: A) -> Self {
        Self {
            base_scheduler,
            noise_dist,
            step_count: 0,
            rng: scirs2_core::random::rng(),
            min_lr,
        }
    }

    /// Sample noise from the configured distribution
    fn sample_noise(&mut self) -> A {
        match self.noise_dist {
            NoiseDistribution::Uniform { min, max } => {
                let min_f64 = min.to_f64().unwrap();
                let max_f64 = max.to_f64().unwrap();
                let sample = self.rng.gen_range(min_f64..max_f64);
                <A as NumCast>::from(sample).unwrap()
            }
            NoiseDistribution::Gaussian { mean, std_dev } => {
                let mean_f64 = mean.to_f64().unwrap();
                let std_dev_f64 = std_dev.to_f64().unwrap();
                // Box-Muller transformation for Gaussian
                let u1: f64 = self.rng.gen_range(0.0..1.0);
                let u2: f64 = self.rng.gen_range(0.0..1.0);
                let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                let sample = mean_f64 + std_dev_f64 * z0;
                <A as NumCast>::from(sample).unwrap()
            }
            NoiseDistribution::Cyclical { amplitude, period } => {
                let period_f: A = <A as NumCast>::from(period).unwrap();
                let step: A = <A as NumCast>::from(self.step_count).unwrap();
                let angle: A = <A as NumCast>::from(2.0).unwrap()
                    * <A as NumCast>::from(std::f64::consts::PI).unwrap()
                    * (step / period_f);
                amplitude * angle.sin()
            }
            NoiseDistribution::Decaying {
                initial_scale,
                final_scale,
                decay_steps,
            } => {
                let decay_steps_a: A = <A as NumCast>::from(decay_steps).unwrap();
                let decay_steps_usize: usize = <usize as NumCast>::from(decay_steps).unwrap();
                let step_count_min = self.step_count.min(decay_steps_usize);
                let step: A = <A as NumCast>::from(step_count_min).unwrap();
                let scale = initial_scale - (step / decay_steps_a) * (initial_scale - final_scale);

                // Sample from uniform distribution and scale by the decaying factor
                let sample = self.rng.gen_range(-1.0..1.0);
                scale * NumCast::from(sample).unwrap()
            }
        }
    }
}

impl<A, S> LearningRateScheduler<A> for NoiseInjectionScheduler<A, S>
where
    A: Float + Debug + ScalarOperand,
    S: LearningRateScheduler<A>,
{
    fn get_learning_rate(&self) -> A {
        // Get the base learning rate from the underlying scheduler
        let base_lr = self.base_scheduler.get_learning_rate();

        // Use fresh RNG to sample noise since get_learning_rate takes &self
        let mut rand_rng = scirs2_core::random::rng();
        let noise = match self.noise_dist {
            NoiseDistribution::Uniform { min, max } => {
                let min_f64 = min.to_f64().unwrap();
                let max_f64 = max.to_f64().unwrap();
                let sample = rand_rng.gen_range(min_f64..max_f64);
                <A as NumCast>::from(sample).unwrap()
            }
            NoiseDistribution::Gaussian { mean, std_dev } => {
                let mean_f64 = mean.to_f64().unwrap();
                let std_dev_f64 = std_dev.to_f64().unwrap();
                // Box-Muller transformation
                let u1: f64 = rand_rng.gen_range(0.0..1.0);
                let u2: f64 = rand_rng.gen_range(0.0..1.0);
                let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                let sample = mean_f64 + std_dev_f64 * z0;
                <A as NumCast>::from(sample).unwrap()
            }
            NoiseDistribution::Cyclical { amplitude, period } => {
                let period_f: A = <A as NumCast>::from(period).unwrap();
                let step: A = <A as NumCast>::from(self.step_count).unwrap();
                let angle: A = <A as NumCast>::from(2.0).unwrap()
                    * <A as NumCast>::from(std::f64::consts::PI).unwrap()
                    * (step / period_f);
                amplitude * angle.sin()
            }
            NoiseDistribution::Decaying {
                initial_scale,
                final_scale,
                decay_steps,
            } => {
                let decay_steps_a: A = <A as NumCast>::from(decay_steps).unwrap();
                let decay_steps_usize: usize = <usize as NumCast>::from(decay_steps).unwrap();
                let step_count_min = self.step_count.min(decay_steps_usize);
                let step: A = <A as NumCast>::from(step_count_min).unwrap();
                let scale = initial_scale - (step / decay_steps_a) * (initial_scale - final_scale);

                // Sample from uniform distribution and scale by the decaying factor
                let sample = rand_rng.gen_range(-1.0..1.0);
                scale * NumCast::from(sample).unwrap()
            }
        };

        // Add noise to base learning rate and ensure it doesn't go below min_lr
        (base_lr + noise).max(self.min_lr)
    }

    fn step(&mut self) -> A {
        // Step the base scheduler
        self.base_scheduler.step();

        // Sample noise for the current step
        let noise = self.sample_noise();

        // Increment the step counter
        self.step_count += 1;

        // Get the base learning rate
        let base_lr = self.base_scheduler.get_learning_rate();

        // Add noise to base learning rate and ensure it doesn't go below min_lr
        (base_lr + noise).max(self.min_lr)
    }

    fn reset(&mut self) {
        self.base_scheduler.reset();
        self.step_count = 0;
    }
}

// Only implement Clone for NoiseInjectionScheduler when S is Clone
impl<A, S> Clone for NoiseInjectionScheduler<A, S>
where
    A: Float + Debug + ScalarOperand,
    S: LearningRateScheduler<A> + Clone,
{
    fn clone(&self) -> Self {
        Self {
            base_scheduler: self.base_scheduler.clone(),
            noise_dist: self.noise_dist,
            step_count: self.step_count,
            rng: scirs2_core::random::rng(),
            min_lr: self.min_lr,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schedulers::ConstantScheduler;

    #[test]
    fn test_uniform_noise() {
        // Create a constant base scheduler
        let base_scheduler = ConstantScheduler::new(0.1);

        // Create a noise injection scheduler with uniform noise
        let mut scheduler = NoiseInjectionScheduler::new(
            base_scheduler,
            NoiseDistribution::Uniform {
                min: -0.02,
                max: 0.02,
            },
            0.001,
        );

        // Get multiple learning rates and check they are within expected range
        let mut rates = Vec::with_capacity(100);
        for _ in 0..100 {
            rates.push(scheduler.step());
        }

        // Check that learning rates are within expected range
        for &rate in &rates {
            assert!((0.08..=0.12).contains(&rate));
        }

        // Check that there is some variation in the learning rates
        let mean = rates.iter().sum::<f64>() / rates.len() as f64;
        let variance = rates.iter().map(|&r| (r - mean).powi(2)).sum::<f64>() / rates.len() as f64;

        // Variance should be non-zero if noise is being added
        assert!(variance > 0.0);
    }

    #[test]
    fn test_gaussian_noise() {
        let base_scheduler = ConstantScheduler::new(0.1);
        let mut scheduler = NoiseInjectionScheduler::new(
            base_scheduler,
            NoiseDistribution::Gaussian {
                mean: 0.0,
                std_dev: 0.01,
            },
            0.001,
        );

        // Collect samples
        let mut rates = Vec::with_capacity(1000);
        for _ in 0..1000 {
            rates.push(scheduler.step());
        }

        // Statistical checks (basic, just to ensure it's working)
        let mean = rates.iter().sum::<f64>() / rates.len() as f64;

        // Mean should be close to 0.1 (base learning rate)
        assert!((mean - 0.1).abs() < 0.01);
    }

    #[test]
    fn test_cyclical_noise() {
        let base_scheduler = ConstantScheduler::new(0.1);
        let mut scheduler = NoiseInjectionScheduler::new(
            base_scheduler,
            NoiseDistribution::Cyclical {
                amplitude: 0.05,
                period: 10,
            },
            0.001,
        );

        // Step for 20 steps (2 complete cycles)
        let mut rates = Vec::with_capacity(20);
        for _ in 0..20 {
            rates.push(scheduler.step());
        }

        // Check that the pattern repeats
        for i in 0..10 {
            // The rate at i should be similar to the rate at i+10 (one period later)
            // Due to how sinusoidal functions work
            assert!((rates[i] - rates[i + 10]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_decaying_noise() {
        let base_scheduler = ConstantScheduler::new(0.1);
        let mut scheduler = NoiseInjectionScheduler::new(
            base_scheduler,
            NoiseDistribution::Decaying {
                initial_scale: 0.05,
                final_scale: 0.001,
                decay_steps: 100,
            },
            0.001,
        );

        // Check that noise magnitude decreases over time

        // Calculate variance for early steps
        let mut early_rates = Vec::with_capacity(50);
        for _ in 0..50 {
            early_rates.push(scheduler.step());
        }
        let early_mean = early_rates.iter().sum::<f64>() / early_rates.len() as f64;
        let early_variance = early_rates
            .iter()
            .map(|&r| (r - early_mean).powi(2))
            .sum::<f64>()
            / early_rates.len() as f64;

        // Calculate variance for later steps
        let mut late_rates = Vec::with_capacity(50);
        for _ in 0..50 {
            late_rates.push(scheduler.step());
        }
        let late_mean = late_rates.iter().sum::<f64>() / late_rates.len() as f64;
        let late_variance = late_rates
            .iter()
            .map(|&r| (r - late_mean).powi(2))
            .sum::<f64>()
            / late_rates.len() as f64;

        // The variance should decrease over time
        assert!(early_variance > late_variance);
    }

    #[test]
    fn test_min_lr() {
        let base_scheduler = ConstantScheduler::new(0.01);
        let mut scheduler = NoiseInjectionScheduler::new(
            base_scheduler,
            NoiseDistribution::Uniform {
                min: -0.1, // This would make the learning rate negative
                max: 0.0,
            },
            0.005, // Minimum learning rate
        );

        // All learning rates should be at least min_lr
        for _ in 0..100 {
            assert!(scheduler.step() >= 0.005);
        }
    }
}
