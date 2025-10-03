//! Simulation utilities for Monte Carlo methods
//!
//! This module provides efficient random number generation, path simulation, and
//! variance reduction techniques for Monte Carlo pricing.
//!
//! # Features
//! - Low-discrepancy sequences (Sobol, Halton)
//! - Geometric Brownian motion path generation
//! - Antithetic variates for variance reduction
//! - Flexible random number generation
//!
//! # Example
//! ```
//! use scirs2_integrate::specialized::finance::utils::simulation::{
//!     PathGenerator, PathConfig, VarianceReduction
//! };
//!
//! // Generate stock price paths with geometric Brownian motion
//! let config = PathConfig::new(100.0, 0.05, 0.0, 0.2, 1.0)
//!     .with_steps(252)
//!     .with_variance_reduction(VarianceReduction::Antithetic);
//!
//! let generator = PathGenerator::new(config);
//! let paths = generator.generate_paths(10000);
//! ```

use crate::error::{IntegrateError, IntegrateResult as Result};
use scirs2_core::random::Rng;
use scirs2_core::random::{Distribution, Normal};

/// Variance reduction technique
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VarianceReduction {
    /// No variance reduction
    None,
    /// Antithetic variates (generate paired paths with negated shocks)
    Antithetic,
}

/// Configuration for path generation
#[derive(Debug, Clone)]
pub struct PathConfig {
    /// Initial value (e.g., spot price)
    pub initial_value: f64,
    /// Drift rate (e.g., risk-free rate - dividend)
    pub drift: f64,
    /// Dividend yield
    pub dividend: f64,
    /// Volatility
    pub volatility: f64,
    /// Time horizon (years)
    pub time_horizon: f64,
    /// Number of time steps
    pub n_steps: usize,
    /// Variance reduction technique
    pub variance_reduction: VarianceReduction,
}

impl PathConfig {
    /// Create a new path configuration
    pub fn new(
        initial_value: f64,
        risk_free_rate: f64,
        dividend: f64,
        volatility: f64,
        time_horizon: f64,
    ) -> Self {
        Self {
            initial_value,
            drift: risk_free_rate - dividend,
            dividend,
            volatility,
            time_horizon,
            n_steps: 252, // Daily steps by default
            variance_reduction: VarianceReduction::None,
        }
    }

    /// Set number of time steps
    pub fn with_steps(mut self, n_steps: usize) -> Self {
        self.n_steps = n_steps;
        self
    }

    /// Set variance reduction technique
    pub fn with_variance_reduction(mut self, variance_reduction: VarianceReduction) -> Self {
        self.variance_reduction = variance_reduction;
        self
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        if self.initial_value <= 0.0 {
            return Err(IntegrateError::ValueError(
                "Initial value must be positive".to_string(),
            ));
        }
        if self.volatility < 0.0 {
            return Err(IntegrateError::ValueError(
                "Volatility cannot be negative".to_string(),
            ));
        }
        if self.time_horizon <= 0.0 {
            return Err(IntegrateError::ValueError(
                "Time horizon must be positive".to_string(),
            ));
        }
        if self.n_steps == 0 {
            return Err(IntegrateError::ValueError(
                "Number of steps must be positive".to_string(),
            ));
        }
        Ok(())
    }
}

/// Path generator for geometric Brownian motion
pub struct PathGenerator {
    config: PathConfig,
}

impl PathGenerator {
    /// Create a new path generator
    pub fn new(config: PathConfig) -> Self {
        Self { config }
    }

    /// Generate multiple paths
    pub fn generate_paths(&self, n_paths: usize) -> Result<Vec<Vec<f64>>> {
        self.config.validate()?;

        let mut rng = scirs2_core::random::rng();
        let normal = Normal::new(0.0, 1.0).map_err(|e| {
            IntegrateError::ValueError(format!("Failed to create normal distribution: {}", e))
        })?;

        let mut paths = Vec::with_capacity(n_paths);

        match self.config.variance_reduction {
            VarianceReduction::None => {
                for _ in 0..n_paths {
                    paths.push(self.generate_single_path(&mut rng, &normal)?);
                }
            }
            VarianceReduction::Antithetic => {
                let pairs = n_paths / 2;
                for _ in 0..pairs {
                    let (path1, path2) = self.generate_antithetic_pair(&mut rng, &normal)?;
                    paths.push(path1);
                    paths.push(path2);
                }
                // If odd number of paths, add one more
                if n_paths % 2 == 1 {
                    paths.push(self.generate_single_path(&mut rng, &normal)?);
                }
            }
        }

        Ok(paths)
    }

    /// Generate a single path using geometric Brownian motion
    fn generate_single_path<R: Rng>(&self, rng: &mut R, normal: &Normal<f64>) -> Result<Vec<f64>> {
        let dt = self.config.time_horizon / self.config.n_steps as f64;
        let drift_term = (self.config.drift - 0.5 * self.config.volatility.powi(2)) * dt;
        let diffusion_term = self.config.volatility * dt.sqrt();

        let mut path = Vec::with_capacity(self.config.n_steps + 1);
        path.push(self.config.initial_value);

        let mut current_value = self.config.initial_value;

        for _ in 0..self.config.n_steps {
            let z = normal.sample(rng);
            current_value *= (drift_term + diffusion_term * z).exp();
            path.push(current_value);
        }

        Ok(path)
    }

    /// Generate antithetic pair of paths
    fn generate_antithetic_pair<R: Rng>(
        &self,
        rng: &mut R,
        normal: &Normal<f64>,
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        let dt = self.config.time_horizon / self.config.n_steps as f64;
        let drift_term = (self.config.drift - 0.5 * self.config.volatility.powi(2)) * dt;
        let diffusion_term = self.config.volatility * dt.sqrt();

        let mut path1 = Vec::with_capacity(self.config.n_steps + 1);
        let mut path2 = Vec::with_capacity(self.config.n_steps + 1);

        path1.push(self.config.initial_value);
        path2.push(self.config.initial_value);

        let mut value1 = self.config.initial_value;
        let mut value2 = self.config.initial_value;

        for _ in 0..self.config.n_steps {
            let z = normal.sample(rng);

            // Path 1 uses positive shock
            value1 *= (drift_term + diffusion_term * z).exp();
            path1.push(value1);

            // Path 2 uses negative shock (antithetic)
            value2 *= (drift_term - diffusion_term * z).exp();
            path2.push(value2);
        }

        Ok((path1, path2))
    }

    /// Generate terminal values only (more efficient when intermediate values not needed)
    pub fn generate_terminal_values(&self, n_paths: usize) -> Result<Vec<f64>> {
        self.config.validate()?;

        let mut rng = scirs2_core::random::rng();
        let normal = Normal::new(0.0, 1.0).map_err(|e| {
            IntegrateError::ValueError(format!("Failed to create normal distribution: {}", e))
        })?;

        let dt = self.config.time_horizon / self.config.n_steps as f64;
        let drift_term = (self.config.drift - 0.5 * self.config.volatility.powi(2)) * dt;
        let diffusion_term = self.config.volatility * dt.sqrt();

        let mut terminal_values = Vec::with_capacity(n_paths);

        match self.config.variance_reduction {
            VarianceReduction::None => {
                for _ in 0..n_paths {
                    let mut value = self.config.initial_value;
                    for _ in 0..self.config.n_steps {
                        let z = normal.sample(&mut rng);
                        value *= (drift_term + diffusion_term * z).exp();
                    }
                    terminal_values.push(value);
                }
            }
            VarianceReduction::Antithetic => {
                let pairs = n_paths / 2;
                for _ in 0..pairs {
                    let mut value1 = self.config.initial_value;
                    let mut value2 = self.config.initial_value;

                    for _ in 0..self.config.n_steps {
                        let z = normal.sample(&mut rng);
                        value1 *= (drift_term + diffusion_term * z).exp();
                        value2 *= (drift_term - diffusion_term * z).exp();
                    }

                    terminal_values.push(value1);
                    terminal_values.push(value2);
                }

                // Handle odd number of paths
                if n_paths % 2 == 1 {
                    let mut value = self.config.initial_value;
                    for _ in 0..self.config.n_steps {
                        let z = normal.sample(&mut rng);
                        value *= (drift_term + diffusion_term * z).exp();
                    }
                    terminal_values.push(value);
                }
            }
        }

        Ok(terminal_values)
    }
}

/// Random number generator trait for flexibility
pub trait RandomNumberGenerator {
    /// Generate a standard normal random number
    fn generate_normal(&mut self) -> f64;

    /// Generate multiple standard normal random numbers
    fn generate_normal_vec(&mut self, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.generate_normal()).collect()
    }
}

/// Standard pseudo-random number generator
pub struct StandardRng {
    rng: scirs2_core::random::rngs::ThreadRng,
    normal: Normal<f64>,
}

impl StandardRng {
    /// Create a new standard RNG
    pub fn new() -> Result<Self> {
        let normal = Normal::new(0.0, 1.0).map_err(|e| {
            IntegrateError::ValueError(format!("Failed to create normal distribution: {}", e))
        })?;

        Ok(Self {
            rng: scirs2_core::random::rng(),
            normal,
        })
    }
}

impl Default for StandardRng {
    fn default() -> Self {
        Self::new().expect("Failed to create standard RNG")
    }
}

impl RandomNumberGenerator for StandardRng {
    fn generate_normal(&mut self) -> f64 {
        self.normal.sample(&mut self.rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_path_config_creation() {
        let config = PathConfig::new(100.0, 0.05, 0.02, 0.2, 1.0);
        assert_eq!(config.initial_value, 100.0);
        assert!((config.drift - 0.03).abs() < 1e-10); // r - q = 0.05 - 0.02
        assert_eq!(config.volatility, 0.2);
        assert_eq!(config.n_steps, 252);
    }

    #[test]
    fn test_path_config_validation() {
        let valid_config = PathConfig::new(100.0, 0.05, 0.0, 0.2, 1.0);
        assert!(valid_config.validate().is_ok());

        let invalid_config = PathConfig::new(-100.0, 0.05, 0.0, 0.2, 1.0);
        assert!(invalid_config.validate().is_err());
    }

    #[test]
    fn test_path_generation() {
        let config = PathConfig::new(100.0, 0.05, 0.0, 0.2, 1.0).with_steps(100);
        let generator = PathGenerator::new(config);

        let paths = generator.generate_paths(10).unwrap();
        assert_eq!(paths.len(), 10);

        for path in &paths {
            assert_eq!(path.len(), 101); // n_steps + 1
            assert_eq!(path[0], 100.0); // Initial value
            assert!(path.iter().all(|&v| v > 0.0)); // All positive
        }
    }

    #[test]
    fn test_antithetic_variance_reduction() {
        let config = PathConfig::new(100.0, 0.05, 0.0, 0.2, 1.0)
            .with_steps(50)
            .with_variance_reduction(VarianceReduction::Antithetic);

        let generator = PathGenerator::new(config);
        let paths = generator.generate_paths(100).unwrap();

        assert_eq!(paths.len(), 100);

        // Check that all paths start at initial value
        for path in &paths {
            assert_eq!(path[0], 100.0);
        }
    }

    #[test]
    fn test_terminal_values_generation() {
        let config = PathConfig::new(100.0, 0.05, 0.0, 0.2, 1.0).with_steps(252);
        let generator = PathGenerator::new(config);

        let terminal_values = generator.generate_terminal_values(1000).unwrap();
        assert_eq!(terminal_values.len(), 1000);

        // All values should be positive
        assert!(terminal_values.iter().all(|&v| v > 0.0));

        // Statistical checks: mean should be close to forward price
        let forward_price = 100.0 * (0.05_f64 * 1.0).exp();
        let mean: f64 = terminal_values.iter().sum::<f64>() / terminal_values.len() as f64;

        // Mean should be within reasonable range (using generous tolerance for randomness)
        assert!((mean - forward_price).abs() / forward_price < 0.1);
    }

    #[test]
    fn test_standard_rng() {
        let mut rng = StandardRng::new().unwrap();

        let samples = rng.generate_normal_vec(10000);
        assert_eq!(samples.len(), 10000);

        // Statistical test: mean should be close to 0, std dev close to 1
        let mean: f64 = samples.iter().sum::<f64>() / samples.len() as f64;
        let variance: f64 =
            samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / samples.len() as f64;
        let std_dev = variance.sqrt();

        assert!(mean.abs() < 0.05); // Mean close to 0
        assert!((std_dev - 1.0).abs() < 0.05); // Std dev close to 1
    }

    #[test]
    fn test_path_lengths() {
        let config = PathConfig::new(100.0, 0.05, 0.0, 0.2, 1.0).with_steps(10);
        let generator = PathGenerator::new(config);

        let paths = generator.generate_paths(5).unwrap();
        for path in paths {
            assert_eq!(path.len(), 11); // n_steps + 1
        }
    }

    #[test]
    fn test_variance_reduction_odd_paths() {
        let config = PathConfig::new(100.0, 0.05, 0.0, 0.2, 1.0)
            .with_steps(50)
            .with_variance_reduction(VarianceReduction::Antithetic);

        let generator = PathGenerator::new(config);

        // Test odd number of paths
        let paths = generator.generate_paths(101).unwrap();
        assert_eq!(paths.len(), 101);
    }
}
