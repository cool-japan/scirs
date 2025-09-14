//! Weight initialization strategies for neural networks

use crate::error::{NeuralError, Result};
use ndarray::{Array, Dimension, IxDyn};
use num_traits::Float;
use rand::Rng;
use std::fmt::Debug;
/// Initialization strategies for neural network weights
#[derive(Debug, Clone, Copy)]
pub enum Initializer {
    /// Zero initialization
    Zeros,
    /// One initialization
    Ones,
    /// Uniform random initialization
    Uniform {
        /// Minimum value
        min: f64,
        /// Maximum value
        max: f64,
    },
    /// Normal random initialization
    Normal {
        /// Mean
        mean: f64,
        /// Standard deviation
        std: f64,
    },
    /// Xavier/Glorot initialization
    Xavier,
    /// He initialization
    He,
    /// LeCun initialization
    LeCun,
}
impl Initializer {
    /// Initialize weights according to the strategy
    ///
    /// # Arguments
    /// * `shape` - Shape of the weights array
    /// * `fan_in` - Number of input connections (for Xavier, He, LeCun)
    /// * `fan_out` - Number of output connections (for Xavier)
    /// * `rng` - Random number generator
    /// # Returns
    /// * Initialized weights array
    pub fn initialize<F: Float + Debug, R: Rng>(
        &self,
        shape: IxDyn,
        fan_in: usize,
        fan_out: usize,
        rng: &mut R,
    ) -> Result<Array<F, IxDyn>> {
        let size = shape.as_array_view().iter().product();
        match self {
            Initializer::Zeros => Ok(Array::zeros(shape)),
            Initializer::Ones => {
                let ones: Vec<F> = (0..size).map(|_| F::one()).collect();
                Array::from_shape_vec(shape, ones).map_err(|e| {
                    NeuralError::InvalidArchitecture(format!("Failed to create array: {e}"))
                })
            }
            Initializer::Uniform { min, max } => {
                let values: Vec<F> = (0..size)
                    .map(|_| {
                        let val = rng.gen_range(*min..*max);
                        F::from(val).ok_or_else(|| {
                            NeuralError::InvalidArchitecture(
                                "Failed to convert random value".to_string(),
                            )
                        })
                    })
                    .collect::<Result<Vec<F>>>()?;
                Array::from_shape_vec(shape, values).map_err(|e| {
                    NeuralError::InvalidArchitecture(format!("Failed to create array: {e}"))
                })
            }
            Initializer::Normal { mean, std } => {
                let values: Vec<F> = (0..(size / 2 + 1))
                    .flat_map(|_| {
                        // Box-Muller transform to generate normal distribution
                        let u1 = rng.gen_range(0.0..1.0);
                        let u2 = rng.gen_range(0.0..1.0);
                        let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                        let z1 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).sin();
                        let val0 = mean + std * z0;
                        let val1 = mean + std * z1;
                        vec![
                            F::from(val0).unwrap_or(F::zero()),
                            F::from(val1).unwrap_or(F::zero()),
                        ]
                    })
                    .take(size)
                    .collect();
                Array::from_shape_vec(shape, values).map_err(|e| {
                    NeuralError::InvalidArchitecture(format!("Failed to create array: {e}"))
                })
            }
            Initializer::Xavier => {
                let limit = (6.0 / (fan_in + fan_out) as f64).sqrt();
                let values: Vec<F> = (0..size)
                    .map(|_| {
                        let val = rng.gen_range(-limit..limit);
                        F::from(val).unwrap_or(F::zero())
                    })
                    .collect();
                Array::from_shape_vec(shape, values).map_err(|e| {
                    NeuralError::InvalidArchitecture(format!("Failed to create array: {e}"))
                })
            }
            Initializer::He => {
                let std = (2.0 / fan_in as f64).sqrt();
                let values: Vec<F> = (0..(size / 2 + 1))
                    .flat_map(|_| {
                        // Box-Muller transform for He initialization
                        let u1 = rng.gen_range(0.0..1.0);
                        let u2 = rng.gen_range(0.0..1.0);
                        let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                        let z1 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).sin();
                        let val0 = std * z0;
                        let val1 = std * z1;
                        vec![
                            F::from(val0).unwrap_or(F::zero()),
                            F::from(val1).unwrap_or(F::zero()),
                        ]
                    })
                    .take(size)
                    .collect();
                Array::from_shape_vec(shape, values).map_err(|e| {
                    NeuralError::InvalidArchitecture(format!("Failed to create array: {e}"))
                })
            }
            Initializer::LeCun => {
                let std = (1.0 / fan_in as f64).sqrt();
                let values: Vec<F> = (0..(size / 2 + 1))
                    .flat_map(|_| {
                        // Box-Muller transform for LeCun initialization
                        let u1 = rng.gen_range(0.0..1.0);
                        let u2 = rng.gen_range(0.0..1.0);
                        let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                        let z1 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).sin();
                        let val0 = std * z0;
                        let val1 = std * z1;
                        vec![
                            F::from(val0).unwrap_or(F::zero()),
                            F::from(val1).unwrap_or(F::zero()),
                        ]
                    })
                    .take(size)
                    .collect();
                Array::from_shape_vec(shape, values).map_err(|e| {
                    NeuralError::InvalidArchitecture(format!("Failed to create array: {e}"))
                })
            }
        }
    }
}
/// Xavier/Glorot uniform initialization
///
/// # Arguments
/// * `shape` - Shape of the weights array
/// # Returns
/// * Initialized weights array
#[allow(dead_code)]
pub fn xavier_uniform<F: Float + Debug>(shape: IxDyn) -> Result<Array<F, IxDyn>> {
    let fan_in = match shape.ndim() {
        0 => 1,
        1 => shape[0],
        _ => shape[0],
    };
    let fan_out = match shape.ndim() {
        1 => 1,
        _ => shape[1],
    };
    let mut rng = rand::rng();
    Initializer::Xavier.initialize(shape, fan_in, fan_out, &mut rng)
}
