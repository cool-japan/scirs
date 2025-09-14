//! Mutation operators for multi-objective optimization
//!
//! Various mutation strategies for maintaining diversity in populations.

use rand::Rng;
use rand_distr::{Distribution, Normal};

/// Trait for mutation operators
pub trait MutationOperator {
    /// Perform mutation on a solution
    fn mutate(&self, solution: &mut [f64], bounds: &[(f64, f64)]);
}

/// Polynomial mutation
#[derive(Debug, Clone)]
pub struct PolynomialMutation {
    mutation_probability: f64,
    distribution_index: f64,
}

impl PolynomialMutation {
    pub fn new(mutation_probability: f64, distribution_index: f64) -> Self {
        Self {
            mutation_probability,
            distribution_index,
        }
    }

    fn calculate_delta(&self, rng: &mut impl Rng) -> f64 {
        let u = rng.random::<f64>();
        if u < 0.5 {
            (2.0 * u).powf(1.0 / (self.distribution_index + 1.0)) - 1.0
        } else {
            1.0 - (2.0 * (1.0 - u)).powf(1.0 / (self.distribution_index + 1.0))
        }
    }
}

impl MutationOperator for PolynomialMutation {
    fn mutate(&self, solution: &mut [f64], bounds: &[(f64, f64)]) {
        let mut rng = rand::rng();
        let n = solution.len();

        for i in 0..n {
            if rng.random::<f64>() <= self.mutation_probability {
                let (lower, upper) = bounds[i];
                let delta = self.calculate_delta(&mut rng);
                let max_delta = upper - lower;

                solution[i] += delta * max_delta;
                solution[i] = solution[i].max(lower).min(upper);
            }
        }
    }
}

/// Gaussian mutation
#[derive(Debug, Clone)]
pub struct GaussianMutation {
    mutation_probability: f64,
    std_dev: f64,
}

impl GaussianMutation {
    pub fn new(mutation_probability: f64, std_dev: f64) -> Self {
        Self {
            mutation_probability,
            std_dev,
        }
    }
}

impl MutationOperator for GaussianMutation {
    fn mutate(&self, solution: &mut [f64], bounds: &[(f64, f64)]) {
        let mut rng = rand::rng();
        let normal = Normal::new(0.0, self.std_dev).unwrap();

        for i in 0..solution.len() {
            if rng.random::<f64>() <= self.mutation_probability {
                let (lower, upper) = bounds[i];
                let perturbation = normal.sample(&mut rng);

                solution[i] += perturbation;
                solution[i] = solution[i].max(lower).min(upper);
            }
        }
    }
}

/// Uniform mutation
#[derive(Debug, Clone)]
pub struct UniformMutation {
    mutation_probability: f64,
}

impl UniformMutation {
    pub fn new(mutation_probability: f64) -> Self {
        Self {
            mutation_probability,
        }
    }
}

impl MutationOperator for UniformMutation {
    fn mutate(&self, solution: &mut [f64], bounds: &[(f64, f64)]) {
        let mut rng = rand::rng();

        for i in 0..solution.len() {
            if rng.random::<f64>() <= self.mutation_probability {
                let (lower, upper) = bounds[i];
                solution[i] = rng.gen_range(lower..upper);
            }
        }
    }
}
