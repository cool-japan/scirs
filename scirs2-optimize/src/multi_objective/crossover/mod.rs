//! Crossover operators for multi-objective optimization
//!
//! Various crossover strategies for generating offspring solutions.

use rand::Rng;

/// Trait for crossover operators
pub trait CrossoverOperator {
    /// Perform crossover between two parents
    fn crossover(&self, parent1: &[f64], parent2: &[f64]) -> (Vec<f64>, Vec<f64>);
}

/// Simulated Binary Crossover (SBX)
#[derive(Debug, Clone)]
pub struct SimulatedBinaryCrossover {
    crossover_probability: f64,
    distribution_index: f64,
}

/// Simulated Binary Crossover (SBX) - Alias for compatibility
pub type SBX = SimulatedBinaryCrossover;

impl SimulatedBinaryCrossover {
    pub fn new(distribution_index: f64, crossover_probability: f64) -> Self {
        Self {
            crossover_probability,
            distribution_index,
        }
    }
}

impl CrossoverOperator for SimulatedBinaryCrossover {
    fn crossover(&self, parent1: &[f64], parent2: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let mut rng = rand::rng();
        let n = parent1.len();
        let mut child1 = vec![0.0; n];
        let mut child2 = vec![0.0; n];

        if rng.random::<f64>() <= self.crossover_probability {
            for i in 0..n {
                if rng.random::<f64>() <= 0.5 {
                    let beta = self.calculate_beta(&mut rng);
                    child1[i] = 0.5 * ((1.0 + beta) * parent1[i] + (1.0 - beta) * parent2[i]);
                    child2[i] = 0.5 * ((1.0 - beta) * parent1[i] + (1.0 + beta) * parent2[i]);
                } else {
                    child1[i] = parent1[i];
                    child2[i] = parent2[i];
                }
            }
        } else {
            child1 = parent1.to_vec();
            child2 = parent2.to_vec();
        }

        (child1, child2)
    }
}

impl SimulatedBinaryCrossover {
    fn calculate_beta(&self, rng: &mut impl Rng) -> f64 {
        let u = rng.random::<f64>();
        if u <= 0.5 {
            (2.0 * u).powf(1.0 / (self.distribution_index + 1.0))
        } else {
            (1.0 / (2.0 * (1.0 - u))).powf(1.0 / (self.distribution_index + 1.0))
        }
    }
}

/// Uniform crossover
#[derive(Debug, Clone)]
pub struct UniformCrossover {
    crossover_probability: f64,
}

impl UniformCrossover {
    pub fn new(crossover_probability: f64) -> Self {
        Self {
            crossover_probability,
        }
    }
}

impl CrossoverOperator for UniformCrossover {
    fn crossover(&self, parent1: &[f64], parent2: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let mut rng = rand::rng();
        let n = parent1.len();
        let mut child1 = vec![0.0; n];
        let mut child2 = vec![0.0; n];

        if rng.random::<f64>() <= self.crossover_probability {
            for i in 0..n {
                if rng.gen_bool(0.5) {
                    child1[i] = parent1[i];
                    child2[i] = parent2[i];
                } else {
                    child1[i] = parent2[i];
                    child2[i] = parent1[i];
                }
            }
        } else {
            child1 = parent1.to_vec();
            child2 = parent2.to_vec();
        }

        (child1, child2)
    }
}
