//! # Fuzzing Framework
//!
//! This module provides fuzzing capabilities for discovering edge cases and potential
//! vulnerabilities in `SciRS2` Core functions. It includes:
//! - Random input generation with configurable constraints
//! - Edge case exploration
//! - Boundary condition testing
//! - Input validation fuzzing

use crate::error::{CoreError, CoreResult};
use crate::testing::{TestConfig, TestResult};
use std::fmt::Debug;
use std::time::{Duration, Instant};

#[cfg(feature = "random")]
use rand::rngs::StdRng;
#[cfg(feature = "random")]
use rand::{Rng, SeedableRng};

/// Fuzzing input generator for different data types
pub trait FuzzingGenerator<T> {
    /// Generate a random input value
    fn generate(&mut self) -> T;

    /// Generate an edge case input
    fn generate_edge_case(&mut self) -> T;

    /// Generate a boundary condition input
    fn generate_boundary(&mut self) -> T;
}

/// Configuration for fuzzing tests
#[derive(Debug, Clone)]
pub struct FuzzingConfig {
    /// Number of random test cases to generate
    pub random_cases: usize,
    /// Number of edge cases to test
    pub edge_cases: usize,
    /// Number of boundary conditions to test
    pub boundary_cases: usize,
    /// Maximum input size for collections
    pub max_input_size: usize,
    /// Minimum input size for collections
    pub min_input_size: usize,
    /// Random seed for reproducible fuzzing
    pub seed: Option<u64>,
    /// Enable shrinking of failing test cases
    pub enable_shrinking: bool,
}

impl Default for FuzzingConfig {
    fn default() -> Self {
        Self {
            random_cases: 1000,
            edge_cases: 100,
            boundary_cases: 100,
            max_input_size: 10000,
            min_input_size: 0,
            seed: None,
            enable_shrinking: true,
        }
    }
}

impl FuzzingConfig {
    /// Create a new fuzzing configuration
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of random test cases
    pub fn with_random_cases(mut self, cases: usize) -> Self {
        self.random_cases = cases;
        self
    }

    /// Set the number of edge cases
    pub fn with_edge_cases(mut self, cases: usize) -> Self {
        self.edge_cases = cases;
        self
    }

    /// Set the number of boundary cases
    pub fn with_boundary_cases(mut self, cases: usize) -> Self {
        self.boundary_cases = cases;
        self
    }

    /// Set the maximum input size
    pub fn with_max_input_size(mut self, size: usize) -> Self {
        self.max_input_size = size;
        self
    }

    /// Set the minimum input size
    pub fn with_min_input_size(mut self, size: usize) -> Self {
        self.min_input_size = size;
        self
    }

    /// Set the random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Enable or disable shrinking
    pub fn with_shrinking(mut self, enable: bool) -> Self {
        self.enable_shrinking = enable;
        self
    }
}

/// Fuzzing result with detailed information about failures
#[derive(Debug, Clone)]
pub struct FuzzingResult {
    /// Total test cases executed
    pub total_cases: usize,
    /// Number of failed cases
    pub failed_cases: usize,
    /// Execution time
    pub duration: Duration,
    /// Failed test cases with inputs and errors
    pub failures: Vec<FuzzingFailure>,
}

/// Information about a fuzzing failure
#[derive(Debug, Clone)]
pub struct FuzzingFailure {
    /// Test case number
    pub case_number: usize,
    /// Input that caused the failure (serialized as string)
    pub input: String,
    /// Error that occurred
    pub error: String,
    /// Type of test case (random, edge, boundary)
    pub case_type: String,
}

/// Fuzzing engine that coordinates test generation and execution
pub struct FuzzingEngine {
    config: FuzzingConfig,
    #[cfg(feature = "random")]
    #[allow(dead_code)]
    rng: StdRng,
}

impl FuzzingEngine {
    /// Create a new fuzzing engine
    pub fn new(config: FuzzingConfig) -> Self {
        #[cfg(feature = "random")]
        let rng = if let Some(seed) = config.seed {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::seed_from_u64(Default::default())
        };

        Self {
            config,
            #[cfg(feature = "random")]
            rng,
        }
    }

    /// Run fuzzing tests on a function with a generator
    pub fn fuzz_function_with_generator<T, F, G>(
        &mut self,
        test_fn: F,
        mut generator: G,
    ) -> CoreResult<FuzzingResult>
    where
        T: Debug + Clone,
        F: Fn(&T) -> CoreResult<()>,
        G: FuzzingGenerator<T>,
    {
        let start_time = Instant::now();
        let mut failures = Vec::new();
        let mut case_number = 0;

        // Test random cases
        for _ in 0..self.config.random_cases {
            let input = generator.generate();
            if let Err(error) = test_fn(&input) {
                failures.push(FuzzingFailure {
                    case_number,
                    input: format!("{input:?}"),
                    error: format!("{error:?}"),
                    case_type: "random".to_string(),
                });
            }
            case_number += 1;
        }

        // Test edge cases
        for _ in 0..self.config.edge_cases {
            let input = generator.generate_edge_case();
            if let Err(error) = test_fn(&input) {
                failures.push(FuzzingFailure {
                    case_number,
                    input: format!("{input:?}"),
                    error: format!("{error:?}"),
                    case_type: "edge".to_string(),
                });
            }
            case_number += 1;
        }

        // Test boundary cases
        for _ in 0..self.config.boundary_cases {
            let input = generator.generate_boundary();
            if let Err(error) = test_fn(&input) {
                failures.push(FuzzingFailure {
                    case_number,
                    input: format!("{input:?}"),
                    error: format!("{error:?}"),
                    case_type: "boundary".to_string(),
                });
            }
            case_number += 1;
        }

        Ok(FuzzingResult {
            total_cases: case_number,
            failed_cases: failures.len(),
            duration: start_time.elapsed(),
            failures,
        })
    }
}

/// Floating-point number fuzzing generator
pub struct FloatFuzzingGenerator {
    #[cfg(feature = "random")]
    rng: StdRng,
    min_value: f64,
    max_value: f64,
}

impl FloatFuzzingGenerator {
    /// Create a new float fuzzing generator
    pub fn new(min_val: f64, max_val: f64) -> Self {
        Self {
            #[cfg(feature = "random")]
            rng: StdRng::seed_from_u64(Default::default()),
            min_value: min_val,
            max_value: max_val,
        }
    }

    /// Create a generator with seed
    #[allow(unused_variables)]
    pub fn with_seed(min_val: f64, max_val: f64, seed: u64) -> Self {
        Self {
            #[cfg(feature = "random")]
            rng: StdRng::seed_from_u64(seed),
            min_value: min_val,
            max_value: max_val,
        }
    }
}

#[allow(deprecated)]
impl FuzzingGenerator<f64> for FloatFuzzingGenerator {
    fn generate(&mut self) -> f64 {
        #[cfg(feature = "random")]
        {
            self.rng.gen_range(self.min_value..=self.max_value)
        }
        #[cfg(not(feature = "random"))]
        {
            // Fallback implementation without random feature
            (self.min_value + self.max_value) / 2.0
        }
    }

    fn generate_edge_case(&mut self) -> f64 {
        #[cfg(feature = "random")]
        {
            let edge_cases = vec![
                0.0,
                -0.0,
                f64::INFINITY,
                f64::NEG_INFINITY,
                f64::NAN,
                f64::MIN,
                f64::MAX,
                f64::MIN_POSITIVE,
                f64::EPSILON,
                -f64::EPSILON,
                1.0,
                -1.0,
            ];

            let valid_edges: Vec<f64> = edge_cases
                .into_iter()
                .filter(|&x| x >= self.min_value && x <= self.max_value && x.is_finite())
                .collect();

            if valid_edges.is_empty() {
                self.generate()
            } else {
                let index = self.rng.gen_range(0..valid_edges.len());
                valid_edges[index]
            }
        }
        #[cfg(not(feature = "random"))]
        {
            // Return a meaningful edge case
            if self.min_value <= 0.0 && self.max_value >= 0.0 {
                0.0
            } else {
                self.min_value
            }
        }
    }

    fn generate_boundary(&mut self) -> f64 {
        #[cfg(feature = "random")]
        {
            match self.rng.gen_range(0..4) {
                0 => self.min_value,
                1 => self.max_value,
                2 => self.min_value + f64::EPSILON,
                _ => self.max_value - f64::EPSILON,
            }
        }
        #[cfg(not(feature = "random"))]
        {
            self.min_value
        }
    }
}

/// Vector fuzzing generator
pub struct VectorFuzzingGenerator {
    #[cfg(feature = "random")]
    rng: StdRng,
    minsize: usize,
    maxsize: usize,
    element_generator: FloatFuzzingGenerator,
}

impl VectorFuzzingGenerator {
    /// Create a new vector fuzzing generator
    pub fn new(min_size: usize, max_size: usize, min_value: f64, max_value: f64) -> Self {
        Self {
            #[cfg(feature = "random")]
            rng: StdRng::seed_from_u64(Default::default()),
            minsize: min_size,
            maxsize: max_size,
            element_generator: FloatFuzzingGenerator::new(min_value, max_value),
        }
    }
}

#[allow(deprecated)]
impl FuzzingGenerator<Vec<f64>> for VectorFuzzingGenerator {
    fn generate(&mut self) -> Vec<f64> {
        #[cfg(feature = "random")]
        let size = self.rng.gen_range(self.minsize..=self.maxsize);
        #[cfg(not(feature = "random"))]
        let size = (self.minsize + self.maxsize) / 2;

        (0..size)
            .map(|_| self.element_generator.generate())
            .collect()
    }

    fn generate_edge_case(&mut self) -> Vec<f64> {
        #[cfg(feature = "random")]
        {
            match self.rng.gen_range(0..4) {
                0 => vec![],                                            // Empty vector
                1 => vec![self.element_generator.generate_edge_case()], // Single element
                2 => {
                    // All same values
                    let value = self.element_generator.generate_edge_case();
                    let size = self.rng.gen_range(2..=10);
                    vec![value; size]
                }
                _ => {
                    // Mixed edge cases
                    let size = self.rng.gen_range(2..=10);
                    (0..size)
                        .map(|_| self.element_generator.generate_edge_case())
                        .collect()
                }
            }
        }
        #[cfg(not(feature = "random"))]
        {
            vec![] // Empty vector as edge case
        }
    }

    fn generate_boundary(&mut self) -> Vec<f64> {
        #[cfg(feature = "random")]
        {
            match self.rng.gen_range(0..3) {
                0 => {
                    // Minimum size
                    let size = self.minsize;
                    (0..size)
                        .map(|_| self.element_generator.generate_boundary())
                        .collect()
                }
                1 => {
                    // Maximum size
                    let size = self.maxsize;
                    (0..size)
                        .map(|_| self.element_generator.generate_boundary())
                        .collect()
                }
                _ => {
                    // Size near boundaries
                    let size = if self.minsize > 0 {
                        self.minsize - 1
                    } else {
                        self.maxsize + 1
                    };
                    (0..size)
                        .map(|_| self.element_generator.generate_boundary())
                        .collect()
                }
            }
        }
        #[cfg(not(feature = "random"))]
        {
            vec![self.element_generator.generate_boundary(); self.minsize]
        }
    }
}

/// High-level fuzzing utilities
pub struct FuzzingUtils;

impl FuzzingUtils {
    /// Fuzz a numerical function with floating-point inputs
    pub fn fuzz_numeric_function<F>(
        function: F,
        config: FuzzingConfig,
        min_value: f64,
        max_value: f64,
    ) -> CoreResult<FuzzingResult>
    where
        F: Fn(f64) -> CoreResult<f64>,
    {
        let mut engine = FuzzingEngine::new(config);
        let generator = FloatFuzzingGenerator::new(min_value, max_value);

        engine.fuzz_function_with_generator(|input: &f64| function(*input).map(|_| ()), generator)
    }

    /// Fuzz a vector function
    pub fn fuzz_vector_function<F>(
        function: F,
        config: FuzzingConfig,
        minsize: usize,
        maxsize: usize,
        min_value: f64,
        max_value: f64,
    ) -> CoreResult<FuzzingResult>
    where
        F: Fn(&[f64]) -> CoreResult<Vec<f64>>,
    {
        let mut engine = FuzzingEngine::new(config);
        let generator = VectorFuzzingGenerator::new(minsize, maxsize, min_value, max_value);

        engine
            .fuzz_function_with_generator(|input: &Vec<f64>| function(input).map(|_| ()), generator)
    }

    /// Create a comprehensive fuzzing test suite
    pub fn create_fuzzing_suite(name: &str, config: TestConfig) -> crate::testing::TestSuite {
        let mut suite = crate::testing::TestSuite::new(name, config);

        // Add standard fuzzing tests
        suite.add_test("numeric_edge_cases", |_runner| {
            let fuzzing_config = FuzzingConfig::default().with_edge_cases(100);
            let result = Self::fuzz_numeric_function(
                |x| {
                    if x.is_finite() && x != 0.0 {
                        Ok(1.0 / x)
                    } else {
                        Err(CoreError::DomainError(crate::error::ErrorContext::new(
                            "Division by zero or infinite input",
                        )))
                    }
                },
                fuzzing_config,
                -1000.0,
                1000.0,
            )?;

            if result.failed_cases > 0 {
                return Ok(TestResult::failure(
                    std::time::Duration::from_secs(1),
                    result.total_cases,
                    format!("Fuzzing found {} failures", result.failed_cases),
                ));
            }

            Ok(TestResult::success(
                std::time::Duration::from_secs(1),
                result.total_cases,
            ))
        });

        suite.add_test("vector_boundary_conditions", |_runner| {
            let fuzzing_config = FuzzingConfig::default().with_boundary_cases(50);
            let result = Self::fuzz_vector_function(
                |v| {
                    if v.is_empty() {
                        Err(CoreError::ValidationError(crate::error::ErrorContext::new(
                            "Empty vector not allowed",
                        )))
                    } else {
                        Ok(v.iter().map(|&x| x * 2.0).collect())
                    }
                },
                fuzzing_config,
                0,
                1000,
                -100.0,
                100.0,
            )?;

            if result.failed_cases > 0 {
                return Ok(TestResult::failure(
                    std::time::Duration::from_secs(1),
                    result.total_cases,
                    format!("Vector fuzzing found {} failures", result.failed_cases),
                ));
            }

            Ok(TestResult::success(
                std::time::Duration::from_secs(1),
                result.total_cases,
            ))
        });

        suite
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_float_fuzzing_generator() {
        let mut generator = FloatFuzzingGenerator::new(-10.0, 10.0);

        // Test normal generation
        for _ in 0..100 {
            let value = generator.generate();
            assert!((-10.0..=10.0).contains(&value));
        }

        // Test edge case generation
        let edge_case = generator.generate_edge_case();
        assert!(edge_case.is_finite());

        // Test boundary generation
        let boundary = generator.generate_boundary();
        assert!(boundary.is_finite());
    }

    #[test]
    fn test_vector_fuzzing_generator() {
        let mut generator = VectorFuzzingGenerator::new(1, 10, -5.0, 5.0);

        // Test normal generation
        let vector = generator.generate();
        assert!(!vector.is_empty() && vector.len() <= 10);
        for &value in &vector {
            assert!((-5.0..=5.0).contains(&value));
        }

        // Test edge case generation
        let edge_vector = generator.generate_edge_case();
        // Edge cases might generate empty vectors or vectors with special values

        // Test boundary generation
        let boundary_vector = generator.generate_boundary();
        // Boundary cases test size limits
    }

    #[test]
    fn test_fuzzing_config() {
        let config = FuzzingConfig::new()
            .with_random_cases(500)
            .with_edge_cases(50)
            .with_boundary_cases(25)
            .with_max_input_size(5000)
            .with_seed(12345);

        assert_eq!(config.random_cases, 500);
        assert_eq!(config.edge_cases, 50);
        assert_eq!(config.boundary_cases, 25);
        assert_eq!(config.max_input_size, 5000);
        assert_eq!(config.seed, Some(12345));
    }
}
