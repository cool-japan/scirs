//! Canonical RNG usage patterns for SciRS2 ecosystem
//! This file demonstrates correct type parameterization and initialization

use scirs2_core::essentials::Normal;
use scirs2_core::random::{seeded_rng, thread_rng, CoreRandom};
use scirs2_core::random::{Distribution, Rng};

/// Example 1: Struct with seeded RNG field
struct DeterministicAlgorithm {
    rng: CoreRandom<scirs2_core::random::rngs::StdRng>,
}

impl DeterministicAlgorithm {
    fn new(seed: u64) -> Self {
        Self {
            rng: seeded_rng(seed),
        }
    }

    fn generate_sample(&mut self) -> f64 {
        let normal = Normal::new(0.0, 1.0).unwrap();
        self.rng.sample(normal)
    }
}

/// Example 2: Struct with thread-local RNG field
struct FastAlgorithm {
    rng: CoreRandom, // Defaults to ThreadRng
}

impl FastAlgorithm {
    fn new() -> Self {
        Self { rng: thread_rng() }
    }

    fn random_value(&mut self) -> f64 {
        self.rng.gen_range(0.0..1.0)
    }
}

/// Example 3: Generic algorithm accepting any RNG type
struct GenericAlgorithm<R: Rng> {
    rng: CoreRandom<R>,
}

impl<R: Rng> GenericAlgorithm<R> {
    fn new(rng: CoreRandom<R>) -> Self {
        Self { rng }
    }

    fn process(&mut self) -> Vec<f64> {
        (0..10).map(|_| self.rng.random()).collect()
    }
}

/// Example 4: Functions accepting RNG parameters
fn process_with_rng<R: Rng>(rng: &mut CoreRandom<R>, size: usize) -> Vec<f64> {
    let normal = Normal::new(0.0, 1.0).unwrap();
    (0..size).map(|_| rng.sample(normal)).collect()
}

/// Example 5: Machine Learning model with reproducible initialization
struct NeuralNetwork {
    rng: CoreRandom<scirs2_core::random::rngs::StdRng>,
    weights: Vec<f64>,
}

impl NeuralNetwork {
    fn new(seed: u64, layer_sizes: &[usize]) -> Self {
        let mut rng = seeded_rng(seed);
        let total_weights: usize = layer_sizes.windows(2).map(|w| w[0] * w[1]).sum();

        // Xavier initialization
        let weights = (0..total_weights)
            .map(|_| {
                let scale = (2.0 / layer_sizes[0] as f64).sqrt();
                rng.gen_range(-scale..scale)
            })
            .collect();

        Self { rng, weights }
    }

    fn add_noise(&mut self, scale: f64) {
        let noise_dist = Normal::new(0.0, scale).unwrap();
        for weight in &mut self.weights {
            *weight += self.rng.sample(noise_dist);
        }
    }
}

/// Example 6: Text sampling with weighted selection
struct TextSampler {
    rng: CoreRandom<scirs2_core::random::rngs::StdRng>,
}

impl TextSampler {
    fn new(seed: u64) -> Self {
        Self {
            rng: seeded_rng(seed),
        }
    }

    fn sample_words(&mut self, words: &[String], n: usize) -> Vec<String> {
        use scirs2_core::random::seq::SliceRandom;
        let mut indices: Vec<usize> = (0..words.len()).collect();
        indices.shuffle(self.rng.rng_mut());

        indices
            .into_iter()
            .take(n)
            .map(|i| words[i].clone())
            .collect()
    }

    fn weighted_sample(&mut self, words: &[(String, f64)]) -> String {
        // Simple weighted sampling without WeightedIndex
        let total_weight: f64 = words.iter().map(|(_, w)| w).sum();
        let mut threshold = self.rng.gen_range(0.0..total_weight);

        for (word, weight) in words {
            threshold -= weight;
            if threshold <= 0.0 {
                return word.clone();
            }
        }

        words.last().unwrap().0.clone()
    }
}

/// Example 7: Clustering with configurable RNG
pub struct KMeans<R: Rng> {
    rng: CoreRandom<R>,
    centers: Vec<Vec<f64>>,
}

impl<R: Rng> KMeans<R> {
    pub fn new(rng: CoreRandom<R>, k: usize, dim: usize) -> Self {
        Self {
            rng,
            centers: vec![vec![0.0; dim]; k],
        }
    }

    pub fn random_init(&mut self, data: &[Vec<f64>]) {
        use scirs2_core::random::seq::SliceRandom;
        let mut indices: Vec<usize> = (0..data.len()).collect();
        indices.shuffle(self.rng.rng_mut());

        for (i, &idx) in indices.iter().take(self.centers.len()).enumerate() {
            self.centers[i] = data[idx].clone();
        }
    }
}

fn main() {
    println!("Testing RNG patterns compilation...");

    // Test deterministic algorithm
    let mut det_algo = DeterministicAlgorithm::new(42);
    let sample = det_algo.generate_sample();
    println!("Deterministic sample: {}", sample);

    // Test fast algorithm
    let mut fast_algo = FastAlgorithm::new();
    let value = fast_algo.random_value();
    println!("Fast random value: {}", value);

    // Test generic algorithm with seeded RNG
    let mut generic_seeded = GenericAlgorithm::new(seeded_rng(123));
    let values = generic_seeded.process();
    println!("Generic seeded values: {:?}", &values[..3]);

    // Test generic algorithm with thread RNG
    let mut generic_thread = GenericAlgorithm::new(thread_rng());
    let values = generic_thread.process();
    println!("Generic thread values: {:?}", &values[..3]);

    // Test function with different RNG types
    let mut seeded = seeded_rng(42);
    let seeded_values = process_with_rng(&mut seeded, 5);
    println!("Function with seeded RNG: {:?}", seeded_values);

    let mut thread = thread_rng();
    let thread_values = process_with_rng(&mut thread, 5);
    println!("Function with thread RNG: {:?}", thread_values);

    // Test neural network
    let mut nn = NeuralNetwork::new(42, &[10, 5, 3]);
    println!("Neural network weights: {} total", nn.weights.len());
    nn.add_noise(0.01);

    // Test text sampler
    let mut sampler = TextSampler::new(42);
    let words = vec!["hello".to_string(), "world".to_string(), "rust".to_string()];
    let sampled = sampler.sample_words(&words, 2);
    println!("Sampled words: {:?}", sampled);

    let weighted_words = vec![("common".to_string(), 0.8), ("rare".to_string(), 0.2)];
    let selected = sampler.weighted_sample(&weighted_words);
    println!("Weighted selection: {}", selected);

    // Test clustering with different RNG types
    let kmeans_det = KMeans::new(seeded_rng(42), 3, 2);
    println!(
        "Deterministic K-means created with {} centers",
        kmeans_det.centers.len()
    );

    let kmeans_fast = KMeans::new(thread_rng(), 3, 2);
    println!(
        "Fast K-means created with {} centers",
        kmeans_fast.centers.len()
    );

    println!("\nAll RNG patterns compiled and executed successfully!");
    println!("These patterns are the canonical examples for SciRS2 ecosystem.");
}
