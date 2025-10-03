//! Tests for the random module functionality
//!
//! These tests cover all random number generation features including:
//! - SeedableRng trait implementation
//! - SliceRandomExt functionality
//! - Advanced distributions (Beta, Categorical, etc.)
//! - Optimized array operations
//! - Thread-safe random state

#[cfg(feature = "random")]
mod random_tests {
    use approx::assert_abs_diff_eq;
    use ndarray::{Array, Ix2};
    use rand::seq::SliceRandom;
    use rand::Rng;
    use scirs2_core::random::optimized_arrays::*;
    use scirs2_core::random::*;
    use scirs2_core::random::{Distribution, Normal, Uniform};

    #[test]
    fn test_sobol_sequence() {
        let mut sobol = quasi_monte_carlo::SobolSequence::dimension(2).unwrap();
        let points = sobol.generate(10);

        assert_eq!(points.nrows(), 10);
        assert_eq!(points.ncols(), 2);
        assert!(points.iter().all(|&x| (0.0..1.0).contains(&x)));
    }

    #[test]
    fn test_halton_sequence() {
        let mut halton = quasi_monte_carlo::HaltonSequence::dimension(2).unwrap();
        let points = halton.generate(10);

        assert_eq!(points.nrows(), 10);
        assert_eq!(points.ncols(), 2);
        assert!(points.iter().all(|&x| (0.0..1.0).contains(&x)));
    }

    #[test]
    fn test_latin_hypercube_sampling() {
        let mut lhs = quasi_monte_carlo::LatinHypercubeSampling::new(2);
        let samples = lhs.sample(10).unwrap();

        assert_eq!(samples.shape(), &[10, 2]);
        assert!(samples.iter().all(|&x| (0.0..1.0).contains(&x)));
    }

    #[test]
    fn test_secure_random() {
        let mut secure_rng = secure::SecureRandom::new();
        let value = secure_rng.random_f64();
        assert!((0.0..1.0).contains(&value));

        let bytes = secure_rng.random_bytes(32);
        assert_eq!(bytes.len(), 32);
    }

    #[test]
    fn test_antithetic_sampling() {
        let mut antithetic = variance_reduction::AntitheticSampling::with_default_rng();
        let (original, antithetic_vals) = antithetic.generate_antithetic_pairs(10);

        assert_eq!(original.len(), 10);
        assert_eq!(antithetic_vals.len(), 10);

        for (o, a) in original.iter().zip(antithetic_vals.iter()) {
            assert_abs_diff_eq!(o + a, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_importance_sampling() {
        let mut sampler = importance_sampling::ImportanceSampler::with_default_rng();

        let target_pdf = |x: f64| (-0.5 * x * x).exp();
        let proposal_pdf = |_x: f64| 1.0; // Uniform on some interval
        let proposal_sampler = |rng: &mut Random<_>| rng.sample(Uniform::new(-3.0, 3.0).unwrap());

        let (samples, weights) =
            sampler.sample_with_weights(target_pdf, proposal_pdf, proposal_sampler, 100);

        assert_eq!(samples.len(), 100);
        assert_eq!(weights.len(), 100);
        assert!(weights.iter().all(|&w| w >= 0.0));
    }

    #[test]
    fn test_multivariate_normal() {
        let mean = vec![0.0, 0.0];
        let cov = vec![vec![1.0, 0.5], vec![0.5, 1.0]];

        let mvn = MultivariateNormal::new(mean, cov).unwrap();
        let mut rng = CoreRandom::default();
        let sample = mvn.sample(&mut rng);

        assert_eq!(sample.len(), 2);
    }

    #[test]
    fn test_dirichlet_distribution() {
        let alphas = vec![1.0, 2.0, 3.0];
        let dirichlet = Dirichlet::new(alphas).unwrap();

        let mut rng = CoreRandom::default();
        let sample = dirichlet.sample(&mut rng);

        assert_eq!(sample.len(), 3);
        assert_abs_diff_eq!(sample.iter().sum::<f64>(), 1.0, epsilon = 1e-10);
        assert!(sample.iter().all(|&x| x >= 0.0));
    }

    #[test]
    fn test_von_mises_distribution() {
        let von_mises = VonMises::mu(0.0, 1.0).unwrap();
        let mut rng = CoreRandom::default();

        let samples = von_mises.sample_vec(&mut rng, 100);
        assert_eq!(samples.len(), 100);
        assert!(samples
            .iter()
            .all(|&x| (0.0..2.0 * std::f64::consts::PI).contains(&x)));
    }

    #[test]
    fn test_thread_local_rng_pool() {
        let pool = parallel::ThreadLocalRngPool::default();

        let result = pool.with_rng(|rng| rng.sample(Uniform::new(0.0, 1.0).unwrap()));

        assert!((0.0..1.0).contains(&result));
    }

    #[test]
    fn test_control_variate() {
        let mut control = variance_reduction::ControlVariate::mean(0.5);

        let target = vec![0.1, 0.3, 0.7, 0.9];
        let control_samples = vec![0.2, 0.4, 0.6, 0.8];

        control.estimate_coefficient(&target, &control_samples);
        let corrected = control.apply_correction(&target, &control_samples);

        assert_eq!(corrected.len(), target.len());
    }

    // Tests for OxiRS-requested features

    #[test]
    fn test_seedable_rng_trait() {
        use rand::SeedableRng;

        // Test SeedableRng trait implementation
        let rng1 = Random::<scirs2_core::random::rngs::StdRng>::seed_from_u64(42);
        let rng2 = Random::<scirs2_core::random::rngs::StdRng>::seed_from_u64(42);

        // Same seed should produce same first value
        let mut rng1 = rng1;
        let mut rng2 = rng2;
        let val1 = rng1.sample(Uniform::new(0.0, 1.0).unwrap());
        let val2 = rng2.sample(Uniform::new(0.0, 1.0).unwrap());

        assert_eq!(val1, val2);
    }

    #[test]
    fn test_seeded_rng_function() {
        let mut rng1 = seeded_rng(12345);
        let mut rng2 = seeded_rng(12345);

        let val1 = rng1.sample(Uniform::new(0.0, 1.0).unwrap());
        let val2 = rng2.sample(Uniform::new(0.0, 1.0).unwrap());

        assert_eq!(val1, val2);
    }

    #[test]
    fn test_thread_rng_function() {
        let mut rng = thread_rng();
        let value = rng.sample(Uniform::new(0.0, 1.0).unwrap());
        assert!((0.0..1.0).contains(&value));
    }

    #[test]
    fn test_slice_random_ext() {
        let mut data = [1, 2, 3, 4, 5];
        let mut rng = CoreRandom::default();

        // Test shuffle - Note: Using SciRS2-Core shuffle extension
        data.shuffle(&mut rng);
        assert_eq!(data.len(), 5);
        assert!(data.contains(&1));
        assert!(data.contains(&5));

        // Test choose - Note: Manual implementation for now, will use SciRS2-Core in POLICY refactor
        let original = [1, 2, 3, 4, 5];
        let index = rng.random_range(0..original.len());
        let choice = &original[index];
        assert!(original.contains(choice));

        // Test choose_multiple - Note: Manual implementation for now, will use SciRS2-Core in POLICY refactor
        let mut indices: Vec<usize> = (0..original.len()).collect();
        for i in 0..3.min(original.len()) {
            let j = rng.random_range(i..original.len());
            indices.swap(i, j);
        }
        let choices: Vec<&i32> = indices.iter().take(3).map(|&i| &original[i]).collect();
        assert_eq!(choices.len(), 3);
        for choice in choices {
            assert!(original.contains(choice));
        }
    }

    #[test]
    fn test_slice_random_convenience_functions() {
        let mut data = vec![1, 2, 3, 4, 5];
        slice_random::shuffle(&mut data);
        assert_eq!(data.len(), 5);

        let original = vec![1, 2, 3, 4, 5];
        let choice = slice_random::choose(&original);
        assert!(choice.is_some());

        let samples = slice_random::sample(&original, 3);
        assert_eq!(samples.len(), 3);
    }

    #[test]
    fn test_beta_distribution() {
        let beta = Beta::new(2.0, 3.0).unwrap();
        let mut rng = CoreRandom::default();

        let sample = beta.sample(&mut rng);
        assert!((0.0..1.0).contains(&sample));

        let samples = beta.sample_vec(&mut rng, 100);
        assert_eq!(samples.len(), 100);
        assert!(samples.iter().all(|&x| (0.0..1.0).contains(&x)));

        // Test error cases
        assert!(Beta::new(-1.0, 2.0).is_err());
        assert!(Beta::new(2.0, -1.0).is_err());
    }

    #[test]
    fn test_categorical_distribution() {
        let weights = vec![0.2, 0.3, 0.5];
        let categorical = Categorical::new(weights).unwrap();
        let mut rng = CoreRandom::default();

        let samples = categorical.sample_vec(&mut rng, 1000);
        assert_eq!(samples.len(), 1000);
        assert!(samples.iter().all(|&x| x < 3));

        // Check basic statistics
        let count_0 = samples.iter().filter(|&&x| x == 0).count();
        let count_1 = samples.iter().filter(|&&x| x == 1).count();
        let count_2 = samples.iter().filter(|&&x| x == 2).count();

        // Should have some samples in each category
        assert!(count_0 > 0);
        assert!(count_1 > 0);
        assert!(count_2 > 0);

        // Test error cases
        assert!(Categorical::new(vec![]).is_err());
        assert!(Categorical::new(vec![-1.0, 0.5]).is_err());
    }

    #[test]
    fn test_weighted_choice_distribution() {
        let items = vec!["A", "B", "C"];
        let weights = vec![0.2, 0.3, 0.5];
        let weighted_choice = WeightedChoice::new(items, weights).unwrap();
        let mut rng = CoreRandom::default();

        let samples = weighted_choice.sample_vec(&mut rng, 100);
        assert_eq!(samples.len(), 100);
        assert!(samples.iter().all(|&x| ["A", "B", "C"].contains(&x)));

        // Test error case
        let items_wrong = vec!["A", "B"];
        let weights_wrong = vec![0.2, 0.3, 0.5];
        assert!(WeightedChoice::new(items_wrong, weights_wrong).is_err());
    }

    #[test]
    fn test_exponential_distribution() {
        let exp_dist = ExponentialDist::new(1.0).unwrap();
        let mut rng = CoreRandom::default();

        let sample = exp_dist.sample(&mut rng);
        assert!(sample >= 0.0);

        let samples = exp_dist.sample_vec(&mut rng, 100);
        assert_eq!(samples.len(), 100);
        assert!(samples.iter().all(|&x| x >= 0.0));

        // Note: Using CoreRandom's sample method instead
        let mut samples = Vec::with_capacity(50);
        for _ in 0..50 {
            samples.push(exp_dist.sample(&mut rng));
        }
        let array = samples;
        assert_eq!(array.len(), 50);

        // Test error case
        assert!(ExponentialDist::new(-1.0).is_err());
    }

    #[test]
    fn test_gamma_distribution() {
        let gamma_dist = GammaDist::new(2.0, 1.0).unwrap();
        let mut rng = CoreRandom::default();

        let sample = gamma_dist.sample(&mut rng);
        assert!(sample >= 0.0);

        let samples = gamma_dist.sample_vec(&mut rng, 100);
        assert_eq!(samples.len(), 100);
        assert!(samples.iter().all(|&x| x >= 0.0));

        // Note: Using CoreRandom's sample method instead
        let mut samples = Vec::with_capacity(50);
        for _ in 0..50 {
            samples.push(gamma_dist.sample(&mut rng));
        }
        let array = samples;
        assert_eq!(array.len(), 50);

        // Test error cases
        assert!(GammaDist::new(-1.0, 1.0).is_err());
        assert!(GammaDist::new(1.0, -1.0).is_err());
    }

    #[test]
    fn test_optimized_array_operations() {
        let mut rng = CoreRandom::default();

        // Test bulk operations don't create multiple RNG instances
        let uniform_array = random_uniform_array(Ix2(10, 10), &mut rng);
        assert_eq!(uniform_array.shape(), &[10, 10]);
        assert!(uniform_array.iter().all(|&x| (0.0..1.0).contains(&x)));

        let normal_array = random_normal_array(Ix2(5, 5), 0.0, 1.0, &mut rng);
        assert_eq!(normal_array.shape(), &[5, 5]);

        let exp_array = random_exponential_array(Ix2(3, 3), 1.0, &mut rng);
        assert_eq!(exp_array.shape(), &[3, 3]);
        assert!(exp_array.iter().all(|&x| x >= 0.0));

        let gamma_array = random_gamma_array(Ix2(4, 4), 2.0, 1.0, &mut rng);
        assert_eq!(gamma_array.shape(), &[4, 4]);
        assert!(gamma_array.iter().all(|&x| x >= 0.0));

        let sparse_array = random_sparse_array(Ix2(6, 6), 0.7, &mut rng);
        assert_eq!(sparse_array.shape(), &[6, 6]);
        let zero_count = sparse_array.iter().filter(|&&x| x == 0.0).count();
        assert!(zero_count > 0); // Should have some zeros due to sparsity

        // Test neural network weight initialization
        let xavier_weights = random_xavier_weights(10, 5, &mut rng);
        assert_eq!(xavier_weights.shape(), &[5, 10]);

        let he_weights = random_he_weights(10, 5, &mut rng);
        assert_eq!(he_weights.shape(), &[5, 10]);
    }

    #[test]
    fn test_optimized_array_trait() {
        use OptimizedArrayRandom;

        let mut rng = CoreRandom::default();
        let shape = Ix2(5, 5);

        // Test random_bulk method
        let array = Array::<f64, _>::random_bulk(shape, Uniform::new(0.0, 1.0).unwrap(), &mut rng);
        assert_eq!(array.shape(), &[5, 5]);
        assert!(array.iter().all(|&x| (0.0..1.0).contains(&x)));

        // Test random_using_bulk method
        let array2 = Array::<i32, _>::random_using_bulk(shape, &mut rng, |rng| {
            rng.sample(Uniform::new(1, 100).unwrap())
        });
        assert_eq!(array2.shape(), &[5, 5]);
        assert!(array2.iter().all(|&x| (1..100).contains(&x)));
    }
}
