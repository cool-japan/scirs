use ndarray::IxDyn;
use rand::seq::SliceRandom;
use rand_distr::{Bernoulli, Normal, Uniform};
use scirs2_core::random::{get_rng, sampling, DistributionExt, Random};

#[allow(dead_code)]
fn main() {
    println!("Random Number Generation Example");

    // Only run the example if the random feature is enabled
    #[cfg(feature = "random")]
    {
        println!("\n--- Basic Random Number Generation ---");
        basic_random_example();

        println!("\n--- Distribution Examples ---");
        distribution_examples();

        println!("\n--- Random Array Generation ---");
        random_array_example();

        println!("\n--- Seeded Random Generation ---");
        seeded_random_example();

        println!("\n--- Thread-Local Random Example ---");
        thread_local_random_example();

        println!("\n--- Sampling Functions Example ---");
        sampling_functions_example();
    }

    #[cfg(not(feature = "random"))]
    println!("Random feature not enabled. Run with --features=\"random\" to see the example.");
}

#[cfg(feature = "random")]
#[allow(dead_code)]
fn basic_random_example() {
    // Create a random number generator
    let mut rng = Random::default();

    // Generate random values
    let value1 = rng.gen_range(1..100);
    let value2 = rng.gen_range(0.0..1.0);
    let coin_flip = rng.random_bool();

    println!("Random integer (1-99): {}", value1);
    println!("Random float (0.saturating_sub(1)): {:.6}", value2);
    println!("Random boolean: {}", coin_flip);

    // Generate a random boolean with a specific probability
    let biased_coin = rng.random_bool_with_chance(0.8);
    println!("Biased coin (80% true): {}", biased_coin);

    // Shuffle a vector
    let mut numbers: Vec<i32> = (1..10).collect();
    println!("Original vector: {:?}", numbers);
    rng.shuffle(&mut numbers);
    println!("Shuffled vector: {:?}", numbers);
}

#[cfg(feature = "random")]
#[allow(dead_code)]
fn distribution_examples() {
    let mut rng = Random::default();

    // Sample from uniform distribution
    let uniform = Uniform::new(0.0, 10.0).unwrap();
    let uniform_sample = rng.sample(uniform);
    println!("Uniform(0, 10) sample: {:.4}", uniform_sample);

    // Sample from normal distribution
    let normal = Normal::new(5.0, 2.0).unwrap();
    let normal_sample = rng.sample(normal);
    println!("Normal(mean=5, std=2) sample: {:.4}", normal_sample);

    // Sample from Bernoulli distribution
    let bernoulli = Bernoulli::new(0.7).unwrap();
    let bernoulli_sample = rng.sample(bernoulli);
    println!("Bernoulli(p=0.7) sample: {}", bernoulli_sample);

    // Generate a vector of samples
    let normal_vec = rng.sample_vec(normal, 10);
    println!("10 samples from Normal(5, 2): {:?}", normal_vec);
}

#[cfg(feature = "random")]
#[allow(dead_code)]
fn random_array_example() {
    let mut rng = Random::default();

    // Create a 1D array with uniform random values
    let uniform = Uniform::new(0.0, 1.0).unwrap();
    let array1d = rng.sample_array(uniform, IxDyn(&[5]));
    println!("1D random array: {}", array1d);

    // Create a 2D array with normal random values
    let normal = Normal::new(0.0, 1.0).unwrap();
    let array2d = rng.sample_array(normal, IxDyn(&[3, 4]));
    println!("2D random array (3x4):");
    println!("{}", array2d);

    // Using the DistributionExt trait
    let dist = Uniform::new(1, 100).unwrap();
    let random_array = dist.random_array(&mut rng, IxDyn(&[2, 3]));
    println!("Random array using DistributionExt:");
    println!("{}", random_array);
}

#[cfg(feature = "random")]
#[allow(dead_code)]
fn seeded_random_example() {
    // Create two random generators with the same seed
    let mut rng1 = Random::seed(42);
    let mut rng2 = Random::seed(42);

    // They should produce the same sequence
    println!("Seeded RNG 1:");
    for _ in 0..3 {
        println!("  {:.6}", rng1.gen_range(0.0..1.0));
    }

    println!("Seeded RNG 2 (same seed):");
    for _ in 0..3 {
        println!("  {:.6}", rng2.gen_range(0.0..1.0));
    }

    // Different seed produces different sequence
    let mut rng3 = Random::seed(43);
    println!("Seeded RNG 3 (different seed):");
    for _ in 0..3 {
        println!("  {:.6}", rng3.gen_range(0.0..1.0));
    }
}

#[cfg(feature = "random")]
#[allow(dead_code)]
fn thread_local_random_example() {
    // Access the thread-local random generator
    let values = get_rng(|rng| {
        // Generate 5 random values
        let mut values = Vec::with_capacity(5);
        for _ in 0..5 {
            values.push(rng.gen_range(0..100));
        }
        values
    });

    println!("Random values from thread-local RNG: {:?}", values);
}

#[cfg(feature = "random")]
#[allow(dead_code)]
fn sampling_functions_example() {
    let mut rng = Random::default();

    // Sample from various distributions using helper functions
    let uniform01 = sampling::random_uniform01(&mut rng);
    let standard_normal = sampling::random_standard_normal(&mut rng);
    let custom_normal = sampling::random_normal(&mut rng, 10.0, 2.0);
    let lognormal = sampling::randomlognormal(&mut rng, 0.0, 1.0);
    let exponential = sampling::random_exponential(&mut rng, 2.0);

    println!("Uniform[0,1): {:.6}", uniform01);
    println!("Standard Normal: {:.6}", standard_normal);
    println!("Normal(10, 2): {:.6}", custom_normal);
    println!("LogNormal(0, 1): {:.6}", lognormal);
    println!("Exponential(2): {:.6}", exponential);

    // Generate arrays of random values
    let random_ints = sampling::random_integers(&mut rng, 1, 100, IxDyn(&[2, 2]));
    println!("\nRandom integers array (1-100):");
    println!("{}", random_ints);

    let random_floats = sampling::random_floats(&mut rng, -1.0, 1.0, IxDyn(&[2, 3]));
    println!("\nRandom floats array (-1 to 1):");
    println!("{}", random_floats);

    // Bootstrap sampling (sampling with replacement)
    let data_size = 100;
    let bootstrap_indices = sampling::bootstrap_indices(&mut rng, data_size, 10);
    println!("\nBootstrap sample indices: {:?}", bootstrap_indices);

    // Sample without replacement
    let subsample_indices = sampling::sample_without_replacement(&mut rng, data_size, 10);
    println!(
        "Subsample indices (without replacement): {:?}",
        subsample_indices
    );
}
