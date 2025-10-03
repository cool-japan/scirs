use scirs2_core::ndarray::{array, Array1};
use scirs2_stats::distributions::multivariate::multinomial::Multinomial;
use statrs::statistics::Statistics;
use std::collections::HashMap;

#[allow(dead_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Multinomial Distribution Example");
    println!("-------------------------------");

    // Create a multinomial distribution for a 3-sided die rolled 10 times
    let n = 10;
    let p = array![0.2, 0.3, 0.5]; // Probabilities for each outcome

    let multinomial = Multinomial::new(n, p.clone())?;

    println!("Created Multinomial distribution:");
    println!("  Number of trials: {}", n);
    println!("  Probabilities: {:?}", p);
    println!();

    // Calculate distribution properties
    println!("Distribution properties:");
    let mean = multinomial.mean();
    println!("  Mean: {:?}", mean);
    let cov = multinomial.cov();
    println!("  Covariance matrix:");
    for i in 0..3 {
        println!("    {:?}", cov.row(i));
    }
    println!();

    // Calculate PMF at a few points
    println!("PMF values:");

    // Equally distributed outcomes
    let x1 = array![2.0, 3.0, 5.0];
    let pmf1 = multinomial.pmf(&x1);
    println!("  PMF at {:?}: {:.6}", x1, pmf1);

    // All outcomes of first type
    let x2 = array![10.0, 0.0, 0.0];
    let pmf2 = multinomial.pmf(&x2);
    println!("  PMF at {:?}: {:.10}", x2, pmf2);

    // Most likely outcome (if probabilities were equal)
    let x3 = array![3.0, 3.0, 4.0];
    let pmf3 = multinomial.pmf(&x3);
    println!("  PMF at {:?}: {:.6}", x3, pmf3);
    println!();

    // Generate random samples
    println!("Generating random samples:");
    let num_samples = 20;
    let samples = multinomial.rvs(num_samples)?;

    for (i, sample) in samples.iter().enumerate().take(5) {
        println!("  Sample {}: {:?}", i + 1, sample);
    }
    println!("  ...");
    println!();

    // Calculate sample statistics
    let mut sample_sum = Array1::<f64>::zeros(p.len());
    for sample in &samples {
        sample_sum += sample;
    }
    let sample_mean = sample_sum / num_samples as f64;

    println!("Sample statistics from {} samples:", num_samples);
    println!("  Sample mean: {:?}", sample_mean);
    println!("  Expected mean: {:?}", mean);
    println!();

    // Create a histogram of the samples
    println!("Frequency counts of outcomes:");
    let mut outcome_counts = HashMap::new();

    for sample in &samples {
        // Convert sample to a tuple for HashMap key
        let key = (sample[0] as u64, sample[1] as u64, sample[2] as u64);
        *outcome_counts.entry(key).or_insert(0) += 1;
    }

    // Display the histogram (top 5 most frequent outcomes)
    let mut counts: Vec<_> = outcome_counts.iter().collect();
    counts.sort_by(|a, b| b.1.cmp(a.1)); // Sort by count in descending order

    for ((x1, x2, x3), count) in counts.iter().take(5) {
        println!(
            "  Outcome [{}, {}, {}]: occurred {} times",
            x1, x2, x3, count
        );
    }

    println!();
    println!("Applications of the Multinomial distribution:");
    println!("1. Modeling outcomes of n independent categorical trials");
    println!("2. Analyzing results of multi-category experiments");
    println!("3. Testing goodness-of-fit for categorical data");
    println!("4. Text analysis (word counts across categories)");
    println!("5. Genetic sequence analysis (nucleotide frequencies)");

    Ok(())
}
