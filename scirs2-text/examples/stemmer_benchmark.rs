use scirs2_text::{LancasterStemmer, PorterStemmer, SnowballStemmer, Stemmer};
use std::error::Error;
use std::time::{Duration, Instant};

fn main() -> Result<(), Box<dyn Error>> {
    println!("Stemmer Benchmark");
    println!("----------------");

    // Create instances of different stemmers
    let porter_stemmer = PorterStemmer::new();
    let snowball_stemmer = SnowballStemmer::new("english")?;
    let lancaster_stemmer = LancasterStemmer::new();

    // Test words to benchmark
    let test_words = vec![
        "running",
        "ran",
        "runs",
        "easily",
        "fishing",
        "fished",
        "troubled",
        "troubling",
        "troubles",
        "production",
        "productive",
        "argument",
        "arguing",
        "university",
        "universities",
        "maximizing",
        "maximum",
        "presumably",
        "multiply",
        "opposition",
        "computational",
        "algorithms",
        "mathematics",
        "scientific",
        "engineering",
        "development",
        "statistics",
        "probability",
        "hypothesis",
        "experiment",
        "observation",
        "distribution",
        "estimation",
        "regression",
        "classification",
        "clustering",
        "dimensionality",
        "transformation",
        "optimization",
        "visualization",
        "inference",
    ];

    // Repeat each test many times to get meaningful timing
    let iterations = 10000;

    // Benchmark Porter stemmer
    let start = Instant::now();
    for _ in 0..iterations {
        for word in &test_words {
            let _ = porter_stemmer.stem(word)?;
        }
    }
    let porter_duration = start.elapsed();

    // Benchmark Snowball stemmer
    let start = Instant::now();
    for _ in 0..iterations {
        for word in &test_words {
            let _ = snowball_stemmer.stem(word)?;
        }
    }
    let snowball_duration = start.elapsed();

    // Benchmark Lancaster stemmer
    let start = Instant::now();
    for _ in 0..iterations {
        for word in &test_words {
            let _ = lancaster_stemmer.stem(word)?;
        }
    }
    let lancaster_duration = start.elapsed();

    // Print results
    println!(
        "Results ({} iterations on {} words):",
        iterations,
        test_words.len()
    );
    println!(
        "Porter stemmer:   {:?} ({:.1} words/ms)",
        porter_duration,
        words_per_ms(test_words.len(), iterations, porter_duration)
    );
    println!(
        "Snowball stemmer: {:?} ({:.1} words/ms)",
        snowball_duration,
        words_per_ms(test_words.len(), iterations, snowball_duration)
    );
    println!(
        "Lancaster stemmer: {:?} ({:.1} words/ms)",
        lancaster_duration,
        words_per_ms(test_words.len(), iterations, lancaster_duration)
    );

    println!("\nRelative performance (Porter = 1.0):");
    println!("Porter:   1.0");
    println!(
        "Snowball: {:.1}",
        porter_duration.as_nanos() as f64 / snowball_duration.as_nanos() as f64
    );
    println!(
        "Lancaster: {:.1}",
        porter_duration.as_nanos() as f64 / lancaster_duration.as_nanos() as f64
    );

    Ok(())
}

fn words_per_ms(word_count: usize, iterations: usize, duration: Duration) -> f64 {
    (word_count * iterations) as f64 / duration.as_millis() as f64
}
