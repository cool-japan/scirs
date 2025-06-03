use scirs2_signal::window::{bartlett, hamming, hann};

fn main() {
    let n = 10;

    println!("Window Analysis for N = {}\n", n);

    // Test Hamming window
    println!("=== Hamming Window ===");
    let hamming_win = hamming(n, true).unwrap();
    print_window_analysis(&hamming_win, "Hamming");

    // Test Hann window
    println!("\n=== Hann Window ===");
    let hann_win = hann(n, true).unwrap();
    print_window_analysis(&hann_win, "Hann");

    // Test Bartlett window
    println!("\n=== Bartlett Window ===");
    let bartlett_win = bartlett(n, true).unwrap();
    print_window_analysis(&bartlett_win, "Bartlett");
}

fn print_window_analysis(window: &[f64], name: &str) {
    // Print all values
    println!("Values:");
    for (i, &val) in window.iter().enumerate() {
        println!("  [{}]: {:.6}", i, val);
    }

    // Find maximum value and position
    let (max_idx, &max_val) = window
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();

    println!("\nMaximum value: {:.6} at index {}", max_val, max_idx);

    // Calculate expected theoretical values
    let n = window.len();
    match name {
        "Hamming" => {
            // Hamming: w[i] = 0.54 - 0.46 * cos(2π * i / (N-1))
            // Maximum at center: i = (N-1)/2
            let _center_idx = (n - 1) as f64 / 2.0;
            let theoretical_max = 0.54 + 0.46; // cos(π) = -1
            println!(
                "Theoretical maximum: {} (at symmetric center)",
                theoretical_max
            );

            // For even N, calculate actual center values
            if n % 2 == 0 {
                let idx1 = n / 2 - 1;
                let idx2 = n / 2;
                let val1 =
                    0.54 - 0.46 * (2.0 * std::f64::consts::PI * idx1 as f64 / (n - 1) as f64).cos();
                let val2 =
                    0.54 - 0.46 * (2.0 * std::f64::consts::PI * idx2 as f64 / (n - 1) as f64).cos();
                println!("Expected at index {}: {:.6}", idx1, val1);
                println!("Expected at index {}: {:.6}", idx2, val2);
            }
        }
        "Hann" => {
            // Hann: w[i] = 0.5 * (1 - cos(2π * i / (N-1)))
            // Maximum at center: cos(π) = -1, so w = 0.5 * (1 - (-1)) = 1.0
            println!("Theoretical maximum: 1.0 (at symmetric center)");

            // For even N, calculate actual center values
            if n % 2 == 0 {
                let idx1 = n / 2 - 1;
                let idx2 = n / 2;
                let val1 =
                    0.5 * (1.0 - (2.0 * std::f64::consts::PI * idx1 as f64 / (n - 1) as f64).cos());
                let val2 =
                    0.5 * (1.0 - (2.0 * std::f64::consts::PI * idx2 as f64 / (n - 1) as f64).cos());
                println!("Expected at index {}: {:.6}", idx1, val1);
                println!("Expected at index {}: {:.6}", idx2, val2);
            }
        }
        "Bartlett" => {
            // Bartlett: triangular window
            // For even N: maximum at indices (N/2-1) and N/2
            // Value: 2*i/(N-1) for i < N/2
            if n % 2 == 0 {
                let idx = n / 2 - 1;
                let expected = 2.0 * idx as f64 / (n - 1) as f64;
                println!(
                    "Expected maximum: {:.6} (at indices {} and {})",
                    expected,
                    idx,
                    idx + 1
                );
            } else {
                let idx = n / 2;
                println!("Expected maximum: 1.0 (at index {})", idx);
            }
        }
        _ => {}
    }
}
