use crate::error::SignalResult;
use crate::features::statistical::calculate_quantile;
use crate::features::statistical::calculate_std;
use std::collections::HashMap;

/// Extract entropy-based features from a time series
pub fn extract_entropy_features(
    signal: &[f64],
    features: &mut HashMap<String, f64>,
) -> SignalResult<()> {
    // Calculate Shannon entropy
    let shannon_entropy = calculate_shannon_entropy(signal);
    features.insert("shannon_entropy".to_string(), shannon_entropy);

    // Calculate approximate entropy
    let approx_entropy = calculate_approximate_entropy(signal, 2, 0.2);
    features.insert("approximate_entropy".to_string(), approx_entropy);

    // Calculate sample entropy
    let sample_entropy = calculate_sample_entropy(signal, 2, 0.2);
    features.insert("sample_entropy".to_string(), sample_entropy);

    // Calculate permutation entropy
    let perm_entropy = calculate_permutation_entropy(signal, 3);
    features.insert("permutation_entropy".to_string(), perm_entropy);

    Ok(())
}

/// Calculate Shannon entropy of a signal
pub fn calculate_shannon_entropy(signal: &[f64]) -> f64 {
    if signal.is_empty() {
        return 0.0;
    }

    // Determine range and bin width for histogram
    let min = signal.iter().copied().fold(f64::INFINITY, f64::min);
    let max = signal.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    if min == max {
        return 0.0; // Constant signal has zero entropy
    }

    // Use Freedman-Diaconis rule for bin width
    let n = signal.len();
    let iqr = {
        let mut sorted = signal.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        calculate_quantile(&sorted, 0.75) - calculate_quantile(&sorted, 0.25)
    };

    let bin_width = 2.0 * iqr / (n as f64).powf(1.0 / 3.0);
    let num_bins = ((max - min) / bin_width).ceil() as usize;
    let num_bins = num_bins.clamp(1, 100); // Limit to avoid excessive bins

    // Create histogram
    let mut histogram = vec![0; num_bins];

    for &value in signal {
        let bin = ((value - min) / (max - min) * num_bins as f64).floor() as usize;
        let bin = bin.min(num_bins - 1); // Ensure index is valid
        histogram[bin] += 1;
    }

    // Calculate entropy
    let mut entropy = 0.0;

    for &count in &histogram {
        if count > 0 {
            let probability = count as f64 / n as f64;
            entropy -= probability * probability.ln();
        }
    }

    entropy
}

/// Calculate approximate entropy of a signal
pub fn calculate_approximate_entropy(signal: &[f64], m: usize, r: f64) -> f64 {
    if signal.is_empty() || m == 0 {
        return 0.0;
    }

    let n = signal.len();
    if n <= m + 1 {
        return 0.0; // Not enough data points
    }

    // Normalize tolerance
    let std_dev = calculate_std(signal);
    let tolerance = r * std_dev;

    if tolerance == 0.0 {
        return 0.0;
    }

    // Calculate phi for different m values
    let phi_m = calculate_phi(signal, m, tolerance);
    let phi_m_plus_1 = calculate_phi(signal, m + 1, tolerance);

    // Return approximate entropy
    phi_m - phi_m_plus_1
}

/// Helper function for approximate entropy calculation
fn calculate_phi(signal: &[f64], m: usize, tolerance: f64) -> f64 {
    let n = signal.len();
    let mut total = 0.0;

    for i in 0..n - m + 1 {
        let template_i = &signal[i..i + m];
        let mut local_count = 0.0;

        for j in 0..n - m + 1 {
            let template_j = &signal[j..j + m];
            let mut max_diff: f64 = 0.0;

            for k in 0..m {
                let diff = (template_i[k] - template_j[k]).abs();
                max_diff = if diff > max_diff { diff } else { max_diff };
            }

            if max_diff <= tolerance {
                local_count += 1.0;
            }
        }

        total += (local_count / (n - m + 1) as f64).ln();
    }

    total / (n - m + 1) as f64
}

/// Calculate sample entropy of a signal
pub fn calculate_sample_entropy(signal: &[f64], m: usize, r: f64) -> f64 {
    if signal.is_empty() || m == 0 {
        return 0.0;
    }

    let n = signal.len();
    if n <= m + 1 {
        return 0.0; // Not enough data points
    }

    // Normalize tolerance
    let std_dev = calculate_std(signal);
    let tolerance = r * std_dev;

    if tolerance == 0.0 {
        return 0.0;
    }

    // Count matches for m and m+1
    let count_m = count_matches(signal, m, tolerance);
    let count_m_plus_1 = count_matches(signal, m + 1, tolerance);

    // Return sample entropy
    if count_m > 0.0 {
        -((count_m_plus_1 / count_m).ln())
    } else {
        f64::MAX
    }
}

/// Helper function for sample entropy calculation
fn count_matches(signal: &[f64], m: usize, tolerance: f64) -> f64 {
    let n = signal.len();
    let mut match_count = 0.0;

    for i in 0..n - m {
        let template_i = &signal[i..i + m];

        for j in i + 1..n - m + 1 {
            let template_j = &signal[j..j + m];
            let mut max_diff: f64 = 0.0;

            for k in 0..m {
                let diff = (template_i[k] - template_j[k]).abs();
                max_diff = if diff > max_diff { diff } else { max_diff };
            }

            if max_diff <= tolerance {
                match_count += 1.0;
            }
        }
    }

    match_count * 2.0 / ((n - m) as f64 * (n - m - 1) as f64)
}

/// Calculate permutation entropy of a signal
pub fn calculate_permutation_entropy(signal: &[f64], order: usize) -> f64 {
    if signal.is_empty() || order < 2 {
        return 0.0;
    }

    let n = signal.len();
    if n < order + 1 {
        return 0.0; // Not enough data points
    }

    // Calculate factorial(order) - number of possible permutations
    let factorial = (1..=order).product::<usize>() as f64;

    // Count occurrences of each permutation pattern
    let mut pattern_counts = HashMap::new();

    for i in 0..n - order + 1 {
        let mut pattern = Vec::with_capacity(order);
        for j in 0..order {
            pattern.push(signal[i + j]);
        }

        // Find rank ordering of the pattern
        let mut idx: Vec<usize> = (0..order).collect();
        idx.sort_by(|&a, &b| pattern[a].partial_cmp(&pattern[b]).unwrap());

        // Convert rank order to a hash key
        let key: String = idx
            .iter()
            .map(|&i| char::from_digit(i as u32, 10).unwrap())
            .collect();

        // Update count for this pattern
        *pattern_counts.entry(key).or_insert(0) += 1;
    }

    // Calculate entropy
    let total_patterns = (n - order + 1) as f64;
    let mut entropy = 0.0;

    for &count in pattern_counts.values() {
        let probability = count as f64 / total_patterns;
        entropy -= probability * probability.ln();
    }

    // Normalize by ln(factorial(order)) to get a value in [0, 1]
    entropy / factorial.ln()
}
