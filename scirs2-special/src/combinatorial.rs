//! Combinatorial functions
//!
//! This module provides various combinatorial functions including:
//! - Factorial and related functions
//! - Binomial coefficients
//! - Permutations and combinations
//! - Stirling numbers
//! - Bell numbers
//! - Bernoulli numbers
//! - Euler numbers

use crate::error::{SpecialError, SpecialResult};
use crate::gamma::gamma;

/// Computes the factorial of a non-negative integer.
///
/// For large values, this uses the gamma function: n! = Γ(n+1).
///
/// # Arguments
///
/// * `n` - Non-negative integer
///
/// # Returns
///
/// * `SpecialResult<f64>` - The factorial value n!
///
/// # Examples
///
/// ```
/// use scirs2_special::factorial;
///
/// assert_eq!(factorial(0).unwrap(), 1.0);
/// assert_eq!(factorial(5).unwrap(), 120.0);
/// assert_eq!(factorial(10).unwrap(), 3628800.0);
/// ```
pub fn factorial(n: u32) -> SpecialResult<f64> {
    if n <= 20 {
        // Use direct calculation for small values
        let mut result = 1.0;
        for i in 1..=n {
            result *= i as f64;
        }
        Ok(result)
    } else {
        // Use gamma function for larger values
        Ok(gamma((n + 1) as f64))
    }
}

/// Computes the double factorial of a non-negative integer.
///
/// Double factorial n!! is defined as:
/// - For even n: n!! = n × (n-2) × (n-4) × ... × 2
/// - For odd n: n!! = n × (n-2) × (n-4) × ... × 1
/// - 0!! = 1 by convention
///
/// # Arguments
///
/// * `n` - Non-negative integer
///
/// # Returns
///
/// * `SpecialResult<f64>` - The double factorial value n!!
///
/// # Examples
///
/// ```
/// use scirs2_special::double_factorial;
///
/// assert_eq!(double_factorial(0).unwrap(), 1.0);
/// assert_eq!(double_factorial(5).unwrap(), 15.0); // 5 × 3 × 1
/// assert_eq!(double_factorial(6).unwrap(), 48.0); // 6 × 4 × 2
/// ```
pub fn double_factorial(n: u32) -> SpecialResult<f64> {
    if n == 0 {
        return Ok(1.0);
    }

    let mut result = 1.0;
    let mut i = n;
    while i > 0 {
        result *= i as f64;
        i = i.saturating_sub(2);
    }
    Ok(result)
}

/// Computes the binomial coefficient "n choose k".
///
/// The binomial coefficient C(n,k) = n! / (k! × (n-k)!) represents
/// the number of ways to choose k items from n items.
///
/// # Arguments
///
/// * `n` - Total number of items
/// * `k` - Number of items to choose
///
/// # Returns
///
/// * `SpecialResult<f64>` - The binomial coefficient C(n,k)
///
/// # Examples
///
/// ```
/// use scirs2_special::binomial;
///
/// assert_eq!(binomial(5, 2).unwrap(), 10.0);
/// assert_eq!(binomial(10, 3).unwrap(), 120.0);
/// assert_eq!(binomial(7, 0).unwrap(), 1.0);
/// assert_eq!(binomial(7, 7).unwrap(), 1.0);
/// ```
pub fn binomial(n: u32, k: u32) -> SpecialResult<f64> {
    if k > n {
        return Ok(0.0);
    }

    if k == 0 || k == n {
        return Ok(1.0);
    }

    // Use symmetry: C(n,k) = C(n,n-k)
    let k = if k > n - k { n - k } else { k };

    // For small values, use direct calculation to avoid precision issues
    if n <= 30 {
        let mut result = 1.0;
        for i in 0..k {
            result = result * (n - i) as f64 / (i + 1) as f64;
        }
        Ok(result)
    } else {
        // Use gamma function for larger values
        let n_fact = gamma((n + 1) as f64);
        let k_fact = gamma((k + 1) as f64);
        let nk_fact = gamma((n - k + 1) as f64);
        Ok(n_fact / (k_fact * nk_fact))
    }
}

/// Computes the number of permutations of n items taken k at a time.
///
/// P(n,k) = n! / (n-k)! represents the number of ways to arrange
/// k items from n items where order matters.
///
/// # Arguments
///
/// * `n` - Total number of items
/// * `k` - Number of items to arrange
///
/// # Returns
///
/// * `SpecialResult<f64>` - The number of permutations P(n,k)
///
/// # Examples
///
/// ```
/// use scirs2_special::permutations;
///
/// assert_eq!(permutations(5, 2).unwrap(), 20.0);
/// assert_eq!(permutations(10, 3).unwrap(), 720.0);
/// assert_eq!(permutations(7, 0).unwrap(), 1.0);
/// ```
pub fn permutations(n: u32, k: u32) -> SpecialResult<f64> {
    if k > n {
        return Ok(0.0);
    }

    if k == 0 {
        return Ok(1.0);
    }

    // For small values, use direct calculation
    if n <= 30 {
        let mut result = 1.0;
        for i in 0..k {
            result *= (n - i) as f64;
        }
        Ok(result)
    } else {
        // Use gamma function for larger values
        let n_fact = gamma((n + 1) as f64);
        let nk_fact = gamma((n - k + 1) as f64);
        Ok(n_fact / nk_fact)
    }
}

/// Computes the Stirling number of the first kind.
///
/// The unsigned Stirling number of the first kind s(n,k) counts
/// the number of permutations of n elements with exactly k cycles.
///
/// # Arguments
///
/// * `n` - Number of elements
/// * `k` - Number of cycles
///
/// # Returns
///
/// * `SpecialResult<f64>` - The unsigned Stirling number s(n,k)
///
/// # Examples
///
/// ```
/// use scirs2_special::stirling_first;
///
/// assert_eq!(stirling_first(0, 0).unwrap(), 1.0);
/// assert_eq!(stirling_first(4, 2).unwrap(), 11.0);
/// assert_eq!(stirling_first(5, 3).unwrap(), 35.0);
/// ```
pub fn stirling_first(n: u32, k: u32) -> SpecialResult<f64> {
    if n == 0 && k == 0 {
        return Ok(1.0);
    }
    if n == 0 || k == 0 || k > n {
        return Ok(0.0);
    }

    // Use recurrence relation: s(n,k) = (n-1) * s(n-1,k) + s(n-1,k-1)
    let mut dp = vec![vec![0.0; (k + 1) as usize]; (n + 1) as usize];
    dp[0][0] = 1.0;

    for i in 1..=n as usize {
        for j in 1..=std::cmp::min(i, k as usize) {
            dp[i][j] = (i - 1) as f64 * dp[i - 1][j] + dp[i - 1][j - 1];
        }
    }

    Ok(dp[n as usize][k as usize])
}

/// Computes the Stirling number of the second kind.
///
/// The Stirling number of the second kind S(n,k) counts the number
/// of ways to partition n elements into exactly k non-empty subsets.
///
/// # Arguments
///
/// * `n` - Number of elements
/// * `k` - Number of subsets
///
/// # Returns
///
/// * `SpecialResult<f64>` - The Stirling number S(n,k)
///
/// # Examples
///
/// ```
/// use scirs2_special::stirling_second;
///
/// assert_eq!(stirling_second(0, 0).unwrap(), 1.0);
/// assert_eq!(stirling_second(4, 2).unwrap(), 7.0);
/// assert_eq!(stirling_second(5, 3).unwrap(), 25.0);
/// ```
pub fn stirling_second(n: u32, k: u32) -> SpecialResult<f64> {
    if n == 0 && k == 0 {
        return Ok(1.0);
    }
    if n == 0 || k == 0 || k > n {
        return Ok(0.0);
    }

    // Use recurrence relation: S(n,k) = k * S(n-1,k) + S(n-1,k-1)
    let mut dp = vec![vec![0.0; (k + 1) as usize]; (n + 1) as usize];
    dp[0][0] = 1.0;

    for i in 1..=n as usize {
        for j in 1..=std::cmp::min(i, k as usize) {
            dp[i][j] = j as f64 * dp[i - 1][j] + dp[i - 1][j - 1];
        }
    }

    Ok(dp[n as usize][k as usize])
}

/// Computes the Bell number B(n).
///
/// The Bell number B(n) counts the number of ways to partition
/// a set of n elements into non-empty subsets.
///
/// # Arguments
///
/// * `n` - Number of elements
///
/// # Returns
///
/// * `SpecialResult<f64>` - The Bell number B(n)
///
/// # Examples
///
/// ```
/// use scirs2_special::bell_number;
///
/// assert_eq!(bell_number(0).unwrap(), 1.0);
/// assert_eq!(bell_number(1).unwrap(), 1.0);
/// assert_eq!(bell_number(2).unwrap(), 2.0);
/// assert_eq!(bell_number(3).unwrap(), 5.0);
/// assert_eq!(bell_number(4).unwrap(), 15.0);
/// ```
pub fn bell_number(n: u32) -> SpecialResult<f64> {
    if n == 0 {
        return Ok(1.0);
    }

    // B(n) = sum of S(n,k) for k from 0 to n
    let mut result = 0.0;
    for k in 0..=n {
        result += stirling_second(n, k)?;
    }
    Ok(result)
}

/// Computes the Bernoulli number B(n).
///
/// The Bernoulli numbers are a sequence of rational numbers defined
/// by the generating function t/(e^t - 1) = sum(B_n * t^n / n!).
///
/// This implementation returns the modern convention where B_1 = -1/2.
///
/// # Arguments
///
/// * `n` - Index of the Bernoulli number
///
/// # Returns
///
/// * `SpecialResult<f64>` - The Bernoulli number B(n)
///
/// # Examples
///
/// ```
/// use scirs2_special::bernoulli_number;
/// use approx::assert_relative_eq;
///
/// assert_eq!(bernoulli_number(0).unwrap(), 1.0);
/// assert_relative_eq!(bernoulli_number(1).unwrap(), -0.5, epsilon = 1e-10);
/// assert_relative_eq!(bernoulli_number(2).unwrap(), 1.0/6.0, epsilon = 1e-10);
/// assert_eq!(bernoulli_number(3).unwrap(), 0.0); // Odd Bernoulli numbers are 0 (except B_1)
/// ```
pub fn bernoulli_number(n: u32) -> SpecialResult<f64> {
    if n == 0 {
        return Ok(1.0);
    }
    if n == 1 {
        return Ok(-0.5);
    }
    if n > 1 && n % 2 == 1 {
        return Ok(0.0); // Odd Bernoulli numbers are zero (except B_1)
    }

    // For small even n, use known values
    match n {
        2 => return Ok(1.0 / 6.0),
        4 => return Ok(-1.0 / 30.0),
        6 => return Ok(1.0 / 42.0),
        8 => return Ok(-1.0 / 30.0),
        10 => return Ok(5.0 / 66.0),
        12 => return Ok(-691.0 / 2730.0),
        _ => {}
    }

    // For larger values, use recurrence relation
    // B_n = -1/(n+1) * sum_{k=0}^{n-1} C(n+1,k) * B_k
    let mut bernoulli = vec![0.0; (n + 1) as usize];
    bernoulli[0] = 1.0;
    if n >= 1 {
        bernoulli[1] = -0.5;
    }

    for m in 2..=(n as usize) {
        if m % 2 == 1 {
            bernoulli[m] = 0.0;
            continue;
        }

        let mut sum = 0.0;
        for (k, &bernoulli_k) in bernoulli.iter().enumerate().take(m) {
            let binom_coeff = binomial((m + 1) as u32, k as u32)?;
            sum += binom_coeff * bernoulli_k;
        }
        bernoulli[m] = -sum / (m + 1) as f64;
    }

    Ok(bernoulli[n as usize])
}

/// Computes the Euler number E(n).
///
/// The Euler numbers are integers defined by the generating function
/// sech(t) = 2/(e^t + e^(-t)) = sum(E_n * t^n / n!).
/// Only even-indexed Euler numbers are non-zero.
///
/// # Arguments
///
/// * `n` - Index of the Euler number
///
/// # Returns
///
/// * `SpecialResult<f64>` - The Euler number E(n)
///
/// # Examples
///
/// ```
/// use scirs2_special::euler_number;
///
/// assert_eq!(euler_number(0).unwrap(), 1.0);
/// assert_eq!(euler_number(1).unwrap(), 0.0);
/// assert_eq!(euler_number(2).unwrap(), -1.0);
/// assert_eq!(euler_number(4).unwrap(), 5.0);
/// ```
pub fn euler_number(n: u32) -> SpecialResult<f64> {
    if n % 2 == 1 {
        return Ok(0.0); // Odd Euler numbers are zero
    }

    // Known values for small even n
    match n {
        0 => return Ok(1.0),
        2 => return Ok(-1.0),
        4 => return Ok(5.0),
        6 => return Ok(-61.0),
        8 => return Ok(1385.0),
        10 => return Ok(-50521.0),
        _ => {}
    }

    // For larger values, use recurrence relation
    // This gets computationally expensive for large n
    if n > 20 {
        return Err(SpecialError::NotImplementedError(
            "Euler numbers for n > 20 not implemented".to_string(),
        ));
    }

    // Use recurrence relation based on generating function
    let mut euler = vec![0.0; (n + 1) as usize];
    euler[0] = 1.0;

    for m in (2..=(n as usize)).step_by(2) {
        let mut sum = 0.0;
        for k in (0..m).step_by(2) {
            let binom_coeff = binomial(m as u32, k as u32)?;
            sum += binom_coeff * euler[k];
        }
        euler[m] = -sum;
    }

    Ok(euler[n as usize])
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_factorial() {
        assert_eq!(factorial(0).unwrap(), 1.0);
        assert_eq!(factorial(1).unwrap(), 1.0);
        assert_eq!(factorial(5).unwrap(), 120.0);
        assert_eq!(factorial(10).unwrap(), 3628800.0);

        // Test larger values using gamma function
        assert_relative_eq!(factorial(15).unwrap(), 1307674368000.0, epsilon = 1.0);
    }

    #[test]
    fn test_double_factorial() {
        assert_eq!(double_factorial(0).unwrap(), 1.0);
        assert_eq!(double_factorial(1).unwrap(), 1.0);
        assert_eq!(double_factorial(2).unwrap(), 2.0);
        assert_eq!(double_factorial(5).unwrap(), 15.0); // 5 × 3 × 1
        assert_eq!(double_factorial(6).unwrap(), 48.0); // 6 × 4 × 2
        assert_eq!(double_factorial(8).unwrap(), 384.0); // 8 × 6 × 4 × 2
    }

    #[test]
    fn test_binomial() {
        assert_eq!(binomial(5, 2).unwrap(), 10.0);
        assert_eq!(binomial(10, 3).unwrap(), 120.0);
        assert_eq!(binomial(7, 0).unwrap(), 1.0);
        assert_eq!(binomial(7, 7).unwrap(), 1.0);
        assert_eq!(binomial(5, 10).unwrap(), 0.0); // k > n

        // Test symmetry
        assert_eq!(binomial(10, 3).unwrap(), binomial(10, 7).unwrap());
    }

    #[test]
    fn test_permutations() {
        assert_eq!(permutations(5, 2).unwrap(), 20.0);
        assert_eq!(permutations(10, 3).unwrap(), 720.0);
        assert_eq!(permutations(7, 0).unwrap(), 1.0);
        assert_eq!(permutations(5, 10).unwrap(), 0.0); // k > n
    }

    #[test]
    fn test_stirling_first() {
        assert_eq!(stirling_first(0, 0).unwrap(), 1.0);
        assert_eq!(stirling_first(4, 2).unwrap(), 11.0);
        assert_eq!(stirling_first(5, 3).unwrap(), 35.0);
        assert_eq!(stirling_first(3, 0).unwrap(), 0.0);
        assert_eq!(stirling_first(0, 3).unwrap(), 0.0);
    }

    #[test]
    fn test_stirling_second() {
        assert_eq!(stirling_second(0, 0).unwrap(), 1.0);
        assert_eq!(stirling_second(4, 2).unwrap(), 7.0);
        assert_eq!(stirling_second(5, 3).unwrap(), 25.0);
        assert_eq!(stirling_second(3, 0).unwrap(), 0.0);
        assert_eq!(stirling_second(0, 3).unwrap(), 0.0);
    }

    #[test]
    fn test_bell_number() {
        assert_eq!(bell_number(0).unwrap(), 1.0);
        assert_eq!(bell_number(1).unwrap(), 1.0);
        assert_eq!(bell_number(2).unwrap(), 2.0);
        assert_eq!(bell_number(3).unwrap(), 5.0);
        assert_eq!(bell_number(4).unwrap(), 15.0);
        assert_eq!(bell_number(5).unwrap(), 52.0);
    }

    #[test]
    fn test_bernoulli_number() {
        assert_eq!(bernoulli_number(0).unwrap(), 1.0);
        assert_relative_eq!(bernoulli_number(1).unwrap(), -0.5, epsilon = 1e-10);
        assert_relative_eq!(bernoulli_number(2).unwrap(), 1.0 / 6.0, epsilon = 1e-10);
        assert_eq!(bernoulli_number(3).unwrap(), 0.0);
        assert_relative_eq!(bernoulli_number(4).unwrap(), -1.0 / 30.0, epsilon = 1e-10);
        assert_eq!(bernoulli_number(5).unwrap(), 0.0);
    }

    #[test]
    fn test_euler_number() {
        assert_eq!(euler_number(0).unwrap(), 1.0);
        assert_eq!(euler_number(1).unwrap(), 0.0);
        assert_eq!(euler_number(2).unwrap(), -1.0);
        assert_eq!(euler_number(3).unwrap(), 0.0);
        assert_eq!(euler_number(4).unwrap(), 5.0);
        assert_eq!(euler_number(6).unwrap(), -61.0);
    }
}
