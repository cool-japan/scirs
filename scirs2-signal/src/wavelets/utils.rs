//! Helper functions for wavelet transforms

/// Helper function to calculate factorial
pub fn factorial(n: usize) -> usize {
    if n <= 1 {
        1
    } else {
        n * factorial(n - 1)
    }
}
