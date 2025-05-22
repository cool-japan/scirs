#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use scirs2_fft::{fft, fft_simd, simd_support_available};

    #[test]
    fn test_simd_fft_basic() {
        // Skip test if SIMD is not available
        if !simd_support_available() {
            println!("SIMD support not available, skipping test");
            return;
        }

        // Create a simple test signal
        let signal: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];

        // Compute FFT with standard implementation
        let standard_result = fft(&signal, None).unwrap();

        // Compute FFT with SIMD implementation
        let simd_result = fft_simd(&signal, None, None).unwrap();

        // Compare results
        assert_eq!(standard_result.len(), simd_result.len());
        for i in 0..standard_result.len() {
            assert_relative_eq!(standard_result[i].re, simd_result[i].re, epsilon = 1e-10);
            assert_relative_eq!(standard_result[i].im, simd_result[i].im, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_simd_detection() {
        // This just verifies that the function runs without error
        let _ = simd_support_available();
    }
}
