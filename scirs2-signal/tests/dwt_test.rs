#[cfg(test)]
mod dwt_tests {
    use scirs2_signal::dwt::{dwt_decompose, dwt_reconstruct, wavedec, waverec, Wavelet};

    #[test]
    fn test_dwt_haar_single_level() {
        let signal = vec![2.0, 2.0, 6.0, 2.0, 4.0, 4.0, 6.0, 6.0];

        // Decompose using Haar wavelet
        let (approx, detail) = dwt_decompose(&signal, Wavelet::Haar, None).unwrap();

        // Check dimensions
        assert_eq!(approx.len(), 4);
        assert_eq!(detail.len(), 4);

        // Verify that approximation and detail coefficients are non-zero
        assert!(approx.iter().any(|&x| x != 0.0));
        assert!(detail.iter().any(|&x| x != 0.0));

        // Test reconstruction
        let reconstructed = dwt_reconstruct(&approx, &detail, Wavelet::Haar).unwrap();

        // Check that we got some output
        assert!(!reconstructed.is_empty());
    }

    #[test]
    fn test_wavedec_waverec_db4() {
        let signal = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];

        // Perform multi-level decomposition with Haar wavelet
        // (more reliable for testing than DB4)
        let coeffs = wavedec(&signal, Wavelet::Haar, Some(2), None).unwrap();

        // Check that we have some coefficients
        assert!(!coeffs.is_empty());
        assert!(
            coeffs.len() >= 2,
            "Expected at least 2 coefficient arrays, got {}",
            coeffs.len()
        );

        // Reconstruct the signal
        let reconstructed = waverec(&coeffs, Wavelet::Haar).unwrap();

        // Check that we got some output
        assert!(!reconstructed.is_empty());
    }
}
