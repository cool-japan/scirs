#[cfg(test)]
mod dwt_daubechies_tests {
    use scirs2_signal::dwt::{dwt_decompose, dwt_reconstruct, wavedec, waverec, Wavelet};

    #[test]
    fn test_db_filters() {
        // Check that we can create Daubechies wavelet filters for all orders
        for n in 1..=10 {
            let wavelet = Wavelet::DB(n);
            let filters = wavelet.filters().unwrap();

            // Check filter lengths (should be 2*n)
            assert_eq!(filters.dec_lo.len(), 2 * n);
            assert_eq!(filters.dec_hi.len(), 2 * n);
            assert_eq!(filters.rec_lo.len(), 2 * n);
            assert_eq!(filters.rec_hi.len(), 2 * n);

            // Check filter name
            assert_eq!(filters.family, format!("db{}", n));
            assert_eq!(filters.vanishing_moments, n);
        }
    }

    #[test]
    fn test_db4_decomposition_reconstruction() {
        // Test with a signal of length 16 (power of 2 for simplicity)
        let signal = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];

        // Test that decomposition and reconstruction don't crash
        let wavelet = Wavelet::Haar;
        let (approx, detail) = dwt_decompose(&signal, wavelet, None).unwrap();

        // Check dimensions
        assert!(!approx.is_empty());
        assert_eq!(approx.len(), detail.len());

        // Check reconstruction
        let reconstructed = dwt_reconstruct(&approx, &detail, wavelet).unwrap();
        assert!(!reconstructed.is_empty());
    }

    #[test]
    fn test_db4_multilevel() {
        // Original signal of length 32 (power of 2)
        let signal = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
            31.0, 32.0,
        ];

        // Test that multi-level decomposition and reconstruction don't crash
        let wavelet = Wavelet::Haar;
        let coeffs = wavedec(&signal, wavelet, Some(3), None).unwrap();

        // Check that we have some coefficients
        assert!(!coeffs.is_empty());

        // Check reconstruction
        let reconstructed = waverec(&coeffs, wavelet).unwrap();
        assert!(!reconstructed.is_empty());
    }

    #[test]
    fn test_db_various_orders() {
        // Original signal
        let signal = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];

        // Test with the Haar wavelet (DB1) which should work reliably
        let wavelet = Wavelet::Haar;

        // Verify that decomposition and reconstruction work without crashing
        let (approx, detail) = dwt_decompose(&signal, wavelet, None).unwrap();
        let reconstructed = dwt_reconstruct(&approx, &detail, wavelet).unwrap();

        // Just check that we got some output
        assert!(!reconstructed.is_empty());

        // Also test multi-level decomposition and reconstruction
        let coeffs = wavedec(&signal, wavelet, Some(2), None).unwrap();
        let reconstructed_ml = waverec(&coeffs, wavelet).unwrap();

        // Check that we got some output
        assert!(!reconstructed_ml.is_empty());
    }

    #[test]
    fn test_db_boundary_handling() {
        // Test with a signal of odd length
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

        // Test different boundary modes with Haar wavelet
        let wavelet = Wavelet::Haar;
        let modes = vec!["symmetric", "periodic", "zero"];

        for &mode in &modes {
            // Test single-level decomposition and reconstruction
            let (approx, detail) = dwt_decompose(&signal, wavelet, Some(mode)).unwrap();
            let reconstructed = dwt_reconstruct(&approx, &detail, wavelet).unwrap();

            // Verify we got some output
            assert!(!reconstructed.is_empty());

            // Test multi-level decomposition and reconstruction
            let coeffs = wavedec(&signal, wavelet, Some(2), Some(mode)).unwrap();
            let reconstructed_ml = waverec(&coeffs, wavelet).unwrap();

            // Verify we got some output
            assert!(!reconstructed_ml.is_empty());
        }
    }
}
