//! Tests for utilities module
//!
//! This module contains unit tests for the utilities module.

#[cfg(test)]
mod spectral_tests {
    use super::super::spectral::*;
    use approx::assert_relative_eq;
    use ndarray::{Array1, Array2};

    #[test]
    fn test_energy_spectral_density() {
        // Create a simple PSD
        let psd = vec![1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0];
        let fs = 100.0; // Sample rate in Hz

        let esd = energy_spectral_density(&psd, fs).unwrap();

        // Check scaling by sample interval (1/fs)
        for (i, &p) in psd.iter().enumerate() {
            assert_relative_eq!(esd[i], p / fs, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_normalized_psd() {
        // Create a simple PSD
        let psd = vec![1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0];

        let norm_psd = normalized_psd(&psd).unwrap();

        // Check sum is 1.0
        let sum: f64 = norm_psd.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-10);

        // Check shape is preserved
        for (i, &p) in psd.iter().enumerate() {
            if i > 0 {
                assert_relative_eq!(norm_psd[i] / norm_psd[0], p / psd[0], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_spectral_centroid() {
        // Create a symmetric PSD with peak in the middle
        let psd = vec![1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0];
        let freqs = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let centroid = spectral_centroid(&psd, &freqs).unwrap();

        // For symmetric PSD with peak in the middle, centroid should be at the middle frequency
        assert_relative_eq!(centroid, 3.0, epsilon = 1e-10);

        // Test non-symmetric PSD
        let psd = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let freqs = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let centroid = spectral_centroid(&psd, &freqs).unwrap();

        // Centroid should be biased toward higher frequencies
        assert!(centroid > 3.0);
    }

    #[test]
    fn test_spectral_spread() {
        // Create a symmetric PSD with peak in the middle
        let psd = vec![1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0];
        let freqs = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let spread = spectral_spread(&psd, &freqs, None).unwrap();

        // Spread should be positive
        assert!(spread > 0.0);

        // Create a very narrow PSD
        let psd = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        let freqs = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let spread = spectral_spread(&psd, &freqs, None).unwrap();

        // Spread should be very small for narrow PSD
        assert!(spread < 0.1);
    }

    #[test]
    fn test_spectral_skewness() {
        // Create a symmetric PSD
        let psd = vec![1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0];
        let freqs = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let skewness = spectral_skewness(&psd, &freqs, None, None).unwrap();

        // Skewness should be close to zero for symmetric distribution
        assert_relative_eq!(skewness, 0.0, epsilon = 1e-10);

        // Create a distribution with more energy at lower frequencies
        let psd = vec![4.0, 3.0, 2.0, 1.0, 0.5, 0.3, 0.1];
        let freqs = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let skewness = spectral_skewness(&psd, &freqs, None, None).unwrap();

        // In spectral analysis, this is considered to have a positive skewness
        // This is because spectral skewness in signal processing often measures
        // the asymmetry of the spectrum relative to its centroid, looking at
        // whether energy is concentrated below or above the centroid
        assert!(
            skewness > 0.0,
            "Expected positive spectral skewness, got {}",
            skewness
        );

        // Create a distribution with more energy at higher frequencies
        let psd = vec![0.1, 0.3, 0.5, 1.0, 2.0, 3.0, 4.0];
        let freqs = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let skewness = spectral_skewness(&psd, &freqs, None, None).unwrap();

        // In spectral analysis with this implementation, a distribution with more energy
        // at high frequencies gives negative skewness. This is due to how the spectral
        // skewness formula is implemented, which is consistent with traditional
        // statistical skewness.
        assert!(
            skewness < 0.0,
            "Expected negative skewness for this PSD, got {}",
            skewness
        );
    }

    #[test]
    fn test_spectral_kurtosis() {
        // Create a flat PSD (uniform distribution)
        let psd = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let freqs = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let kurtosis = spectral_kurtosis(&psd, &freqs, None, None).unwrap();

        // Kurtosis should be negative for uniform distribution (platykurtic)
        assert!(kurtosis < 0.0);

        // Create a peaked PSD (leptokurtic)
        let psd = vec![0.1, 0.2, 0.5, 5.0, 0.5, 0.2, 0.1];
        let freqs = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let kurtosis = spectral_kurtosis(&psd, &freqs, None, None).unwrap();

        // Kurtosis should be positive for peaked distribution
        assert!(kurtosis > 0.0);
    }

    #[test]
    fn test_spectral_flatness() {
        // Create a flat PSD (white noise-like)
        let psd = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        let flatness = spectral_flatness(&psd).unwrap();

        // Flatness should be close to 1.0 for flat PSD
        assert_relative_eq!(flatness, 1.0, epsilon = 1e-10);

        // Create a PSD with a single peak (tone-like)
        let psd = vec![0.01, 0.01, 0.01, 1.0, 0.01, 0.01, 0.01];

        let flatness = spectral_flatness(&psd).unwrap();

        // Flatness should be close to 0.0 for peak PSD
        assert!(flatness < 0.3);
    }

    #[test]
    fn test_spectral_flux() {
        // Create two identical PSDs
        let psd1 = vec![1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0];
        let psd2 = vec![1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0];

        let flux_l1 = spectral_flux(&psd1, &psd2, "l1").unwrap();
        let flux_l2 = spectral_flux(&psd1, &psd2, "l2").unwrap();
        let flux_max = spectral_flux(&psd1, &psd2, "max").unwrap();

        // Flux should be 0.0 for identical PSDs
        assert_relative_eq!(flux_l1, 0.0, epsilon = 1e-10);
        assert_relative_eq!(flux_l2, 0.0, epsilon = 1e-10);
        assert_relative_eq!(flux_max, 0.0, epsilon = 1e-10);

        // Create two different PSDs
        let psd1 = vec![1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0];
        let psd2 = vec![0.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0];

        let flux_l1 = spectral_flux(&psd1, &psd2, "l1").unwrap();
        let flux_l2 = spectral_flux(&psd1, &psd2, "l2").unwrap();
        let flux_max = spectral_flux(&psd1, &psd2, "max").unwrap();

        // Flux should be positive for different PSDs
        assert!(flux_l1 > 0.0);
        assert!(flux_l2 > 0.0);
        assert!(flux_max > 0.0);

        // Test different norms
        assert!(flux_l1 >= flux_l2);
        assert!(flux_l2 >= flux_max);
    }

    #[test]
    fn test_spectral_rolloff() {
        // Create a PSD with energy concentrated in first half
        let psd = vec![1.0, 2.0, 3.0, 4.0, 0.1, 0.1, 0.1];
        let freqs = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let rolloff_95 = spectral_rolloff(&psd, &freqs, 0.95).unwrap();
        let rolloff_50 = spectral_rolloff(&psd, &freqs, 0.50).unwrap();

        // 95% rolloff should be higher than 50% rolloff
        assert!(rolloff_95 >= rolloff_50);

        // 50% rolloff should be near the middle frequency for this distribution
        assert!(rolloff_50 <= 3.0);

        // Create a PSD with energy concentrated in second half
        let psd = vec![0.1, 0.1, 0.1, 0.1, 3.0, 4.0, 5.0];
        let freqs = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let rolloff = spectral_rolloff(&psd, &freqs, 0.95).unwrap();

        // Rolloff should be in the higher frequency range
        assert!(rolloff >= 5.0);
    }

    // Tests for new spectral functions
    #[test]
    fn test_spectral_crest() {
        // Create a flat PSD (white noise-like)
        let psd = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        let crest = spectral_crest(&psd).unwrap();

        // Crest factor should be 1.0 for flat PSD
        assert_relative_eq!(crest, 1.0, epsilon = 1e-10);

        // Create a peaked PSD
        let psd = vec![0.1, 0.1, 0.1, 1.0, 0.1, 0.1, 0.1];

        let crest = spectral_crest(&psd).unwrap();

        // Crest factor should be high for peaked PSD
        // The exact value depends on the calculation method
        assert!(crest > 1.0);
    }

    #[test]
    fn test_spectral_decrease() {
        // Create a spectrum with decreasing amplitude
        let psd = vec![5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.2];
        let freqs = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let decrease = spectral_decrease(&psd, &freqs).unwrap();

        // Decrease should be negative for decreasing spectrum
        assert!(decrease < 0.0);

        // Create a spectrum with increasing amplitude
        let psd = vec![0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0];
        let freqs = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let decrease = spectral_decrease(&psd, &freqs).unwrap();

        // Decrease should be positive for increasing spectrum
        assert!(decrease > 0.0);
    }

    #[test]
    fn test_spectral_slope() {
        // Create a spectrum with negative slope
        let psd = vec![5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.2];
        let freqs = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let slope = spectral_slope(&psd, &freqs).unwrap();

        // Slope should be negative
        assert!(slope < 0.0);

        // Create a spectrum with positive slope
        let psd = vec![0.2, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0];
        let freqs = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let slope = spectral_slope(&psd, &freqs).unwrap();

        // Slope should be positive
        assert!(slope > 0.0);
    }

    #[test]
    fn test_spectral_contrast() {
        // Create a spectrum with clear peaks and valleys
        let psd = vec![0.1, 1.0, 0.1, 1.0, 0.1, 1.0, 0.1];
        let freqs = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let contrast = spectral_contrast(&psd, &freqs, 2).unwrap();

        // Should have correct number of bands
        assert_eq!(contrast.len(), 2);

        // All contrast values should be positive
        for &c in &contrast {
            assert!(c >= 0.0);
        }

        // Create a flat spectrum
        let psd = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let freqs = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let contrast = spectral_contrast(&psd, &freqs, 2).unwrap();

        // Contrast should be low for flat spectrum
        for &c in &contrast {
            assert!(c < 0.1);
        }
    }

    #[test]
    fn test_spectral_bandwidth() {
        // Create a peaked spectrum
        let psd = vec![0.1, 0.2, 0.5, 1.0, 0.5, 0.2, 0.1];
        let freqs = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let bandwidth_3db = spectral_bandwidth(&psd, &freqs, -3.0).unwrap();
        let bandwidth_6db = spectral_bandwidth(&psd, &freqs, -6.0).unwrap();

        // -6dB bandwidth should be wider than -3dB bandwidth
        assert!(bandwidth_6db > bandwidth_3db);

        // Bandwidths should be positive
        assert!(bandwidth_3db > 0.0);
        assert!(bandwidth_6db > 0.0);
    }

    #[test]
    fn test_dominant_frequency() {
        // Create a spectrum with a clear peak
        let psd = vec![0.1, 0.2, 0.5, 5.0, 0.5, 0.2, 0.1];
        let freqs = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let (dominant_freq, magnitude) = dominant_frequency(&psd, &freqs).unwrap();

        // Dominant frequency should be at the peak (index 3, frequency 3.0)
        assert_relative_eq!(dominant_freq, 3.0, epsilon = 1e-10);
        assert_relative_eq!(magnitude, 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_dominant_frequencies() {
        // Create a spectrum with multiple peaks
        let psd = vec![0.1, 2.0, 0.1, 5.0, 0.1, 3.0, 0.1];
        let freqs = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let peaks = dominant_frequencies(&psd, &freqs, 3, 1.0).unwrap();

        // Should find 3 peaks
        assert_eq!(peaks.len(), 3);

        // Peaks should be in descending order of magnitude
        assert!(peaks[0].1 >= peaks[1].1);
        assert!(peaks[1].1 >= peaks[2].1);

        // First peak should be at frequency 3.0 with magnitude 5.0
        assert_relative_eq!(peaks[0].0, 3.0, epsilon = 1e-10);
        assert_relative_eq!(peaks[0].1, 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_error_handling() {
        // Test with empty arrays
        assert!(energy_spectral_density(&[] as &[f64], 100.0).is_err());
        assert!(normalized_psd(&[] as &[f64]).is_err());
        assert!(spectral_centroid(&[] as &[f64], &[] as &[f64]).is_err());
        assert!(spectral_spread(&[] as &[f64], &[] as &[f64], None).is_err());
        assert!(spectral_flatness(&[] as &[f64]).is_err());
        assert!(spectral_flux(&[] as &[f64], &[] as &[f64], "l2").is_err());
        assert!(spectral_crest(&[] as &[f64]).is_err());
        assert!(spectral_decrease(&[] as &[f64], &[] as &[f64]).is_err());
        assert!(spectral_slope(&[] as &[f64], &[] as &[f64]).is_err());
        assert!(spectral_contrast(&[] as &[f64], &[] as &[f64], 2).is_err());
        assert!(spectral_bandwidth(&[] as &[f64], &[] as &[f64], -3.0).is_err());
        assert!(dominant_frequency(&[] as &[f64], &[] as &[f64]).is_err());
        assert!(dominant_frequencies(&[] as &[f64], &[] as &[f64], 3, 1.0).is_err());

        // Test with mismatched array lengths
        let psd = vec![1.0, 2.0, 3.0];
        let freqs = vec![0.0, 1.0];
        assert!(spectral_centroid(&psd, &freqs).is_err());
        assert!(spectral_decrease(&psd, &freqs).is_err());
        assert!(spectral_bandwidth(&psd, &freqs, -3.0).is_err());

        // Test with invalid parameters
        assert!(energy_spectral_density(&[1.0, 2.0], -1.0).is_err());
        assert!(spectral_rolloff(&[1.0, 2.0], &[0.0, 1.0], 2.0).is_err());
        assert!(spectral_flux(&[1.0, 2.0], &[1.0, 2.0], "invalid_norm").is_err());
        assert!(spectral_contrast(&[1.0, 2.0], &[0.0, 1.0], 0).is_err());
        assert!(dominant_frequencies(&[1.0, 2.0], &[0.0, 1.0], 0, 1.0).is_err());
        assert!(dominant_frequencies(&[1.0, 2.0], &[0.0, 1.0], 1, -1.0).is_err());
    }

    #[test]
    fn test_with_ndarray() {
        // Test functions with ndarray types
        let psd = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0]);
        let freqs = Array1::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let centroid = spectral_centroid(&psd.to_vec(), &freqs.to_vec()).unwrap();
        assert_relative_eq!(centroid, 3.0, epsilon = 1e-10);

        // Create 2D array and take a slice
        let psd_2d =
            Array2::from_shape_vec((1, 7), vec![1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0]).unwrap();
        let psd_slice = psd_2d.slice(ndarray::s![0, ..]);
        let psd_slice_vec: Vec<f64> = psd_slice.iter().cloned().collect();

        let spread = spectral_spread(&psd_slice_vec, &freqs.to_vec(), None).unwrap();
        assert!(spread > 0.0);

        // Test new functions with ndarray
        let crest = spectral_crest(&psd.to_vec()).unwrap();
        assert!(crest >= 1.0);

        let (dominant_freq, _) = dominant_frequency(&psd.to_vec(), &freqs.to_vec()).unwrap();
        assert_relative_eq!(dominant_freq, 3.0, epsilon = 1e-10);
    }
}
