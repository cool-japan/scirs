//! Time series dimensionality reduction methods
//!
//! This module provides various dimensionality reduction techniques specifically
//! designed for time series data, including PCA, Functional PCA, Dynamic Time Warping
//! barycenter averaging, and symbolic approximation methods.
//!
//! # Key Features
//!
//! - **Principal Component Analysis (PCA)**: Traditional PCA adapted for time series
//! - **Functional PCA**: PCA for functional time series data
//! - **Dynamic Time Warping (DTW) Barycenter**: Averaging for irregular time series
//! - **Symbolic Approximation**: Discrete representation methods
//! - **Adaptive Methods**: Data-driven dimension selection
//! - **Cross-validation**: Model selection and validation
//!
//! # Example
//!
//! ```rust
//! use scirs2_core::ndarray::Array2;
//! use scirs2_series::dimensionality_reduction::{PCAConfig, apply_pca};
//!
//! // Create sample time series data matrix (n_series × n_timepoints)
//! let data = Array2::from_shape_vec((5, 100), (0..500).map(|x| x as f64).collect()).expect("Operation failed");
//!
//! // Configure PCA
//! let config = PCAConfig {
//!     n_components: Some(3),
//!     center_data: true,
//!     scale_data: true,
//!     ..Default::default()
//! };
//!
//! // Apply PCA transformation
//! let result = apply_pca(&data, &config).expect("Operation failed");
//! println!("Explained variance ratio: {:?}", result.explained_variance_ratio);
//! ```

mod dtw;
mod functional_pca;
mod pca;
mod symbolic;

// Re-export PCA types and functions
pub use pca::{apply_pca, PCAConfig, PCAResult};

// Re-export Functional PCA types and functions
pub use functional_pca::{
    apply_functional_pca, BasisType, FunctionalPCA, FunctionalPCAConfig, FunctionalPCAResult,
};

// Re-export DTW types and functions
pub use dtw::{
    compute_dtw_barycenter, BarycenterInit, DTWBarycenterConfig, DTWBarycenterResult, DTWDistance,
};

// Re-export Symbolic approximation types and functions
pub use symbolic::{
    apply_symbolic_approximation, compute_reconstruction_error, reconstruct_from_sax,
    SymbolicApproximationConfig, SymbolicApproximationResult, SymbolicDistance, SymbolicMethod,
};

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{Array1, Array2};

    #[test]
    fn test_pca_basic() {
        let data = Array2::from_shape_vec((10, 5), (0..50).map(|x| x as f64).collect())
            .expect("Operation failed");
        let config = PCAConfig::default();

        let result = apply_pca(&data, &config).expect("Operation failed");

        assert_eq!(result.transformed_data.nrows(), 10);
        assert!(result.n_components_selected > 0);
        assert!(!result.explained_variance.is_empty());
    }

    #[test]
    fn test_pca_configuration() {
        let data = Array2::from_shape_vec((20, 10), (0..200).map(|x| x as f64).collect())
            .expect("Operation failed");

        let config = PCAConfig {
            n_components: Some(3),
            center_data: true,
            scale_data: true,
            ..Default::default()
        };

        let result = apply_pca(&data, &config).expect("Operation failed");

        assert_eq!(result.n_components_selected, 3);
        assert_eq!(result.transformed_data.ncols(), 3);
        assert_eq!(result.components.ncols(), 3);
    }

    #[test]
    fn test_functional_pca_basic() {
        let functional_data =
            Array2::from_shape_vec((5, 20), (0..100).map(|x| (x as f64 * 0.1).sin()).collect())
                .expect("Operation failed");

        let config = FunctionalPCAConfig::default();
        let result = apply_functional_pca(&functional_data, &config).expect("Operation failed");

        assert!(result.functional_components.nrows() > 0);
        assert!(!result.explained_variance.is_empty());
    }

    #[test]
    fn test_dtw_barycenter_basic() {
        let ts1 = Array1::from_vec(vec![1.0, 2.0, 3.0, 2.0, 1.0]);
        let ts2 = Array1::from_vec(vec![0.5, 1.5, 2.5, 1.5, 0.5]);
        let _timeseries = vec![ts1, ts2];

        let config = DTWBarycenterConfig::default();
        let result = compute_dtw_barycenter(&_timeseries, &config).expect("Operation failed");

        assert!(!result.barycenter.is_empty());
        assert_eq!(result.distances.len(), 2);
        assert!(result.iterations > 0);
    }

    #[test]
    fn test_symbolic_approximation_sax() {
        let _timeseries = Array1::from_shape_fn(100, |i| (i as f64 * 0.1).sin());
        let config = SymbolicApproximationConfig::default();

        let result = apply_symbolic_approximation(&_timeseries, &config).expect("Operation failed");

        assert!(!result.symbolic_sequence.is_empty());
        assert!(result.compression_ratio > 1.0);
    }

    #[test]
    fn test_pca_edge_cases() {
        // Test with minimal data
        let data =
            Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).expect("Operation failed");
        let config = PCAConfig::default();

        let result = apply_pca(&data, &config).expect("Operation failed");
        assert!(result.n_components_selected <= 2);
    }

    #[test]
    fn test_dtw_single_series() {
        let ts = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let _timeseries = vec![ts];

        let config = DTWBarycenterConfig::default();
        let result = compute_dtw_barycenter(&_timeseries, &config).expect("Operation failed");

        assert_eq!(result.barycenter.len(), 3);
        assert_eq!(result.distances.len(), 1);
    }

    // -----------------------------------------------------------------------
    // Tests for newly-implemented APCA, PLA, Persist, and SAX reconstruction
    // -----------------------------------------------------------------------

    #[test]
    fn test_symbolic_approximation_apca() {
        // Sinusoidal signal — APCA should compress and produce correct segment count
        let ts = Array1::from_shape_fn(100, |i| (i as f64 * 0.2).sin());
        let config = SymbolicApproximationConfig {
            method: SymbolicMethod::APCA,
            nsegments: 10,
            ..Default::default()
        };

        let result = apply_symbolic_approximation(&ts, &config)
            .expect("APCA symbolic approximation should succeed");

        assert_eq!(
            result.symbolic_sequence.len(),
            10,
            "APCA: expected 10 symbols"
        );
        assert!(
            result.compression_ratio > 1.0,
            "APCA: compression ratio should be > 1"
        );
        assert!(
            result.reconstruction_error >= 0.0,
            "APCA: error should be non-negative"
        );

        // All symbols should be valid alphabet characters
        for &sym in &result.symbolic_sequence {
            assert!(
                sym.is_alphabetic(),
                "APCA: symbol should be alphabetic, got '{sym}'"
            );
        }
    }

    #[test]
    fn test_symbolic_approximation_apca_constant() {
        // Constant series: all segments should have the same representative value
        let ts = Array1::from_elem(50, 3.0_f64);
        let config = SymbolicApproximationConfig {
            method: SymbolicMethod::APCA,
            nsegments: 5,
            normalize_data: false,
            ..Default::default()
        };

        let result = apply_symbolic_approximation(&ts, &config)
            .expect("APCA on constant series should succeed");

        // Reconstruction error should be ~0 for a constant series
        assert!(
            result.reconstruction_error < 1e-10,
            "APCA: reconstruction error on constant series should be ~0"
        );
    }

    #[test]
    fn test_symbolic_approximation_pla() {
        // Linear ramp: PLA should be near-perfect
        let ts = Array1::from_shape_fn(60, |i| i as f64);
        let config = SymbolicApproximationConfig {
            method: SymbolicMethod::PLA,
            nsegments: 6,
            normalize_data: false,
            ..Default::default()
        };

        let result = apply_symbolic_approximation(&ts, &config)
            .expect("PLA symbolic approximation should succeed");

        assert_eq!(result.symbolic_sequence.len(), 6, "PLA: expected 6 symbols");
        assert!(
            result.compression_ratio > 1.0,
            "PLA: compression ratio should be > 1"
        );

        // For a perfect linear ramp, per-segment error should be small
        assert!(
            result.reconstruction_error < 10.0,
            "PLA: reconstruction error on linear ramp should be small"
        );
    }

    #[test]
    fn test_symbolic_approximation_persist() {
        // Monotone increasing: all Persist symbols should be 'c' (up) after first
        let ts = Array1::from_shape_fn(50, |i| i as f64);
        let config = SymbolicApproximationConfig {
            method: SymbolicMethod::Persist,
            nsegments: 10,
            normalize_data: false,
            ..Default::default()
        };

        let result = apply_symbolic_approximation(&ts, &config)
            .expect("Persist symbolic approximation should succeed");

        assert_eq!(
            result.symbolic_sequence.len(),
            10,
            "Persist: expected 10 symbols"
        );

        // First symbol is 'b' (no predecessor), rest should be 'c' (up) for a rising ramp
        assert_eq!(
            result.symbolic_sequence[0], 'b',
            "Persist: first symbol should be 'b'"
        );
        for &sym in &result.symbolic_sequence[1..] {
            assert_eq!(
                sym, 'c',
                "Persist: rising ramp should yield all 'c' after first"
            );
        }
    }

    #[test]
    fn test_sax_reconstruction_roundtrip() {
        // Verify that reconstructed series has correct length and contains finite values
        let ts = Array1::from_shape_fn(100, |i| (i as f64 * 0.1).sin());
        let config = SymbolicApproximationConfig::default();

        // Step 1: SAX encode
        let sax_result =
            apply_symbolic_approximation(&ts, &config).expect("SAX encoding should succeed");

        // Step 2: Reconstruct
        let reconstructed = reconstruct_from_sax(
            &sax_result.symbolic_sequence,
            &sax_result.breakpoints,
            ts.len(),
            config.nsegments,
        )
        .expect("SAX reconstruction should succeed");

        assert_eq!(
            reconstructed.len(),
            ts.len(),
            "Reconstructed length should match original"
        );
        assert!(
            reconstructed.iter().all(|v| v.is_finite()),
            "All reconstructed values should be finite"
        );

        // Reconstruction error computed via compute_reconstruction_error should be finite
        let err = compute_reconstruction_error(&ts, &reconstructed);
        assert!(err.is_finite(), "Reconstruction error should be finite");
        assert!(err >= 0.0, "Reconstruction error should be non-negative");
    }

    #[test]
    fn test_sax_reconstruction_constant() {
        // Constant series reconstructed from SAX should also be approximately constant
        let ts = Array1::from_elem(40, 0.0_f64);
        let config = SymbolicApproximationConfig {
            nsegments: 8,
            normalize_data: false,
            ..Default::default()
        };

        let sax =
            apply_symbolic_approximation(&ts, &config).expect("SAX on constant should succeed");
        let recon = reconstruct_from_sax(&sax.symbolic_sequence, &sax.breakpoints, ts.len(), 8)
            .expect("SAX reconstruction on constant should succeed");

        assert_eq!(recon.len(), ts.len());
        // Reconstructed values should be finite
        assert!(recon.iter().all(|v| v.is_finite()));
    }
}
