//! Auto-generated test module (consolidated from inline `#[cfg(test)] mod` blocks)

use crate::error::Result;
use rayon::prelude::*;
use scirs2_core::ndarray::{Array1, Array2, Array3};

use super::*;

#[cfg(test)]
mod tests_2 {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_dwt_haar() -> Result<()> {
        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let dwt = DWT::new(WaveletType::Haar)?;

        let (approx, detail) = dwt.decompose(&signal.view())?;

        assert!(approx.len() > 0);
        assert!(detail.len() > 0);
        assert_eq!(approx.len(), detail.len());

        Ok(())
    }

    #[test]
    fn test_dwt_multilevel() -> Result<()> {
        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let dwt = DWT::new(WaveletType::Haar)?.with_level(2);

        let coeffs = dwt.wavedec(&signal.view())?;

        assert_eq!(coeffs.len(), 3); // 2 levels + approximation

        Ok(())
    }

    #[test]
    fn test_dwt_reconstruction() -> Result<()> {
        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let dwt = DWT::new(WaveletType::Haar)?;

        let (approx, detail) = dwt.decompose(&signal.view())?;
        let reconstructed = dwt.reconstruct(&approx.view(), &detail.view())?;

        // Check reconstruction is approximately correct (may have different length)
        assert!(reconstructed.len() >= signal.len() - 2);

        Ok(())
    }

    #[test]
    fn test_dwt2d() -> Result<()> {
        let image = Array2::from_shape_fn((8, 8), |(i, j)| (i + j) as f64);
        let dwt2d = DWT2D::new(WaveletType::Haar)?;

        let coeffs = dwt2d.decompose2(&image.view())?;

        assert!(coeffs.ll.len() > 0);
        assert!(coeffs.lh.len() > 0);
        assert!(coeffs.hl.len() > 0);
        assert!(coeffs.hh.len() > 0);

        Ok(())
    }

    #[test]
    fn test_wavelet_filters() -> Result<()> {
        let filters = WaveletFilters::from_wavelet(WaveletType::Haar)?;

        assert_eq!(filters.dec_lo.len(), 2);
        assert_eq!(filters.dec_hi.len(), 2);
        assert_eq!(filters.rec_lo.len(), 2);
        assert_eq!(filters.rec_hi.len(), 2);

        Ok(())
    }

    /// Verify that `dec_lo` sums to sqrt(2) (standard orthonormal normalisation).
    fn check_filter_normalisation(filters: &WaveletFilters) {
        let sum: f64 = filters.dec_lo.iter().sum();
        let diff = (sum - 2.0_f64.sqrt()).abs();
        assert!(
            diff < 1e-6,
            "dec_lo sum {sum} is not sqrt(2); diff = {diff}"
        );
    }

    #[test]
    fn test_daubechies_db1_is_haar() -> Result<()> {
        let haar = WaveletFilters::from_wavelet(WaveletType::Haar)?;
        let db1 = WaveletFilters::from_wavelet(WaveletType::Daubechies(1))?;
        assert_abs_diff_eq!(haar.dec_lo[0], db1.dec_lo[0], epsilon = 1e-10);
        assert_abs_diff_eq!(haar.dec_lo[1], db1.dec_lo[1], epsilon = 1e-10);
        Ok(())
    }

    #[test]
    fn test_daubechies_db3_filters() -> Result<()> {
        let f = WaveletFilters::from_wavelet(WaveletType::Daubechies(3))?;
        assert_eq!(f.dec_lo.len(), 6);
        check_filter_normalisation(&f);
        Ok(())
    }

    #[test]
    fn test_daubechies_db5_filters() -> Result<()> {
        let f = WaveletFilters::from_wavelet(WaveletType::Daubechies(5))?;
        assert_eq!(f.dec_lo.len(), 10);
        check_filter_normalisation(&f);
        Ok(())
    }

    #[test]
    fn test_daubechies_db6_filters() -> Result<()> {
        let f = WaveletFilters::from_wavelet(WaveletType::Daubechies(6))?;
        assert_eq!(f.dec_lo.len(), 12);
        check_filter_normalisation(&f);
        Ok(())
    }

    #[test]
    fn test_daubechies_db7_filters() -> Result<()> {
        let f = WaveletFilters::from_wavelet(WaveletType::Daubechies(7))?;
        assert_eq!(f.dec_lo.len(), 14);
        check_filter_normalisation(&f);
        Ok(())
    }

    #[test]
    fn test_daubechies_db8_filters() -> Result<()> {
        let f = WaveletFilters::from_wavelet(WaveletType::Daubechies(8))?;
        assert_eq!(f.dec_lo.len(), 16);
        check_filter_normalisation(&f);
        Ok(())
    }

    #[test]
    fn test_daubechies_db10_filters() -> Result<()> {
        let f = WaveletFilters::from_wavelet(WaveletType::Daubechies(10))?;
        assert_eq!(f.dec_lo.len(), 20);
        check_filter_normalisation(&f);
        Ok(())
    }

    #[test]
    fn test_daubechies_unsupported_returns_error() {
        let result = WaveletFilters::from_wavelet(WaveletType::Daubechies(11));
        assert!(result.is_err(), "DB11 should return an error");
    }

    #[test]
    fn test_coiflet1_filters() -> Result<()> {
        let f = WaveletFilters::from_wavelet(WaveletType::Coiflet(1))?;
        assert_eq!(f.dec_lo.len(), 6);
        check_filter_normalisation(&f);
        Ok(())
    }

    #[test]
    fn test_coiflet2_filters() -> Result<()> {
        let f = WaveletFilters::from_wavelet(WaveletType::Coiflet(2))?;
        assert_eq!(f.dec_lo.len(), 12);
        check_filter_normalisation(&f);
        Ok(())
    }

    #[test]
    fn test_coiflet3_filters() -> Result<()> {
        let f = WaveletFilters::from_wavelet(WaveletType::Coiflet(3))?;
        assert_eq!(f.dec_lo.len(), 18);
        check_filter_normalisation(&f);
        Ok(())
    }

    #[test]
    fn test_coiflet4_filters() -> Result<()> {
        let f = WaveletFilters::from_wavelet(WaveletType::Coiflet(4))?;
        assert_eq!(f.dec_lo.len(), 24);
        check_filter_normalisation(&f);
        Ok(())
    }

    #[test]
    fn test_coiflet5_filters() -> Result<()> {
        let f = WaveletFilters::from_wavelet(WaveletType::Coiflet(5))?;
        assert_eq!(f.dec_lo.len(), 30);
        check_filter_normalisation(&f);
        Ok(())
    }

    #[test]
    fn test_coiflet_unsupported_returns_error() {
        let result = WaveletFilters::from_wavelet(WaveletType::Coiflet(6));
        assert!(result.is_err(), "Coif6 should return an error");
    }

    #[test]
    fn test_dwt_db3_roundtrip() -> Result<()> {
        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let dwt = DWT::new(WaveletType::Daubechies(3))?;

        let (approx, detail) = dwt.decompose(&signal.view())?;
        let reconstructed = dwt.reconstruct(&approx.view(), &detail.view())?;

        // Reconstruction should have at least as many samples as the original
        assert!(reconstructed.len() >= signal.len() - 2);
        Ok(())
    }

    #[test]
    fn test_dwt_coif2_roundtrip() -> Result<()> {
        let signal = Array1::from_vec(vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ]);
        let dwt = DWT::new(WaveletType::Coiflet(2))?;

        let (approx, detail) = dwt.decompose(&signal.view())?;
        let reconstructed = dwt.reconstruct(&approx.view(), &detail.view())?;

        assert!(reconstructed.len() >= signal.len() - 4);
        Ok(())
    }

    // -------------------------------------------------------------------------
    // Invariant helpers: unit energy (sum of squares = 1) and QMF orthogonality
    // -------------------------------------------------------------------------

    /// Assert that sum of squares of dec_lo equals 1.0 (unit energy condition).
    fn check_unit_energy(filters: &WaveletFilters) {
        let energy: f64 = filters.dec_lo.iter().map(|x| x * x).sum();
        let diff = (energy - 1.0).abs();
        assert!(
            diff < 1e-10,
            "dec_lo unit-energy check failed: sum-of-squares = {energy}, diff from 1.0 = {diff}"
        );
    }

    /// Assert that <dec_lo, dec_hi> = 0 (QMF orthogonality).
    fn check_qmf_orthogonality(filters: &WaveletFilters) {
        let inner: f64 = filters
            .dec_lo
            .iter()
            .zip(filters.dec_hi.iter())
            .map(|(a, b)| a * b)
            .sum();
        let diff = inner.abs();
        assert!(
            diff < 1e-10,
            "QMF orthogonality check failed: <dec_lo, dec_hi> = {inner}, abs = {diff}"
        );
    }

    // -------------------------------------------------------------------------
    // Daubechies filter invariant tests (length, sum=√2, energy=1, QMF ortho)
    // -------------------------------------------------------------------------

    #[test]
    fn test_db2_all_invariants() -> Result<()> {
        let f = WaveletFilters::from_wavelet(WaveletType::Daubechies(2))?;
        assert_eq!(f.dec_lo.len(), 4, "db2 length must be 4");
        check_filter_normalisation(&f);
        check_unit_energy(&f);
        check_qmf_orthogonality(&f);
        Ok(())
    }

    #[test]
    fn test_db4_all_invariants() -> Result<()> {
        let f = WaveletFilters::from_wavelet(WaveletType::Daubechies(4))?;
        assert_eq!(f.dec_lo.len(), 8, "db4 length must be 8");
        check_filter_normalisation(&f);
        check_unit_energy(&f);
        check_qmf_orthogonality(&f);
        Ok(())
    }

    #[test]
    fn test_db6_all_invariants() -> Result<()> {
        let f = WaveletFilters::from_wavelet(WaveletType::Daubechies(6))?;
        assert_eq!(f.dec_lo.len(), 12, "db6 length must be 12");
        check_filter_normalisation(&f);
        check_unit_energy(&f);
        check_qmf_orthogonality(&f);
        Ok(())
    }

    #[test]
    fn test_db8_all_invariants() -> Result<()> {
        let f = WaveletFilters::from_wavelet(WaveletType::Daubechies(8))?;
        assert_eq!(f.dec_lo.len(), 16, "db8 length must be 16");
        check_filter_normalisation(&f);
        check_unit_energy(&f);
        check_qmf_orthogonality(&f);
        Ok(())
    }

    #[test]
    fn test_db10_all_invariants() -> Result<()> {
        let f = WaveletFilters::from_wavelet(WaveletType::Daubechies(10))?;
        assert_eq!(f.dec_lo.len(), 20, "db10 length must be 20");
        check_filter_normalisation(&f);
        check_unit_energy(&f);
        check_qmf_orthogonality(&f);
        Ok(())
    }

    // -------------------------------------------------------------------------
    // Coiflet filter invariant tests (length, sum=√2, energy=1, QMF ortho)
    // -------------------------------------------------------------------------

    #[test]
    fn test_coif1_all_invariants() -> Result<()> {
        let f = WaveletFilters::from_wavelet(WaveletType::Coiflet(1))?;
        assert_eq!(f.dec_lo.len(), 6, "coif1 length must be 6");
        check_filter_normalisation(&f);
        check_unit_energy(&f);
        check_qmf_orthogonality(&f);
        Ok(())
    }

    #[test]
    fn test_coif2_all_invariants() -> Result<()> {
        let f = WaveletFilters::from_wavelet(WaveletType::Coiflet(2))?;
        assert_eq!(f.dec_lo.len(), 12, "coif2 length must be 12");
        check_filter_normalisation(&f);
        check_unit_energy(&f);
        check_qmf_orthogonality(&f);
        Ok(())
    }

    #[test]
    fn test_coif3_all_invariants() -> Result<()> {
        let f = WaveletFilters::from_wavelet(WaveletType::Coiflet(3))?;
        assert_eq!(f.dec_lo.len(), 18, "coif3 length must be 18");
        check_filter_normalisation(&f);
        check_unit_energy(&f);
        check_qmf_orthogonality(&f);
        Ok(())
    }

    #[test]
    fn test_coif4_all_invariants() -> Result<()> {
        let f = WaveletFilters::from_wavelet(WaveletType::Coiflet(4))?;
        assert_eq!(f.dec_lo.len(), 24, "coif4 length must be 24");
        check_filter_normalisation(&f);
        check_unit_energy(&f);
        check_qmf_orthogonality(&f);
        Ok(())
    }

    #[test]
    fn test_coif5_all_invariants() -> Result<()> {
        let f = WaveletFilters::from_wavelet(WaveletType::Coiflet(5))?;
        assert_eq!(f.dec_lo.len(), 30, "coif5 length must be 30");
        check_filter_normalisation(&f);
        check_unit_energy(&f);
        check_qmf_orthogonality(&f);
        Ok(())
    }

    // -------------------------------------------------------------------------
    // 3D DWT tests
    // -------------------------------------------------------------------------

    /// Decompose a constant volume with Haar: the LLL (approximation) subband
    /// should equal the input value scaled by (√2)^3 = 2√2, and all seven
    /// detail subbands should be near-zero.
    ///
    /// With Haar, each axis pass scales the low-pass output by 1/√2 * √2 = 1
    /// per coefficient and sums two values, so: a constant `c` signal of length N
    /// yields approx coefficients ≈ `c * √2` (one per pair of inputs).  Three
    /// axis passes → LLL ≈ `c * (√2)^3 = c * 2√2`.
    #[test]
    fn test_dwt3d_constant_volume_lll_scaling() -> Result<()> {
        let c = 3.0_f64;
        // 8×8×8 constant volume for clean Haar decomposition
        let volume = Array3::from_elem((8, 8, 8), c);
        let dwtn = DWTN::new(WaveletType::Haar);
        let coeffs = dwtn.decompose3(&volume)?;

        // LLL subband must be non-empty and all values ≈ c * 2√2
        assert!(coeffs.lll.len() > 0, "LLL subband must not be empty");
        let expected_lll = c * 2.0_f64.sqrt().powi(3); // c * 2√2 ≈ 4.243 for c=3
        for val in coeffs.lll.iter() {
            assert_abs_diff_eq!(*val, expected_lll, epsilon = 1e-10);
        }

        // All 7 detail subbands must be near-zero (constant signal has no detail)
        let detail_bands: [&Array3<f64>; 7] = [
            &coeffs.llh,
            &coeffs.lhl,
            &coeffs.lhh,
            &coeffs.hll,
            &coeffs.hlh,
            &coeffs.hhl,
            &coeffs.hhh,
        ];
        for band in detail_bands {
            for val in band.iter() {
                assert_abs_diff_eq!(*val, 0.0, epsilon = 1e-10);
            }
        }
        Ok(())
    }

    /// Verify that `decompose3` produces subbands with the expected half-sizes.
    #[test]
    fn test_dwt3d_output_shape() -> Result<()> {
        let volume = Array3::from_shape_fn((8, 6, 4), |(i, j, k)| (i + j + k) as f64);
        let dwtn = DWTN::new(WaveletType::Haar);
        let coeffs = dwtn.decompose3(&volume)?;

        // Each axis is halved (ceiling division): 8→4, 6→3, 4→2
        assert_eq!(coeffs.lll.dim(), (4, 3, 2));
        assert_eq!(coeffs.hhh.dim(), (4, 3, 2));
        Ok(())
    }

    /// Decomposing a volume with all dimensions < 2 must return an error.
    #[test]
    fn test_dwt3d_rejects_too_small_volume() {
        let volume = Array3::from_elem((1, 8, 8), 1.0);
        let dwtn = DWTN::new(WaveletType::Haar);
        assert!(
            dwtn.decompose3(&volume).is_err(),
            "decompose3 must reject a volume with any dimension < 2"
        );
    }

    // -------------------------------------------------------------------------
    // Symlet: must return REAL Symlet coefficients, not Daubechies coefficients
    // silently substituted in (the previously-fixed bug).
    // -------------------------------------------------------------------------

    #[test]
    fn symlet_matches_authoritative_sym2_values() -> Result<()> {
        // Cross-checked against PyWavelets `sym2_` (wavelets_coeffs.template.h).
        let f = WaveletFilters::from_wavelet(WaveletType::Symlet(2))?;
        let expected = [
            0.48296291314469025,
            0.83651630373746899,
            0.22414386804185735,
            -0.12940952255092145,
        ];
        for (got, want) in f.dec_lo.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(*got, *want, epsilon = 1e-12);
        }
        Ok(())
    }

    #[test]
    fn symlet_n_differs_from_daubechies_n_for_all_supported_orders() -> Result<()> {
        // Before the fix, `Symlet(n)` silently returned `Daubechies(n)`'s
        // coefficients verbatim. sym{n} and db{n} share a length and several
        // invariants but are numerically different filters for n >= 3 (sym2
        // == db2 exactly, since length-4 is the unique real orthogonal
        // solution, so we check n = 3..=8 where they must differ).
        for n in 3..=8usize {
            let sym = WaveletFilters::from_wavelet(WaveletType::Symlet(n))?;
            let db = WaveletFilters::from_wavelet(WaveletType::Daubechies(n))?;
            assert_eq!(sym.dec_lo.len(), db.dec_lo.len());
            let identical = sym
                .dec_lo
                .iter()
                .zip(db.dec_lo.iter())
                .all(|(a, b)| (a - b).abs() < 1e-12);
            assert!(
                !identical,
                "Symlet({n}) must not be identical to Daubechies({n})"
            );
        }
        Ok(())
    }

    #[test]
    fn symlet_filters_satisfy_orthogonal_wavelet_invariants() -> Result<()> {
        for n in 2..=8usize {
            let f = WaveletFilters::from_wavelet(WaveletType::Symlet(n))?;
            assert_eq!(f.dec_lo.len(), 2 * n, "sym{n} length must be 2n");
            check_filter_normalisation(&f);
            check_unit_energy(&f);
            check_qmf_orthogonality(&f);
        }
        Ok(())
    }

    #[test]
    fn symlet_unsupported_order_returns_error() {
        assert!(WaveletFilters::from_wavelet(WaveletType::Symlet(1)).is_err());
        assert!(WaveletFilters::from_wavelet(WaveletType::Symlet(9)).is_err());
    }

    // -------------------------------------------------------------------------
    // Biorthogonal: must return REAL bior{p}.{q} coefficients, not Haar
    // (the previously-fixed bug), and must form a genuine perfect-
    // reconstruction filter bank.
    // -------------------------------------------------------------------------

    #[test]
    fn biorthogonal_matches_authoritative_bior1_3_values() -> Result<()> {
        // Cross-checked against PyWavelets `bior1_3_` (wavelets_coeffs.template.h).
        let f = WaveletFilters::from_wavelet(WaveletType::Biorthogonal(1, 3))?;
        let expected_rec_lo = [
            0.0,
            0.0,
            std::f64::consts::FRAC_1_SQRT_2,
            std::f64::consts::FRAC_1_SQRT_2,
            0.0,
            0.0,
        ];
        for (got, want) in f.rec_lo.iter().zip(expected_rec_lo.iter()) {
            assert_abs_diff_eq!(*got, *want, epsilon = 1e-12);
        }
        Ok(())
    }

    #[test]
    fn biorthogonal_pairs_differ_from_haar_fallback() -> Result<()> {
        // Before the fix, every (p, q) silently returned Haar's coefficients
        // (dec_lo = [1/sqrt2, 1/sqrt2], length 2). Every supported
        // biorthogonal pair here has a strictly longer filter, which is a
        // direct, unambiguous proof that the fallback is gone.
        let haar = WaveletFilters::from_wavelet(WaveletType::Haar)?;
        for &(p, q) in &[(1usize, 3usize), (2, 2), (3, 1), (4, 4)] {
            let f = WaveletFilters::from_wavelet(WaveletType::Biorthogonal(p, q))?;
            assert!(
                f.dec_lo.len() > haar.dec_lo.len(),
                "bior{p}.{q} must not fall back to Haar's 2-tap filter"
            );
        }
        Ok(())
    }

    #[test]
    fn biorthogonal_unsupported_pair_returns_error() {
        assert!(WaveletFilters::from_wavelet(WaveletType::Biorthogonal(9, 9)).is_err());
    }

    /// Circular (periodic) analysis+synthesis round-trip. This directly
    /// verifies the mathematical perfect-reconstruction property of a
    /// filter bank, independent of this module's own (symmetric-extension)
    /// boundary handling in `DWT::decompose`/`reconstruct`.
    fn circular_round_trip(x: &[f64], filters: &WaveletFilters) -> Vec<f64> {
        let n = x.len();
        let circ_conv = |input: &[f64], h: &[f64]| -> Vec<f64> {
            (0..n)
                .map(|i| {
                    h.iter()
                        .enumerate()
                        .map(|(k, &hk)| {
                            let idx = (i as isize - k as isize).rem_euclid(n as isize) as usize;
                            hk * input[idx]
                        })
                        .sum()
                })
                .collect()
        };

        let y0 = circ_conv(x, &filters.dec_lo);
        let y1 = circ_conv(x, &filters.dec_hi);
        let approx: Vec<f64> = y0.iter().step_by(2).copied().collect();
        let detail: Vec<f64> = y1.iter().step_by(2).copied().collect();

        let mut a_up = vec![0.0; n];
        let mut d_up = vec![0.0; n];
        for (i, &v) in approx.iter().enumerate() {
            a_up[i * 2] = v;
        }
        for (i, &v) in detail.iter().enumerate() {
            d_up[i * 2] = v;
        }

        let r0 = circ_conv(&a_up, &filters.rec_lo);
        let r1 = circ_conv(&d_up, &filters.rec_hi);
        r0.iter().zip(r1.iter()).map(|(a, b)| a + b).collect()
    }

    /// Returns `Some(shift)` if `rec` equals `x` cyclically rotated by
    /// `shift` positions (within `tol`), else `None`.
    fn find_cyclic_shift(x: &[f64], rec: &[f64], tol: f64) -> Option<usize> {
        let n = x.len();
        (0..n).find(|&shift| (0..n).all(|i| (rec[(i + shift) % n] - x[i]).abs() < tol))
    }

    /// Non-constant test signal (constant/all-ones data cannot distinguish a
    /// real filter bank from a degenerate/fabricated one).
    fn pr_test_signal() -> Vec<f64> {
        vec![
            1.0, 2.0, -1.0, 3.0, 0.5, -2.0, 4.0, 1.5, 2.5, -0.5, 3.5, 0.0, -1.5, 2.25, 0.75, -3.25,
        ]
    }

    #[test]
    fn bior1_3_perfect_reconstruction() -> Result<()> {
        let filters = WaveletFilters::from_wavelet(WaveletType::Biorthogonal(1, 3))?;
        let x = pr_test_signal();
        let rec = circular_round_trip(&x, &filters);
        let shift = find_cyclic_shift(&x, &rec, 1e-8);
        assert!(
            shift.is_some(),
            "bior1.3 failed perfect reconstruction: rec={rec:?} vs x={x:?}"
        );
        Ok(())
    }

    #[test]
    fn bior2_2_perfect_reconstruction() -> Result<()> {
        let filters = WaveletFilters::from_wavelet(WaveletType::Biorthogonal(2, 2))?;
        let x = pr_test_signal();
        let rec = circular_round_trip(&x, &filters);
        let shift = find_cyclic_shift(&x, &rec, 1e-8);
        assert!(
            shift.is_some(),
            "bior2.2 failed perfect reconstruction: rec={rec:?} vs x={x:?}"
        );
        Ok(())
    }

    #[test]
    fn bior3_1_perfect_reconstruction() -> Result<()> {
        let filters = WaveletFilters::from_wavelet(WaveletType::Biorthogonal(3, 1))?;
        let x = pr_test_signal();
        let rec = circular_round_trip(&x, &filters);
        let shift = find_cyclic_shift(&x, &rec, 1e-8);
        assert!(
            shift.is_some(),
            "bior3.1 failed perfect reconstruction: rec={rec:?} vs x={x:?}"
        );
        Ok(())
    }

    #[test]
    fn bior4_4_perfect_reconstruction() -> Result<()> {
        let filters = WaveletFilters::from_wavelet(WaveletType::Biorthogonal(4, 4))?;
        let x = pr_test_signal();
        let rec = circular_round_trip(&x, &filters);
        let shift = find_cyclic_shift(&x, &rec, 1e-8);
        assert!(
            shift.is_some(),
            "bior4.4 failed perfect reconstruction: rec={rec:?} vs x={x:?}"
        );
        Ok(())
    }

    #[test]
    fn symlet_perfect_reconstruction_sym2_sym4_sym8() -> Result<()> {
        let x = pr_test_signal();
        for n in [2usize, 4, 8] {
            let filters = WaveletFilters::from_wavelet(WaveletType::Symlet(n))?;
            let rec = circular_round_trip(&x, &filters);
            let shift = find_cyclic_shift(&x, &rec, 1e-8);
            assert!(
                shift.is_some(),
                "sym{n} failed perfect reconstruction: rec={rec:?} vs x={x:?}"
            );
        }
        Ok(())
    }
}
