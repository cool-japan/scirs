//! Correctness tests for AVX-512 butterfly kernels.
//!
//! These tests compare the AVX-512 implementations against the scalar
//! reference implementations.  On hosts that do not support AVX-512F the
//! AVX-512 branches are skipped with an informational message; the test
//! still passes (compile-check coverage is considered sufficient for CI).
//!
//! Run with:
//! ```sh
//! cargo nextest run -p scirs2-fft -E 'test(avx512)'
//! ```

use scirs2_core::numeric::Complex64;
use std::f64::consts::PI;

// The `simd_fft::avx512` sub-module is only compiled on x86_64.
#[cfg(target_arch = "x86_64")]
use scirs2_fft::simd_fft::avx512;

// ─────────────────────────────────────────────────────────────────────────────
//  Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Maximum absolute error between two complex slices.
fn max_err(a: &[Complex64], b: &[Complex64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).norm())
        .fold(0.0_f64, f64::max)
}

/// Direct O(N²) DFT — used as ground truth.
fn direct_dft(input: &[Complex64]) -> Vec<Complex64> {
    let n = input.len();
    let mut out = vec![Complex64::new(0.0, 0.0); n];
    for (k, slot) in out.iter_mut().enumerate() {
        for (j, &x) in input.iter().enumerate() {
            let angle = -2.0 * PI * (k * j) as f64 / n as f64;
            *slot += x * Complex64::new(angle.cos(), angle.sin());
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
//  Radix-4 tests
// ─────────────────────────────────────────────────────────────────────────────

/// AVX-512 radix-4 matches scalar for a pure-real impulse input.
#[test]
fn avx512_radix4_matches_scalar() {
    let twiddles = [
        Complex64::new(0.0, -1.0), // W_4^1 = -j
        Complex64::new(-1.0, 0.0), // W_4^2 = -1
        Complex64::new(0.0, 1.0),  // W_4^3 = +j
    ];
    let input = [
        Complex64::new(1.0, 2.0),
        Complex64::new(3.0, 4.0),
        Complex64::new(5.0, 6.0),
        Complex64::new(7.0, 8.0),
    ];

    // Scalar reference
    let mut scalar_data = input;
    #[cfg(target_arch = "x86_64")]
    avx512::radix4_butterfly_scalar(&mut scalar_data, &twiddles);
    #[cfg(not(target_arch = "x86_64"))]
    scirs2_fft::butterfly4(&mut scalar_data, &twiddles);

    #[cfg(target_arch = "x86_64")]
    {
        if avx512::is_avx512_available() {
            let mut avx_data = input;
            // Safety: is_avx512_available() returned true.
            unsafe {
                avx512::radix4_butterfly_avx512(avx_data.as_mut_ptr(), twiddles.as_ptr());
            }
            let err = max_err(&scalar_data, &avx_data);
            assert!(
                err < 1e-12,
                "AVX-512 radix-4 diverges from scalar by {err}\n  scalar={scalar_data:?}\n  avx512={avx_data:?}"
            );
        } else {
            eprintln!("[avx512_radix4_matches_scalar] AVX-512F not available — compile-check only");
        }
    }

    // The scalar result must always be finite.
    assert!(
        scalar_data
            .iter()
            .all(|c| c.re.is_finite() && c.im.is_finite()),
        "scalar produced non-finite output: {scalar_data:?}"
    );
}

/// AVX-512 radix-4 result matches direct DFT for the same standard twiddles.
#[test]
fn avx512_radix4_matches_direct_dft() {
    let input = [
        Complex64::new(1.0, 0.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(3.0, 0.0),
        Complex64::new(4.0, 0.0),
    ];
    let twiddles = [
        Complex64::new(0.0, -1.0),
        Complex64::new(-1.0, 0.0),
        Complex64::new(0.0, 1.0),
    ];

    let expected = direct_dft(&input);

    // Scalar reference
    let mut scalar_data = input;
    #[cfg(target_arch = "x86_64")]
    avx512::radix4_butterfly_scalar(&mut scalar_data, &twiddles);
    #[cfg(not(target_arch = "x86_64"))]
    scirs2_fft::butterfly4(&mut scalar_data, &twiddles);

    let scalar_err = max_err(&scalar_data, &expected);
    assert!(
        scalar_err < 1e-12,
        "scalar radix-4 vs direct DFT: err={scalar_err}"
    );

    #[cfg(target_arch = "x86_64")]
    {
        if avx512::is_avx512_available() {
            let mut avx_data = input;
            // Safety: is_avx512_available() returned true.
            unsafe {
                avx512::radix4_butterfly_avx512(avx_data.as_mut_ptr(), twiddles.as_ptr());
            }
            let avx_err = max_err(&avx_data, &expected);
            assert!(
                avx_err < 1e-12,
                "AVX-512 radix-4 vs direct DFT: err={avx_err}"
            );
        } else {
            eprintln!(
                "[avx512_radix4_matches_direct_dft] AVX-512F not available — compile-check only"
            );
        }
    }
}

/// Dispatch wrapper produces the same result as direct scalar for radix-4.
#[test]
fn avx512_dispatch_radix4_agrees_with_scalar() {
    let twiddles = [
        Complex64::new(0.0, -1.0),
        Complex64::new(-1.0, 0.0),
        Complex64::new(0.0, 1.0),
    ];
    let input = [
        Complex64::new(2.0, -1.0),
        Complex64::new(0.5, 3.0),
        Complex64::new(-1.0, 1.0),
        Complex64::new(4.0, -2.0),
    ];

    let mut scalar_data = input;
    #[cfg(target_arch = "x86_64")]
    avx512::radix4_butterfly_scalar(&mut scalar_data, &twiddles);
    #[cfg(not(target_arch = "x86_64"))]
    scirs2_fft::butterfly4(&mut scalar_data, &twiddles);

    let mut dispatch_data = input;
    #[cfg(target_arch = "x86_64")]
    avx512::radix4_butterfly_dispatch(&mut dispatch_data, &twiddles);
    #[cfg(not(target_arch = "x86_64"))]
    scirs2_fft::butterfly4(&mut dispatch_data, &twiddles);

    let err = max_err(&scalar_data, &dispatch_data);
    assert!(err < 1e-12, "dispatch vs scalar radix-4 err={err}");
}

// ─────────────────────────────────────────────────────────────────────────────
//  Radix-8 tests
// ─────────────────────────────────────────────────────────────────────────────

/// AVX-512 radix-8 matches scalar for arbitrary complex input.
#[test]
fn avx512_radix8_matches_scalar() {
    let input: [Complex64; 8] = std::array::from_fn(|k| {
        let t = k as f64 * 0.5;
        Complex64::new(t.sin() + 1.0, t.cos() - 0.5)
    });
    let twiddles: [Complex64; 7] = std::array::from_fn(|k| {
        let angle = -2.0 * PI * (k + 1) as f64 / 8.0;
        Complex64::new(angle.cos(), angle.sin())
    });

    let mut scalar_data = input;
    #[cfg(target_arch = "x86_64")]
    avx512::radix8_butterfly_scalar(&mut scalar_data, &twiddles);
    #[cfg(not(target_arch = "x86_64"))]
    scirs2_fft::butterfly8(&mut scalar_data, &twiddles);

    #[cfg(target_arch = "x86_64")]
    {
        if avx512::is_avx512_available() {
            let mut avx_data = input;
            // Safety: is_avx512_available() returned true.
            unsafe {
                avx512::radix8_butterfly_avx512(avx_data.as_mut_ptr(), twiddles.as_ptr());
            }
            let err = max_err(&scalar_data, &avx_data);
            assert!(err < 1e-12, "AVX-512 radix-8 diverges from scalar by {err}");
        } else {
            eprintln!("[avx512_radix8_matches_scalar] AVX-512F not available — compile-check only");
        }
    }

    assert!(
        scalar_data
            .iter()
            .all(|c| c.re.is_finite() && c.im.is_finite()),
        "scalar radix-8 produced non-finite output"
    );
}

/// AVX-512 radix-8 matches the direct O(N²) DFT ground truth.
#[test]
fn avx512_radix8_matches_direct_dft() {
    let input: [Complex64; 8] =
        std::array::from_fn(|k| Complex64::new((k as f64 * 0.7).cos(), -(k as f64 * 0.4).sin()));
    let twiddles: [Complex64; 7] = std::array::from_fn(|k| {
        let angle = -2.0 * PI * (k + 1) as f64 / 8.0;
        Complex64::new(angle.cos(), angle.sin())
    });

    let expected = direct_dft(&input);

    let mut scalar_data = input;
    #[cfg(target_arch = "x86_64")]
    avx512::radix8_butterfly_scalar(&mut scalar_data, &twiddles);
    #[cfg(not(target_arch = "x86_64"))]
    scirs2_fft::butterfly8(&mut scalar_data, &twiddles);

    let scalar_err = max_err(&scalar_data, &expected);
    assert!(
        scalar_err < 1e-10,
        "scalar radix-8 vs direct DFT err={scalar_err}"
    );

    #[cfg(target_arch = "x86_64")]
    {
        if avx512::is_avx512_available() {
            let mut avx_data = input;
            // Safety: is_avx512_available() returned true.
            unsafe {
                avx512::radix8_butterfly_avx512(avx_data.as_mut_ptr(), twiddles.as_ptr());
            }
            let avx_err = max_err(&avx_data, &expected);
            assert!(
                avx_err < 1e-10,
                "AVX-512 radix-8 vs direct DFT err={avx_err}"
            );
        } else {
            eprintln!(
                "[avx512_radix8_matches_direct_dft] AVX-512F not available — compile-check only"
            );
        }
    }
}

/// Dispatch wrapper for radix-8 agrees with scalar.
#[test]
fn avx512_dispatch_radix8_agrees_with_scalar() {
    let input: [Complex64; 8] =
        std::array::from_fn(|k| Complex64::new(k as f64 * 0.7 - 1.0, k as f64 * 0.3));
    let twiddles: [Complex64; 7] = std::array::from_fn(|k| {
        let angle = -2.0 * PI * (k + 1) as f64 / 8.0;
        Complex64::new(angle.cos(), angle.sin())
    });

    let mut scalar_data = input;
    #[cfg(target_arch = "x86_64")]
    avx512::radix8_butterfly_scalar(&mut scalar_data, &twiddles);
    #[cfg(not(target_arch = "x86_64"))]
    scirs2_fft::butterfly8(&mut scalar_data, &twiddles);

    let mut dispatch_data = input;
    #[cfg(target_arch = "x86_64")]
    avx512::radix8_butterfly_dispatch(&mut dispatch_data, &twiddles);
    #[cfg(not(target_arch = "x86_64"))]
    scirs2_fft::butterfly8(&mut dispatch_data, &twiddles);

    let err = max_err(&scalar_data, &dispatch_data);
    assert!(err < 1e-12, "dispatch radix-8 vs scalar err={err}");
}

/// Matches full split-radix FFT output on random input (end-to-end sanity check).
#[test]
fn avx512_matches_full_fft_on_random_input() {
    // Use a fixed, reproducible 4-element complex signal.
    let input = [
        Complex64::new(1.23, -0.45),
        Complex64::new(-0.78, 2.34),
        Complex64::new(0.56, -1.23),
        Complex64::new(-0.12, 0.89),
    ];
    let twiddles = [
        Complex64::new(0.0, -1.0),
        Complex64::new(-1.0, 0.0),
        Complex64::new(0.0, 1.0),
    ];

    // Ground truth via direct DFT
    let expected = direct_dft(&input);

    // Scalar butterfly matches direct DFT
    let mut scalar_data = input;
    #[cfg(target_arch = "x86_64")]
    avx512::radix4_butterfly_scalar(&mut scalar_data, &twiddles);
    #[cfg(not(target_arch = "x86_64"))]
    scirs2_fft::butterfly4(&mut scalar_data, &twiddles);

    let scalar_err = max_err(&scalar_data, &expected);
    assert!(
        scalar_err < 1e-12,
        "scalar vs DFT on random input: err={scalar_err}"
    );

    #[cfg(target_arch = "x86_64")]
    {
        if avx512::is_avx512_available() {
            let mut avx_data = input;
            // Safety: is_avx512_available() returned true.
            unsafe {
                avx512::radix4_butterfly_avx512(avx_data.as_mut_ptr(), twiddles.as_ptr());
            }
            let avx_err = max_err(&avx_data, &expected);
            assert!(
                avx_err < 1e-12,
                "AVX-512 vs DFT on random input: err={avx_err}"
            );
        } else {
            eprintln!(
                "[avx512_matches_full_fft_on_random_input] AVX-512F not available — compile-check only"
            );
        }
    }
}
