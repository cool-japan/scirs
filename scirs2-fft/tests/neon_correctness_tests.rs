//! Correctness tests for ARM NEON butterfly kernels.
//!
//! These tests compare the NEON implementations against the scalar
//! reference implementations and direct O(N²) DFT ground truth.
//!
//! On AArch64 hosts (Apple Silicon, AWS Graviton, etc.) NEON is always
//! available and all tests execute the NEON path.  On non-AArch64 hosts
//! the NEON-specific assertions are skipped at compile time (the scalar
//! assertions still run for regression coverage).
//!
//! Run with:
//! ```sh
//! cargo nextest run -p scirs2-fft -E 'test(neon) | test(sve)'
//! ```

use scirs2_core::numeric::Complex64;
use std::f64::consts::PI;

// The `simd_fft::neon` sub-module is only compiled on AArch64.
#[cfg(target_arch = "aarch64")]
use scirs2_fft::simd_fft::neon;

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

/// Standard 4-point twiddle factors: `[W_4^1, W_4^2, W_4^3] = [−j, −1, +j]`.
fn twiddles4() -> [Complex64; 3] {
    [
        Complex64::new(0.0, -1.0), // W_4^1 = -j
        Complex64::new(-1.0, 0.0), // W_4^2 = -1
        Complex64::new(0.0, 1.0),  // W_4^3 = +j
    ]
}

/// Standard 8-point twiddle factors: `[W_8^1, …, W_8^7]`.
fn twiddles8() -> [Complex64; 7] {
    std::array::from_fn(|k| {
        let angle = -2.0 * PI * (k + 1) as f64 / 8.0;
        Complex64::new(angle.cos(), angle.sin())
    })
}

// ─────────────────────────────────────────────────────────────────────────────
//  Radix-4 tests
// ─────────────────────────────────────────────────────────────────────────────

/// NEON radix-4 matches scalar for a complex impulse input.
#[test]
fn neon_radix4_matches_scalar() {
    let twiddles = twiddles4();
    let input = [
        Complex64::new(1.0, 2.0),
        Complex64::new(3.0, 4.0),
        Complex64::new(5.0, 6.0),
        Complex64::new(7.0, 8.0),
    ];

    // Scalar reference
    let mut scalar_data = input;
    #[cfg(target_arch = "aarch64")]
    neon::radix4_butterfly_scalar(&mut scalar_data, &twiddles);
    #[cfg(not(target_arch = "aarch64"))]
    scirs2_fft::butterfly4(&mut scalar_data, &twiddles);

    #[cfg(target_arch = "aarch64")]
    {
        let mut neon_data = input;
        // Safety: NEON is always available on AArch64.
        unsafe {
            neon::radix4_butterfly_neon(neon_data.as_mut_ptr(), twiddles.as_ptr());
        }
        let err = max_err(&scalar_data, &neon_data);
        assert!(
            err < 1e-12,
            "NEON radix-4 diverges from scalar by {err}\n  scalar={scalar_data:?}\n  neon={neon_data:?}"
        );
    }

    // Scalar path must always produce finite values.
    assert!(
        scalar_data
            .iter()
            .all(|c| c.re.is_finite() && c.im.is_finite()),
        "scalar produced non-finite output: {scalar_data:?}"
    );
}

/// NEON radix-4 result matches direct DFT for the same standard twiddles.
#[test]
fn neon_radix4_matches_direct_dft() {
    let input = [
        Complex64::new(1.0, 0.0),
        Complex64::new(2.0, 0.0),
        Complex64::new(3.0, 0.0),
        Complex64::new(4.0, 0.0),
    ];
    let twiddles = twiddles4();

    let expected = direct_dft(&input);

    // Scalar must match direct DFT.
    let mut scalar_data = input;
    #[cfg(target_arch = "aarch64")]
    neon::radix4_butterfly_scalar(&mut scalar_data, &twiddles);
    #[cfg(not(target_arch = "aarch64"))]
    scirs2_fft::butterfly4(&mut scalar_data, &twiddles);

    let scalar_err = max_err(&scalar_data, &expected);
    assert!(
        scalar_err < 1e-12,
        "scalar radix-4 vs direct DFT: err={scalar_err}"
    );

    #[cfg(target_arch = "aarch64")]
    {
        let mut neon_data = input;
        // Safety: NEON is always available on AArch64.
        unsafe {
            neon::radix4_butterfly_neon(neon_data.as_mut_ptr(), twiddles.as_ptr());
        }
        let neon_err = max_err(&neon_data, &expected);
        assert!(
            neon_err < 1e-12,
            "NEON radix-4 vs direct DFT: err={neon_err}"
        );
    }
}

/// Dispatch wrapper produces the same result as direct scalar for radix-4.
#[test]
fn neon_dispatch_radix4_agrees_with_scalar() {
    let twiddles = twiddles4();
    let input = [
        Complex64::new(2.0, -1.0),
        Complex64::new(0.5, 3.0),
        Complex64::new(-1.0, 1.0),
        Complex64::new(4.0, -2.0),
    ];

    let mut scalar_data = input;
    #[cfg(target_arch = "aarch64")]
    neon::radix4_butterfly_scalar(&mut scalar_data, &twiddles);
    #[cfg(not(target_arch = "aarch64"))]
    scirs2_fft::butterfly4(&mut scalar_data, &twiddles);

    let mut dispatch_data = input;
    #[cfg(target_arch = "aarch64")]
    neon::radix4_butterfly_dispatch(&mut dispatch_data, &twiddles);
    #[cfg(not(target_arch = "aarch64"))]
    scirs2_fft::butterfly4(&mut dispatch_data, &twiddles);

    let err = max_err(&scalar_data, &dispatch_data);
    assert!(err < 1e-12, "dispatch vs scalar radix-4 err={err}");
}

// ─────────────────────────────────────────────────────────────────────────────
//  Radix-8 tests
// ─────────────────────────────────────────────────────────────────────────────

/// NEON radix-8 matches scalar for arbitrary complex input.
#[test]
fn neon_radix8_matches_scalar() {
    let input: [Complex64; 8] = std::array::from_fn(|k| {
        let t = k as f64 * 0.5;
        Complex64::new(t.sin() + 1.0, t.cos() - 0.5)
    });
    let twiddles = twiddles8();

    let mut scalar_data = input;
    #[cfg(target_arch = "aarch64")]
    neon::radix8_butterfly_scalar(&mut scalar_data, &twiddles);
    #[cfg(not(target_arch = "aarch64"))]
    scirs2_fft::butterfly8(&mut scalar_data, &twiddles);

    #[cfg(target_arch = "aarch64")]
    {
        let mut neon_data = input;
        // Safety: NEON is always available on AArch64.
        unsafe {
            neon::radix8_butterfly_neon(neon_data.as_mut_ptr(), twiddles.as_ptr());
        }
        let err = max_err(&scalar_data, &neon_data);
        assert!(err < 1e-12, "NEON radix-8 diverges from scalar by {err}");
    }

    assert!(
        scalar_data
            .iter()
            .all(|c| c.re.is_finite() && c.im.is_finite()),
        "scalar radix-8 produced non-finite output"
    );
}

/// NEON radix-8 matches the direct O(N²) DFT ground truth.
#[test]
fn neon_radix8_matches_direct_dft() {
    let input: [Complex64; 8] =
        std::array::from_fn(|k| Complex64::new((k as f64 * 0.7).cos(), -(k as f64 * 0.4).sin()));
    let twiddles = twiddles8();

    let expected = direct_dft(&input);

    let mut scalar_data = input;
    #[cfg(target_arch = "aarch64")]
    neon::radix8_butterfly_scalar(&mut scalar_data, &twiddles);
    #[cfg(not(target_arch = "aarch64"))]
    scirs2_fft::butterfly8(&mut scalar_data, &twiddles);

    let scalar_err = max_err(&scalar_data, &expected);
    assert!(
        scalar_err < 1e-10,
        "scalar radix-8 vs direct DFT err={scalar_err}"
    );

    #[cfg(target_arch = "aarch64")]
    {
        let mut neon_data = input;
        // Safety: NEON is always available on AArch64.
        unsafe {
            neon::radix8_butterfly_neon(neon_data.as_mut_ptr(), twiddles.as_ptr());
        }
        let neon_err = max_err(&neon_data, &expected);
        assert!(
            neon_err < 1e-10,
            "NEON radix-8 vs direct DFT err={neon_err}"
        );
    }
}

/// Dispatch wrapper for radix-8 agrees with scalar.
#[test]
fn neon_dispatch_radix8_agrees_with_scalar() {
    let input: [Complex64; 8] =
        std::array::from_fn(|k| Complex64::new(k as f64 * 0.7 - 1.0, k as f64 * 0.3));
    let twiddles = twiddles8();

    let mut scalar_data = input;
    #[cfg(target_arch = "aarch64")]
    neon::radix8_butterfly_scalar(&mut scalar_data, &twiddles);
    #[cfg(not(target_arch = "aarch64"))]
    scirs2_fft::butterfly8(&mut scalar_data, &twiddles);

    let mut dispatch_data = input;
    #[cfg(target_arch = "aarch64")]
    neon::radix8_butterfly_dispatch(&mut dispatch_data, &twiddles);
    #[cfg(not(target_arch = "aarch64"))]
    scirs2_fft::butterfly8(&mut dispatch_data, &twiddles);

    let err = max_err(&scalar_data, &dispatch_data);
    assert!(err < 1e-12, "dispatch radix-8 vs scalar err={err}");
}

/// NEON radix-4 matches full DFT output on a reproducible complex signal.
#[test]
fn neon_matches_full_fft_on_fixed_input() {
    let input = [
        Complex64::new(1.23, -0.45),
        Complex64::new(-0.78, 2.34),
        Complex64::new(0.56, -1.23),
        Complex64::new(-0.12, 0.89),
    ];
    let twiddles = twiddles4();

    let expected = direct_dft(&input);

    let mut scalar_data = input;
    #[cfg(target_arch = "aarch64")]
    neon::radix4_butterfly_scalar(&mut scalar_data, &twiddles);
    #[cfg(not(target_arch = "aarch64"))]
    scirs2_fft::butterfly4(&mut scalar_data, &twiddles);

    let scalar_err = max_err(&scalar_data, &expected);
    assert!(
        scalar_err < 1e-12,
        "scalar vs DFT on fixed input: err={scalar_err}"
    );

    #[cfg(target_arch = "aarch64")]
    {
        let mut neon_data = input;
        // Safety: NEON is always available on AArch64.
        unsafe {
            neon::radix4_butterfly_neon(neon_data.as_mut_ptr(), twiddles.as_ptr());
        }
        let neon_err = max_err(&neon_data, &expected);
        assert!(
            neon_err < 1e-12,
            "NEON vs DFT on fixed input: err={neon_err}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  SVE tests
// ─────────────────────────────────────────────────────────────────────────────

/// SVE availability check must not panic.
#[test]
fn sve_availability_check_does_not_panic() {
    #[cfg(target_arch = "aarch64")]
    {
        let available = neon::sve::is_sve_available();
        // Just check it returns without panic; value depends on hardware.
        eprintln!("[sve] SVE available on this host: {available}");
    }
    // On non-AArch64 this is a no-op; no assertion needed.
}

/// SVE radix-4 butterfly matches scalar (dispatch to NEON when SVE absent).
#[test]
fn sve_radix4_matches_scalar() {
    let twiddles = twiddles4();
    let input = [
        Complex64::new(1.0, 2.0),
        Complex64::new(3.0, 4.0),
        Complex64::new(5.0, 6.0),
        Complex64::new(7.0, 8.0),
    ];

    let mut scalar_data = input;
    #[cfg(target_arch = "aarch64")]
    neon::radix4_butterfly_scalar(&mut scalar_data, &twiddles);
    #[cfg(not(target_arch = "aarch64"))]
    scirs2_fft::butterfly4(&mut scalar_data, &twiddles);

    #[cfg(target_arch = "aarch64")]
    {
        let mut sve_data = input;
        neon::sve::radix4_butterfly_sve(&mut sve_data, &twiddles);
        let err = max_err(&scalar_data, &sve_data);
        assert!(err < 1e-12, "SVE radix-4 diverges from scalar by {err}");
    }

    assert!(
        scalar_data
            .iter()
            .all(|c| c.re.is_finite() && c.im.is_finite()),
        "scalar produced non-finite output"
    );
}

/// SVE radix-8 butterfly matches scalar.
#[test]
fn sve_radix8_matches_scalar() {
    let input: [Complex64; 8] =
        std::array::from_fn(|k| Complex64::new(k as f64 * 0.3 + 0.1, -(k as f64 * 0.2)));
    let twiddles = twiddles8();

    let mut scalar_data = input;
    #[cfg(target_arch = "aarch64")]
    neon::radix8_butterfly_scalar(&mut scalar_data, &twiddles);
    #[cfg(not(target_arch = "aarch64"))]
    scirs2_fft::butterfly8(&mut scalar_data, &twiddles);

    #[cfg(target_arch = "aarch64")]
    {
        let mut sve_data = input;
        neon::sve::radix8_butterfly_sve(&mut sve_data, &twiddles);
        let err = max_err(&scalar_data, &sve_data);
        assert!(err < 1e-12, "SVE radix-8 diverges from scalar by {err}");
    }

    assert!(
        scalar_data
            .iter()
            .all(|c| c.re.is_finite() && c.im.is_finite()),
        "scalar produced non-finite output"
    );
}
