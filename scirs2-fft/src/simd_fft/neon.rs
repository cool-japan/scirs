//! ARM NEON and SVE accelerated radix-4 and radix-8 FFT butterfly kernels.
//!
//! This module provides NEON-intrinsics-based butterfly kernels compiled with
//! `#[target_feature(enable = "neon")]` so the compiler can emit NEON-encoded
//! instructions.  On AArch64, NEON is architecturally mandatory (always
//! available), so there is no runtime capability check needed beyond the
//! compile-time target check.
//!
//! ## Design notes
//!
//! Each butterfly matches the **exact algorithm** of its scalar sibling in
//! `src/butterfly.rs` and the AVX-512 implementation in `simd_fft/avx512.rs`
//! — a full DFT-matrix multiply with derived twiddle powers — so that the
//! dispatch wrappers can freely swap between the two paths and produce
//! numerically equivalent results.
//!
//! ### Radix-4 twiddle convention
//! `twiddles = [W_4^1, W_4^2, W_4^3]`.  Derived: `w4 = w2²`, `w6 = w4·w2`, `w9 = w3³`.
//!
//! ```text
//! X[0] = x0 + x1         + x2       + x3
//! X[1] = x0 + w1·x1      + w2·x2    + w3·x3
//! X[2] = x0 + w2·x1      + w4·x2    + w6·x3
//! X[3] = x0 + w3·x1      + w6·x2    + w9·x3
//! ```
//!
//! ### Radix-8 twiddle convention
//! `twiddles[0..7]` = `[W_8^1, W_8^2, …, W_8^7]`.  `w[0] = 1`, `w[m] = twiddles[m-1]`.
//!
//! ```text
//! X[k] = Σ_{n=0}^{7} x[n] · w[(n·k) mod 8]
//! ```
//!
//! ### NEON complex multiplication
//! Each `float64x2_t` encodes a single `Complex<f64>` as `[re, im]` (little-endian
//! lane order: lane 0 = re, lane 1 = im).
//!
//! ```text
//! (a + bi)(c + di) = (ac − bd) + (ad + bc)i
//! ```
//!
//! Computed via:
//! 1. `ac = vmulq_f64(a, [c, c])`  — `[a_re·c, a_im·c]`
//! 2. `ad = vmulq_f64([a_im, a_re], [d, d])`  — `[a_im·d, a_re·d]`
//! 3. result = `vfmsq_f64(ac, [−1, +1], ad_neg)` ... or simply:
//!    - `result[0] = ac[0] − a_im·d`   (fmsub)
//!    - `result[1] = ac[1] + a_re·d`   (fmadd)
//!
//! We use `vfmsq_f64` / `vfmaq_f64` for fused multiply-subtract/add.

use scirs2_core::numeric::Complex64;
use std::arch::aarch64::*;

// ─────────────────────────────────────────────────────────────────────────────
//  Scalar reference implementations
// ─────────────────────────────────────────────────────────────────────────────

/// Scalar radix-4 DFT butterfly — identical algorithm to `butterfly::butterfly4`.
///
/// `twiddles[0..3]` = `[W_4^1, W_4^2, W_4^3]`.
#[inline(always)]
pub fn radix4_butterfly_scalar(a: &mut [Complex64; 4], twiddles: &[Complex64; 3]) {
    let x0 = a[0];
    let x1 = a[1];
    let x2 = a[2];
    let x3 = a[3];

    let w1 = twiddles[0];
    let w2 = twiddles[1];
    let w3 = twiddles[2];

    let w4 = w2 * w2;
    let w6 = w4 * w2;
    let w9 = w3 * w3 * w3;

    a[0] = x0 + x1 + x2 + x3;
    a[1] = x0 + w1 * x1 + w2 * x2 + w3 * x3;
    a[2] = x0 + w2 * x1 + w4 * x2 + w6 * x3;
    a[3] = x0 + w3 * x1 + w6 * x2 + w9 * x3;
}

/// Scalar radix-8 DFT butterfly — identical algorithm to `butterfly::butterfly8`.
///
/// `twiddles[0..7]` = `[W_8^1, …, W_8^7]`.
#[inline(always)]
pub fn radix8_butterfly_scalar(a: &mut [Complex64; 8], twiddles: &[Complex64; 7]) {
    let w = [
        Complex64::new(1.0, 0.0),
        twiddles[0],
        twiddles[1],
        twiddles[2],
        twiddles[3],
        twiddles[4],
        twiddles[5],
        twiddles[6],
    ];

    let input = *a;
    for k in 0..8 {
        let mut sum = Complex64::new(0.0, 0.0);
        for n in 0..8 {
            let idx = (n * k) % 8;
            sum += input[n] * w[idx];
        }
        a[k] = sum;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  NEON complex-multiply helper
// ─────────────────────────────────────────────────────────────────────────────

/// Complex multiplication of two `float64x2_t` registers, each encoding
/// one `Complex<f64>` as `[re, im]` (lane 0 = real, lane 1 = imaginary).
///
/// Computes `(a.re + a.im·i) * (b.re + b.im·i)`:
/// ```text
/// re_out = a.re·b.re − a.im·b.im
/// im_out = a.re·b.im + a.im·b.re
/// ```
///
/// Implementation:
/// 1. Broadcast `b.re` to both lanes: `b_re = [b.re, b.re]`
/// 2. Broadcast `b.im` to both lanes: `b_im = [b.im, b.im]`
/// 3. `ac  = a * b_re  = [a.re·b.re, a.im·b.re]`
/// 4. `a_swap = [a.im, a.re]`  (swap lanes via `vextq_f64::<1>`)
/// 5. `result = vfmaq_f64(vfmsq_f64(b_im, a_swap, ...), ac, ...)` …
///    simplified: use fmsub for real part, fmadd for imaginary:
///    - `re_out = a.re·b.re − a.im·b.im`  = fma(−a.im, b.im, a.re·b.re)
///    - `im_out = a.im·b.re + a.re·b.im`  = fma( a.re, b.im, a.im·b.re)
///
/// # Safety
/// Requires AArch64 with NEON (always true on AArch64).
#[target_feature(enable = "neon")]
#[inline]
unsafe fn cmul_f64(a: float64x2_t, b: float64x2_t) -> float64x2_t {
    // Broadcast real/imag parts of b
    let b_re = vdupq_laneq_f64::<0>(b); // [b.re, b.re]
    let b_im = vdupq_laneq_f64::<1>(b); // [b.im, b.im]

    // a_swap = [a.im, a.re]
    let a_swap = vextq_f64::<1>(a, a); // [lane1, lane0] = [a.im, a.re]

    // ac = [a.re·b.re, a.im·b.re]
    let ac = vmulq_f64(a, b_re);

    // result:
    //   lane0 (re): a.re·b.re − a.im·b.im  = fmsub(a.im, b.im, from ac[0])
    //   lane1 (im): a.im·b.re + a.re·b.im  = fmadd(a.re, b.im, from ac[1])
    //
    // vfmsq_f64(a, b, c)  = a − b*c  → not what we want directly.
    // Use: vfmaq_f64(acc, x, y)  = acc + x*y
    //      vfmsq_f64(acc, x, y)  = acc − x*y
    //
    // For the two lanes simultaneously we need mixed sign.
    // Strategy: build `[a.re·b.im, a.re·b.im]` and `[a.im·b.im, a.im·b.im]`
    // is expensive. Instead compute per-lane with two FMAs:
    //
    //   partial = vfmsq_f64(ac, a_swap, b_im)
    //           = ac − a_swap * b_im
    //           = [a.re·b.re − a.im·b.im,  a.im·b.re − a.re·b.im]
    //
    //   That gives lane0 = re_out ✓, lane1 = -(im_out) ✗
    //
    // Then negate lane1 with a sign flip vector [-0.0, +0.0]:
    //   result = partial XOR [+0.0, −0.0] on lane1 → but no such xor in safe NEON.
    //
    // Cleaner: two separate FMAs, recombine:
    let re_out = vfmsq_f64(ac, a_swap, b_im); // [a.re·b.re - a.im·b.im,  a.im·b.re - a.re·b.im]
                                              // lane0 of re_out is correct; lane1 has wrong sign, ignore it.
                                              // Compute im lane separately:
    let im_vec = vfmaq_f64(ac, a_swap, b_im); // [a.re·b.re + a.im·b.im,  a.im·b.re + a.re·b.im]
                                              // lane1 of im_vec = a.im·b.re + a.re·b.im = im_out ✓; lane0 is garbage.
                                              //
                                              // Combine: pick lane0 from re_out, lane1 from im_vec.
                                              // vextq_f64 doesn't work here.  Use vzip or vtrn:
                                              //   vcombine_f64(vget_low_f64(re_out), vget_high_f64(im_vec))
                                              //   lane0 of f64x2 = vget_low_f64 (64-bit element 0 = lower half)
                                              //   lane1 of f64x2 = vget_high_f64 (64-bit element 1 = upper half)
                                              //
                                              // BUT: float64x2_t is [lane0=low, lane1=high]
                                              //   re_out lane0 = low  → vget_low_f64(re_out) gives the f64x1 holding lane0
                                              //   im_vec lane1 = high → vget_high_f64(im_vec) gives the f64x1 holding lane1
    vcombine_f64(vget_low_f64(re_out), vget_high_f64(im_vec))
}

// ─────────────────────────────────────────────────────────────────────────────
//  NEON radix-4 butterfly
// ─────────────────────────────────────────────────────────────────────────────

/// NEON radix-4 DFT butterfly.
///
/// Computes `X[k] = Σ_{n=0}^{3} x[n] · W_4^{n·k}` in-place using the same
/// DFT-matrix algorithm as `butterfly4` / `radix4_butterfly_scalar`.
///
/// `a` must point to 4 contiguous `Complex64` values (64 bytes total).
/// `twiddles` must point to 3 contiguous twiddle factors `[W^1, W^2, W^3]`.
///
/// # Safety
/// Caller must ensure `a` and `twiddles` are valid, non-null, and point to
/// the correct number of initialized `Complex64` values.  NEON is always
/// available on AArch64, so no runtime feature check is required.
#[target_feature(enable = "neon")]
pub unsafe fn radix4_butterfly_neon(a: *mut Complex64, twiddles: *const Complex64) {
    let p = a as *const f64;
    let tp = twiddles as *const f64;

    // Load inputs: each float64x2_t = one Complex64 = [re, im]
    let x0 = vld1q_f64(p); // a[0]
    let x1 = vld1q_f64(p.add(2)); // a[1]
    let x2 = vld1q_f64(p.add(4)); // a[2]
    let x3 = vld1q_f64(p.add(6)); // a[3]

    // Load twiddle factors
    let w1 = vld1q_f64(tp); // W^1 = twiddles[0]
    let w2 = vld1q_f64(tp.add(2)); // W^2 = twiddles[1]
    let w3 = vld1q_f64(tp.add(4)); // W^3 = twiddles[2]

    // Derived twiddle powers (mirror of avx512.rs and butterfly.rs):
    //   w4 = w2 * w2
    //   w6 = w4 * w2
    //   w9 = w3 * w3 * w3
    let w4 = cmul_f64(w2, w2);
    let w6 = cmul_f64(w4, w2);
    let w3sq = cmul_f64(w3, w3);
    let w9 = cmul_f64(w3sq, w3);

    // X[0] = x0 + x1 + x2 + x3
    let out0 = vaddq_f64(vaddq_f64(x0, x1), vaddq_f64(x2, x3));

    // X[1] = x0 + w1·x1 + w2·x2 + w3·x3
    let out1 = vaddq_f64(
        vaddq_f64(x0, cmul_f64(w1, x1)),
        vaddq_f64(cmul_f64(w2, x2), cmul_f64(w3, x3)),
    );

    // X[2] = x0 + w2·x1 + w4·x2 + w6·x3
    let out2 = vaddq_f64(
        vaddq_f64(x0, cmul_f64(w2, x1)),
        vaddq_f64(cmul_f64(w4, x2), cmul_f64(w6, x3)),
    );

    // X[3] = x0 + w3·x1 + w6·x2 + w9·x3
    let out3 = vaddq_f64(
        vaddq_f64(x0, cmul_f64(w3, x1)),
        vaddq_f64(cmul_f64(w6, x2), cmul_f64(w9, x3)),
    );

    // Store results
    let q = a as *mut f64;
    vst1q_f64(q, out0);
    vst1q_f64(q.add(2), out1);
    vst1q_f64(q.add(4), out2);
    vst1q_f64(q.add(6), out3);
}

// ─────────────────────────────────────────────────────────────────────────────
//  NEON radix-8 butterfly
// ─────────────────────────────────────────────────────────────────────────────

/// NEON radix-8 DFT butterfly.
///
/// Computes `X[k] = Σ_{n=0}^{7} x[n] · W_8^{n·k}` in-place.
/// `twiddles[0..7]` = `[W_8^1, …, W_8^7]`.
///
/// Implements the same DFT-matrix multiply as `butterfly8` / `radix8_butterfly_scalar`.
///
/// # Safety
/// `a` must point to 8 valid `Complex64` values; `twiddles` to 7.
#[target_feature(enable = "neon")]
pub unsafe fn radix8_butterfly_neon(a: *mut Complex64, twiddles: *const Complex64) {
    let p = a as *const f64;
    let tp = twiddles as *const f64;

    // Load all 8 input values
    let x = [
        vld1q_f64(p),         // a[0]
        vld1q_f64(p.add(2)),  // a[1]
        vld1q_f64(p.add(4)),  // a[2]
        vld1q_f64(p.add(6)),  // a[3]
        vld1q_f64(p.add(8)),  // a[4]
        vld1q_f64(p.add(10)), // a[5]
        vld1q_f64(p.add(12)), // a[6]
        vld1q_f64(p.add(14)), // a[7]
    ];

    // W_8^0 = 1 + 0i
    let w0 = {
        let re: f64 = 1.0;
        let im: f64 = 0.0;
        vld1q_f64([re, im].as_ptr())
    };

    // Twiddle power table: w[m] = W_8^m for m = 0..7
    // w[0] = 1, w[1..7] = twiddles[0..6]
    let w = [
        w0,
        vld1q_f64(tp),         // W_8^1 = twiddles[0]
        vld1q_f64(tp.add(2)),  // W_8^2 = twiddles[1]
        vld1q_f64(tp.add(4)),  // W_8^3 = twiddles[2]
        vld1q_f64(tp.add(6)),  // W_8^4 = twiddles[3]
        vld1q_f64(tp.add(8)),  // W_8^5 = twiddles[4]
        vld1q_f64(tp.add(10)), // W_8^6 = twiddles[5]
        vld1q_f64(tp.add(12)), // W_8^7 = twiddles[6]
    ];

    // X[k] = Σ_{n=0}^{7} x[n] * w[(n*k) % 8]
    let mut out = [vdupq_n_f64(0.0); 8];
    for k in 0..8 {
        let mut sum = vdupq_n_f64(0.0);
        for n in 0..8 {
            let idx = (n * k) % 8;
            sum = vaddq_f64(sum, cmul_f64(w[idx], x[n]));
        }
        out[k] = sum;
    }

    // Store results
    let q = a as *mut f64;
    for k in 0..8 {
        vst1q_f64(q.add(k * 2), out[k]);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Safe runtime-dispatch wrappers
// ─────────────────────────────────────────────────────────────────────────────

/// Returns `true` if NEON is available at runtime.
///
/// On AArch64, NEON is architecturally mandatory and always returns `true`.
#[inline]
pub fn is_neon_available() -> bool {
    // NEON is mandatory on AArch64
    true
}

/// Runtime-dispatch radix-4 butterfly.
///
/// On AArch64: always calls [`radix4_butterfly_neon`] (NEON is mandatory).
///
/// This function is always safe to call on AArch64.
pub fn radix4_butterfly_dispatch(a: &mut [Complex64; 4], twiddles: &[Complex64; 3]) {
    // Safety: NEON is always available on AArch64.
    unsafe {
        radix4_butterfly_neon(a.as_mut_ptr(), twiddles.as_ptr());
    }
}

/// Runtime-dispatch radix-8 butterfly.
///
/// On AArch64: always calls [`radix8_butterfly_neon`] (NEON is mandatory).
pub fn radix8_butterfly_dispatch(a: &mut [Complex64; 8], twiddles: &[Complex64; 7]) {
    // Safety: NEON is always available on AArch64.
    unsafe {
        radix8_butterfly_neon(a.as_mut_ptr(), twiddles.as_ptr());
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  SVE (Scalable Vector Extension) — runtime-gated sub-module
// ─────────────────────────────────────────────────────────────────────────────

/// SVE sub-module.  SVE is architecturally optional on AArch64 and requires
/// a runtime feature check.  The butterfly implementations in this module
/// delegate to the NEON path (which SVE CPUs can execute in their NEON-compat
/// mode) until stable SVE intrinsics (`svfloat64_t` etc.) are stabilised in
/// the Rust standard library.
pub mod sve {
    use super::{radix4_butterfly_neon, radix8_butterfly_neon};
    use scirs2_core::numeric::Complex64;

    /// Returns `true` if SVE is available at runtime.
    pub fn is_sve_available() -> bool {
        std::arch::is_aarch64_feature_detected!("sve")
    }

    /// Radix-4 butterfly dispatched to SVE when available, NEON otherwise.
    ///
    /// Full `svfloat64_t` SVE intrinsics require nightly Rust and an explicit
    /// `#[target_feature(enable = "sve")]` block; until those APIs stabilise
    /// this function runs the NEON path (valid and correct on SVE-capable CPUs).
    pub fn radix4_butterfly_sve(a: &mut [Complex64; 4], twiddles: &[Complex64; 3]) {
        // Safety: NEON is always available on AArch64 (SVE CPUs are a strict superset).
        unsafe {
            radix4_butterfly_neon(a.as_mut_ptr(), twiddles.as_ptr());
        }
    }

    /// Radix-8 butterfly dispatched to SVE when available, NEON otherwise.
    pub fn radix8_butterfly_sve(a: &mut [Complex64; 8], twiddles: &[Complex64; 7]) {
        // Safety: NEON is always available on AArch64.
        unsafe {
            radix8_butterfly_neon(a.as_mut_ptr(), twiddles.as_ptr());
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn max_err(a: &[Complex64], b: &[Complex64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).norm())
            .fold(0.0_f64, f64::max)
    }

    /// Scalar radix-4 matches known 4-point DFT output.
    #[test]
    fn test_scalar_radix4_matches_known() {
        let input = [
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 0.0),
            Complex64::new(3.0, 0.0),
            Complex64::new(4.0, 0.0),
        ];
        let twiddles = [
            Complex64::new(0.0, -1.0), // W_4^1 = -j
            Complex64::new(-1.0, 0.0), // W_4^2 = -1
            Complex64::new(0.0, 1.0),  // W_4^3 = +j
        ];
        let mut data = input;
        radix4_butterfly_scalar(&mut data, &twiddles);
        assert!((data[0] - Complex64::new(10.0, 0.0)).norm() < 1e-12);
        assert!((data[1] - Complex64::new(-2.0, 2.0)).norm() < 1e-12);
        assert!((data[2] - Complex64::new(-2.0, 0.0)).norm() < 1e-12);
        assert!((data[3] - Complex64::new(-2.0, -2.0)).norm() < 1e-12);
    }

    /// Scalar radix-8 matches direct DFT.
    #[test]
    fn test_scalar_radix8_matches_direct_dft() {
        let input: [Complex64; 8] = std::array::from_fn(|k| Complex64::new(k as f64 + 1.0, 0.0));
        let twiddles: [Complex64; 7] = std::array::from_fn(|k| {
            let angle = -2.0 * PI * (k + 1) as f64 / 8.0;
            Complex64::new(angle.cos(), angle.sin())
        });
        let mut data = input;
        radix8_butterfly_scalar(&mut data, &twiddles);

        // Reference via direct DFT
        let mut reference = [Complex64::new(0.0, 0.0); 8];
        for k in 0..8 {
            for n in 0..8 {
                let angle = -2.0 * PI * (n * k) as f64 / 8.0;
                reference[k] += input[n] * Complex64::new(angle.cos(), angle.sin());
            }
        }
        let err = max_err(&data, &reference);
        assert!(err < 1e-10, "radix8 scalar err={err}");
    }

    /// NEON radix-4 matches scalar radix-4.
    #[test]
    fn test_neon_radix4_matches_scalar() {
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

        let mut ref_data = input;
        radix4_butterfly_scalar(&mut ref_data, &twiddles);

        let mut neon_data = input;
        // Safety: NEON is always available on AArch64.
        unsafe { radix4_butterfly_neon(neon_data.as_mut_ptr(), twiddles.as_ptr()) };

        let err = max_err(&ref_data, &neon_data);
        assert!(
            err < 1e-12,
            "NEON radix-4 diverges from scalar by {err}\n  scalar={ref_data:?}\n  neon={neon_data:?}"
        );
    }

    /// NEON radix-8 matches scalar radix-8.
    #[test]
    fn test_neon_radix8_matches_scalar() {
        let input: [Complex64; 8] = std::array::from_fn(|k| {
            let t = k as f64 * 0.5;
            Complex64::new(t.sin() + 1.0, t.cos() - 0.5)
        });
        let twiddles: [Complex64; 7] = std::array::from_fn(|k| {
            let angle = -2.0 * PI * (k + 1) as f64 / 8.0;
            Complex64::new(angle.cos(), angle.sin())
        });

        let mut ref_data = input;
        radix8_butterfly_scalar(&mut ref_data, &twiddles);

        let mut neon_data = input;
        // Safety: NEON is always available on AArch64.
        unsafe { radix8_butterfly_neon(neon_data.as_mut_ptr(), twiddles.as_ptr()) };

        let err = max_err(&ref_data, &neon_data);
        assert!(err < 1e-12, "NEON radix-8 diverges from scalar by {err}");
    }

    /// Dispatch wrapper for radix-4 agrees with scalar.
    #[test]
    fn test_dispatch_radix4_agrees_with_scalar() {
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

        let mut ref_data = input;
        radix4_butterfly_scalar(&mut ref_data, &twiddles);

        let mut dispatch_data = input;
        radix4_butterfly_dispatch(&mut dispatch_data, &twiddles);

        let err = max_err(&ref_data, &dispatch_data);
        assert!(err < 1e-12, "dispatch vs scalar radix-4 err={err}");
    }

    /// Dispatch wrapper for radix-8 agrees with scalar.
    #[test]
    fn test_dispatch_radix8_agrees_with_scalar() {
        let input: [Complex64; 8] =
            std::array::from_fn(|k| Complex64::new(k as f64 * 0.7 - 1.0, k as f64 * 0.3));
        let twiddles: [Complex64; 7] = std::array::from_fn(|k| {
            let angle = -2.0 * PI * (k + 1) as f64 / 8.0;
            Complex64::new(angle.cos(), angle.sin())
        });

        let mut ref_data = input;
        radix8_butterfly_scalar(&mut ref_data, &twiddles);

        let mut dispatch_data = input;
        radix8_butterfly_dispatch(&mut dispatch_data, &twiddles);

        let err = max_err(&ref_data, &dispatch_data);
        assert!(err < 1e-12, "dispatch radix-8 vs scalar err={err}");
    }
}
