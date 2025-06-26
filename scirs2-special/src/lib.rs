//! Special functions module
//!
//! This module provides implementations of various special mathematical functions,
//! following SciPy's `special` module with enhanced numerical stability and precision.
//!
//! ## Overview
//!
//! * Gamma and Beta functions
//! * Bessel functions
//! * Combinatorial functions (factorials, binomial coefficients, Stirling numbers, etc.)
//! * Statistical functions (logistic, softmax, log-softmax, sinc, etc.)
//! * Orthogonal polynomials
//! * Error functions
//! * Airy functions
//! * Exponential and logarithmic integrals
//! * Elliptic integrals and functions
//! * Hypergeometric functions
//! * Spherical harmonics
//! * Mathieu functions
//! * Zeta functions
//! * Lambert W function
//! * Wright Omega function
//! * Logarithmic integral
//!
//! ## Performance Features
//!
//! * **GPU Acceleration** (with `gpu` feature): Automatically accelerates array operations
//!   on supported hardware for gamma, Bessel, and error functions
//! * **Memory-Efficient Processing**: Chunked processing for large arrays to avoid
//!   memory overflow and improve cache efficiency
//! * **SIMD Optimizations**: Vectorized implementations for improved performance
//! * **Parallel Processing**: Multi-threaded execution for large arrays
//!
//! ## Examples
//!
//! Basic usage:
//! ```
//! use scirs2_special::gamma;
//!
//! let gamma_value = gamma(5.0f64);
//! assert!((gamma_value - 24.0).abs() < 1e-10);
//! ```
//!
//! Memory-efficient processing for large arrays:
//! ```no_run
//! use ndarray::Array1;
//! use scirs2_special::memory_efficient::gamma_chunked;
//!
//! let large_array = Array1::linspace(0.1, 10.0, 1_000_000);
//! let result = gamma_chunked(&large_array, None).unwrap();
//! ```
//!
//! GPU acceleration (requires `gpu` feature):
//! ```ignore
//! use ndarray::Array1;
//! use scirs2_special::gpu_ops::gamma_gpu;
//!
//! let input = Array1::linspace(0.1, 10.0, 100_000);
//! let mut output = Array1::zeros(100_000);
//! gamma_gpu(&input.view(), &mut output.view_mut()).unwrap();
//! ```

// Export error types
pub mod error;
pub use error::{SpecialError, SpecialResult};

// Modules
mod airy;
pub mod array_ops;
pub mod bessel;
mod combinatorial;
mod constants;
mod coulomb;
mod elliptic;
mod erf;
mod fresnel;
mod gamma;
#[cfg(feature = "gpu")]
pub mod gpu_ops;
mod hypergeometric;
mod kelvin;
mod lambert;
mod logint;
mod mathieu;
pub mod memory_efficient;
pub mod optimizations;
mod orthogonal;
mod parabolic;
pub mod precision;
mod property_tests;
mod simd_ops;
mod spherical_harmonics;
mod spheroidal;
mod statistical;
mod struve;
mod validation;
mod wright;
mod wright_bessel;
mod wright_simplified;
mod zeta;

// Re-export common functions
// Note: These functions require various trait bounds in their implementation,
// including Float, FromPrimitive, Debug, AddAssign, etc.
pub use airy::{ai, aip, bi, bip};
// Complex Airy functions
pub use airy::complex::{ai_complex, aip_complex, bi_complex, bip_complex};
pub use bessel::{
    // Regular Bessel functions
    i0,
    i1,
    iv,
    j0,
    // Derivatives of Bessel functions
    j0_prime,
    j1,
    j1_prime,
    jn,
    jn_prime,
    jv,
    jv_prime,
    k0,
    k1,
    kv,
    // Spherical Bessel functions
    spherical_jn,
    spherical_yn,
    y0,
    y0_prime,
    y1,
    y1_prime,
    yn,
    yn_prime,
};
pub use combinatorial::{
    bell_number, bernoulli_number, binomial, double_factorial, euler_number, factorial,
    permutations, stirling_first, stirling_second,
};
pub use coulomb::{coulomb_f, coulomb_g, coulomb_h_minus, coulomb_h_plus, coulomb_phase_shift};
pub use elliptic::{
    elliptic_e, elliptic_e_inc, elliptic_f, elliptic_k, elliptic_pi, jacobi_cn, jacobi_dn,
    jacobi_sn,
};
pub use fresnel::{
    fresnel, fresnel_complex, fresnelc, fresnels, mod_fresnel_minus, mod_fresnel_plus,
};
pub use gamma::{
    beta, betainc, betainc_regularized, betaincinv, betaln, digamma, gamma, gammaln, loggamma,
};
// Complex gamma functions
pub use gamma::complex::{beta_complex, digamma_complex, gamma_complex, loggamma_complex};
// Complex Bessel functions
pub use bessel::complex::{i0_complex, j0_complex, j1_complex, jn_complex, jv_complex, k0_complex};
// Complex error functions
pub use erf::complex::{erf_complex, erfc_complex, erfcx_complex, faddeeva_complex};
pub use hypergeometric::{hyp1f1, hyp2f1, ln_pochhammer, pochhammer};
pub use kelvin::{bei, beip, ber, berp, kei, keip, kelvin, ker, kerp};
pub use lambert::{lambert_w, lambert_w_real};
pub use logint::{chi, ci, e1, expint, li, li_complex, polylog, shi, si};
pub use mathieu::{
    mathieu_a, mathieu_b, mathieu_cem, mathieu_even_coef, mathieu_odd_coef, mathieu_sem,
};
pub use orthogonal::{
    chebyshev, gegenbauer, hermite, hermite_prob, jacobi, laguerre, laguerre_generalized, legendre,
    legendre_assoc,
};
pub use parabolic::{pbdv, pbdv_seq, pbvv, pbvv_seq, pbwa};
pub use spherical_harmonics::{sph_harm, sph_harm_complex};
pub use spheroidal::{
    obl_ang1, obl_cv, obl_cv_seq, obl_rad1, obl_rad2, pro_ang1, pro_cv, pro_cv_seq, pro_rad1,
    pro_rad2,
};
pub use statistical::{
    expm1_array, log1p_array, log_abs_gamma, log_softmax, logistic, logistic_derivative, logsumexp,
    sinc, sinc_array, softmax,
};
pub use struve::{it2_struve0, it_mod_struve0, it_struve0, mod_struve, struve};
pub use wright::{wright_omega_optimized, wright_omega_real_optimized};
pub use wright_bessel::{wright_bessel, wright_bessel_complex, wright_bessel_zeros};
pub use wright_simplified::{wright_omega, wright_omega_real};
pub use zeta::{hurwitz_zeta, zeta, zetac};

// SIMD operations (when enabled)
#[cfg(feature = "simd")]
pub use simd_ops::{
    benchmark_simd_performance, erf_f32_simd, exp_f32_simd, gamma_f32_simd, gamma_f64_simd,
    j0_f32_simd, vectorized_special_ops,
};

// Parallel operations (when enabled)
#[cfg(feature = "parallel")]
pub use simd_ops::{
    adaptive_gamma_processing, benchmark_parallel_performance, gamma_f64_parallel, j0_f64_parallel,
};

// Combined SIMD+Parallel operations (when both enabled)
#[cfg(all(feature = "simd", feature = "parallel"))]
pub use simd_ops::gamma_f32_simd_parallel;

// Error function and related functions
pub use erf::{erf, erfc, erfcinv, erfinv};

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use num_complex::Complex64;

    #[test]
    fn test_gamma_function() {
        // Test integer values
        assert_relative_eq!(gamma(1.0), 1.0, epsilon = 1e-10);
        assert_relative_eq!(gamma(2.0), 1.0, epsilon = 1e-10);
        assert_relative_eq!(gamma(3.0), 2.0, epsilon = 1e-10);
        assert_relative_eq!(gamma(4.0), 6.0, epsilon = 1e-10);
        assert_relative_eq!(gamma(5.0), 24.0, epsilon = 1e-10);
    }

    #[test]
    fn test_lambert_w() {
        // Test principal branch (k=0)
        let w = lambert_w(Complex64::new(1.0, 0.0), 0, 1e-8).unwrap();
        let expected = Complex64::new(0.567_143_290_409_783_8, 0.0);
        assert!((w - expected).norm() < 1e-10);

        // Test w * exp(w) = z
        let z = Complex64::new(1.0, 0.0);
        let w_exp_w = w * w.exp();
        assert!((w_exp_w - z).norm() < 1e-10);

        // Test branch k = 1
        let w_b1 = lambert_w(Complex64::new(1.0, 0.0), 1, 1e-8).unwrap();
        assert!(w_b1.im > 0.0);

        // Test branch k = -1
        let w_bm1 = lambert_w(Complex64::new(1.0, 0.0), -1, 1e-8).unwrap();
        assert!(w_bm1.im < 0.0);

        // Test real function
        let w_real = lambert_w_real(1.0, 1e-8).unwrap();
        assert!((w_real - 0.567_143_290_409_783_8).abs() < 1e-10);
    }
}
