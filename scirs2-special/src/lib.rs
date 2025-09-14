#![allow(deprecated)]
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
pub mod error_context;
pub mod error_wrappers;
pub use error::{SpecialError, SpecialResult};

// Modules
mod airy;
#[cfg(feature = "high-precision")]
pub mod arbitrary_precision;
#[cfg(feature = "gpu")]
pub mod array_ops;
pub mod bessel;
mod bessel_zeros;
mod boxcox;
mod carlson;
mod combinatorial;
mod constants;
pub mod convenience;
mod coulomb;
pub mod cross_validation;
mod distributions;
pub mod edge_case_tests;
mod ellipsoidal;
mod elliptic;
mod erf;
#[cfg(test)]
mod extended_property_tests;
pub mod extended_scipy_validation;
mod fresnel;
mod gamma;
#[cfg(feature = "gpu")]
pub mod gpu_context_manager;
#[cfg(feature = "gpu")]
pub mod gpu_ops;
mod hypergeometric;
mod incomplete_gamma;
mod information_theory;
mod kelvin;
mod lambert;
mod logint;
mod mathieu;
pub mod memory_efficient;
pub mod optimizations;
mod orthogonal;
mod parabolic;
pub mod performance_benchmarks;
pub mod physics_engineering;
pub mod precision;
mod property_tests;
pub mod python_interop;
#[cfg(test)]
mod quickcheck_tests;
mod simd_ops;
mod spherical_harmonics;
mod spheroidal;
pub mod stability_analysis;
mod statistical;
mod struve;
mod utility;
mod validation;
#[cfg(feature = "plotting")]
pub mod visualization;
mod voigt;
mod wright;
mod wright_bessel;
mod wright_simplified;
mod zeta;

// Re-export common functions
// Note: These functions require various trait bounds in their implementation,
// including Float, FromPrimitive, Debug, AddAssign, etc.
pub use airy::{ai, ai_zeros, aie, aip, airye, bi, bi_zeros, bie, bip, itairy};
// Complex Airy functions
pub use airy::complex::{ai_complex, aip_complex, bi_complex, bip_complex};
pub use bessel::{
    h1vp,
    h2vp,
    // Hankel functions
    hankel1,
    hankel1e,
    hankel2,
    hankel2e,
    // Regular Bessel functions
    i0,
    // Derivatives of modified Bessel functions
    i0_prime,
    i0e,
    i1,
    i1_prime,
    i1e,
    iv,
    iv_prime,
    ive,
    ivp,
    j0,
    // Derivatives of Bessel functions
    j0_prime,
    j0e,
    j1,
    j1_prime,
    j1e,
    jn,
    jn_prime,
    jne,
    jv,
    jv_prime,
    jve,
    // SciPy-compatible derivative interfaces
    jvp,
    k0,
    k0_prime,
    k0e,
    k1,
    k1_prime,
    k1e,
    kv,
    kv_prime,
    kve,
    kvp,
    // Spherical Bessel functions
    spherical_jn,
    spherical_yn,
    y0,
    y0_prime,
    y0e,
    y1,
    y1_prime,
    y1e,
    yn,
    yn_prime,
    yne,
    yvp,
};
pub use bessel_zeros::{
    besselpoly,
    // Bessel utilities
    itj0y0,
    // Zeros of Bessel functions
    j0_zeros,
    j1_zeros,
    jn_zeros,
    jnjnp_zeros,
    jnp_zeros,
    jnyn_zeros,
    y0_zeros,
    y1_zeros,
    yn_zeros,
};
pub use boxcox::{
    boxcox, boxcox1p, boxcox1p_array, boxcox_array, inv_boxcox, inv_boxcox1p, inv_boxcox1p_array,
    inv_boxcox_array,
};
pub use carlson::{elliprc, elliprd, elliprf, elliprf_array, elliprg, elliprj};
pub use combinatorial::{
    bell_number, bernoulli_number, binomial, comb, double_factorial, euler_number, factorial,
    factorial2, factorialk, perm, permutations, stirling2, stirling_first, stirling_second,
};
pub use coulomb::{coulomb_f, coulomb_g, coulomb_h_plus, coulomb_hminus, coulomb_phase_shift};
pub use distributions::{
    // Binomial distribution
    bdtr,
    bdtr_array,
    bdtrc,
    bdtri,
    bdtrik,
    bdtrin,
    // Beta distribution inverse functions
    btdtria,
    btdtrib,
    // Chi-square distribution
    chdtr,
    chdtrc,
    chdtri,
    // F distribution
    fdtr,
    fdtrc,
    fdtridfd,
    // Gamma distribution
    gdtr,
    gdtrc,
    gdtria,
    gdtrib,
    gdtrix,
    kolmogi,
    // Kolmogorov-Smirnov distribution
    kolmogorov,
    log_ndtr,
    // Negative binomial distribution
    nbdtr,
    nbdtrc,
    nbdtri,
    // Normal distribution
    ndtr,
    ndtr_array,
    ndtri,
    ndtri_exp,
    // Poisson distribution
    pdtr,
    pdtrc,
    pdtri,
    pdtrik,
    // Student's t distribution
    stdtr,
};
pub use ellipsoidal::{
    ellip_harm, ellip_harm_2, ellip_harm_array, ellip_harm_coefficients, ellip_harm_complex,
    ellip_normal,
};
pub use elliptic::{
    ellipe, ellipeinc, ellipj, ellipk, ellipkinc, ellipkm1, elliptic_e, elliptic_e_inc, elliptic_f,
    elliptic_k, elliptic_pi, jacobi_cn, jacobi_dn, jacobi_sn,
};
pub use fresnel::{
    fresnel, fresnel_complex, fresnelc, fresnels, mod_fresnel_plus, mod_fresnelminus,
};
pub use gamma::{
    beta,
    // Safe versions with error handling
    beta_safe,
    betainc,
    betainc_regularized,
    betaincinv,
    betaln,
    digamma,
    digamma_safe,
    gamma,
    gamma_safe,
    gammaln,
    loggamma,
    polygamma,
};
pub use incomplete_gamma::{
    gammainc, gammainc_lower, gammainc_upper, gammaincc, gammainccinv, gammaincinv, gammasgn,
    gammastar,
};
// Complex gamma functions
pub use gamma::complex::{beta_complex, digamma_complex, gamma_complex, loggamma_complex};
// Complex Bessel functions
pub use bessel::complex::{i0_complex, j0_complex, j1_complex, jn_complex, jv_complex, k0_complex};
// Complex error functions
pub use erf::complex::{erf_complex, erfc_complex, erfcx_complex, faddeeva_complex};
pub use hypergeometric::{hyp0f1, hyp1f1, hyp2f1, hyperu, ln_pochhammer, pochhammer};
pub use information_theory::{
    binary_entropy, cross_entropy, entr, entr_array, entropy, huber, huber_loss, kl_div,
    kl_divergence, pseudo_huber, rel_entr,
};
pub use kelvin::{bei, beip, ber, berp, kei, keip, kelvin, ker, kerp};
pub use lambert::{lambert_w, lambert_w_real};
pub use logint::{chi, ci, e1, expint, li, li_complex, polylog, shi, shichi, si, sici, spence};
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
pub use utility::{
    agm,
    // Basic functions
    cbrt,
    // Array operations
    cbrt_array,
    cosdg,
    // Accurate computations
    cosm1,
    cotdg,
    // Special functions
    diric,
    exp10,
    exp10_array,
    exp2,
    // Statistical functions (SciPy compatibility)
    expit,
    expit_array,
    expm1_array_utility,
    exprel,
    gradient,
    log1p_array_utility,
    log_expit,
    logit,
    logit_array,
    owens_t,
    powm1,
    // Trigonometric in degrees
    radian,
    round,
    round_array,
    sindg,
    softplus,
    spherical_distance,
    tandg,
    xlog1py,
    // Additional convenience functions for SciPy parity
    xlog1py_scalar,
    xlogy,
};
pub use voigt::{
    pseudo_voigt, voigt_profile, voigt_profile_array, voigt_profile_fwhm, voigt_profile_fwhm_array,
    voigt_profile_normalized,
};
pub use wright::{wright_omega_optimized, wright_omega_real_optimized};
pub use wright_bessel::{
    log_wright_bessel, wright_bessel, wright_bessel_complex, wright_bessel_zeros,
};
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
pub use erf::{dawsn, erf, erfc, erfcinv, erfcx, erfi, erfinv, wofz};

// Arbitrary precision functions (when enabled)
#[cfg(feature = "high-precision")]
pub use arbitrary_precision::{
    bessel::{bessel_j_ap, bessel_j_mp, bessel_y_ap, bessel_y_mp},
    cleanup_cache,
    error_function::{erf_ap, erf_mp, erfc_ap, erfc_mp},
    gamma::{gamma_ap, gamma_mp, log_gamma_ap, log_gamma_mp},
    to_complex64, to_f64, PrecisionContext,
};

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
