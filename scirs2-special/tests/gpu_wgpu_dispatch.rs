//! Integration tests for wgpu compute dispatch (scirs2-special).
//!
//! Gated on `wgpu_kernels` feature. Tests gracefully skip when no GPU adapter is available.

#![cfg(feature = "wgpu_kernels")]

use scirs2_special::gpu_kernels::wgsl::{
    bessel_j0_batch_wgpu, erf_batch_wgpu, erfc_batch_wgpu, erfinv_batch_wgpu, gamma_batch_wgpu,
    lgamma_batch_wgpu, WgslDispatchError,
};

fn is_no_gpu(e: &WgslDispatchError) -> bool {
    matches!(e, WgslDispatchError::GpuNotAvailable)
}

/// Run a test that either passes GPU validation or gracefully skips when no adapter is present.
macro_rules! gpu_test {
    ($label:expr, $call:expr, $body:expr) => {{
        match $call {
            Err(e) if is_no_gpu(&e) => {
                println!("{}: skipping — no GPU adapter", $label);
            }
            Err(e) => panic!("unexpected error: {:?}", e),
            Ok(ys) => $body(ys),
        }
    }};
}

#[test]
fn test_gamma_wgpu_batch() {
    let xs = vec![1.0_f64, 2.0, 3.0, 0.5];
    gpu_test!("test_gamma_wgpu_batch", gamma_batch_wgpu(&xs), |ys: Vec<
        f64,
    >| {
        assert_eq!(ys.len(), xs.len());
        // Γ(1)=1, Γ(2)=1, Γ(3)=2, Γ(0.5)=√π≈1.7724538509
        let expected = [1.0_f64, 1.0, 2.0, 1.772_453_850_9];
        for (i, (y, e)) in ys.iter().zip(expected.iter()).enumerate() {
            assert!(
                (y - e).abs() < 1e-3,
                "gamma(xs[{}]={}) = {} (expected {})",
                i,
                xs[i],
                y,
                e
            );
        }
    });
}

#[test]
fn test_erf_wgpu_batch() {
    let xs = vec![0.0_f64, 1.0, -1.0, 2.0];
    gpu_test!("test_erf_wgpu_batch", erf_batch_wgpu(&xs), |ys: Vec<
        f64,
    >| {
        assert_eq!(ys.len(), xs.len());
        // erf(0)=0, erf(1)≈0.8427, erf(-1)≈-0.8427, erf(2)≈0.9953
        let expected = [0.0_f64, 0.842_700_79, -0.842_700_79, 0.995_322_27];
        for (i, (y, e)) in ys.iter().zip(expected.iter()).enumerate() {
            assert!(
                (y - e).abs() < 1e-4,
                "erf(xs[{}]={}) = {} (expected {})",
                i,
                xs[i],
                y,
                e
            );
        }
    });
}

#[test]
fn test_bessel_j0_wgpu_batch() {
    let xs = vec![0.0_f64, 2.4048];
    gpu_test!(
        "test_bessel_j0_wgpu_batch",
        bessel_j0_batch_wgpu(&xs),
        |ys: Vec<f64>| {
            assert_eq!(ys.len(), xs.len());
            // J0(0)=1, J0(2.4048)≈0 (first zero)
            assert!((ys[0] - 1.0).abs() < 1e-3);
            assert!(ys[1].abs() < 0.01);
        }
    );
}

#[test]
fn test_lgamma_wgpu_batch() {
    let xs = vec![1.0_f64, 2.0, 3.0, 0.5];
    // lgamma(1)=0, lgamma(2)=0, lgamma(3)=ln(2), lgamma(0.5)=ln(√π)≈0.5724
    let ln2 = std::f64::consts::LN_2;
    // ln(sqrt(pi)) = 0.5 * ln(pi)
    let ln_sqrt_pi = 0.5_f64 * std::f64::consts::PI.ln();
    let expected = [0.0_f64, 0.0, ln2, ln_sqrt_pi];

    gpu_test!(
        "test_lgamma_wgpu_batch",
        lgamma_batch_wgpu(&xs),
        |ys: Vec<f64>| {
            assert_eq!(ys.len(), xs.len());
            for (i, (y, e)) in ys.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (y - e).abs() < 1e-3,
                    "lgamma(xs[{}]={}) = {} (expected {})",
                    i,
                    xs[i],
                    y,
                    e
                );
            }
        }
    );
}

#[test]
fn test_erfc_wgpu_batch() {
    let xs = vec![0.0_f64, 1.0, -1.0, 2.0];
    // erfc(0)=1, erfc(1)≈0.15729921, erfc(-1)≈1.84270079, erfc(2)≈0.00467773
    let expected = [1.0_f64, 0.157_299_21, 1.842_700_79, 0.004_677_73];
    gpu_test!("test_erfc_wgpu_batch", erfc_batch_wgpu(&xs), |ys: Vec<
        f64,
    >| {
        assert_eq!(ys.len(), xs.len());
        for (i, (y, e)) in ys.iter().zip(expected.iter()).enumerate() {
            assert!(
                (y - e).abs() < 1e-3,
                "erfc(xs[{}]={}) = {} (expected {})",
                i,
                xs[i],
                y,
                e
            );
        }
    });
}

#[test]
fn test_erfinv_wgpu_batch() {
    let xs = vec![0.0_f64, 0.5, -0.5, 0.9];
    // erfinv(0)=0, erfinv(0.5)≈0.4769, erfinv(-0.5)≈-0.4769, erfinv(0.9)≈1.1631
    let expected = [0.0_f64, 0.476_936_276_2, -0.476_936_276_2, 1.163_087_154];
    gpu_test!(
        "test_erfinv_wgpu_batch",
        erfinv_batch_wgpu(&xs),
        |ys: Vec<f64>| {
            assert_eq!(ys.len(), xs.len());
            for (i, (y, e)) in ys.iter().zip(expected.iter()).enumerate() {
                // f32 shader → tolerate up to 0.01 absolute error
                assert!(
                    (y - e).abs() < 0.01,
                    "erfinv(xs[{}]={}) = {} (expected {})",
                    i,
                    xs[i],
                    y,
                    e
                );
            }
        }
    );
}

#[test]
fn test_empty_input() {
    let xs: Vec<f64> = vec![];
    // Empty input should always succeed (returns empty vec)
    match gamma_batch_wgpu(&xs) {
        Ok(ys) => assert!(ys.is_empty()),
        Err(WgslDispatchError::GpuNotAvailable) => {}
        Err(e) => panic!("unexpected error on empty input: {:?}", e),
    }
}
