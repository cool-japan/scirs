// Copyright (c) 2025, `SciRS2` Team
//
// Licensed under the Apache License, Version 2.0
// (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0)
//

//! Value-correctness tests for `svd`/`inverse` dispatched through
//! `GpuNdarray<f32>` — split out of `gpu_ndarray.rs` to keep it under the
//! workspace's 2000-line-per-file limit (see `neural.rs`/
//! `neural_backward_tests.rs` for the same pattern).
//!
//! `svd`/`inverse` dispatched through `GpuNdarray<f32>` used to silently
//! return a fabricated identity/ones "decomposition" for any input (see
//! `gpu_ndarray.rs`'s module-level "CPU-fallback operations" doc, and
//! `array_function`'s own former "placeholder" comments). The existing
//! GPU-adapter-gated smoke test
//! (`tests/gpu_ndarray_dispatch_smoke.rs::svd_falls_back_to_cpu`) only
//! asserts output *shapes*, so it could never have caught that. These tests
//! replay the same reconstruction-property checks already used for the
//! plain-`NdarrayWrapper` CPU path
//! (`operations::tests::test_svd_reconstructs_original_matrix` /
//! `test_inverse_is_a_real_inverse`) against the real dispatch path here,
//! using non-constant, non-orthogonal/non-identity data so a fabricated
//! placeholder provably fails. Both gracefully skip (pass trivially) on
//! hosts without a GPU adapter, matching this crate's established
//! convention for `wgpu`-gated tests.

use super::*;
use crate::array_protocol::{inverse, svd};

/// Uploads `data` to the GPU, or returns `None` if no adapter is available
/// (graceful skip).
fn try_make(data: &[f32], shape: Vec<usize>) -> Option<GpuNdarray<f32>> {
    if !is_gpu_available() {
        return None;
    }
    GpuNdarray::<f32>::from_data(data, shape).ok()
}

/// Downcasts a dispatch result to `GpuNdarray<f32>` and downloads it as an
/// owned 2-D array.
fn download_2d(result: &dyn ArrayProtocol) -> ndarray::Array2<f32> {
    result
        .as_any()
        .downcast_ref::<GpuNdarray<f32>>()
        .expect("dispatch result should be GpuNdarray<f32>")
        .to_ndarray()
        .expect("GPU readback should succeed")
        .into_dimensionality::<ndarray::Ix2>()
        .expect("result should be 2-D")
}

/// Downcasts a dispatch result to `GpuNdarray<f32>` and downloads it as an
/// owned 1-D array.
fn download_1d(result: &dyn ArrayProtocol) -> ndarray::Array1<f32> {
    result
        .as_any()
        .downcast_ref::<GpuNdarray<f32>>()
        .expect("dispatch result should be GpuNdarray<f32>")
        .to_ndarray()
        .expect("GPU readback should succeed")
        .into_dimensionality::<ndarray::Ix1>()
        .expect("result should be 1-D")
}

#[test]
fn svd_dispatch_reconstructs_original_matrix_or_skips() {
    // Non-square, non-constant, non-orthogonal: an identity/ones
    // fabrication can never satisfy the reconstruction check below.
    const M: usize = 4;
    const N: usize = 3;
    let data: Vec<f32> = (0..(M * N))
        .map(|i| ((i + 1) as f32) * 1.3 - ((i % 3) as f32) * 0.7)
        .collect();

    let Some(g) = try_make(&data, vec![M, N]) else {
        println!("No GPU adapter — skipping svd_dispatch_reconstructs_original_matrix_or_skips");
        return;
    };

    let (u, s, vt) = svd(&g).expect("GPU-dispatched svd should succeed");
    let s = download_1d(s.as_ref());
    // The old placeholder returned `s` as all-ones regardless of input.
    assert!(
        s.iter().any(|&v| (v - 1.0).abs() > 1e-4),
        "singular values look like the old all-ones placeholder: {s:?}"
    );

    let u = download_2d(u.as_ref());
    let vt = download_2d(vt.as_ref());

    let mut sigma = ndarray::Array2::<f32>::zeros((u.ncols(), vt.nrows()));
    for (i, &sv) in s.iter().enumerate() {
        sigma[[i, i]] = sv;
    }
    let reconstructed = u.dot(&sigma).dot(&vt);

    let original =
        ndarray::Array2::from_shape_vec((M, N), data).expect("data should reshape to (M, N)");
    for i in 0..M {
        for j in 0..N {
            assert!(
                (reconstructed[[i, j]] - original[[i, j]]).abs() < 1e-3,
                "GPU-dispatched SVD reconstruction mismatch at ({i},{j}): {got} vs {expected}",
                got = reconstructed[[i, j]],
                expected = original[[i, j]]
            );
        }
    }
}

#[test]
fn inverse_dispatch_is_a_real_inverse_or_skips() {
    // Non-symmetric, well-conditioned, non-identity invertible matrix.
    const N: usize = 3;
    let data: Vec<f32> = vec![4.0, 3.0, 0.0, 2.0, 5.0, 1.0, 1.0, 0.0, 3.0];

    let Some(g) = try_make(&data, vec![N, N]) else {
        println!("No GPU adapter — skipping inverse_dispatch_is_a_real_inverse_or_skips");
        return;
    };

    let inv = inverse(&g).expect("GPU-dispatched inverse should succeed");
    let inv = download_2d(inv.as_ref());

    // The old placeholder always returned the identity matrix outright.
    // Assert the real defining property (A @ inv(A) == I) rather than
    // merely a result of the right shape, which a fabricated identity would
    // also trivially have.
    let original =
        ndarray::Array2::from_shape_vec((N, N), data).expect("data should reshape to (N, N)");
    let product = original.dot(&inv);
    for i in 0..N {
        for j in 0..N {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (product[[i, j]] - expected).abs() < 1e-3,
                "GPU-dispatched inverse's A*inv(A) mismatch at ({i},{j}): {got} vs {expected}",
                got = product[[i, j]]
            );
        }
    }
}
