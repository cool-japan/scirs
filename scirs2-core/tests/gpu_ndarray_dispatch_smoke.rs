// Copyright (c) 2025, `SciRS2` Team
//
// Licensed under the Apache License, Version 2.0
// (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0)
//

//! Smoke tests for `GpuNdarray<f32>` ArrayProtocol dispatch.
//!
//! All tests are gated on the `array_protocol_wgpu` feature.
//! On hosts without a GPU adapter, tests skip gracefully (they pass).

#[cfg(feature = "array_protocol_wgpu")]
mod gpu_ndarray_dispatch_smoke {
    use scirs2_core::array_protocol::{
        self as ap,
        gpu_ndarray::{is_gpu_available, GpuNdarray},
        ArrayProtocol,
    };

    /// Helper — try to construct a `GpuNdarray<f32>` from flat data + shape.
    /// Returns `None` if no GPU adapter is available (graceful skip).
    fn try_make(data: &[f32], shape: Vec<usize>) -> Option<GpuNdarray<f32>> {
        match GpuNdarray::<f32>::from_data(data, shape) {
            Ok(g) => Some(g),
            Err(e) => {
                let msg = e.to_string();
                if msg.contains("adapter")
                    || msg.contains("Adapter")
                    || msg.contains("GPU")
                    || msg.contains("no suitable")
                {
                    println!("No GPU adapter — skipping test ({msg})");
                    None
                } else {
                    panic!("Unexpected error creating GpuNdarray: {e}");
                }
            }
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    // Test 1: add_matches_cpu_or_skips
    // ──────────────────────────────────────────────────────────────────────

    /// Verifies that GPU add (dispatched via ArrayProtocol) produces values
    /// matching element-wise CPU addition, or skips when no adapter available.
    #[test]
    fn add_matches_cpu_or_skips() {
        // Use 4096 elements to exceed the GPU threshold
        const N: usize = 4096;
        let a_data: Vec<f32> = (0..N).map(|i| i as f32).collect();
        let b_data: Vec<f32> = (0..N).map(|i| (N - i) as f32).collect();
        let expected: Vec<f32> = a_data
            .iter()
            .zip(b_data.iter())
            .map(|(x, y)| x + y)
            .collect();

        let Some(a_gpu) = try_make(&a_data, vec![N]) else {
            return;
        };
        let Some(b_gpu) = try_make(&b_data, vec![N]) else {
            return;
        };

        let result = ap::add(&a_gpu, &b_gpu).expect("add failed");
        let result_gpu = result
            .as_any()
            .downcast_ref::<GpuNdarray<f32>>()
            .expect("result must be GpuNdarray<f32>");
        let got = result_gpu.to_vec().expect("readback failed");

        assert_eq!(got.len(), N, "result length mismatch");
        for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-4,
                "add mismatch at index {i}: got {g}, expected {e}"
            );
        }
        println!("add_matches_cpu_or_skips: passed with N={N}");
    }

    // ──────────────────────────────────────────────────────────────────────
    // Test 2: matmul_matches_cpu_or_skips
    // ──────────────────────────────────────────────────────────────────────

    /// Verifies GPU matmul produces C[i,j] = sum_k A[i,k]*B[k,j].
    #[test]
    fn matmul_matches_cpu_or_skips() {
        // 64×64 matrices → 4096 elements, triggers GPU
        const M: usize = 64;
        const K: usize = 64;
        const N: usize = 64;

        let a_data: Vec<f32> = (0..(M * K)).map(|i| i as f32 * 0.01).collect();
        let b_data: Vec<f32> = (0..(K * N)).map(|i| (i % 10) as f32).collect();

        // CPU reference
        let mut expected = vec![0.0f32; M * N];
        for i in 0..M {
            for j in 0..N {
                let mut s = 0.0f32;
                for k in 0..K {
                    s += a_data[i * K + k] * b_data[k * N + j];
                }
                expected[i * N + j] = s;
            }
        }

        let Some(a_gpu) = try_make(&a_data, vec![M, K]) else {
            return;
        };
        let Some(b_gpu) = try_make(&b_data, vec![K, N]) else {
            return;
        };

        let result = ap::matmul(&a_gpu, &b_gpu).expect("matmul failed");
        let result_gpu = result
            .as_any()
            .downcast_ref::<GpuNdarray<f32>>()
            .expect("result must be GpuNdarray<f32>");
        let got = result_gpu.to_vec().expect("readback failed");

        assert_eq!(got.len(), M * N, "matmul result length");
        for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).abs() < 0.1,
                "matmul mismatch at flat {i}: got {g}, expected {e}"
            );
        }
        println!("matmul_matches_cpu_or_skips: passed [{M}x{K}]x[{K}x{N}]");
    }

    // ──────────────────────────────────────────────────────────────────────
    // Test 3: sum_full_reduction
    // ──────────────────────────────────────────────────────────────────────

    /// Verifies `sum(axis=None)` returns the correct scalar total.
    #[test]
    fn sum_full_reduction() {
        const N: usize = 4096;
        let data: Vec<f32> = (0..N).map(|i| (i % 100) as f32).collect();
        let expected: f32 = data.iter().sum();

        let Some(g) = try_make(&data, vec![N]) else {
            return;
        };

        let result_box = ap::sum(&g, None).expect("sum failed");
        let got = result_box
            .downcast_ref::<f32>()
            .copied()
            .expect("sum must return f32 scalar");

        assert!(
            (got - expected).abs() < expected * 1e-3,
            "sum mismatch: got {got}, expected {expected}"
        );
        println!("sum_full_reduction: {got} ≈ {expected}");
    }

    // ──────────────────────────────────────────────────────────────────────
    // Test 4: transpose_2d
    // ──────────────────────────────────────────────────────────────────────

    /// Verifies GPU transpose of a 64×128 matrix produces a 128×64 result.
    #[test]
    fn transpose_2d() {
        const ROWS: usize = 64;
        const COLS: usize = 128;
        let data: Vec<f32> = (0..(ROWS * COLS)).map(|i| i as f32).collect();

        // CPU reference: T[j,i] = data[i*COLS + j]
        let mut expected = vec![0.0f32; COLS * ROWS];
        for i in 0..ROWS {
            for j in 0..COLS {
                expected[j * ROWS + i] = data[i * COLS + j];
            }
        }

        let Some(g) = try_make(&data, vec![ROWS, COLS]) else {
            return;
        };

        let result = ap::transpose(&g).expect("transpose failed");
        assert_eq!(result.shape(), &[COLS, ROWS], "transposed shape");

        let result_gpu = result
            .as_any()
            .downcast_ref::<GpuNdarray<f32>>()
            .expect("result must be GpuNdarray<f32>");
        let got = result_gpu.to_vec().expect("readback failed");

        assert_eq!(got.len(), COLS * ROWS, "transpose result length");
        for (i, (&g_val, &e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (g_val - e).abs() < 1e-4,
                "transpose mismatch at flat {i}: got {g_val}, expected {e}"
            );
        }
        println!("transpose_2d: [{ROWS}x{COLS}] → [{COLS}x{ROWS}] correct");
    }

    // ──────────────────────────────────────────────────────────────────────
    // Test 5: reshape_is_zero_copy
    // ──────────────────────────────────────────────────────────────────────

    /// Verifies that reshape returns an array sharing the same underlying buffer.
    #[test]
    fn reshape_is_zero_copy() {
        let data: Vec<f32> = (0..4096).map(|i| i as f32).collect();

        let Some(g) = try_make(&data, vec![64, 64]) else {
            return;
        };

        let reshaped = ap::reshape(&g, &[4096]).expect("reshape failed");
        let reshaped_gpu = reshaped
            .as_any()
            .downcast_ref::<GpuNdarray<f32>>()
            .expect("result must be GpuNdarray<f32>");

        // Zero-copy: both must point to the same wgpu::Buffer allocation
        assert!(
            std::sync::Arc::ptr_eq(g.buffer_arc(), reshaped_gpu.buffer_arc()),
            "reshape must reuse the same Arc<Buffer> (zero-copy)"
        );
        assert_eq!(reshaped_gpu.shape(), &[4096], "reshaped shape");
        println!("reshape_is_zero_copy: buffer ptr matches");
    }

    // ──────────────────────────────────────────────────────────────────────
    // Test 6: concatenate_axis0
    // ──────────────────────────────────────────────────────────────────────

    /// Verifies concatenation along axis=0 produces correct shape and data.
    #[test]
    fn concatenate_axis0() {
        // Two 32×64 matrices → 64×64 after concat on axis=0
        const ROWS: usize = 32;
        const COLS: usize = 64;
        let a_data: Vec<f32> = (0..(ROWS * COLS)).map(|i| i as f32).collect();
        let b_data: Vec<f32> = (0..(ROWS * COLS))
            .map(|i| (i + ROWS * COLS) as f32)
            .collect();

        let Some(a_gpu) = try_make(&a_data, vec![ROWS, COLS]) else {
            return;
        };
        let Some(b_gpu) = try_make(&b_data, vec![ROWS, COLS]) else {
            return;
        };

        let parts: Vec<&dyn ap::ArrayProtocol> = vec![&a_gpu, &b_gpu];
        let result = ap::concatenate(&parts, 0).expect("concatenate failed");

        assert_eq!(result.shape(), &[ROWS * 2, COLS], "concatenated shape");

        let result_gpu = result
            .as_any()
            .downcast_ref::<GpuNdarray<f32>>()
            .expect("result must be GpuNdarray<f32>");
        let got = result_gpu.to_vec().expect("readback failed");

        assert_eq!(got.len(), ROWS * 2 * COLS, "concatenate result length");
        // First half should match a_data
        for (i, (&g, &e)) in got[..ROWS * COLS].iter().zip(a_data.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-4,
                "concat a_data mismatch at {i}: got {g}, expected {e}"
            );
        }
        // Second half should match b_data
        for (i, (&g, &e)) in got[ROWS * COLS..].iter().zip(b_data.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-4,
                "concat b_data mismatch at {i}: got {g}, expected {e}"
            );
        }
        println!(
            "concatenate_axis0: [{ROWS}x{COLS}] × 2 → [{ROWSx2}x{COLS}]",
            ROWSx2 = ROWS * 2
        );
    }

    // ──────────────────────────────────────────────────────────────────────
    // Test 9: gpu_ndarray_concatenate_axis1_or_skips (Wave 77)
    // ──────────────────────────────────────────────────────────────────────

    /// Concatenates two [64, 128] matrices along axis=1 to get [64, 256].
    /// Uses shapes above GPU_THRESHOLD (8192 elements each) to exercise the GPU kernel.
    #[test]
    fn gpu_ndarray_concatenate_axis1_or_skips() {
        const ROWS: usize = 64;
        const COLS: usize = 128;
        let n = ROWS * COLS;
        let a_data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b_data: Vec<f32> = (0..n).map(|i| (i + n) as f32).collect();

        let Some(a_gpu) = try_make(&a_data, vec![ROWS, COLS]) else {
            return;
        };
        let Some(b_gpu) = try_make(&b_data, vec![ROWS, COLS]) else {
            return;
        };

        let parts: Vec<&dyn ap::ArrayProtocol> = vec![&a_gpu, &b_gpu];
        let result = ap::concatenate(&parts, 1).expect("concatenate axis=1 failed");

        assert_eq!(result.shape(), &[ROWS, COLS * 2], "concat axis=1 shape");

        let result_gpu = result
            .as_any()
            .downcast_ref::<GpuNdarray<f32>>()
            .expect("result must be GpuNdarray<f32>");
        let got = result_gpu.to_vec().expect("readback failed");
        assert_eq!(got.len(), ROWS * COLS * 2);

        // Verify: row r, col c: c < COLS → a_data[r*COLS+c], else b_data[r*COLS+(c-COLS)]
        for r in 0..ROWS {
            for c in 0..(COLS * 2) {
                let flat = r * (COLS * 2) + c;
                let expected = if c < COLS {
                    a_data[r * COLS + c]
                } else {
                    b_data[r * COLS + (c - COLS)]
                };
                assert!(
                    (got[flat] - expected).abs() < 1e-5,
                    "concat axis=1 mismatch at [{r},{c}]: got {}, expected {expected}",
                    got[flat]
                );
            }
        }
        println!(
            "gpu_ndarray_concatenate_axis1_or_skips: [{ROWS}x{COLS}]x2 → [{ROWS}x{}]",
            COLS * 2
        );
    }

    // ──────────────────────────────────────────────────────────────────────
    // Test 10: gpu_ndarray_sum_axis0_rank3_or_skips (Wave 77)
    // ──────────────────────────────────────────────────────────────────────

    /// Sum a [8, 64, 64] tensor along axis=0 → [64, 64].
    /// Input has 32768 elements, exceeding the GPU threshold.
    #[test]
    fn gpu_ndarray_sum_axis0_rank3_or_skips() {
        const D0: usize = 8;
        const D1: usize = 64;
        const D2: usize = 64;
        let n = D0 * D1 * D2;
        let data: Vec<f32> = (0..n).map(|i| (i % 17) as f32).collect();

        // CPU reference
        let mut expected = vec![0.0f32; D1 * D2];
        for i in 0..D0 {
            for j in 0..D1 {
                for k in 0..D2 {
                    expected[j * D2 + k] += data[i * D1 * D2 + j * D2 + k];
                }
            }
        }

        let Some(g) = try_make(&data, vec![D0, D1, D2]) else {
            return;
        };

        let result = ap::sum(&g, Some(0)).expect("sum axis=0 failed");
        let result_gpu = result
            .downcast_ref::<Box<dyn ap::ArrayProtocol>>()
            .expect("result must be Box<dyn ArrayProtocol>")
            .as_any()
            .downcast_ref::<GpuNdarray<f32>>()
            .expect("result must be GpuNdarray<f32>");

        assert_eq!(result_gpu.shape(), &[D1, D2], "sum axis=0 shape");
        let got = result_gpu.to_vec().expect("readback failed");
        assert_eq!(got.len(), D1 * D2);

        for (i, (&g_val, &e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (g_val - e).abs() < 1e-3,
                "sum axis=0 mismatch at {i}: got {g_val}, expected {e}"
            );
        }
        println!("gpu_ndarray_sum_axis0_rank3_or_skips: [{D0}x{D1}x{D2}] sum axis=0 → [{D1}x{D2}]");
    }

    // ──────────────────────────────────────────────────────────────────────
    // Test 11: gpu_ndarray_sum_axis2_rank3_or_skips (Wave 77)
    // ──────────────────────────────────────────────────────────────────────

    /// Sum a [8, 64, 64] tensor along axis=2 → [8, 64].
    /// Input has 32768 elements, exceeding the GPU threshold.
    #[test]
    fn gpu_ndarray_sum_axis2_rank3_or_skips() {
        const D0: usize = 8;
        const D1: usize = 64;
        const D2: usize = 64;
        let n = D0 * D1 * D2;
        let data: Vec<f32> = (0..n).map(|i| (i % 13) as f32).collect();

        // CPU reference
        let mut expected = vec![0.0f32; D0 * D1];
        for i in 0..D0 {
            for j in 0..D1 {
                for k in 0..D2 {
                    expected[i * D1 + j] += data[i * D1 * D2 + j * D2 + k];
                }
            }
        }

        let Some(g) = try_make(&data, vec![D0, D1, D2]) else {
            return;
        };

        let result = ap::sum(&g, Some(2)).expect("sum axis=2 failed");
        let result_gpu = result
            .downcast_ref::<Box<dyn ap::ArrayProtocol>>()
            .expect("result must be Box<dyn ArrayProtocol>")
            .as_any()
            .downcast_ref::<GpuNdarray<f32>>()
            .expect("result must be GpuNdarray<f32>");

        assert_eq!(result_gpu.shape(), &[D0, D1], "sum axis=2 shape");
        let got = result_gpu.to_vec().expect("readback failed");
        assert_eq!(got.len(), D0 * D1);

        for (i, (&g_val, &e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (g_val - e).abs() < 1e-3,
                "sum axis=2 mismatch at {i}: got {g_val}, expected {e}"
            );
        }
        println!("gpu_ndarray_sum_axis2_rank3_or_skips: [{D0}x{D1}x{D2}] sum axis=2 → [{D0}x{D1}]");
    }

    // ──────────────────────────────────────────────────────────────────────
    // Test 7: svd_falls_back_to_cpu
    // ──────────────────────────────────────────────────────────────────────

    /// SVD must always fall back to CPU (no GPU kernel). Verifies it
    /// returns three ArrayProtocol objects with correct shapes.
    #[test]
    fn svd_falls_back_to_cpu() {
        const M: usize = 4;
        const N: usize = 4;
        let data: Vec<f32> = (0..(M * N)).map(|i| (i + 1) as f32).collect();

        let Some(g) = try_make(&data, vec![M, N]) else {
            return;
        };

        let (u, s, vt) = ap::svd(&g).expect("svd failed");

        assert_eq!(u.shape(), &[M, M], "SVD U shape");
        assert_eq!(s.shape(), &[M.min(N)], "SVD S shape");
        assert_eq!(vt.shape(), &[N, N], "SVD Vt shape");
        println!(
            "svd_falls_back_to_cpu: shapes correct ({M}x{M}, {k}, {N}x{N})",
            k = M.min(N)
        );
    }

    // ──────────────────────────────────────────────────────────────────────
    // Test 8: dispatch_below_threshold_uses_cpu
    // ──────────────────────────────────────────────────────────────────────

    /// Arrays with fewer than 4096 elements skip GPU dispatch and compute
    /// on CPU (no GPU round-trip).  Verifies numerical correctness.
    #[test]
    fn dispatch_below_threshold_uses_cpu() {
        // 16 elements — well below GPU_THRESHOLD (4096)
        const N: usize = 16;
        let a_data: Vec<f32> = (0..N).map(|i| i as f32).collect();
        let b_data: Vec<f32> = (0..N).map(|i| (N - i) as f32).collect();
        let expected: Vec<f32> = a_data
            .iter()
            .zip(b_data.iter())
            .map(|(x, y)| x + y)
            .collect();

        let Some(a_gpu) = try_make(&a_data, vec![N]) else {
            return;
        };
        let Some(b_gpu) = try_make(&b_data, vec![N]) else {
            return;
        };

        let result = ap::add(&a_gpu, &b_gpu).expect("add (below threshold) failed");
        let result_gpu = result
            .as_any()
            .downcast_ref::<GpuNdarray<f32>>()
            .expect("result must be GpuNdarray<f32>");
        let got = result_gpu.to_vec().expect("readback failed");

        for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-5,
                "below-threshold add mismatch at {i}: got {g}, expected {e}"
            );
        }
        println!("dispatch_below_threshold_uses_cpu: N={N} add correct (CPU path)");

        // Confirm GPU availability flag is consistent
        let gpu_flag = is_gpu_available();
        println!("GPU available: {gpu_flag}");
    }
}
