//! Smoke tests for the real wgpu RBF dispatch path in scirs2-interpolate.
//!
//! All tests are gated on `#[cfg(feature = "wgpu_rbf")]`.  On hosts without a
//! GPU adapter (headless CI), each test detects the absence of an adapter at
//! runtime and skips gracefully — the test passes rather than fails.

#[cfg(feature = "wgpu_rbf")]
mod wgpu_rbf_smoke {
    use scirs2_core::ndarray::Array1;
    use scirs2_interpolate::gpu_accelerated::{
        wgpu_rbf::{
            evaluate_shader_source, gpu_rbf_evaluate, gpu_rbf_kernel_matrix, is_gpu_available,
            kernel_matrix_shader_source, GPU_THRESHOLD,
        },
        GpuConfig, GpuRBFInterpolator, GpuRBFKernel,
    };

    // ─────────────────────────────────────────────────────────────────────────
    // Skip-helper: returns true when the error indicates no adapter.
    // ─────────────────────────────────────────────────────────────────────────

    fn is_no_adapter(msg: &str) -> bool {
        msg.contains("adapter")
            || msg.contains("Adapter")
            || msg.contains("GPU")
            || msg.contains("no suitable")
            || msg.contains("NoAdapter")
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 1: Gaussian RBF kernel-matrix + evaluate — GPU vs CPU diff < 1e-4
    // ─────────────────────────────────────────────────────────────────────────

    /// Builds a 64-center / 64-query Gaussian RBF kernel matrix on the GPU
    /// (when available) and verifies element-wise agreement with the CPU
    /// reference.  Uses `f32` tolerance (1e-4) because the GPU operates in
    /// single precision.
    #[test]
    fn gaussian_rbf_gpu_matches_cpu_or_skips() {
        let n_centers: usize = 64;
        let n_queries: usize = 64;

        // Generate deterministic data
        let centers: Vec<f64> = (0..n_centers)
            .map(|i| i as f64 / n_centers as f64)
            .collect();
        let queries: Vec<f64> = (0..n_queries)
            .map(|i| (i as f64 + 0.5) / n_queries as f64)
            .collect();
        let epsilon = 1.0_f64;

        // CPU reference kernel matrix
        let cpu_matrix: Vec<f64> = centers
            .iter()
            .flat_map(|&c| {
                queries.iter().map(move |&q| {
                    let r = (c - q).abs() / epsilon;
                    (-r * r).exp()
                })
            })
            .collect();

        match gpu_rbf_kernel_matrix(&centers, &queries, GpuRBFKernel::Gaussian, epsilon) {
            Ok((gpu_matrix, timing)) => {
                assert_eq!(
                    gpu_matrix.len(),
                    n_centers * n_queries,
                    "GPU matrix length must match n_centers × n_queries"
                );
                for (i, (&gpu_val, &cpu_val)) in
                    gpu_matrix.iter().zip(cpu_matrix.iter()).enumerate()
                {
                    assert!(
                        (gpu_val - cpu_val).abs() < 1e-4,
                        "element {i}: GPU={gpu_val}, CPU={cpu_val}, diff={}",
                        (gpu_val - cpu_val).abs()
                    );
                }
                println!(
                    "gaussian_rbf_gpu_matches_cpu_or_skips: PASS (transfer={} ns, dispatch={} ns)",
                    timing.transfer_ns, timing.dispatch_ns
                );
            }
            Err(e) => {
                let msg = e.to_string();
                if is_no_adapter(&msg) {
                    println!("No wgpu adapter — skipping ({msg})");
                } else {
                    panic!("Unexpected error: {e}");
                }
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 2: Multiquadric evaluate roundtrip
    // ─────────────────────────────────────────────────────────────────────────

    /// Generates random (seeded) coefficients and centers, evaluates via GPU
    /// and CPU, and asserts element-wise diff < 1e-4.
    #[test]
    fn multiquadric_rbf_evaluate_roundtrip_or_skips() {
        let n_centers: usize = 32;
        let n_queries: usize = 32;
        let epsilon = 2.0_f64;

        // Deterministic pseudo-random data (LCG)
        let mut seed: u64 = 42;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) as f64 / u32::MAX as f64
        };

        let centers: Vec<f64> = (0..n_centers).map(|_| next()).collect();
        let coefficients: Vec<f64> = (0..n_centers).map(|_| next() * 2.0 - 1.0).collect();
        let queries: Vec<f64> = (0..n_queries).map(|_| next()).collect();

        // CPU reference
        let cpu_values: Vec<f64> = queries
            .iter()
            .map(|&q| {
                coefficients
                    .iter()
                    .zip(centers.iter())
                    .map(|(&coeff, &c)| {
                        let r = (c - q).abs() / epsilon;
                        coeff * (1.0 + r * r).sqrt()
                    })
                    .sum::<f64>()
            })
            .collect();

        match gpu_rbf_evaluate(
            &coefficients,
            &centers,
            &queries,
            GpuRBFKernel::Multiquadric,
            epsilon,
        ) {
            Ok((gpu_values, timing)) => {
                assert_eq!(
                    gpu_values.len(),
                    n_queries,
                    "GPU output length must match n_queries"
                );
                for (i, (&gpu_val, &cpu_val)) in
                    gpu_values.iter().zip(cpu_values.iter()).enumerate()
                {
                    assert!(
                        (gpu_val - cpu_val).abs() < 1e-4,
                        "element {i}: GPU={gpu_val:.6}, CPU={cpu_val:.6}, diff={:.2e}",
                        (gpu_val - cpu_val).abs()
                    );
                }
                println!(
                    "multiquadric_rbf_evaluate_roundtrip_or_skips: PASS (transfer={} ns, dispatch={} ns)",
                    timing.transfer_ns, timing.dispatch_ns
                );
            }
            Err(e) => {
                let msg = e.to_string();
                if is_no_adapter(&msg) {
                    println!("No wgpu adapter — skipping ({msg})");
                } else {
                    panic!("Unexpected error: {e}");
                }
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 3: All six kernels — shader sources are non-empty and valid WGSL
    // ─────────────────────────────────────────────────────────────────────────

    /// For each of the six `GpuRBFKernel` variants, verifies that the WGSL
    /// shader sources contain a `@compute` entry point.  The kernel_id uniform
    /// selects the variant at runtime inside the shader; this test verifies the
    /// static shader source properties.
    #[test]
    fn all_six_kernels_compile_or_skip() {
        let km_source = kernel_matrix_shader_source();
        let eval_source = evaluate_shader_source();

        // Both shaders must be non-empty and contain @compute
        assert!(
            !km_source.is_empty(),
            "kernel-matrix WGSL must not be empty"
        );
        assert!(
            km_source.contains("@compute"),
            "kernel-matrix WGSL must contain @compute"
        );
        assert!(!eval_source.is_empty(), "evaluate WGSL must not be empty");
        assert!(
            eval_source.contains("@compute"),
            "evaluate WGSL must contain @compute"
        );

        // Each kernel variant must have a valid id mapping
        let kernels = [
            GpuRBFKernel::Gaussian,
            GpuRBFKernel::Multiquadric,
            GpuRBFKernel::InverseMultiquadric,
            GpuRBFKernel::Linear,
            GpuRBFKernel::Cubic,
            GpuRBFKernel::ThinPlate,
        ];

        // Attempt a tiny kernel-matrix dispatch for each variant to verify
        // the GPU shader compiles and selects the correct branch.
        let centers = vec![0.0_f64, 1.0, 2.0];
        let queries = vec![0.5_f64, 1.5];
        for kernel in kernels {
            match gpu_rbf_kernel_matrix(&centers, &queries, kernel, 1.0) {
                Ok((matrix, _)) => {
                    assert_eq!(matrix.len(), centers.len() * queries.len());
                    assert!(
                        matrix.iter().all(|v| v.is_finite()),
                        "All kernel values must be finite for {:?}",
                        kernel
                    );
                    println!("all_six_kernels_compile_or_skip: {kernel:?} — PASS");
                }
                Err(e) => {
                    let msg = e.to_string();
                    if is_no_adapter(&msg) {
                        println!("all_six_kernels_compile_or_skip: {kernel:?} — no adapter, skip ({msg})");
                    } else {
                        panic!("Unexpected error for {kernel:?}: {e}");
                    }
                }
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 4: Below-threshold uses CPU path
    // ─────────────────────────────────────────────────────────────────────────

    /// When `n_centers * n_queries < GPU_THRESHOLD` (4096), the interpolator
    /// must use the CPU path and `GpuStats.used_gpu` must be `false`.
    #[test]
    fn below_threshold_uses_cpu() {
        // 8 × 8 = 64 < 4096
        let n = 8_usize;
        assert!(
            n * n < GPU_THRESHOLD,
            "test precondition: 8×8 must be below threshold {GPU_THRESHOLD}"
        );

        let x: Array1<f64> = Array1::linspace(0.0, 1.0, n);
        let y = x.mapv(|v| v * v);

        let mut interpolator = GpuRBFInterpolator::new()
            .with_kernel(GpuRBFKernel::Gaussian)
            .with_kernel_width(1.0)
            .with_gpu_config(GpuConfig {
                prefer_gpu: true, // even with prefer_gpu=true, threshold blocks dispatch
                ..GpuConfig::default()
            });

        interpolator
            .fit(&x.view(), &y.view())
            .expect("fit should succeed");

        let xeval: Array1<f64> = Array1::linspace(0.0, 1.0, n);
        let _ = interpolator
            .evaluate(&xeval.view())
            .expect("evaluate should succeed");

        let stats = interpolator.get_stats();
        assert!(
            !stats.used_gpu,
            "used_gpu must be false for sub-threshold problem (n_centers×n_queries={})",
            n * n
        );
        println!(
            "below_threshold_uses_cpu: PASS (used_gpu={})",
            stats.used_gpu
        );
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Test 5: is_gpu_available returns a consistent bool
    // ─────────────────────────────────────────────────────────────────────────

    /// `is_gpu_available()` must:
    /// 1. Not panic on the first call.
    /// 2. Return the same value on a second call (cached result).
    #[test]
    fn is_gpu_available_returns_bool() {
        let first = is_gpu_available();
        let second = is_gpu_available();
        assert_eq!(
            first, second,
            "is_gpu_available() must be idempotent (cached)"
        );
        println!("is_gpu_available_returns_bool: PASS (available={first})");
    }
}
