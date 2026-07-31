//! Smoke tests for GPU-accelerated Conjugate Gradient and Newton-CG optimizers.
//!
//! Test 1 (`cg_gpu_matches_cpu_on_quadratic_or_skips`): Run both CPU and GPU
//! CG on a 100-dimensional quadratic with `gpu_threshold_override = Some(1)`
//! so GPU dispatch fires at n=100 even though the default threshold is 4096.
//! Asserts solutions agree within 1e-3, or skips gracefully when no wgpu adapter
//! is present.
//!
//! Test 2 (`newton_gpu_matches_cpu_or_skips`): Same quadratic, Newton-CG CPU vs
//! GPU, same skip pattern.
//!
//! Test 3 (`cg_gpu_disabled_fallsback_to_cpu`): Set `use_gpu = false`, run a
//! 1000-D separable quadratic and assert correctness — no GPU adapter needed.

#[cfg(feature = "wgpu")]
mod gpu_tests {
    use scirs2_core::ndarray::{Array1, ArrayView1};
    use scirs2_optimize::unconstrained::{
        minimize_conjugate_gradient, minimize_newton_cg, Options,
    };

    /// 100-D separable quadratic: f(x) = sum_i (x_i - 1)^2, minimum at all-ones.
    fn quadratic_100d(x: &ArrayView1<f64>) -> f64 {
        x.iter().map(|&xi| (xi - 1.0).powi(2)).sum()
    }

    #[test]
    fn cg_gpu_matches_cpu_on_quadratic_or_skips() {
        let n = 100usize;
        let x0_cpu = Array1::from_elem(n, 0.0f64);
        let x0_gpu = x0_cpu.clone();

        // CPU: disable GPU path explicitly.
        let opt_cpu = Options {
            max_iter: 1000,
            use_gpu: false,
            ..Options::default()
        };

        let result_cpu = minimize_conjugate_gradient(
            quadratic_100d,
            None::<fn(&ArrayView1<f64>) -> Array1<f64>>,
            x0_cpu,
            &opt_cpu,
        );

        let cpu_result = match result_cpu {
            Ok(r) => r,
            Err(e) => {
                panic!("CPU CG failed: {e}");
            }
        };

        // GPU attempt: use gpu_threshold_override = Some(1) to force GPU dispatch at n=100.
        let opt_gpu = Options {
            max_iter: 1000,
            use_gpu: true,
            gpu_threshold_override: Some(1),
            ..Options::default()
        };

        let result_gpu = minimize_conjugate_gradient(
            quadratic_100d,
            None::<fn(&ArrayView1<f64>) -> Array1<f64>>,
            x0_gpu,
            &opt_gpu,
        );

        let gpu_result = match result_gpu {
            Ok(r) => r,
            Err(e) => {
                // If the error is adapter-related, skip gracefully.
                let msg = e.to_string();
                if scirs2_core::testing::gpu_availability::is_gpu_unavailable_error(&msg) {
                    scirs2_core::testing::gpu_availability::print_gpu_skip(
                        "cg_gpu_matches_cpu_on_quadratic_or_skips",
                        &msg,
                    );
                    return;
                }
                panic!("Unexpected GPU CG error: {e}");
            }
        };

        // Compare solutions (GPU uses f32 internally, so 1e-3 is the target tolerance).
        let cpu_x: &Array1<f64> = &cpu_result.x;
        let gpu_x: &Array1<f64> = &gpu_result.x;
        let diff = (cpu_x - gpu_x)
            .mapv(f64::abs)
            .iter()
            .cloned()
            .fold(0.0f64, f64::max);

        let cpu_fun = cpu_result.fun;
        let gpu_fun = gpu_result.fun;
        assert!(
            diff < 1e-3,
            "CPU and GPU CG solutions differ by {diff:.2e} (max element-wise), exceeds 1e-3.\n\
             CPU f={cpu_fun:.4e}, GPU f={gpu_fun:.4e}",
        );

        println!("cg_gpu_matches_cpu_on_quadratic_or_skips passed: max diff = {diff:.2e}");
    }

    #[test]
    fn newton_gpu_matches_cpu_or_skips() {
        let n = 100usize;
        let x0_cpu = Array1::from_elem(n, 0.0f64);
        let x0_gpu = x0_cpu.clone();

        // CPU: disable GPU path explicitly.
        let opt_cpu = Options {
            max_iter: 50,
            use_gpu: false,
            ..Options::default()
        };

        let result_cpu = minimize_newton_cg(quadratic_100d, x0_cpu, &opt_cpu);

        let cpu_result = match result_cpu {
            Ok(r) => r,
            Err(e) => {
                panic!("CPU Newton-CG failed: {e}");
            }
        };

        // GPU attempt: use gpu_threshold_override = Some(1) to force GPU dispatch at n=100.
        let opt_gpu = Options {
            max_iter: 50,
            use_gpu: true,
            gpu_threshold_override: Some(1),
            ..Options::default()
        };

        let result_gpu = minimize_newton_cg(quadratic_100d, x0_gpu, &opt_gpu);

        let gpu_result = match result_gpu {
            Ok(r) => r,
            Err(e) => {
                let msg = e.to_string();
                if scirs2_core::testing::gpu_availability::is_gpu_unavailable_error(&msg) {
                    scirs2_core::testing::gpu_availability::print_gpu_skip(
                        "newton_gpu_matches_cpu_or_skips",
                        &msg,
                    );
                    return;
                }
                panic!("Unexpected GPU Newton-CG error: {e}");
            }
        };

        // Compare solutions.
        let cpu_x: &Array1<f64> = &cpu_result.x;
        let gpu_x: &Array1<f64> = &gpu_result.x;
        let diff = (cpu_x - gpu_x)
            .mapv(f64::abs)
            .iter()
            .cloned()
            .fold(0.0f64, f64::max);

        let cpu_fun = cpu_result.fun;
        let gpu_fun = gpu_result.fun;
        assert!(
            diff < 1e-3,
            "CPU and GPU Newton-CG solutions differ by {diff:.2e} (max element-wise), exceeds 1e-3.\n\
             CPU f={cpu_fun:.4e}, GPU f={gpu_fun:.4e}",
        );

        println!("newton_gpu_matches_cpu_or_skips passed: max diff = {diff:.2e}");
    }

    #[test]
    fn cg_gpu_disabled_fallsback_to_cpu() {
        // 1000-D separable quadratic: f(x) = sum_i (x_i - 1)^2, minimum at all-ones.
        let n = 1000usize;
        let x0 = Array1::from_elem(n, 0.0f64);

        let quadratic_1000d =
            |x: &ArrayView1<f64>| -> f64 { x.iter().map(|&xi| (xi - 1.0).powi(2)).sum() };

        // Force CPU path — no adapter needed.
        let opts = Options {
            max_iter: 500,
            use_gpu: false,
            ..Options::default()
        };

        let result = minimize_conjugate_gradient(
            quadratic_1000d,
            None::<fn(&ArrayView1<f64>) -> Array1<f64>>,
            x0,
            &opts,
        )
        .expect("CPU CG with use_gpu=false should not fail");

        let fun_val = result.fun;
        assert!(
            fun_val < 10.0,
            "Expected f(x) < 10.0 on 1000-D quadratic with CPU CG, got {fun_val:.4e}"
        );

        println!(
            "cg_gpu_disabled_fallsback_to_cpu passed: f={fun_val:.4e} after {} iters",
            result.nit
        );
    }
}

// When the `gpu` feature is not enabled, define a placeholder test so the file is not empty.
#[cfg(not(feature = "wgpu"))]
#[test]
fn cg_newton_gpu_feature_not_enabled() {
    println!("gpu feature not enabled; CG and Newton GPU tests are skipped at compile time.");
}
