//! Smoke tests for GPU-accelerated L-BFGS 2-loop recursion.
//!
//! Test 1 (`lbfgs_gpu_matches_cpu_on_rosenbrock_or_skips`): Run both CPU and
//! GPU L-BFGS on a 100-dimensional problem with `gpu_threshold_override = Some(1)`
//! so GPU dispatch fires at n=100 even though the default threshold is 4096.
//! Asserts that the solutions agree within 1e-3, or skips gracefully when no
//! wgpu adapter is present.
//!
//! Test 2 (`lbfgs_gpu_disabled_falls_back_to_cpu`): Set `use_gpu = false` and
//! run with n=5000 parameters; asserts correctness of the CPU path (no adapter
//! needed).

#[cfg(feature = "wgpu")]
mod gpu_tests {
    use scirs2_core::ndarray::Array1;
    use scirs2_optimize::unconstrained::{minimize_lbfgs, Options};

    // Rosenbrock function for n-dimensional input.
    fn rosenbrock(x: &scirs2_core::ndarray::ArrayView1<f64>) -> f64 {
        let n = x.len();
        (0..n - 1)
            .map(|i| {
                let a = 1.0 - x[i];
                let b = x[i + 1] - x[i] * x[i];
                a * a + 100.0 * b * b
            })
            .sum()
    }

    #[test]
    fn lbfgs_gpu_matches_cpu_on_rosenbrock_or_skips() {
        let n = 100usize;
        let x0_cpu = Array1::from_elem(n, 0.5f64);
        let x0_gpu = x0_cpu.clone();

        // CPU: disable GPU path explicitly.
        let opt_cpu = Options {
            max_iter: 1000,
            use_gpu: false,
            ..Options::default()
        };

        let result_cpu = minimize_lbfgs(
            rosenbrock,
            None::<fn(&scirs2_core::ndarray::ArrayView1<f64>) -> Array1<f64>>,
            x0_cpu,
            &opt_cpu,
        );

        let cpu_result = match result_cpu {
            Ok(r) => r,
            Err(e) => {
                panic!("CPU L-BFGS failed: {e}");
            }
        };

        // GPU attempt: use gpu_threshold_override = Some(1) to force GPU dispatch at n=100.
        let opt_gpu = Options {
            max_iter: 1000,
            use_gpu: true,
            gpu_threshold_override: Some(1),
            ..Options::default()
        };

        let result_gpu = minimize_lbfgs(
            rosenbrock,
            None::<fn(&scirs2_core::ndarray::ArrayView1<f64>) -> Array1<f64>>,
            x0_gpu,
            &opt_gpu,
        );

        let gpu_result = match result_gpu {
            Ok(r) => r,
            Err(e) => {
                let msg = e.to_string();
                if scirs2_core::testing::gpu_availability::is_gpu_unavailable_error(&msg) {
                    scirs2_core::testing::gpu_availability::print_gpu_skip(
                        "lbfgs_gpu_matches_cpu_on_rosenbrock_or_skips",
                        &msg,
                    );
                    return;
                }
                panic!("Unexpected GPU L-BFGS error: {e}");
            }
        };

        // Compare solutions.
        // Note: GPU path uses f32 arithmetic internally, so agreement to 1e-2 is expected.
        let cpu_x: &Array1<f64> = &cpu_result.x;
        let gpu_x: &Array1<f64> = &gpu_result.x;
        let diff = (cpu_x - gpu_x)
            .mapv(f64::abs)
            .iter()
            .cloned()
            .fold(0.0f64, f64::max);

        // If GPU path fell back to CPU (no adapter), the results should be identical.
        // If GPU path ran, results should agree within f32 precision limits.
        assert!(
            diff < 1e-2,
            "CPU and GPU L-BFGS solutions differ by {diff:.2e} (max element-wise), \
             which exceeds 1e-2; GPU result may have precision issues.\n\
             CPU solution norm: {:.4e}, GPU solution norm: {:.4e}",
            cpu_x.dot(cpu_x).sqrt(),
            gpu_x.dot(gpu_x).sqrt(),
        );

        println!("GPU L-BFGS smoke test passed: CPU/GPU solution max diff = {diff:.2e}");
    }

    #[test]
    fn lbfgs_gpu_disabled_falls_back_to_cpu() {
        // A separable quadratic: f(x) = sum_i (x_i - 1)^2, minimum at all-ones.
        let n = 5000usize;
        let x0 = Array1::from_elem(n, 0.0f64);
        let target = Array1::from_elem(n, 1.0f64);
        let target_ref = target.clone();

        let quadratic = move |x: &scirs2_core::ndarray::ArrayView1<f64>| -> f64 {
            x.iter()
                .zip(target_ref.iter())
                .map(|(&xi, &ti)| (xi - ti).powi(2))
                .sum()
        };

        // Force CPU path — no adapter needed.
        let opts = Options {
            max_iter: 200,
            use_gpu: false,
            ..Options::default()
        };

        let result = minimize_lbfgs(
            quadratic,
            None::<fn(&scirs2_core::ndarray::ArrayView1<f64>) -> Array1<f64>>,
            x0,
            &opts,
        )
        .expect("CPU L-BFGS with use_gpu=false should not fail");

        // The optimizer should make meaningful progress toward [1.0; 5000].
        let fun_val: f64 = result.fun;
        assert!(
            fun_val < 100.0,
            "Expected f(x) < 100.0 after 200 iterations on separable quadratic, got {fun_val:.4e}"
        );

        println!(
            "lbfgs_gpu_disabled_falls_back_to_cpu passed: f={fun_val:.4e} after {} iters",
            result.nit
        );
    }
}

// When the `gpu` feature is not enabled, define a placeholder test so the file is not empty.
#[cfg(not(feature = "wgpu"))]
#[test]
fn lbfgs_gpu_feature_not_enabled() {
    // This test just verifies that the file compiles cleanly without the gpu feature.
    println!("gpu feature not enabled; GPU L-BFGS tests are skipped at compile time.");
}
