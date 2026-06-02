//! Smoke tests for `GpuKernel::eval_batch` end-to-end wgpu dispatch.
//!
//! All tests inside the `gpu` cfg block require the crate-level `gpu` feature.
//! On hosts without a wgpu adapter (headless CI, virtual machines) each test
//! detects the missing adapter at runtime, prints a diagnostic, and passes.

#[cfg(feature = "gpu")]
mod gpu_eval_smoke {
    use scirs2_symbolic::compile::gpu::JitDispatch;
    use scirs2_symbolic::compile::gpu::{GpuError, GpuKernel, GPU_DISPATCH_THRESHOLD};
    use scirs2_symbolic::compile::{to_gpu, to_jit_auto};
    use scirs2_symbolic::eml::op::LoweredOp;

    /// Helper: recognise adapter-absent errors and skip gracefully.
    fn is_no_adapter(msg: &str) -> bool {
        msg.contains("adapter")
            || msg.contains("Adapter")
            || msg.contains("GPU")
            || msg.contains("no suitable")
            || msg.contains("NoAdapter")
    }

    // ── Test 1: linear formula f(x) = 2x + 1 ────────────────────────────────

    /// Build a `GpuKernel` for `f(x) = 2*x + 1` and evaluate 4 rows.
    /// Expected results: `[1.0, 3.0, 5.0, 7.0]` (f32 tolerance 1e-4).
    /// Skips gracefully when no wgpu adapter is available.
    #[test]
    fn eval_batch_constant_kernel_or_skips() {
        // f(x) = 2*x + 1
        let x = LoweredOp::Var(0);
        let two_x = LoweredOp::Mul(Box::new(LoweredOp::Const(2.0)), Box::new(x));
        let op = LoweredOp::Add(Box::new(two_x), Box::new(LoweredOp::Const(1.0)));

        let kernel: GpuKernel = to_gpu(&op).expect("WGSL shader generation must not fail");

        // 4 rows, 1 variable each
        let inputs: Vec<Vec<f64>> = (0u32..4).map(|i| vec![i as f64]).collect();

        match kernel.eval_batch(&inputs) {
            Ok(out) => {
                let expected = [1.0_f64, 3.0, 5.0, 7.0];
                assert_eq!(out.len(), 4, "output length must equal row count");
                for (i, (&got, &exp)) in out.iter().zip(expected.iter()).enumerate() {
                    assert!(
                        (got - exp).abs() < 1e-4,
                        "row {i}: got {got}, expected {exp} (f32 tolerance 1e-4)"
                    );
                }
                println!("eval_batch_constant_kernel_or_skips: GPU dispatch verified for 4 rows");
            }
            Err(e) => {
                let msg = e.to_string();
                if is_no_adapter(&msg) {
                    println!("No wgpu adapter available — skipping ({msg})");
                } else {
                    panic!("Unexpected error: {e}");
                }
            }
        }
    }

    // ── Test 2: transcendental formula f(x) = sin(x) + cos(x) ───────────────

    /// Build a `GpuKernel` for `f(x) = sin(x) + cos(x)` and evaluate 128 rows
    /// uniformly spaced in `[0, 2π]`. GPU vs CPU `eval_real` within 1e-3
    /// (f32 transcendental precision).
    /// Skips gracefully when no wgpu adapter is available.
    #[test]
    fn eval_batch_transcendental_kernel_or_skips() {
        use scirs2_symbolic::eml::eval::eval_real;
        use std::f64::consts::TAU;

        // f(x) = sin(x) + cos(x)
        let x = LoweredOp::Var(0);
        let sin_x = LoweredOp::Sin(Box::new(x.clone()));
        let cos_x = LoweredOp::Cos(Box::new(x));
        let op = LoweredOp::Add(Box::new(sin_x), Box::new(cos_x));

        let kernel: GpuKernel = to_gpu(&op).expect("WGSL shader generation must not fail");

        const N: usize = 128;
        let inputs: Vec<Vec<f64>> = (0..N).map(|i| vec![i as f64 * TAU / (N as f64)]).collect();

        // Compute CPU reference values using EvalCtx with each row's bindings
        use scirs2_symbolic::eml::eval::EvalCtx;
        let cpu_refs: Vec<f64> = inputs
            .iter()
            .map(|row| {
                let ctx = EvalCtx::new(row.as_slice());
                eval_real(&op, &ctx).unwrap_or(f64::NAN)
            })
            .collect();

        match kernel.eval_batch(&inputs) {
            Ok(out) => {
                assert_eq!(out.len(), N, "output length must equal row count");
                let mut max_err = 0.0_f64;
                for (i, (&got, &cpu)) in out.iter().zip(cpu_refs.iter()).enumerate() {
                    let err = (got - cpu).abs();
                    if err > max_err {
                        max_err = err;
                    }
                    assert!(
                        err < 1e-3,
                        "row {i}: GPU={got}, CPU={cpu}, |diff|={err} (tolerance 1e-3)"
                    );
                }
                println!(
                    "eval_batch_transcendental_kernel_or_skips: GPU vs CPU max_err={max_err:.2e} for {N} rows"
                );
            }
            Err(e) => {
                let msg = e.to_string();
                if is_no_adapter(&msg) {
                    println!("No wgpu adapter available — skipping ({msg})");
                } else {
                    panic!("Unexpected error: {e}");
                }
            }
        }
    }

    // ── Test 3: to_jit_auto returns Gpu above threshold ──────────────────────

    /// Use `to_jit_auto` with n=200_000 (above `GPU_DISPATCH_THRESHOLD=100_000`).
    /// Asserts `JitDispatch::Gpu` is returned. When a GPU adapter is available,
    /// also evaluates a simple expression and compares to CPU.
    /// Skips GPU evaluation gracefully when no adapter is available.
    #[test]
    fn to_jit_auto_returns_gpu_above_threshold_or_skips() {
        // f(x) = x + 0 (identity) — trivially easy for correctness check
        let op = LoweredOp::Add(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Const(0.0)));

        const BATCH: usize = 200_000;
        const { assert!(BATCH >= GPU_DISPATCH_THRESHOLD) };

        match to_jit_auto(&op, BATCH) {
            Ok(JitDispatch::Gpu(kernel)) => {
                println!("to_jit_auto returned Gpu dispatch as expected (batch={BATCH})");

                // Evaluate a small slice (3 rows) to verify correctness
                let inputs: Vec<Vec<f64>> = vec![vec![1.0], vec![2.0], vec![3.0]];
                match kernel.eval_batch(&inputs) {
                    Ok(out) => {
                        assert_eq!(out.len(), 3, "output length mismatch");
                        for (i, (&got, &exp)) in
                            out.iter().zip([1.0_f64, 2.0, 3.0].iter()).enumerate()
                        {
                            assert!(
                                (got - exp).abs() < 1e-3,
                                "row {i}: GPU={got}, expected {exp} (tolerance 1e-3)"
                            );
                        }
                        println!(
                            "to_jit_auto_returns_gpu_above_threshold_or_skips: GPU result verified"
                        );
                    }
                    Err(e) => {
                        let msg = e.to_string();
                        if is_no_adapter(&msg) {
                            println!("No wgpu adapter available for eval_batch — skipping dispatch check ({msg})");
                        } else {
                            panic!("Unexpected eval_batch error: {e}");
                        }
                    }
                }
            }
            Ok(JitDispatch::Cpu(_)) => {
                // to_jit_auto always returns Gpu for batch >= threshold; Cpu here
                // means the threshold logic is broken.
                panic!(
                    "to_jit_auto returned Cpu for batch={BATCH} >= GPU_DISPATCH_THRESHOLD={GPU_DISPATCH_THRESHOLD}"
                );
            }
            Err(e) => {
                let msg = e.to_string();
                if is_no_adapter(&msg) {
                    println!("to_jit_auto: no GPU adapter — skipping ({msg})");
                } else {
                    panic!("Unexpected to_jit_auto error: {e}");
                }
            }
        }
    }

    // ── Bonus: empty-batch is always an error ─────────────────────────────────

    #[test]
    fn eval_batch_empty_input_is_always_error() {
        let op = LoweredOp::Var(0);
        let kernel: GpuKernel = to_gpu(&op).expect("WGSL shader generation");
        match kernel.eval_batch(&[]) {
            Err(GpuError::EmptyInput) => {}
            other => panic!("expected EmptyInput for zero-row batch, got {other:?}"),
        }
    }
}
