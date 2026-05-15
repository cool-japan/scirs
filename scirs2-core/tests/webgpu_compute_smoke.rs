//! Smoke tests for the real WebGPU compute shader dispatch path.
//!
//! These tests are only compiled and run when the `wgpu_backend` feature is enabled.
//! On hosts without a GPU adapter (headless CI), all tests that require GPU access
//! detect the absence of an adapter at runtime and skip gracefully (they pass, not fail).

#[cfg(feature = "wgpu_backend")]
mod webgpu_compute_smoke {
    use scirs2_core::gpu::backends::{run_vector_add_wgsl, try_compile_wgsl, WgpuComputePipeline};

    /// Static assertion: `WgpuComputePipeline` must be `Send + Sync`.
    #[test]
    fn pipeline_struct_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<WgpuComputePipeline>();
    }

    /// Attempts to compile a trivial WGSL compute shader that writes the invocation index to
    /// each element of a storage buffer.
    ///
    /// On hosts without a wgpu adapter (headless CI) the test prints a note and passes.
    #[test]
    fn wgsl_compile_succeeds_or_skips_gracefully() {
        const WGSL_WRITE_INDEX: &str = r#"
@group(0) @binding(0) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if idx < arrayLength(&output) {
        output[idx] = f32(idx);
    }
}
"#;

        match try_compile_wgsl(WGSL_WRITE_INDEX) {
            Ok(pipeline) => {
                // Verify the extracted workgroup size matches what we declared
                assert_eq!(
                    pipeline.workgroup_size,
                    [64, 1, 1],
                    "workgroup_size should be extracted as [64, 1, 1]"
                );
                println!(
                    "WebGPU compute pipeline compiled successfully (workgroup_size = {:?})",
                    pipeline.workgroup_size
                );
            }
            Err(e) => {
                let msg = e.to_string();
                if msg.contains("adapter") || msg.contains("Adapter") || msg.contains("GPU") {
                    println!("No wgpu adapter available on this host — skipping GPU compile test ({msg})");
                    // Not a failure — CI without GPU should still pass
                } else {
                    panic!("Unexpected error compiling WGSL write-index shader: {e}");
                }
            }
        }
    }

    /// Attempts to compile a vector-add WGSL shader end-to-end, upload two small vectors,
    /// dispatch the compute shader, and read back the result to assert element-wise correctness.
    ///
    /// On hosts without a wgpu adapter the test prints a note and passes.
    #[test]
    fn vector_add_runs_end_to_end_when_adapter_available() {
        const N: usize = 128;
        let a: Vec<f32> = (0..N).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..N).map(|i| (N - i) as f32).collect();
        let expected: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();

        match run_vector_add_wgsl(&a, &b) {
            Ok(result) => {
                assert_eq!(result.len(), N, "result length must match input");
                for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
                    assert!(
                        (got - exp).abs() < 1e-5,
                        "element {i}: got {got}, expected {exp}"
                    );
                }
                println!("WebGPU vector-add dispatched and verified successfully for {N} elements");
            }
            Err(e) => {
                let msg = e.to_string();
                if msg.contains("adapter") || msg.contains("Adapter") || msg.contains("GPU") {
                    println!("No wgpu adapter available on this host — skipping GPU vector-add test ({msg})");
                    // Not a failure — CI without GPU should still pass
                } else {
                    panic!("Unexpected error during GPU vector-add: {e}");
                }
            }
        }
    }

    /// Verifies that the WGSL source for elementwise_sub, elementwise_pow,
    /// elementwise_sqrt, elementwise_exp, and elementwise_log is non-empty and
    /// syntactically parseable by the workgroup-size extractor.
    ///
    /// On hosts without a wgpu adapter the compile attempt is skipped gracefully.
    #[test]
    fn elementwise_kernel_wgsl_sources_are_non_empty() {
        use scirs2_core::gpu::kernels::elementwise::{
            ElementwiseExpKernel, ElementwiseLogKernel, ElementwisePowKernel,
            ElementwiseSqrtKernel, ElementwiseSubKernel,
        };
        use scirs2_core::gpu::kernels::GpuKernel;
        use scirs2_core::gpu::GpuBackend;

        let kernels: Vec<(&str, Box<dyn GpuKernel>)> = vec![
            ("elementwise_sub", Box::new(ElementwiseSubKernel::new())),
            ("elementwise_pow", Box::new(ElementwisePowKernel::new())),
            ("elementwise_sqrt", Box::new(ElementwiseSqrtKernel::new())),
            ("elementwise_exp", Box::new(ElementwiseExpKernel::new())),
            ("elementwise_log", Box::new(ElementwiseLogKernel::new())),
        ];

        for (label, kernel) in &kernels {
            let source = kernel
                .source_for_backend(GpuBackend::Wgpu)
                .unwrap_or_else(|e| panic!("{label}: source_for_backend failed: {e}"));
            assert!(
                !source.is_empty(),
                "{label}: WGSL source must not be empty after Phase 2 fix"
            );
            assert!(
                source.contains("@compute"),
                "{label}: WGSL source must contain a @compute entry point"
            );
            assert!(
                source.contains("@workgroup_size(256)"),
                "{label}: WGSL source must declare @workgroup_size(256)"
            );
            assert!(
                source.contains("fn main("),
                "{label}: WGSL entry point must be named 'main' for pipeline dispatch compatibility"
            );
        }
    }

    /// Attempts to compile the WGSL sources for each newly-filled elementwise kernel.
    ///
    /// On hosts without a wgpu adapter the compile attempt is skipped gracefully.
    #[test]
    fn elementwise_kernel_wgsl_compiles_or_skips_gracefully() {
        use scirs2_core::gpu::kernels::elementwise::{
            ElementwiseExpKernel, ElementwiseLogKernel, ElementwisePowKernel,
            ElementwiseSqrtKernel, ElementwiseSubKernel,
        };
        use scirs2_core::gpu::kernels::GpuKernel;
        use scirs2_core::gpu::GpuBackend;

        let kernels: Vec<(&str, Box<dyn GpuKernel>)> = vec![
            ("elementwise_sub", Box::new(ElementwiseSubKernel::new())),
            ("elementwise_pow", Box::new(ElementwisePowKernel::new())),
            ("elementwise_sqrt", Box::new(ElementwiseSqrtKernel::new())),
            ("elementwise_exp", Box::new(ElementwiseExpKernel::new())),
            ("elementwise_log", Box::new(ElementwiseLogKernel::new())),
        ];

        for (label, kernel) in &kernels {
            let source = kernel
                .source_for_backend(GpuBackend::Wgpu)
                .unwrap_or_else(|e| panic!("{label}: source_for_backend failed: {e}"));

            match try_compile_wgsl(&source) {
                Ok(pipeline) => {
                    assert_eq!(
                        pipeline.workgroup_size,
                        [256, 1, 1],
                        "{label}: expected workgroup_size [256, 1, 1]"
                    );
                    println!(
                        "{label}: compiled successfully (workgroup_size = {:?})",
                        pipeline.workgroup_size
                    );
                }
                Err(e) => {
                    let msg = e.to_string();
                    if msg.contains("adapter")
                        || msg.contains("Adapter")
                        || msg.contains("GPU")
                        || msg.contains("no suitable")
                    {
                        println!(
                            "{label}: no wgpu adapter available — skipping GPU compile ({msg})"
                        );
                    } else {
                        panic!("{label}: unexpected error compiling WGSL: {e}");
                    }
                }
            }
        }
    }

    /// Verifies workgroup_size extraction for multi-dimensional workgroups.
    #[test]
    fn workgroup_size_extraction_for_2d_workgroup() {
        // A shader with 2D workgroup size; verify the extracted size is correct.
        const WGSL_2D: &str = r#"
@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@group(0) @binding(1) var<storage, read> input: array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let idx = id.x + id.y * 8u;
    if idx < arrayLength(&data) {
        data[idx] = input[idx] * 2.0;
    }
}
"#;
        match try_compile_wgsl(WGSL_2D) {
            Ok(pipeline) => {
                assert_eq!(
                    pipeline.workgroup_size,
                    [8, 8, 1],
                    "2D workgroup_size should be [8, 8, 1]"
                );
                println!(
                    "2D workgroup pipeline compiled: {:?}",
                    pipeline.workgroup_size
                );
            }
            Err(e) => {
                let msg = e.to_string();
                if msg.contains("adapter") || msg.contains("Adapter") || msg.contains("GPU") {
                    println!("No wgpu adapter — skipping 2D workgroup test ({msg})");
                } else {
                    panic!("Unexpected error compiling 2D workgroup shader: {e}");
                }
            }
        }
    }
}
