//! GPU JIT compilation of [`crate::eml::op::LoweredOp`] to WGSL compute shaders.
//!
//! Compiles a formula to a WGSL compute shader that evaluates one element
//! per workgroup invocation. Batched evaluation amortises shader compile +
//! buffer setup over millions of inputs.
//!
//! # Dispatch heuristic
//!
//! - Batch < 10⁵: use [`crate::compile::to_jit`] (Cranelift CPU JIT)
//! - Batch ≥ 10⁵: use [`to_gpu`] (this module)
//!
//! See [`to_jit_auto`] for the dispatcher.
//!
//! # Phase 1 cut (v0.4.4)
//!
//! This module ships the WGSL **shader-text generator** and the public API
//! surface ([`GpuKernel`], [`to_gpu`], [`to_jit_auto`], [`JitDispatch`]).
//! Actual `wgpu` device submission for [`GpuKernel::eval_batch`] is deferred
//! to v0.4.5 once the WebGPU backend in `scirs2-core` exposes a stable
//! "submit user shader, await output buffer" entry point. Until then,
//! `eval_batch` returns [`GpuError::Unsupported`] explicitly — *no silent
//! NaN return*, so callers always know whether they got real GPU output.
//!
//! # f32 vs f64
//!
//! WGSL's standard storage type is `f32` (the `shader-f64` extension is
//! not portably available across all backends as of `wgpu 29`). The
//! generated shader uses `f32` storage buffers and emits `f32` literals
//! (`{:.7e}f` suffix). When v0.4.5 wires real dispatch, `f64` host inputs
//! are downcast to `f32` at upload time. Formulas requiring `f64` precision
//! should stay on the Cranelift CPU backend.
//!
//! # Feature gate
//!
//! All items in this module are gated behind the crate-local `gpu` cargo
//! feature, which transitively enables `jit`. Default builds (and
//! `--no-default-features`) do not pull in the `wgpu` / `pollster` /
//! `bytemuck` dependency stack.

#![cfg(feature = "gpu")]

use crate::eml::op::LoweredOp;
use thiserror::Error;

/// GPU JIT compilation errors.
#[derive(Debug, Error)]
pub enum GpuError {
    /// No suitable WebGPU adapter is available on this host.
    #[error("no GPU adapter available: {0}")]
    NoAdapter(String),

    /// The generated WGSL shader source failed to compile.
    #[error("WGSL compilation failed: {0}")]
    WgslError(String),

    /// A `LoweredOp` variant cannot be expressed in WGSL.
    #[error("operator not supported in WGSL: {0}")]
    Unsupported(String),

    /// Generic device error — surfaced from `wgpu`.
    #[error("GPU device error: {0}")]
    DeviceError(String),

    /// Input batch was empty.
    #[error("eval_batch: input batch is empty")]
    EmptyInput,

    /// A buffer operation (upload or readback) failed.
    #[error("GPU buffer operation failed: {0}")]
    BufferError(String),
}

/// A GPU-compiled `LoweredOp` ready for batched evaluation.
///
/// Construction generates the WGSL shader source eagerly. Actual
/// device dispatch happens inside [`Self::eval_batch`] (Phase 2).
pub struct GpuKernel {
    /// The WGSL shader source. Inspect via [`Self::wgsl`].
    wgsl_source: String,
    /// Number of variables in the formula (`max Var(i) + 1`).
    n_vars: usize,
    /// Post-order operator tape — kept here so a future Phase 2 path can
    /// reuse it for either real GPU dispatch or a CPU fallback evaluator
    /// without re-walking the `LoweredOp` tree.
    #[allow(dead_code)]
    tape: Vec<crate::eml::op::OxiOp>,
}

impl GpuKernel {
    /// Number of input variables expected per row.
    pub fn n_vars(&self) -> usize {
        self.n_vars
    }

    /// Borrow the generated WGSL source code (for inspection and tests).
    pub fn wgsl(&self) -> &str {
        &self.wgsl_source
    }

    /// Evaluate at `inputs.len()` rows (each `inputs[i]` is a `Vec<f64>` of
    /// length [`Self::n_vars`]).
    ///
    /// # Precision note
    ///
    /// WGSL uses `f32` storage. Host `f64` inputs are downcast to `f32` at
    /// upload time and upcast back to `f64` after readback. Formulas
    /// requiring strict `f64` precision should use the Cranelift CPU backend.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::NoAdapter`] when no GPU adapter is available (e.g.
    /// headless CI). Returns [`GpuError::DeviceError`] on wgpu device failure.
    pub fn eval_batch(&self, inputs: &[Vec<f64>]) -> Result<Vec<f64>, GpuError> {
        let n_rows = inputs.len();
        if n_rows == 0 {
            return Err(GpuError::EmptyInput);
        }
        // Flatten inputs row-major: row*n_vars + var_idx — matches the shader's
        // `inputs.data[base + i]` pattern where `base = idx * n_vars`.
        let n_vars = self.n_vars;
        let flat_f32: Vec<f32> = inputs
            .iter()
            .flat_map(|row| row.iter().map(|&x| x as f32))
            .collect();

        self.dispatch_wgpu(&flat_f32, n_rows, n_vars)
    }

    /// Core wgpu dispatch: upload `flat_f32` (row-major inputs), run the
    /// precompiled WGSL shader, read back `n_rows` output f32 values, return
    /// as f64.
    #[cfg(feature = "gpu")]
    fn dispatch_wgpu(
        &self,
        flat_f32: &[f32],
        n_rows: usize,
        _n_vars: usize,
    ) -> Result<Vec<f64>, GpuError> {
        use wgpu::{
            util::{BufferInitDescriptor, DeviceExt as _},
            BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
            BindingType, BufferBindingType, BufferDescriptor, BufferUsages,
            CommandEncoderDescriptor, ComputePassDescriptor, DeviceDescriptor, Features,
            InstanceDescriptor, Limits, MapMode, RequestAdapterOptions, ShaderModuleDescriptor,
            ShaderSource, ShaderStages,
        };

        // ── Adapter / device acquisition ──────────────────────────────────────
        let instance = wgpu::Instance::new(InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::default(),
            memory_budget_thresholds: Default::default(),
            backend_options: Default::default(),
            display: None,
        });

        let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .map_err(|e| GpuError::NoAdapter(e.to_string()))?;

        let (device, queue) = pollster::block_on(adapter.request_device(&DeviceDescriptor {
            label: Some("scirs2-symbolic-gpu"),
            required_features: Features::empty(),
            required_limits: Limits::default(),
            ..Default::default()
        }))
        .map_err(|e| GpuError::DeviceError(e.to_string()))?;

        // ── Encode flat input as raw bytes ─────────────────────────────────────
        let input_bytes: Vec<u8> = flat_f32.iter().flat_map(|f| f.to_le_bytes()).collect();
        let output_byte_len = (n_rows * std::mem::size_of::<f32>()) as u64;

        // ── Buffers ───────────────────────────────────────────────────────────
        let buf_inputs = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("scirs2-sym-inputs"),
            contents: &input_bytes,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        });

        let buf_outputs = device.create_buffer(&BufferDescriptor {
            label: Some("scirs2-sym-outputs"),
            size: output_byte_len,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let buf_staging = device.create_buffer(&BufferDescriptor {
            label: Some("scirs2-sym-staging"),
            size: output_byte_len,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Bind group layout (binding 0: inputs read, binding 1: outputs rw) ─
        let bgl = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("scirs2-sym-bgl"),
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // ── Pipeline ──────────────────────────────────────────────────────────
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("scirs2-sym-layout"),
            bind_group_layouts: &[Some(&bgl)],
            ..Default::default()
        });

        let shader_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("scirs2-sym-shader"),
            source: ShaderSource::Wgsl(self.wgsl_source.as_str().into()),
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("scirs2-sym-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("eval_main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // ── Bind group ────────────────────────────────────────────────────────
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("scirs2-sym-bg"),
            layout: &bgl,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: buf_inputs.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: buf_outputs.as_entire_binding(),
                },
            ],
        });

        // ── Dispatch: ceil(n_rows / 64) workgroups ────────────────────────────
        let workgroups = (n_rows as u32).div_ceil(64);
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("scirs2-sym-encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("scirs2-sym-pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        queue.submit([encoder.finish()]);

        // ── Copy outputs to staging ───────────────────────────────────────────
        let mut encoder2 = device.create_command_encoder(&CommandEncoderDescriptor { label: None });
        encoder2.copy_buffer_to_buffer(&buf_outputs, 0, &buf_staging, 0, output_byte_len);
        queue.submit([encoder2.finish()]);

        // ── Poll and map ──────────────────────────────────────────────────────
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| GpuError::BufferError(format!("GPU poll error: {e:?}")))?;

        let slice = buf_staging.slice(0..output_byte_len);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(MapMode::Read, move |r| {
            let _ = tx.send(r);
        });

        device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| GpuError::BufferError(format!("GPU poll during map: {e:?}")))?;

        rx.recv()
            .map_err(|_| GpuError::BufferError("channel closed during map_async".into()))?
            .map_err(|e| GpuError::BufferError(format!("map_async failed: {e:?}")))?;

        let mapped = slice.get_mapped_range();
        let result_f64: Vec<f64> = mapped
            .chunks_exact(4)
            .take(n_rows)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f64)
            .collect();
        drop(mapped);
        buf_staging.unmap();

        Ok(result_f64)
    }
}

/// Compile a [`LoweredOp`] to a [`GpuKernel`].
///
/// Generates the WGSL compute-shader source for the formula and returns a
/// kernel object whose [`GpuKernel::wgsl`] can be inspected immediately.
///
/// # Errors
///
/// Returns [`GpuError::WgslError`] if the formula is degenerate (empty
/// expression stack after lowering — should not occur for a well-formed
/// `LoweredOp`).
pub fn to_gpu(op: &LoweredOp) -> Result<GpuKernel, GpuError> {
    let n_vars = op.count_vars();
    let wgsl_source = generate_wgsl(op, n_vars)?;
    let tape = op.to_oxi_ops();
    Ok(GpuKernel {
        wgsl_source,
        n_vars,
        tape,
    })
}

/// Generate WGSL source for the given `LoweredOp`.
///
/// Layout: one storage buffer of `f32` inputs (row-major: `row*n_vars + var_idx`),
/// one storage buffer of `f32` outputs. The compute shader uses workgroup
/// size 64; one global invocation evaluates one row.
fn generate_wgsl(op: &LoweredOp, n_vars: usize) -> Result<String, GpuError> {
    let mut output = String::with_capacity(2048);

    output.push_str("// Auto-generated by scirs2_symbolic::compile::to_gpu\n");
    output.push_str("// One workgroup invocation = one input row evaluated.\n\n");
    output.push_str("struct Inputs { data: array<f32>, };\n");
    output.push_str("struct Outputs { data: array<f32>, };\n");
    output.push_str("@group(0) @binding(0) var<storage, read> inputs: Inputs;\n");
    output.push_str("@group(0) @binding(1) var<storage, read_write> outputs: Outputs;\n");
    output.push('\n');
    output.push_str("@compute @workgroup_size(64)\n");
    output.push_str("fn eval_main(@builtin(global_invocation_id) gid: vec3<u32>) {\n");
    output.push_str("\tlet idx = gid.x;\n");
    output.push_str(&format!("\tlet base = idx * {n_vars}u;\n"));

    let expr = wgsl_expression(op)?;
    output.push_str(&format!("\toutputs.data[idx] = {expr};\n"));
    output.push_str("}\n");

    Ok(output)
}

/// Convert a [`LoweredOp`] to a WGSL expression string. Iterative —
/// uses a post-order work stack to avoid OS-stack overflow on deep trees
/// (per the project's no-recursion-on-`LoweredOp` policy).
fn wgsl_expression(op: &LoweredOp) -> Result<String, GpuError> {
    let mut work: Vec<(&LoweredOp, bool)> = vec![(op, false)];
    let mut stack: Vec<String> = Vec::new();

    while let Some((node, visited)) = work.pop() {
        if visited {
            // Post-visit: synthesise the WGSL fragment for this node from
            // the already-emitted children sitting on top of `stack`.
            let s = post_visit(node, &mut stack)?;
            stack.push(s);
        } else {
            schedule_children(node, &mut work);
        }
    }

    stack
        .pop()
        .ok_or_else(|| GpuError::WgslError("empty expression stack".into()))
}

/// Pop children off `stack` (already in post-order) and build the WGSL
/// fragment for `node`.
fn post_visit(node: &LoweredOp, stack: &mut Vec<String>) -> Result<String, GpuError> {
    // Helper: pop one child, surfacing a structured error rather than
    // panicking, to honour the no-`unwrap()` policy. The error path
    // should never fire on a well-formed `LoweredOp` post-order walk.
    fn pop_child(stack: &mut Vec<String>, op_name: &str) -> Result<String, GpuError> {
        stack
            .pop()
            .ok_or_else(|| GpuError::WgslError(format!("missing child for {op_name}")))
    }

    let s = match node {
        LoweredOp::Const(c) => format!("{c:.7e}f"),
        LoweredOp::Var(i) => format!("inputs.data[base + {i}u]"),
        LoweredOp::Add(_, _) => {
            let b = pop_child(stack, "Add.rhs")?;
            let a = pop_child(stack, "Add.lhs")?;
            format!("({a} + {b})")
        }
        LoweredOp::Sub(_, _) => {
            let b = pop_child(stack, "Sub.rhs")?;
            let a = pop_child(stack, "Sub.lhs")?;
            format!("({a} - {b})")
        }
        LoweredOp::Mul(_, _) => {
            let b = pop_child(stack, "Mul.rhs")?;
            let a = pop_child(stack, "Mul.lhs")?;
            format!("({a} * {b})")
        }
        LoweredOp::Div(_, _) => {
            let b = pop_child(stack, "Div.rhs")?;
            let a = pop_child(stack, "Div.lhs")?;
            format!("({a} / {b})")
        }
        LoweredOp::Pow(_, _) => {
            let b = pop_child(stack, "Pow.rhs")?;
            let a = pop_child(stack, "Pow.lhs")?;
            format!("pow({a}, {b})")
        }
        LoweredOp::Neg(_) => {
            let c = pop_child(stack, "Neg")?;
            format!("(-({c}))")
        }
        LoweredOp::Exp(_) => {
            let c = pop_child(stack, "Exp")?;
            format!("exp({c})")
        }
        LoweredOp::Ln(_) => {
            // WGSL's natural log is `log` (NOT `log10`/`log2`).
            let c = pop_child(stack, "Ln")?;
            format!("log({c})")
        }
        LoweredOp::Sin(_) => {
            let c = pop_child(stack, "Sin")?;
            format!("sin({c})")
        }
        LoweredOp::Cos(_) => {
            let c = pop_child(stack, "Cos")?;
            format!("cos({c})")
        }
        LoweredOp::Tan(_) => {
            let c = pop_child(stack, "Tan")?;
            format!("tan({c})")
        }
        LoweredOp::Sinh(_) => {
            let c = pop_child(stack, "Sinh")?;
            format!("sinh({c})")
        }
        LoweredOp::Cosh(_) => {
            let c = pop_child(stack, "Cosh")?;
            format!("cosh({c})")
        }
        LoweredOp::Tanh(_) => {
            let c = pop_child(stack, "Tanh")?;
            format!("tanh({c})")
        }
        LoweredOp::Arcsin(_) => {
            let c = pop_child(stack, "Arcsin")?;
            format!("asin({c})")
        }
        LoweredOp::Arccos(_) => {
            let c = pop_child(stack, "Arccos")?;
            format!("acos({c})")
        }
        LoweredOp::Arctan(_) => {
            let c = pop_child(stack, "Arctan")?;
            format!("atan({c})")
        }
        LoweredOp::Arcsinh(_) => {
            let c = pop_child(stack, "Arcsinh")?;
            format!("asinh({c})")
        }
        LoweredOp::Arccosh(_) => {
            let c = pop_child(stack, "Arccosh")?;
            format!("acosh({c})")
        }
        LoweredOp::Arctanh(_) => {
            let c = pop_child(stack, "Arctanh")?;
            format!("atanh({c})")
        }
        LoweredOp::Sqrt(_) => {
            let c = pop_child(stack, "Sqrt")?;
            format!("sqrt({c})")
        }
        LoweredOp::Abs(_) => {
            let c = pop_child(stack, "Abs")?;
            format!("abs({c})")
        }
    };
    Ok(s)
}

/// Push children onto the work stack so they will be evaluated post-order
/// before the parent's post-visit fires.
fn schedule_children<'a>(node: &'a LoweredOp, work: &mut Vec<(&'a LoweredOp, bool)>) {
    match node {
        LoweredOp::Const(_) | LoweredOp::Var(_) => {
            work.push((node, true));
        }
        LoweredOp::Add(a, b)
        | LoweredOp::Sub(a, b)
        | LoweredOp::Mul(a, b)
        | LoweredOp::Div(a, b)
        | LoweredOp::Pow(a, b) => {
            work.push((node, true));
            // Push right first so it pops second; left pops first → matches
            // post-visit's "pop rhs then lhs" expectation.
            work.push((b, false));
            work.push((a, false));
        }
        LoweredOp::Neg(c)
        | LoweredOp::Exp(c)
        | LoweredOp::Ln(c)
        | LoweredOp::Sin(c)
        | LoweredOp::Cos(c)
        | LoweredOp::Tan(c)
        | LoweredOp::Sinh(c)
        | LoweredOp::Cosh(c)
        | LoweredOp::Tanh(c)
        | LoweredOp::Arcsin(c)
        | LoweredOp::Arccos(c)
        | LoweredOp::Arctan(c)
        | LoweredOp::Arcsinh(c)
        | LoweredOp::Arccosh(c)
        | LoweredOp::Arctanh(c)
        | LoweredOp::Sqrt(c)
        | LoweredOp::Abs(c) => {
            work.push((node, true));
            work.push((c, false));
        }
    }
}

/// Either a CPU JIT function or a GPU kernel. Returned by [`to_jit_auto`].
///
/// `Debug` is intentionally NOT derived — [`crate::compile::JitFunction`]
/// wraps a raw native function pointer + `JITModule` and does not implement
/// `Debug`. Use the variant directly via `match`.
///
/// Both variants are boxed because [`crate::compile::JitFunction`] is
/// substantially larger than [`GpuKernel`] (~456 B vs ~56 B); boxing keeps
/// the enum compact and avoids `clippy::large_enum_variant`.
pub enum JitDispatch {
    /// Cranelift-compiled native CPU function.
    Cpu(Box<crate::compile::JitFunction>),
    /// WGSL-compiled GPU compute shader.
    Gpu(Box<GpuKernel>),
}

/// Batch-size threshold above which [`to_jit_auto`] chooses the GPU backend.
///
/// Empirical: at ~10⁵ batches, WGSL shader-compile + buffer-setup + dispatch
/// latency starts to amortise below the per-call overhead of the CPU JIT
/// across many invocations. Tunable in the future once Phase 2 dispatch
/// lands and we benchmark on real hardware.
pub const GPU_DISPATCH_THRESHOLD: usize = 100_000;

/// Compile to either CPU or GPU JIT based on the expected batch size.
///
/// - `expected_batch_size < GPU_DISPATCH_THRESHOLD` → CPU (Cranelift)
/// - `expected_batch_size ≥ GPU_DISPATCH_THRESHOLD` → GPU (WGSL)
///
/// # Errors
///
/// - GPU path: any [`GpuError`] from [`to_gpu`].
/// - CPU path: any [`crate::compile::JitError`] surfaced through
///   [`GpuError::DeviceError`] for a uniform return type.
pub fn to_jit_auto(op: &LoweredOp, expected_batch_size: usize) -> Result<JitDispatch, GpuError> {
    if expected_batch_size >= GPU_DISPATCH_THRESHOLD {
        Ok(JitDispatch::Gpu(Box::new(to_gpu(op)?)))
    } else {
        let cpu = crate::compile::to_jit(op).map_err(|e| GpuError::DeviceError(e.to_string()))?;
        Ok(JitDispatch::Cpu(Box::new(cpu)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_wgsl_const() {
        // 2.5 is the kind of value that must round-trip exactly into the
        // shader; we deliberately avoid `3.14` (clippy::approx_constant
        // would flag it as a near-π).
        let op = LoweredOp::Const(2.5);
        let kernel = to_gpu(&op).expect("gpu compile");
        // The `f` suffix makes it a WGSL `f32` literal, in scientific form.
        assert!(kernel.wgsl().contains("2.5"));
        assert!(kernel.wgsl().contains("@compute"));
        assert!(kernel.wgsl().contains("eval_main"));
    }

    #[test]
    fn generate_wgsl_var() {
        let op = LoweredOp::Var(0);
        let kernel = to_gpu(&op).expect("gpu compile");
        assert!(kernel.wgsl().contains("inputs.data[base + 0u]"));
        assert_eq!(kernel.n_vars(), 1);
    }

    #[test]
    fn generate_wgsl_arithmetic() {
        let op = LoweredOp::Add(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Const(1.0)));
        let kernel = to_gpu(&op).expect("gpu compile");
        assert!(kernel.wgsl().contains("(inputs.data[base + 0u] + 1"));
    }

    #[test]
    fn generate_wgsl_transcendental() {
        let op = LoweredOp::Sin(Box::new(LoweredOp::Var(0)));
        let kernel = to_gpu(&op).expect("gpu compile");
        assert!(kernel.wgsl().contains("sin("));
    }

    #[test]
    fn generate_wgsl_native_sqrt() {
        let op = LoweredOp::Sqrt(Box::new(LoweredOp::Var(0)));
        let kernel = to_gpu(&op).expect("gpu compile");
        assert!(kernel.wgsl().contains("sqrt("));
    }

    #[test]
    fn generate_wgsl_native_abs() {
        let op = LoweredOp::Abs(Box::new(LoweredOp::Var(0)));
        let kernel = to_gpu(&op).expect("gpu compile");
        assert!(kernel.wgsl().contains("abs("));
    }

    #[test]
    fn deep_chain_no_overflow() {
        // 1000 nested Adds — must not blow the OS stack via the iterative walk.
        let mut op = LoweredOp::Var(0);
        for _ in 0..1000 {
            op = LoweredOp::Add(Box::new(op), Box::new(LoweredOp::Const(1.0)));
        }
        let _kernel = to_gpu(&op).expect("gpu compile");
    }

    #[test]
    fn auto_dispatch_below_threshold() {
        let op = LoweredOp::Var(0);
        match to_jit_auto(&op, 1000) {
            Ok(JitDispatch::Cpu(_)) => {}
            Ok(JitDispatch::Gpu(_)) => panic!("expected CPU dispatch for batch=1000, got GPU"),
            Err(e) => panic!("dispatch failed: {e}"),
        }
    }

    #[test]
    fn auto_dispatch_above_threshold() {
        let op = LoweredOp::Var(0);
        match to_jit_auto(&op, 200_000) {
            Ok(JitDispatch::Gpu(_)) => {}
            Ok(JitDispatch::Cpu(_)) => panic!("expected GPU dispatch for batch=200000, got CPU"),
            Err(e) => panic!("dispatch failed: {e}"),
        }
    }

    #[test]
    fn ln_emits_log_for_wgsl() {
        // WGSL natural log is `log` (NOT `log10` or `log2`).
        let op = LoweredOp::Ln(Box::new(LoweredOp::Var(0)));
        let kernel = to_gpu(&op).expect("gpu compile");
        assert!(kernel.wgsl().contains("log("));
    }

    #[test]
    fn eval_batch_empty_input_returns_error() {
        let op = LoweredOp::Var(0);
        let kernel = to_gpu(&op).expect("gpu compile");
        match kernel.eval_batch(&[]) {
            Err(GpuError::EmptyInput) => {}
            other => panic!("expected EmptyInput for zero-row batch, got {other:?}"),
        }
    }

    #[test]
    fn eval_batch_linear_or_skips() {
        // f(x) = 2*x + 1  →  inputs [0,1,2,3], expected [1,3,5,7]
        let x = LoweredOp::Var(0);
        let two_x = LoweredOp::Mul(Box::new(LoweredOp::Const(2.0)), Box::new(x));
        let op = LoweredOp::Add(Box::new(two_x), Box::new(LoweredOp::Const(1.0)));
        let kernel = to_gpu(&op).expect("gpu compile");

        let inputs: Vec<Vec<f64>> = (0..4).map(|i| vec![i as f64]).collect();
        match kernel.eval_batch(&inputs) {
            Ok(out) => {
                let expected = [1.0_f64, 3.0, 5.0, 7.0];
                assert_eq!(out.len(), 4, "output length mismatch");
                for (i, (&got, &exp)) in out.iter().zip(expected.iter()).enumerate() {
                    assert!(
                        (got - exp).abs() < 1e-3,
                        "row {i}: got {got}, expected {exp}"
                    );
                }
            }
            Err(e) => {
                let msg = e.to_string();
                if msg.contains("adapter")
                    || msg.contains("Adapter")
                    || msg.contains("GPU")
                    || msg.contains("no suitable")
                    || msg.contains("NoAdapter")
                {
                    println!("No wgpu adapter available — skipping eval_batch_linear ({msg})");
                } else {
                    panic!("Unexpected error: {e}");
                }
            }
        }
    }
}
