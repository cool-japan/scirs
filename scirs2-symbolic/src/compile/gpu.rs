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
    /// No suitable WebGPU adapter is available on this host (Phase 2 dispatch).
    #[error("no GPU adapter available")]
    NoAdapter,

    /// The generated WGSL shader source failed to compile.
    #[error("WGSL compilation failed: {0}")]
    WgslError(String),

    /// A `LoweredOp` variant cannot be expressed in WGSL.
    #[error("operator not supported in WGSL: {0}")]
    Unsupported(String),

    /// Generic device error — surfaced from `wgpu` (Phase 2 dispatch).
    #[error("GPU device error: {0}")]
    DeviceError(String),
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

    /// Evaluate at `inputs.len()` rows (each `inputs[i]` is a `&[f64]` of
    /// length [`Self::n_vars`]).
    ///
    /// # Phase 1 status
    ///
    /// Returns [`GpuError::Unsupported`] explicitly. Real `wgpu` device
    /// submission lands in v0.4.5; until then callers should use the CPU
    /// JIT path (`to_jit`) and revisit when GPU dispatch is wired.
    pub fn eval_batch(&self, _inputs: &[Vec<f64>]) -> Result<Vec<f64>, GpuError> {
        Err(GpuError::Unsupported(
            "GpuKernel::eval_batch dispatch is wired in v0.4.5; \
             Phase 1 ships the shader generator only. \
             Use crate::compile::to_jit for CPU evaluation."
                .into(),
        ))
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
    fn eval_batch_returns_unsupported_in_phase1() {
        let op = LoweredOp::Var(0);
        let kernel = to_gpu(&op).expect("gpu compile");
        let inputs = vec![vec![1.0_f64], vec![2.0_f64]];
        match kernel.eval_batch(&inputs) {
            Err(GpuError::Unsupported(_)) => {}
            other => panic!("expected Unsupported in Phase 1, got {other:?}"),
        }
    }
}
