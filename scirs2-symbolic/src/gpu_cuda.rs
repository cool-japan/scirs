#![cfg(feature = "cuda")]
//! Pure-Rust CUDA path for batched `f64` [`LoweredOp`] evaluation via the
//! `oxicuda` ecosystem.
//!
//! This module is an optional, **off-by-default** CUDA acceleration path for
//! evaluating a single [`crate::eml::op::LoweredOp`] over many input rows. It is
//! built entirely on the pure-Rust `oxicuda` crates (`oxicuda-ptx`,
//! `oxicuda-launch`, `oxicuda-driver`, `oxicuda-memory`), compiled only when the
//! crate-local `cuda` feature is enabled, and it does **not** route through
//! `scirs2-core`. The existing CPU, Cranelift-JIT, and `wgpu` paths are untouched.
//!
//! ## Availability (runtime-probed, NVIDIA-only)
//!
//! `oxicuda-driver` loads `libcuda` at runtime. On a machine with no NVIDIA
//! driver (for example this development Mac), initialization fails and the
//! public functions return a [`CudaEvalError`] variant rather than panicking.
//! Call [`cuda_is_available`] first to probe for a usable device; it never
//! panics and returns `false` when CUDA is unavailable. Importantly, the crate
//! still *compiles* on macOS with `--features cuda` — only the runtime probe
//! returns `false`.
//!
//! ## The CUSTOM-KERNEL pattern (vs scirs2-fft's library-call pattern)
//!
//! Unlike `scirs2-fft`'s CUDA path — which calls into the prebuilt
//! `oxicuda-fft` *library* (a fixed C2C transform) — this module **generates a
//! bespoke PTX kernel** for the supplied formula using `oxicuda-ptx`'s
//! instruction-level DSL, then dispatches it with `oxicuda-launch`. The kernel
//! name is `sym_eval` and the PTX is produced by [`generate_ptx`], which is a
//! fully **GPU-free** code generator: it can be inspected and unit-tested on any
//! platform without a CUDA device present. Only the actual device launch
//! (inside [`cuda_eval_batch`]) requires NVIDIA hardware.
//!
//! ## f64-native — a genuine precision advantage over the wgpu path
//!
//! The existing `wgpu` JIT path in [`crate::compile::gpu`] downcasts host `f64`
//! inputs to `f32` at upload time, because WGSL's portable storage type is
//! `f32` (the `shader-f64` extension is not portably available across `wgpu`
//! backends). This CUDA path is **`f64`-native end to end**: variables and
//! constants are uploaded as `f64`, every arithmetic instruction is the
//! double-precision PTX op (`add.f64`, `sub.f64`, `fma.rn.f64`), and the result
//! is read back as `f64`. For formulas that need full double precision, this is
//! a real win over the f32 wgpu path.
//!
//! ## Fixed-arity 5-parameter kernel design
//!
//! The generated `sym_eval` kernel always takes exactly five parameters in this
//! order:
//!
//! 1. `out_ptr` (`u64`) — device pointer to the `f64` output buffer (`n` rows).
//! 2. `n` (`u32`) — number of rows (thread guard).
//! 3. `vars_ptr` (`u64`) — device pointer to the row-major `f64` variable matrix
//!    (`n * n_vars` elements; `vars_ptr[(row*n_vars + i)]` is variable `i` of
//!    `row`).
//! 4. `n_vars` (`u32`) — number of variables per row.
//! 5. `consts_ptr` (`u64`) — device pointer to the `f64` constant buffer
//!    (`n_consts + 1` elements; the trailing entry is a literal `0.0`, see below).
//!
//! One thread evaluates one row. This fixed five-argument shape is what lets the
//! host side use a single concrete `oxicuda_launch::KernelArgs` tuple
//! `(u64, u32, u64, u32, u64)` regardless of the formula.
//!
//! ## The constant buffer + fma-with-trailing-zero trick
//!
//! `oxicuda-ptx`'s public DSL currently exposes **no `f64` immediate
//! materialization** — there is no `mov_imm_f64`, no public
//! `mul_f64(Register, Register)`, the low-level `emit()` is private, and the
//! `raw_ptx` auto-declaration helper does not recognize an `f64` register
//! prefix. To keep `f64` precision without a usable immediate, every real
//! `Const(c)` is hoisted into a host-side **constants device buffer** and loaded
//! from global memory inside the kernel. The buffer carries one extra trailing
//! `0.0` at index `n_consts`; multiplication `a * b` is then emitted as
//! `fma.rn.f64(a, b, 0.0)` (fused multiply-add `a*b + 0`), which is exactly the
//! product and needs no `mul.f64` builder method.
//!
//! ## Supported op subset and extensibility
//!
//! [`generate_ptx`] (and hence [`cuda_eval_batch`]) currently supports the
//! arithmetic core `{Const, Var, Add, Sub, Mul}`. A pre-pass
//! ([`validate_supported`]) walks the tree iteratively and returns
//! [`CudaEvalError::Unsupported`] on the first variant outside that set — *no
//! silent wrong answer*. The `LoweredOp` → PTX walk is a straightforward
//! iterative post-order over the tree, so extending it (e.g. `Div` via
//! `rcp.rn.f64` + `fma`, or the transcendentals via range-reduction helpers) is
//! a matter of adding post-visit arms; the device-management boilerplate stays
//! unchanged.

use crate::eml::op::LoweredOp;
use oxicuda_driver::Module;
use oxicuda_launch::{grid_size_for, Kernel, LaunchParams};
use oxicuda_memory::DeviceBuffer;
use oxicuda_ptx::arch::SmVersion;
use oxicuda_ptx::builder::{BodyBuilder, KernelBuilder};
use oxicuda_ptx::ir::{PtxType, Register};
use std::sync::Arc;
use thiserror::Error;

/// Probe whether a usable NVIDIA CUDA device is available at runtime.
///
/// Never panics. Returns `false` when the CUDA driver cannot be initialized
/// (for example on non-NVIDIA platforms such as macOS) or when no device is
/// present. Call this before [`cuda_eval_batch`] to decide whether the CUDA
/// path is usable on the current host.
pub fn cuda_is_available() -> bool {
    oxicuda_driver::init().is_ok()
        && oxicuda_driver::device::Device::count()
            .map(|c| c > 0)
            .unwrap_or(false)
}

/// Errors from the CUDA `LoweredOp` evaluation path.
#[derive(Debug, Error)]
pub enum CudaEvalError {
    /// PTX kernel generation (codegen) failed — surfaced from `oxicuda-ptx`.
    #[error("PTX generation failed: {0}")]
    PtxGen(String),

    /// A CUDA kernel launch (module load, kernel lookup, launch, or sync) failed.
    #[error("CUDA kernel launch failed: {0}")]
    Launch(String),

    /// The CUDA backend could not be initialized (driver/device/context/stream).
    #[error("CUDA backend unavailable: {0}")]
    BackendError(String),

    /// A `LoweredOp` variant is not supported by the PTX code generator.
    #[error("operator not supported in CUDA PTX path: {0}")]
    Unsupported(String),

    /// The input batch was empty.
    #[error("cuda_eval_batch: input batch is empty")]
    EmptyInput,

    /// An input row did not match the expected variable count.
    #[error("input shape mismatch: {0}")]
    ShapeMismatch(String),
}

/// Returns the static name of a [`LoweredOp`] variant (for diagnostics).
fn variant_name(op: &LoweredOp) -> &'static str {
    match op {
        LoweredOp::Const(_) => "Const",
        LoweredOp::Var(_) => "Var",
        LoweredOp::Add(_, _) => "Add",
        LoweredOp::Sub(_, _) => "Sub",
        LoweredOp::Mul(_, _) => "Mul",
        LoweredOp::Div(_, _) => "Div",
        LoweredOp::Pow(_, _) => "Pow",
        LoweredOp::Neg(_) => "Neg",
        LoweredOp::Exp(_) => "Exp",
        LoweredOp::Ln(_) => "Ln",
        LoweredOp::Sin(_) => "Sin",
        LoweredOp::Cos(_) => "Cos",
        LoweredOp::Tan(_) => "Tan",
        LoweredOp::Sinh(_) => "Sinh",
        LoweredOp::Cosh(_) => "Cosh",
        LoweredOp::Tanh(_) => "Tanh",
        LoweredOp::Arcsin(_) => "Arcsin",
        LoweredOp::Arccos(_) => "Arccos",
        LoweredOp::Arctan(_) => "Arctan",
        LoweredOp::Arcsinh(_) => "Arcsinh",
        LoweredOp::Arccosh(_) => "Arccosh",
        LoweredOp::Arctanh(_) => "Arctanh",
        LoweredOp::Sqrt(_) => "Sqrt",
        LoweredOp::Abs(_) => "Abs",
    }
}

/// Pre-pass validator: confirm the whole expression is within the supported
/// op subset `{Const, Var, Add, Sub, Mul}`.
///
/// Iterative work-stack walk (no recursion on `LoweredOp`). On the first
/// variant outside the supported set — at any depth — returns
/// [`CudaEvalError::Unsupported`] naming that variant. The validator descends
/// into children of supported nodes so a deeply nested unsupported operator is
/// still caught. This pre-pass exists because the `KernelBuilder::body` closure
/// returns `()` and therefore cannot itself surface an error.
fn validate_supported(op: &LoweredOp) -> Result<(), CudaEvalError> {
    let mut work: Vec<&LoweredOp> = vec![op];
    while let Some(node) = work.pop() {
        match node {
            LoweredOp::Const(_) | LoweredOp::Var(_) => {}
            LoweredOp::Add(a, b) | LoweredOp::Sub(a, b) | LoweredOp::Mul(a, b) => {
                work.push(a);
                work.push(b);
            }
            other => {
                return Err(CudaEvalError::Unsupported(variant_name(other).to_string()));
            }
        }
    }
    Ok(())
}

/// Collect every `Const(c)` value in the expression in iterative **post-order**.
///
/// The ordering must match [`emit_walk`]'s post-order exactly, so that
/// `const_regs[k]` (loaded from the device constants buffer in the same order)
/// aligns with the `k`-th constant the walk consumes. `Var` and operator nodes
/// contribute no value; the walk just descends through them.
fn collect_consts(op: &LoweredOp) -> Vec<f64> {
    let mut consts: Vec<f64> = Vec::new();
    let mut work: Vec<(&LoweredOp, bool)> = vec![(op, false)];
    while let Some((node, visited)) = work.pop() {
        if visited {
            if let LoweredOp::Const(c) = node {
                consts.push(*c);
            }
        } else {
            schedule_children(node, &mut work);
        }
    }
    consts
}

/// Push the children of `node` onto the work stack so they are visited in
/// post-order before the parent's post-visit fires.
///
/// Covers **all** [`LoweredOp`] variants so the iterative walk never gets stuck,
/// mirroring the `schedule_children` shape used by [`crate::compile::gpu`]. For
/// binary nodes the right child is pushed first so it pops second, which makes
/// the post-visit "pop rhs then lhs" ordering correct.
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
            // Push right first so it pops second; left pops first → matches the
            // post-visit "pop rhs then lhs" expectation.
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

/// Emit the PTX instructions evaluating `op` for the current thread's row,
/// returning the [`Register`] holding the `f64` result.
///
/// Iterative post-order walk (no recursion on `LoweredOp`). Maintains its own
/// `const_counter` that increments on each `Const` post-visit; because the walk
/// visits constants in the **same** post-order as [`collect_consts`],
/// `const_regs[const_counter]` is exactly the right pre-loaded constant. `Mul`
/// is lowered to `fma.rn.f64(a, b, 0.0)` (see module docs). On the (validated,
/// unreachable) event of an unsupported node, a comment is emitted and the
/// trailing zero register is pushed, so the function is total without any
/// `unwrap`/`expect`.
fn emit_walk(
    b: &mut BodyBuilder<'_>,
    op: &LoweredOp,
    var_regs: &[Register],
    const_regs: &[Register],
    n_consts: usize,
    zero_reg: &Register,
) -> Register {
    let mut const_counter: usize = 0;
    let mut stack: Vec<Register> = Vec::new();
    let mut work: Vec<(&LoweredOp, bool)> = vec![(op, false)];

    while let Some((node, visited)) = work.pop() {
        if visited {
            match node {
                LoweredOp::Const(_) => {
                    // `const_counter` aligns with `collect_consts` post-order.
                    let reg = const_regs
                        .get(const_counter)
                        .cloned()
                        .unwrap_or_else(|| zero_reg.clone());
                    // Guard (defensive; the index cannot exceed n_consts here).
                    if const_counter >= n_consts {
                        b.comment("const index overflow");
                    }
                    const_counter += 1;
                    stack.push(reg);
                }
                LoweredOp::Var(i) => {
                    let reg = var_regs
                        .get(*i)
                        .cloned()
                        .unwrap_or_else(|| zero_reg.clone());
                    stack.push(reg);
                }
                LoweredOp::Add(_, _) => {
                    let b_reg = stack.pop().unwrap_or_else(|| zero_reg.clone());
                    let a_reg = stack.pop().unwrap_or_else(|| zero_reg.clone());
                    stack.push(b.add_f64(a_reg, b_reg));
                }
                LoweredOp::Sub(_, _) => {
                    let b_reg = stack.pop().unwrap_or_else(|| zero_reg.clone());
                    let a_reg = stack.pop().unwrap_or_else(|| zero_reg.clone());
                    stack.push(b.sub_f64(a_reg, b_reg));
                }
                LoweredOp::Mul(_, _) => {
                    let b_reg = stack.pop().unwrap_or_else(|| zero_reg.clone());
                    let a_reg = stack.pop().unwrap_or_else(|| zero_reg.clone());
                    // a * b == fma(a, b, 0.0): no public mul.f64 builder exists.
                    stack.push(b.fma_f64(a_reg, b_reg, zero_reg.clone()));
                }
                _ => {
                    // Unreachable: validate_supported() rejected this earlier.
                    b.comment("unreachable: validated by pre-pass");
                    stack.push(zero_reg.clone());
                }
            }
        } else {
            schedule_children(node, &mut work);
        }
    }

    stack.pop().unwrap_or_else(|| {
        b.comment("empty result stack");
        zero_reg.clone()
    })
}

/// Generate the PTX text for the `sym_eval` kernel evaluating `op`.
///
/// This is a fully **GPU-free** code generator: it builds and returns the PTX
/// string without touching any CUDA device, so it can be inspected and tested
/// on any platform. It is general over the supported op subset
/// `{Const, Var, Add, Sub, Mul}` (validated up front by [`validate_supported`]).
///
/// - `n_vars` — number of variables per row (the kernel reads
///   `vars_ptr[(tid*n_vars + i)]` for variable `i`).
/// - `n_consts` — number of real `Const` values in the expression. The device
///   constants buffer is expected to have `n_consts + 1` entries: the real
///   constants followed by a trailing `0.0` (index `n_consts`) used to express
///   `Mul` as `fma(a, b, 0.0)`.
///
/// # Errors
///
/// Returns [`CudaEvalError::Unsupported`] if the expression contains a variant
/// outside the supported subset, or [`CudaEvalError::PtxGen`] if `oxicuda-ptx`
/// fails to emit the module.
pub fn generate_ptx(
    op: &LoweredOp,
    n_vars: usize,
    n_consts: usize,
) -> Result<String, CudaEvalError> {
    validate_supported(op)?;

    // The body closure is `'static`; capture owned data (Copy scalars + a clone
    // of the op tree).
    let op_owned = op.clone();

    KernelBuilder::new("sym_eval")
        .target(SmVersion::Sm80)
        .param("out_ptr", PtxType::U64)
        .param("n", PtxType::U32)
        .param("vars_ptr", PtxType::U64)
        .param("n_vars", PtxType::U32)
        .param("consts_ptr", PtxType::U64)
        .body(move |b| {
            let tid = b.global_thread_id_x();
            let n_reg = b.load_param_u32("n");
            let tid_for_guard = tid.clone();
            b.if_lt_u32(tid_for_guard, n_reg, |b| {
                let vars_ptr_reg = b.load_param_u64("vars_ptr");
                let nvars_reg = b.load_param_u32("n_vars");
                let consts_ptr_reg = b.load_param_u64("consts_ptr");
                let eight = b.mov_imm_u32(8);

                // Pre-load each variable for this row:
                //   Var(i) value = vars_ptr[(tid*n_vars + i) * 8 bytes] (f64).
                let var_regs: Vec<Register> = (0..n_vars)
                    .map(|i| {
                        let i_imm = b.mov_imm_u32(i as u32);
                        // tid*n_vars + i
                        let idx32 = b.mad_lo_u32(tid.clone(), nvars_reg.clone(), i_imm);
                        // *8 bytes -> u64
                        let off64 = b.mul_wide_u32_to_u64(idx32, eight.clone());
                        let addr = b.add_u64(vars_ptr_reg.clone(), off64);
                        b.load_global_f64(addr)
                    })
                    .collect();

                // Pre-load each constant (0..n_consts) AND the trailing zero at
                // index n_consts (used to express Mul as fma(a, b, 0.0)).
                let const_regs: Vec<Register> = (0..(n_consts + 1))
                    .map(|k| {
                        let k_imm = b.mov_imm_u32(k as u32);
                        let coff = b.mul_wide_u32_to_u64(k_imm, eight.clone());
                        let caddr = b.add_u64(consts_ptr_reg.clone(), coff);
                        b.load_global_f64(caddr)
                    })
                    .collect();

                let zero_reg = const_regs[n_consts].clone();
                let result = emit_walk(b, &op_owned, &var_regs, &const_regs, n_consts, &zero_reg);

                // out_addr = out_ptr + tid*8
                let out_ptr_reg = b.load_param_u64("out_ptr");
                let out_off = b.mul_wide_u32_to_u64(tid.clone(), eight.clone());
                let out_addr = b.add_u64(out_ptr_reg, out_off);
                b.store_global_f64(out_addr, result);
            });
            b.ret();
        })
        .build()
        .map_err(|e| CudaEvalError::PtxGen(format!("{e}")))
}

/// Map an `oxicuda` driver/memory error into [`CudaEvalError::Launch`].
fn launch_err(e: oxicuda_driver::CudaError) -> CudaEvalError {
    CudaEvalError::Launch(format!("oxicuda: {e}"))
}

/// Initialize the CUDA driver and build a stream bound to device 0.
///
/// Returns the owning [`oxicuda_driver::Context`] (in an `Arc`) alongside a
/// [`oxicuda_driver::stream::Stream`]; the caller must keep the context alive
/// for the lifetime of any kernel launch. All failures map to
/// [`CudaEvalError::BackendError`].
fn build_handle(
) -> Result<(Arc<oxicuda_driver::Context>, oxicuda_driver::stream::Stream), CudaEvalError> {
    oxicuda_driver::init()
        .map_err(|e| CudaEvalError::BackendError(format!("CUDA unavailable: {e}")))?;
    let count = oxicuda_driver::device::Device::count()
        .map_err(|e| CudaEvalError::BackendError(format!("device count: {e}")))?;
    if count <= 0 {
        return Err(CudaEvalError::BackendError(
            "no NVIDIA CUDA device available".into(),
        ));
    }
    let dev = oxicuda_driver::device::Device::get(0)
        .map_err(|e| CudaEvalError::BackendError(format!("device get: {e}")))?;
    let ctx = Arc::new(
        oxicuda_driver::Context::new(&dev)
            .map_err(|e| CudaEvalError::BackendError(format!("context: {e}")))?,
    );
    let stream = oxicuda_driver::stream::Stream::new(&ctx)
        .map_err(|e| CudaEvalError::BackendError(format!("stream: {e}")))?;
    Ok((ctx, stream))
}

/// Evaluate a single [`LoweredOp`] over a batch of input rows on a CUDA device,
/// keeping full `f64` precision.
///
/// `inputs[r]` is the variable row for row `r`: a `Vec<f64>` of length
/// `op.count_vars()`. Returns one `f64` result per row.
///
/// The flow is: collect constants → generate the bespoke `sym_eval` PTX kernel
/// → build a device handle → upload the row-major variable matrix and the
/// constants buffer (with trailing `0.0`) → launch `sym_eval` → synchronize →
/// copy results back.
///
/// # Errors
///
/// - [`CudaEvalError::EmptyInput`] when `inputs` is empty.
/// - [`CudaEvalError::ShapeMismatch`] when a row's length differs from
///   `op.count_vars()`.
/// - [`CudaEvalError::Unsupported`] / [`CudaEvalError::PtxGen`] from
///   [`generate_ptx`].
/// - [`CudaEvalError::BackendError`] when no NVIDIA CUDA device is available.
/// - [`CudaEvalError::Launch`] on module load, kernel lookup, launch, sync, or
///   buffer transfer failure.
pub fn cuda_eval_batch(op: &LoweredOp, inputs: &[Vec<f64>]) -> Result<Vec<f64>, CudaEvalError> {
    if inputs.is_empty() {
        return Err(CudaEvalError::EmptyInput);
    }
    let n_vars = op.count_vars();

    // Validate every row's shape up front.
    for (r, row) in inputs.iter().enumerate() {
        if row.len() != n_vars {
            return Err(CudaEvalError::ShapeMismatch(format!(
                "row {r} has len {} but expected {n_vars}",
                row.len()
            )));
        }
    }

    let consts = collect_consts(op);
    let n_consts = consts.len();
    // Constants buffer: the real constants followed by a trailing 0.0 used to
    // express Mul as fma(a, b, 0.0). Length is n_consts + 1.
    let mut consts_host = consts;
    consts_host.push(0.0);

    // Flatten inputs row-major: row[i] at r*n_vars + i.
    let vars_host: Vec<f64> = inputs.iter().flat_map(|row| row.iter().copied()).collect();
    let n_rows = inputs.len();

    let ptx = generate_ptx(op, n_vars, n_consts)?;

    // Keep `_ctx` alive across the launch so device memory + the loaded module
    // remain valid.
    let (_ctx, stream) = build_handle()?;

    let d_vars = DeviceBuffer::from_host(&vars_host).map_err(launch_err)?;
    let d_consts = DeviceBuffer::from_host(&consts_host).map_err(launch_err)?;
    let d_out = DeviceBuffer::<f64>::alloc(n_rows).map_err(launch_err)?;

    let module = Arc::new(
        Module::from_ptx(&ptx)
            .map_err(|e| CudaEvalError::Launch(format!("module from_ptx: {e}")))?,
    );
    let kernel = Kernel::from_module(module, "sym_eval")
        .map_err(|e| CudaEvalError::Launch(format!("kernel from_module: {e}")))?;

    let block = 256u32;
    let grid = grid_size_for(n_rows as u32, block);
    let params = LaunchParams::new(grid, block);

    let args = (
        d_out.as_device_ptr(),
        n_rows as u32,
        d_vars.as_device_ptr(),
        n_vars as u32,
        d_consts.as_device_ptr(),
    );

    kernel
        .launch(&params, &stream, &args)
        .map_err(|e| CudaEvalError::Launch(format!("launch: {e}")))?;
    stream
        .synchronize()
        .map_err(|e| CudaEvalError::Launch(format!("sync: {e}")))?;

    let mut host_out = vec![0.0f64; n_rows];
    d_out.copy_to_host(&mut host_out).map_err(launch_err)?;
    Ok(host_out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eml::{eval_real, EvalCtx};

    #[test]
    fn generate_ptx_for_linear_succeeds() {
        // 2*x + 1 — two constants (2.0 and 1.0), one variable.
        let op = LoweredOp::Add(
            Box::new(LoweredOp::Mul(
                Box::new(LoweredOp::Const(2.0)),
                Box::new(LoweredOp::Var(0)),
            )),
            Box::new(LoweredOp::Const(1.0)),
        );
        let ptx = generate_ptx(&op, 1, 2).expect("ptx");
        assert!(ptx.contains(".entry sym_eval"), "missing entry: {ptx}");
        assert!(ptx.contains("sm_80"), "missing target: {ptx}");
        assert!(
            ptx.contains("ld.global.f64"),
            "missing ld.global.f64: {ptx}"
        );
        assert!(
            ptx.contains("st.global.f64"),
            "missing st.global.f64: {ptx}"
        );
        assert!(ptx.contains("fma"), "missing fma: {ptx}");
        assert!(ptx.contains("add.f64"), "missing add.f64: {ptx}");

        // x*y — no constants, two variables; at least two var loads + an fma.
        let op = LoweredOp::Mul(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Var(1)));
        let op_ptx = generate_ptx(&op, 2, 0).expect("ptx");
        assert!(op_ptx.contains("fma"), "missing fma: {op_ptx}");
        assert!(
            op_ptx.contains("ld.global.f64"),
            "missing ld.global.f64: {op_ptx}"
        );
        assert!(
            op_ptx.matches("ld.global.f64").count() >= 2,
            "expected >=2 global f64 loads: {op_ptx}"
        );

        // Sin is outside the supported subset → Unsupported.
        let op = LoweredOp::Sin(Box::new(LoweredOp::Var(0)));
        assert!(matches!(
            generate_ptx(&op, 1, 0),
            Err(CudaEvalError::Unsupported(_))
        ));
    }

    #[test]
    fn cuda_eval_linear_or_skip() {
        if !cuda_is_available() {
            eprintln!("skipping: no NVIDIA CUDA device");
            assert!(!cuda_is_available());
            return;
        }

        // 2*x + 1 over rows 0..4.
        let op = LoweredOp::Add(
            Box::new(LoweredOp::Mul(
                Box::new(LoweredOp::Const(2.0)),
                Box::new(LoweredOp::Var(0)),
            )),
            Box::new(LoweredOp::Const(1.0)),
        );
        let inputs: Vec<Vec<f64>> = (0..4).map(|i| vec![i as f64]).collect();
        let out = cuda_eval_batch(&op, &inputs).expect("cuda eval");
        for (r, row) in inputs.iter().enumerate() {
            let want = eval_real(&op, &EvalCtx::new(row)).expect("cpu");
            assert!(
                (out[r] - want).abs() < 1e-9,
                "row {r}: got {}, want {want}",
                out[r]
            );
        }

        // x*y over two rows.
        let op2 = LoweredOp::Mul(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Var(1)));
        let inputs2 = vec![vec![2.0, 3.0], vec![1.5, 4.0]];
        let out2 = cuda_eval_batch(&op2, &inputs2).expect("cuda eval");
        for (r, row) in inputs2.iter().enumerate() {
            let want = eval_real(&op2, &EvalCtx::new(row)).expect("cpu");
            assert!(
                (out2[r] - want).abs() < 1e-9,
                "row {r}: got {}, want {want}",
                out2[r]
            );
        }
    }

    /// §3b PTX→JIT→launch coverage: deeply nested 5-level `{Const,Var,Add,Sub,Mul}`
    /// tree over 3 variables, batched across 10 diverse input rows.
    ///
    /// Formula: `((x0*3.0 + x1*2.0) - x2) * (x0 - x1*0.5 + x2*1.5 - 7.0)`
    ///
    /// This tree has:
    /// - 3 variables (Var(0)..Var(2))
    /// - 5 constants in post-order: [3.0, 2.0, 0.5, 1.5, 7.0]
    /// - 5 levels deep (deepest path: Mul→Sub→Add→Mul→Var)
    /// - All five supported op kinds exercised
    ///
    /// Tolerance rationale: pure arithmetic ({Add,Sub,fma-as-Mul}), no
    /// transcendental polynomial; rounding per operation is ~1 ULP per op,
    /// accumulating to well under 1e-9 for a depth-5 tree.
    #[test]
    fn cuda_eval_deep_nest_or_skip() {
        if !cuda_is_available() {
            eprintln!("skipping: no NVIDIA CUDA device");
            assert!(!cuda_is_available());
            return;
        }

        // ((x0*3.0 + x1*2.0) - x2)
        let lhs = LoweredOp::Sub(
            Box::new(LoweredOp::Add(
                Box::new(LoweredOp::Mul(
                    Box::new(LoweredOp::Var(0)),
                    Box::new(LoweredOp::Const(3.0)),
                )),
                Box::new(LoweredOp::Mul(
                    Box::new(LoweredOp::Var(1)),
                    Box::new(LoweredOp::Const(2.0)),
                )),
            )),
            Box::new(LoweredOp::Var(2)),
        );
        // (x0 - x1*0.5 + x2*1.5 - 7.0)  =  ((x0 - x1*0.5) + x2*1.5) - 7.0
        let rhs = LoweredOp::Sub(
            Box::new(LoweredOp::Add(
                Box::new(LoweredOp::Sub(
                    Box::new(LoweredOp::Var(0)),
                    Box::new(LoweredOp::Mul(
                        Box::new(LoweredOp::Var(1)),
                        Box::new(LoweredOp::Const(0.5)),
                    )),
                )),
                Box::new(LoweredOp::Mul(
                    Box::new(LoweredOp::Var(2)),
                    Box::new(LoweredOp::Const(1.5)),
                )),
            )),
            Box::new(LoweredOp::Const(7.0)),
        );
        let op = LoweredOp::Mul(Box::new(lhs), Box::new(rhs));

        // 10 rows: integers, negatives, non-integer, zero, large values.
        let inputs: Vec<Vec<f64>> = vec![
            vec![1.0, 2.0, 3.0],
            vec![0.0, 0.0, 0.0],
            vec![2.0, 1.0, 0.5],
            vec![-1.0, 3.0, -2.0],
            vec![5.0, -2.0, 1.0],
            vec![0.5, 0.5, 0.5],
            vec![10.0, -5.0, 3.0],
            vec![-3.0, -3.0, -3.0],
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, -1.0],
        ];

        let out = cuda_eval_batch(&op, &inputs).expect("cuda eval deep nest");
        // 1e-9: pure arithmetic fma chain; mathematically exact ops mean the
        // only error source is IEEE-754 rounding (~1e-15 per step, depth-5 tree).
        let tol = 1e-9_f64;
        for (r, row) in inputs.iter().enumerate() {
            let want = eval_real(&op, &EvalCtx::new(row)).expect("cpu eval");
            assert!(
                (out[r] - want).abs() < tol,
                "deep nest row {r} {row:?}: gpu={}, cpu={want}, |diff|={}",
                out[r],
                (out[r] - want).abs()
            );
        }
    }

    /// §3b non-device rejection test: `generate_ptx` and `cuda_eval_batch` must
    /// return `Err(CudaEvalError::Unsupported(_))` for any op outside the
    /// supported set `{Const, Var, Add, Sub, Mul}`.
    ///
    /// No CUDA device is required: `validate_supported` is called inside
    /// `generate_ptx` *before* `build_handle`, so the error is returned from
    /// the expression-tree pre-pass without touching any driver API.
    #[test]
    fn unsupported_op_rejects_without_device() {
        // Div — rejected at the top level.
        let div_op = LoweredOp::Div(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Const(1.0)));
        assert!(
            matches!(
                generate_ptx(&div_op, 1, 1),
                Err(CudaEvalError::Unsupported(_))
            ),
            "generate_ptx(Div) must return Unsupported"
        );
        // cuda_eval_batch rejects at the same validate_supported pre-pass —
        // no device access because the error fires before build_handle().
        assert!(
            matches!(
                cuda_eval_batch(&div_op, &[vec![2.0]]),
                Err(CudaEvalError::Unsupported(_))
            ),
            "cuda_eval_batch(Div) must return Unsupported without touching GPU"
        );

        // Pow — also unsupported.
        let pow_op = LoweredOp::Pow(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Const(2.0)));
        assert!(
            matches!(
                generate_ptx(&pow_op, 1, 1),
                Err(CudaEvalError::Unsupported(_))
            ),
            "generate_ptx(Pow) must return Unsupported"
        );

        // Ln — transcendental, unsupported.
        let ln_op = LoweredOp::Ln(Box::new(LoweredOp::Var(0)));
        assert!(
            matches!(
                generate_ptx(&ln_op, 1, 0),
                Err(CudaEvalError::Unsupported(_))
            ),
            "generate_ptx(Ln) must return Unsupported"
        );

        // Sin buried at depth 3 inside an otherwise supported tree — the
        // validator must find it and reject the whole expression.
        let nested_unsupported = LoweredOp::Add(
            Box::new(LoweredOp::Mul(
                Box::new(LoweredOp::Var(0)),
                Box::new(LoweredOp::Const(2.0)),
            )),
            Box::new(LoweredOp::Sin(Box::new(LoweredOp::Var(1)))),
        );
        assert!(
            matches!(
                generate_ptx(&nested_unsupported, 2, 1),
                Err(CudaEvalError::Unsupported(_))
            ),
            "generate_ptx with deeply buried Sin must return Unsupported"
        );
    }
}
