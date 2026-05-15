//! Cranelift JIT compiler for [`LoweredOp`].
//!
//! Compiles a `LoweredOp` formula to a native function with the C ABI
//! signature `extern "C" fn(*const f64, usize) -> f64`. The first
//! argument is a pointer into the variable bindings slice; the second
//! is its length (used for an upfront safety bounds check on the host
//! side, not by the emitted code).
//!
//! # Adapted from oxieml v0.1.0, `src/jit.rs`
//!
//! The IR-emission scaffolding (signature construction, libm symbol
//! registration, `JITModule` lifecycle, `extern "C"` ABI) is ported with
//! attribution. Two divergences:
//!
//! 1. We walk the **flat post-order tape** produced by
//!    [`LoweredOp::to_oxi_ops`] rather than recursing on the algebraic
//!    tree — same semantics, simpler loop, already iterative.
//! 2. We add native [`OxiOp::Sqrt`] / [`OxiOp::Abs`] handlers using
//!    Cranelift's direct `sqrt` and `fabs` intrinsics — oxieml lowers
//!    these to `Pow(_, 0.5)` and `sqrt(x²)`, but our `LoweredOp` keeps
//!    them native (see `eml::op` rustdoc).

#![cfg(feature = "jit")]

// Adapted from oxieml v0.1.0, src/jit.rs
//
// The libm symbol-registration pattern, declare_extern_fns scaffolding,
// and call_extern1/call_extern2 helpers are ported with attribution.
// Native Sqrt/Abs branches and the to_oxi_ops tape-driven emit loop are
// our own.

use crate::eml::op::{LoweredOp, OxiOp};
use cranelift_codegen::ir::types::F64;
use cranelift_codegen::ir::{AbiParam, Function, InstBuilder, MemFlags, Signature, UserFuncName};
use cranelift_codegen::ir::{FuncRef, Value};
use cranelift_codegen::isa::CallConv;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{FuncId, Linkage, Module};
use std::sync::Arc;
use thiserror::Error;

/// JIT compilation errors.
#[derive(Debug, Error)]
pub enum JitError {
    /// The Cranelift backend failed at some stage of code emission.
    #[error("Cranelift error: {0}")]
    Cranelift(String),

    /// Attempt to evaluate with a `vars` slice shorter than the formula
    /// requires (the largest `Var(i)` index plus one).
    #[error("JIT eval needs {needed} variable bindings, got {provided}")]
    NotEnoughVars {
        /// Required minimum slice length.
        needed: usize,
        /// Length of the slice the caller passed.
        provided: usize,
    },

    /// The compiled tape produced no result on the value stack — should
    /// never fire on well-formed `LoweredOp` input; surfaced as an error
    /// rather than a panic to honour the no-`unwrap()` policy.
    #[error("internal: empty value stack after JIT emit (malformed LoweredOp?)")]
    EmptyResult,

    /// A `Var(i)` index is too large to encode as an `i32` byte offset
    /// (`i * 8 > i32::MAX`). At ~268 million variables, no real workload
    /// hits this; treated as an error rather than a silent overflow.
    #[error("Var index {0} too large to encode as JIT memory offset")]
    VarIndexOverflow(usize),
}

/// A JIT-compiled `LoweredOp` ready for repeated evaluation.
///
/// The compiled machine code is owned by an internal Cranelift
/// [`JITModule`]; dropping `JitFunction` releases the code mapping.
/// Use [`Arc`] (e.g. via [`crate::compile::cache::JitCache`]) to share
/// across threads.
pub struct JitFunction {
    /// Raw native function pointer. Lifetime tied to `_module` below.
    func: unsafe extern "C" fn(*const f64, usize) -> f64,
    /// Minimum required variable-slice length.
    n_vars: usize,
    /// Owns the `JITModule`; dropping invalidates the function pointer.
    /// `_` because we never read it back — its `Drop` does the work.
    _module: JITModule,
}

// SAFETY: After `module.finalize_definitions()`, the JITModule no longer
// mutates shared state — the machine code is mapped read-execute. The
// raw function pointer is a read-only reference into that immutable
// mapping. Concurrent reads from multiple threads are sound.
unsafe impl Send for JitFunction {}
// SAFETY: `JitFunction::eval` is a pure function of its inputs; the
// emitted code only reads from `vars[0..n_vars]` via the pointer and
// has no mutable shared state. Concurrent calls are sound.
unsafe impl Sync for JitFunction {}

impl JitFunction {
    /// Evaluate the compiled function at the given variable bindings.
    ///
    /// Returns [`JitError::NotEnoughVars`] if `vars` is shorter than the
    /// minimum the formula requires (the largest `Var(i)` index + 1).
    pub fn eval_checked(&self, vars: &[f64]) -> Result<f64, JitError> {
        if vars.len() < self.n_vars {
            return Err(JitError::NotEnoughVars {
                needed: self.n_vars,
                provided: vars.len(),
            });
        }
        // SAFETY: The emitted code only reads `vars[0..n_vars]` via the
        // raw pointer; the bounds check above guarantees that range is
        // in-bounds for `vars`.
        Ok(unsafe { (self.func)(vars.as_ptr(), vars.len()) })
    }

    /// Evaluate without bounds checking (debug-asserts the length).
    ///
    /// Panics in debug builds when `vars.len() < self.n_vars`. Use
    /// [`Self::eval_checked`] when the caller needs a `Result`.
    pub fn eval(&self, vars: &[f64]) -> f64 {
        debug_assert!(
            vars.len() >= self.n_vars,
            "JitFunction::eval needs {} vars, got {}",
            self.n_vars,
            vars.len()
        );
        // SAFETY: The debug assert documents the precondition; the
        // emitted code reads only `vars[0..n_vars]`.
        unsafe { (self.func)(vars.as_ptr(), vars.len()) }
    }

    /// Minimum required `vars` slice length (largest `Var(i)` + 1).
    pub fn n_vars(&self) -> usize {
        self.n_vars
    }
}

/// Compile a [`LoweredOp`] to a JIT [`JitFunction`].
///
/// The returned function takes an `&[f64]` of variable bindings indexed
/// by the `Var(i)` indices in the source `LoweredOp`. Compilation is
/// pure: no global state mutated, two compiles of the same `LoweredOp`
/// produce equivalent (though not pointer-equal) functions.
///
/// # Errors
///
/// Returns [`JitError::Cranelift`] on any Cranelift-internal failure,
/// [`JitError::VarIndexOverflow`] if a `Var(i)` index would overflow
/// the IR's `i32` memory offset, or [`JitError::EmptyResult`] on an
/// internally malformed `LoweredOp`.
pub fn to_jit(op: &LoweredOp) -> Result<JitFunction, JitError> {
    // Flatten to post-order tape — reuses the existing iterative walk
    // from `eml::op`.
    let tape = op.to_oxi_ops();
    let n_vars = op.count_vars();

    // ── JIT module ─────────────────────────────────────────────────────
    let mut jit_builder = JITBuilder::new(cranelift_module::default_libcall_names())
        .map_err(|e| JitError::Cranelift(format!("JITBuilder::new: {e}")))?;

    // Register libm symbols explicitly. Without this, dynamic-linker
    // resolution fails on platforms where libm is not in the default
    // search path (notably musl-libc static builds).
    register_libm_symbols(&mut jit_builder);

    let mut module = JITModule::new(jit_builder);
    let call_conv = module.target_config().default_call_conv;
    let ptr_ty = module.target_config().pointer_type();

    // ── Main signature: extern "C" fn(*const f64, usize) -> f64 ───────
    let mut main_sig = Signature::new(call_conv);
    main_sig.params.push(AbiParam::new(ptr_ty));
    main_sig.params.push(AbiParam::new(ptr_ty));
    main_sig.returns.push(AbiParam::new(F64));

    let main_func_id = module
        .declare_function("__scirs2_symbolic_jit_eval", Linkage::Local, &main_sig)
        .map_err(|e| JitError::Cranelift(format!("declare_function: {e}")))?;

    // ── External libm declarations ─────────────────────────────────────
    let extern_ids = declare_extern_fns(&mut module, call_conv)?;

    // ── Build the IR ──────────────────────────────────────────────────
    let mut ctx = module.make_context();
    ctx.func =
        Function::with_name_signature(UserFuncName::user(0, main_func_id.as_u32()), main_sig);

    {
        let mut fb_ctx = FunctionBuilderContext::new();
        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fb_ctx);

        let entry_block = builder.create_block();
        builder.append_block_params_for_function_params(entry_block);
        builder.switch_to_block(entry_block);
        builder.seal_block(entry_block);

        let vars_ptr = builder.block_params(entry_block)[0];

        // Pre-declare each external libm function as a FuncRef once, so
        // we don't redeclare per-call.
        let extern_refs = extern_ids.declare_in_function(&mut module, &mut builder);

        let mut vstack: Vec<Value> = Vec::with_capacity(tape.len().min(64));

        for opcode in &tape {
            emit_op(opcode, &mut builder, &mut vstack, vars_ptr, &extern_refs)?;
        }

        let result = vstack.pop().ok_or(JitError::EmptyResult)?;

        builder.ins().return_(&[result]);
        builder.finalize();
    }

    // ── Compile + finalize ────────────────────────────────────────────
    module
        .define_function(main_func_id, &mut ctx)
        .map_err(|e| JitError::Cranelift(format!("define_function: {e}")))?;
    module.clear_context(&mut ctx);
    module
        .finalize_definitions()
        .map_err(|e| JitError::Cranelift(format!("finalize_definitions: {e}")))?;

    let raw_ptr = module.get_finalized_function(main_func_id);
    // SAFETY: The pointer is valid for the lifetime of `module`, which
    // we move into `JitFunction` immediately below.
    let func: unsafe extern "C" fn(*const f64, usize) -> f64 =
        unsafe { std::mem::transmute(raw_ptr) };

    Ok(JitFunction {
        func,
        n_vars,
        _module: module,
    })
}

// ─── libm symbol registration ────────────────────────────────────────────

/// Register the libm functions we lower transcendentals to.
///
/// `f64::ln` is double-cast through `fn(f64) -> f64` because Cranelift
/// requires a `*const u8`-castable function pointer; the intrinsic
/// methods on `f64` are dispatched via the trait system on stable.
fn register_libm_symbols(builder: &mut JITBuilder) {
    builder.symbol("exp", f64::exp as *const u8);
    builder.symbol("log", (f64::ln as fn(f64) -> f64) as *const u8);
    builder.symbol("sin", f64::sin as *const u8);
    builder.symbol("cos", f64::cos as *const u8);
    builder.symbol("tan", f64::tan as *const u8);
    builder.symbol("sinh", f64::sinh as *const u8);
    builder.symbol("cosh", f64::cosh as *const u8);
    builder.symbol("tanh", f64::tanh as *const u8);
    builder.symbol("asin", f64::asin as *const u8);
    builder.symbol("acos", f64::acos as *const u8);
    builder.symbol("atan", f64::atan as *const u8);
    builder.symbol("asinh", f64::asinh as *const u8);
    builder.symbol("acosh", f64::acosh as *const u8);
    builder.symbol("atanh", f64::atanh as *const u8);
    builder.symbol("pow", f64::powf as *const u8);
}

// ─── External function declarations ──────────────────────────────────────

/// Holds the `FuncId` for every external libm function we may emit.
struct ExternIds {
    exp: FuncId,
    log: FuncId,
    sin: FuncId,
    cos: FuncId,
    tan: FuncId,
    sinh: FuncId,
    cosh: FuncId,
    tanh: FuncId,
    asin: FuncId,
    acos: FuncId,
    atan: FuncId,
    asinh: FuncId,
    acosh: FuncId,
    atanh: FuncId,
    pow: FuncId,
}

/// Per-function libm `FuncRef`s (the in-function declarations).
struct ExternRefs {
    exp: FuncRef,
    log: FuncRef,
    sin: FuncRef,
    cos: FuncRef,
    tan: FuncRef,
    sinh: FuncRef,
    cosh: FuncRef,
    tanh: FuncRef,
    asin: FuncRef,
    acos: FuncRef,
    atan: FuncRef,
    asinh: FuncRef,
    acosh: FuncRef,
    atanh: FuncRef,
    pow: FuncRef,
}

impl ExternIds {
    fn declare_in_function(
        &self,
        module: &mut JITModule,
        builder: &mut FunctionBuilder<'_>,
    ) -> ExternRefs {
        ExternRefs {
            exp: module.declare_func_in_func(self.exp, builder.func),
            log: module.declare_func_in_func(self.log, builder.func),
            sin: module.declare_func_in_func(self.sin, builder.func),
            cos: module.declare_func_in_func(self.cos, builder.func),
            tan: module.declare_func_in_func(self.tan, builder.func),
            sinh: module.declare_func_in_func(self.sinh, builder.func),
            cosh: module.declare_func_in_func(self.cosh, builder.func),
            tanh: module.declare_func_in_func(self.tanh, builder.func),
            asin: module.declare_func_in_func(self.asin, builder.func),
            acos: module.declare_func_in_func(self.acos, builder.func),
            atan: module.declare_func_in_func(self.atan, builder.func),
            asinh: module.declare_func_in_func(self.asinh, builder.func),
            acosh: module.declare_func_in_func(self.acosh, builder.func),
            atanh: module.declare_func_in_func(self.atanh, builder.func),
            pow: module.declare_func_in_func(self.pow, builder.func),
        }
    }
}

fn sig_unary(call_conv: CallConv) -> Signature {
    let mut sig = Signature::new(call_conv);
    sig.params.push(AbiParam::new(F64));
    sig.returns.push(AbiParam::new(F64));
    sig
}

fn sig_binary(call_conv: CallConv) -> Signature {
    let mut sig = Signature::new(call_conv);
    sig.params.push(AbiParam::new(F64));
    sig.params.push(AbiParam::new(F64));
    sig.returns.push(AbiParam::new(F64));
    sig
}

fn declare_extern_fns(module: &mut JITModule, call_conv: CallConv) -> Result<ExternIds, JitError> {
    let s_un = sig_unary(call_conv);
    let s_bi = sig_binary(call_conv);

    let mut decl = |name: &str, sig: &Signature| -> Result<FuncId, JitError> {
        module
            .declare_function(name, Linkage::Import, sig)
            .map_err(|e| JitError::Cranelift(format!("declare {name}: {e}")))
    };

    Ok(ExternIds {
        exp: decl("exp", &s_un)?,
        log: decl("log", &s_un)?,
        sin: decl("sin", &s_un)?,
        cos: decl("cos", &s_un)?,
        tan: decl("tan", &s_un)?,
        sinh: decl("sinh", &s_un)?,
        cosh: decl("cosh", &s_un)?,
        tanh: decl("tanh", &s_un)?,
        asin: decl("asin", &s_un)?,
        acos: decl("acos", &s_un)?,
        atan: decl("atan", &s_un)?,
        asinh: decl("asinh", &s_un)?,
        acosh: decl("acosh", &s_un)?,
        atanh: decl("atanh", &s_un)?,
        pow: decl("pow", &s_bi)?,
    })
}

// ─── IR emission ────────────────────────────────────────────────────────

fn emit_op(
    opcode: &OxiOp,
    builder: &mut FunctionBuilder<'_>,
    vstack: &mut Vec<Value>,
    vars_ptr: Value,
    extern_refs: &ExternRefs,
) -> Result<(), JitError> {
    match opcode {
        OxiOp::Const(c) => {
            let v = builder.ins().f64const(*c);
            vstack.push(v);
        }
        OxiOp::Var(i) => {
            let byte_offset = i.checked_mul(8).ok_or(JitError::VarIndexOverflow(*i))?;
            let offset = i32::try_from(byte_offset).map_err(|_| JitError::VarIndexOverflow(*i))?;
            let v = builder
                .ins()
                .load(F64, MemFlags::trusted(), vars_ptr, offset);
            vstack.push(v);
        }
        OxiOp::Add => {
            let (a, b) = pop2(vstack)?;
            vstack.push(builder.ins().fadd(a, b));
        }
        OxiOp::Sub => {
            let (a, b) = pop2(vstack)?;
            vstack.push(builder.ins().fsub(a, b));
        }
        OxiOp::Mul => {
            let (a, b) = pop2(vstack)?;
            vstack.push(builder.ins().fmul(a, b));
        }
        OxiOp::Div => {
            let (a, b) = pop2(vstack)?;
            vstack.push(builder.ins().fdiv(a, b));
        }
        OxiOp::Neg => {
            let a = pop1(vstack)?;
            vstack.push(builder.ins().fneg(a));
        }
        OxiOp::Sqrt => {
            // Native — Cranelift `sqrt` intrinsic, no libm round-trip.
            let a = pop1(vstack)?;
            vstack.push(builder.ins().sqrt(a));
        }
        OxiOp::Abs => {
            // Native — Cranelift `fabs` intrinsic.
            let a = pop1(vstack)?;
            vstack.push(builder.ins().fabs(a));
        }
        OxiOp::Exp => {
            let a = pop1(vstack)?;
            vstack.push(call_unary(builder, extern_refs.exp, a)?);
        }
        OxiOp::Ln => {
            let a = pop1(vstack)?;
            vstack.push(call_unary(builder, extern_refs.log, a)?);
        }
        OxiOp::Sin => {
            let a = pop1(vstack)?;
            vstack.push(call_unary(builder, extern_refs.sin, a)?);
        }
        OxiOp::Cos => {
            let a = pop1(vstack)?;
            vstack.push(call_unary(builder, extern_refs.cos, a)?);
        }
        OxiOp::Tan => {
            let a = pop1(vstack)?;
            vstack.push(call_unary(builder, extern_refs.tan, a)?);
        }
        OxiOp::Sinh => {
            let a = pop1(vstack)?;
            vstack.push(call_unary(builder, extern_refs.sinh, a)?);
        }
        OxiOp::Cosh => {
            let a = pop1(vstack)?;
            vstack.push(call_unary(builder, extern_refs.cosh, a)?);
        }
        OxiOp::Tanh => {
            let a = pop1(vstack)?;
            vstack.push(call_unary(builder, extern_refs.tanh, a)?);
        }
        OxiOp::Arcsin => {
            let a = pop1(vstack)?;
            vstack.push(call_unary(builder, extern_refs.asin, a)?);
        }
        OxiOp::Arccos => {
            let a = pop1(vstack)?;
            vstack.push(call_unary(builder, extern_refs.acos, a)?);
        }
        OxiOp::Arctan => {
            let a = pop1(vstack)?;
            vstack.push(call_unary(builder, extern_refs.atan, a)?);
        }
        OxiOp::Arcsinh => {
            let a = pop1(vstack)?;
            vstack.push(call_unary(builder, extern_refs.asinh, a)?);
        }
        OxiOp::Arccosh => {
            let a = pop1(vstack)?;
            vstack.push(call_unary(builder, extern_refs.acosh, a)?);
        }
        OxiOp::Arctanh => {
            let a = pop1(vstack)?;
            vstack.push(call_unary(builder, extern_refs.atanh, a)?);
        }
        OxiOp::Pow => {
            let (a, b) = pop2(vstack)?;
            vstack.push(call_binary(builder, extern_refs.pow, a, b)?);
        }
    }
    Ok(())
}

fn pop1(vstack: &mut Vec<Value>) -> Result<Value, JitError> {
    vstack.pop().ok_or(JitError::EmptyResult)
}

fn pop2(vstack: &mut Vec<Value>) -> Result<(Value, Value), JitError> {
    let b = vstack.pop().ok_or(JitError::EmptyResult)?;
    let a = vstack.pop().ok_or(JitError::EmptyResult)?;
    Ok((a, b))
}

fn call_unary(
    builder: &mut FunctionBuilder<'_>,
    func_ref: FuncRef,
    arg: Value,
) -> Result<Value, JitError> {
    let call = builder.ins().call(func_ref, &[arg]);
    let results = builder.inst_results(call);
    results.first().copied().ok_or(JitError::EmptyResult)
}

fn call_binary(
    builder: &mut FunctionBuilder<'_>,
    func_ref: FuncRef,
    a: Value,
    b: Value,
) -> Result<Value, JitError> {
    let call = builder.ins().call(func_ref, &[a, b]);
    let results = builder.inst_results(call);
    results.first().copied().ok_or(JitError::EmptyResult)
}

// ─── thread-safety: cache uses Arc<JitFunction> ──────────────────────────

/// Convenience shorthand: `Arc<JitFunction>` for cache use.
pub type SharedJit = Arc<JitFunction>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eml::eval::{eval_real, EvalCtx};

    fn b(op: LoweredOp) -> Box<LoweredOp> {
        Box::new(op)
    }

    #[test]
    fn jit_const() {
        let op = LoweredOp::Const(2.5);
        let f = to_jit(&op).expect("compile");
        assert_eq!(f.eval(&[]), 2.5);
        assert_eq!(f.n_vars(), 0);
    }

    #[test]
    fn jit_var() {
        let op = LoweredOp::Var(0);
        let f = to_jit(&op).expect("compile");
        assert_eq!(f.eval(&[42.0]), 42.0);
        assert_eq!(f.n_vars(), 1);
    }

    #[test]
    fn jit_add_two_vars() {
        let op = LoweredOp::Add(b(LoweredOp::Var(0)), b(LoweredOp::Var(1)));
        let f = to_jit(&op).expect("compile");
        assert_eq!(f.eval(&[3.0, 4.0]), 7.0);
        assert_eq!(f.n_vars(), 2);
    }

    #[test]
    fn jit_sub() {
        let op = LoweredOp::Sub(b(LoweredOp::Var(0)), b(LoweredOp::Var(1)));
        let f = to_jit(&op).expect("compile");
        assert_eq!(f.eval(&[10.0, 3.0]), 7.0);
    }

    #[test]
    fn jit_mul() {
        let op = LoweredOp::Mul(b(LoweredOp::Var(0)), b(LoweredOp::Var(1)));
        let f = to_jit(&op).expect("compile");
        assert_eq!(f.eval(&[6.0, 7.0]), 42.0);
    }

    #[test]
    fn jit_div() {
        let op = LoweredOp::Div(b(LoweredOp::Var(0)), b(LoweredOp::Var(1)));
        let f = to_jit(&op).expect("compile");
        assert_eq!(f.eval(&[20.0, 4.0]), 5.0);
    }

    #[test]
    fn jit_neg() {
        let op = LoweredOp::Neg(b(LoweredOp::Var(0)));
        let f = to_jit(&op).expect("compile");
        assert_eq!(f.eval(&[7.0]), -7.0);
    }

    #[test]
    fn jit_quadratic() {
        // f(x) = x² + 2x + 1
        let x = LoweredOp::Var(0);
        let x_sq = LoweredOp::Mul(b(x.clone()), b(x.clone()));
        let two_x = LoweredOp::Mul(b(LoweredOp::Const(2.0)), b(x));
        let one = LoweredOp::Const(1.0);
        let f_op = LoweredOp::Add(b(LoweredOp::Add(b(x_sq), b(two_x))), b(one));
        let f = to_jit(&f_op).expect("compile");
        // f(3) = 9 + 6 + 1 = 16
        assert_eq!(f.eval(&[3.0]), 16.0);
        // f(0) = 1
        assert_eq!(f.eval(&[0.0]), 1.0);
    }

    #[test]
    fn jit_sqrt_native() {
        let op = LoweredOp::Sqrt(b(LoweredOp::Var(0)));
        let f = to_jit(&op).expect("compile");
        assert!((f.eval(&[4.0]) - 2.0).abs() < 1e-15);
        assert!((f.eval(&[2.0]) - std::f64::consts::SQRT_2).abs() < 1e-15);
    }

    #[test]
    fn jit_abs_native() {
        let op = LoweredOp::Abs(b(LoweredOp::Var(0)));
        let f = to_jit(&op).expect("compile");
        assert_eq!(f.eval(&[3.0]), 3.0);
        assert_eq!(f.eval(&[-3.0]), 3.0);
        assert_eq!(f.eval(&[0.0]), 0.0);
    }

    #[test]
    fn jit_sin_libm() {
        // The critical regression test — the planned stub would have
        // returned the input unchanged here.
        let op = LoweredOp::Sin(b(LoweredOp::Var(0)));
        let f = to_jit(&op).expect("compile");
        let x = 0.7;
        let got = f.eval(&[x]);
        let want = x.sin();
        assert!(
            (got - want).abs() < 1e-15,
            "JIT sin({x}) = {got}, libm sin = {want}"
        );
        // Parity with eval_real
        let bindings = vec![x];
        let ctx = EvalCtx::new(&bindings);
        let interp = eval_real(&op, &ctx).expect("interp");
        assert!((got - interp).abs() < 1e-15);
    }

    #[test]
    fn jit_cos_libm() {
        let op = LoweredOp::Cos(b(LoweredOp::Var(0)));
        let f = to_jit(&op).expect("compile");
        let x = 1.2;
        let got = f.eval(&[x]);
        assert!((got - x.cos()).abs() < 1e-15);
    }

    #[test]
    fn jit_exp_libm() {
        let op = LoweredOp::Exp(b(LoweredOp::Var(0)));
        let f = to_jit(&op).expect("compile");
        let x = 1.5;
        assert!((f.eval(&[x]) - x.exp()).abs() < 1e-12);
    }

    #[test]
    fn jit_ln_libm() {
        let op = LoweredOp::Ln(b(LoweredOp::Var(0)));
        let f = to_jit(&op).expect("compile");
        let x = std::f64::consts::E;
        // ln(e) = 1 exactly to machine precision.
        assert!((f.eval(&[x]) - x.ln()).abs() < 1e-12);
        assert!((f.eval(&[x]) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn jit_pow_libm() {
        let op = LoweredOp::Pow(b(LoweredOp::Var(0)), b(LoweredOp::Const(3.0)));
        let f = to_jit(&op).expect("compile");
        let x = 2.0;
        assert!((f.eval(&[x]) - 8.0).abs() < 1e-12);
    }

    #[test]
    fn jit_composite_transcendental() {
        // f(x) = sin(x)² + cos(x)² → 1 (Pythagorean identity)
        let x = LoweredOp::Var(0);
        let s = LoweredOp::Sin(b(x.clone()));
        let c = LoweredOp::Cos(b(x));
        let s2 = LoweredOp::Mul(b(s.clone()), b(s));
        let c2 = LoweredOp::Mul(b(c.clone()), b(c));
        let op = LoweredOp::Add(b(s2), b(c2));
        let f = to_jit(&op).expect("compile");
        for &x in &[0.0_f64, 0.5, 1.0, 1.7, std::f64::consts::PI, -2.4] {
            assert!(
                (f.eval(&[x]) - 1.0).abs() < 1e-14,
                "Pythagorean identity violated at x={x}: got {}",
                f.eval(&[x])
            );
        }
    }

    #[test]
    fn jit_parity_with_eval_real_random_formula() {
        // f(x, y) = sqrt(x² + y²) + ln(1 + exp(x))
        let x = LoweredOp::Var(0);
        let y = LoweredOp::Var(1);
        let x2 = LoweredOp::Mul(b(x.clone()), b(x.clone()));
        let y2 = LoweredOp::Mul(b(y.clone()), b(y));
        let r = LoweredOp::Sqrt(b(LoweredOp::Add(b(x2), b(y2))));
        let softplus = LoweredOp::Ln(b(LoweredOp::Add(
            b(LoweredOp::Const(1.0)),
            b(LoweredOp::Exp(b(x))),
        )));
        let op = LoweredOp::Add(b(r), b(softplus));
        let f = to_jit(&op).expect("compile");
        for (xv, yv) in [(1.0, 2.0), (0.5, -0.3), (-1.5, 4.0), (0.0, 0.0)] {
            let bindings = vec![xv, yv];
            let ctx = EvalCtx::new(&bindings);
            let interp = eval_real(&op, &ctx).expect("interp");
            let jit = f.eval(&[xv, yv]);
            assert!(
                (interp - jit).abs() < 1e-12,
                "parity broken at ({xv}, {yv}): interp={interp} jit={jit}"
            );
        }
    }

    #[test]
    fn jit_eval_checked_too_few_vars() {
        let op = LoweredOp::Add(b(LoweredOp::Var(0)), b(LoweredOp::Var(1)));
        let f = to_jit(&op).expect("compile");
        let res = f.eval_checked(&[1.0]);
        match res {
            Err(JitError::NotEnoughVars { needed, provided }) => {
                assert_eq!(needed, 2);
                assert_eq!(provided, 1);
            }
            other => panic!("expected NotEnoughVars, got {other:?}"),
        }
    }

    #[test]
    fn jit_handles_inverse_trig() {
        // arctan(tan(x)) = x for x in (-π/2, π/2)
        let x = LoweredOp::Var(0);
        let inner = LoweredOp::Tan(b(x));
        let op = LoweredOp::Arctan(b(inner));
        let f = to_jit(&op).expect("compile");
        for &xv in &[-1.4_f64, -0.5, 0.0, 0.5, 1.4] {
            let got = f.eval(&[xv]);
            assert!((got - xv).abs() < 1e-12, "arctan(tan({xv})) = {got}");
        }
    }

    #[test]
    fn jit_handles_hyperbolic() {
        // sinh(x)² - cosh(x)² = -1
        let x = LoweredOp::Var(0);
        let s = LoweredOp::Sinh(b(x.clone()));
        let c = LoweredOp::Cosh(b(x));
        let s2 = LoweredOp::Mul(b(s.clone()), b(s));
        let c2 = LoweredOp::Mul(b(c.clone()), b(c));
        let op = LoweredOp::Sub(b(s2), b(c2));
        let f = to_jit(&op).expect("compile");
        for &xv in &[0.0_f64, 0.7, -1.2, 2.5] {
            let got = f.eval(&[xv]);
            assert!(
                (got - (-1.0)).abs() < 1e-12,
                "sinh²({xv})-cosh²({xv}) = {got}"
            );
        }
    }

    #[test]
    fn jit_send_sync_marker() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<JitFunction>();
        assert_send_sync::<SharedJit>();
    }
}
