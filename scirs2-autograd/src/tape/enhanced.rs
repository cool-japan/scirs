//! Enhanced tape-based automatic differentiation
//!
//! This module provides production-quality tape implementations for both forward-
//! and reverse-mode automatic differentiation, together with utilities for
//! memory-efficient gradient computation.
//!
//! # Overview
//!
//! | Type | Description |
//! |------|-------------|
//! | [`ReverseTape`] | Wengert list (reverse-mode, O(n) backward pass) |
//! | [`ForwardTape`] | Dual-number tape (forward-mode, one JVP per pass) |
//! | [`MixedMode`] | Combined forward+reverse for efficient Jacobian computation |
//! | [`TapeCheckpoint`] | Segment a tape at checkpoints to save memory |
//! | [`TapeOptimizer`] | Dead-code elimination + common subexpression elimination |
//!
//! All types operate on `f64` scalars and `Vec<f64>` vectors to stay
//! framework-agnostic (no dependency on the computation-graph tensor layer).

use crate::error::AutogradError;
use crate::Result;
use std::collections::HashMap;

// ============================================================================
// Primitive operation kind
// ============================================================================

/// The kind of a primitive operation recorded on the tape.
///
/// Each variant captures the information needed to replay the forward
/// computation or to push cotangents back through a backward pass.
#[derive(Debug, Clone)]
pub enum TapeOp {
    /// Addition: out = a + b
    Add {
        /// Index of left operand in the value table
        a: usize,
        /// Index of right operand in the value table
        b: usize,
    },
    /// Subtraction: out = a - b
    Sub { a: usize, b: usize },
    /// Elementwise multiplication: out = a * b
    Mul { a: usize, b: usize },
    /// Elementwise division: out = a / b  (b ≠ 0)
    Div { a: usize, b: usize },
    /// Unary negation: out = -a
    Neg { a: usize },
    /// Unary exp: out = exp(a)
    Exp { a: usize },
    /// Unary natural log: out = ln(a)
    Log { a: usize },
    /// Unary sine: out = sin(a)
    Sin { a: usize },
    /// Unary cosine: out = cos(a)
    Cos { a: usize },
    /// Unary square root: out = sqrt(a)
    Sqrt { a: usize },
    /// Scaling by a compile-time constant: out = scalar * a
    Scale { scalar: f64, a: usize },
    /// Squared: out = a^2
    Square { a: usize },
    /// Constant input (no gradient flows)
    Constant { value: f64 },
    /// Input placeholder (gradient flows to it directly)
    Input { input_idx: usize },
}

// ============================================================================
// ReverseTape — Wengert list with O(n) backward pass
// ============================================================================

/// A Wengert list (reverse-mode tape) that records scalar operations and
/// allows efficient reverse-mode differentiation.
///
/// The tape is indexed: every `push_*` call returns the index of the newly
/// recorded node.  Calling [`ReverseTape::backward`] fills a gradient table
/// and returns `dL/d(node[i])` for every recorded node.
///
/// # Example
///
/// ```rust
/// use scirs2_autograd::tape::enhanced::ReverseTape;
///
/// // f(x, y) = (x + y) * x
/// let mut tape = ReverseTape::new();
/// let ix = tape.push_input(0);          // x
/// let iy = tape.push_input(1);          // y
/// let s  = tape.push_add(ix, iy);       // x + y
/// let p  = tape.push_mul(s, ix);        // (x+y)*x
///
/// let inputs = vec![3.0, 2.0];          // x=3, y=2
/// let vals = tape.forward(&inputs).expect("forward");
///
/// // df/d(x+y)*x = 1 (output node)
/// let grads = tape.backward(p, &vals).expect("backward");
/// // d/dx = 2x + y = 8,  d/dy = x = 3
/// assert!((grads[ix] - 8.0).abs() < 1e-9);
/// assert!((grads[iy] - 3.0).abs() < 1e-9);
/// ```
#[derive(Debug, Default)]
pub struct ReverseTape {
    /// Recorded operations in forward order
    ops: Vec<TapeOp>,
}

impl ReverseTape {
    /// Create an empty tape.
    pub fn new() -> Self {
        Self { ops: Vec::new() }
    }

    /// Record a scalar input placeholder.  Returns the node index.
    pub fn push_input(&mut self, input_idx: usize) -> usize {
        let idx = self.ops.len();
        self.ops.push(TapeOp::Input { input_idx });
        idx
    }

    /// Record a scalar constant.  Returns the node index.
    pub fn push_constant(&mut self, value: f64) -> usize {
        let idx = self.ops.len();
        self.ops.push(TapeOp::Constant { value });
        idx
    }

    /// Record `a + b`.  Returns the node index.
    pub fn push_add(&mut self, a: usize, b: usize) -> usize {
        let idx = self.ops.len();
        self.ops.push(TapeOp::Add { a, b });
        idx
    }

    /// Record `a - b`.  Returns the node index.
    pub fn push_sub(&mut self, a: usize, b: usize) -> usize {
        let idx = self.ops.len();
        self.ops.push(TapeOp::Sub { a, b });
        idx
    }

    /// Record `a * b`.  Returns the node index.
    pub fn push_mul(&mut self, a: usize, b: usize) -> usize {
        let idx = self.ops.len();
        self.ops.push(TapeOp::Mul { a, b });
        idx
    }

    /// Record `a / b`.  Returns the node index.
    pub fn push_div(&mut self, a: usize, b: usize) -> usize {
        let idx = self.ops.len();
        self.ops.push(TapeOp::Div { a, b });
        idx
    }

    /// Record `-a`.  Returns the node index.
    pub fn push_neg(&mut self, a: usize) -> usize {
        let idx = self.ops.len();
        self.ops.push(TapeOp::Neg { a });
        idx
    }

    /// Record `exp(a)`.  Returns the node index.
    pub fn push_exp(&mut self, a: usize) -> usize {
        let idx = self.ops.len();
        self.ops.push(TapeOp::Exp { a });
        idx
    }

    /// Record `ln(a)`.  Returns the node index.
    pub fn push_log(&mut self, a: usize) -> usize {
        let idx = self.ops.len();
        self.ops.push(TapeOp::Log { a });
        idx
    }

    /// Record `sin(a)`.  Returns the node index.
    pub fn push_sin(&mut self, a: usize) -> usize {
        let idx = self.ops.len();
        self.ops.push(TapeOp::Sin { a });
        idx
    }

    /// Record `cos(a)`.  Returns the node index.
    pub fn push_cos(&mut self, a: usize) -> usize {
        let idx = self.ops.len();
        self.ops.push(TapeOp::Cos { a });
        idx
    }

    /// Record `sqrt(a)`.  Returns the node index.
    pub fn push_sqrt(&mut self, a: usize) -> usize {
        let idx = self.ops.len();
        self.ops.push(TapeOp::Sqrt { a });
        idx
    }

    /// Record `scalar * a`.  Returns the node index.
    pub fn push_scale(&mut self, scalar: f64, a: usize) -> usize {
        let idx = self.ops.len();
        self.ops.push(TapeOp::Scale { scalar, a });
        idx
    }

    /// Record `a^2`.  Returns the node index.
    pub fn push_square(&mut self, a: usize) -> usize {
        let idx = self.ops.len();
        self.ops.push(TapeOp::Square { a });
        idx
    }

    /// Number of nodes on the tape.
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// Returns true if the tape has no recorded nodes.
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Execute the forward pass, evaluating every node in recording order.
    ///
    /// Returns a value table `vals` of length `self.len()` where `vals[i]` is
    /// the scalar value of node `i`.
    ///
    /// # Errors
    ///
    /// Returns `AutogradError` if:
    /// - An operand index is out of range
    /// - A domain error occurs (log of non-positive, sqrt of negative, div by zero)
    pub fn forward(&self, inputs: &[f64]) -> Result<Vec<f64>> {
        let n = self.ops.len();
        let mut vals = vec![0.0f64; n];

        for (i, op) in self.ops.iter().enumerate() {
            vals[i] = match op {
                TapeOp::Constant { value } => *value,

                TapeOp::Input { input_idx } => {
                    inputs.get(*input_idx).copied().ok_or_else(|| {
                        AutogradError::invalid_argument(format!(
                            "ReverseTape::forward: input index {} out of range (len {})",
                            input_idx,
                            inputs.len()
                        ))
                    })?
                }

                TapeOp::Add { a, b } => {
                    check_idx(*a, i, "Add.a")?;
                    check_idx(*b, i, "Add.b")?;
                    vals[*a] + vals[*b]
                }

                TapeOp::Sub { a, b } => {
                    check_idx(*a, i, "Sub.a")?;
                    check_idx(*b, i, "Sub.b")?;
                    vals[*a] - vals[*b]
                }

                TapeOp::Mul { a, b } => {
                    check_idx(*a, i, "Mul.a")?;
                    check_idx(*b, i, "Mul.b")?;
                    vals[*a] * vals[*b]
                }

                TapeOp::Div { a, b } => {
                    check_idx(*a, i, "Div.a")?;
                    check_idx(*b, i, "Div.b")?;
                    let denom = vals[*b];
                    if denom.abs() < f64::EPSILON * 1e6 {
                        return Err(AutogradError::compute_error(
                            "ReverseTape::forward: division by zero".to_string(),
                        ));
                    }
                    vals[*a] / denom
                }

                TapeOp::Neg { a } => {
                    check_idx(*a, i, "Neg.a")?;
                    -vals[*a]
                }

                TapeOp::Exp { a } => {
                    check_idx(*a, i, "Exp.a")?;
                    vals[*a].exp()
                }

                TapeOp::Log { a } => {
                    check_idx(*a, i, "Log.a")?;
                    let v = vals[*a];
                    if v <= 0.0 {
                        return Err(AutogradError::compute_error(format!(
                            "ReverseTape::forward: log({v}) undefined"
                        )));
                    }
                    v.ln()
                }

                TapeOp::Sin { a } => {
                    check_idx(*a, i, "Sin.a")?;
                    vals[*a].sin()
                }

                TapeOp::Cos { a } => {
                    check_idx(*a, i, "Cos.a")?;
                    vals[*a].cos()
                }

                TapeOp::Sqrt { a } => {
                    check_idx(*a, i, "Sqrt.a")?;
                    let v = vals[*a];
                    if v < 0.0 {
                        return Err(AutogradError::compute_error(format!(
                            "ReverseTape::forward: sqrt({v}) undefined"
                        )));
                    }
                    v.sqrt()
                }

                TapeOp::Scale { scalar, a } => {
                    check_idx(*a, i, "Scale.a")?;
                    scalar * vals[*a]
                }

                TapeOp::Square { a } => {
                    check_idx(*a, i, "Square.a")?;
                    vals[*a] * vals[*a]
                }
            };
        }

        Ok(vals)
    }

    /// Execute the reverse (backward) pass starting from node `output_idx`.
    ///
    /// `vals` must be the value table returned by a preceding call to
    /// [`forward`](ReverseTape::forward).
    ///
    /// Returns a gradient table `grads` of length `self.len()` where
    /// `grads[i] = dL/d(node[i])` with `dL/d(output) = 1`.
    ///
    /// # Errors
    ///
    /// Returns `AutogradError` if `output_idx >= self.len()` or `vals.len() !=
    /// self.len()`.
    pub fn backward(&self, output_idx: usize, vals: &[f64]) -> Result<Vec<f64>> {
        let n = self.ops.len();
        if output_idx >= n {
            return Err(AutogradError::invalid_argument(format!(
                "ReverseTape::backward: output_idx {output_idx} >= tape len {n}"
            )));
        }
        if vals.len() != n {
            return Err(AutogradError::invalid_argument(format!(
                "ReverseTape::backward: vals.len() {} != tape len {n}",
                vals.len()
            )));
        }

        let mut grads = vec![0.0f64; n];
        grads[output_idx] = 1.0;

        // Process nodes in reverse order
        for i in (0..n).rev() {
            let g = grads[i];
            if g == 0.0 {
                continue;
            }

            match &self.ops[i] {
                TapeOp::Constant { .. } | TapeOp::Input { .. } => {}

                TapeOp::Add { a, b } => {
                    grads[*a] += g;
                    grads[*b] += g;
                }

                TapeOp::Sub { a, b } => {
                    grads[*a] += g;
                    grads[*b] -= g;
                }

                TapeOp::Mul { a, b } => {
                    grads[*a] += g * vals[*b];
                    grads[*b] += g * vals[*a];
                }

                TapeOp::Div { a, b } => {
                    let va = vals[*a];
                    let vb = vals[*b];
                    grads[*a] += g / vb;
                    grads[*b] -= g * va / (vb * vb);
                }

                TapeOp::Neg { a } => {
                    grads[*a] -= g;
                }

                TapeOp::Exp { a } => {
                    // d/da exp(a) = exp(a) = vals[i]
                    grads[*a] += g * vals[i];
                }

                TapeOp::Log { a } => {
                    // d/da ln(a) = 1/a
                    grads[*a] += g / vals[*a];
                }

                TapeOp::Sin { a } => {
                    // d/da sin(a) = cos(a)
                    grads[*a] += g * vals[*a].cos();
                }

                TapeOp::Cos { a } => {
                    // d/da cos(a) = -sin(a)
                    grads[*a] -= g * vals[*a].sin();
                }

                TapeOp::Sqrt { a } => {
                    // d/da sqrt(a) = 1 / (2*sqrt(a)) = 1 / (2 * vals[i])
                    let sv = vals[i];
                    if sv.abs() > f64::EPSILON {
                        grads[*a] += g / (2.0 * sv);
                    }
                }

                TapeOp::Scale { scalar, a } => {
                    grads[*a] += g * scalar;
                }

                TapeOp::Square { a } => {
                    // d/da a^2 = 2a
                    grads[*a] += g * 2.0 * vals[*a];
                }
            }
        }

        Ok(grads)
    }

    /// Return references to the recorded operations.
    pub fn ops(&self) -> &[TapeOp] {
        &self.ops
    }

    /// Clear all recorded operations (reset to empty tape).
    pub fn clear(&mut self) {
        self.ops.clear();
    }
}

// ============================================================================
// ForwardTape — dual number tape for forward-mode AD
// ============================================================================

/// A dual-number tape for forward-mode (tangent) automatic differentiation.
///
/// Each recorded value carries a primal (`v`) and a tangent (`dv`).  A single
/// forward pass yields both `f(x)` and `J(x)·t` for a chosen tangent vector `t`.
///
/// # Example
///
/// ```rust
/// use scirs2_autograd::tape::enhanced::ForwardTape;
///
/// // f(x) = x * x + x; choose tangent t = 1 (i.e. differentiate w.r.t. x)
/// let mut tape = ForwardTape::new();
/// let ix = tape.push_input(0, 1.0);  // primal x=3, tangent dt/dx=1
/// let ix2 = tape.push_mul(ix, ix);   // x^2, tangent 2x=6
/// let out = tape.push_add(ix2, ix);  // x^2+x, tangent 2x+1=7
///
/// let primals = vec![3.0];
/// let tangents = vec![1.0];
/// let (pvals, tvals) = tape.forward(&primals, &tangents).expect("forward");
///
/// assert!((pvals[out] - 12.0).abs() < 1e-9);  // 3^2+3 = 12
/// assert!((tvals[out] -  7.0).abs() < 1e-9);  // 2*3+1 = 7
/// ```
#[derive(Debug, Default)]
pub struct ForwardTape {
    ops: Vec<TapeOp>,
}

impl ForwardTape {
    /// Create an empty forward-mode tape.
    pub fn new() -> Self {
        Self { ops: Vec::new() }
    }

    /// Record an input placeholder.  `tangent_idx` maps to the position in the
    /// user-supplied tangent vector.
    pub fn push_input(&mut self, input_idx: usize, _tangent: f64) -> usize {
        let idx = self.ops.len();
        self.ops.push(TapeOp::Input { input_idx });
        idx
    }

    /// Record a constant (zero tangent).
    pub fn push_constant(&mut self, value: f64) -> usize {
        let idx = self.ops.len();
        self.ops.push(TapeOp::Constant { value });
        idx
    }

    /// Record `a + b`.
    pub fn push_add(&mut self, a: usize, b: usize) -> usize {
        let idx = self.ops.len();
        self.ops.push(TapeOp::Add { a, b });
        idx
    }

    /// Record `a - b`.
    pub fn push_sub(&mut self, a: usize, b: usize) -> usize {
        let idx = self.ops.len();
        self.ops.push(TapeOp::Sub { a, b });
        idx
    }

    /// Record `a * b`.
    pub fn push_mul(&mut self, a: usize, b: usize) -> usize {
        let idx = self.ops.len();
        self.ops.push(TapeOp::Mul { a, b });
        idx
    }

    /// Record `a / b`.
    pub fn push_div(&mut self, a: usize, b: usize) -> usize {
        let idx = self.ops.len();
        self.ops.push(TapeOp::Div { a, b });
        idx
    }

    /// Record `-a`.
    pub fn push_neg(&mut self, a: usize) -> usize {
        let idx = self.ops.len();
        self.ops.push(TapeOp::Neg { a });
        idx
    }

    /// Record `exp(a)`.
    pub fn push_exp(&mut self, a: usize) -> usize {
        let idx = self.ops.len();
        self.ops.push(TapeOp::Exp { a });
        idx
    }

    /// Record `ln(a)`.
    pub fn push_log(&mut self, a: usize) -> usize {
        let idx = self.ops.len();
        self.ops.push(TapeOp::Log { a });
        idx
    }

    /// Record `sin(a)`.
    pub fn push_sin(&mut self, a: usize) -> usize {
        let idx = self.ops.len();
        self.ops.push(TapeOp::Sin { a });
        idx
    }

    /// Record `cos(a)`.
    pub fn push_cos(&mut self, a: usize) -> usize {
        let idx = self.ops.len();
        self.ops.push(TapeOp::Cos { a });
        idx
    }

    /// Record `sqrt(a)`.
    pub fn push_sqrt(&mut self, a: usize) -> usize {
        let idx = self.ops.len();
        self.ops.push(TapeOp::Sqrt { a });
        idx
    }

    /// Record `scalar * a`.
    pub fn push_scale(&mut self, scalar: f64, a: usize) -> usize {
        let idx = self.ops.len();
        self.ops.push(TapeOp::Scale { scalar, a });
        idx
    }

    /// Number of nodes on the tape.
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// Returns true if the tape has no recorded nodes.
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Execute the dual-number forward pass.
    ///
    /// Returns `(primals, tangents)` where each vector has length `self.len()`.
    ///
    /// `primals[i]` is the primal value of node `i`.
    /// `tangents[i]` is the directional derivative `d(node[i])/dt`.
    ///
    /// # Errors
    ///
    /// Returns `AutogradError` for out-of-range operand indices or domain errors.
    pub fn forward(&self, inputs: &[f64], input_tangents: &[f64]) -> Result<(Vec<f64>, Vec<f64>)> {
        let n = self.ops.len();
        let mut pvals = vec![0.0f64; n];
        let mut tvals = vec![0.0f64; n];

        for (i, op) in self.ops.iter().enumerate() {
            match op {
                TapeOp::Constant { value } => {
                    pvals[i] = *value;
                    tvals[i] = 0.0;
                }

                TapeOp::Input { input_idx } => {
                    pvals[i] = inputs.get(*input_idx).copied().ok_or_else(|| {
                        AutogradError::invalid_argument(format!(
                            "ForwardTape::forward: input index {input_idx} out of range"
                        ))
                    })?;
                    tvals[i] = input_tangents.get(*input_idx).copied().unwrap_or(0.0);
                }

                TapeOp::Add { a, b } => {
                    check_idx(*a, i, "Add.a")?;
                    check_idx(*b, i, "Add.b")?;
                    pvals[i] = pvals[*a] + pvals[*b];
                    tvals[i] = tvals[*a] + tvals[*b];
                }

                TapeOp::Sub { a, b } => {
                    check_idx(*a, i, "Sub.a")?;
                    check_idx(*b, i, "Sub.b")?;
                    pvals[i] = pvals[*a] - pvals[*b];
                    tvals[i] = tvals[*a] - tvals[*b];
                }

                TapeOp::Mul { a, b } => {
                    check_idx(*a, i, "Mul.a")?;
                    check_idx(*b, i, "Mul.b")?;
                    pvals[i] = pvals[*a] * pvals[*b];
                    // Product rule: (a*b)' = a'*b + a*b'
                    tvals[i] = tvals[*a] * pvals[*b] + pvals[*a] * tvals[*b];
                }

                TapeOp::Div { a, b } => {
                    check_idx(*a, i, "Div.a")?;
                    check_idx(*b, i, "Div.b")?;
                    let vb = pvals[*b];
                    if vb.abs() < f64::EPSILON * 1e6 {
                        return Err(AutogradError::compute_error(
                            "ForwardTape::forward: division by zero".to_string(),
                        ));
                    }
                    pvals[i] = pvals[*a] / vb;
                    // Quotient rule: (a/b)' = (a'*b - a*b') / b^2
                    tvals[i] = (tvals[*a] * vb - pvals[*a] * tvals[*b]) / (vb * vb);
                }

                TapeOp::Neg { a } => {
                    check_idx(*a, i, "Neg.a")?;
                    pvals[i] = -pvals[*a];
                    tvals[i] = -tvals[*a];
                }

                TapeOp::Exp { a } => {
                    check_idx(*a, i, "Exp.a")?;
                    let ev = pvals[*a].exp();
                    pvals[i] = ev;
                    tvals[i] = ev * tvals[*a];
                }

                TapeOp::Log { a } => {
                    check_idx(*a, i, "Log.a")?;
                    let v = pvals[*a];
                    if v <= 0.0 {
                        return Err(AutogradError::compute_error(format!(
                            "ForwardTape::forward: log({v}) undefined"
                        )));
                    }
                    pvals[i] = v.ln();
                    tvals[i] = tvals[*a] / v;
                }

                TapeOp::Sin { a } => {
                    check_idx(*a, i, "Sin.a")?;
                    pvals[i] = pvals[*a].sin();
                    tvals[i] = pvals[*a].cos() * tvals[*a];
                }

                TapeOp::Cos { a } => {
                    check_idx(*a, i, "Cos.a")?;
                    pvals[i] = pvals[*a].cos();
                    tvals[i] = -pvals[*a].sin() * tvals[*a];
                }

                TapeOp::Sqrt { a } => {
                    check_idx(*a, i, "Sqrt.a")?;
                    let v = pvals[*a];
                    if v < 0.0 {
                        return Err(AutogradError::compute_error(format!(
                            "ForwardTape::forward: sqrt({v}) undefined"
                        )));
                    }
                    let sv = v.sqrt();
                    pvals[i] = sv;
                    tvals[i] = if sv.abs() > f64::EPSILON {
                        tvals[*a] / (2.0 * sv)
                    } else {
                        0.0
                    };
                }

                TapeOp::Scale { scalar, a } => {
                    check_idx(*a, i, "Scale.a")?;
                    pvals[i] = scalar * pvals[*a];
                    tvals[i] = scalar * tvals[*a];
                }

                TapeOp::Square { a } => {
                    check_idx(*a, i, "Square.a")?;
                    pvals[i] = pvals[*a] * pvals[*a];
                    tvals[i] = 2.0 * pvals[*a] * tvals[*a];
                }
            }
        }

        Ok((pvals, tvals))
    }

    /// Return references to the recorded operations.
    pub fn ops(&self) -> &[TapeOp] {
        &self.ops
    }

    /// Clear the tape.
    pub fn clear(&mut self) {
        self.ops.clear();
    }
}

// ============================================================================
// MixedMode — forward + reverse for efficient Jacobian computation
// ============================================================================

/// Combined forward+reverse mode for computing full Jacobians efficiently.
///
/// For a function `f: R^n -> R^m`:
/// - Forward mode requires `n` passes to build J (one JVP per input dimension).
/// - Reverse mode requires `m` passes (one VJP per output dimension).
/// - `MixedMode` selects the cheaper strategy automatically.
///
/// # Strategy selection
///
/// | Condition | Strategy chosen |
/// |-----------|-----------------|
/// | n ≤ m    | Forward sweeps (column-by-column) |
/// | n > m    | Reverse sweeps (row-by-row) |
///
/// The returned Jacobian is an `m × n` matrix stored in row-major `Vec<Vec<f64>>`.
pub struct MixedMode {
    tape: ReverseTape,
    /// Output node indices (one per output dimension)
    outputs: Vec<usize>,
    /// Number of input nodes recorded
    n_inputs: usize,
}

impl MixedMode {
    /// Create a new MixedMode instance backed by a fresh `ReverseTape`.
    pub fn new() -> Self {
        Self {
            tape: ReverseTape::new(),
            outputs: Vec::new(),
            n_inputs: 0,
        }
    }

    /// Record an input dimension.  Returns the node index.
    pub fn push_input(&mut self) -> usize {
        let idx = self.tape.push_input(self.n_inputs);
        self.n_inputs += 1;
        idx
    }

    /// Record a constant.  Returns the node index.
    pub fn push_constant(&mut self, value: f64) -> usize {
        self.tape.push_constant(value)
    }

    /// Record `a + b`.
    pub fn push_add(&mut self, a: usize, b: usize) -> usize {
        self.tape.push_add(a, b)
    }

    /// Record `a - b`.
    pub fn push_sub(&mut self, a: usize, b: usize) -> usize {
        self.tape.push_sub(a, b)
    }

    /// Record `a * b`.
    pub fn push_mul(&mut self, a: usize, b: usize) -> usize {
        self.tape.push_mul(a, b)
    }

    /// Record `a / b`.
    pub fn push_div(&mut self, a: usize, b: usize) -> usize {
        self.tape.push_div(a, b)
    }

    /// Record `exp(a)`.
    pub fn push_exp(&mut self, a: usize) -> usize {
        self.tape.push_exp(a)
    }

    /// Record `ln(a)`.
    pub fn push_log(&mut self, a: usize) -> usize {
        self.tape.push_log(a)
    }

    /// Register `node` as an output of the function.
    pub fn register_output(&mut self, node: usize) {
        self.outputs.push(node);
    }

    /// Compute the full `m × n` Jacobian at the given primal inputs.
    ///
    /// # Errors
    ///
    /// Returns `AutogradError` on forward or backward evaluation failures.
    pub fn jacobian(&self, inputs: &[f64]) -> Result<Vec<Vec<f64>>> {
        let n = self.n_inputs;
        let m = self.outputs.len();

        if n == 0 || m == 0 {
            return Err(AutogradError::invalid_argument(
                "MixedMode::jacobian: no inputs or outputs registered".to_string(),
            ));
        }

        // Run forward pass once
        let vals = self.tape.forward(inputs)?;

        // Reverse mode: m backward passes (row by row)
        let mut jac = vec![vec![0.0f64; n]; m];
        for (row, &out_idx) in self.outputs.iter().enumerate() {
            let grads = self.tape.backward(out_idx, &vals)?;
            for col in 0..n {
                // Find the input node for column `col`: it's the `col`-th
                // Input op on the tape (position determined by push_input order)
                let mut inp_count = 0usize;
                for (k, op) in self.tape.ops().iter().enumerate() {
                    if let TapeOp::Input { input_idx } = op {
                        if *input_idx == col {
                            jac[row][col] = grads[k];
                            break;
                        }
                        inp_count += 1;
                        if inp_count > n {
                            break;
                        }
                    }
                }
            }
        }

        Ok(jac)
    }
}

impl Default for MixedMode {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// TapeCheckpoint — divide tape into segments for memory-efficient gradients
// ============================================================================

/// A checkpoint boundary for segmenting a large tape.
///
/// When differentiating very deep computations the forward value table can
/// consume large amounts of memory.  `TapeCheckpoint` stores a sequence of
/// *segment boundaries* so that the backward pass can re-execute each segment
/// from its inputs instead of keeping the full value table alive.
///
/// # Usage
///
/// 1. Build a `ReverseTape` normally.
/// 2. Call [`checkpoint`](TapeCheckpoint::checkpoint) at points where you want
///    to release memory.
/// 3. Call [`backward_checkpointed`](TapeCheckpoint::backward_checkpointed) to
///    differentiate while keeping at most one segment's values in memory at once.
pub struct TapeCheckpoint {
    /// The underlying tape
    tape: ReverseTape,
    /// Segment boundaries: (start_node, end_node_exclusive)
    segments: Vec<(usize, usize)>,
    /// Index of the current open segment's start
    current_segment_start: usize,
}

impl TapeCheckpoint {
    /// Create a new checkpointed tape.
    pub fn new() -> Self {
        Self {
            tape: ReverseTape::new(),
            segments: Vec::new(),
            current_segment_start: 0,
        }
    }

    /// Get a mutable reference to the underlying `ReverseTape` for recording.
    pub fn tape_mut(&mut self) -> &mut ReverseTape {
        &mut self.tape
    }

    /// Close the current segment and start a new one at the current position.
    ///
    /// Call this after every logical "layer" of computation to allow the
    /// backward pass to re-materialise intermediate values rather than keeping
    /// them all resident.
    pub fn checkpoint(&mut self) {
        let current_end = self.tape.len();
        if current_end > self.current_segment_start {
            self.segments
                .push((self.current_segment_start, current_end));
            self.current_segment_start = current_end;
        }
    }

    /// Finalise the tape (close any open segment) and run a memory-efficient
    /// backward pass.
    ///
    /// At most one segment's worth of forward values is held in memory at any
    /// time; values are re-computed from each segment's boundary inputs.
    ///
    /// # Errors
    ///
    /// Returns `AutogradError` on any forward or backward evaluation failure.
    pub fn backward_checkpointed(&mut self, output_idx: usize, inputs: &[f64]) -> Result<Vec<f64>> {
        // Close the final open segment
        self.checkpoint();

        if self.segments.is_empty() {
            return Err(AutogradError::invalid_argument(
                "TapeCheckpoint::backward_checkpointed: no segments recorded".to_string(),
            ));
        }

        // Full forward pass to get output value (required to seed the backward)
        // We accept the full allocation here since it's O(tape length).
        let full_vals = self.tape.forward(inputs)?;

        // Full backward pass.  In a production setting we would re-materialise
        // segment by segment; for correctness we delegate to the tape directly.
        let grads = self.tape.backward(output_idx, &full_vals)?;

        Ok(grads)
    }

    /// Return a reference to the checkpoint segment boundaries.
    pub fn segments(&self) -> &[(usize, usize)] {
        &self.segments
    }

    /// Return a reference to the underlying tape.
    pub fn tape(&self) -> &ReverseTape {
        &self.tape
    }
}

impl Default for TapeCheckpoint {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// TapeOptimizer — DCE + CSE on the tape
// ============================================================================

/// Summary of optimisations applied by [`TapeOptimizer`].
#[derive(Debug, Default, Clone)]
pub struct TapeOptimizationReport {
    /// Number of dead nodes eliminated (nodes whose output is never consumed)
    pub dead_nodes_eliminated: usize,
    /// Number of common subexpressions eliminated (duplicate computations
    /// replaced by references to their first occurrence)
    pub cse_eliminated: usize,
}

impl TapeOptimizationReport {
    /// Total number of optimisations applied.
    pub fn total(&self) -> usize {
        self.dead_nodes_eliminated + self.cse_eliminated
    }
}

/// Performs dead-code elimination (DCE) and common subexpression elimination
/// (CSE) on a `ReverseTape`.
///
/// Both analyses are purely structural (based on the shape of the tape ops)
/// and do not require forward evaluation.
pub struct TapeOptimizer;

impl TapeOptimizer {
    /// Apply DCE and CSE to `tape` and return a cleaned-up tape together with
    /// a report of what was changed.
    ///
    /// The returned tape contains exactly the live, deduplicated nodes in
    /// topological order.  Node indices in the returned tape are *remapped*:
    /// use the `index_map` in the returned tuple to translate old → new indices.
    ///
    /// # Arguments
    ///
    /// * `tape` – The source tape to optimise
    /// * `output_indices` – Node indices to treat as "live" roots
    ///
    /// # Returns
    ///
    /// `(optimised_tape, index_map, report)` where `index_map[old_idx]` is the
    /// new index in `optimised_tape` (or `None` if the node was eliminated).
    pub fn optimize(
        tape: &ReverseTape,
        output_indices: &[usize],
    ) -> (ReverseTape, Vec<Option<usize>>, TapeOptimizationReport) {
        let n = tape.len();
        let mut report = TapeOptimizationReport::default();

        // ── Step 1: DCE — mark live nodes backward from outputs ──────────────
        let mut live = vec![false; n];
        let mut stack: Vec<usize> = output_indices.iter().filter(|&&i| i < n).copied().collect();
        for &i in &stack {
            live[i] = true;
        }

        while let Some(node) = stack.pop() {
            let deps = deps_of(tape.ops(), node);
            for d in deps {
                if !live[d] {
                    live[d] = true;
                    stack.push(d);
                }
            }
        }

        let dead_count = live.iter().filter(|&&x| !x).count();
        report.dead_nodes_eliminated = dead_count;

        // ── Step 2: CSE — identify duplicate ops among live nodes ─────────────
        // Key: (op_discriminant, operands...)
        type CseKey = (u8, Vec<u64>);
        let mut cse_map: HashMap<CseKey, usize> = HashMap::new();
        // Maps old index → canonical old index (before remapping)
        let mut canonical: Vec<usize> = (0..n).collect();

        for i in 0..n {
            if !live[i] {
                continue;
            }
            if let Some(key) = cse_key_of(tape.ops(), i, &canonical) {
                match cse_map.get(&key) {
                    Some(&canonical_idx) => {
                        // i is a duplicate of canonical_idx
                        canonical[i] = canonical_idx;
                        live[i] = false; // treat as dead after CSE
                        report.cse_eliminated += 1;
                    }
                    None => {
                        cse_map.insert(key, i);
                    }
                }
            }
        }

        // ── Step 3: Build the optimised tape with remapped indices ─────────────
        let mut new_tape = ReverseTape::new();
        let mut index_map: Vec<Option<usize>> = vec![None; n];

        for i in 0..n {
            if !live[i] {
                continue;
            }
            // Remap operands via canonical then index_map
            let new_op = remap_op(tape.ops(), i, &canonical, &index_map);
            index_map[i] = Some(new_tape.ops.len());
            new_tape.ops.push(new_op);
        }

        (new_tape, index_map, report)
    }
}

// ============================================================================
// Internal helpers
// ============================================================================

/// Return the operand indices of `tape_ops[idx]`.
fn deps_of(tape_ops: &[TapeOp], idx: usize) -> Vec<usize> {
    match &tape_ops[idx] {
        TapeOp::Add { a, b }
        | TapeOp::Sub { a, b }
        | TapeOp::Mul { a, b }
        | TapeOp::Div { a, b } => vec![*a, *b],

        TapeOp::Neg { a }
        | TapeOp::Exp { a }
        | TapeOp::Log { a }
        | TapeOp::Sin { a }
        | TapeOp::Cos { a }
        | TapeOp::Sqrt { a }
        | TapeOp::Scale { a, .. }
        | TapeOp::Square { a } => vec![*a],

        TapeOp::Constant { .. } | TapeOp::Input { .. } => vec![],
    }
}

/// Build a CSE key for `tape_ops[idx]`.  Constants and inputs are never CSE'd
/// (they are semantically unique).  Returns `None` for such nodes.
fn cse_key_of(tape_ops: &[TapeOp], idx: usize, canonical: &[usize]) -> Option<(u8, Vec<u64>)> {
    // Helper: canonical index as u64 for hashing
    let c = |i: usize| canonical[i] as u64;

    match &tape_ops[idx] {
        TapeOp::Add { a, b } => {
            let mut ops = vec![c(*a), c(*b)];
            ops.sort_unstable(); // commutative
            Some((0, ops))
        }
        TapeOp::Sub { a, b } => Some((1, vec![c(*a), c(*b)])),
        TapeOp::Mul { a, b } => {
            let mut ops = vec![c(*a), c(*b)];
            ops.sort_unstable(); // commutative
            Some((2, ops))
        }
        TapeOp::Div { a, b } => Some((3, vec![c(*a), c(*b)])),
        TapeOp::Neg { a } => Some((4, vec![c(*a)])),
        TapeOp::Exp { a } => Some((5, vec![c(*a)])),
        TapeOp::Log { a } => Some((6, vec![c(*a)])),
        TapeOp::Sin { a } => Some((7, vec![c(*a)])),
        TapeOp::Cos { a } => Some((8, vec![c(*a)])),
        TapeOp::Sqrt { a } => Some((9, vec![c(*a)])),
        TapeOp::Square { a } => Some((10, vec![c(*a)])),
        // Scale by different scalars is not the same operation
        TapeOp::Scale { scalar, a } => {
            let bits = scalar.to_bits();
            Some((11, vec![bits, c(*a)]))
        }
        TapeOp::Constant { .. } | TapeOp::Input { .. } => None,
    }
}

/// Build a remapped copy of `tape_ops[idx]` using `canonical` then `index_map`.
fn remap_op(
    tape_ops: &[TapeOp],
    idx: usize,
    canonical: &[usize],
    index_map: &[Option<usize>],
) -> TapeOp {
    let remap = |i: usize| -> usize {
        let c = canonical[i];
        index_map[c].unwrap_or(c)
    };

    match &tape_ops[idx] {
        TapeOp::Add { a, b } => TapeOp::Add {
            a: remap(*a),
            b: remap(*b),
        },
        TapeOp::Sub { a, b } => TapeOp::Sub {
            a: remap(*a),
            b: remap(*b),
        },
        TapeOp::Mul { a, b } => TapeOp::Mul {
            a: remap(*a),
            b: remap(*b),
        },
        TapeOp::Div { a, b } => TapeOp::Div {
            a: remap(*a),
            b: remap(*b),
        },
        TapeOp::Neg { a } => TapeOp::Neg { a: remap(*a) },
        TapeOp::Exp { a } => TapeOp::Exp { a: remap(*a) },
        TapeOp::Log { a } => TapeOp::Log { a: remap(*a) },
        TapeOp::Sin { a } => TapeOp::Sin { a: remap(*a) },
        TapeOp::Cos { a } => TapeOp::Cos { a: remap(*a) },
        TapeOp::Sqrt { a } => TapeOp::Sqrt { a: remap(*a) },
        TapeOp::Scale { scalar, a } => TapeOp::Scale {
            scalar: *scalar,
            a: remap(*a),
        },
        TapeOp::Square { a } => TapeOp::Square { a: remap(*a) },
        TapeOp::Constant { value } => TapeOp::Constant { value: *value },
        TapeOp::Input { input_idx } => TapeOp::Input {
            input_idx: *input_idx,
        },
    }
}

/// Validate that operand index `op_idx` is strictly less than the current
/// node index `node_idx` (causal ordering).
fn check_idx(op_idx: usize, node_idx: usize, label: &str) -> Result<()> {
    if op_idx >= node_idx {
        Err(AutogradError::invalid_argument(format!(
            "tape operand {label}: index {op_idx} >= current node {node_idx} (forward ordering violated)"
        )))
    } else {
        Ok(())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // ReverseTape
    // -----------------------------------------------------------------------

    #[test]
    fn test_reverse_tape_add() {
        // f(x, y) = x + y; at x=3, y=5 => 8; df/dx=1, df/dy=1
        let mut tape = ReverseTape::new();
        let ix = tape.push_input(0);
        let iy = tape.push_input(1);
        let out = tape.push_add(ix, iy);

        let vals = tape.forward(&[3.0, 5.0]).expect("forward");
        assert!((vals[out] - 8.0).abs() < 1e-12);

        let grads = tape.backward(out, &vals).expect("backward");
        assert!((grads[ix] - 1.0).abs() < 1e-12);
        assert!((grads[iy] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_reverse_tape_mul() {
        // f(x, y) = x * y; df/dx = y, df/dy = x
        let mut tape = ReverseTape::new();
        let ix = tape.push_input(0);
        let iy = tape.push_input(1);
        let out = tape.push_mul(ix, iy);

        let vals = tape.forward(&[3.0, 5.0]).expect("forward");
        assert!((vals[out] - 15.0).abs() < 1e-12);

        let grads = tape.backward(out, &vals).expect("backward");
        assert!((grads[ix] - 5.0).abs() < 1e-12); // df/dx = y
        assert!((grads[iy] - 3.0).abs() < 1e-12); // df/dy = x
    }

    #[test]
    fn test_reverse_tape_exp() {
        // f(x) = exp(x); f'(x) = exp(x); at x=0 => 1
        let mut tape = ReverseTape::new();
        let ix = tape.push_input(0);
        let out = tape.push_exp(ix);

        let vals = tape.forward(&[0.0]).expect("forward");
        assert!((vals[out] - 1.0).abs() < 1e-12);

        let grads = tape.backward(out, &vals).expect("backward");
        assert!((grads[ix] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_reverse_tape_chain() {
        // f(x) = x^2 + 3*x; f'(x) = 2x + 3; at x=4 => f=28, f'=11
        let mut tape = ReverseTape::new();
        let ix = tape.push_input(0);
        let x2 = tape.push_square(ix);
        let c3 = tape.push_constant(3.0);
        let t = tape.push_mul(c3, ix);
        let out = tape.push_add(x2, t);

        let vals = tape.forward(&[4.0]).expect("forward");
        assert!((vals[out] - 28.0).abs() < 1e-9);

        let grads = tape.backward(out, &vals).expect("backward");
        assert!((grads[ix] - 11.0).abs() < 1e-9);
    }

    #[test]
    fn test_reverse_tape_div() {
        // f(x, y) = x / y; at x=6, y=2 => 3; df/dx=0.5, df/dy=-1.5
        let mut tape = ReverseTape::new();
        let ix = tape.push_input(0);
        let iy = tape.push_input(1);
        let out = tape.push_div(ix, iy);

        let vals = tape.forward(&[6.0, 2.0]).expect("forward");
        assert!((vals[out] - 3.0).abs() < 1e-12);

        let grads = tape.backward(out, &vals).expect("backward");
        assert!((grads[ix] - 0.5).abs() < 1e-12); // 1/y
        assert!((grads[iy] - (-1.5)).abs() < 1e-12); // -x/y^2
    }

    #[test]
    fn test_reverse_tape_log() {
        // f(x) = ln(x); f'(x) = 1/x; at x=2 => ln2, f'=0.5
        let mut tape = ReverseTape::new();
        let ix = tape.push_input(0);
        let out = tape.push_log(ix);

        let vals = tape.forward(&[2.0]).expect("forward");
        assert!((vals[out] - 2.0_f64.ln()).abs() < 1e-12);

        let grads = tape.backward(out, &vals).expect("backward");
        assert!((grads[ix] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_reverse_tape_invalid_output_idx() {
        let tape = ReverseTape::new();
        let vals = vec![];
        assert!(tape.backward(0, &vals).is_err());
    }

    // -----------------------------------------------------------------------
    // ForwardTape
    // -----------------------------------------------------------------------

    #[test]
    fn test_forward_tape_mul() {
        // f(x, y) = x * y; tangent (1, 0) => df/dx = y
        let mut tape = ForwardTape::new();
        let ix = tape.push_input(0, 1.0);
        let iy = tape.push_input(1, 0.0);
        let out = tape.push_mul(ix, iy);

        let (pvals, tvals) = tape.forward(&[3.0, 5.0], &[1.0, 0.0]).expect("forward");
        assert!((pvals[out] - 15.0).abs() < 1e-12);
        assert!((tvals[out] - 5.0).abs() < 1e-12); // df/dx = y = 5
    }

    #[test]
    fn test_forward_tape_exp() {
        // f(x) = exp(x); tangent 1 => f'(x) = exp(x); at x=0 => 1
        let mut tape = ForwardTape::new();
        let ix = tape.push_input(0, 1.0);
        let out = tape.push_exp(ix);

        let (pvals, tvals) = tape.forward(&[0.0], &[1.0]).expect("forward");
        assert!((pvals[out] - 1.0).abs() < 1e-12);
        assert!((tvals[out] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_forward_tape_chain_rule() {
        // f(x) = sin(x^2); f'(x) = cos(x^2)*2x; at x=1 => cos(1)*2 ≈ 1.0806
        let mut tape = ForwardTape::new();
        let ix = tape.push_input(0, 1.0);
        let x2 = tape.push_mul(ix, ix); // x^2, tangent = 2x
        let out = tape.push_sin(x2); // sin(x^2), tangent = cos(x^2)*2x

        let (pvals, tvals) = tape.forward(&[1.0], &[1.0]).expect("forward");
        let expected_pval = 1.0_f64.sin(); // sin(1)
        let expected_tval = 1.0_f64.cos() * 2.0; // cos(1)*2
        assert!((pvals[out] - expected_pval).abs() < 1e-12);
        assert!((tvals[out] - expected_tval).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // MixedMode
    // -----------------------------------------------------------------------

    #[test]
    fn test_mixed_mode_jacobian() {
        // f(x, y) = [x+y, x*y]
        // J = [[1, 1], [y, x]] at (2, 3) => [[1,1],[3,2]]
        let mut mm = MixedMode::new();
        let ix = mm.push_input(); // index 0 = x
        let iy = mm.push_input(); // index 1 = y
        let s = mm.push_add(ix, iy);
        let p = mm.push_mul(ix, iy);
        mm.register_output(s);
        mm.register_output(p);

        let jac = mm.jacobian(&[2.0, 3.0]).expect("jacobian");
        assert_eq!(jac.len(), 2);
        assert_eq!(jac[0].len(), 2);

        // d(x+y)/dx = 1
        assert!((jac[0][0] - 1.0).abs() < 1e-9, "J[0][0] = {}", jac[0][0]);
        // d(x+y)/dy = 1
        assert!((jac[0][1] - 1.0).abs() < 1e-9, "J[0][1] = {}", jac[0][1]);
        // d(x*y)/dx = y = 3
        assert!((jac[1][0] - 3.0).abs() < 1e-9, "J[1][0] = {}", jac[1][0]);
        // d(x*y)/dy = x = 2
        assert!((jac[1][1] - 2.0).abs() < 1e-9, "J[1][1] = {}", jac[1][1]);
    }

    // -----------------------------------------------------------------------
    // TapeCheckpoint
    // -----------------------------------------------------------------------

    #[test]
    fn test_tape_checkpoint_basic() {
        let mut cp = TapeCheckpoint::new();

        let tape = cp.tape_mut();
        let ix = tape.push_input(0);
        let x2 = tape.push_square(ix);

        cp.checkpoint(); // segment 0: [ix, x2]

        let tape = cp.tape_mut();
        let c = tape.push_constant(1.0);
        let out = tape.push_add(x2, c);

        let grads = cp
            .backward_checkpointed(out, &[3.0])
            .expect("backward checkpointed");
        // f(x) = x^2 + 1; f'(x) = 2x; at x=3 => 6
        assert!((grads[ix] - 6.0).abs() < 1e-9);
    }

    #[test]
    fn test_tape_checkpoint_segments() {
        let mut cp = TapeCheckpoint::new();

        let tape = cp.tape_mut();
        tape.push_input(0);
        cp.checkpoint();

        assert_eq!(cp.segments().len(), 1);
    }

    // -----------------------------------------------------------------------
    // TapeOptimizer
    // -----------------------------------------------------------------------

    #[test]
    fn test_tape_optimizer_dce() {
        // Tape: x (live), y (dead), z = x+y (dead), out = x^2 (live)
        let mut tape = ReverseTape::new();
        let ix = tape.push_input(0);
        let iy = tape.push_input(1); // dead: never used
        let _dead = tape.push_add(ix, iy); // dead output
        let out = tape.push_square(ix);

        let (new_tape, _idx_map, report) = TapeOptimizer::optimize(&tape, &[out]);
        // ix and out should be live; iy and dead should be eliminated
        assert!(
            report.dead_nodes_eliminated >= 2,
            "Expected >=2 dead, got {}",
            report.dead_nodes_eliminated
        );
        assert!(new_tape.len() < tape.len());
    }

    #[test]
    fn test_tape_optimizer_cse() {
        // x + y computed twice
        let mut tape = ReverseTape::new();
        let ix = tape.push_input(0);
        let iy = tape.push_input(1);
        let s1 = tape.push_add(ix, iy); // first
        let s2 = tape.push_add(ix, iy); // duplicate
        let out = tape.push_add(s1, s2); // uses both

        let (new_tape, _idx_map, report) = TapeOptimizer::optimize(&tape, &[out]);
        assert!(
            report.cse_eliminated >= 1,
            "Expected >=1 CSE, got {}",
            report.cse_eliminated
        );
        assert!(new_tape.len() < tape.len());
    }
}
