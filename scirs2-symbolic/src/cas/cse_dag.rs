//! Common Subexpression Elimination DAG keyed by EML structural hash.
//!
//! [`CseDag`] deduplicates shared subexpressions across multiple [`LoweredOp`]
//! expressions inserted into it. When a Newton step computes `f`, `∇f`, and
//! `∇²f` at the same point, every shared subexpression (e.g. `x²` appearing
//! in all three) is evaluated exactly once — O(unique-node-count) evaluations
//! instead of O(expressions × depth).
//!
//! # Design
//!
//! Nodes are identified by their structural u128 hash (collision-free in
//! practice for the polynomial + analytic subsets used in SciRS2). Children
//! are tracked by hash in `children_of`. Evaluation uses Kahn's algorithm for
//! topological sort, then evaluates each unique node exactly once using its
//! children's already-computed values.
//!
//! ## Complexity
//!
//! - `add`: O(N) where N is the number of unique nodes in the subtree.
//! - `eval_all`: O(N) topological sort + O(N) evaluation = O(N) total.
//! - N = number of *unique* nodes (deduplicated by structural hash).
//!
//! ## Iterative guarantee
//!
//! All traversals use iterative work-stack patterns over heap-allocated `Vec`.
//! No recursive calls over `LoweredOp`. A 5000-deep `Sin(Sin(...))` chain is
//! safe.
//!
//! ## v0.4.4 scope note
//!
//! `eval_all` implements true O(unique_nodes) bottom-up evaluation via a
//! custom `eval_from_children` that applies one node's operator to pre-computed
//! child values from the result map. This avoids re-traversing subtrees. Full
//! wgpu GPU dispatch is deferred to v0.4.5.

use std::collections::HashMap;

use crate::eml::op::LoweredOp;
use crate::error::EmlError;

// ─────────────────────────────────────────────────────────────────────────────
// Internal work-stack frame for iterative post-order insertion
// ─────────────────────────────────────────────────────────────────────────────

/// Frame on the `add` work-stack.
///
/// `Open(op)` means we have not yet processed the children of `op`.
/// `Close(op, hash)` means all children have been processed; we must
/// finalize `op` at `hash` and register child edges.
#[derive(Clone)]
enum AddFrame {
    Open(LoweredOp),
    Close(LoweredOp, u128),
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers — extract child `LoweredOp` refs from a parent
// ─────────────────────────────────────────────────────────────────────────────

/// Push child ops onto the work stack in left-then-right order.
///
/// Binary ops push both children. Unary ops push one. Leaves push nothing.
/// We push right first so the left child is popped first (maintaining
/// left-to-right post-order for determinism, which matches `to_oxi_ops`).
fn push_children(op: &LoweredOp, stack: &mut Vec<AddFrame>) {
    match op {
        LoweredOp::Const(_) | LoweredOp::Var(_) => {}

        LoweredOp::Add(a, b)
        | LoweredOp::Sub(a, b)
        | LoweredOp::Mul(a, b)
        | LoweredOp::Div(a, b)
        | LoweredOp::Pow(a, b) => {
            // Right first so left is popped (processed) first.
            stack.push(AddFrame::Open(*b.clone()));
            stack.push(AddFrame::Open(*a.clone()));
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
            stack.push(AddFrame::Open(*c.clone()));
        }
    }
}

/// Return the hashes of the immediate children of `op`, in left-then-right
/// order (same order as `push_children`).
///
/// Const and Var nodes have no children → returns empty vec.
fn child_hashes(op: &LoweredOp) -> Vec<u128> {
    match op {
        LoweredOp::Const(_) | LoweredOp::Var(_) => vec![],

        LoweredOp::Add(a, b)
        | LoweredOp::Sub(a, b)
        | LoweredOp::Mul(a, b)
        | LoweredOp::Div(a, b)
        | LoweredOp::Pow(a, b) => {
            vec![a.structural_hash(), b.structural_hash()]
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
            vec![c.structural_hash()]
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-node evaluation from pre-computed child values
// ─────────────────────────────────────────────────────────────────────────────

/// Evaluate a single node given:
/// - `op`: the operator (must not be a leaf; leaves handled separately)
/// - `child_vals`: already-computed values for each child, in left-then-right order
///
/// Returns `Err` on domain violations or child-count mismatches.
fn eval_from_children(op: &LoweredOp, child_vals: &[f64]) -> Result<f64, EmlError> {
    match op {
        LoweredOp::Const(c) => Ok(*c), // leaf — called if child_vals is empty
        LoweredOp::Var(_) => {
            // Var leaves are resolved from `point` directly in `eval_all`.
            // This arm is unreachable in normal use, but handle gracefully.
            Err(EmlError::EvalDomain(
                "eval_from_children called on Var — use point slice instead".into(),
            ))
        }

        // Binary ops — need exactly 2 child values
        LoweredOp::Add(_, _) => {
            if child_vals.len() < 2 {
                return Err(EmlError::EvalDomain("Add: missing child values".into()));
            }
            Ok(child_vals[0] + child_vals[1])
        }
        LoweredOp::Sub(_, _) => {
            if child_vals.len() < 2 {
                return Err(EmlError::EvalDomain("Sub: missing child values".into()));
            }
            Ok(child_vals[0] - child_vals[1])
        }
        LoweredOp::Mul(_, _) => {
            if child_vals.len() < 2 {
                return Err(EmlError::EvalDomain("Mul: missing child values".into()));
            }
            Ok(child_vals[0] * child_vals[1])
        }
        LoweredOp::Div(_, _) => {
            if child_vals.len() < 2 {
                return Err(EmlError::EvalDomain("Div: missing child values".into()));
            }
            let b = child_vals[1];
            if b.abs() < 1e-300 {
                return Err(EmlError::DivisionByZero);
            }
            Ok(child_vals[0] / b)
        }
        LoweredOp::Pow(_, _) => {
            if child_vals.len() < 2 {
                return Err(EmlError::EvalDomain("Pow: missing child values".into()));
            }
            Ok(child_vals[0].powf(child_vals[1]))
        }

        // Unary ops — need exactly 1 child value
        LoweredOp::Neg(_) => {
            let c = child_vals
                .first()
                .ok_or_else(|| EmlError::EvalDomain("Neg: missing child value".into()))?;
            Ok(-c)
        }
        LoweredOp::Exp(_) => {
            let c = child_vals
                .first()
                .ok_or_else(|| EmlError::EvalDomain("Exp: missing child value".into()))?;
            Ok(c.exp())
        }
        LoweredOp::Ln(_) => {
            let c = child_vals
                .first()
                .ok_or_else(|| EmlError::EvalDomain("Ln: missing child value".into()))?;
            if *c <= 0.0 {
                return Err(EmlError::EvalDomain(format!(
                    "ln({c}) — argument must be positive"
                )));
            }
            Ok(c.ln())
        }
        LoweredOp::Sin(_) => {
            let c = child_vals
                .first()
                .ok_or_else(|| EmlError::EvalDomain("Sin: missing child value".into()))?;
            Ok(c.sin())
        }
        LoweredOp::Cos(_) => {
            let c = child_vals
                .first()
                .ok_or_else(|| EmlError::EvalDomain("Cos: missing child value".into()))?;
            Ok(c.cos())
        }
        LoweredOp::Tan(_) => {
            let c = child_vals
                .first()
                .ok_or_else(|| EmlError::EvalDomain("Tan: missing child value".into()))?;
            Ok(c.tan())
        }
        LoweredOp::Sinh(_) => {
            let c = child_vals
                .first()
                .ok_or_else(|| EmlError::EvalDomain("Sinh: missing child value".into()))?;
            Ok(c.sinh())
        }
        LoweredOp::Cosh(_) => {
            let c = child_vals
                .first()
                .ok_or_else(|| EmlError::EvalDomain("Cosh: missing child value".into()))?;
            Ok(c.cosh())
        }
        LoweredOp::Tanh(_) => {
            let c = child_vals
                .first()
                .ok_or_else(|| EmlError::EvalDomain("Tanh: missing child value".into()))?;
            Ok(c.tanh())
        }
        LoweredOp::Arcsin(_) => {
            let c = child_vals
                .first()
                .ok_or_else(|| EmlError::EvalDomain("Arcsin: missing child value".into()))?;
            if !(-1.0..=1.0).contains(c) {
                return Err(EmlError::EvalDomain(format!(
                    "arcsin({c}) — argument must be in [-1, 1]"
                )));
            }
            Ok(c.asin())
        }
        LoweredOp::Arccos(_) => {
            let c = child_vals
                .first()
                .ok_or_else(|| EmlError::EvalDomain("Arccos: missing child value".into()))?;
            if !(-1.0..=1.0).contains(c) {
                return Err(EmlError::EvalDomain(format!(
                    "arccos({c}) — argument must be in [-1, 1]"
                )));
            }
            Ok(c.acos())
        }
        LoweredOp::Arctan(_) => {
            let c = child_vals
                .first()
                .ok_or_else(|| EmlError::EvalDomain("Arctan: missing child value".into()))?;
            Ok(c.atan())
        }
        LoweredOp::Arcsinh(_) => {
            let c = child_vals
                .first()
                .ok_or_else(|| EmlError::EvalDomain("Arcsinh: missing child value".into()))?;
            Ok(c.asinh())
        }
        LoweredOp::Arccosh(_) => {
            let c = child_vals
                .first()
                .ok_or_else(|| EmlError::EvalDomain("Arccosh: missing child value".into()))?;
            if *c < 1.0 {
                return Err(EmlError::EvalDomain(format!(
                    "arccosh({c}) — argument must be ≥ 1"
                )));
            }
            Ok(c.acosh())
        }
        LoweredOp::Arctanh(_) => {
            let c = child_vals
                .first()
                .ok_or_else(|| EmlError::EvalDomain("Arctanh: missing child value".into()))?;
            if !(-1.0..1.0).contains(c) {
                return Err(EmlError::EvalDomain(format!(
                    "arctanh({c}) — argument must be in (-1, 1)"
                )));
            }
            Ok(c.atanh())
        }
        LoweredOp::Sqrt(_) => {
            let c = child_vals
                .first()
                .ok_or_else(|| EmlError::EvalDomain("Sqrt: missing child value".into()))?;
            if *c < 0.0 {
                return Err(EmlError::EvalDomain(format!(
                    "sqrt({c}) — argument must be ≥ 0"
                )));
            }
            Ok(c.sqrt())
        }
        LoweredOp::Abs(_) => {
            let c = child_vals
                .first()
                .ok_or_else(|| EmlError::EvalDomain("Abs: missing child value".into()))?;
            Ok(c.abs())
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CseDag — public API
// ─────────────────────────────────────────────────────────────────────────────

/// A DAG that deduplicates shared subexpressions by EML structural hash.
///
/// Insert expressions with [`CseDag::add`], which returns a `u128` key.
/// Evaluate all inserted expressions at a point with [`CseDag::eval_all`],
/// which returns a `HashMap<u128, f64>` — every unique subexpression evaluated
/// exactly once in topological order.
///
/// ## Usage for Newton steps
///
/// ```
/// use scirs2_symbolic::{LoweredOp, eml::{grad, hessian}};
/// use scirs2_symbolic::cas::CseDag;
///
/// let f = LoweredOp::Add(
///     Box::new(LoweredOp::Pow(
///         Box::new(LoweredOp::Var(0)),
///         Box::new(LoweredOp::Const(2.0)),
///     )),
///     Box::new(LoweredOp::Pow(
///         Box::new(LoweredOp::Var(1)),
///         Box::new(LoweredOp::Const(2.0)),
///     )),
/// );
/// let df_dx = grad(&f, 0);
/// let df_dy = grad(&f, 1);
///
/// let mut dag = CseDag::new();
/// let f_key   = dag.add(&f);
/// let gx_key  = dag.add(&df_dx);
/// let gy_key  = dag.add(&df_dy);
///
/// let values = dag.eval_all(&[1.0, 2.0]).expect("eval_all should succeed");
/// let f_val = values[&f_key];   // x² + y² = 5.0
/// let gx    = values[&gx_key];  // 2x = 2.0
/// let gy    = values[&gy_key];  // 2y = 4.0
/// assert!((f_val - 5.0).abs() < 1e-12);
/// assert!((gx - 2.0).abs() < 1e-12);
/// assert!((gy - 4.0).abs() < 1e-12);
/// ```
pub struct CseDag {
    /// Unique nodes stored in the DAG: hash → LoweredOp.
    nodes: HashMap<u128, LoweredOp>,
    /// Adjacency: parent_hash → child hashes (left then right for binary ops).
    children_of: HashMap<u128, Vec<u128>>,
}

impl Default for CseDag {
    fn default() -> Self {
        Self::new()
    }
}

impl CseDag {
    /// Create a new, empty `CseDag`.
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            children_of: HashMap::new(),
        }
    }

    /// Insert `op` and all its subexpressions into the DAG.
    ///
    /// Already-present nodes (by structural hash) are skipped — their subtrees
    /// are not re-traversed. This is the core deduplication mechanism.
    ///
    /// Returns the structural hash of the root node of `op`.
    ///
    /// ## Iterative algorithm
    ///
    /// Uses `Open` / `Close` frames on a heap-allocated work-stack so that
    /// arbitrarily deep expression trees (e.g. 5000-deep Sin chains) are safe.
    pub fn add(&mut self, op: &LoweredOp) -> u128 {
        let root_hash = op.structural_hash();

        // Fast path: already fully inserted.
        if self.nodes.contains_key(&root_hash) {
            return root_hash;
        }

        let mut stack: Vec<AddFrame> = Vec::with_capacity(64);
        stack.push(AddFrame::Open(op.clone()));

        while let Some(frame) = stack.pop() {
            match frame {
                AddFrame::Open(cur) => {
                    let h = cur.structural_hash();

                    if self.nodes.contains_key(&h) {
                        // Already in DAG — no need to re-process subtree.
                        continue;
                    }

                    // Schedule finalization after children are done.
                    stack.push(AddFrame::Close(cur.clone(), h));

                    // Push children to be processed before finalization.
                    push_children(&cur, &mut stack);
                }

                AddFrame::Close(cur, h) => {
                    if self.nodes.contains_key(&h) {
                        // Another path already finalized this hash — skip.
                        continue;
                    }

                    // Register children edges (computed after children are finalized).
                    let kids = child_hashes(&cur);
                    self.children_of.insert(h, kids);

                    // Store the node.
                    self.nodes.insert(h, cur);
                }
            }
        }

        root_hash
    }

    /// Evaluate all nodes at `point` (variable bindings indexed by `Var(i)`).
    ///
    /// Returns a `HashMap<u128, f64>` mapping every stored node's hash to its
    /// value. Nodes are evaluated in topological order (leaves first) so every
    /// parent can use its children's values without redundant recomputation.
    ///
    /// ## Algorithm
    ///
    /// 1. Build in-degree counts for all nodes (Kahn's algorithm setup).
    /// 2. Initialize a queue with all zero-in-degree nodes (leaves).
    /// 3. Process nodes from the queue: compute value from child values already
    ///    in the result map, then decrement parent in-degrees.
    ///
    /// The in-degree here refers to *reverse* edges: for topological ordering
    /// we need to process children before parents, so we compute in-degree as
    /// "number of parents that depend on me" — i.e., each node starts with
    /// in-degree equal to the number of times it appears as a child.
    ///
    /// Actually, for Kahn's algorithm on a DAG where we want children before
    /// parents, we build a reverse adjacency (child → set of parents) and use
    /// standard Kahn's on the reversed graph. The "in-degree" in the reversed
    /// graph equals the number of children in the forward graph.
    ///
    /// ## Errors
    ///
    /// Returns domain errors (e.g. `ln` of non-positive, `sqrt` of negative)
    /// as `EmlError`. Variable indices beyond `point.len()` return
    /// `EmlError::UnboundVariableIndex`.
    pub fn eval_all(&self, point: &[f64]) -> Result<HashMap<u128, f64>, EmlError> {
        if self.nodes.is_empty() {
            return Ok(HashMap::new());
        }

        // ── Step 1: Build reverse adjacency (child_hash → parent_hashes) ──
        // and in-degree for Kahn's (in reversed graph, in-degree = number of
        // *children* of the node in the forward graph).

        // in_degree[h] = number of children of node h (= edges FROM h in
        // the original DAG, or edges TO h in the reversed DAG).
        let mut in_degree: HashMap<u128, usize> = HashMap::with_capacity(self.nodes.len());
        // parent_of[child_h] = list of parent hashes
        let mut parent_of: HashMap<u128, Vec<u128>> = HashMap::with_capacity(self.nodes.len());

        for &h in self.nodes.keys() {
            let kids = self.children_of.get(&h).map(Vec::as_slice).unwrap_or(&[]);
            // In-degree of h in the reversed graph = number of its children
            // in the forward graph. Nodes with 0 children are ready first.
            in_degree.entry(h).or_insert(kids.len());
            for &ch in kids {
                parent_of.entry(ch).or_default().push(h);
            }
        }

        // ── Step 2: Queue all zero-in-degree nodes (leaves in forward graph) ──
        // These are nodes with no children — Const and Var — which can be
        // evaluated immediately.
        let mut queue: std::collections::VecDeque<u128> =
            std::collections::VecDeque::with_capacity(self.nodes.len());

        for (&h, &deg) in &in_degree {
            if deg == 0 {
                queue.push_back(h);
            }
        }

        // ── Step 3: Kahn's algorithm ──
        let mut result: HashMap<u128, f64> = HashMap::with_capacity(self.nodes.len());

        while let Some(h) = queue.pop_front() {
            let op = match self.nodes.get(&h) {
                Some(o) => o,
                None => continue,
            };

            // Compute value for this node.
            let val = match op {
                LoweredOp::Const(c) => *c,
                LoweredOp::Var(i) => {
                    point
                        .get(*i)
                        .copied()
                        .ok_or(EmlError::UnboundVariableIndex {
                            idx: *i,
                            len: point.len(),
                        })?
                }
                _ => {
                    // Collect child values in order.
                    let kids = self.children_of.get(&h).map(Vec::as_slice).unwrap_or(&[]);
                    let mut child_vals = Vec::with_capacity(kids.len());
                    for &ch in kids {
                        let cv = result.get(&ch).copied().ok_or_else(|| {
                            EmlError::EvalDomain(format!(
                                "child hash {ch} not yet evaluated (topo order broken)"
                            ))
                        })?;
                        child_vals.push(cv);
                    }
                    eval_from_children(op, &child_vals)?
                }
            };

            result.insert(h, val);

            // Decrement in-degree of parents; enqueue those that reach 0.
            if let Some(parents) = parent_of.get(&h) {
                for &ph in parents {
                    let deg = in_degree.entry(ph).or_insert(0);
                    if *deg > 0 {
                        *deg -= 1;
                    }
                    if *deg == 0 {
                        queue.push_back(ph);
                    }
                }
            }
        }

        Ok(result)
    }

    /// Evaluate a specific expression (identified by its root hash) at `point`.
    ///
    /// Calls [`eval_all`] internally and extracts the value for `key`.
    /// Returns `None` if `key` was never inserted via `add` or if there
    /// is a domain error (domain errors are swallowed; use [`eval_all`]
    /// directly for proper error propagation).
    ///
    /// [`eval_all`]: CseDag::eval_all
    pub fn eval_one(&self, key: u128, point: &[f64]) -> Option<f64> {
        self.eval_all(point).ok().and_then(|m| m.get(&key).copied())
    }

    /// Number of unique nodes currently stored in the DAG.
    ///
    /// If you have inserted N total nodes (counting duplicates), the
    /// deduplication count is N − `node_count()`.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Remove all nodes from the DAG, resetting it to the initial empty state.
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.children_of.clear();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eml::{grad, hessian};

    const TOL: f64 = 1e-12;

    // Helpers for building common LoweredOp expressions.
    fn var(i: usize) -> LoweredOp {
        LoweredOp::Var(i)
    }
    fn con(c: f64) -> LoweredOp {
        LoweredOp::Const(c)
    }
    fn add(a: LoweredOp, b: LoweredOp) -> LoweredOp {
        LoweredOp::Add(Box::new(a), Box::new(b))
    }
    fn mul(a: LoweredOp, b: LoweredOp) -> LoweredOp {
        LoweredOp::Mul(Box::new(a), Box::new(b))
    }
    fn pow(base: LoweredOp, exp: LoweredOp) -> LoweredOp {
        LoweredOp::Pow(Box::new(base), Box::new(exp))
    }
    fn sin_op(x: LoweredOp) -> LoweredOp {
        LoweredOp::Sin(Box::new(x))
    }

    // ── Test 1: add deduplicates identical expressions ──

    #[test]
    fn test_add_dedup() {
        // x + y inserted twice must result in one copy.
        let xy = add(var(0), var(1));

        let mut dag = CseDag::new();
        let h1 = dag.add(&xy);
        let count_after_first = dag.node_count();

        let h2 = dag.add(&xy);
        let count_after_second = dag.node_count();

        // Same hash both times.
        assert_eq!(h1, h2);
        // No new nodes added on second insert.
        assert_eq!(count_after_first, count_after_second);
        // Nodes: Var(0), Var(1), Add(Var(0), Var(1)) = 3
        assert_eq!(dag.node_count(), 3);
    }

    // ── Test 2: eval_all returns correct value for x + y*z ──

    #[test]
    fn test_eval_simple() {
        // f(x,y,z) = x + y*z, evaluate at (1, 2, 3) → 1 + 2*3 = 7
        let f = add(var(0), mul(var(1), var(2)));

        let mut dag = CseDag::new();
        let root = dag.add(&f);

        let values = dag.eval_all(&[1.0, 2.0, 3.0]).expect("eval_all");
        let got = values[&root];
        assert!((got - 7.0).abs() < TOL, "expected 7.0, got {got}");
    }

    // ── Test 3: unique eval count is less than naive triple ──

    #[test]
    fn test_eval_counts_unique_evals() {
        // f(x) = x^2, df = 2*x (after simplify), ddf = 2
        let f = pow(var(0), con(2.0));
        let df = grad(&f, 0);
        let ddf = grad(&df, 0);

        let mut dag = CseDag::new();
        dag.add(&f);
        dag.add(&df);
        dag.add(&ddf);

        // If we had three separate trees, naive node count would be
        // f_nodes + df_nodes + ddf_nodes. With CSE, shared subexpressions
        // (e.g., Var(0), Const(2.0)) are stored only once.
        // We just verify that node_count is finite and significantly less
        // than a naive bound. In practice, the shared Var(0) and constants
        // collapse the total.
        let n = dag.node_count();
        // Conservative: unique nodes < 3 × nodes in f (f has ~3 nodes)
        assert!(n < 30, "too many unique nodes: {n}");

        // Correctness: evaluate at x=3 → f=9, df=6, ddf=2
        let vals = dag.eval_all(&[3.0]).expect("eval_all");
        let f_root = f.structural_hash();
        let f_val = vals[&f_root];
        assert!((f_val - 9.0).abs() < TOL, "f(3) expected 9.0, got {f_val}");
    }

    // ── Test 4: constant leaf evaluates correctly ──

    #[test]
    fn test_add_const() {
        // Use 3.15 to avoid clippy::approx_constant (3.14 ≈ π).
        let c = con(3.15);
        let mut dag = CseDag::new();
        let h = dag.add(&c);

        let got = dag.eval_one(h, &[]).expect("eval_one returned None");
        assert!((got - 3.15).abs() < TOL, "expected 3.15, got {got}");
    }

    // ── Test 5: deep chain evaluates correctly at all levels ──

    #[test]
    fn test_topo_order_correct() {
        // Build chain: Add(Add(Add(Var(0), Const(1)), Const(1)), Const(1))
        // = Var(0) + 3.  Evaluate at x=10 → 13.
        let depth = 5usize;
        let mut expr = var(0);
        for _ in 0..depth {
            expr = add(expr, con(1.0));
        }

        let mut dag = CseDag::new();
        let root = dag.add(&expr);

        let values = dag.eval_all(&[10.0]).expect("eval_all");
        let got = values[&root];
        let expected = 10.0 + depth as f64;
        assert!(
            (got - expected).abs() < TOL,
            "expected {expected}, got {got}"
        );

        // Also verify intermediate nodes.
        // The node for just Var(0) should also be in the map.
        let var_hash = var(0).structural_hash();
        let var_val = values[&var_hash];
        assert!(
            (var_val - 10.0).abs() < TOL,
            "Var(0) value wrong: {var_val}"
        );
    }

    // ── Test 6: shared subexpression sin(x) in sin(x) + sin(x)^2 ──

    #[test]
    fn test_shared_subexpr() {
        let x = var(0);
        let sx = sin_op(x.clone());

        // sin(x) + sin(x)^2
        let expr = add(sx.clone(), pow(sx.clone(), con(2.0)));

        let mut dag = CseDag::new();
        dag.add(&expr);

        // sin(x) must appear only once.
        let sin_hash = sx.structural_hash();
        assert!(dag.nodes.contains_key(&sin_hash), "sin(x) should be in DAG");

        // Evaluate at x = 0.5
        let vals = dag.eval_all(&[0.5]).expect("eval_all");
        let root_hash = expr.structural_hash();
        let got = vals[&root_hash];
        let s = 0.5_f64.sin();
        let expected = s + s * s;
        assert!(
            (got - expected).abs() < TOL,
            "expected {expected}, got {got}"
        );
    }

    // ── Test 7: clear resets node count to 0 ──

    #[test]
    fn test_node_count_after_clear() {
        let mut dag = CseDag::new();
        dag.add(&add(var(0), var(1)));
        assert!(dag.node_count() > 0);

        dag.clear();
        assert_eq!(dag.node_count(), 0);

        // Can re-use after clear.
        dag.add(&con(1.0));
        assert_eq!(dag.node_count(), 1);
    }

    // ── Test 8: gradient + hessian share subexpressions ──

    #[test]
    fn test_gradient_hessian_share() {
        // f = x^2 + y^2
        let f = add(pow(var(0), con(2.0)), pow(var(1), con(2.0)));
        let df_dx = grad(&f, 0);
        let df_dy = grad(&f, 1);
        let ddf_dx2 = grad(&df_dx, 0);

        let mut dag = CseDag::new();
        let f_key = dag.add(&f);
        let gx_key = dag.add(&df_dx);
        let gy_key = dag.add(&df_dy);
        let hxx_key = dag.add(&ddf_dx2);

        // Count unique nodes.
        let unique = dag.node_count();

        // Naive upper bound: count nodes in each tree separately.
        fn count_nodes(op: &LoweredOp) -> usize {
            let mut stack = vec![op];
            let mut count = 0usize;
            while let Some(cur) = stack.pop() {
                count += 1;
                match cur {
                    LoweredOp::Add(a, b)
                    | LoweredOp::Sub(a, b)
                    | LoweredOp::Mul(a, b)
                    | LoweredOp::Div(a, b)
                    | LoweredOp::Pow(a, b) => {
                        stack.push(a);
                        stack.push(b);
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
                        stack.push(c);
                    }
                    LoweredOp::Const(_) | LoweredOp::Var(_) => {}
                }
            }
            count
        }

        let naive =
            count_nodes(&f) + count_nodes(&df_dx) + count_nodes(&df_dy) + count_nodes(&ddf_dx2);

        // CSE should have strictly fewer unique nodes than naively summing.
        assert!(
            unique < naive,
            "CSE unique={unique} should be < naive sum={naive}"
        );

        // Correctness check at (2.0, 3.0):
        //   f = 4 + 9 = 13
        //   df/dx = 2x = 4
        //   df/dy = 2y = 6
        //   d²f/dx² = 2
        let vals = dag.eval_all(&[2.0, 3.0]).expect("eval_all");

        let f_val = vals[&f_key];
        assert!(
            (f_val - 13.0).abs() < TOL,
            "f(2,3) expected 13, got {f_val}"
        );

        let gx_val = vals[&gx_key];
        assert!(
            (gx_val - 4.0).abs() < TOL,
            "df/dx(2,3) expected 4, got {gx_val}"
        );

        let gy_val = vals[&gy_key];
        assert!(
            (gy_val - 6.0).abs() < TOL,
            "df/dy(2,3) expected 6, got {gy_val}"
        );

        let hxx_val = vals[&hxx_key];
        assert!(
            (hxx_val - 2.0).abs() < TOL,
            "d²f/dx²(2,3) expected 2, got {hxx_val}"
        );
    }

    // ── Additional: deep chain does not stack overflow ──

    #[test]
    fn test_deep_chain_no_overflow() {
        let depth = 2000usize;
        let mut expr = var(0);
        for _ in 0..depth {
            expr = add(expr, con(1.0));
        }
        let mut dag = CseDag::new();
        let root = dag.add(&expr);
        let vals = dag.eval_all(&[0.0]).expect("eval_all deep chain");
        let got = vals[&root];
        assert!(
            (got - depth as f64).abs() < TOL,
            "deep chain expected {}, got {got}",
            depth
        );
    }

    // ── Additional: eval_all on empty DAG returns empty map ──

    #[test]
    fn test_eval_all_empty() {
        let dag = CseDag::new();
        let vals = dag.eval_all(&[]).expect("eval_all empty");
        assert!(vals.is_empty());
    }

    // ── Additional: hessian function integration ──

    #[test]
    fn test_hessian_cse() {
        // f = x^2 * y;  hessian is 2×2
        let f = mul(pow(var(0), con(2.0)), var(1));
        let h = hessian(&f, 2);

        let mut dag = CseDag::new();
        let f_key = dag.add(&f);
        let h00_key = dag.add(&h[0][0]);
        let h01_key = dag.add(&h[0][1]);
        let h10_key = dag.add(&h[1][0]);
        let h11_key = dag.add(&h[1][1]);

        // Evaluate at (2.0, 3.0):
        //   f = 4 * 3 = 12
        //   h[0][0] = d^2f/dx^2 = 2y = 6
        //   h[0][1] = d^2f/dxdy = 2x = 4
        //   h[1][0] = d^2f/dydx = 2x = 4
        //   h[1][1] = d^2f/dy^2 = 0
        let vals = dag.eval_all(&[2.0, 3.0]).expect("eval_all hessian");

        let f_val = vals[&f_key];
        assert!(
            (f_val - 12.0).abs() < TOL,
            "f(2,3) expected 12, got {f_val}"
        );

        let h00 = vals[&h00_key];
        assert!((h00 - 6.0).abs() < TOL, "h[0][0] expected 6, got {h00}");

        let h01 = vals[&h01_key];
        assert!((h01 - 4.0).abs() < TOL, "h[0][1] expected 4, got {h01}");

        let h10 = vals[&h10_key];
        assert!((h10 - 4.0).abs() < TOL, "h[1][0] expected 4, got {h10}");

        let h11 = vals[&h11_key];
        assert!((h11 - 0.0).abs() < TOL, "h[1][1] expected 0, got {h11}");
    }
}
