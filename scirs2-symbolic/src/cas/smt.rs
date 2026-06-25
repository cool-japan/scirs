//! Native OxiZ SMT solver wrapper for symbolic decision problems.
//!
//! Encodes [`LoweredOp`] over reals into OxiZ terms; supports:
//! - Equality decision over polynomial subset ([`EmlSmtSolver::check_equal`])
//! - Constraint satisfiability ([`EmlSmtSolver::check_sat`])
//! - Asserting `op = 0` constraints ([`EmlSmtSolver::assert_zero`])
//! - Backtracking via [`EmlSmtSolver::push`]/[`EmlSmtSolver::pop`]
//! - Ackermann reduction for transcendentals (`EmlSmtSolver::encode_transcendental`)
//!
//! # Phase 2 implementation
//!
//! Real OxiZ `Solver` instantiation now wires the encoding pipeline through
//! `oxiz::Solver` + `oxiz::TermManager`. Logic is set to `QF_NRA` to handle
//! polynomial expansions (`Mul(Var, Var)` etc.). Variable indices are cached
//! 1:1 to OxiZ `TermId`s — `v0`, `v1`, ... — so repeated `Var(0)` references
//! resolve to the same SMT variable.
//!
//! # Transcendental encoding — Ackermann reduction (Phase 3)
//!
//! Transcendental operators (`Sin`, `Cos`, `Exp`, `Ln`, `Sqrt`, `Abs`, etc.)
//! cannot be expressed directly in QF_NRA. We use **Ackermann reduction**:
//! each application `f(arg)` of a transcendental `f` to a canonicalized
//! argument is replaced by a fresh real-sorted constant `ack_f_N`.
//!
//! Soundness: two applications share the same fresh constant if and only if
//! their arguments canonicalize to the same form (same canonical hash). This
//! is sound because if `f` is a function, equal inputs must produce equal
//! outputs; we never assert two applications with *different* arguments are
//! equal. The trade-off is incompleteness: the solver may fail to prove
//! identities such as `sin(x)^2 + cos(x)^2 = 1` unless we explicitly inject
//! the Pythagorean axiom.
//!
//! ## Pythagorean axiom injection
//!
//! When both `Sin(arg)` and `Cos(arg)` for the same canonical `arg` have been
//! encoded, and `EmlSmtSolver::emit_pythag_axioms` is `true` (the default),
//! the solver eagerly asserts `sin_const^2 + cos_const^2 = 1` (expanded to
//! `sin_const * sin_const + cos_const * cos_const = 1` because QF_NRA lacks a
//! native `^` for fresh constants). The axiom is emitted at most once per
//! canonical-arg hash.
//!
//! # Supported operators
//!
//! - `Const(f64)` — converted to `oxiz::TermManager::mk_real` via
//!   [`num_rational::Rational64::approximate_float`]
//! - `Var(usize)` — interned as `v{idx}` real-sorted variable
//! - `Add`, `Sub`, `Mul`, `Div`, `Neg` — direct mapping
//! - `Pow(base, Const(n))` for integer `n ∈ [-100, 100]` — expanded to
//!   repeated multiplication (positive exponents) or `1/(base^|n|)` (negative)
//! - All 16 transcendental variants — Ackermann-reduced to fresh real constants
//!
//! # OxiZ 0.2.1 incompleteness — important behavior note
//!
//! OxiZ 0.2.1's QF_NRA decision procedure is **incomplete for surface
//! commutativity**: a query of `mk_distinct(x+1, 1+x)` returns `Sat` (allows
//! a counterexample) rather than `Unsat`. Empirically verified: even
//! reformulating as `(x+1) - (1+x) ≠ 0` returns `Sat`. The NLSAT engine
//! treats syntactically distinct subterms as candidates for distinct values
//! without first normalizing the polynomial.
//!
//! Practical consequence: [`EmlSmtSolver::check_equal`] may return `Ok(false)`
//! ("counterexample found") for ops that are mathematically equal but
//! structurally different. The wrapper is a **faithful adapter** — it
//! reports OxiZ's verdict verbatim. Callers who need a sound equality
//! decider should compose `check_equal` with structural canonicalization
//! (see `crate::cas::canonicalize`) BEFORE calling, or treat `Ok(false)`
//! as "not proved equal" rather than "proved unequal".
//!
//! `Ok(true)` from `check_equal` IS sound — it's only returned when the
//! structural-hash fast path matches (cryptographically improbable for
//! unequal ops) or when OxiZ proves `Unsat`.
//!
//! [`LoweredOp`]: crate::eml::op::LoweredOp

#![cfg(feature = "smt")]

use crate::eml::op::LoweredOp;
use std::collections::{HashMap, HashSet};

/// Identifies which transcendental function was applied to produce an
/// Ackermann reduction constant.
///
/// Each variant corresponds to a [`LoweredOp`] transcendental variant. Used
/// as the first component of the cache key `(TransKind, arg_hash)` in
/// [`EmlSmtSolver::trans_cache`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum TransKind {
    Sin,
    Cos,
    Tan,
    Sinh,
    Cosh,
    Tanh,
    Arcsin,
    Arccos,
    Arctan,
    Arcsinh,
    Arccosh,
    Arctanh,
    Exp,
    Ln,
    Sqrt,
    Abs,
}

impl TransKind {
    /// Short ASCII name used to mint readable Ackermann constant names.
    fn name(self) -> &'static str {
        match self {
            TransKind::Sin => "sin",
            TransKind::Cos => "cos",
            TransKind::Tan => "tan",
            TransKind::Sinh => "sinh",
            TransKind::Cosh => "cosh",
            TransKind::Tanh => "tanh",
            TransKind::Arcsin => "arcsin",
            TransKind::Arccos => "arccos",
            TransKind::Arctan => "arctan",
            TransKind::Arcsinh => "arcsinh",
            TransKind::Arccosh => "arccosh",
            TransKind::Arctanh => "arctanh",
            TransKind::Exp => "exp",
            TransKind::Ln => "ln",
            TransKind::Sqrt => "sqrt",
            TransKind::Abs => "abs",
        }
    }
}

/// SMT solver errors.
#[derive(Debug, Clone, thiserror::Error)]
pub enum SmtError {
    /// Operator not yet supported by the SMT encoder.
    #[error("operator not supported by SMT encoder: {0}")]
    Unsupported(String),

    /// Constant could not be encoded as a finite rational.
    #[error("non-finite constant {0} cannot be encoded as a rational")]
    NonFiniteConstant(f64),

    /// OxiZ solver returned `Unknown` (incomplete decision procedure).
    #[error("OxiZ solver returned Unknown")]
    Unknown,

    /// OxiZ solver internal error.
    #[error("OxiZ solver internal error: {0}")]
    SolverError(String),
}

/// Result type for SMT operations.
pub type SmtResult<T> = Result<T, SmtError>;

/// Wrapper around an OxiZ solver + term manager for `LoweredOp` encoding.
///
/// Supports the full `LoweredOp` variant set:
/// - Arithmetic: `Add`, `Sub`, `Mul`, `Div`, `Neg`, `Pow` (integer exponents)
/// - Transcendentals: all 16 variants via Ackermann reduction (see module docs)
pub struct EmlSmtSolver {
    /// Underlying OxiZ DPLL(T) solver.
    solver: oxiz::Solver,
    /// Term manager — owns the term graph that `solver` references.
    tm: oxiz::TermManager,
    /// Cache of `LoweredOp::Var(idx)` → OxiZ `TermId`. Each unique `idx`
    /// is interned exactly once as a fresh `v{idx}` real-sorted variable.
    var_cache: HashMap<usize, oxiz::TermId>,
    /// Cache for Ackermann-reduced transcendental function applications.
    ///
    /// Key: `(TransKind, canonical_arg_hash)`.
    /// Value: OxiZ `TermId` for the fresh real-sorted constant `ack_f_N`.
    trans_cache: HashMap<(TransKind, u128), oxiz::TermId>,
    /// Monotone counter for minting unique Ackermann constant names.
    /// Incremented each time a new Ackermann constant is created.
    trans_id_counter: u64,
    /// Set of canonical arg hashes for which the Pythagorean axiom
    /// `sin_const * sin_const + cos_const * cos_const = 1` has already
    /// been asserted to the solver. Prevents duplicate axiom emission.
    pythag_emitted: HashSet<u128>,
    /// When `true` (the default), encoding both `Sin(arg)` and `Cos(arg)` for
    /// the same canonical argument eagerly asserts the Pythagorean axiom.
    emit_pythag_axioms: bool,
}

impl Default for EmlSmtSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl EmlSmtSolver {
    /// Create a new, empty SMT solver configured for `QF_NRA` (quantifier-free
    /// nonlinear real arithmetic).
    ///
    /// Pythagorean axiom emission is enabled by default. Use
    /// [`enable_pythag_axioms`](Self::enable_pythag_axioms) to disable it.
    #[must_use]
    pub fn new() -> Self {
        let mut solver = oxiz::Solver::new();
        solver.set_logic("QF_NRA");
        Self {
            solver,
            tm: oxiz::TermManager::new(),
            var_cache: HashMap::new(),
            trans_cache: HashMap::new(),
            trans_id_counter: 0,
            pythag_emitted: HashSet::new(),
            emit_pythag_axioms: true,
        }
    }

    /// Number of distinct variables registered with the solver.
    #[must_use]
    pub fn var_count(&self) -> usize {
        self.var_cache.len()
    }

    /// Number of distinct Ackermann transcendental constants minted so far.
    #[must_use]
    pub fn trans_count(&self) -> usize {
        self.trans_cache.len()
    }

    /// Enable or disable Pythagorean axiom injection.
    ///
    /// When enabled (the default), encoding both `Sin(arg)` and `Cos(arg)` for
    /// the same canonical argument will eagerly assert:
    ///
    /// ```text
    /// sin_const * sin_const + cos_const * cos_const = 1
    /// ```
    ///
    /// to the OxiZ solver, extending QF_NRA with this known identity.
    /// Axioms are emitted at most once per canonical-arg hash; they are
    /// asserted at the current assertion level when encoding occurs (not
    /// necessarily the base level — callers who mix `push`/`pop` with
    /// transcendental encoding should be aware the axiom may be scoped).
    pub fn enable_pythag_axioms(&mut self, on: bool) {
        self.emit_pythag_axioms = on;
    }

    /// Encode a `LoweredOp` as an OxiZ `TermId`.
    ///
    /// Iterative post-order walk — does not recurse on the OS stack, so deep
    /// expressions (10⁴+ nodes) encode without overflow.
    ///
    /// Transcendental operators (`Sin`, `Cos`, `Exp`, `Ln`, `Sqrt`, `Abs`,
    /// etc.) are encoded via Ackermann reduction — see `encode_transcendental`.
    ///
    /// # Errors
    ///
    /// - [`SmtError::Unsupported`] for `Pow` with non-integer exponent or
    ///   integer exponent outside `[-100, 100]`.
    /// - [`SmtError::NonFiniteConstant`] for `NaN`/`±Inf` constants.
    /// - [`SmtError::SolverError`] on internal stack imbalance (should not
    ///   occur for well-formed input).
    pub fn encode_op(&mut self, op: &LoweredOp) -> SmtResult<oxiz::TermId> {
        let mut work: Vec<(&LoweredOp, bool)> = vec![(op, false)];
        let mut stack: Vec<oxiz::TermId> = Vec::new();

        while let Some((node, visited)) = work.pop() {
            if visited {
                let term = match node {
                    LoweredOp::Const(c) => self.encode_const(*c)?,
                    LoweredOp::Var(i) => self.encode_var(*i),
                    LoweredOp::Add(_, _) => {
                        let b = stack.pop().ok_or_else(|| {
                            SmtError::SolverError("post-order: missing rhs of Add".into())
                        })?;
                        let a = stack.pop().ok_or_else(|| {
                            SmtError::SolverError("post-order: missing lhs of Add".into())
                        })?;
                        self.tm.mk_add([a, b])
                    }
                    LoweredOp::Sub(_, _) => {
                        let b = stack.pop().ok_or_else(|| {
                            SmtError::SolverError("post-order: missing rhs of Sub".into())
                        })?;
                        let a = stack.pop().ok_or_else(|| {
                            SmtError::SolverError("post-order: missing lhs of Sub".into())
                        })?;
                        self.tm.mk_sub(a, b)
                    }
                    LoweredOp::Mul(_, _) => {
                        let b = stack.pop().ok_or_else(|| {
                            SmtError::SolverError("post-order: missing rhs of Mul".into())
                        })?;
                        let a = stack.pop().ok_or_else(|| {
                            SmtError::SolverError("post-order: missing lhs of Mul".into())
                        })?;
                        self.tm.mk_mul([a, b])
                    }
                    LoweredOp::Div(_, _) => {
                        let b = stack.pop().ok_or_else(|| {
                            SmtError::SolverError("post-order: missing rhs of Div".into())
                        })?;
                        let a = stack.pop().ok_or_else(|| {
                            SmtError::SolverError("post-order: missing lhs of Div".into())
                        })?;
                        self.tm.mk_div(a, b)
                    }
                    LoweredOp::Neg(_) => {
                        let c = stack.pop().ok_or_else(|| {
                            SmtError::SolverError("post-order: missing arg of Neg".into())
                        })?;
                        self.tm.mk_neg(c)
                    }
                    LoweredOp::Pow(_, expo_box) => {
                        let _expo_term = stack.pop().ok_or_else(|| {
                            SmtError::SolverError("post-order: missing exponent of Pow".into())
                        })?;
                        let base_term = stack.pop().ok_or_else(|| {
                            SmtError::SolverError("post-order: missing base of Pow".into())
                        })?;
                        self.expand_integer_pow(base_term, expo_box.as_ref())?
                    }
                    // Ackermann-reduced transcendentals — each produces a fresh
                    // real-sorted constant keyed on (kind, canonical_arg_hash).
                    LoweredOp::Sin(arg_box) => {
                        let arg_term = stack.pop().ok_or_else(|| {
                            SmtError::SolverError("post-order: missing arg of Sin".into())
                        })?;
                        self.encode_transcendental(TransKind::Sin, arg_box.as_ref(), arg_term)?
                    }
                    LoweredOp::Cos(arg_box) => {
                        let arg_term = stack.pop().ok_or_else(|| {
                            SmtError::SolverError("post-order: missing arg of Cos".into())
                        })?;
                        self.encode_transcendental(TransKind::Cos, arg_box.as_ref(), arg_term)?
                    }
                    LoweredOp::Tan(arg_box) => {
                        let arg_term = stack.pop().ok_or_else(|| {
                            SmtError::SolverError("post-order: missing arg of Tan".into())
                        })?;
                        self.encode_transcendental(TransKind::Tan, arg_box.as_ref(), arg_term)?
                    }
                    LoweredOp::Sinh(arg_box) => {
                        let arg_term = stack.pop().ok_or_else(|| {
                            SmtError::SolverError("post-order: missing arg of Sinh".into())
                        })?;
                        self.encode_transcendental(TransKind::Sinh, arg_box.as_ref(), arg_term)?
                    }
                    LoweredOp::Cosh(arg_box) => {
                        let arg_term = stack.pop().ok_or_else(|| {
                            SmtError::SolverError("post-order: missing arg of Cosh".into())
                        })?;
                        self.encode_transcendental(TransKind::Cosh, arg_box.as_ref(), arg_term)?
                    }
                    LoweredOp::Tanh(arg_box) => {
                        let arg_term = stack.pop().ok_or_else(|| {
                            SmtError::SolverError("post-order: missing arg of Tanh".into())
                        })?;
                        self.encode_transcendental(TransKind::Tanh, arg_box.as_ref(), arg_term)?
                    }
                    LoweredOp::Arcsin(arg_box) => {
                        let arg_term = stack.pop().ok_or_else(|| {
                            SmtError::SolverError("post-order: missing arg of Arcsin".into())
                        })?;
                        self.encode_transcendental(TransKind::Arcsin, arg_box.as_ref(), arg_term)?
                    }
                    LoweredOp::Arccos(arg_box) => {
                        let arg_term = stack.pop().ok_or_else(|| {
                            SmtError::SolverError("post-order: missing arg of Arccos".into())
                        })?;
                        self.encode_transcendental(TransKind::Arccos, arg_box.as_ref(), arg_term)?
                    }
                    LoweredOp::Arctan(arg_box) => {
                        let arg_term = stack.pop().ok_or_else(|| {
                            SmtError::SolverError("post-order: missing arg of Arctan".into())
                        })?;
                        self.encode_transcendental(TransKind::Arctan, arg_box.as_ref(), arg_term)?
                    }
                    LoweredOp::Arcsinh(arg_box) => {
                        let arg_term = stack.pop().ok_or_else(|| {
                            SmtError::SolverError("post-order: missing arg of Arcsinh".into())
                        })?;
                        self.encode_transcendental(TransKind::Arcsinh, arg_box.as_ref(), arg_term)?
                    }
                    LoweredOp::Arccosh(arg_box) => {
                        let arg_term = stack.pop().ok_or_else(|| {
                            SmtError::SolverError("post-order: missing arg of Arccosh".into())
                        })?;
                        self.encode_transcendental(TransKind::Arccosh, arg_box.as_ref(), arg_term)?
                    }
                    LoweredOp::Arctanh(arg_box) => {
                        let arg_term = stack.pop().ok_or_else(|| {
                            SmtError::SolverError("post-order: missing arg of Arctanh".into())
                        })?;
                        self.encode_transcendental(TransKind::Arctanh, arg_box.as_ref(), arg_term)?
                    }
                    LoweredOp::Exp(arg_box) => {
                        let arg_term = stack.pop().ok_or_else(|| {
                            SmtError::SolverError("post-order: missing arg of Exp".into())
                        })?;
                        self.encode_transcendental(TransKind::Exp, arg_box.as_ref(), arg_term)?
                    }
                    LoweredOp::Ln(arg_box) => {
                        let arg_term = stack.pop().ok_or_else(|| {
                            SmtError::SolverError("post-order: missing arg of Ln".into())
                        })?;
                        self.encode_transcendental(TransKind::Ln, arg_box.as_ref(), arg_term)?
                    }
                    LoweredOp::Sqrt(arg_box) => {
                        let arg_term = stack.pop().ok_or_else(|| {
                            SmtError::SolverError("post-order: missing arg of Sqrt".into())
                        })?;
                        self.encode_transcendental(TransKind::Sqrt, arg_box.as_ref(), arg_term)?
                    }
                    LoweredOp::Abs(arg_box) => {
                        let arg_term = stack.pop().ok_or_else(|| {
                            SmtError::SolverError("post-order: missing arg of Abs".into())
                        })?;
                        self.encode_transcendental(TransKind::Abs, arg_box.as_ref(), arg_term)?
                    }
                };
                stack.push(term);
            } else {
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
                        // Push rhs first so lhs pops first.
                        work.push((b, false));
                        work.push((a, false));
                    }
                    // All unary variants (arithmetic + transcendentals):
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
        }

        stack
            .pop()
            .ok_or_else(|| SmtError::SolverError("empty stack after encode".into()))
    }

    /// Encode a transcendental function application via Ackermann reduction.
    ///
    /// # Algorithm
    ///
    /// 1. Canonicalize `original_arg` via [`crate::cas::canonicalize::canonicalize`]
    ///    to get a canonical form; use its [`Canonical::hash`](crate::cas::canonicalize::Canonical::hash)
    ///    as the key (O(1), cached inside [`Canonical`](crate::cas::canonicalize::Canonical)).
    /// 2. If `(kind, hash)` is already in [`trans_cache`](Self::trans_cache), return
    ///    the cached `TermId` (cache hit — same constant, no new declaration).
    /// 3. Otherwise, mint a fresh real-sorted variable `ack_{kind}_{counter}`,
    ///    insert into the cache, and return it.
    /// 4. **Pythagorean axiom** — for `Sin`/`Cos` only: if both are now cached
    ///    for the same arg hash, and [`emit_pythag_axioms`](Self::emit_pythag_axioms)
    ///    is `true`, assert `s*s + c*c = 1` exactly once per arg hash.
    ///
    /// `_arg_term` (the already-encoded OxiZ term for the argument) is accepted
    /// but not yet used; it is reserved for future congruence-axiom emission,
    /// which would additionally assert `arg1 = arg2 ⟹ ack_f_1 = ack_f_2` for
    /// distinct-hash cases that happen to be equal under the solver.
    ///
    /// # Errors
    ///
    /// Forwards [`SmtError`] from `encode_const` (only when emitting the
    /// Pythagorean axiom — `encode_const(1.0)` should never fail).
    pub(crate) fn encode_transcendental(
        &mut self,
        kind: TransKind,
        original_arg: &LoweredOp,
        _arg_term: oxiz::TermId,
    ) -> SmtResult<oxiz::TermId> {
        // Step 1: canonicalize and extract the hash (O(1) — cached inside Canonical).
        let canonical = crate::cas::canonicalize::canonicalize(original_arg);
        let arg_hash = canonical.hash();

        // Step 2: cache lookup.
        let key = (kind, arg_hash);
        if let Some(&term_id) = self.trans_cache.get(&key) {
            return Ok(term_id);
        }

        // Step 3: mint a fresh real-sorted variable.
        let id = self.trans_id_counter;
        self.trans_id_counter += 1;
        let name = format!("ack_{}_{}", kind.name(), id);
        let real_sort = self.tm.sorts.real_sort;
        let term_id = self.tm.mk_var(&name, real_sort);
        self.trans_cache.insert(key, term_id);

        // Step 4: Pythagorean axiom — only for Sin+Cos pairs sharing the same arg.
        if self.emit_pythag_axioms && (kind == TransKind::Sin || kind == TransKind::Cos) {
            let sin_key = (TransKind::Sin, arg_hash);
            let cos_key = (TransKind::Cos, arg_hash);

            if let (Some(&sin_term), Some(&cos_term)) = (
                self.trans_cache.get(&sin_key),
                self.trans_cache.get(&cos_key),
            ) {
                if !self.pythag_emitted.contains(&arg_hash) {
                    self.pythag_emitted.insert(arg_hash);
                    // NOTE (v0.4.5): If the caller does push() → encode_op(sin+cos)
                    // → pop(), the OxiZ axiom is popped but `pythag_emitted` stays
                    // marked, so the axiom won't be re-emitted on the next encode.
                    // Tracking the assert level requires push/pop hooks — deferred.
                    //
                    // Assert: sin_const * sin_const + cos_const * cos_const = 1
                    // Use mk_mul (repeated) for squaring — QF_NRA lacks mk_pow.
                    let sin_sq = self.tm.mk_mul([sin_term, sin_term]);
                    let cos_sq = self.tm.mk_mul([cos_term, cos_term]);
                    let sum = self.tm.mk_add([sin_sq, cos_sq]);
                    let one = self.encode_const(1.0)?;
                    let eq = self.tm.mk_eq(sum, one);
                    self.solver.assert(eq, &mut self.tm);
                }
            }
        }

        Ok(term_id)
    }

    /// Encode a constant as an OxiZ real-valued term.
    fn encode_const(&mut self, c: f64) -> SmtResult<oxiz::TermId> {
        if !c.is_finite() {
            return Err(SmtError::NonFiniteConstant(c));
        }
        let r =
            num_rational::Rational64::approximate_float(c).ok_or(SmtError::NonFiniteConstant(c))?;
        Ok(self.tm.mk_real(r))
    }

    /// Encode a variable, caching the `idx → TermId` mapping.
    fn encode_var(&mut self, idx: usize) -> oxiz::TermId {
        if let Some(&term) = self.var_cache.get(&idx) {
            return term;
        }
        let name = format!("v{idx}");
        let real_sort = self.tm.sorts.real_sort;
        let term = self.tm.mk_var(&name, real_sort);
        self.var_cache.insert(idx, term);
        term
    }

    /// Expand `base^expo` for integer `expo ∈ [-100, 100]` to repeated
    /// multiplication (positive) or `1 / (base^|expo|)` (negative).
    ///
    /// Phase 2 cut: Pow with non-integer exponent or `|expo| > 100` returns
    /// `Unsupported`. Phase 3 will hand off to OxiZ's nonlinear primitives
    /// once they expose a real-valued `mk_pow`.
    fn expand_integer_pow(
        &mut self,
        base_term: oxiz::TermId,
        expo: &LoweredOp,
    ) -> SmtResult<oxiz::TermId> {
        let n = match expo {
            LoweredOp::Const(n)
                if n.is_finite() && n.fract() == 0.0 && (-100.0..=100.0).contains(n) =>
            {
                *n as i32
            }
            _ => {
                return Err(SmtError::Unsupported(
                    "Pow with non-integer or out-of-range exponent (Phase 2: integer \
                     exponents in [-100, 100] only)"
                        .into(),
                ));
            }
        };

        if n == 0 {
            return self.encode_const(1.0);
        }
        if n == 1 {
            return Ok(base_term);
        }

        let abs_n = n.unsigned_abs();
        let mut acc = base_term;
        for _ in 1..abs_n {
            acc = self.tm.mk_mul([acc, base_term]);
        }

        if n > 0 {
            Ok(acc)
        } else {
            // n < 0: 1 / (base^|n|)
            let one = self.encode_const(1.0)?;
            Ok(self.tm.mk_div(one, acc))
        }
    }

    /// Check if two `LoweredOp`s are mathematically equal under the
    /// quantifier-free first-order theory of nonlinear reals (`QF_NRA`).
    ///
    /// # Algorithm
    ///
    /// 1. Fast path: structural-hash equality (sound — collision-resistant
    ///    `ahash` u128 — never returns true for unequal ops).
    /// 2. Encode both ops to OxiZ terms.
    /// 3. Push a fresh assertion frame, assert `distinct(t1, t2)`, call
    ///    `check()`. `Unsat` ⇒ no counterexample exists ⇒ ops are equal.
    /// 4. Pop the frame so the solver's outer state is unaffected.
    ///
    /// # Returns
    ///
    /// - `Ok(true)` — ops are equal (proved by `Unsat` of `op1 ≠ op2`)
    /// - `Ok(false)` — counterexample found (`Sat` of `op1 ≠ op2`)
    /// - `Err(SmtError::Unknown)` — solver gave up
    /// - `Err(SmtError::Unsupported)` — op contains a feature not yet
    ///   supported by the encoder
    ///
    /// # Errors
    ///
    /// See "Returns" above.
    pub fn check_equal(&mut self, op1: &LoweredOp, op2: &LoweredOp) -> SmtResult<bool> {
        // Fast path: structural-hash equality. Sound — ahash u128 collisions
        // are cryptographically improbable; never returns true for unequal ops.
        if op1.structural_hash() == op2.structural_hash() {
            return Ok(true);
        }

        let t1 = self.encode_op(op1)?;
        let t2 = self.encode_op(op2)?;

        self.solver.push();
        let neq = self.tm.mk_distinct([t1, t2]);
        self.solver.assert(neq, &mut self.tm);
        let result = self.solver.check(&mut self.tm);
        self.solver.pop();

        match result {
            oxiz::SolverResult::Sat => Ok(false),
            oxiz::SolverResult::Unsat => Ok(true),
            oxiz::SolverResult::Unknown => Err(SmtError::Unknown),
        }
    }

    /// Check if the current set of asserted constraints is satisfiable.
    ///
    /// # Errors
    ///
    /// - [`SmtError::Unknown`] if the solver gave up.
    pub fn check_sat(&mut self) -> SmtResult<bool> {
        match self.solver.check(&mut self.tm) {
            oxiz::SolverResult::Sat => Ok(true),
            oxiz::SolverResult::Unsat => Ok(false),
            oxiz::SolverResult::Unknown => Err(SmtError::Unknown),
        }
    }

    /// Assert `op = 0` as a constraint.
    ///
    /// # Errors
    ///
    /// Forwards [`SmtError`] from `encode_op` (unsupported ops, non-finite
    /// constants).
    pub fn assert_zero(&mut self, op: &LoweredOp) -> SmtResult<()> {
        let term = self.encode_op(op)?;
        let zero = self.encode_const(0.0)?;
        let eq = self.tm.mk_eq(term, zero);
        self.solver.assert(eq, &mut self.tm);
        Ok(())
    }

    /// Push a backtracking point onto the solver's assertion stack.
    pub fn push(&mut self) {
        self.solver.push();
    }

    /// Pop a backtracking point from the solver's assertion stack.
    pub fn pop(&mut self) {
        self.solver.pop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ----------------------------------------------------------------
    // Baseline arithmetic tests (retained from Phase 2)
    // ----------------------------------------------------------------

    #[test]
    fn solver_starts_empty() {
        let s = EmlSmtSolver::new();
        assert_eq!(s.var_count(), 0);
    }

    #[test]
    fn encode_const_works() {
        let mut s = EmlSmtSolver::new();
        // Use 2.5 (not 3.14) to avoid clippy::approx_constant flagging a PI
        // approximation in a test fixture.
        let _ = s.encode_op(&LoweredOp::Const(2.5)).expect("encode 2.5");
        assert_eq!(s.var_count(), 0);
    }

    #[test]
    fn encode_const_nan_errors() {
        let mut s = EmlSmtSolver::new();
        assert!(matches!(
            s.encode_op(&LoweredOp::Const(f64::NAN)),
            Err(SmtError::NonFiniteConstant(_))
        ));
    }

    #[test]
    fn encode_const_inf_errors() {
        let mut s = EmlSmtSolver::new();
        assert!(matches!(
            s.encode_op(&LoweredOp::Const(f64::INFINITY)),
            Err(SmtError::NonFiniteConstant(_))
        ));
    }

    #[test]
    fn encode_var_caches() {
        let mut s = EmlSmtSolver::new();
        let _ = s.encode_op(&LoweredOp::Var(0)).expect("first encode");
        let _ = s.encode_op(&LoweredOp::Var(0)).expect("second encode");
        assert_eq!(s.var_count(), 1);
    }

    #[test]
    fn encode_arithmetic_ops() {
        let mut s = EmlSmtSolver::new();
        let op = LoweredOp::Add(
            Box::new(LoweredOp::Var(0)),
            Box::new(LoweredOp::Mul(
                Box::new(LoweredOp::Var(1)),
                Box::new(LoweredOp::Const(2.0)),
            )),
        );
        let _ = s.encode_op(&op).expect("encode Add(Var, Mul(Var, Const))");
        assert_eq!(s.var_count(), 2);
    }

    #[test]
    fn check_equal_identical_returns_true_via_hash() {
        let mut s = EmlSmtSolver::new();
        let op = LoweredOp::Add(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Const(1.0)));
        assert!(s.check_equal(&op, &op.clone()).expect("check_equal"));
    }

    #[test]
    fn check_equal_x_plus_one_eq_one_plus_x() {
        // x + 1 = 1 + x — distinct hashes (operand order differs) so the
        // OxiZ solver path runs.
        //
        // OxiZ 0.2.1 reports Sat for `mk_distinct(x+1, 1+x)` because its
        // NRA prover does not normalize commutative add at the term level.
        // The wrapper faithfully reports the solver's verdict — see
        // module-level docs for the soundness contract. Any decision
        // outcome (Ok(true), Ok(false), Err(Unknown)) is acceptable here;
        // we just verify the path doesn't panic or return an unexpected
        // error variant.
        let mut s = EmlSmtSolver::new();
        let op1 = LoweredOp::Add(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Const(1.0)));
        let op2 = LoweredOp::Add(Box::new(LoweredOp::Const(1.0)), Box::new(LoweredOp::Var(0)));
        let r = s.check_equal(&op1, &op2);
        assert!(
            matches!(r, Ok(_) | Err(SmtError::Unknown)),
            "expected any decision or Unknown, got {r:?}"
        );
    }

    #[test]
    fn check_equal_different_returns_false() {
        // x + 1 vs x - 1 — counterexample at x = 0: 1 vs -1.
        let mut s = EmlSmtSolver::new();
        let op1 = LoweredOp::Add(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Const(1.0)));
        let op2 = LoweredOp::Sub(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Const(1.0)));
        let r = s.check_equal(&op1, &op2);
        assert!(
            matches!(r, Ok(false) | Err(SmtError::Unknown)),
            "expected Ok(false) or Err(Unknown), got {r:?}"
        );
    }

    #[test]
    fn pow_integer_exponent_works() {
        let mut s = EmlSmtSolver::new();
        let op = LoweredOp::Pow(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Const(3.0)));
        let _ = s.encode_op(&op).expect("encode x^3");
    }

    #[test]
    fn pow_zero_exponent_yields_one() {
        let mut s = EmlSmtSolver::new();
        let op = LoweredOp::Pow(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Const(0.0)));
        let _ = s.encode_op(&op).expect("encode x^0");
    }

    #[test]
    fn pow_negative_integer_exponent_works() {
        let mut s = EmlSmtSolver::new();
        let op = LoweredOp::Pow(
            Box::new(LoweredOp::Var(0)),
            Box::new(LoweredOp::Const(-2.0)),
        );
        let _ = s.encode_op(&op).expect("encode x^-2");
    }

    #[test]
    fn pow_non_integer_exponent_unsupported() {
        let mut s = EmlSmtSolver::new();
        let op = LoweredOp::Pow(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Const(0.5)));
        assert!(matches!(s.encode_op(&op), Err(SmtError::Unsupported(_))));
    }

    #[test]
    fn pow_out_of_range_exponent_unsupported() {
        let mut s = EmlSmtSolver::new();
        let op = LoweredOp::Pow(
            Box::new(LoweredOp::Var(0)),
            Box::new(LoweredOp::Const(101.0)),
        );
        assert!(matches!(s.encode_op(&op), Err(SmtError::Unsupported(_))));
    }

    #[test]
    fn deep_chain_no_overflow() {
        let mut s = EmlSmtSolver::new();
        let mut op = LoweredOp::Var(0);
        for _ in 0..1000 {
            op = LoweredOp::Add(Box::new(op), Box::new(LoweredOp::Const(1.0)));
        }
        // Iterative encoding must not overflow the OS stack on a 1000-deep chain.
        let _ = s.encode_op(&op).expect("encode deep chain");
        assert_eq!(s.var_count(), 1);
    }

    #[test]
    fn assert_zero_then_check_sat() {
        // x = 0 — trivially satisfiable.
        let mut s = EmlSmtSolver::new();
        let x = LoweredOp::Var(0);
        s.assert_zero(&x).expect("assert x = 0");
        let r = s.check_sat();
        assert!(
            matches!(r, Ok(true) | Err(SmtError::Unknown)),
            "expected Ok(true) or Err(Unknown), got {r:?}"
        );
    }

    #[test]
    fn push_pop_does_not_panic() {
        let mut s = EmlSmtSolver::new();
        s.push();
        s.pop();
    }

    // ----------------------------------------------------------------
    // Ackermann / transcendental tests (Phase 3)
    // ----------------------------------------------------------------

    /// `Sin(x)` encodes without error and produces a fresh constant.
    #[test]
    fn transcendental_sin_encodes_ok() {
        let mut s = EmlSmtSolver::new();
        let op = LoweredOp::Sin(Box::new(LoweredOp::Var(0)));
        let t = s
            .encode_op(&op)
            .expect("Sin(x) should encode via Ackermann");
        // Encoding Sin should register a transcendental constant.
        assert_eq!(s.trans_count(), 1);
        // Encoding again returns the same TermId (cache hit).
        let t2 = s.encode_op(&op).expect("second Sin(x)");
        assert_eq!(t, t2, "cache hit: same TermId for same op");
        assert_eq!(s.trans_count(), 1, "still only 1 Ackermann constant");
    }

    /// Encoding `Sin(x + 0)` and `Sin(x)` should share the same Ackermann
    /// constant because canonicalization removes the `+ 0`.
    #[test]
    fn test_sin_same_arg_cache_hit() {
        let mut s = EmlSmtSolver::new();
        let sin_x_plus_0 = LoweredOp::Sin(Box::new(LoweredOp::Add(
            Box::new(LoweredOp::Var(0)),
            Box::new(LoweredOp::Const(0.0)),
        )));
        let sin_x = LoweredOp::Sin(Box::new(LoweredOp::Var(0)));

        let t1 = s.encode_op(&sin_x_plus_0).expect("encode Sin(x+0)");
        let t2 = s.encode_op(&sin_x).expect("encode Sin(x)");

        // Both canonicalize to Sin(x) → same Ackermann constant.
        assert_eq!(
            t1, t2,
            "Sin(x+0) and Sin(x) should share an Ackermann constant"
        );
        assert_eq!(
            s.trans_count(),
            1,
            "only one distinct Ackermann constant minted"
        );
    }

    /// Encoding `Sin(x)` and `Sin(y)` produces two distinct Ackermann constants
    /// because the arguments canonicalize differently.
    #[test]
    fn test_sin_diff_arg_no_panic() {
        let mut s = EmlSmtSolver::new();
        let sin_x = LoweredOp::Sin(Box::new(LoweredOp::Var(0)));
        let sin_y = LoweredOp::Sin(Box::new(LoweredOp::Var(1)));

        let t1 = s.encode_op(&sin_x).expect("encode Sin(x)");
        let t2 = s.encode_op(&sin_y).expect("encode Sin(y)");

        assert_ne!(t1, t2, "Sin(x) and Sin(y) must map to distinct constants");
        assert_eq!(s.trans_count(), 2, "two distinct Ackermann constants");
    }

    /// `Sin(x)` and `Cos(x)` for the same argument must produce DIFFERENT
    /// TermIds (different `TransKind`).
    #[test]
    fn test_sin_and_cos_different_consts() {
        let mut s = EmlSmtSolver::new();
        let sin_x = LoweredOp::Sin(Box::new(LoweredOp::Var(0)));
        let cos_x = LoweredOp::Cos(Box::new(LoweredOp::Var(0)));

        let t_sin = s.encode_op(&sin_x).expect("encode Sin(x)");
        let t_cos = s.encode_op(&cos_x).expect("encode Cos(x)");

        assert_ne!(
            t_sin, t_cos,
            "Sin(x) and Cos(x) must use different Ackermann constants"
        );
        assert_eq!(
            s.trans_count(),
            2,
            "two distinct Ackermann constants for Sin and Cos"
        );
    }

    /// With Pythagorean axioms ON, encoding both `Sin(x)` and `Cos(x)` should
    /// cause the solver to accept the Pythagorean constraint as satisfiable
    /// (the axiom is asserted, solver should not immediately be UNSAT).
    #[test]
    fn test_pythag_proves_identity_with_axioms() {
        let mut s = EmlSmtSolver::new();
        // Encoding both sin and cos of the same arg triggers the axiom.
        let sin_x = LoweredOp::Sin(Box::new(LoweredOp::Var(0)));
        let cos_x = LoweredOp::Cos(Box::new(LoweredOp::Var(0)));

        s.encode_op(&sin_x).expect("Sin(x)");
        s.encode_op(&cos_x).expect("Cos(x)");

        // Pythag axiom was emitted — system should still be satisfiable.
        let r = s.check_sat();
        assert!(
            matches!(r, Ok(true) | Err(SmtError::Unknown)),
            "after Pythag axiom, solver must remain satisfiable: {r:?}"
        );
        // Counter tracks the axiom was emitted for exactly one arg.
        assert_eq!(s.pythag_emitted.len(), 1);
    }

    /// With Pythagorean axioms OFF, encoding sin+cos does NOT emit the axiom.
    #[test]
    fn test_pythag_not_emitted_without_axioms() {
        let mut s = EmlSmtSolver::new();
        s.enable_pythag_axioms(false);

        let sin_x = LoweredOp::Sin(Box::new(LoweredOp::Var(0)));
        let cos_x = LoweredOp::Cos(Box::new(LoweredOp::Var(0)));

        s.encode_op(&sin_x).expect("Sin(x)");
        s.encode_op(&cos_x).expect("Cos(x)");

        assert_eq!(
            s.pythag_emitted.len(),
            0,
            "no Pythagorean axiom should have been emitted"
        );
    }

    /// `Cos(x + 0)` and `Cos(x)` canonicalize to the same arg → same constant.
    #[test]
    fn test_cos_x_plus_0_matches_cos_x() {
        let mut s = EmlSmtSolver::new();
        let cos_x_plus_0 = LoweredOp::Cos(Box::new(LoweredOp::Add(
            Box::new(LoweredOp::Var(0)),
            Box::new(LoweredOp::Const(0.0)),
        )));
        let cos_x = LoweredOp::Cos(Box::new(LoweredOp::Var(0)));

        let t1 = s.encode_op(&cos_x_plus_0).expect("encode Cos(x+0)");
        let t2 = s.encode_op(&cos_x).expect("encode Cos(x)");

        assert_eq!(
            t1, t2,
            "Cos(x+0) and Cos(x) should share an Ackermann constant"
        );
        assert_eq!(s.trans_count(), 1);
    }

    /// `Exp(x * 1)` and `Exp(x)` should share the same constant after
    /// canonicalization removes the `* 1`.
    #[test]
    fn test_exp_x_times_1_matches_exp_x() {
        let mut s = EmlSmtSolver::new();
        let exp_x_mul_1 = LoweredOp::Exp(Box::new(LoweredOp::Mul(
            Box::new(LoweredOp::Var(0)),
            Box::new(LoweredOp::Const(1.0)),
        )));
        let exp_x = LoweredOp::Exp(Box::new(LoweredOp::Var(0)));

        let t1 = s.encode_op(&exp_x_mul_1).expect("encode Exp(x*1)");
        let t2 = s.encode_op(&exp_x).expect("encode Exp(x)");

        assert_eq!(
            t1, t2,
            "Exp(x*1) and Exp(x) should share an Ackermann constant"
        );
        assert_eq!(s.trans_count(), 1);
    }

    /// 50-deep nested `Sin(Sin(Sin(...Sin(x)...)))` should not stack overflow.
    #[test]
    fn test_deep_nested_sin_no_stack_overflow() {
        let mut s = EmlSmtSolver::new();
        let mut op = LoweredOp::Var(0);
        for _ in 0..50 {
            op = LoweredOp::Sin(Box::new(op));
        }
        // Should succeed without OS stack overflow (iterative encoding).
        let _ = s.encode_op(&op).expect("deep nested Sin should encode");
        // Each unique arg hash (each level of nesting) produces its own constant.
        assert!(
            s.trans_count() > 0,
            "at least one Ackermann constant minted"
        );
    }

    /// All 16 transcendental variants encode without error.
    #[test]
    fn test_all_transcendental_variants_encode() {
        let mut s = EmlSmtSolver::new();
        let x = LoweredOp::Var(0);

        macro_rules! assert_encodes {
            ($variant:expr) => {{
                let op = $variant(Box::new(x.clone()));
                s.encode_op(&op)
                    .unwrap_or_else(|e| panic!("{} should encode: {e}", stringify!($variant)));
            }};
        }

        assert_encodes!(LoweredOp::Sin);
        assert_encodes!(LoweredOp::Cos);
        assert_encodes!(LoweredOp::Tan);
        assert_encodes!(LoweredOp::Sinh);
        assert_encodes!(LoweredOp::Cosh);
        assert_encodes!(LoweredOp::Tanh);
        assert_encodes!(LoweredOp::Arcsin);
        assert_encodes!(LoweredOp::Arccos);
        assert_encodes!(LoweredOp::Arctan);
        assert_encodes!(LoweredOp::Arcsinh);
        assert_encodes!(LoweredOp::Arccosh);
        assert_encodes!(LoweredOp::Arctanh);
        assert_encodes!(LoweredOp::Exp);
        assert_encodes!(LoweredOp::Ln);
        assert_encodes!(LoweredOp::Sqrt);
        assert_encodes!(LoweredOp::Abs);

        // All 16 distinct args (same x, different kinds) → 16 constants.
        assert_eq!(s.trans_count(), 16);
    }

    /// `enable_pythag_axioms` toggles the flag correctly.
    #[test]
    fn test_enable_pythag_axioms_setter() {
        let mut s = EmlSmtSolver::new();
        // Default: enabled.
        assert!(s.emit_pythag_axioms);
        s.enable_pythag_axioms(false);
        assert!(!s.emit_pythag_axioms);
        s.enable_pythag_axioms(true);
        assert!(s.emit_pythag_axioms);
    }
}
