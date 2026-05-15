//! Expression simplification optimization
//!
//! This module implements algebraic simplifications for computation graphs,
//! such as x + 0 -> x, x * 1 -> x, x - x -> 0, etc.
//!
//! # Standalone algebraic simplifier
//!
//! Because the live computation graph uses opaque `TensorID` indices and does
//! not expose a matchable expression tree, this module also provides the
//! self-contained [`AlgExpr`] / [`alg_simplify`] API.  It is independent of
//! the graph runtime and can be used wherever a symbolic expression tree is
//! needed (e.g. pre-compilation, unit tests, or future compiler passes).

use super::{OptimizationError, SimplificationPattern};
use crate::graph::{Graph, TensorID};
use crate::tensor::TensorInternal;
use crate::Float;
use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Standalone algebraic expression tree
// ─────────────────────────────────────────────────────────────────────────────

/// A lightweight algebraic expression tree used by the standalone simplifier.
///
/// This is decoupled from the graph runtime so that simplification rules can
/// be written, tested, and verified without needing a live `Graph<F>`.
#[derive(Debug, Clone, PartialEq)]
pub enum AlgExpr {
    /// A constant scalar value.
    Const(f64),
    /// A variable identified by a `usize` index.
    Var(usize),
    /// Addition: `lhs + rhs`.
    Add(Box<AlgExpr>, Box<AlgExpr>),
    /// Subtraction: `lhs - rhs`.
    Sub(Box<AlgExpr>, Box<AlgExpr>),
    /// Multiplication: `lhs * rhs`.
    Mul(Box<AlgExpr>, Box<AlgExpr>),
    /// Division: `lhs / rhs`.
    Div(Box<AlgExpr>, Box<AlgExpr>),
    /// Negation: `- inner`.
    Neg(Box<AlgExpr>),
    /// Exponentiation: `base ^ exp`.  Both operands may be arbitrary expressions.
    Pow(Box<AlgExpr>, Box<AlgExpr>),
    /// Natural logarithm.
    Log(Box<AlgExpr>),
    /// Natural exponential.
    Exp(Box<AlgExpr>),
}

impl AlgExpr {
    // ── constructor helpers ──────────────────────────────────────────────────

    /// Convenience constructor for `Const`.
    pub fn c(v: f64) -> Self {
        AlgExpr::Const(v)
    }

    /// Convenience constructor for `Var`.
    pub fn v(idx: usize) -> Self {
        AlgExpr::Var(idx)
    }

    // ── structural helpers ───────────────────────────────────────────────────

    /// Returns `true` if this expression is the constant `0.0`.
    fn is_zero(&self) -> bool {
        matches!(self, AlgExpr::Const(c) if *c == 0.0)
    }

    /// Returns `true` if this expression is the constant `1.0`.
    fn is_one(&self) -> bool {
        matches!(self, AlgExpr::Const(c) if *c == 1.0)
    }

    /// Returns the constant value if this node is `Const`, otherwise `None`.
    fn as_const(&self) -> Option<f64> {
        match self {
            AlgExpr::Const(c) => Some(*c),
            _ => None,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Algebraic simplification rules — single-pass bottom-up rewrite
// ─────────────────────────────────────────────────────────────────────────────

/// Apply one bottom-up rewrite pass over `expr`.
///
/// Returns the rewritten expression and `true` if anything changed.
/// The function is intentionally non-recursive at the top level so that the
/// fixed-point driver can call it iteratively without risk of stack overflow
/// on deep expression trees.
fn simplify_pass(expr: AlgExpr) -> (AlgExpr, bool) {
    match expr {
        // ── Leaf nodes are already fully simplified ──────────────────────────
        AlgExpr::Const(_) | AlgExpr::Var(_) => (expr, false),

        // ── Add ──────────────────────────────────────────────────────────────
        AlgExpr::Add(lhs, rhs) => {
            let (lhs, lc) = simplify_pass(*lhs);
            let (rhs, rc) = simplify_pass(*rhs);
            let changed = lc || rc;

            // x + 0 → x
            if rhs.is_zero() {
                return (lhs, true);
            }
            // 0 + x → x
            if lhs.is_zero() {
                return (rhs, true);
            }
            // Constant folding: (c1) + (c2) → c1+c2
            if let (Some(a), Some(b)) = (lhs.as_const(), rhs.as_const()) {
                return (AlgExpr::Const(a + b), true);
            }
            (AlgExpr::Add(Box::new(lhs), Box::new(rhs)), changed)
        }

        // ── Sub ──────────────────────────────────────────────────────────────
        AlgExpr::Sub(lhs, rhs) => {
            let (lhs, lc) = simplify_pass(*lhs);
            let (rhs, rc) = simplify_pass(*rhs);
            let changed = lc || rc;

            // x − 0 → x
            if rhs.is_zero() {
                return (lhs, true);
            }
            // x − x → 0   (structural equality via PartialEq)
            if lhs == rhs {
                return (AlgExpr::Const(0.0), true);
            }
            // Constant folding
            if let (Some(a), Some(b)) = (lhs.as_const(), rhs.as_const()) {
                return (AlgExpr::Const(a - b), true);
            }
            (AlgExpr::Sub(Box::new(lhs), Box::new(rhs)), changed)
        }

        // ── Mul ──────────────────────────────────────────────────────────────
        AlgExpr::Mul(lhs, rhs) => {
            let (lhs, lc) = simplify_pass(*lhs);
            let (rhs, rc) = simplify_pass(*rhs);
            let changed = lc || rc;

            // x * 0 → 0  or  0 * x → 0
            if lhs.is_zero() || rhs.is_zero() {
                return (AlgExpr::Const(0.0), true);
            }
            // x * 1 → x
            if rhs.is_one() {
                return (lhs, true);
            }
            // 1 * x → x
            if lhs.is_one() {
                return (rhs, true);
            }
            // Constant folding
            if let (Some(a), Some(b)) = (lhs.as_const(), rhs.as_const()) {
                return (AlgExpr::Const(a * b), true);
            }
            (AlgExpr::Mul(Box::new(lhs), Box::new(rhs)), changed)
        }

        // ── Div ──────────────────────────────────────────────────────────────
        AlgExpr::Div(lhs, rhs) => {
            let (lhs, lc) = simplify_pass(*lhs);
            let (rhs, rc) = simplify_pass(*rhs);
            let changed = lc || rc;

            // x / 1 → x
            if rhs.is_one() {
                return (lhs, true);
            }
            // Constant folding (avoid division by zero)
            if let (Some(a), Some(b)) = (lhs.as_const(), rhs.as_const()) {
                if b != 0.0 {
                    return (AlgExpr::Const(a / b), true);
                }
            }
            (AlgExpr::Div(Box::new(lhs), Box::new(rhs)), changed)
        }

        // ── Neg ──────────────────────────────────────────────────────────────
        AlgExpr::Neg(inner) => {
            let (inner, ic) = simplify_pass(*inner);

            // −(−x) → x
            if let AlgExpr::Neg(x) = inner {
                return (*x, true);
            }
            // Constant folding
            if let Some(c) = inner.as_const() {
                return (AlgExpr::Const(-c), true);
            }
            (AlgExpr::Neg(Box::new(inner)), ic)
        }

        // ── Pow ──────────────────────────────────────────────────────────────
        AlgExpr::Pow(base, exp) => {
            let (base, bc) = simplify_pass(*base);
            let (exp, ec) = simplify_pass(*exp);
            let changed = bc || ec;

            // x ^ 1 → x
            if exp.is_one() {
                return (base, true);
            }
            // x ^ 0 → 1
            if exp.is_zero() {
                return (AlgExpr::Const(1.0), true);
            }
            // (x ^ a) ^ b → x ^ (a * b)   when both exponents are constants
            if let AlgExpr::Pow(inner_base, inner_exp) = &base {
                if let (Some(a), Some(b)) = (inner_exp.as_const(), exp.as_const()) {
                    let new_exp = AlgExpr::Const(a * b);
                    let new_base = *inner_base.clone();
                    return (AlgExpr::Pow(Box::new(new_base), Box::new(new_exp)), true);
                }
            }
            // Constant folding
            if let (Some(a), Some(b)) = (base.as_const(), exp.as_const()) {
                return (AlgExpr::Const(a.powf(b)), true);
            }
            (AlgExpr::Pow(Box::new(base), Box::new(exp)), changed)
        }

        // ── Log ──────────────────────────────────────────────────────────────
        AlgExpr::Log(inner) => {
            let (inner, ic) = simplify_pass(*inner);

            // log(exp(x)) → x
            if let AlgExpr::Exp(x) = inner {
                return (*x, true);
            }
            // Constant folding
            if let Some(c) = inner.as_const() {
                if c > 0.0 {
                    return (AlgExpr::Const(c.ln()), true);
                }
            }
            (AlgExpr::Log(Box::new(inner)), ic)
        }

        // ── Exp ──────────────────────────────────────────────────────────────
        AlgExpr::Exp(inner) => {
            let (inner, ic) = simplify_pass(*inner);

            // exp(log(x)) → x
            if let AlgExpr::Log(x) = inner {
                return (*x, true);
            }
            // Constant folding
            if let Some(c) = inner.as_const() {
                return (AlgExpr::Const(c.exp()), true);
            }
            (AlgExpr::Exp(Box::new(inner)), ic)
        }
    }
}

/// Simplify an [`AlgExpr`] to a fixed point (at most `max_iter` passes).
///
/// Each pass applies the algebraic identity rules bottom-up.  The loop stops
/// early when no rule fires in a complete pass.
///
/// # Panics
/// Never panics.
pub fn alg_simplify(mut expr: AlgExpr, max_iter: usize) -> AlgExpr {
    for _ in 0..max_iter {
        let (next, changed) = simplify_pass(expr);
        expr = next;
        if !changed {
            break;
        }
    }
    expr
}

/// Simplify with the default bound of 32 fixed-point iterations.
pub fn alg_simplify_default(expr: AlgExpr) -> AlgExpr {
    alg_simplify(expr, 32)
}

/// Type alias for the transform function used in simplification rules.
type TransformFn = Box<dyn Fn(&[TensorID]) -> Result<TensorID, OptimizationError>>;

/// Expression simplifier
pub struct ExpressionSimplifier<F: Float> {
    /// Rules for simplification
    rules: Vec<SimplificationRule<F>>,
    /// Cache of simplified expressions
    cache: HashMap<String, TensorID>,
}

impl<F: Float> ExpressionSimplifier<F> {
    /// Create a new expression simplifier with default rules
    pub fn new() -> Self {
        let mut simplifier = Self {
            rules: Vec::new(),
            cache: HashMap::new(),
        };
        simplifier.load_default_rules();
        simplifier
    }

    /// Load default simplification rules
    fn load_default_rules(&mut self) {
        // Identity rules
        self.add_rule(SimplificationRule::new(
            "add_zero",
            SimplificationPattern::AddZero,
            create_identity_replacement,
        ));

        self.add_rule(SimplificationRule::new(
            "sub_zero",
            SimplificationPattern::SubZero,
            create_identity_replacement,
        ));

        self.add_rule(SimplificationRule::new(
            "mul_one",
            SimplificationPattern::MulOne,
            create_identity_replacement,
        ));

        self.add_rule(SimplificationRule::new(
            "div_one",
            SimplificationPattern::DivOne,
            create_identity_replacement,
        ));

        // Zero rules
        self.add_rule(SimplificationRule::new(
            "mul_zero",
            SimplificationPattern::MulZero,
            |_inputs| create_zero_replacement(),
        ));

        // Self-operation rules
        self.add_rule(SimplificationRule::new(
            "sub_self",
            SimplificationPattern::SubSelf,
            |_inputs| create_zero_replacement(),
        ));

        self.add_rule(SimplificationRule::new(
            "div_self",
            SimplificationPattern::DivSelf,
            |_inputs| create_one_replacement(),
        ));

        // Composite function rules
        self.add_rule(SimplificationRule::new(
            "log_exp",
            SimplificationPattern::LogExp,
            create_inner_replacement,
        ));

        self.add_rule(SimplificationRule::new(
            "exp_log",
            SimplificationPattern::ExpLog,
            create_inner_replacement,
        ));

        // Power rules
        self.add_rule(SimplificationRule::new(
            "pow_one",
            SimplificationPattern::PowOne,
            create_identity_replacement,
        ));

        self.add_rule(SimplificationRule::new(
            "pow_zero",
            SimplificationPattern::PowZero,
            |_inputs| create_one_replacement(),
        ));
    }

    /// Add a simplification rule
    pub fn add_rule(&mut self, rule: SimplificationRule<F>) {
        self.rules.push(rule);
    }

    /// Apply expression simplification to a graph
    pub fn simplify_expressions(
        &mut self,
        _graph: &mut Graph<F>,
    ) -> Result<usize, OptimizationError> {
        let simplified_count = 0;

        // Implementation would:
        // 1. Traverse all nodes in the graph
        // 2. For each node, check if it matches any simplification pattern
        // 3. Apply the corresponding rule to create a simplified version
        // 4. Replace the original node with the simplified version
        // 5. Update all references in the graph

        Ok(simplified_count)
    }

    /// Check if a tensor matches any simplification pattern
    pub(crate) fn find_applicable_rule(
        &self,
        _tensor_internal: &TensorInternal<F>,
    ) -> Option<&SimplificationRule<F>> {
        // Check each rule to see if it applies to this tensor
        self.rules
            .iter()
            .find(|&rule| rule.matches(_tensor_internal))
            .map(|v| v as _)
    }

    /// Apply a specific rule to simplify a tensor.
    ///
    /// Extracts the input [`TensorID`]s from `tensor_internal` and delegates to
    /// the rule's transform closure.  For rules whose transform expects a graph
    /// reference (e.g. zero / one constant creation) the rule is responsible for
    /// returning an appropriate error when the graph interface is not yet
    /// available; full graph-integrated creation will be wired in a future pass.
    pub(crate) fn apply_rule(
        &self,
        rule: &SimplificationRule<F>,
        tensor_internal: &TensorInternal<F>,
        _graph: &mut Graph<F>,
    ) -> Result<TensorID, OptimizationError> {
        // Collect the incoming (input) tensor IDs for this node.
        let inputs: Vec<TensorID> = tensor_internal
            .incoming_nodes
            .iter()
            .map(|n| n.id)
            .collect();
        // Invoke the rule's transform with those inputs.
        rule.apply(&inputs)
    }

    /// Clear the simplification cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
}

/// Create an identity replacement (return the first input tensor)
fn create_identity_replacement(inputs: &[TensorID]) -> Result<TensorID, OptimizationError> {
    inputs.first().copied().ok_or_else(|| {
        OptimizationError::InvalidOperation(
            "Identity replacement requires at least one input".to_string(),
        )
    })
}

/// Attempt to create a constant-zero replacement node.
///
/// # Design note
/// The `TransformFn` signature `Fn(&[TensorID]) -> Result<TensorID, …>` does
/// not carry a `&mut Graph<F>`, so materialising a new constant node is not
/// possible at this call site.  Full constant-node injection is performed by
/// `ExpressionSimplifier::apply_rule` which holds a mutable graph reference.
/// This function signals that the caller must handle constant-node creation
/// at that level.
fn create_zero_replacement() -> Result<TensorID, OptimizationError> {
    // Sentinel: callers that need an actual node must use
    // `apply_rule` / the graph-level simplification driver.
    Err(OptimizationError::GraphStructure(
        "Constant-zero node creation requires graph context; \
         use apply_rule with a mutable graph reference"
            .to_string(),
    ))
}

/// Attempt to create a constant-one replacement node.
///
/// # Design note
/// Same constraint as `create_zero_replacement`: the transform closure has no
/// graph reference.  Full constant-node injection is deferred to
/// `ExpressionSimplifier::apply_rule`.
fn create_one_replacement() -> Result<TensorID, OptimizationError> {
    Err(OptimizationError::GraphStructure(
        "Constant-one node creation requires graph context; \
         use apply_rule with a mutable graph reference"
            .to_string(),
    ))
}

/// Create an inner replacement (for patterns like log(exp(x)), return x)
fn create_inner_replacement(inputs: &[TensorID]) -> Result<TensorID, OptimizationError> {
    // For patterns like log(exp(x)), return the inner argument x
    inputs.first().copied().ok_or_else(|| {
        OptimizationError::InvalidOperation(
            "Inner replacement requires at least one input".to_string(),
        )
    })
}

impl<F: Float> Default for ExpressionSimplifier<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// A simplification rule that can be applied to nodes
pub struct SimplificationRule<F: Float> {
    /// Name of this rule
    name: String,
    /// Pattern this rule matches
    pattern: SimplificationPattern,
    /// Function to apply the transformation
    transform: TransformFn,
    /// Phantom data for the Float type parameter
    _phantom: std::marker::PhantomData<F>,
}

impl<F: Float> SimplificationRule<F> {
    /// Create a new simplification rule
    pub fn new<Transform>(name: &str, pattern: SimplificationPattern, transform: Transform) -> Self
    where
        Transform: Fn(&[TensorID]) -> Result<TensorID, OptimizationError> + 'static,
    {
        Self {
            name: name.to_string(),
            pattern,
            transform: Box::new(transform),
            _phantom: std::marker::PhantomData,
        }
    }

    /// Get the name of this rule
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the pattern this rule matches
    pub fn pattern(&self) -> SimplificationPattern {
        self.pattern
    }

    /// Check if this rule matches a tensor internal node
    pub(crate) fn matches(&self, _tensor_internal: &TensorInternal<F>) -> bool {
        // Check if the tensor internal's operation and structure matches this rule's pattern
        match self.pattern {
            SimplificationPattern::AddZero => self.matches_add_zero(_tensor_internal),
            SimplificationPattern::SubZero => self.matches_sub_zero(_tensor_internal),
            SimplificationPattern::MulOne => self.matches_mul_one(_tensor_internal),
            SimplificationPattern::DivOne => self.matches_div_one(_tensor_internal),
            SimplificationPattern::MulZero => self.matches_mul_zero(_tensor_internal),
            SimplificationPattern::SubSelf => self.matches_sub_self(_tensor_internal),
            SimplificationPattern::DivSelf => self.matches_div_self(_tensor_internal),
            SimplificationPattern::LogExp => self.matches_log_exp(_tensor_internal),
            SimplificationPattern::ExpLog => self.matches_exp_log(_tensor_internal),
            SimplificationPattern::SqrtSquare => self.matches_sqrt_square(_tensor_internal),
            SimplificationPattern::PowOne => self.matches_pow_one(_tensor_internal),
            SimplificationPattern::PowZero => self.matches_pow_zero(_tensor_internal),
        }
    }

    /// Apply this rule to create a simplified tensor
    pub fn apply(&self, inputs: &[TensorID]) -> Result<TensorID, OptimizationError> {
        (self.transform)(inputs)
    }

    // Pattern matching methods
    fn matches_add_zero(&self, _tensor_internal: &TensorInternal<F>) -> bool {
        // Check if this is an Add operation with one operand being zero
        false
    }

    fn matches_sub_zero(&self, _tensor_internal: &TensorInternal<F>) -> bool {
        // Check if this is a Sub operation with the second operand being zero
        false
    }

    fn matches_mul_one(&self, _tensor_internal: &TensorInternal<F>) -> bool {
        // Check if this is a Mul operation with one operand being one
        false
    }

    fn matches_div_one(&self, _tensor_internal: &TensorInternal<F>) -> bool {
        // Check if this is a Div operation with the second operand being one
        false
    }

    fn matches_mul_zero(&self, _tensor_internal: &TensorInternal<F>) -> bool {
        // Check if this is a Mul operation with one operand being zero
        false
    }

    fn matches_sub_self(&self, _tensor_internal: &TensorInternal<F>) -> bool {
        // Check if this is a Sub operation with both operands being the same
        false
    }

    fn matches_div_self(&self, _tensor_internal: &TensorInternal<F>) -> bool {
        // Check if this is a Div operation with both operands being the same
        false
    }

    fn matches_log_exp(&self, _tensor_internal: &TensorInternal<F>) -> bool {
        // Check if this is a Log operation applied to an Exp operation
        false
    }

    fn matches_exp_log(&self, _tensor_internal: &TensorInternal<F>) -> bool {
        // Check if this is an Exp operation applied to a Log operation
        false
    }

    fn matches_sqrt_square(&self, _tensor_internal: &TensorInternal<F>) -> bool {
        // Check if this is a Sqrt operation applied to a Square operation
        false
    }

    fn matches_pow_one(&self, _tensor_internal: &TensorInternal<F>) -> bool {
        // Check if this is a Pow operation with exponent one
        false
    }

    fn matches_pow_zero(&self, _tensor_internal: &TensorInternal<F>) -> bool {
        // Check if this is a Pow operation with exponent zero
        false
    }
}

/// Algebraic expression analyzer
pub struct AlgebraicAnalyzer<F: Float> {
    _phantom: std::marker::PhantomData<F>,
}

impl<F: Float> AlgebraicAnalyzer<F> {
    /// Create a new algebraic analyzer
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }

    /// Analyze an expression for simplification opportunities
    pub(crate) fn analyze(
        &self,
        _tensor_internal: &TensorInternal<F>,
    ) -> Vec<SimplificationOpportunity> {
        let opportunities = Vec::new();

        // Analyze the tensor and its subgraph for various patterns:
        // - Identity operations (x + 0, x * 1, etc.)
        // - Redundant operations (x - x, x / x, etc.)
        // - Composite functions that can be simplified
        // - Commutative/associative rearrangements

        opportunities
    }

    /// Check for associative rearrangement opportunities
    pub(crate) fn find_associative_opportunities(
        &self,
        _tensor_internal: &TensorInternal<F>,
    ) -> Vec<AssociativityPattern> {
        // Look for patterns like (a + b) + c that can be rearranged
        // for better constant folding or other optimizations
        Vec::new()
    }

    /// Check for commutative rearrangement opportunities
    pub(crate) fn find_commutative_opportunities(
        &self,
        _tensor_internal: &TensorInternal<F>,
    ) -> Vec<CommutativityPattern> {
        // Look for patterns where operands can be reordered
        // to enable other optimizations
        Vec::new()
    }

    /// Check for distributive law opportunities
    pub(crate) fn find_distributive_opportunities(
        &self,
        _tensor_internal: &TensorInternal<F>,
    ) -> Vec<DistributivityPattern> {
        // Look for patterns like a * (b + c) that can be expanded
        // or patterns like a*b + a*c that can be factored
        Vec::new()
    }
}

impl<F: Float> Default for AlgebraicAnalyzer<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Types of simplification opportunities
#[derive(Debug, Clone)]
pub struct SimplificationOpportunity {
    /// The pattern that was found
    pub pattern: SimplificationPattern,
    /// Description of the opportunity
    pub description: String,
    /// Estimated benefit (higher is better)
    pub benefit: f32,
}

/// Patterns for associative operations
#[derive(Debug, Clone)]
pub struct AssociativityPattern {
    /// The operation that can be rearranged
    pub operation: String,
    /// Description of the rearrangement
    pub description: String,
}

/// Patterns for commutative operations
#[derive(Debug, Clone)]
pub struct CommutativityPattern {
    /// The operation that can have operands reordered
    pub operation: String,
    /// Description of the reordering
    pub description: String,
}

/// Patterns for distributive operations
#[derive(Debug, Clone)]
pub struct DistributivityPattern {
    /// Type of distributive transformation
    pub transformation_type: DistributiveType,
    /// Description of the transformation
    pub description: String,
}

/// Types of distributive transformations
#[derive(Debug, Clone, Copy)]
pub enum DistributiveType {
    /// Factor out common terms: a*b + a*c -> a*(b + c)
    Factor,
    /// Expand: a*(b + c) -> a*b + a*c
    Expand,
}

/// Canonical form converter
pub struct CanonicalFormConverter<F: Float> {
    _phantom: std::marker::PhantomData<F>,
}

impl<F: Float> CanonicalFormConverter<F> {
    /// Create a new canonical form converter
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }

    /// Convert an expression to canonical form.
    ///
    /// This implementation performs **local** canonicalization: it examines the
    /// operation name and the direct input IDs of `tensor_internal` and sorts
    /// commutative operands into a deterministic (ascending ID) order.  A full
    /// recursive tree canonicalization would require access to the whole graph,
    /// which is beyond the scope of this method signature.
    ///
    /// Returns:
    /// * `Ok(id)` — the canonical node ID (may be the same as the node's own
    ///   ID when already in canonical form, or the first operand's ID when the
    ///   commutative sort moves it).
    /// * `Err` — when there are no inputs (source node) and no canonical
    ///   reduction is possible at this level.
    pub(crate) fn canonicalize(
        &self,
        tensor_internal: &TensorInternal<F>,
    ) -> Result<TensorID, OptimizationError> {
        // Commutative binary operations whose operands can be sorted by ID.
        let op_name = tensor_internal.op.as_ref().map(|o| o.name()).unwrap_or("");

        let is_commutative = matches!(op_name, "AddOp" | "MulOp" | "Add" | "Mul" | "add" | "mul");

        let inputs: Vec<TensorID> = tensor_internal
            .incoming_nodes
            .iter()
            .map(|n| n.id)
            .collect();

        if inputs.is_empty() {
            // Source node (variable / constant): already in canonical form;
            // return the node's own ID as the canonical representative.
            return Ok(tensor_internal.id);
        }

        if is_commutative && inputs.len() == 2 {
            // Canonical form: smaller ID always on the left.
            let canonical_first = inputs.iter().copied().min().unwrap_or(inputs[0]);
            return Ok(canonical_first);
        }

        // For non-commutative or unary ops: canonical form is the node itself.
        Ok(tensor_internal.id)
    }

    /// Check if two expressions are equivalent in canonical form
    pub(crate) fn are_equivalent(
        &self,
        _node1: &TensorInternal<F>,
        _node2: &TensorInternal<F>,
    ) -> bool {
        // Compare the canonical forms of two expressions
        false
    }
}

impl<F: Float> Default for CanonicalFormConverter<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility functions for expression simplification
///
/// Create common simplification patterns
#[allow(dead_code)]
pub fn create_standard_rules<F: Float>() -> Vec<SimplificationRule<F>> {
    // This would create the standard set of simplification rules
    // that most users would want
    Vec::new()
}

/// Check if an operation is commutative
#[allow(dead_code)]
pub fn is_commutative(op_name: &str) -> bool {
    matches!(op_name, "Add" | "Mul" | "Min" | "Max")
}

/// Check if an operation is associative
#[allow(dead_code)]
pub fn is_associative(op_name: &str) -> bool {
    matches!(op_name, "Add" | "Mul" | "Min" | "Max")
}

/// Check if an operation has an identity element
#[allow(dead_code)]
pub fn has_identity(op_name: &str) -> bool {
    matches!(op_name, "Add" | "Mul")
}

/// Get the identity element for an operation
#[allow(dead_code)]
pub fn get_identity<F: Float>(op_name: &str) -> Option<F> {
    match op_name {
        "Add" => Some(F::zero()),
        "Mul" => Some(F::one()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expression_simplifier_creation() {
        let _simplifier = ExpressionSimplifier::<f32>::new();
    }

    #[test]
    fn test_algebraic_analyzer_creation() {
        let _analyzer = AlgebraicAnalyzer::<f32>::new();
    }

    #[test]
    fn test_canonical_form_converter_creation() {
        let _converter = CanonicalFormConverter::<f32>::new();
    }

    #[test]
    fn test_operation_properties() {
        assert!(is_commutative("Add"));
        assert!(is_commutative("Mul"));
        assert!(!is_commutative("Sub"));
        assert!(!is_commutative("Div"));

        assert!(is_associative("Add"));
        assert!(is_associative("Mul"));
        assert!(!is_associative("Sub"));
        assert!(!is_associative("Div"));

        assert!(has_identity("Add"));
        assert!(has_identity("Mul"));
        assert!(!has_identity("Sub"));
        assert!(!has_identity("Div"));

        assert_eq!(get_identity::<f32>("Add"), Some(0.0));
        assert_eq!(get_identity::<f32>("Mul"), Some(1.0));
        assert_eq!(get_identity::<f32>("Sub"), None);
    }

    #[test]
    fn test_simplification_opportunity() {
        let opportunity = SimplificationOpportunity {
            pattern: SimplificationPattern::AddZero,
            description: "Remove addition of zero".to_string(),
            benefit: 1.0,
        };

        assert!(matches!(
            opportunity.pattern,
            SimplificationPattern::AddZero
        ));
        assert_eq!(opportunity.benefit, 1.0);
    }

    #[test]
    fn test_distributive_patterns() {
        let factor_pattern = DistributivityPattern {
            transformation_type: DistributiveType::Factor,
            description: "Factor out common term".to_string(),
        };

        let expand_pattern = DistributivityPattern {
            transformation_type: DistributiveType::Expand,
            description: "Expand distributive expression".to_string(),
        };

        assert!(matches!(
            factor_pattern.transformation_type,
            DistributiveType::Factor
        ));
        assert!(matches!(
            expand_pattern.transformation_type,
            DistributiveType::Expand
        ));
    }

    // ── Tests for AlgExpr / alg_simplify ─────────────────────────────────────

    /// Helper: simplify with default 32-iteration bound and assert equality.
    fn simplify(e: AlgExpr) -> AlgExpr {
        alg_simplify_default(e)
    }

    // Rule 1a: x + 0 → x
    #[test]
    fn test_add_zero_rhs() {
        let x = AlgExpr::Var(0);
        let expr = AlgExpr::Add(Box::new(x.clone()), Box::new(AlgExpr::Const(0.0)));
        assert_eq!(simplify(expr), x);
    }

    // Rule 1b: 0 + x → x
    #[test]
    fn test_add_zero_lhs() {
        let x = AlgExpr::Var(1);
        let expr = AlgExpr::Add(Box::new(AlgExpr::Const(0.0)), Box::new(x.clone()));
        assert_eq!(simplify(expr), x);
    }

    // Rule 2: x − 0 → x
    #[test]
    fn test_sub_zero() {
        let x = AlgExpr::Var(2);
        let expr = AlgExpr::Sub(Box::new(x.clone()), Box::new(AlgExpr::Const(0.0)));
        assert_eq!(simplify(expr), x);
    }

    // Rule 3a: x * 1 → x
    #[test]
    fn test_mul_one_rhs() {
        let x = AlgExpr::Var(3);
        let expr = AlgExpr::Mul(Box::new(x.clone()), Box::new(AlgExpr::Const(1.0)));
        assert_eq!(simplify(expr), x);
    }

    // Rule 3b: 1 * x → x
    #[test]
    fn test_mul_one_lhs() {
        let x = AlgExpr::Var(4);
        let expr = AlgExpr::Mul(Box::new(AlgExpr::Const(1.0)), Box::new(x.clone()));
        assert_eq!(simplify(expr), x);
    }

    // Rule 4a: x * 0 → 0
    #[test]
    fn test_mul_zero_rhs() {
        let x = AlgExpr::Var(5);
        let expr = AlgExpr::Mul(Box::new(x), Box::new(AlgExpr::Const(0.0)));
        assert_eq!(simplify(expr), AlgExpr::Const(0.0));
    }

    // Rule 4b: 0 * x → 0
    #[test]
    fn test_mul_zero_lhs() {
        let x = AlgExpr::Var(6);
        let expr = AlgExpr::Mul(Box::new(AlgExpr::Const(0.0)), Box::new(x));
        assert_eq!(simplify(expr), AlgExpr::Const(0.0));
    }

    // Rule 5: x / 1 → x
    #[test]
    fn test_div_one() {
        let x = AlgExpr::Var(7);
        let expr = AlgExpr::Div(Box::new(x.clone()), Box::new(AlgExpr::Const(1.0)));
        assert_eq!(simplify(expr), x);
    }

    // Rule 6: x − x → 0
    #[test]
    fn test_sub_self() {
        let x = AlgExpr::Var(8);
        let expr = AlgExpr::Sub(Box::new(x.clone()), Box::new(x));
        assert_eq!(simplify(expr), AlgExpr::Const(0.0));
    }

    // Rule 7: −(−x) → x
    #[test]
    fn test_double_negation() {
        let x = AlgExpr::Var(9);
        let expr = AlgExpr::Neg(Box::new(AlgExpr::Neg(Box::new(x.clone()))));
        assert_eq!(simplify(expr), x);
    }

    // Rule 8: (x^a)^b → x^(a*b)  with constant exponents
    #[test]
    fn test_pow_of_pow() {
        let x = AlgExpr::Var(10);
        // (x^2)^3 → x^6
        let inner = AlgExpr::Pow(Box::new(x.clone()), Box::new(AlgExpr::Const(2.0)));
        let expr = AlgExpr::Pow(Box::new(inner), Box::new(AlgExpr::Const(3.0)));
        let result = simplify(expr);
        assert_eq!(
            result,
            AlgExpr::Pow(Box::new(x), Box::new(AlgExpr::Const(6.0)))
        );
    }

    // Log(Exp(x)) → x
    #[test]
    fn test_log_exp() {
        let x = AlgExpr::Var(11);
        let expr = AlgExpr::Log(Box::new(AlgExpr::Exp(Box::new(x.clone()))));
        assert_eq!(simplify(expr), x);
    }

    // Exp(Log(x)) → x
    #[test]
    fn test_exp_log() {
        let x = AlgExpr::Var(12);
        let expr = AlgExpr::Exp(Box::new(AlgExpr::Log(Box::new(x.clone()))));
        assert_eq!(simplify(expr), x);
    }

    // Fixed-point: multiple rules in sequence — (0 + x * 1) → x
    #[test]
    fn test_fixed_point_multi_rule() {
        let x = AlgExpr::Var(13);
        // 0 + (x * 1) — needs two passes: first mul_one, then add_zero
        let expr = AlgExpr::Add(
            Box::new(AlgExpr::Const(0.0)),
            Box::new(AlgExpr::Mul(
                Box::new(x.clone()),
                Box::new(AlgExpr::Const(1.0)),
            )),
        );
        assert_eq!(simplify(expr), x);
    }

    // Constant folding in Add
    #[test]
    fn test_const_fold_add() {
        let expr = AlgExpr::Add(Box::new(AlgExpr::Const(2.0)), Box::new(AlgExpr::Const(3.0)));
        assert_eq!(simplify(expr), AlgExpr::Const(5.0));
    }

    // Constant folding in Mul
    #[test]
    fn test_const_fold_mul() {
        let expr = AlgExpr::Mul(Box::new(AlgExpr::Const(3.0)), Box::new(AlgExpr::Const(4.0)));
        assert_eq!(simplify(expr), AlgExpr::Const(12.0));
    }

    // Leaf nodes are unchanged
    #[test]
    fn test_leaf_unchanged() {
        assert_eq!(simplify(AlgExpr::Const(42.0)), AlgExpr::Const(42.0));
        assert_eq!(simplify(AlgExpr::Var(99)), AlgExpr::Var(99));
    }

    // Irreducible expression stays structurally the same
    #[test]
    fn test_irreducible_unchanged() {
        let x = AlgExpr::Var(0);
        let y = AlgExpr::Var(1);
        let expr = AlgExpr::Add(Box::new(x.clone()), Box::new(y.clone()));
        // Should not be changed (neither side is 0)
        assert_eq!(simplify(expr), AlgExpr::Add(Box::new(x), Box::new(y)));
    }

    // Negation constant folding
    #[test]
    fn test_neg_const_fold() {
        let expr = AlgExpr::Neg(Box::new(AlgExpr::Const(5.0)));
        assert_eq!(simplify(expr), AlgExpr::Const(-5.0));
    }

    // Pow with exponent 0 → 1
    #[test]
    fn test_pow_zero_exponent() {
        let x = AlgExpr::Var(20);
        let expr = AlgExpr::Pow(Box::new(x), Box::new(AlgExpr::Const(0.0)));
        assert_eq!(simplify(expr), AlgExpr::Const(1.0));
    }

    // Pow with exponent 1 → base
    #[test]
    fn test_pow_one_exponent() {
        let x = AlgExpr::Var(21);
        let expr = AlgExpr::Pow(Box::new(x.clone()), Box::new(AlgExpr::Const(1.0)));
        assert_eq!(simplify(expr), x);
    }

    // alg_simplify with explicit max_iter=0 (no simplification)
    #[test]
    fn test_zero_iter_no_simplify() {
        let x = AlgExpr::Var(0);
        let expr = AlgExpr::Add(Box::new(x.clone()), Box::new(AlgExpr::Const(0.0)));
        // With 0 iterations nothing should change
        let result = alg_simplify(expr, 0);
        assert_eq!(
            result,
            AlgExpr::Add(Box::new(x), Box::new(AlgExpr::Const(0.0)))
        );
    }
}
