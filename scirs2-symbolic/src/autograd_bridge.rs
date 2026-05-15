//! Bridge to scirs2-autograd: symbolic gradient as the native AD tape backend.
//!
//! # Phase 1 — API surface only
//!
//! This module ships the trait surface (`SymbolicTape`, `TapeNode`) and a
//! reference implementation that lowers `LoweredOp` to a flat `Vec<TapeNode>`
//! in post-order. The actual scirs2-autograd cross-crate integration —
//! available in `scirs2-autograd` under the `symbolic` feature via
//! `eml_scalar_op` — was completed in v0.4.4. See
//! `scirs2-autograd::symbolic_backend` for the public API.
//!
//! # Why design-freedom unlock C?
//!
//! Direct integration (Phase 3) bypasses scirs2-autograd's existing
//! float-tape backend whenever a tensor's provenance is `LoweredOp`.
//! This means:
//! - **No information loss**: symbolic structure preserved through
//!   `Tensor` operations
//! - **Dispatch-by-provenance**: no feature flag — `Tensor + Tensor`
//!   keeps EML if both are EML, or falls back to float if either is
//! - **JIT-compilable result**: the end-of-graph tape can be Cranelift-JIT'd
//!   for ~50-100x speedup over the autograd interpreter
//!
//! # Phase 1 surface
//!
//! ```
//! use scirs2_symbolic::autograd_bridge::SymbolicTape;
//! use scirs2_symbolic::eml::LoweredOp;
//!
//! let op = LoweredOp::Mul(
//!     Box::new(LoweredOp::Var(0)),
//!     Box::new(LoweredOp::Var(0)),
//! );
//! let tape = SymbolicTape::from_lowered(&op);
//! assert!(!tape.is_empty());
//! ```

use crate::eml::op::LoweredOp;

/// A single node on the symbolic tape — corresponds to a `LoweredOp` variant.
///
/// `TapeNode` is the unit of dispatch for the future scirs2-autograd
/// integration: each tensor operation maps to one `TapeNode`, and the
/// `SymbolicTape` is the topologically-sorted sequence of operations
/// recorded during forward computation.
#[derive(Clone, Debug, PartialEq)]
pub enum TapeNode {
    /// A constant value.
    Const(f64),
    /// A variable input (indexed by external Var ID).
    Input(usize),
    /// Binary operator with operand indices into the tape.
    Binary {
        /// The operator kind (mirrors `LoweredOp` binary variants).
        op: BinaryKind,
        /// Index of left operand in the tape.
        lhs: usize,
        /// Index of right operand in the tape.
        rhs: usize,
    },
    /// Unary operator with operand index.
    Unary {
        /// The operator kind (mirrors `LoweredOp` unary variants).
        op: UnaryKind,
        /// Index of operand in the tape.
        arg: usize,
    },
}

/// Binary operator kinds for `TapeNode::Binary`.
#[allow(missing_docs)] // variants are self-documenting
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BinaryKind {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
}

/// Unary operator kinds for `TapeNode::Unary`.
#[allow(missing_docs)] // variants are self-documenting
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UnaryKind {
    Neg,
    Exp,
    Ln,
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
    Sqrt,
    Abs,
}

/// A symbolic tape — flat sequence of `TapeNode`s in topological (post-order) order.
///
/// The last node is the result; earlier nodes are operands. Each `Binary` /
/// `Unary` node references operands by index into earlier positions.
#[derive(Clone, Debug, Default)]
pub struct SymbolicTape {
    nodes: Vec<TapeNode>,
}

impl SymbolicTape {
    /// New, empty tape.
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    /// Build from a `LoweredOp` via iterative post-order flatten.
    ///
    /// No recursion — uses a heap work-stack mirroring the pattern in
    /// [`LoweredOp::to_oxi_ops`]. An auxiliary `index_stack` tracks where
    /// each emitted child landed in `self.nodes`, so the parent operator
    /// can reference them by index.
    pub fn from_lowered(op: &LoweredOp) -> Self {
        let mut tape = Self::new();
        let mut work: Vec<(&LoweredOp, bool)> = vec![(op, false)];
        let mut index_stack: Vec<usize> = Vec::new();

        while let Some((node, visited)) = work.pop() {
            if visited {
                let new_idx = tape.nodes.len();
                let new_node = match node {
                    LoweredOp::Const(c) => TapeNode::Const(*c),
                    LoweredOp::Var(i) => TapeNode::Input(*i),
                    LoweredOp::Add(_, _) => {
                        let rhs = index_stack.pop().expect("post-order: binary rhs index");
                        let lhs = index_stack.pop().expect("post-order: binary lhs index");
                        TapeNode::Binary {
                            op: BinaryKind::Add,
                            lhs,
                            rhs,
                        }
                    }
                    LoweredOp::Sub(_, _) => {
                        let rhs = index_stack.pop().expect("post-order: binary rhs index");
                        let lhs = index_stack.pop().expect("post-order: binary lhs index");
                        TapeNode::Binary {
                            op: BinaryKind::Sub,
                            lhs,
                            rhs,
                        }
                    }
                    LoweredOp::Mul(_, _) => {
                        let rhs = index_stack.pop().expect("post-order: binary rhs index");
                        let lhs = index_stack.pop().expect("post-order: binary lhs index");
                        TapeNode::Binary {
                            op: BinaryKind::Mul,
                            lhs,
                            rhs,
                        }
                    }
                    LoweredOp::Div(_, _) => {
                        let rhs = index_stack.pop().expect("post-order: binary rhs index");
                        let lhs = index_stack.pop().expect("post-order: binary lhs index");
                        TapeNode::Binary {
                            op: BinaryKind::Div,
                            lhs,
                            rhs,
                        }
                    }
                    LoweredOp::Pow(_, _) => {
                        let rhs = index_stack.pop().expect("post-order: binary rhs index");
                        let lhs = index_stack.pop().expect("post-order: binary lhs index");
                        TapeNode::Binary {
                            op: BinaryKind::Pow,
                            lhs,
                            rhs,
                        }
                    }
                    LoweredOp::Neg(_) => {
                        let arg = index_stack.pop().expect("post-order: unary arg index");
                        TapeNode::Unary {
                            op: UnaryKind::Neg,
                            arg,
                        }
                    }
                    LoweredOp::Exp(_) => {
                        let arg = index_stack.pop().expect("post-order: unary arg index");
                        TapeNode::Unary {
                            op: UnaryKind::Exp,
                            arg,
                        }
                    }
                    LoweredOp::Ln(_) => {
                        let arg = index_stack.pop().expect("post-order: unary arg index");
                        TapeNode::Unary {
                            op: UnaryKind::Ln,
                            arg,
                        }
                    }
                    LoweredOp::Sin(_) => {
                        let arg = index_stack.pop().expect("post-order: unary arg index");
                        TapeNode::Unary {
                            op: UnaryKind::Sin,
                            arg,
                        }
                    }
                    LoweredOp::Cos(_) => {
                        let arg = index_stack.pop().expect("post-order: unary arg index");
                        TapeNode::Unary {
                            op: UnaryKind::Cos,
                            arg,
                        }
                    }
                    LoweredOp::Tan(_) => {
                        let arg = index_stack.pop().expect("post-order: unary arg index");
                        TapeNode::Unary {
                            op: UnaryKind::Tan,
                            arg,
                        }
                    }
                    LoweredOp::Sinh(_) => {
                        let arg = index_stack.pop().expect("post-order: unary arg index");
                        TapeNode::Unary {
                            op: UnaryKind::Sinh,
                            arg,
                        }
                    }
                    LoweredOp::Cosh(_) => {
                        let arg = index_stack.pop().expect("post-order: unary arg index");
                        TapeNode::Unary {
                            op: UnaryKind::Cosh,
                            arg,
                        }
                    }
                    LoweredOp::Tanh(_) => {
                        let arg = index_stack.pop().expect("post-order: unary arg index");
                        TapeNode::Unary {
                            op: UnaryKind::Tanh,
                            arg,
                        }
                    }
                    LoweredOp::Arcsin(_) => {
                        let arg = index_stack.pop().expect("post-order: unary arg index");
                        TapeNode::Unary {
                            op: UnaryKind::Arcsin,
                            arg,
                        }
                    }
                    LoweredOp::Arccos(_) => {
                        let arg = index_stack.pop().expect("post-order: unary arg index");
                        TapeNode::Unary {
                            op: UnaryKind::Arccos,
                            arg,
                        }
                    }
                    LoweredOp::Arctan(_) => {
                        let arg = index_stack.pop().expect("post-order: unary arg index");
                        TapeNode::Unary {
                            op: UnaryKind::Arctan,
                            arg,
                        }
                    }
                    LoweredOp::Arcsinh(_) => {
                        let arg = index_stack.pop().expect("post-order: unary arg index");
                        TapeNode::Unary {
                            op: UnaryKind::Arcsinh,
                            arg,
                        }
                    }
                    LoweredOp::Arccosh(_) => {
                        let arg = index_stack.pop().expect("post-order: unary arg index");
                        TapeNode::Unary {
                            op: UnaryKind::Arccosh,
                            arg,
                        }
                    }
                    LoweredOp::Arctanh(_) => {
                        let arg = index_stack.pop().expect("post-order: unary arg index");
                        TapeNode::Unary {
                            op: UnaryKind::Arctanh,
                            arg,
                        }
                    }
                    LoweredOp::Sqrt(_) => {
                        let arg = index_stack.pop().expect("post-order: unary arg index");
                        TapeNode::Unary {
                            op: UnaryKind::Sqrt,
                            arg,
                        }
                    }
                    LoweredOp::Abs(_) => {
                        let arg = index_stack.pop().expect("post-order: unary arg index");
                        TapeNode::Unary {
                            op: UnaryKind::Abs,
                            arg,
                        }
                    }
                };
                tape.nodes.push(new_node);
                index_stack.push(new_idx);
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
                        // Push right first so it pops second; left pops first
                        // and lands in the index_stack first too.
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
        }

        tape
    }

    /// Number of nodes on the tape.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// True if the tape is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Read-only access to the underlying nodes.
    pub fn nodes(&self) -> &[TapeNode] {
        &self.nodes
    }

    /// Index of the result node (the last node).
    pub fn result_index(&self) -> Option<usize> {
        if self.nodes.is_empty() {
            None
        } else {
            Some(self.nodes.len() - 1)
        }
    }
}

/// Convert a `LoweredOp` to a `SymbolicTape`. Convenience trait on `LoweredOp`.
pub trait ToTape {
    /// Convert to a symbolic tape.
    fn to_tape(&self) -> SymbolicTape;
}

impl ToTape for LoweredOp {
    fn to_tape(&self) -> SymbolicTape {
        SymbolicTape::from_lowered(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_tape() {
        let tape = SymbolicTape::new();
        assert!(tape.is_empty());
        assert_eq!(tape.len(), 0);
        assert!(tape.result_index().is_none());
    }

    #[test]
    fn const_tape() {
        let op = LoweredOp::Const(2.5);
        let tape = SymbolicTape::from_lowered(&op);
        assert_eq!(tape.len(), 1);
        assert_eq!(tape.nodes()[0], TapeNode::Const(2.5));
        assert_eq!(tape.result_index(), Some(0));
    }

    #[test]
    fn var_tape() {
        let op = LoweredOp::Var(2);
        let tape = SymbolicTape::from_lowered(&op);
        assert_eq!(tape.len(), 1);
        assert_eq!(tape.nodes()[0], TapeNode::Input(2));
    }

    #[test]
    fn binary_tape_add() {
        let op = LoweredOp::Add(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Const(1.0)));
        let tape = SymbolicTape::from_lowered(&op);
        // Post-order: [Var(0), Const(1.0), Add{lhs=0, rhs=1}]
        assert_eq!(tape.len(), 3);
        assert_eq!(tape.nodes()[0], TapeNode::Input(0));
        assert_eq!(tape.nodes()[1], TapeNode::Const(1.0));
        match &tape.nodes()[2] {
            TapeNode::Binary {
                op: BinaryKind::Add,
                lhs,
                rhs,
            } => {
                assert_eq!(*lhs, 0);
                assert_eq!(*rhs, 1);
            }
            other => panic!("expected Binary Add, got {:?}", other),
        }
    }

    #[test]
    fn unary_tape_sin() {
        let op = LoweredOp::Sin(Box::new(LoweredOp::Var(0)));
        let tape = SymbolicTape::from_lowered(&op);
        assert_eq!(tape.len(), 2);
        assert_eq!(tape.nodes()[0], TapeNode::Input(0));
        match &tape.nodes()[1] {
            TapeNode::Unary {
                op: UnaryKind::Sin,
                arg,
            } => assert_eq!(*arg, 0),
            other => panic!("expected Unary Sin, got {:?}", other),
        }
    }

    #[test]
    fn nested_tape() {
        // Mul(Add(x, 1), x)
        let op = LoweredOp::Mul(
            Box::new(LoweredOp::Add(
                Box::new(LoweredOp::Var(0)),
                Box::new(LoweredOp::Const(1.0)),
            )),
            Box::new(LoweredOp::Var(0)),
        );
        let tape = SymbolicTape::from_lowered(&op);
        // Post-order: [Var(0), Const(1), Add(0,1), Var(0), Mul(2,3)]
        assert_eq!(tape.len(), 5);
    }

    #[test]
    fn deep_chain_no_overflow() {
        // 5000-deep right-chain — must not blow the OS stack
        let mut op = LoweredOp::Var(0);
        for _ in 0..5000 {
            op = LoweredOp::Add(Box::new(op), Box::new(LoweredOp::Const(1.0)));
        }
        let tape = SymbolicTape::from_lowered(&op);
        assert_eq!(tape.len(), 1 + 5000 * 2); // 1 leaf + 5000 (Const + Add) pairs
    }

    #[test]
    fn to_tape_trait_works() {
        let op = LoweredOp::Var(0);
        let tape = op.to_tape();
        assert_eq!(tape.len(), 1);
    }

    #[test]
    fn binary_kind_eq() {
        assert_eq!(BinaryKind::Add, BinaryKind::Add);
        assert_ne!(BinaryKind::Add, BinaryKind::Sub);
    }

    #[test]
    fn unary_kind_eq() {
        assert_eq!(UnaryKind::Sin, UnaryKind::Sin);
        assert_ne!(UnaryKind::Sin, UnaryKind::Cos);
    }
}
