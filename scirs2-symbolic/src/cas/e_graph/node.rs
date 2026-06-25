//! E-graph node types: `ClassId`, `NodeKind`, `ENode`, `EClass`.
//!
//! # `NodeKind` bit-layout for `Const`
//!
//! Constant f64 values are stored as their raw `u64` bit pattern via
//! `f64::to_bits()`. This means `-0.0` and `+0.0` hash differently and
//! different NaN bit patterns produce different nodes. This is intentional
//! and documented here to avoid surprises: the e-graph treats syntactic
//! bit-equality as node equality, not semantic equality.

use crate::cas::pattern::{BinaryKind, UnaryKind};
use crate::eml::LoweredOp;

/// Opaque identifier for an e-class.
///
/// Ids are assigned monotonically by `EGraph::add`. After a `union(a, b)`,
/// one id becomes the representative; the other may be defunct. Always use
/// `egraph.find(id)` to get the canonical representative.
///
/// Implements `Ord` so that `Vec<ClassId>` can be sorted to ensure
/// deterministic iteration order during equality saturation —
/// `HashMap` iteration order is non-deterministic (randomised hash seed), so
/// collecting and sorting the class ids before the saturation inner loop
/// eliminates iteration-order sensitivity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ClassId(pub u32);

/// The operation type stored in an [`ENode`], mirroring [`LoweredOp`].
///
/// Unlike `LoweredOp`, children are represented by `ClassId` in the
/// `ENode.children` field rather than being embedded recursively — this
/// breaks the recursive ownership required by tree-structured IR and is
/// what enables e-graph sharing.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum NodeKind {
    /// Constant, stored as raw f64 bits for hash/eq purposes.
    Const(u64),
    /// Variable reference by index.
    Var(usize),
    // Unary ops
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
    // Binary ops
    Add,
    Sub,
    Mul,
    Div,
    Pow,
}

/// An e-node: a single operator with children referencing e-classes.
///
/// The `children` field is a boxed slice. For leaves (`Const`, `Var`) it is
/// empty; for unary ops it has one element; for binary ops it has two.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ENode {
    pub kind: NodeKind,
    pub children: Box<[ClassId]>,
}

impl ENode {
    /// Construct a leaf node (no children).
    pub fn leaf(kind: NodeKind) -> Self {
        ENode {
            kind,
            children: Box::new([]),
        }
    }

    /// Construct a unary node.
    pub fn unary(kind: NodeKind, child: ClassId) -> Self {
        ENode {
            kind,
            children: Box::new([child]),
        }
    }

    /// Construct a binary node.
    pub fn binary(kind: NodeKind, left: ClassId, right: ClassId) -> Self {
        ENode {
            kind,
            children: Box::new([left, right]),
        }
    }
}

/// An e-class: a set of equivalent e-nodes plus parent back-edges.
///
/// # Parent back-edges
///
/// `parents` contains `(parent_enode, parent_class_id)` pairs for every
/// node whose children include this class. These are used by `rebuild` to
/// re-check the hashcons after a union changes which class is canonical.
pub struct EClass {
    pub id: ClassId,
    pub nodes: Vec<ENode>,
    pub parents: Vec<(ENode, ClassId)>,
}

impl EClass {
    /// Create a new e-class with one initial node.
    pub fn new(id: ClassId, initial: ENode) -> Self {
        EClass {
            id,
            nodes: vec![initial],
            parents: Vec::new(),
        }
    }
}

/// Convert a `NodeKind` back to the `LoweredOp` *operator shape* with
/// placeholder children. Used during extraction.
///
/// This does not fill in children — callers must supply them.
pub(crate) fn lowered_leaf(kind: &NodeKind) -> Option<LoweredOp> {
    match kind {
        NodeKind::Const(bits) => Some(LoweredOp::Const(f64::from_bits(*bits))),
        NodeKind::Var(i) => Some(LoweredOp::Var(*i)),
        _ => None,
    }
}

/// Convert a `NodeKind` + one child `LoweredOp` into a `LoweredOp`.
pub(crate) fn lowered_unary(kind: &NodeKind, child: LoweredOp) -> Option<LoweredOp> {
    let b = Box::new(child);
    match kind {
        NodeKind::Neg => Some(LoweredOp::Neg(b)),
        NodeKind::Exp => Some(LoweredOp::Exp(b)),
        NodeKind::Ln => Some(LoweredOp::Ln(b)),
        NodeKind::Sin => Some(LoweredOp::Sin(b)),
        NodeKind::Cos => Some(LoweredOp::Cos(b)),
        NodeKind::Tan => Some(LoweredOp::Tan(b)),
        NodeKind::Sinh => Some(LoweredOp::Sinh(b)),
        NodeKind::Cosh => Some(LoweredOp::Cosh(b)),
        NodeKind::Tanh => Some(LoweredOp::Tanh(b)),
        NodeKind::Arcsin => Some(LoweredOp::Arcsin(b)),
        NodeKind::Arccos => Some(LoweredOp::Arccos(b)),
        NodeKind::Arctan => Some(LoweredOp::Arctan(b)),
        NodeKind::Arcsinh => Some(LoweredOp::Arcsinh(b)),
        NodeKind::Arccosh => Some(LoweredOp::Arccosh(b)),
        NodeKind::Arctanh => Some(LoweredOp::Arctanh(b)),
        NodeKind::Sqrt => Some(LoweredOp::Sqrt(b)),
        NodeKind::Abs => Some(LoweredOp::Abs(b)),
        _ => None,
    }
}

/// Convert a `NodeKind` + two child `LoweredOp`s into a `LoweredOp`.
pub(crate) fn lowered_binary(
    kind: &NodeKind,
    left: LoweredOp,
    right: LoweredOp,
) -> Option<LoweredOp> {
    let (bl, br) = (Box::new(left), Box::new(right));
    match kind {
        NodeKind::Add => Some(LoweredOp::Add(bl, br)),
        NodeKind::Sub => Some(LoweredOp::Sub(bl, br)),
        NodeKind::Mul => Some(LoweredOp::Mul(bl, br)),
        NodeKind::Div => Some(LoweredOp::Div(bl, br)),
        NodeKind::Pow => Some(LoweredOp::Pow(bl, br)),
        _ => None,
    }
}

/// Convert a `LoweredOp` leaf to a `NodeKind` (no children).
pub(crate) fn node_kind_of_leaf(op: &LoweredOp) -> Option<NodeKind> {
    match op {
        LoweredOp::Const(v) => Some(NodeKind::Const(v.to_bits())),
        LoweredOp::Var(i) => Some(NodeKind::Var(*i)),
        _ => None,
    }
}

/// Convert a unary `LoweredOp`'s outer kind (ignoring the child) to a `NodeKind`.
pub(crate) fn node_kind_of_unary(op: &LoweredOp) -> Option<NodeKind> {
    match op {
        LoweredOp::Neg(_) => Some(NodeKind::Neg),
        LoweredOp::Exp(_) => Some(NodeKind::Exp),
        LoweredOp::Ln(_) => Some(NodeKind::Ln),
        LoweredOp::Sin(_) => Some(NodeKind::Sin),
        LoweredOp::Cos(_) => Some(NodeKind::Cos),
        LoweredOp::Tan(_) => Some(NodeKind::Tan),
        LoweredOp::Sinh(_) => Some(NodeKind::Sinh),
        LoweredOp::Cosh(_) => Some(NodeKind::Cosh),
        LoweredOp::Tanh(_) => Some(NodeKind::Tanh),
        LoweredOp::Arcsin(_) => Some(NodeKind::Arcsin),
        LoweredOp::Arccos(_) => Some(NodeKind::Arccos),
        LoweredOp::Arctan(_) => Some(NodeKind::Arctan),
        LoweredOp::Arcsinh(_) => Some(NodeKind::Arcsinh),
        LoweredOp::Arccosh(_) => Some(NodeKind::Arccosh),
        LoweredOp::Arctanh(_) => Some(NodeKind::Arctanh),
        LoweredOp::Sqrt(_) => Some(NodeKind::Sqrt),
        LoweredOp::Abs(_) => Some(NodeKind::Abs),
        _ => None,
    }
}

/// Convert a binary `LoweredOp`'s outer kind (ignoring children) to a `NodeKind`.
pub(crate) fn node_kind_of_binary(op: &LoweredOp) -> Option<NodeKind> {
    match op {
        LoweredOp::Add(_, _) => Some(NodeKind::Add),
        LoweredOp::Sub(_, _) => Some(NodeKind::Sub),
        LoweredOp::Mul(_, _) => Some(NodeKind::Mul),
        LoweredOp::Div(_, _) => Some(NodeKind::Div),
        LoweredOp::Pow(_, _) => Some(NodeKind::Pow),
        _ => None,
    }
}

/// Convert a `UnaryKind` to `NodeKind`.
pub(crate) fn node_kind_of_unary_kind(k: UnaryKind) -> NodeKind {
    match k {
        UnaryKind::Neg => NodeKind::Neg,
        UnaryKind::Exp => NodeKind::Exp,
        UnaryKind::Ln => NodeKind::Ln,
        UnaryKind::Sin => NodeKind::Sin,
        UnaryKind::Cos => NodeKind::Cos,
        UnaryKind::Tan => NodeKind::Tan,
        UnaryKind::Sinh => NodeKind::Sinh,
        UnaryKind::Cosh => NodeKind::Cosh,
        UnaryKind::Tanh => NodeKind::Tanh,
        UnaryKind::Arcsin => NodeKind::Arcsin,
        UnaryKind::Arccos => NodeKind::Arccos,
        UnaryKind::Arctan => NodeKind::Arctan,
        UnaryKind::Arcsinh => NodeKind::Arcsinh,
        UnaryKind::Arccosh => NodeKind::Arccosh,
        UnaryKind::Arctanh => NodeKind::Arctanh,
        UnaryKind::Sqrt => NodeKind::Sqrt,
        UnaryKind::Abs => NodeKind::Abs,
    }
}

/// Convert a `BinaryKind` to `NodeKind`.
pub(crate) fn node_kind_of_binary_kind(k: BinaryKind) -> NodeKind {
    match k {
        BinaryKind::Add => NodeKind::Add,
        BinaryKind::Sub => NodeKind::Sub,
        BinaryKind::Mul => NodeKind::Mul,
        BinaryKind::Div => NodeKind::Div,
        BinaryKind::Pow => NodeKind::Pow,
    }
}
