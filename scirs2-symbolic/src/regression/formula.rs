//! Discovered formula representation.
//!
//! A [`DiscoveredFormula`] pairs a [`LoweredOp`] (the algebraic structure)
//! with its training-set [`Fitness`], plus a node count for parsimony.

use crate::eml::LoweredOp;
use crate::regression::Fitness;

/// A formula discovered by symbolic regression.
#[derive(Clone, Debug)]
pub struct DiscoveredFormula {
    /// The `LoweredOp` representation of the formula.
    pub op: LoweredOp,
    /// Fitness on the training data.
    pub fitness: Fitness,
    /// Number of nodes (parsimony measure).
    pub node_count: usize,
    /// Variable count (max var index + 1).
    pub n_vars: usize,
}

impl DiscoveredFormula {
    /// New formula with the given op and fitness.
    pub fn new(op: LoweredOp, fitness: Fitness) -> Self {
        let node_count = count_nodes(&op);
        let n_vars = op.count_vars();
        Self {
            op,
            fitness,
            node_count,
            n_vars,
        }
    }
}

/// Iterative O(N) node count over a [`LoweredOp`] tree.
///
/// Exposed at `pub(crate)` so the search engine in
/// [`mod@crate::regression::discover`] can size-cap candidate ops without
/// reconstructing a [`DiscoveredFormula`] just for the count.
pub(crate) fn count_nodes(op: &LoweredOp) -> usize {
    let mut count = 0;
    let mut work: Vec<&LoweredOp> = vec![op];
    while let Some(n) = work.pop() {
        count += 1;
        match n {
            LoweredOp::Const(_) | LoweredOp::Var(_) => {}
            LoweredOp::Add(a, b)
            | LoweredOp::Sub(a, b)
            | LoweredOp::Mul(a, b)
            | LoweredOp::Div(a, b)
            | LoweredOp::Pow(a, b) => {
                work.push(a);
                work.push(b);
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
                work.push(c);
            }
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn count_leaf() {
        assert_eq!(count_nodes(&LoweredOp::Var(0)), 1);
        assert_eq!(count_nodes(&LoweredOp::Const(2.0)), 1);
    }

    #[test]
    fn count_binary() {
        let op = LoweredOp::Add(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Const(1.0)));
        assert_eq!(count_nodes(&op), 3);
    }

    #[test]
    fn count_unary_chain() {
        // Sin(Cos(Var(0))) = 3 nodes
        let op = LoweredOp::Sin(Box::new(LoweredOp::Cos(Box::new(LoweredOp::Var(0)))));
        assert_eq!(count_nodes(&op), 3);
    }

    #[test]
    fn formula_records_node_count() {
        let op = LoweredOp::Mul(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Var(0)));
        let f = DiscoveredFormula::new(op, Fitness::worst());
        assert_eq!(f.node_count, 3);
        assert_eq!(f.n_vars, 1);
    }
}
