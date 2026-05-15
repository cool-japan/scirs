//! `LoweredOp` and `OxiOp` — flat operator IR and stack-machine tape.
//!
//! Many CAS algorithms (gradient, simplify, JIT, eval) are easier on a flat
//! operator IR than on the recursive [`crate::eml::tree::EmlTree`]. The
//! lowering pass ([`crate::eml::lower::lower`]) walks an `EmlTree` post-order
//! and recognises canonical shapes (e.g. `eml(x, 1) → exp(x)`) emitting
//! matching `LoweredOp` variants.
//!
//! # Adapted from oxieml v0.1.0, `src/lower.rs`
//!
//! Variant set is preserved with two additions: `Sqrt` and `Abs` are
//! native here (oxieml lowers them to `Pow(_, 0.5)` and `sqrt(square(_))`).
//! Native variants enable cleaner gradient handling at boundary points
//! (`d/dx sqrt(x)` at `x=0`, `d/dx |x|` at `x=0`).

#![allow(missing_docs)] // variants are self-documenting

use crate::eml::hash::hash_u128;

/// Flat operator IR.
///
/// Variants form an algebraic tree (`Box` children). Includes native `Sqrt`
/// and `Abs` which oxieml lowers to `Pow(_, 0.5)` / `sqrt(x²)`; native
/// variants enable cleaner gradient handling at boundary points.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum LoweredOp {
    Const(f64),
    Var(usize),
    Add(Box<LoweredOp>, Box<LoweredOp>),
    Sub(Box<LoweredOp>, Box<LoweredOp>),
    Mul(Box<LoweredOp>, Box<LoweredOp>),
    Div(Box<LoweredOp>, Box<LoweredOp>),
    Pow(Box<LoweredOp>, Box<LoweredOp>),
    Neg(Box<LoweredOp>),
    Exp(Box<LoweredOp>),
    Ln(Box<LoweredOp>),
    Sin(Box<LoweredOp>),
    Cos(Box<LoweredOp>),
    Tan(Box<LoweredOp>),
    Sinh(Box<LoweredOp>),
    Cosh(Box<LoweredOp>),
    Tanh(Box<LoweredOp>),
    Arcsin(Box<LoweredOp>),
    Arccos(Box<LoweredOp>),
    Arctan(Box<LoweredOp>),
    Arcsinh(Box<LoweredOp>),
    Arccosh(Box<LoweredOp>),
    Arctanh(Box<LoweredOp>),
    /// Native — divergence from oxieml (which uses `Pow(_, 0.5)`).
    Sqrt(Box<LoweredOp>),
    /// Native — divergence from oxieml (which uses `sqrt(square(_))`).
    Abs(Box<LoweredOp>),
}

/// Stack-machine tape opcodes (for fast iterative evaluation).
///
/// Mirrors [`LoweredOp`] but without `Box` children — the [`LoweredOp::to_oxi_ops`]
/// flatten produces a `Vec<OxiOp>` post-order; the eval loop pushes/pops
/// a value stack.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum OxiOp {
    Const(f64),
    Var(usize),
    Add,
    Sub,
    Mul,
    Div,
    Pow,
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

impl LoweredOp {
    /// Flatten this `LoweredOp` to a post-order `Vec<OxiOp>` tape.
    ///
    /// Iterative — no recursion. Capacity preallocated to a reasonable
    /// initial estimate; the underlying `Vec` grows on demand.
    ///
    /// # Invariant
    ///
    /// The returned tape is suitable for stack-machine evaluation: each
    /// binary operator pops two operands (left then right), each unary
    /// operator pops one. Left operands are always emitted before right
    /// operands.
    pub fn to_oxi_ops(&self) -> Vec<OxiOp> {
        let mut output = Vec::with_capacity(64);
        let mut work: Vec<(&LoweredOp, bool)> = vec![(self, false)];
        while let Some((op, visited)) = work.pop() {
            if visited {
                // Post-order: emit the operator
                match op {
                    LoweredOp::Const(c) => output.push(OxiOp::Const(*c)),
                    LoweredOp::Var(i) => output.push(OxiOp::Var(*i)),
                    LoweredOp::Add(_, _) => output.push(OxiOp::Add),
                    LoweredOp::Sub(_, _) => output.push(OxiOp::Sub),
                    LoweredOp::Mul(_, _) => output.push(OxiOp::Mul),
                    LoweredOp::Div(_, _) => output.push(OxiOp::Div),
                    LoweredOp::Pow(_, _) => output.push(OxiOp::Pow),
                    LoweredOp::Neg(_) => output.push(OxiOp::Neg),
                    LoweredOp::Exp(_) => output.push(OxiOp::Exp),
                    LoweredOp::Ln(_) => output.push(OxiOp::Ln),
                    LoweredOp::Sin(_) => output.push(OxiOp::Sin),
                    LoweredOp::Cos(_) => output.push(OxiOp::Cos),
                    LoweredOp::Tan(_) => output.push(OxiOp::Tan),
                    LoweredOp::Sinh(_) => output.push(OxiOp::Sinh),
                    LoweredOp::Cosh(_) => output.push(OxiOp::Cosh),
                    LoweredOp::Tanh(_) => output.push(OxiOp::Tanh),
                    LoweredOp::Arcsin(_) => output.push(OxiOp::Arcsin),
                    LoweredOp::Arccos(_) => output.push(OxiOp::Arccos),
                    LoweredOp::Arctan(_) => output.push(OxiOp::Arctan),
                    LoweredOp::Arcsinh(_) => output.push(OxiOp::Arcsinh),
                    LoweredOp::Arccosh(_) => output.push(OxiOp::Arccosh),
                    LoweredOp::Arctanh(_) => output.push(OxiOp::Arctanh),
                    LoweredOp::Sqrt(_) => output.push(OxiOp::Sqrt),
                    LoweredOp::Abs(_) => output.push(OxiOp::Abs),
                }
            } else {
                // Pre-order: schedule post-visit + push children
                match op {
                    LoweredOp::Const(_) | LoweredOp::Var(_) => {
                        // Leaf — emit immediately on next pop.
                        work.push((op, true));
                    }
                    LoweredOp::Add(a, b)
                    | LoweredOp::Sub(a, b)
                    | LoweredOp::Mul(a, b)
                    | LoweredOp::Div(a, b)
                    | LoweredOp::Pow(a, b) => {
                        work.push((op, true));
                        // Push right first so it pops second; left pops first.
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
                        work.push((op, true));
                        work.push((c, false));
                    }
                }
            }
        }
        output
    }

    /// O(N) iterative variable count (max var index + 1, or 0 if no vars).
    pub fn count_vars(&self) -> usize {
        let mut max_idx: Option<usize> = None;
        let mut work: Vec<&LoweredOp> = vec![self];
        while let Some(op) = work.pop() {
            match op {
                LoweredOp::Const(_) => {}
                LoweredOp::Var(i) => {
                    max_idx = Some(max_idx.map_or(*i, |m| m.max(*i)));
                }
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
        max_idx.map(|i| i + 1).unwrap_or(0)
    }

    /// Structural u128 hash. Iterative; produces same value across processes
    /// (uses two-seed `ahash` from `crate::eml::hash`).
    ///
    /// The hash is computed from the post-order tape so two structurally
    /// identical `LoweredOp` trees always hash to the same u128.
    pub fn structural_hash(&self) -> u128 {
        let ops = self.to_oxi_ops();
        // Encode ops as a deterministic byte sequence.
        let mut tape: Vec<u8> = Vec::with_capacity(ops.len() * 9);
        for op in &ops {
            match op {
                OxiOp::Const(c) => {
                    tape.push(0u8);
                    tape.extend_from_slice(&c.to_bits().to_le_bytes());
                }
                OxiOp::Var(i) => {
                    tape.push(1u8);
                    tape.extend_from_slice(&(*i as u64).to_le_bytes());
                }
                OxiOp::Add => tape.push(10u8),
                OxiOp::Sub => tape.push(11u8),
                OxiOp::Mul => tape.push(12u8),
                OxiOp::Div => tape.push(13u8),
                OxiOp::Pow => tape.push(14u8),
                OxiOp::Neg => tape.push(15u8),
                OxiOp::Exp => tape.push(20u8),
                OxiOp::Ln => tape.push(21u8),
                OxiOp::Sin => tape.push(30u8),
                OxiOp::Cos => tape.push(31u8),
                OxiOp::Tan => tape.push(32u8),
                OxiOp::Sinh => tape.push(33u8),
                OxiOp::Cosh => tape.push(34u8),
                OxiOp::Tanh => tape.push(35u8),
                OxiOp::Arcsin => tape.push(40u8),
                OxiOp::Arccos => tape.push(41u8),
                OxiOp::Arctan => tape.push(42u8),
                OxiOp::Arcsinh => tape.push(43u8),
                OxiOp::Arccosh => tape.push(44u8),
                OxiOp::Arctanh => tape.push(45u8),
                OxiOp::Sqrt => tape.push(50u8),
                OxiOp::Abs => tape.push(51u8),
            }
        }
        hash_u128(tape.as_slice())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn const_leaf() {
        let op = LoweredOp::Const(2.71);
        let tape = op.to_oxi_ops();
        assert_eq!(tape, vec![OxiOp::Const(2.71)]);
        assert_eq!(op.count_vars(), 0);
    }

    #[test]
    fn var_leaf() {
        let op = LoweredOp::Var(5);
        assert_eq!(op.count_vars(), 6);
        let tape = op.to_oxi_ops();
        assert_eq!(tape, vec![OxiOp::Var(5)]);
    }

    #[test]
    fn binary_op_post_order() {
        // Add(Var(0), Const(1.0)) → [Var(0), Const(1.0), Add]
        let op = LoweredOp::Add(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Const(1.0)));
        let tape = op.to_oxi_ops();
        assert_eq!(tape, vec![OxiOp::Var(0), OxiOp::Const(1.0), OxiOp::Add]);
    }

    #[test]
    fn unary_op_post_order() {
        // Sin(Var(0)) → [Var(0), Sin]
        let op = LoweredOp::Sin(Box::new(LoweredOp::Var(0)));
        assert_eq!(op.to_oxi_ops(), vec![OxiOp::Var(0), OxiOp::Sin]);
    }

    #[test]
    fn nested_post_order() {
        // Mul(Add(Var(0), Const(1)), Var(0)) → [Var(0), Const(1), Add, Var(0), Mul]
        let op = LoweredOp::Mul(
            Box::new(LoweredOp::Add(
                Box::new(LoweredOp::Var(0)),
                Box::new(LoweredOp::Const(1.0)),
            )),
            Box::new(LoweredOp::Var(0)),
        );
        let tape = op.to_oxi_ops();
        assert_eq!(tape.len(), 5);
        assert_eq!(tape[0], OxiOp::Var(0));
        assert_eq!(tape[1], OxiOp::Const(1.0));
        assert_eq!(tape[2], OxiOp::Add);
        assert_eq!(tape[3], OxiOp::Var(0));
        assert_eq!(tape[4], OxiOp::Mul);
    }

    #[test]
    fn sqrt_and_abs_native() {
        // Sqrt and Abs flatten to single-byte tape entries.
        let s = LoweredOp::Sqrt(Box::new(LoweredOp::Var(0)));
        assert_eq!(s.to_oxi_ops(), vec![OxiOp::Var(0), OxiOp::Sqrt]);
        let a = LoweredOp::Abs(Box::new(LoweredOp::Var(1)));
        assert_eq!(a.to_oxi_ops(), vec![OxiOp::Var(1), OxiOp::Abs]);
    }

    #[test]
    fn count_vars_max_index_plus_one() {
        // Mix of Var(2) and Var(7) → count = 8.
        let op = LoweredOp::Add(Box::new(LoweredOp::Var(2)), Box::new(LoweredOp::Var(7)));
        assert_eq!(op.count_vars(), 8);
    }

    #[test]
    fn count_vars_no_vars() {
        let op = LoweredOp::Add(
            Box::new(LoweredOp::Const(1.0)),
            Box::new(LoweredOp::Const(2.0)),
        );
        assert_eq!(op.count_vars(), 0);
    }

    #[test]
    fn structural_hash_deterministic() {
        let a = LoweredOp::Add(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Const(1.0)));
        let b = LoweredOp::Add(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Const(1.0)));
        assert_eq!(a.structural_hash(), b.structural_hash());
    }

    #[test]
    fn structural_hash_distinguishes_shape() {
        let a = LoweredOp::Add(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Const(1.0)));
        let b = LoweredOp::Sub(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Const(1.0)));
        assert_ne!(a.structural_hash(), b.structural_hash());
    }

    #[test]
    fn structural_hash_distinguishes_var_indices() {
        let a = LoweredOp::Var(0);
        let b = LoweredOp::Var(1);
        assert_ne!(a.structural_hash(), b.structural_hash());
    }

    #[test]
    fn structural_hash_distinguishes_const_values() {
        let a = LoweredOp::Const(1.0);
        let b = LoweredOp::Const(2.0);
        assert_ne!(a.structural_hash(), b.structural_hash());
    }

    #[test]
    fn deep_left_chain_no_overflow() {
        // Build Add(Add(Add(... Var(0), Const(1)), Const(1)), Const(1))
        let mut op = LoweredOp::Var(0);
        for _ in 0..10_000 {
            op = LoweredOp::Add(Box::new(op), Box::new(LoweredOp::Const(1.0)));
        }
        // Must not overflow the OS stack.
        // Structure: 1 Var leaf + 10000 Const leaves + 10000 Add ops = 20001 tape entries.
        let tape = op.to_oxi_ops();
        assert_eq!(tape.len(), 20_001);
        assert_eq!(op.count_vars(), 1);
        let _ = op.structural_hash();
    }

    #[test]
    fn deep_right_chain_no_overflow() {
        // Mirror of the left-chain stress.
        let mut op = LoweredOp::Var(0);
        for _ in 0..10_000 {
            op = LoweredOp::Add(Box::new(LoweredOp::Const(1.0)), Box::new(op));
        }
        let tape = op.to_oxi_ops();
        assert_eq!(tape.len(), 20_001);
    }
}
