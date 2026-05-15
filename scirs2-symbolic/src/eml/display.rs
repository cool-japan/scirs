//! Display impls for [`EmlTree`] / [`EmlNode`] / [`LoweredOp`] plus a LaTeX
//! export for [`LoweredOp`].
//!
//! All renderers use the iterative `to_oxi_ops()` post-order tape + an
//! explicit value stack, so deeply nested expressions never overflow the OS
//! stack.
//!
//! # Operator representation
//!
//! [`EmlTree`] / [`EmlNode`] are rendered in the canonical `eml(left, right)`
//! form (matches [`crate::eml::parser::to_compact_string`] exactly).
//!
//! [`LoweredOp`] is rendered with **always-parenthesised** binary operators
//! to side-step precedence reasoning: `Mul(Add(a, b), c)` → `(a + b) * c`,
//! never `a + b * c`. Unary functions get function-call notation
//! (`sin(x)`, `exp(x + 1)` etc.).
//!
//! # LaTeX
//!
//! [`to_latex`] recognises π and e as exact constants (within `1e-12`) and
//! emits `\pi` / `e` respectively. All other constants render as decimal
//! (integers without a trailing `.0`).

use crate::eml::op::{LoweredOp, OxiOp};
use crate::eml::parser::to_compact_string;
use crate::eml::tree::{EmlNode, EmlTree};
use std::fmt;
use std::sync::Arc;

// ============================================================================
// Display impls — EmlTree / EmlNode
// ============================================================================

impl fmt::Display for EmlTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&to_compact_string(self))
    }
}

impl fmt::Display for EmlNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Wrap in a tree so we can reuse the iterative formatter. Allocates
        // an `Arc<EmlNode>` per call — acceptable for human-readable display.
        let tree = EmlTree::from_node(Arc::new(self.clone()));
        f.write_str(&to_compact_string(&tree))
    }
}

// ============================================================================
// Display impl — LoweredOp (infix, always-parenthesised binary ops)
// ============================================================================

impl fmt::Display for LoweredOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&render_infix(self))
    }
}

/// Render a [`LoweredOp`] as a parenthesised infix string.
///
/// Iterative: walks the post-order tape and uses a string stack to compose
/// each subexpression. Binary operators are always wrapped in parentheses;
/// unary operators use function-call notation.
fn render_infix(op: &LoweredOp) -> String {
    let ops = op.to_oxi_ops();
    let mut stack: Vec<String> = Vec::with_capacity(ops.len());
    for o in &ops {
        match o {
            OxiOp::Const(c) => stack.push(format_const_plain(*c)),
            OxiOp::Var(i) => stack.push(format!("x{i}")),
            OxiOp::Add => apply_binary(&mut stack, "+"),
            OxiOp::Sub => apply_binary(&mut stack, "-"),
            OxiOp::Mul => apply_binary(&mut stack, "*"),
            OxiOp::Div => apply_binary(&mut stack, "/"),
            OxiOp::Pow => apply_binary(&mut stack, "^"),
            OxiOp::Neg => apply_unary_prefix(&mut stack, "-"),
            OxiOp::Exp => apply_unary_call(&mut stack, "exp"),
            OxiOp::Ln => apply_unary_call(&mut stack, "ln"),
            OxiOp::Sin => apply_unary_call(&mut stack, "sin"),
            OxiOp::Cos => apply_unary_call(&mut stack, "cos"),
            OxiOp::Tan => apply_unary_call(&mut stack, "tan"),
            OxiOp::Sinh => apply_unary_call(&mut stack, "sinh"),
            OxiOp::Cosh => apply_unary_call(&mut stack, "cosh"),
            OxiOp::Tanh => apply_unary_call(&mut stack, "tanh"),
            OxiOp::Arcsin => apply_unary_call(&mut stack, "arcsin"),
            OxiOp::Arccos => apply_unary_call(&mut stack, "arccos"),
            OxiOp::Arctan => apply_unary_call(&mut stack, "arctan"),
            OxiOp::Arcsinh => apply_unary_call(&mut stack, "arcsinh"),
            OxiOp::Arccosh => apply_unary_call(&mut stack, "arccosh"),
            OxiOp::Arctanh => apply_unary_call(&mut stack, "arctanh"),
            OxiOp::Sqrt => apply_unary_call(&mut stack, "sqrt"),
            OxiOp::Abs => apply_unary_abs(&mut stack),
        }
    }
    stack.pop().unwrap_or_default()
}

fn apply_binary(stack: &mut Vec<String>, sym: &str) {
    let b = stack
        .pop()
        .expect("post-order invariant: right operand on stack");
    let a = stack
        .pop()
        .expect("post-order invariant: left operand on stack");
    stack.push(format!("({a} {sym} {b})"));
}

fn apply_unary_prefix(stack: &mut Vec<String>, sym: &str) {
    let c = stack.pop().expect("post-order invariant: operand on stack");
    stack.push(format!("({sym}{c})"));
}

fn apply_unary_call(stack: &mut Vec<String>, name: &str) {
    let c = stack.pop().expect("post-order invariant: operand on stack");
    stack.push(format!("{name}({c})"));
}

fn apply_unary_abs(stack: &mut Vec<String>) {
    let c = stack.pop().expect("post-order invariant: operand on stack");
    stack.push(format!("|{c}|"));
}

// ============================================================================
// LaTeX export — LoweredOp → LaTeX string
// ============================================================================

/// Render a [`LoweredOp`] as a LaTeX math expression.
///
/// - π (within `1e-12` of [`std::f64::consts::PI`]) renders as `\pi`.
/// - e (within `1e-12` of [`std::f64::consts::E`]) renders as `e`.
/// - All other finite constants render as decimal (integers with no `.0`).
/// - Variables `Var(i)` render as `x_{i}`.
/// - Binary operators are wrapped in `\left( \right)` for visual clarity,
///   except `Div` (which uses `\frac{}{}`), `Pow` (`a^{b}`), and `Mul`
///   (which uses `\cdot` without parentheses around the product itself).
/// - Inverse trig / hyperbolic functions without a single-token LaTeX
///   command (e.g. `arcsinh`) use `\operatorname{...}`.
///
/// # Examples
///
/// ```
/// use scirs2_symbolic::eml::display::to_latex;
/// use scirs2_symbolic::eml::op::LoweredOp;
///
/// let pi_op = LoweredOp::Const(std::f64::consts::PI);
/// assert_eq!(to_latex(&pi_op), "\\pi");
/// ```
pub fn to_latex(op: &LoweredOp) -> String {
    let ops = op.to_oxi_ops();
    let mut stack: Vec<String> = Vec::with_capacity(ops.len());
    for o in &ops {
        match o {
            OxiOp::Const(c) => stack.push(format_latex_const(*c)),
            OxiOp::Var(i) => stack.push(format!("x_{{{i}}}")),
            OxiOp::Add => latex_binary_paren(&mut stack, "+"),
            OxiOp::Sub => latex_binary_paren(&mut stack, "-"),
            OxiOp::Mul => latex_mul(&mut stack),
            OxiOp::Div => latex_div(&mut stack),
            OxiOp::Pow => latex_pow(&mut stack),
            OxiOp::Neg => latex_neg(&mut stack),
            OxiOp::Exp => latex_exp(&mut stack),
            OxiOp::Ln => latex_func(&mut stack, "\\ln"),
            OxiOp::Sin => latex_func(&mut stack, "\\sin"),
            OxiOp::Cos => latex_func(&mut stack, "\\cos"),
            OxiOp::Tan => latex_func(&mut stack, "\\tan"),
            OxiOp::Sinh => latex_func(&mut stack, "\\sinh"),
            OxiOp::Cosh => latex_func(&mut stack, "\\cosh"),
            OxiOp::Tanh => latex_func(&mut stack, "\\tanh"),
            OxiOp::Arcsin => latex_func(&mut stack, "\\arcsin"),
            OxiOp::Arccos => latex_func(&mut stack, "\\arccos"),
            OxiOp::Arctan => latex_func(&mut stack, "\\arctan"),
            OxiOp::Arcsinh => latex_func_op(&mut stack, "arcsinh"),
            OxiOp::Arccosh => latex_func_op(&mut stack, "arccosh"),
            OxiOp::Arctanh => latex_func_op(&mut stack, "arctanh"),
            OxiOp::Sqrt => latex_sqrt(&mut stack),
            OxiOp::Abs => latex_abs(&mut stack),
        }
    }
    stack.pop().unwrap_or_default()
}

fn latex_binary_paren(stack: &mut Vec<String>, sym: &str) {
    let b = stack.pop().expect("post-order invariant: right operand");
    let a = stack.pop().expect("post-order invariant: left operand");
    stack.push(format!("\\left({a} {sym} {b}\\right)"));
}

fn latex_mul(stack: &mut Vec<String>) {
    let b = stack.pop().expect("post-order invariant: right operand");
    let a = stack.pop().expect("post-order invariant: left operand");
    stack.push(format!("{a} \\cdot {b}"));
}

fn latex_div(stack: &mut Vec<String>) {
    let b = stack.pop().expect("post-order invariant: right operand");
    let a = stack.pop().expect("post-order invariant: left operand");
    stack.push(format!("\\frac{{{a}}}{{{b}}}"));
}

fn latex_pow(stack: &mut Vec<String>) {
    let b = stack.pop().expect("post-order invariant: right operand");
    let a = stack.pop().expect("post-order invariant: left operand");
    stack.push(format!("{a}^{{{b}}}"));
}

fn latex_neg(stack: &mut Vec<String>) {
    let c = stack.pop().expect("post-order invariant: operand");
    stack.push(format!("-{c}"));
}

fn latex_exp(stack: &mut Vec<String>) {
    let c = stack.pop().expect("post-order invariant: operand");
    stack.push(format!("e^{{{c}}}"));
}

fn latex_func(stack: &mut Vec<String>, cmd: &str) {
    let c = stack.pop().expect("post-order invariant: operand");
    stack.push(format!("{cmd}\\left({c}\\right)"));
}

fn latex_func_op(stack: &mut Vec<String>, name: &str) {
    let c = stack.pop().expect("post-order invariant: operand");
    stack.push(format!("\\operatorname{{{name}}}\\left({c}\\right)"));
}

fn latex_sqrt(stack: &mut Vec<String>) {
    let c = stack.pop().expect("post-order invariant: operand");
    stack.push(format!("\\sqrt{{{c}}}"));
}

fn latex_abs(stack: &mut Vec<String>) {
    let c = stack.pop().expect("post-order invariant: operand");
    stack.push(format!("\\left|{c}\\right|"));
}

// ============================================================================
// Constant formatting helpers
// ============================================================================

/// Render a finite constant for plain (non-LaTeX) display.
///
/// Integers print without a `.0` suffix. Non-integers use `f64`'s default
/// `Display` (no scientific notation for typical user values).
fn format_const_plain(c: f64) -> String {
    if c.is_finite() && c.fract() == 0.0 && c.abs() < 1e15 {
        format!("{}", c as i64)
    } else {
        format!("{c}")
    }
}

/// Render a constant for LaTeX, matching π and e exactly within `1e-12`.
fn format_latex_const(c: f64) -> String {
    if (c - std::f64::consts::PI).abs() < 1e-12 {
        "\\pi".to_string()
    } else if (c - std::f64::consts::E).abs() < 1e-12 {
        "e".to_string()
    } else {
        format_const_plain(c)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eml::tree::EmlTree;

    // ----------------------------------------------------------------
    // Display: EmlTree / EmlNode
    // ----------------------------------------------------------------

    #[test]
    fn display_eml_tree_one() {
        assert_eq!(format!("{}", EmlTree::one()), "1");
    }

    #[test]
    fn display_eml_tree_var() {
        assert_eq!(format!("{}", EmlTree::var(3)), "x3");
    }

    #[test]
    fn display_eml_tree_compound() {
        let t = EmlTree::eml(&EmlTree::var(0), &EmlTree::one());
        assert_eq!(format!("{t}"), "eml(x0, 1)");
    }

    #[test]
    fn display_eml_node() {
        let n = EmlNode::Var(7);
        assert_eq!(format!("{n}"), "x7");
    }

    // ----------------------------------------------------------------
    // Display: LoweredOp (infix, parenthesised)
    // ----------------------------------------------------------------

    #[test]
    fn display_const_integer() {
        assert_eq!(format!("{}", LoweredOp::Const(2.0)), "2");
    }

    #[test]
    fn display_const_decimal() {
        assert_eq!(format!("{}", LoweredOp::Const(2.5)), "2.5");
    }

    #[test]
    fn display_var() {
        assert_eq!(format!("{}", LoweredOp::Var(0)), "x0");
    }

    #[test]
    fn display_add() {
        let op = LoweredOp::Add(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Const(1.0)));
        assert_eq!(format!("{op}"), "(x0 + 1)");
    }

    #[test]
    fn display_nested_mul_add() {
        // (x0 + 1) * x0
        let op = LoweredOp::Mul(
            Box::new(LoweredOp::Add(
                Box::new(LoweredOp::Var(0)),
                Box::new(LoweredOp::Const(1.0)),
            )),
            Box::new(LoweredOp::Var(0)),
        );
        assert_eq!(format!("{op}"), "((x0 + 1) * x0)");
    }

    #[test]
    fn display_unary_call() {
        let op = LoweredOp::Sin(Box::new(LoweredOp::Var(0)));
        assert_eq!(format!("{op}"), "sin(x0)");
    }

    #[test]
    fn display_neg() {
        let op = LoweredOp::Neg(Box::new(LoweredOp::Var(0)));
        assert_eq!(format!("{op}"), "(-x0)");
    }

    #[test]
    fn display_abs() {
        let op = LoweredOp::Abs(Box::new(LoweredOp::Var(0)));
        assert_eq!(format!("{op}"), "|x0|");
    }

    // ----------------------------------------------------------------
    // LaTeX
    // ----------------------------------------------------------------

    #[test]
    fn latex_const_pi() {
        let op = LoweredOp::Const(std::f64::consts::PI);
        assert_eq!(to_latex(&op), "\\pi");
    }

    #[test]
    fn latex_const_e() {
        let op = LoweredOp::Const(std::f64::consts::E);
        assert_eq!(to_latex(&op), "e");
    }

    #[test]
    fn latex_const_integer() {
        let op = LoweredOp::Const(2.0);
        assert_eq!(to_latex(&op), "2");
    }

    #[test]
    fn latex_var() {
        let op = LoweredOp::Var(0);
        assert_eq!(to_latex(&op), "x_{0}");
    }

    #[test]
    fn latex_sin() {
        let op = LoweredOp::Sin(Box::new(LoweredOp::Var(0)));
        assert_eq!(to_latex(&op), "\\sin\\left(x_{0}\\right)");
    }

    #[test]
    fn latex_div() {
        let op = LoweredOp::Div(Box::new(LoweredOp::Const(1.0)), Box::new(LoweredOp::Var(0)));
        assert_eq!(to_latex(&op), "\\frac{1}{x_{0}}");
    }

    #[test]
    fn latex_pow() {
        let op = LoweredOp::Pow(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Const(2.0)));
        assert_eq!(to_latex(&op), "x_{0}^{2}");
    }

    #[test]
    fn latex_mul() {
        let op = LoweredOp::Mul(Box::new(LoweredOp::Const(2.0)), Box::new(LoweredOp::Var(0)));
        assert_eq!(to_latex(&op), "2 \\cdot x_{0}");
    }

    #[test]
    fn latex_exp() {
        let op = LoweredOp::Exp(Box::new(LoweredOp::Var(0)));
        assert_eq!(to_latex(&op), "e^{x_{0}}");
    }

    #[test]
    fn latex_sqrt() {
        let op = LoweredOp::Sqrt(Box::new(LoweredOp::Var(0)));
        assert_eq!(to_latex(&op), "\\sqrt{x_{0}}");
    }

    #[test]
    fn latex_abs() {
        let op = LoweredOp::Abs(Box::new(LoweredOp::Var(0)));
        assert_eq!(to_latex(&op), "\\left|x_{0}\\right|");
    }

    #[test]
    fn latex_arcsinh_uses_operatorname() {
        let op = LoweredOp::Arcsinh(Box::new(LoweredOp::Var(0)));
        assert_eq!(to_latex(&op), "\\operatorname{arcsinh}\\left(x_{0}\\right)");
    }

    #[test]
    fn latex_neg() {
        let op = LoweredOp::Neg(Box::new(LoweredOp::Var(0)));
        assert_eq!(to_latex(&op), "-x_{0}");
    }

    #[test]
    fn latex_nested() {
        // sin(x0 + 1)
        let op = LoweredOp::Sin(Box::new(LoweredOp::Add(
            Box::new(LoweredOp::Var(0)),
            Box::new(LoweredOp::Const(1.0)),
        )));
        assert_eq!(
            to_latex(&op),
            "\\sin\\left(\\left(x_{0} + 1\\right)\\right)"
        );
    }

    // ----------------------------------------------------------------
    // Stack-safety smoke (deep ops must not overflow)
    // ----------------------------------------------------------------

    #[test]
    fn display_deep_chain_no_overflow() {
        let mut op = LoweredOp::Var(0);
        for _ in 0..1000 {
            op = LoweredOp::Add(Box::new(op), Box::new(LoweredOp::Const(1.0)));
        }
        let s = format!("{op}");
        assert!(s.starts_with('('));
        assert!(s.ends_with(')'));
    }

    #[test]
    fn latex_deep_chain_no_overflow() {
        let mut op = LoweredOp::Var(0);
        for _ in 0..1000 {
            op = LoweredOp::Add(Box::new(op), Box::new(LoweredOp::Const(1.0)));
        }
        let s = to_latex(&op);
        assert!(s.contains("\\left("));
    }
}
