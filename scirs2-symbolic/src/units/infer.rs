//! Dimensional inference for [`LoweredOp`] expressions.
//!
//! [`infer_dimension`] walks a symbolic expression bottom-up and infers the
//! physical dimension of the result, given the dimensions of each `Var(i)`.
//!
//! # Rules
//!
//! | Operation | Requirement | Result dimension |
//! |---|---|---|
//! | `Const` | none | dimensionless |
//! | `Var(i)` | `i < var_dims.len()` | `var_dims[i]` |
//! | `Add` / `Sub` | operands must have equal dimensions | same as operands |
//! | `Mul` | none | product of operand dimensions |
//! | `Div` | none | quotient of operand dimensions |
//! | `Neg` / `Abs` | none | same as child |
//! | `Pow(_, Const(n))` with integer `n` | base may have any dimension | `base_dim^n` |
//! | `Pow(_, _)` general | base must be dimensionless | dimensionless |
//! | `Sin`, `Cos`, `Tan`, `Sinh`, `Cosh`, `Tanh`, `Arcsin`, `Arccos`, `Arctan`, `Arcsinh`, `Arccosh`, `Arctanh`, `Exp`, `Ln` | arg must be dimensionless | dimensionless |
//! | `Sqrt` | all arg exponents must be even | `arg_dim^(1/2)` |

use crate::eml::op::LoweredOp;
use crate::units::aware::UnitError;
use crate::units::dimension::Dimension;

/// Infer the physical [`Dimension`] of a [`LoweredOp`] expression.
///
/// `var_dims[i]` is the dimension of `Var(i)`. If `op` references a `Var(i)`
/// with `i >= var_dims.len()`, returns [`UnitError::VarIndexOutOfRange`].
///
/// All reasoning is purely structural; no numerical evaluation is performed.
///
/// # Algorithm
///
/// Iterative post-order walk with a work-stack and a result-stack — no
/// recursion, bounded stack usage.
///
/// # Errors
///
/// - [`UnitError::VarIndexOutOfRange`] — a `Var(i)` index has no corresponding entry in `var_dims`.
/// - [`UnitError::Mismatch`] — operands of `Add` or `Sub` have different dimensions.
/// - [`UnitError::NonDimensionlessTranscendental`] — a trig/exp/ln function received a dimensional argument.
/// - [`UnitError::NonIntegerPower`] — a non-integer-constant exponent was applied to a dimensional base.
/// - [`UnitError::NonSquareSqrt`] — `Sqrt` applied to a dimension with odd exponents.
/// - [`UnitError::Internal`] — indicates a bug in the traversal logic (should never occur).
pub fn infer_dimension(op: &LoweredOp, var_dims: &[Dimension]) -> Result<Dimension, UnitError> {
    // The work-stack holds either an Open (schedule children first) or
    // Close (combine already-computed children) entry.
    enum Work<'a> {
        /// Schedule this node for processing: push children, then push Close(node).
        Open(&'a LoweredOp),
        /// Combine the top `n_children` entries of `result_stack` for this node.
        Close(&'a LoweredOp, usize),
    }

    let mut work_stack: Vec<Work<'_>> = vec![Work::Open(op)];
    let mut result_stack: Vec<Dimension> = Vec::new();

    while let Some(item) = work_stack.pop() {
        match item {
            Work::Open(node) => match node {
                // Leaves — push result directly, no children to schedule.
                LoweredOp::Const(_) => {
                    result_stack.push(Dimension::dimensionless());
                }
                LoweredOp::Var(i) => {
                    let dim = var_dims
                        .get(*i)
                        .copied()
                        .ok_or(UnitError::VarIndexOutOfRange(*i))?;
                    result_stack.push(dim);
                }

                // Unary operators — 1 child.
                LoweredOp::Neg(c) | LoweredOp::Abs(c) => {
                    work_stack.push(Work::Close(node, 1));
                    work_stack.push(Work::Open(c));
                }
                LoweredOp::Sin(c)
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
                | LoweredOp::Exp(c)
                | LoweredOp::Ln(c)
                | LoweredOp::Sqrt(c) => {
                    work_stack.push(Work::Close(node, 1));
                    work_stack.push(Work::Open(c));
                }

                // Binary operators — 2 children.
                // Push right first so it is processed second; left is processed
                // first and lands at the bottom of the pair in result_stack.
                LoweredOp::Add(l, r)
                | LoweredOp::Sub(l, r)
                | LoweredOp::Mul(l, r)
                | LoweredOp::Div(l, r)
                | LoweredOp::Pow(l, r) => {
                    work_stack.push(Work::Close(node, 2));
                    work_stack.push(Work::Open(r));
                    work_stack.push(Work::Open(l));
                }
            },

            Work::Close(node, n_children) => {
                // Pop `n_children` from result_stack.
                // Stack is LIFO: the last child pushed (= rightmost) was processed
                // last, so its result sits on top. For n=2: top = right, next = left.
                match node {
                    // --- Unary: passthrough ---
                    LoweredOp::Neg(_) | LoweredOp::Abs(_) => {
                        debug_assert_eq!(n_children, 1);
                        // child_dim is already at the top; nothing to change.
                    }

                    // --- Additive: operands must match ---
                    LoweredOp::Add(_, _) | LoweredOp::Sub(_, _) => {
                        debug_assert_eq!(n_children, 2);
                        let r_dim = result_stack.pop().ok_or(UnitError::Internal(
                            "result_stack underflow at Add/Sub right",
                        ))?;
                        let l_dim = result_stack.pop().ok_or(UnitError::Internal(
                            "result_stack underflow at Add/Sub left",
                        ))?;
                        if l_dim != r_dim {
                            return Err(UnitError::Mismatch(l_dim, r_dim));
                        }
                        result_stack.push(l_dim);
                    }

                    // --- Mul: product of dimensions ---
                    LoweredOp::Mul(_, _) => {
                        debug_assert_eq!(n_children, 2);
                        let r_dim = result_stack
                            .pop()
                            .ok_or(UnitError::Internal("result_stack underflow at Mul right"))?;
                        let l_dim = result_stack
                            .pop()
                            .ok_or(UnitError::Internal("result_stack underflow at Mul left"))?;
                        result_stack.push(l_dim.product(&r_dim));
                    }

                    // --- Div: quotient of dimensions ---
                    LoweredOp::Div(_, _) => {
                        debug_assert_eq!(n_children, 2);
                        let r_dim = result_stack
                            .pop()
                            .ok_or(UnitError::Internal("result_stack underflow at Div right"))?;
                        let l_dim = result_stack
                            .pop()
                            .ok_or(UnitError::Internal("result_stack underflow at Div left"))?;
                        result_stack.push(l_dim.quotient(&r_dim));
                    }

                    // --- Pow: depends on base dimension + exponent shape ---
                    LoweredOp::Pow(_, exp_op) => {
                        debug_assert_eq!(n_children, 2);
                        // Stack: top = exp_dim, next = base_dim
                        // (right child was processed last → on top)
                        let _exp_dim = result_stack
                            .pop()
                            .ok_or(UnitError::Internal("result_stack underflow at Pow exp"))?;
                        let base_dim = result_stack
                            .pop()
                            .ok_or(UnitError::Internal("result_stack underflow at Pow base"))?;

                        if base_dim.is_dimensionless() {
                            // Dimensionless^anything = dimensionless.
                            result_stack.push(Dimension::dimensionless());
                        } else if let LoweredOp::Const(v) = exp_op.as_ref() {
                            // Integer-constant exponent on a dimensional base.
                            if v.fract() == 0.0 {
                                let n = *v as i32;
                                result_stack.push(base_dim.power_int(n));
                            } else {
                                return Err(UnitError::NonIntegerPower(base_dim, *v));
                            }
                        } else {
                            // Non-constant exponent on a dimensional base.
                            return Err(UnitError::NonIntegerPower(base_dim, f64::NAN));
                        }
                    }

                    // --- Transcendentals: arg must be dimensionless → result dimensionless ---
                    LoweredOp::Sin(_) => {
                        debug_assert_eq!(n_children, 1);
                        let child = result_stack
                            .pop()
                            .ok_or(UnitError::Internal("result_stack underflow at Sin"))?;
                        if !child.is_dimensionless() {
                            return Err(UnitError::NonDimensionlessTranscendental {
                                op: "sin",
                                dim: child,
                            });
                        }
                        result_stack.push(Dimension::dimensionless());
                    }
                    LoweredOp::Cos(_) => {
                        debug_assert_eq!(n_children, 1);
                        let child = result_stack
                            .pop()
                            .ok_or(UnitError::Internal("result_stack underflow at Cos"))?;
                        if !child.is_dimensionless() {
                            return Err(UnitError::NonDimensionlessTranscendental {
                                op: "cos",
                                dim: child,
                            });
                        }
                        result_stack.push(Dimension::dimensionless());
                    }
                    LoweredOp::Tan(_) => {
                        debug_assert_eq!(n_children, 1);
                        let child = result_stack
                            .pop()
                            .ok_or(UnitError::Internal("result_stack underflow at Tan"))?;
                        if !child.is_dimensionless() {
                            return Err(UnitError::NonDimensionlessTranscendental {
                                op: "tan",
                                dim: child,
                            });
                        }
                        result_stack.push(Dimension::dimensionless());
                    }
                    LoweredOp::Sinh(_) => {
                        debug_assert_eq!(n_children, 1);
                        let child = result_stack
                            .pop()
                            .ok_or(UnitError::Internal("result_stack underflow at Sinh"))?;
                        if !child.is_dimensionless() {
                            return Err(UnitError::NonDimensionlessTranscendental {
                                op: "sinh",
                                dim: child,
                            });
                        }
                        result_stack.push(Dimension::dimensionless());
                    }
                    LoweredOp::Cosh(_) => {
                        debug_assert_eq!(n_children, 1);
                        let child = result_stack
                            .pop()
                            .ok_or(UnitError::Internal("result_stack underflow at Cosh"))?;
                        if !child.is_dimensionless() {
                            return Err(UnitError::NonDimensionlessTranscendental {
                                op: "cosh",
                                dim: child,
                            });
                        }
                        result_stack.push(Dimension::dimensionless());
                    }
                    LoweredOp::Tanh(_) => {
                        debug_assert_eq!(n_children, 1);
                        let child = result_stack
                            .pop()
                            .ok_or(UnitError::Internal("result_stack underflow at Tanh"))?;
                        if !child.is_dimensionless() {
                            return Err(UnitError::NonDimensionlessTranscendental {
                                op: "tanh",
                                dim: child,
                            });
                        }
                        result_stack.push(Dimension::dimensionless());
                    }
                    LoweredOp::Arcsin(_) => {
                        debug_assert_eq!(n_children, 1);
                        let child = result_stack
                            .pop()
                            .ok_or(UnitError::Internal("result_stack underflow at Arcsin"))?;
                        if !child.is_dimensionless() {
                            return Err(UnitError::NonDimensionlessTranscendental {
                                op: "arcsin",
                                dim: child,
                            });
                        }
                        result_stack.push(Dimension::dimensionless());
                    }
                    LoweredOp::Arccos(_) => {
                        debug_assert_eq!(n_children, 1);
                        let child = result_stack
                            .pop()
                            .ok_or(UnitError::Internal("result_stack underflow at Arccos"))?;
                        if !child.is_dimensionless() {
                            return Err(UnitError::NonDimensionlessTranscendental {
                                op: "arccos",
                                dim: child,
                            });
                        }
                        result_stack.push(Dimension::dimensionless());
                    }
                    LoweredOp::Arctan(_) => {
                        debug_assert_eq!(n_children, 1);
                        let child = result_stack
                            .pop()
                            .ok_or(UnitError::Internal("result_stack underflow at Arctan"))?;
                        if !child.is_dimensionless() {
                            return Err(UnitError::NonDimensionlessTranscendental {
                                op: "arctan",
                                dim: child,
                            });
                        }
                        result_stack.push(Dimension::dimensionless());
                    }
                    LoweredOp::Arcsinh(_) => {
                        debug_assert_eq!(n_children, 1);
                        let child = result_stack
                            .pop()
                            .ok_or(UnitError::Internal("result_stack underflow at Arcsinh"))?;
                        if !child.is_dimensionless() {
                            return Err(UnitError::NonDimensionlessTranscendental {
                                op: "arcsinh",
                                dim: child,
                            });
                        }
                        result_stack.push(Dimension::dimensionless());
                    }
                    LoweredOp::Arccosh(_) => {
                        debug_assert_eq!(n_children, 1);
                        let child = result_stack
                            .pop()
                            .ok_or(UnitError::Internal("result_stack underflow at Arccosh"))?;
                        if !child.is_dimensionless() {
                            return Err(UnitError::NonDimensionlessTranscendental {
                                op: "arccosh",
                                dim: child,
                            });
                        }
                        result_stack.push(Dimension::dimensionless());
                    }
                    LoweredOp::Arctanh(_) => {
                        debug_assert_eq!(n_children, 1);
                        let child = result_stack
                            .pop()
                            .ok_or(UnitError::Internal("result_stack underflow at Arctanh"))?;
                        if !child.is_dimensionless() {
                            return Err(UnitError::NonDimensionlessTranscendental {
                                op: "arctanh",
                                dim: child,
                            });
                        }
                        result_stack.push(Dimension::dimensionless());
                    }
                    LoweredOp::Exp(_) => {
                        debug_assert_eq!(n_children, 1);
                        let child = result_stack
                            .pop()
                            .ok_or(UnitError::Internal("result_stack underflow at Exp"))?;
                        if !child.is_dimensionless() {
                            return Err(UnitError::NonDimensionlessTranscendental {
                                op: "exp",
                                dim: child,
                            });
                        }
                        result_stack.push(Dimension::dimensionless());
                    }
                    LoweredOp::Ln(_) => {
                        debug_assert_eq!(n_children, 1);
                        let child = result_stack
                            .pop()
                            .ok_or(UnitError::Internal("result_stack underflow at Ln"))?;
                        if !child.is_dimensionless() {
                            return Err(UnitError::NonDimensionlessTranscendental {
                                op: "ln",
                                dim: child,
                            });
                        }
                        result_stack.push(Dimension::dimensionless());
                    }

                    // --- Sqrt: arg exponents must all be even ---
                    LoweredOp::Sqrt(_) => {
                        debug_assert_eq!(n_children, 1);
                        let child = result_stack
                            .pop()
                            .ok_or(UnitError::Internal("result_stack underflow at Sqrt"))?;
                        match child.sqrt_dim() {
                            Some(d) => result_stack.push(d),
                            None => return Err(UnitError::NonSquareSqrt(child)),
                        }
                    }

                    // Leaf nodes should never appear in a Close entry.
                    LoweredOp::Const(_) | LoweredOp::Var(_) => {
                        return Err(UnitError::Internal("leaf node in Close entry"));
                    }
                }
            }
        }
    }

    result_stack
        .pop()
        .ok_or(UnitError::Internal("result_stack empty after traversal"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eml::op::LoweredOp;
    use crate::units::dimension::Dimension;

    // Helper: dimensionless constant node.
    fn konst(v: f64) -> LoweredOp {
        LoweredOp::Const(v)
    }

    // Helper: variable node.
    fn var(i: usize) -> LoweredOp {
        LoweredOp::Var(i)
    }

    #[test]
    fn const_is_dimensionless() {
        let result = infer_dimension(&konst(5.0), &[]).expect("const infer");
        assert!(result.is_dimensionless());
    }

    #[test]
    fn var_uses_table() {
        let length = Dimension::length();
        let result = infer_dimension(&var(0), &[length]).expect("var infer");
        assert_eq!(result, Dimension::length());
    }

    #[test]
    fn var_index_out_of_range() {
        let result = infer_dimension(&var(3), &[Dimension::length()]);
        assert!(matches!(result, Err(UnitError::VarIndexOutOfRange(3))));
    }

    #[test]
    fn add_same_dimension_succeeds() {
        let op = LoweredOp::Add(Box::new(var(0)), Box::new(var(1)));
        let dims = [Dimension::length(), Dimension::length()];
        let result = infer_dimension(&op, &dims).expect("add infer");
        assert_eq!(result, Dimension::length());
    }

    #[test]
    fn add_mismatch_errs() {
        let op = LoweredOp::Add(Box::new(var(0)), Box::new(var(1)));
        let dims = [Dimension::length(), Dimension::time()];
        let result = infer_dimension(&op, &dims);
        assert!(matches!(result, Err(UnitError::Mismatch(_, _))));
    }

    #[test]
    fn sub_same_dimension_succeeds() {
        let op = LoweredOp::Sub(Box::new(var(0)), Box::new(var(1)));
        let dims = [Dimension::mass(), Dimension::mass()];
        let result = infer_dimension(&op, &dims).expect("sub infer");
        assert_eq!(result, Dimension::mass());
    }

    #[test]
    fn mul_produces_product() {
        // [m] * [s] = [m·s]
        let op = LoweredOp::Mul(Box::new(var(0)), Box::new(var(1)));
        let dims = [Dimension::length(), Dimension::time()];
        let result = infer_dimension(&op, &dims).expect("mul infer");
        assert_eq!(result, Dimension::length().product(&Dimension::time()));
    }

    #[test]
    fn div_produces_quotient() {
        // [m] / [s] = [m·s^-1] = velocity
        let op = LoweredOp::Div(Box::new(var(0)), Box::new(var(1)));
        let dims = [Dimension::length(), Dimension::time()];
        let result = infer_dimension(&op, &dims).expect("div infer");
        assert_eq!(result, Dimension::velocity());
    }

    #[test]
    fn neg_preserves_dimension() {
        let op = LoweredOp::Neg(Box::new(var(0)));
        let dims = [Dimension::mass()];
        let result = infer_dimension(&op, &dims).expect("neg infer");
        assert_eq!(result, Dimension::mass());
    }

    #[test]
    fn abs_preserves_dimension() {
        let op = LoweredOp::Abs(Box::new(var(0)));
        let dims = [Dimension::force()];
        let result = infer_dimension(&op, &dims).expect("abs infer");
        assert_eq!(result, Dimension::force());
    }

    #[test]
    fn pow_integer_const_exponent() {
        // [m]^2 = [m^2]
        let op = LoweredOp::Pow(Box::new(var(0)), Box::new(konst(2.0)));
        let dims = [Dimension::length()];
        let result = infer_dimension(&op, &dims).expect("pow int infer");
        assert_eq!(result, Dimension::length().power_int(2));
    }

    #[test]
    fn pow_fractional_const_on_dimensional_errs() {
        // [m]^0.5 → NonIntegerPower
        let op = LoweredOp::Pow(Box::new(var(0)), Box::new(konst(0.5)));
        let dims = [Dimension::length()];
        let result = infer_dimension(&op, &dims);
        assert!(matches!(result, Err(UnitError::NonIntegerPower(_, _))));
    }

    #[test]
    fn pow_dimensionless_base_any_exponent() {
        // [1]^2.5 = [1] (dimensionless base, any exponent is fine)
        let op = LoweredOp::Pow(Box::new(konst(2.0)), Box::new(konst(2.5)));
        let result = infer_dimension(&op, &[]).expect("dimensionless pow infer");
        assert!(result.is_dimensionless());
    }

    #[test]
    fn sin_dimensionless_ok() {
        let op = LoweredOp::Sin(Box::new(konst(1.0)));
        let result = infer_dimension(&op, &[]).expect("sin infer");
        assert!(result.is_dimensionless());
    }

    #[test]
    fn sin_dimensional_errs() {
        let op = LoweredOp::Sin(Box::new(var(0)));
        let dims = [Dimension::length()];
        let result = infer_dimension(&op, &dims);
        assert!(matches!(
            result,
            Err(UnitError::NonDimensionlessTranscendental { op: "sin", .. })
        ));
    }

    #[test]
    fn exp_dimensional_errs() {
        let op = LoweredOp::Exp(Box::new(var(0)));
        let dims = [Dimension::time()];
        let result = infer_dimension(&op, &dims);
        assert!(matches!(
            result,
            Err(UnitError::NonDimensionlessTranscendental { op: "exp", .. })
        ));
    }

    #[test]
    fn sqrt_of_area_is_length() {
        // sqrt(m^2) = m
        let op = LoweredOp::Sqrt(Box::new(var(0)));
        let area = Dimension::length().power_int(2);
        let result = infer_dimension(&op, &[area]).expect("sqrt area infer");
        assert_eq!(result, Dimension::length());
    }

    #[test]
    fn sqrt_of_odd_dimension_errs() {
        // sqrt(m^1) → NonSquareSqrt
        let op = LoweredOp::Sqrt(Box::new(var(0)));
        let dims = [Dimension::length()];
        let result = infer_dimension(&op, &dims);
        assert!(matches!(result, Err(UnitError::NonSquareSqrt(_))));
    }

    #[test]
    fn nested_velocity_expression() {
        // v = length / time: Div(Var(0), Var(1)) with [m, s]
        let op = LoweredOp::Div(Box::new(var(0)), Box::new(var(1)));
        let dims = [Dimension::length(), Dimension::time()];
        let result = infer_dimension(&op, &dims).expect("velocity infer");
        assert_eq!(result, Dimension::velocity());
    }

    #[test]
    fn kinetic_energy_formula() {
        // (1/2) * m * v^2 = [kg] * [m·s^-1]^2 = [kg·m^2·s^-2] = energy
        // Mul(Mul(Const(0.5), Var(0)), Pow(Var(1), Const(2)))
        let half = konst(0.5);
        let mass_node = var(0);
        let vel_sq = LoweredOp::Pow(Box::new(var(1)), Box::new(konst(2.0)));
        let m_times_half = LoweredOp::Mul(Box::new(half), Box::new(mass_node));
        let ke = LoweredOp::Mul(Box::new(m_times_half), Box::new(vel_sq));

        let dims = [Dimension::mass(), Dimension::velocity()];
        let result = infer_dimension(&ke, &dims).expect("kinetic energy infer");
        assert_eq!(result, Dimension::energy());
    }

    #[test]
    fn deep_chain_no_stack_overflow() {
        // Build Mul(Mul(Mul(... Var(0), Var(0)), Var(0)), Var(0)) depth 1000.
        let mut op = var(0);
        for _ in 0..1000 {
            op = LoweredOp::Mul(Box::new(op), Box::new(var(0)));
        }
        // All Var(0) are length; result is length^1001.
        let dims = [Dimension::length()];
        let result = infer_dimension(&op, &dims).expect("deep mul infer");
        assert_eq!(result, Dimension::length().power_int(1001));
    }

    #[test]
    fn const_times_var() {
        // Const is dimensionless; [1] * [m] = [m]
        let op = LoweredOp::Mul(Box::new(konst(2.5)), Box::new(var(0)));
        let dims = [Dimension::length()];
        let result = infer_dimension(&op, &dims).expect("const*var infer");
        assert_eq!(result, Dimension::length());
    }
}
