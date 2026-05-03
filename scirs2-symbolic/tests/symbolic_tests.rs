//! Integration tests for scirs2-symbolic.

use scirs2_symbolic::{diff, diff_n, eval, simplify, simplify_full, Expr, SymbolicError};
use std::collections::HashMap;

/// Build a variable binding map from a slice of `(&str, f64)` pairs.
fn vars<'a>(pairs: &[(&'a str, f64)]) -> HashMap<&'a str, f64> {
    pairs.iter().cloned().collect()
}

// ─── Eval tests ───────────────────────────────────────────────────────────────

#[test]
fn test_eval_const() {
    assert_eq!(eval(&Expr::from(42.0), &vars(&[])).unwrap(), 42.0);
}

#[test]
fn test_eval_var_bound() {
    let x = Expr::var("x");
    assert_eq!(eval(&x, &vars(&[("x", 5.0)])).unwrap(), 5.0);
}

#[test]
fn test_eval_var_unbound_returns_error() {
    let x = Expr::var("x");
    assert!(matches!(
        eval(&x, &vars(&[])),
        Err(SymbolicError::UnboundVariable(_))
    ));
}

#[test]
fn test_eval_add() {
    let e = Expr::from(3.0) + Expr::from(4.0);
    assert_eq!(eval(&e, &vars(&[])).unwrap(), 7.0);
}

#[test]
fn test_eval_sub() {
    let e = Expr::from(10.0) - Expr::from(3.0);
    assert_eq!(eval(&e, &vars(&[])).unwrap(), 7.0);
}

#[test]
fn test_eval_mul() {
    let e = Expr::from(3.0) * Expr::from(4.0);
    assert_eq!(eval(&e, &vars(&[])).unwrap(), 12.0);
}

#[test]
fn test_eval_div() {
    let e = Expr::from(10.0) / Expr::from(4.0);
    assert!((eval(&e, &vars(&[])).unwrap() - 2.5).abs() < 1e-12);
}

#[test]
fn test_eval_div_by_zero_errors() {
    let e = Expr::from(1.0) / Expr::from(0.0);
    assert!(matches!(
        eval(&e, &vars(&[])),
        Err(SymbolicError::DivisionByZero)
    ));
}

#[test]
fn test_eval_neg() {
    let e = -Expr::from(5.0);
    assert_eq!(eval(&e, &vars(&[])).unwrap(), -5.0);
}

#[test]
fn test_eval_pow() {
    let e = Expr::var("x").pow(Expr::from(3.0));
    assert!((eval(&e, &vars(&[("x", 2.0)])).unwrap() - 8.0).abs() < 1e-12);
}

#[test]
fn test_eval_sin_cos() {
    let x = Expr::var("x");
    let sin_val = eval(&x.clone().sin(), &vars(&[("x", 0.0)])).unwrap();
    let cos_val = eval(&x.cos(), &vars(&[("x", 0.0)])).unwrap();
    assert!(sin_val.abs() < 1e-12);
    assert!((cos_val - 1.0).abs() < 1e-12);
}

#[test]
fn test_eval_exp_ln() {
    let e = Expr::from(1.0_f64).exp();
    let v = eval(&e, &vars(&[])).unwrap();
    assert!((v - std::f64::consts::E).abs() < 1e-12);

    let ln_e = Expr::from(std::f64::consts::E).ln();
    assert!((eval(&ln_e, &vars(&[])).unwrap() - 1.0).abs() < 1e-12);
}

#[test]
fn test_eval_ln_of_nonpositive_errors() {
    let e = Expr::from(-1.0).ln();
    assert!(matches!(
        eval(&e, &vars(&[])),
        Err(SymbolicError::DomainError(_))
    ));
}

#[test]
fn test_eval_sqrt() {
    let e = Expr::from(4.0).sqrt();
    assert!((eval(&e, &vars(&[])).unwrap() - 2.0).abs() < 1e-12);
}

#[test]
fn test_eval_sqrt_negative_errors() {
    let e = Expr::from(-1.0).sqrt();
    assert!(matches!(
        eval(&e, &vars(&[])),
        Err(SymbolicError::DomainError(_))
    ));
}

#[test]
fn test_eval_abs() {
    let e = Expr::from(-3.5).abs();
    assert!((eval(&e, &vars(&[])).unwrap() - 3.5).abs() < 1e-12);
}

#[test]
fn test_eval_complex_multivar() {
    // f(x, y) = x*y + sin(x)  at (x=2, y=3) → 6 + sin(2)
    let x = Expr::var("x");
    let y = Expr::var("y");
    let f = x.clone() * y + x.clone().sin();
    let v = eval(&f, &vars(&[("x", 2.0), ("y", 3.0)])).unwrap();
    let expected = 6.0 + 2.0_f64.sin();
    assert!((v - expected).abs() < 1e-12);
}

// ─── Simplify tests ───────────────────────────────────────────────────────────

#[test]
fn test_simplify_zero_plus_x() {
    let x = Expr::var("x");
    assert_eq!(simplify(&(Expr::from(0.0) + x.clone())), x);
}

#[test]
fn test_simplify_x_plus_zero() {
    let x = Expr::var("x");
    assert_eq!(simplify(&(x.clone() + Expr::from(0.0))), x);
}

#[test]
fn test_simplify_one_times_x() {
    let x = Expr::var("x");
    assert_eq!(simplify(&(Expr::from(1.0) * x.clone())), x);
}

#[test]
fn test_simplify_x_times_one() {
    let x = Expr::var("x");
    assert_eq!(simplify(&(x.clone() * Expr::from(1.0))), x);
}

#[test]
fn test_simplify_zero_times_x() {
    let x = Expr::var("x");
    assert_eq!(simplify(&(Expr::from(0.0) * x)), Expr::from(0.0));
}

#[test]
fn test_simplify_x_minus_zero() {
    let x = Expr::var("x");
    assert_eq!(simplify(&(x.clone() - Expr::from(0.0))), x);
}

#[test]
fn test_simplify_x_minus_x() {
    let x = Expr::var("x");
    assert_eq!(simplify(&(x.clone() - x)), Expr::from(0.0));
}

#[test]
fn test_simplify_constant_fold_add() {
    assert_eq!(
        simplify(&(Expr::from(3.0) + Expr::from(4.0))),
        Expr::from(7.0)
    );
}

#[test]
fn test_simplify_constant_fold_mul() {
    assert_eq!(
        simplify(&(Expr::from(3.0) * Expr::from(4.0))),
        Expr::from(12.0)
    );
}

#[test]
fn test_simplify_neg_neg() {
    let x = Expr::var("x");
    assert_eq!(simplify(&(-(-x.clone()))), x);
}

#[test]
fn test_simplify_pow_zero_exp() {
    let x = Expr::var("x");
    assert_eq!(simplify(&x.pow(Expr::from(0.0))), Expr::from(1.0));
}

#[test]
fn test_simplify_pow_one_exp() {
    let x = Expr::var("x");
    assert_eq!(simplify(&x.clone().pow(Expr::from(1.0))), x);
}

#[test]
fn test_simplify_exp_ln_inverse() {
    let x = Expr::var("x");
    // exp(ln(x)) → x
    let e = x.clone().ln().exp();
    assert_eq!(simplify(&e), x);
}

#[test]
fn test_simplify_ln_exp_inverse() {
    let x = Expr::var("x");
    // ln(exp(x)) → x
    let e = x.clone().exp().ln();
    assert_eq!(simplify(&e), x);
}

#[test]
fn test_simplify_div_by_one() {
    let x = Expr::var("x");
    assert_eq!(simplify(&(x.clone() / Expr::from(1.0))), x);
}

#[test]
fn test_simplify_full_nested() {
    // (0 + (1 * x)) should fully simplify to x in multiple passes
    let x = Expr::var("x");
    let e = Expr::from(0.0) + (Expr::from(1.0) * x.clone());
    assert_eq!(simplify_full(&e, 5), x);
}

// ─── Diff tests ───────────────────────────────────────────────────────────────

#[test]
fn test_diff_const_is_zero() {
    let e = Expr::from(5.0);
    let de = simplify(&diff(&e, "x"));
    assert_eq!(eval(&de, &vars(&[])).unwrap(), 0.0);
}

#[test]
fn test_diff_var_wrt_self_is_one() {
    let x = Expr::var("x");
    let dx = simplify(&diff(&x, "x"));
    assert_eq!(dx, Expr::from(1.0));
}

#[test]
fn test_diff_var_wrt_other_is_zero() {
    let x = Expr::var("x");
    let dy = simplify(&diff(&x, "y"));
    assert_eq!(dy, Expr::from(0.0));
}

#[test]
fn test_diff_sum_rule() {
    // d/dx (x + x) = 2  (simplified)
    let x = Expr::var("x");
    let f = x.clone() + x.clone();
    let df = simplify(&diff(&f, "x"));
    // 1 + 1 = 2
    assert_eq!(eval(&df, &vars(&[])).unwrap(), 2.0);
}

#[test]
fn test_diff_product_rule() {
    // d/dx (x * sin(x)) = sin(x) + x*cos(x)
    let x = Expr::var("x");
    let f = x.clone() * x.clone().sin();
    let df = simplify(&diff(&f, "x"));
    let v = eval(&df, &vars(&[("x", 1.0)])).unwrap();
    let expected = 1.0_f64.sin() + 1.0 * 1.0_f64.cos();
    assert!((v - expected).abs() < 1e-10, "expected {expected}, got {v}");
}

#[test]
fn test_diff_power_rule_x_squared() {
    // d/dx x² = 2x  →  at x=3: 6
    let x = Expr::var("x");
    let f = x.clone().pow(Expr::from(2.0));
    let df = simplify(&diff(&f, "x"));
    let v = eval(&df, &vars(&[("x", 3.0)])).unwrap();
    assert!((v - 6.0).abs() < 1e-10, "expected 6.0, got {v}");
}

#[test]
fn test_diff_power_rule_x_cubed() {
    // d/dx x³ = 3x²  →  at x=2: 12
    let x = Expr::var("x");
    let f = x.clone().pow(Expr::from(3.0));
    let df = simplify(&diff(&f, "x"));
    let v = eval(&df, &vars(&[("x", 2.0)])).unwrap();
    assert!((v - 12.0).abs() < 1e-10, "expected 12.0, got {v}");
}

#[test]
fn test_diff_sin_is_cos() {
    // d/dx sin(x) = cos(x)  →  at x=0: 1
    let x = Expr::var("x");
    let f = x.clone().sin();
    let df = simplify(&diff(&f, "x"));
    let v = eval(&df, &vars(&[("x", 0.0)])).unwrap();
    assert!((v - 1.0).abs() < 1e-10, "expected 1.0, got {v}");
}

#[test]
fn test_diff_cos_is_neg_sin() {
    // d/dx cos(x) = -sin(x)  →  at x=π/2: -1
    let x = Expr::var("x");
    let f = x.clone().cos();
    let df = simplify(&diff(&f, "x"));
    let pi_half = std::f64::consts::PI / 2.0;
    let v = eval(&df, &vars(&[("x", pi_half)])).unwrap();
    assert!((v - (-1.0)).abs() < 1e-10, "expected -1.0, got {v}");
}

#[test]
fn test_diff_chain_rule_exp_x_squared() {
    // d/dx exp(x²) = 2x * exp(x²)  →  at x=1: 2e
    let x = Expr::var("x");
    let f = x.clone().pow(Expr::from(2.0)).exp();
    let df = simplify(&diff(&f, "x"));
    let v = eval(&df, &vars(&[("x", 1.0)])).unwrap();
    let expected = 2.0 * std::f64::consts::E;
    assert!((v - expected).abs() < 1e-8, "expected {expected}, got {v}");
}

#[test]
fn test_diff_ln() {
    // d/dx ln(x) = 1/x  →  at x=2: 0.5
    let x = Expr::var("x");
    let f = x.clone().ln();
    let df = simplify(&diff(&f, "x"));
    let v = eval(&df, &vars(&[("x", 2.0)])).unwrap();
    assert!((v - 0.5).abs() < 1e-10, "expected 0.5, got {v}");
}

#[test]
fn test_diff_exp() {
    // d/dx exp(x) = exp(x)  →  at x=0: 1
    let x = Expr::var("x");
    let f = x.clone().exp();
    let df = simplify(&diff(&f, "x"));
    let v = eval(&df, &vars(&[("x", 0.0)])).unwrap();
    assert!((v - 1.0).abs() < 1e-10, "expected 1.0, got {v}");
}

#[test]
fn test_diff_quotient_rule() {
    // d/dx (x / (x+1))  →  at x=1: 1/(x+1)²|x=1 = 0.25
    let x = Expr::var("x");
    let f = x.clone() / (x.clone() + Expr::from(1.0));
    let df = simplify(&diff(&f, "x"));
    let v = eval(&df, &vars(&[("x", 1.0)])).unwrap();
    assert!((v - 0.25).abs() < 1e-8, "expected 0.25, got {v}");
}

#[test]
fn test_diff_negation() {
    // d/dx (-x²) = -2x  →  at x=3: -6
    let x = Expr::var("x");
    let f = -x.clone().pow(Expr::from(2.0));
    let df = simplify(&diff(&f, "x"));
    let v = eval(&df, &vars(&[("x", 3.0)])).unwrap();
    assert!((v - (-6.0)).abs() < 1e-10, "expected -6.0, got {v}");
}

#[test]
fn test_diff_second_derivative_sin() {
    // d²/dx² sin(x) = -sin(x)  →  at x=1: -sin(1)
    let x = Expr::var("x");
    let f = x.clone().sin();
    let df = simplify(&diff(&f, "x"));
    let ddf = simplify(&diff(&df, "x"));
    let v = eval(&ddf, &vars(&[("x", 1.0)])).unwrap();
    let expected = -1.0_f64.sin();
    assert!((v - expected).abs() < 1e-8, "expected {expected}, got {v}");
}

#[test]
fn test_diff_n_third_derivative_x4() {
    // d³/dx³ (x⁴) = 24x  →  at x=2: 48
    let x = Expr::var("x");
    let f = x.clone().pow(Expr::from(4.0));
    let d3f = diff_n(&f, "x", 3);
    let v = eval(&d3f, &vars(&[("x", 2.0)])).unwrap();
    assert!((v - 48.0).abs() < 1e-6, "expected 48.0, got {v}");
}

#[test]
fn test_diff_n_zero_returns_original() {
    let x = Expr::var("x");
    let f = x.clone().pow(Expr::from(2.0));
    let d0f = diff_n(&f, "x", 0);
    // Should be x², evaluate at 3 → 9
    let v = eval(&d0f, &vars(&[("x", 3.0)])).unwrap();
    assert!((v - 9.0).abs() < 1e-10);
}

#[test]
fn test_diff_sqrt() {
    // d/dx sqrt(x) = 1/(2*sqrt(x))  →  at x=4: 0.25
    let x = Expr::var("x");
    let f = x.clone().sqrt();
    let df = simplify(&diff(&f, "x"));
    let v = eval(&df, &vars(&[("x", 4.0)])).unwrap();
    assert!((v - 0.25).abs() < 1e-10, "expected 0.25, got {v}");
}

// ─── Display tests ────────────────────────────────────────────────────────────

#[test]
fn test_display_const_integer() {
    assert_eq!(format!("{}", Expr::from(3.0)), "3");
}

#[test]
fn test_display_const_float() {
    assert_eq!(format!("{}", Expr::from(3.5)), "3.5");
}

#[test]
fn test_display_var() {
    assert_eq!(format!("{}", Expr::var("theta")), "theta");
}

#[test]
fn test_display_add() {
    let e = Expr::var("x") + Expr::from(1.0);
    assert_eq!(format!("{e}"), "(x + 1)");
}

#[test]
fn test_display_mul() {
    let e = Expr::from(2.0) * Expr::var("x");
    assert_eq!(format!("{e}"), "(2 * x)");
}

#[test]
fn test_display_sin() {
    let e = Expr::var("x").sin();
    assert_eq!(format!("{e}"), "sin(x)");
}

#[test]
fn test_display_neg() {
    let e = -Expr::var("x");
    assert_eq!(format!("{e}"), "(-x)");
}

// ─── Expr utility tests ───────────────────────────────────────────────────────

#[test]
fn test_expr_node_count() {
    let x = Expr::var("x");
    // x + 1 has 3 nodes (Add, Var, Const)
    let e = x.clone() + Expr::from(1.0);
    assert_eq!(e.node_count(), 3);
}

#[test]
fn test_expr_variables() {
    let x = Expr::var("x");
    let y = Expr::var("y");
    let f = x.clone() * y.clone() + x;
    let vars_set = f.variables();
    assert!(vars_set.contains("x"));
    assert!(vars_set.contains("y"));
    assert_eq!(vars_set.len(), 2);
}

#[test]
fn test_expr_contains_var() {
    let x = Expr::var("x");
    let f = x.clone() + Expr::from(1.0);
    assert!(f.contains_var("x"));
    assert!(!f.contains_var("y"));
}

#[test]
fn test_expr_is_zero_is_one() {
    assert!(Expr::zero().is_zero());
    assert!(!Expr::one().is_zero());
    assert!(Expr::one().is_one());
    assert!(!Expr::zero().is_one());
}

#[test]
fn test_expr_as_const() {
    assert_eq!(Expr::from(3.0).as_const(), Some(3.0));
    assert_eq!(Expr::var("x").as_const(), None);
}

// ─── Error Display tests ──────────────────────────────────────────────────────

#[test]
fn test_error_display_unbound() {
    let e = SymbolicError::UnboundVariable("y".to_string());
    assert!(e.to_string().contains("y"));
}

#[test]
fn test_error_display_division_by_zero() {
    let e = SymbolicError::DivisionByZero;
    assert!(e.to_string().contains("zero"));
}

#[test]
fn test_error_display_domain() {
    let e = SymbolicError::DomainError("test".to_string());
    assert!(e.to_string().contains("test"));
}
