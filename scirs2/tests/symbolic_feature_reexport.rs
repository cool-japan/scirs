//! Compile-time check that the `symbolic` feature re-exports `scirs2_symbolic` correctly.
//!
//! When the `symbolic` feature is enabled, `scirs2::symbolic` must resolve to
//! `scirs2_symbolic`, and its public API (`Expr`, `diff`, `simplify`, `eval`) must be
//! accessible through the facade.
//!
//! The real assertion here is compilation — if the `use` statements below fail to
//! compile, the feature wiring is broken.
//!
//! Compile-check:
//!   cargo check -p scirs2 --features symbolic

// When the `symbolic` feature is active, the module and its key types must be visible.
#[cfg(feature = "symbolic")]
#[allow(unused_imports)]
use scirs2::symbolic as _;

#[cfg(feature = "symbolic")]
mod symbolic_reexport {
    use scirs2::symbolic::{diff, eval, simplify, Expr};
    use std::collections::HashMap;

    #[test]
    fn can_see_symbolic_module() {
        // Build a simple expression through the facade and verify it evaluates correctly.
        let x = Expr::var("x");
        let f = x.clone().pow(Expr::from(2.0)); // x²
        let df = simplify(&diff(&f, "x")); // 2x

        let mut vars = HashMap::new();
        vars.insert("x", 3.0_f64);
        // At x=3: 2*3 = 6
        let result = eval(&df, &vars).expect("eval should succeed");
        assert!((result - 6.0).abs() < 1e-10, "expected 6.0, got {result}");
    }

    #[test]
    fn symbolic_error_is_accessible() {
        use scirs2::symbolic::SymbolicError;
        let x = Expr::var("x");
        let err = eval(&x, &HashMap::new());
        assert!(matches!(err, Err(SymbolicError::UnboundVariable(_))));
    }
}

#[cfg(not(feature = "symbolic"))]
mod no_symbolic {
    #[test]
    fn symbolic_feature_not_enabled() {
        // Normal CI path — symbolic feature disabled; nothing to verify at runtime.
    }
}
