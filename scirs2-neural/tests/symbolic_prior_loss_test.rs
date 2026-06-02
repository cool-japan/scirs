//! Integration tests for `SymbolicPriorLoss` in `scirs2-neural`.
//!
//! These tests are gated on the `symbolic` feature and verify the end-to-end
//! behaviour of the symbolic-prior regularisation loss.

#[cfg(feature = "symbolic")]
mod tests {
    use scirs2_neural::losses::symbolic_prior_loss::SymbolicPriorLoss;
    use scirs2_symbolic::neural_priors::SeriesPrior;
    use scirs2_symbolic::LoweredOp;

    /// Build a trivial AR(1) prior: `f(window) = window[lookback - 1]`.
    fn ar1_prior(lookback: usize) -> SeriesPrior {
        let var_idx = lookback.saturating_sub(1);
        SeriesPrior {
            formulas: vec![(LoweredOp::Var(var_idx), 1.0)],
            variable_names: (0..lookback)
                .map(|j| format!("y[t-{}]", lookback - j))
                .collect(),
            lookback,
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // Test 1 — total loss ≈ base_loss when neural matches prior exactly
    // ─────────────────────────────────────────────────────────────────────

    #[test]
    fn symbolic_prior_loss_zero_when_neural_matches() {
        let prior = ar1_prior(1); // prior formula = Var(0) = window[0]
        let loss_fn = SymbolicPriorLoss::new(prior, 1.0);

        // window=[5.5]; Var(0)=5.5; neural_pred=5.5 → reg=0 → total=base_loss
        let total = loss_fn
            .total_loss(0.5, 5.5, &[5.5])
            .expect("should succeed");

        assert!(
            (total - 0.5).abs() < 1e-10,
            "expected total ≈ 0.5 (base_loss), got {total}"
        );
    }

    // ─────────────────────────────────────────────────────────────────────
    // Test 2 — total loss > base_loss when neural deviates from all priors
    // ─────────────────────────────────────────────────────────────────────

    #[test]
    fn symbolic_prior_loss_penalizes_deviation() {
        let prior = ar1_prior(1); // Var(0) = window[0]
        let loss_fn = SymbolicPriorLoss::new(prior, 1.0);

        // window=[0.0]; Var(0)=0.0; neural_pred=2.0 → reg=(2-0)^2*1=4.0 → total=4.5
        let total = loss_fn
            .total_loss(0.5, 2.0, &[0.0])
            .expect("should succeed");

        assert!(
            total > 0.5,
            "total loss should exceed base_loss when neural deviates; got {total}"
        );
        // Exact value: 0.5 + 4.0 = 4.5
        assert!(
            (total - 4.5).abs() < 1e-10,
            "expected total ≈ 4.5, got {total}"
        );
    }

    // ─────────────────────────────────────────────────────────────────────
    // Test 3 — construction and basic sanity without panics
    // ─────────────────────────────────────────────────────────────────────

    #[test]
    fn symbolic_prior_loss_construction() {
        // Build a multi-formula prior to verify no panic during construction.
        let prior = SeriesPrior {
            formulas: vec![
                (LoweredOp::Var(0), 0.95),
                (
                    LoweredOp::Mul(Box::new(LoweredOp::Var(0)), Box::new(LoweredOp::Const(0.9))),
                    0.88,
                ),
            ],
            variable_names: vec!["y[t-1]".to_string()],
            lookback: 1,
        };

        let loss_fn = SymbolicPriorLoss::new(prior, 0.5);

        // Verify lambda stored correctly.
        assert!((loss_fn.lambda - 0.5).abs() < 1e-15);
        assert_eq!(loss_fn.prior.lookback, 1);
        assert_eq!(loss_fn.prior.formulas.len(), 2);

        // Evaluate: window=[1.0], neural_pred=1.0
        // Var(0)=1.0 → prior_preds=[1.0, 0.9]
        // min_k |1.0 - preds[k]|^2 = min(0, 0.01) = 0.0 → reg=0
        let total = loss_fn
            .total_loss(1.0, 1.0, &[1.0])
            .expect("evaluation should not panic");
        assert!(
            (total - 1.0).abs() < 1e-10,
            "expected total≈1.0 (base_loss), got {total}"
        );
    }
}
