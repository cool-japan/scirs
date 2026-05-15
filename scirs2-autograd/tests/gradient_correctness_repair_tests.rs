//! Wave 73 — Gradient correctness repair tests
//!
//! Covers:
//! 1. `extract_diag` forward-pass correctness (single canonical definition)
//! 2. `ScalarMulOp` first derivative correctness
//! 3. `ScalarMulOp` second derivative (was broken by .eval() collapse; now symbolic)
//! 4. `ScalarMulOp` third derivative is zero for quadratic inputs
//! 5. `jit_fusion` recognises `MatmulEpilogue` and `BatchedMatmulReduction` patterns

use ag::tensor_ops as T;
use scirs2_autograd as ag;
use scirs2_autograd::jit_fusion::{FusionConfig, FusionKindJit, JitFusionEngine, JitNode, JitOp};
use scirs2_core::ndarray::array;

// ---------------------------------------------------------------------------
// Test 1 — extract_diag forward pass (single canonical definition)
// ---------------------------------------------------------------------------

#[test]
fn extract_diag_correct_for_3x3() {
    ag::run(|g| {
        // 3 × 3 matrix with known diagonal [10, 20, 30]
        let data = array![[10.0_f64, 1.0, 2.0], [3.0, 20.0, 4.0], [5.0, 6.0, 30.0]].into_dyn();
        let m = T::convert_to_tensor(data, g);
        let d = T::extract_diag(m);
        let result = d.eval(g).expect("extract_diag eval");

        assert_eq!(result.shape(), &[3], "shape should be [3]");
        assert!((result[[0]] - 10.0).abs() < 1e-12, "diag[0] should be 10");
        assert!((result[[1]] - 20.0).abs() < 1e-12, "diag[1] should be 20");
        assert!((result[[2]] - 30.0).abs() < 1e-12, "diag[2] should be 30");
    });
}

// ---------------------------------------------------------------------------
// Test 2 — ScalarMulOp first derivative
//
// f(x) = 2 * x^2  →  f'(x) = 4*x  →  at x=3: f'(3) = 12
//
// Uses `T::variable` (differentiable) rather than `T::convert_to_tensor`
// (non-differentiable) so the backprop path is correctly identified.
// ---------------------------------------------------------------------------

#[test]
fn scalar_mul_first_derivative() {
    ag::run(|g| {
        let x_data = array![3.0_f64].into_dyn();
        // variable() creates a differentiable tensor; convert_to_tensor sets
        // is_differentiable=false and would silently yield zero gradients.
        let x = T::variable(x_data, g);

        // f = 2 * x^2
        let x_sq = T::mul(x, x); // x^2 — multiply x by itself
        let f = T::scalar_mul(x_sq, 2.0);

        let grads = T::grad(&[f], &[x]);
        assert!(!grads.is_empty(), "gradient vec is non-empty");
        let g_val = grads[0].eval(g).expect("gradient eval");
        // df/dx = 2 * 2*x = 4*x; at x=3 => 12
        assert!(
            (g_val[[0]] - 12.0).abs() < 1e-10,
            "first derivative should be 12, got {}",
            g_val[[0]]
        );
    });
}

// ---------------------------------------------------------------------------
// Test 3 — ScalarMulOp second derivative (KEY TEST)
//
// f(x) = 2 * x^2  →  f''(x) = 4  (constant)
//
// With the old `.eval()` implementation the second gradient chain was broken
// because the tape collapsed to a float constant, yielding 0.  The fix
// propagates symbolically so the second derivative is correctly 4.
// ---------------------------------------------------------------------------

#[test]
fn scalar_mul_second_derivative() {
    ag::run(|g| {
        let x_data = array![3.0_f64].into_dyn();
        let x = T::variable(x_data, g);

        // f = 2 * x^2
        let x_sq = T::mul(x, x);
        let f = T::scalar_mul(x_sq, 2.0);

        // First derivative: df/dx = 4*x
        let first_grads = T::grad(&[f], &[x]);
        assert!(!first_grads.is_empty(), "first grad vec non-empty");
        let df = &first_grads[0];

        // Second derivative: d²f/dx² = 4
        let second_grads = T::grad(&[*df], &[x]);
        assert!(!second_grads.is_empty(), "second grad vec non-empty");
        let g2_val = second_grads[0].eval(g).expect("second derivative eval");
        assert!(
            (g2_val[[0]] - 4.0).abs() < 1e-8,
            "second derivative of 2*x^2 should be 4, got {}",
            g2_val[[0]]
        );
    });
}

// ---------------------------------------------------------------------------
// Test 4 — ScalarMulOp third derivative is zero for a quadratic
//
// f(x) = 2 * x^2  →  f'''(x) = 0
// ---------------------------------------------------------------------------

#[test]
fn scalar_mul_third_derivative_is_zero() {
    ag::run(|g| {
        let x_data = array![2.0_f64].into_dyn();
        let x = T::variable(x_data, g);

        let x_sq = T::mul(x, x);
        let f = T::scalar_mul(x_sq, 2.0);

        let g1 = T::grad(&[f], &[x]);
        assert!(!g1.is_empty());
        let g2 = T::grad(&[g1[0]], &[x]);
        assert!(!g2.is_empty());
        let g3 = T::grad(&[g2[0]], &[x]);
        assert!(!g3.is_empty());
        let g3_val = g3[0].eval(g).expect("third derivative eval");
        assert!(
            g3_val[[0]].abs() < 1e-8,
            "third derivative of 2*x^2 should be 0, got {}",
            g3_val[[0]]
        );
    });
}

// ---------------------------------------------------------------------------
// Test 5 — jit_fusion MatmulEpilogue: matmul → bias_add is fusable
// ---------------------------------------------------------------------------

#[test]
fn jit_fusion_matmul_bias_add_fusable_as_matmul_epilogue() {
    let engine = JitFusionEngine::new(FusionConfig::default());

    // Graph: Input0, Input1 → MatMul → BiasAdd (← Input2)
    let graph = vec![
        JitNode::new(0, JitOp::Input, vec![]),
        JitNode::new(1, JitOp::Input, vec![]),
        JitNode::new(2, JitOp::MatMul, vec![0, 1]),
        JitNode::new(3, JitOp::Input, vec![]),
        JitNode::new(4, JitOp::BiasAdd, vec![2, 3]),
    ];

    let fusions = engine.detect_fusions(&graph);

    // Either LinearBias or MatmulEpilogue should be detected — both are correct.
    let detected = fusions
        .iter()
        .any(|f| f.kind == FusionKindJit::LinearBias || f.kind == FusionKindJit::MatmulEpilogue);
    assert!(
        detected,
        "matmul → bias_add should be detected as LinearBias or MatmulEpilogue; got: {:?}",
        fusions.iter().map(|f| f.kind).collect::<Vec<_>>()
    );
}

// ---------------------------------------------------------------------------
// Test 6 — jit_fusion BatchedMatmulReduction: matmul → reduce_sum is fusable
// ---------------------------------------------------------------------------

#[test]
fn jit_fusion_batched_matmul_reduce_sum_fusable() {
    let engine = JitFusionEngine::new(FusionConfig::default());

    // Graph: Input0, Input1 → MatMul → ReduceSum
    let graph = vec![
        JitNode::new(0, JitOp::Input, vec![]),
        JitNode::new(1, JitOp::Input, vec![]),
        JitNode::new(2, JitOp::MatMul, vec![0, 1]),
        JitNode::new(3, JitOp::ReduceSum, vec![2]),
    ];

    let fusions = engine.detect_fusions(&graph);

    let detected = fusions
        .iter()
        .any(|f| f.kind == FusionKindJit::BatchedMatmulReduction);
    assert!(
        detected,
        "matmul → reduce_sum should be detected as BatchedMatmulReduction; got: {:?}",
        fusions.iter().map(|f| f.kind).collect::<Vec<_>>()
    );
}

// ---------------------------------------------------------------------------
// Test 7 — jit_fusion BatchedMatmulReduction: batched matmul → reduce_mean
// ---------------------------------------------------------------------------

#[test]
fn jit_fusion_batched_matmul_reduce_mean_fusable() {
    let engine = JitFusionEngine::new(FusionConfig::default());

    // Graph: Input0, Input1 → BatchedMatMul → ReduceMean
    let graph = vec![
        JitNode::new(0, JitOp::Input, vec![]),
        JitNode::new(1, JitOp::Input, vec![]),
        JitNode::new(2, JitOp::BatchedMatMul, vec![0, 1]),
        JitNode::new(3, JitOp::ReduceMean, vec![2]),
    ];

    let fusions = engine.detect_fusions(&graph);

    let detected = fusions
        .iter()
        .any(|f| f.kind == FusionKindJit::BatchedMatmulReduction);
    assert!(
        detected,
        "BatchedMatMul → ReduceMean should be detected as BatchedMatmulReduction; got: {:?}",
        fusions.iter().map(|f| f.kind).collect::<Vec<_>>()
    );
}

// ---------------------------------------------------------------------------
// Test 8 — jit_fusion MatmulEpilogue: longer chain (matmul → add → relu → scale)
// ---------------------------------------------------------------------------

#[test]
fn jit_fusion_matmul_epilogue_longer_chain() {
    let engine = JitFusionEngine::new(FusionConfig::default());

    // matmul(0,1) → add(2,3) → relu(4) → scale(5)
    let graph = vec![
        JitNode::new(0, JitOp::Input, vec![]),
        JitNode::new(1, JitOp::Input, vec![]),
        JitNode::new(2, JitOp::MatMul, vec![0, 1]),
        JitNode::new(3, JitOp::Input, vec![]),
        JitNode::new(4, JitOp::Add, vec![2, 3]),
        JitNode::new(5, JitOp::Relu, vec![4]),
        JitNode::new(6, JitOp::Scale, vec![5]),
    ];

    let fusions = engine.detect_fusions(&graph);

    // At least LinearBiasActivation or MatmulEpilogue should fire.
    let detected = fusions.iter().any(|f| {
        f.kind == FusionKindJit::LinearBiasActivation || f.kind == FusionKindJit::MatmulEpilogue
    });
    assert!(
        detected,
        "matmul → add → relu → scale should yield a matmul-epilogue-style fusion; got: {:?}",
        fusions.iter().map(|f| f.kind).collect::<Vec<_>>()
    );
}
