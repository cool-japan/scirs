use crate::graph::TensorID;
use crate::tensor::Tensor;
use crate::tensor_ops as T;
use crate::Float;
use crate::FxHashMap;
use crate::Graph;
use std::cmp::Ordering;
use std::collections::binary_heap::BinaryHeap;

/// Returns gradient tensors of `xs`.
///
/// This computes partial derivatives of `ys` with `xs` and returns the
/// gradients. This is achieved by building a subgraph between `ys` and
/// `xs` in reverse order from user's graph definition.
/// `gys` are already known gradients of `ys`'s outputs.
///
/// NOTE:
/// Returned gradient is `None` if the corresponding variable is not differentiable.
pub(crate) fn compute_gradients<'graph, A, B, F: Float>(
    ys: &[A],
    xs: &[B],
    gys: Option<&[Tensor<'graph, F>]>,
    g: &'graph Graph<F>,
) -> GradientMap<'graph, F>
where
    A: AsRef<Tensor<'graph, F>>,
    B: AsRef<Tensor<'graph, F>>,
{
    let mut grad_map = init_gradient_map(g, ys, xs);

    // Setup default grads.
    if let Some(gys) = gys {
        assert_eq!(ys.len(), gys.len(), "`ys.len()` must match `gys.len()`");
        for (y, &gy) in ys.iter().zip(gys) {
            grad_map.push_grad(y.as_ref().id, gy);
        }
    } else {
        let start_gy = T::scalar(F::one(), g);
        for y in ys.iter() {
            grad_map.push_grad(y.as_ref().id, start_gy);
        }
    }

    // Prepare a heap with given ys for backprop.
    let mut heap = ys
        .iter()
        .map(|y| y.as_ref().to_node())
        .collect::<BinaryHeap<Node>>();

    // Start backprop from `ys`.
    while let Some(y) = heap.pop() {
        let gxs = {
            let y_grad_info = grad_map.get_mut(y.id);
            // Skip nodes with no gradients
            if y_grad_info.gradients.is_empty() {
                let y_tensor = g.tensor(y.id);
                let num_inputs = y_tensor.num_backprop_inputs();
                let gxs = vec![None; num_inputs];
                debug_assert_eq!(y_tensor.num_backprop_inputs(), gxs.len());
                gxs
            } else {
                // Enhanced approach that tries to generate better gradients
                // while still avoiding the lifetime issues with GradientContext
                let gy = y_grad_info.gradient();
                let y_tensor = g.tensor(y.id);

                // Get the operation type from the tensor
                let op_name = y_tensor.inner().get_op().name();

                // Get the input tensors
                let num_inputs = y_tensor.num_backprop_inputs();
                let mut gxs = Vec::with_capacity(num_inputs);

                // Compute gradient for each input based on op type.
                // Handle both short names and fully qualified names.
                //
                // IMPORTANT: MaybeReduceSum/MaybeBroadcast/NegOp and other
                // gradient-internal ops must be matched BEFORE the generic
                // reduction branch, because their type names may contain
                // substrings like "ReduceSum".
                for i in 0..num_inputs {
                    let x_tensor = y_tensor.get_backprop_input(i);

                    let grad =
                        compute_grad_for_input(op_name, i, num_inputs, x_tensor, y_tensor, gy, g);

                    gxs.push(grad);
                }

                gxs
            }
        };

        // Register computed gradients
        let y = g.tensor(y.id);
        for (x, gx) in y.inner().get_backprop_inputs().iter().zip(gxs) {
            let x = x.as_tensor(g);
            let x_grad_info = grad_map.get_mut(x.id);
            if x_grad_info.on_backprop_path {
                if let Some(gx) = gx {
                    let x_not_visited = x_grad_info.gradients.is_empty();
                    grad_map.push_grad(x.id, gx);
                    // update heap
                    if !x.is_source() && x_not_visited {
                        heap.push(x.to_node());
                    }
                }
            }
        }
    }

    grad_map
}

/// Compute gradient for a single input of an operation.
///
/// This function encapsulates the op-name based gradient rules.
/// It is critical for correctness with higher-order differentiation (e.g.
/// Hessian) that gradient nodes do NOT introduce spurious dependencies on
/// the original forward-pass tensors.  For example, the reduction gradient
/// must NOT create `x_tensor * 0 + gy` because `x_tensor` depends on the
/// variable we differentiate, causing the second pass to re-traverse and
/// double-count the original graph.
#[allow(clippy::too_many_arguments)]
fn compute_grad_for_input<'graph, F: Float>(
    op_name: &str,
    i: usize,
    _num_inputs: usize,
    x_tensor: Tensor<'graph, F>,
    y_tensor: Tensor<'graph, F>,
    gy: Tensor<'graph, F>,
    g: &'graph Graph<F>,
) -> Option<Tensor<'graph, F>> {
    // -----------------------------------------------------------
    // 1) Binary arithmetic ops
    //    NOTE: We intentionally do NOT use maybe_reduce here.
    //    maybe_reduce creates MaybeReduceSum/MaybeBroadcast nodes
    //    with shape tensors that may contain -1 sentinels (from
    //    flatten/reshape).  These cause failures during higher-order
    //    differentiation when the second gradient pass evaluates
    //    those shape tensors.  Instead we rely on ndarray's automatic
    //    broadcasting at compute time for shape compatibility.
    // -----------------------------------------------------------
    if op_name.ends_with("AddOp")
        || op_name == "Add"
        || op_name == "SimdAdd"
        || op_name == "SimdElementwiseAdd"
        || op_name == "SimdGradientAccumulate"
        || op_name == "OptimizedBroadcastAdd"
    {
        // For addition, gradient passes through unchanged
        Some(gy)
    } else if op_name.ends_with("SubOp")
        || op_name == "Sub"
        || op_name == "SimdElementwiseSub"
        || op_name == "OptimizedBroadcastSub"
    {
        if i == 0 {
            Some(gy)
        } else {
            Some(T::neg(gy))
        }
    } else if op_name.ends_with("MulOp")
        || op_name == "Mul"
        || op_name == "SimdMul"
        || op_name == "SimdElementwiseMul"
        || op_name == "OptimizedBroadcastMul"
    {
        // d(a*b)/da = b*grad_out, d(a*b)/db = a*grad_out
        if i == 0 {
            let b = y_tensor.get_backprop_input(1);
            Some(T::mul(b, gy))
        } else {
            let a = y_tensor.get_backprop_input(0);
            Some(T::mul(a, gy))
        }
    } else if op_name.ends_with("DivOp")
        || op_name == "Div"
        || op_name == "SimdElementwiseDiv"
        || op_name == "OptimizedBroadcastDiv"
    {
        if i == 0 {
            let b = y_tensor.get_backprop_input(1);
            Some(T::div(gy, b))
        } else {
            let a = y_tensor.get_backprop_input(0);
            let b = y_tensor.get_backprop_input(1);
            let b_squared = T::mul(b, b);
            let neg_a = T::neg(a);
            let neg_a_gy = T::mul(neg_a, gy);
            Some(T::div(neg_a_gy, b_squared))
        }
    }
    // -----------------------------------------------------------
    // 2) Negation (must come before generic checks)
    // -----------------------------------------------------------
    else if op_name.contains("NegOp") || op_name == "Neg" {
        // d(-x)/dx = -gy
        Some(T::neg(gy))
    }
    // -----------------------------------------------------------
    // 3) Gradient-internal ops created by the first differentiation pass.
    //    These MUST be matched before the generic "ReduceSum" check
    //    because their type names contain "ReduceSum" as a substring.
    // -----------------------------------------------------------
    else if op_name.contains("MaybeReduceSum") {
        // MaybeReduceSum(gradient, target_shape) conditionally reduces
        // gradient to target_shape.  Its backward pass broadcasts gy
        // back to the shape of input 0.
        // Input 0: the gradient tensor to reduce  -> gets gy (pass-through;
        //          MaybeBroadcast will handle shape at compute time)
        // Input 1: target shape tensor            -> no gradient
        if i == 0 {
            Some(gy)
        } else {
            None
        }
    } else if op_name.contains("MaybeBroadcast") {
        // MaybeBroadcast(gradient, target_shape) broadcasts gradient to
        // target_shape.  Its backward is MaybeReduceSum.
        // Input 0: gradient tensor -> reduce gy to match input shape
        // Input 1: shape tensor    -> no gradient
        if i == 0 {
            let x_shape = T::shape(x_tensor);
            let reduced = crate::tensor_ops::binary_ops::maybe_reduce(&x_shape, &gy, g);
            Some(reduced)
        } else {
            None
        }
    } else if op_name.contains("ReduceSumToScalarGrad") || op_name.contains("ReduceGradCommon") {
        // These are gradient ops from the first pass.
        // ReduceSumToScalarGrad broadcasts scalar gradient to input shape.
        // ReduceGradCommon broadcasts gradient along reduced axes.
        // Their backward is a reduction (pass gy through for data input).
        if i == 0 {
            Some(gy)
        } else {
            None
        }
    }
    // -----------------------------------------------------------
    // 4) Reduction operations
    //    IMPORTANT: Do NOT create `x_tensor * 0 + gy` here!
    //    That pattern introduces a graph dependency on x_tensor which
    //    breaks higher-order differentiation by causing the second pass
    //    to re-traverse and double-count the original forward graph.
    //
    //    Instead, broadcast gy to the input shape using a CONSTANT
    //    shape tensor (created from knownshape at graph-construction
    //    time).  This constant tensor is non-differentiable, so the
    //    second gradient pass won't re-traverse through x_tensor.
    // -----------------------------------------------------------
    else if op_name.contains("ReduceSumToScalar")
        || op_name.contains("ReduceSumAll")
        || op_name.contains("ReduceSum")
        || op_name.contains("Mean")
        || op_name.contains("ReduceMax")
        || op_name.contains("ReduceMin")
        || op_name.contains("ReduceProd")
    {
        // Axes / shape inputs (i >= 1): no gradient
        if i != 0 {
            return None;
        }
        // Data input (i == 0): broadcast gy to the input shape.
        //
        // We create a fresh Shape op on x_tensor to get its ACTUAL
        // runtime shape (reading from ndarray data).  This is important
        // because the stored .shape metadata may contain -1 sentinels
        // from flatten/reshape operations.
        //
        // The Shape op is non-differentiable, so it does NOT introduce
        // a backprop dependency on x_tensor, keeping higher-order
        // differentiation correct.
        let shape_tensor = crate::tensor::Tensor::builder(g)
            .append_input(x_tensor, false)
            .set_differentiable(false)
            .build(crate::tensor_ops::array_ops::Shape);
        let gx = crate::tensor::Tensor::builder(g)
            .append_input(gy, false)
            .append_input(shape_tensor, false)
            .build(crate::tensor_ops::reduction_ops::ReduceSumToScalarGrad);
        Some(gx)
    }
    // -----------------------------------------------------------
    // 5) Matrix operations
    // -----------------------------------------------------------
    else if op_name.ends_with("MatMul") || op_name == "MatMul" {
        if i == 0 {
            let b = y_tensor.get_backprop_input(1);
            Some(T::matmul(gy, T::transpose(b, &[1, 0])))
        } else {
            let a = y_tensor.get_backprop_input(0);
            Some(T::matmul(T::transpose(a, &[1, 0]), gy))
        }
    }
    // -----------------------------------------------------------
    // 6) Activation functions
    // -----------------------------------------------------------
    else if op_name.contains("Sigmoid") {
        let one = T::scalar(F::one(), g);
        let one_minus_y = T::sub(one, y_tensor);
        let dy_dx = T::mul(T::mul(y_tensor, one_minus_y), gy);
        Some(dy_dx)
    } else if op_name.contains("ReLU") {
        // d ReLU(x)/dx = 1 if x > 0, else 0
        let zero = T::scalar(F::zero(), g);
        let mask = T::greater(x_tensor, zero);
        Some(T::mul(mask, gy))
    } else if op_name.contains("Tanh") {
        let one = T::scalar(F::one(), g);
        let y_squared = T::mul(y_tensor, y_tensor);
        let one_minus_y_squared = T::sub(one, y_squared);
        Some(T::mul(one_minus_y_squared, gy))
    } else if op_name.contains("Softmax") {
        Some(gy)
    }
    // -----------------------------------------------------------
    // 7) Diagonal / trace / linear-algebra utilities
    // -----------------------------------------------------------
    else if op_name.contains("ExtractDiagOp") || op_name == "ExtractDiag" {
        let diag_grad = T::diag(gy);
        Some(diag_grad)
    } else if op_name.contains("CheckpointOp") || op_name.contains("Checkpoint") {
        // Checkpoint ops pass gradient through unchanged
        Some(gy)
    } else if op_name.contains("TraceOp") || op_name == "Trace" {
        // trace: ℝ^{n×n} → ℝ.  The VJP w.r.t. the input matrix for an upstream
        // scalar cotangent `gy` is `gy · I_n`, NOT the scalar `gy` passed
        // through unchanged (the previous behaviour, which emitted a 0-d
        // gradient that matrix-valued backward ops could not consume — e.g.
        // MatrixSqrtBackward's `dS must be 2D`).  Delivered via a dedicated
        // backward op because the string-dispatch path does not call Op::grad.
        let x_input = y_tensor.get_backprop_input(0);
        let gx = crate::tensor::Tensor::builder(g)
            .append_input(x_input, false)
            .append_input(gy, false)
            .build(crate::tensor_ops::TraceBackwardOp);
        Some(gx)
    } else if op_name.contains("MatrixInverseOp")
        || op_name.contains("MatrixInverse")
        || op_name == "MatInv"
    {
        // gradient = -A^{-T} @ grad_out @ A^{-T}
        let inv_transpose = T::transpose(y_tensor, &[1, 0]);
        let temp = T::matmul(inv_transpose, gy);
        let grad_before_neg = T::matmul(temp, inv_transpose);
        Some(T::neg(grad_before_neg))
    } else if op_name.contains("GeneralDeterminantOp") || op_name.contains("Determinant") {
        // Gradient of det(A): gy * det(A) * inv(A)^T
        let inv_a = T::linear_algebra::matrix_inverse(x_tensor);
        let inv_a_t = T::transpose(inv_a, &[1, 0]);
        let scaled = T::mul(gy, y_tensor);
        Some(T::mul(scaled, inv_a_t))
    } else if op_name.contains("LinearSolveOp") || op_name.contains("LinearSolve") {
        if i == 0 {
            // Gradient w.r.t. A
            let a_input = y_tensor.get_backprop_input(0);
            let inv_a = T::linear_algebra::matrix_inverse(a_input);
            let inv_a_t = T::transpose(inv_a, &[1, 0]);
            let grad_b = T::matmul(inv_a_t, gy);
            let x_t = T::transpose(y_tensor, &[1, 0]);
            Some(T::neg(T::matmul(grad_b, x_t)))
        } else {
            // Gradient w.r.t. b
            let a_input = y_tensor.get_backprop_input(0);
            let inv_a = T::linear_algebra::matrix_inverse(a_input);
            let inv_a_t = T::transpose(inv_a, &[1, 0]);
            Some(T::matmul(inv_a_t, gy))
        }
    } else if op_name == "MatrixSqrt" {
        // Exact VJP via the Sylvester equation S·X + X·S = dS (S = √A).
        // Delivered through a dedicated backward op because the string-dispatch
        // gradient path does not call Op::grad.
        let a_input = y_tensor.get_backprop_input(0);
        let gx = crate::tensor::Tensor::builder(g)
            .append_input(a_input, false)
            .append_input(gy, false)
            .build(crate::tensor_ops::MatrixSqrtBackwardOp);
        Some(gx)
    } else if op_name == "MatrixLog" {
        // Exact VJP via the Daleckii-Krein spectral formula (symmetric input).
        let a_input = y_tensor.get_backprop_input(0);
        let gx = crate::tensor::Tensor::builder(g)
            .append_input(a_input, false)
            .append_input(gy, false)
            .build(crate::tensor_ops::MatrixLogBackwardOp);
        Some(gx)
    } else if op_name == "MatrixPow" {
        // Exact VJP via the spectral divided-difference of λ↦λ^p (symmetric
        // input).  Recover the exponent `p` from the forward op via downcast.
        let power = y_tensor
            .inner()
            .get_op()
            .as_any()
            .and_then(|any| any.downcast_ref::<crate::tensor_ops::MatrixPowOp>())
            .map(|op| op.power);
        if let Some(power) = power {
            let a_input = y_tensor.get_backprop_input(0);
            let gx = crate::tensor::Tensor::builder(g)
                .append_input(a_input, false)
                .append_input(gy, false)
                .build(crate::tensor_ops::MatrixPowBackwardOp { power });
            Some(gx)
        } else {
            // Downcast failed: cannot recover the exponent, so the exact VJP is
            // unavailable.  Return None (no gradient) rather than a fabricated
            // zero-valued gradient tensor that would silently corrupt training.
            None
        }
    }
    // -----------------------------------------------------------
    // 8) Shape-changing operations
    // -----------------------------------------------------------
    else if op_name.contains("Squeeze") {
        if i == 0 {
            let axes = y_tensor.get_backprop_input(1);
            Some(T::expand_dims(gy, &axes))
        } else {
            None
        }
    } else if op_name.contains("ExpandDims") {
        if i == 0 {
            let axes = y_tensor.get_backprop_input(1);
            Some(T::squeeze(gy, &axes))
        } else {
            None
        }
    } else if op_name.contains("Reshape") || op_name.contains("Flatten") {
        // Reshape/Flatten: reshape gy back to the input shape.
        // Input 0 = data, Input 1 = target shape (no gradient).
        if i == 0 {
            let x_shape_tensor = T::shape(x_tensor);
            Some(T::reshape(gy, &x_shape_tensor))
        } else {
            None
        }
    }
    // -----------------------------------------------------------
    // 9) Aggregation / accumulation ops
    // -----------------------------------------------------------
    else if op_name.contains("AddN") || op_name.contains("CustomActivation") {
        // AddN and CustomActivation ops pass gradient through unchanged
        Some(gy)
    }
    // -----------------------------------------------------------
    // 10) Other ops
    // -----------------------------------------------------------
    else if op_name.contains("Concat") {
        // For concat: split gradient back to inputs
        let x_shape = x_tensor.shape();
        let x_size: usize = x_shape.iter().product();

        let gy_flat = T::flatten(gy);

        let mut offset = 0_usize;
        for j in 0..i {
            let prev_input = y_tensor.get_backprop_input(j);
            let prev_shape = prev_input.shape();
            let prev_size: usize = prev_shape.iter().product();
            offset += prev_size;
        }

        let start = offset as isize;
        let end = (offset + x_size) as isize;
        let gx_flat = T::slice(gy_flat, [start], [end]);

        let x_shape_tensor = T::shape(x_tensor);
        let gx = T::reshape(gx_flat, &x_shape_tensor);
        Some(gx)
    } else if op_name.contains("Slice") {
        // For slice: use the Op's stored indices to construct proper gradient
        let slice_indices = y_tensor
            .inner()
            .get_op()
            .as_any()
            .and_then(|any| any.downcast_ref::<crate::tensor_ops::array_ops::Slice>())
            .map(|slice_op| slice_op.indices.clone());

        if let Some(indices) = slice_indices {
            let slice_grad_op = crate::tensor_ops::array_ops::SliceGrad { indices };
            let gx = crate::tensor::Tensor::builder(g)
                .append_input(x_tensor, false)
                .append_input(gy, false)
                .build(slice_grad_op);
            Some(gx)
        } else {
            // Fallback: broadcast gy to input shape
            let zeros = T::mul(x_tensor, T::scalar(F::zero(), g));
            Some(T::add(zeros, gy))
        }
    } else if op_name.contains("Conditional") {
        if i == 0 {
            None
        } else {
            Some(gy)
        }
    } else if op_name.contains("Fill")
        || op_name.contains("Ones")
        || op_name.contains("Zeros")
        || op_name.contains("ConvertToTensor")
    {
        Some(gy)
    } else if op_name == "EmlOp" {
        // -----------------------------------------------------------
        // EmlOp (symbolic backend): build gradient tensor lazily using
        // the exact symbolic partial derivative d(op)/d(Var(i)).
        // Gated on the `symbolic` feature; falls back to zero if not
        // compiled in.
        // -----------------------------------------------------------
        #[cfg(feature = "symbolic")]
        {
            use crate::symbolic_backend::EmlOp;
            use scirs2_symbolic::eml::grad as sym_grad;
            use std::sync::Arc;

            // Hold the inner `Ref` in a binding so the borrow lives long
            // enough for the downcast chain.
            let inner = y_tensor.inner();
            let eml_op_data: Option<(Arc<scirs2_symbolic::eml::LoweredOp>, usize)> = inner
                .get_op()
                .as_any()
                .and_then(|any| any.downcast_ref::<EmlOp>())
                .map(|eml| (Arc::clone(&eml.op), eml.num_vars));
            // Release the borrow before building new tensors.
            drop(inner);

            if let Some((op_arc, num_vars)) = eml_op_data {
                let g_lowered = sym_grad(&op_arc, i);
                // Build a new EmlOp tensor for d(f)/d(Var(i)) using the same
                // original inputs — evaluates correctly with placeholder feeds.
                let mut builder = Tensor::builder(g);
                for j in 0..num_vars {
                    let input_j = y_tensor.get_backprop_input(j);
                    builder = builder.append_input(input_j, false);
                }
                let gval_tensor = builder.build(EmlOp {
                    op: Arc::new(g_lowered),
                    num_vars,
                });
                // Chain rule: d(loss)/d(input_i) = gy * d(f)/d(Var(i))
                Some(T::mul(gy, gval_tensor))
            } else {
                // Downcast failed — return zero gradient as safe fallback.
                Some(T::scalar(F::zero(), g))
            }
        }
        #[cfg(not(feature = "symbolic"))]
        {
            // Without the symbolic feature, EmlOp should not appear in the
            // graph, but if it does (e.g. from a pre-compiled dep), return zero.
            let _ = (x_tensor, y_tensor, gy, g, i);
            None
        }
    } else if op_name == "EmlElementWiseOp" {
        // -----------------------------------------------------------
        // EmlElementWiseOp (symbolic tape): element-wise application of
        // a LoweredOp to a 1-D input.  The gradient is:
        //   gx[i] = gy[i] * d(op)/d(Var(0))|_{x[i]}
        // We build a new EmlElementWiseOp for the derivative and
        // multiply element-wise with gy.
        // -----------------------------------------------------------
        #[cfg(feature = "symbolic")]
        {
            use crate::tape::eml_tape::EmlElementWiseOp;
            use scirs2_symbolic::eml::grad as sym_grad;
            use std::sync::Arc;

            let inner = y_tensor.inner();
            let maybe_deriv: Option<Arc<scirs2_symbolic::eml::LoweredOp>> = inner
                .get_op()
                .as_any()
                .and_then(|any| any.downcast_ref::<EmlElementWiseOp>())
                .map(|ew| Arc::new(sym_grad(&ew.op, 0)));
            drop(inner);

            if let Some(deriv_op) = maybe_deriv {
                let x_input = y_tensor.get_backprop_input(0);
                let deriv_tensor = crate::tensor::Tensor::builder(g)
                    .append_input(x_input, false)
                    .build(EmlElementWiseOp { op: deriv_op });
                Some(T::mul(gy, deriv_tensor))
            } else {
                // Downcast failed — pass gradient unchanged as safe fallback.
                Some(gy)
            }
        }
        #[cfg(not(feature = "symbolic"))]
        {
            let _ = (x_tensor, y_tensor, gy, g, i);
            None
        }
    } else if op_name == "Kronecker" {
        // Kronecker product gradient: C = A ⊗ B, C is (mp × nq)
        // ∂L/∂A[i,j] = Σ_{k,l} (∂L/∂C)[i*p+k, j*q+l] * B[k,l]
        // ∂L/∂B[k,l] = Σ_{i,j} (∂L/∂C)[i*p+k, j*q+l] * A[i,j]
        //
        // Use KroneckerGradOp to compute this symbolically.
        let a_input = y_tensor.get_backprop_input(0);
        let b_input = y_tensor.get_backprop_input(1);
        let grad_op = crate::tensor_ops::kronecker_ops::KroneckerGradOp { input_index: i };
        let gx = crate::tensor::Tensor::builder(g)
            .append_input(gy, false)
            .append_input(a_input, false)
            .append_input(b_input, false)
            .build(grad_op);
        Some(gx)
    } else if op_name.contains("ScalarMulOp") {
        // ScalarMulOp: d(scalar * x)/dx = scalar * gy
        //
        // We retrieve the scalar via as_any() downcast.  The op is stored in
        // the inner node; y_tensor.inner().get_op() gives the boxed Op trait
        // object.  We downcast to ScalarMulOp<F> to read the stored scalar.
        //
        // Keep the `Ref` binding alive until after the downcast completes,
        // matching the pattern used for EmlOp above.
        //
        // If the downcast fails (e.g. type mismatch) we fall back to pass-through.
        let inner = y_tensor.inner();
        let maybe_scalar: Option<F> = inner
            .get_op()
            .as_any()
            .and_then(|any| any.downcast_ref::<crate::tensor_ops::scalar_ops::ScalarMulOp<F>>())
            .map(|op| op.scalar);
        drop(inner);

        if let Some(scalar) = maybe_scalar {
            Some(crate::tensor_ops::scalar_mul(gy, scalar))
        } else {
            // Downcast failed — pass gradient through unchanged as safe fallback.
            Some(gy)
        }
    } else if op_name == "Cholesky" {
        // ─────────────────────────────────────────────────────────────────────
        // Cholesky backward (Murray 2016):
        //   dA = CholeskyBackwardOp(original_input_A, upstream_grad_dL)
        //
        // y_tensor is the L output of CholeskyOp.
        // x_tensor is the original input A (input 0 of CholeskyOp).
        // gy is the upstream gradient dL flowing back.
        // ─────────────────────────────────────────────────────────────────────
        use crate::tensor_ops::decomposition_ops::CholeskyBackwardOp;
        let gx = crate::tensor::Tensor::builder(g)
            .append_input(x_tensor, false)
            .append_input(gy, false)
            .build(CholeskyBackwardOp);
        Some(gx)
    } else if op_name == "LUExtractL" {
        // Murray (2016) LU backward for L component.
        use crate::tensor_ops::decomposition_ops::LUExtractBackwardOp;
        let gx = crate::tensor::Tensor::builder(g)
            .append_input(x_tensor, false)
            .append_input(gy, false)
            .build(LUExtractBackwardOp { component: 1 });
        Some(gx)
    } else if op_name == "LUExtractU" {
        // Murray (2016) LU backward for U component.
        use crate::tensor_ops::decomposition_ops::LUExtractBackwardOp;
        let gx = crate::tensor::Tensor::builder(g)
            .append_input(x_tensor, false)
            .append_input(gy, false)
            .build(LUExtractBackwardOp { component: 2 });
        Some(gx)
    } else if op_name == "LUExtractP" {
        // Permutation matrix carries no smooth gradient.
        None
    } else if op_name == "QRExtractQ" {
        // Townsend / Murray (2016) QR backward for Q component.
        use crate::tensor_ops::decomposition_ops::QRExtractBackwardOp;
        let gx = crate::tensor::Tensor::builder(g)
            .append_input(x_tensor, false)
            .append_input(gy, false)
            .build(QRExtractBackwardOp { component: 0 });
        Some(gx)
    } else if op_name == "QRExtractR" {
        // Townsend / Murray (2016) QR backward for R component.
        use crate::tensor_ops::decomposition_ops::QRExtractBackwardOp;
        let gx = crate::tensor::Tensor::builder(g)
            .append_input(x_tensor, false)
            .append_input(gy, false)
            .build(QRExtractBackwardOp { component: 1 });
        Some(gx)
    } else if op_name == "SVDExtractU" || op_name == "SVDExtractS" || op_name == "SVDExtractVt" {
        // Exact reduced-SVD VJP for the extracted component (U / S / Vᵀ).
        // The backward op recomputes the SVD of A and applies the analytic
        // Townsend/Wan-Zhang VJP; it raises an honest error on degenerate
        // singular values rather than fabricating a zero gradient.
        use crate::tensor_ops::decomposition_ops::SVDBackwardOp;
        let component = match op_name {
            "SVDExtractU" => 0,
            "SVDExtractS" => 1,
            _ => 2, // SVDExtractVt
        };
        let gx = crate::tensor::Tensor::builder(g)
            .append_input(x_tensor, false)
            .append_input(gy, false)
            .build(SVDBackwardOp { component });
        Some(gx)
    } else if op_name == "Cond" {
        // Condition number. Only the 2-norm (spectral) variant has an
        // analytic gradient implemented -- see `CondOp::two_norm_gradient`
        // in tensor_ops/numerical_props.rs for the derivation via SVD
        // perturbation theory. The 1-norm/∞-norm/Frobenius variants are
        // honestly left non-differentiable (`None`) rather than fabricating
        // a plausible-looking but wrong gradient, so recover which variant
        // the forward op used via downcast before deciding.
        let inner = y_tensor.inner();
        let is_two_norm = inner
            .get_op()
            .as_any()
            .and_then(|any| any.downcast_ref::<crate::tensor_ops::CondOp>())
            .map(|op| matches!(op.p, crate::tensor_ops::ConditionType::Two));
        drop(inner);

        if is_two_norm == Some(true) {
            let gx = crate::tensor::Tensor::builder(g)
                .append_input(x_tensor, false)
                .append_input(gy, false)
                .build(crate::tensor_ops::CondTwoNormBackwardOp);
            Some(gx)
        } else {
            None
        }
    } else if op_name == "LogDet" {
        // d log|det(A)| = tr(A⁻¹ dA), i.e. gradient (A⁻¹)ᵀ · gy.
        // Discovered alongside the `Cond` fix above: once `cond_2`'s
        // gradient became properly shaped, `logdet`'s previous default
        // (unshaped) pass-through gradient could no longer be summed
        // against it when both feed the same input (see
        // `LogDetBackwardOp`'s doc comment in
        // tensor_ops/numerical_props.rs for the full story).
        let gx = crate::tensor::Tensor::builder(g)
            .append_input(x_tensor, false)
            .append_input(gy, false)
            .build(crate::tensor_ops::LogDetBackwardOp);
        Some(gx)
    } else if op_name == "Rank" {
        // Matrix rank is a discrete, piecewise-constant function of A --
        // its gradient is zero almost everywhere and undefined at rank
        // transitions. `None` here means "no contribution", which is the
        // honest answer (and, discovered alongside the `Cond`/`LogDet`
        // fixes above, also keeps its edge out of the shape-inconsistent
        // scalar pass-through that the generic default below would
        // otherwise produce).
        None
    } else {
        // Default case: pass through gradient for unknown ops.
        // This is generally safer than returning None (zero gradient)
        // for many element-wise or simple operations.
        Some(gy)
    }
}

// a graph node in a gradient subgraph
struct Node {
    id: usize,
    topo_rank: usize,
}

impl Ord for Node {
    // Compares the ranks in topological ordering
    fn cmp(&self, other: &Self) -> Ordering {
        self.topo_rank.cmp(&other.topo_rank)
    }
}

impl PartialOrd for Node {
    #[inline]
    // Compares the ranks in topological ordering
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.topo_rank.cmp(&other.topo_rank))
    }
}

impl Eq for Node {}

impl PartialEq for Node {
    #[inline]
    fn eq(&self, other: &Node) -> bool {
        self.id == other.id
    }
}

impl<'tensor, T: Float> Tensor<'tensor, T> {
    #[inline]
    #[allow(clippy::wrong_self_convention)]
    fn to_node(&'tensor self) -> Node {
        Node {
            id: self.id,
            topo_rank: self.graph.topo_rank(self.id),
        }
    }
}

pub(crate) struct GradientMap<'graph, F: Float> {
    inner: FxHashMap<TensorID, GradientInfo<'graph, F>>,
}

impl<'graph, F: Float> GradientMap<'graph, F> {
    pub(crate) fn extract_grad(
        &mut self,
        x: impl AsRef<Tensor<'graph, F>>,
    ) -> Option<Tensor<'graph, F>> {
        if let Some(info) = self.inner.get_mut(&x.as_ref().id) {
            if info.on_backprop_path {
                if info.gradients.is_empty() {
                    // No gradients yet, create a zero gradient
                    let g = x.as_ref().graph();
                    let shape = x.as_ref().shape();
                    let data_len = shape.iter().product();
                    let zero_data = vec![F::zero(); data_len];
                    let zero_grad = Tensor::from_vec(zero_data, shape, g);
                    info.gradients.push(zero_grad);
                }
                return Some(info.gradient());
            }
        }
        // can't differentiate!
        None
    }

    #[inline]
    fn get_mut(&mut self, key: TensorID) -> &mut GradientInfo<'graph, F> {
        self.inner.get_mut(&key).expect("Operation failed")
    }

    #[inline]
    fn push_grad(&mut self, key: TensorID, grad: Tensor<'graph, F>) {
        self.inner
            .get_mut(&key)
            .expect("Operation failed")
            .gradients
            .push(grad);
    }
}

// GradientInfo is keyed by a TensorID and holds its gradient info for back-prop
struct GradientInfo<'graph, F: Float> {
    gradients: Vec<Tensor<'graph, F>>,
    on_backprop_path: bool,
}

impl<'graph, F: Float> GradientInfo<'graph, F> {
    #[inline]
    fn new(on_backprop_path: bool) -> GradientInfo<'graph, F> {
        GradientInfo {
            on_backprop_path,
            gradients: Vec::new(),
        }
    }

    #[inline]
    fn gradient(&mut self) -> Tensor<'graph, F> {
        if self.gradients.is_empty() {
            panic!("No gradients available")
        } else if self.gradients.len() > 1 {
            // the accumulated gradients are added together at this time.
            self.gradients[0] = T::add_n(self.gradients.as_slice());
        }
        self.gradients[0]
    }
}

#[inline]
#[allow(dead_code)]
fn has_child_on_path<T: Float>(
    parent: Tensor<T>,
    path: &FxHashMap<usize, GradientInfo<T>>,
) -> bool {
    let inner = parent.inner();
    for child in inner.get_backprop_inputs() {
        if path
            .get(&child.id)
            .expect("Operation failed")
            .on_backprop_path
        {
            return true;
        }
    }
    false
}

// checks `candidate` node is an xs node or not.
#[inline]
#[allow(dead_code)]
fn is_given_xs<'graph, F: Float, A>(candidate: usize, xs: &[A]) -> bool
where
    A: AsRef<Tensor<'graph, F>>,
{
    for x in xs {
        if x.as_ref().id == candidate {
            return true;
        }
    }
    false
}

// Go backward from ys and collect reachable nodes.
// Nodes between `ys` and `xs` are marked as `on_backprop_path`.
#[allow(dead_code)]
fn init_gradient_map<'graph, A, B, F: Float>(
    g: &'graph Graph<F>,
    ys: &[A],
    xs: &[B],
) -> GradientMap<'graph, F>
where
    A: AsRef<Tensor<'graph, F>>,
    B: AsRef<Tensor<'graph, F>>,
{
    let mut map = FxHashMap::<TensorID, GradientInfo<F>>::default();

    // Builds GradientInfo while performing depth-first-search.

    // dfs_stack: (node, should_visit)
    let mut dfs_stack: Vec<(TensorID, bool)> = ys.iter().map(|y| (y.as_ref().id, false)).collect();
    while let Some((curr_id, should_visit)) = dfs_stack.pop() {
        let curr_node = g.tensor(curr_id);
        if should_visit {
            let on_backprop_path = curr_node.is_differentiable()
                && (is_given_xs(curr_id, xs) || has_child_on_path(curr_node, &map));
            map.insert(curr_id, GradientInfo::new(on_backprop_path));
        } else {
            // Put self on the stack top (should visit next time)
            dfs_stack.push((curr_id, true));
            // Push children as necessary
            let curr_node = curr_node.inner();
            for child in curr_node.get_backprop_inputs() {
                let child = child.as_tensor(g);
                if let std::collections::hash_map::Entry::Vacant(e) = map.entry(child.id) {
                    if child.is_source() || !child.is_differentiable() {
                        // Add to result, but don't allow any more recursive search
                        // because there will be no `xs` nodes in this direction....
                        e.insert(GradientInfo::new(
                            child.is_differentiable() && is_given_xs(child.id, xs),
                        ));
                    } else {
                        // Recurse
                        dfs_stack.push((child.id, false));
                    }
                }
            }
        }
    }
    GradientMap { inner: map }
}
