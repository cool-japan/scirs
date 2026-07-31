use crate::graph::TensorID;
use crate::tensor::Tensor;
use crate::tensor_ops as T;
use crate::Float;
use crate::FxHashMap;
use crate::Graph;
use std::any::TypeId;
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
                let gy = y_grad_info.gradient();
                let y_tensor = g.tensor(y.id);

                // `concrete_type_id()` returns a `TypeId`; the temporary `Ref` on the node
                // is released at the end of this statement so that the VJP below is free
                // to take a mutable borrow of the graph.
                //
                // NOTE: this is deliberately `concrete_type_id()`, NOT `name()`. `name()`
                // can be overridden to return an arbitrary caller-supplied string (see
                // `CustomGradientWrapper::name`, which forwards the *user's* name), and
                // several unrelated internal op structs intentionally share the same
                // `name()` string across modules. Keying the override table below on
                // `name()` would let a caller-supplied name select another op's VJP;
                // `TypeId` identifies the actual concrete Rust type and cannot be spoofed.
                let op_type_id = y_tensor.inner().get_op().concrete_type_id();

                let num_inputs = y_tensor.num_backprop_inputs();

                // 1) A small, explicit override table, consulted FIRST. It holds only
                //    rules that must NOT come from `Op::grad`: gradient-internal ops
                //    whose backward has to stay higher-order safe, and a handful of ops
                //    whose `Op::grad` is a known stub.
                // 2) Everything else is dispatched to the op's own `Op::grad`.
                match compute_override_grads(op_type_id, num_inputs, y_tensor, gy, g) {
                    Some(gxs) => gxs,
                    None => compute_grads_via_op(y_tensor, gy, num_inputs, g),
                }
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

/// Explicit override table for the backward pass, consulted *before* `Op::grad`.
///
/// Returns `Some(gxs)` — one entry per backprop input — when `op_type_id` names an op
/// whose backward must **not** come from its `Op::grad` implementation, and `None` to
/// fall through to `Op::grad`.
///
/// # Why this is keyed on `TypeId`, not `Op::name()`
///
/// [`crate::op::Op::name`] can be overridden to return an arbitrary caller-supplied
/// string: `CustomGradientWrapper::name` (in `custom_gradient.rs`) forwards the *user's*
/// name verbatim, and `custom_unary_op`'s `ClosureOp` takes its name as a caller-supplied
/// `&'static str` too. Several unrelated internal op structs also intentionally share the
/// same `name()` string across modules (e.g. two distinct `Cholesky`-family ops in
/// different decomposition backends). If this table matched on strings, a user op named
/// e.g. `"Cond"` would silently receive `CondOp`'s gradient rule instead of its own
/// (`CustomGradientOp::backward`) — a caller-supplied string would be selecting another
/// op's VJP. [`std::any::TypeId`] identifies the actual concrete Rust type behind
/// `&dyn Op<F>` via [`crate::op::Op::concrete_type_id`] and cannot be spoofed this way:
/// it is a default trait method with no per-op override, so it always reflects reality.
///
/// Only two kinds of rule belong here:
///
/// 1. Gradient-internal ops created by a previous differentiation pass
///    (`MaybeReduceSum`, `MaybeBroadcast`, `ReduceSumToScalarGrad`, `ReduceGradCommon`)
///    plus the hand-written matrix backward ops (`TraceOp`, `CondOp`, `LogDetOp`,
///    `RankOp`).  It is critical for correctness with higher-order differentiation
///    (e.g. Hessian) that gradient nodes do NOT introduce spurious dependencies on the
///    original forward-pass tensors.  For example, a reduction gradient must NOT create
///    `x_tensor * 0 + gy`, because `x_tensor` depends on the variable we differentiate,
///    causing the second pass to re-traverse and double-count the original graph.
/// 2. Ops whose `Op::grad` is a known stub, where an explicit rule here keeps the live
///    behaviour honest until the op's own VJP is implemented.
///
/// Everything else is dispatched to `Op::grad` by [`compute_grads_via_op`].
fn compute_override_grads<'graph, F: Float>(
    op_type_id: TypeId,
    num_inputs: usize,
    y_tensor: Tensor<'graph, F>,
    gy: Tensor<'graph, F>,
    g: &'graph Graph<F>,
) -> Option<Vec<Option<Tensor<'graph, F>>>> {
    // Helper: gradient for input 0 only, `None` for the remaining (shape/axis) inputs.
    let first_only = |gx: Option<Tensor<'graph, F>>| {
        let mut out = vec![None; num_inputs];
        if let Some(slot) = out.first_mut() {
            *slot = gx;
        }
        Some(out)
    };

    // `symbolic_backend::EmlOp` / `tape::eml_tape::EmlElementWiseOp` only exist behind
    // the `symbolic` feature (see their `#[cfg(feature = "symbolic")]`-gated `mod`
    // declarations in lib.rs / tape/mod.rs), so the `TypeId::of::<...>()` comparisons
    // below must be feature-gated too -- `cfg!(feature = "symbolic")` alone would not
    // help, since it still requires the type *path* to resolve for both branches.
    // Without the feature, these types cannot exist anywhere in the graph, so the
    // matches are simply hard-coded to `false`.
    #[cfg(feature = "symbolic")]
    let is_eml_op = op_type_id == TypeId::of::<crate::symbolic_backend::EmlOp>();
    #[cfg(not(feature = "symbolic"))]
    let is_eml_op = false;

    #[cfg(feature = "symbolic")]
    let is_eml_elementwise_op =
        op_type_id == TypeId::of::<crate::tape::eml_tape::EmlElementWiseOp>();
    #[cfg(not(feature = "symbolic"))]
    let is_eml_elementwise_op = false;

    // -----------------------------------------------------------
    // 1) Gradient-internal ops created by the first differentiation pass.
    // -----------------------------------------------------------
    if op_type_id == TypeId::of::<crate::tensor_ops::binary_ops::MaybeReduceSum>() {
        // MaybeReduceSum(gradient, target_shape) conditionally reduces `gradient` to
        // `target_shape`.  Input 0 gets `gy` (pass-through; MaybeBroadcast handles the
        // shape at compute time), input 1 is a shape tensor and has no gradient.
        first_only(Some(gy))
    } else if op_type_id == TypeId::of::<crate::tensor_ops::binary_ops::MaybeBroadcast>() {
        // Backward of a broadcast is a reduction back to the input shape.
        let x_tensor = y_tensor.get_backprop_input(0);
        let x_shape = T::shape(x_tensor);
        let reduced = crate::tensor_ops::binary_ops::maybe_reduce(&x_shape, &gy, g);
        first_only(Some(reduced))
    } else if op_type_id == TypeId::of::<crate::tensor_ops::reduction_ops::ReduceSumToScalarGrad>()
        || op_type_id == TypeId::of::<crate::tensor_ops::reduction_ops::ReduceGradCommon>()
    {
        // Backward-of-a-backward: both broadcast a gradient over reduced axes, so their
        // own backward is a reduction, delivered here by passing `gy` through for the
        // data input (the shape/axis inputs are non-differentiable).
        first_only(Some(gy))
    }
    // -----------------------------------------------------------
    // 2) Matrix utilities with hand-written backward ops.
    // -----------------------------------------------------------
    else if op_type_id == TypeId::of::<crate::tensor_ops::TraceOp>() {
        // trace: R^{n x n} -> R.  The VJP w.r.t. the input matrix for an upstream scalar
        // cotangent `gy` is `gy * I_n`, NOT the scalar `gy` passed through unchanged
        // (which emits a 0-d gradient that matrix-valued backward ops cannot consume).
        let x_input = y_tensor.get_backprop_input(0);
        let gx = crate::tensor::Tensor::builder(g)
            .append_input(x_input, false)
            .append_input(gy, false)
            .build(crate::tensor_ops::TraceBackwardOp);
        first_only(Some(gx))
    } else if op_type_id == TypeId::of::<crate::tensor_ops::CondOp>() {
        // Condition number.  Only the 2-norm (spectral) variant has an analytic gradient
        // implemented -- see `CondOp::two_norm_gradient` in tensor_ops/numerical_props.rs
        // for the derivation via SVD perturbation theory.  The 1-norm / inf-norm /
        // Frobenius variants are honestly left non-differentiable (`None`) rather than
        // fabricating a plausible-looking but wrong gradient, so recover which variant the
        // forward op used via downcast before deciding. The downcast below is guaranteed
        // to succeed: the `TypeId` check above already proved the concrete type.
        let inner = y_tensor.inner();
        let is_two_norm = inner
            .get_op()
            .as_any()
            .and_then(|any| any.downcast_ref::<crate::tensor_ops::CondOp>())
            .map(|op| matches!(op.p, crate::tensor_ops::ConditionType::Two));
        drop(inner);

        if is_two_norm == Some(true) {
            let x_tensor = y_tensor.get_backprop_input(0);
            let gx = crate::tensor::Tensor::builder(g)
                .append_input(x_tensor, false)
                .append_input(gy, false)
                .build(crate::tensor_ops::CondTwoNormBackwardOp);
            first_only(Some(gx))
        } else {
            first_only(None)
        }
    } else if op_type_id == TypeId::of::<crate::tensor_ops::LogDetOp>() {
        // d log|det(A)| = tr(A^-1 dA), i.e. gradient (A^-1)^T * gy.  Delivered through a
        // dedicated backward op so that the result is matrix-shaped and can be summed
        // against other matrix-shaped gradients feeding the same input.
        let x_tensor = y_tensor.get_backprop_input(0);
        let gx = crate::tensor::Tensor::builder(g)
            .append_input(x_tensor, false)
            .append_input(gy, false)
            .build(crate::tensor_ops::LogDetBackwardOp);
        first_only(Some(gx))
    } else if op_type_id == TypeId::of::<crate::tensor_ops::RankOp<F>>() {
        // Matrix rank is a discrete, piecewise-constant function of A -- its gradient is
        // zero almost everywhere and undefined at rank transitions.  `None` is the honest
        // answer and also keeps its edge out of any shape-inconsistent accumulation.
        Some(vec![None; num_inputs])
    }
    // -----------------------------------------------------------
    // 3) Checkpointing: forward is the identity, so the gradient passes through.
    //    (`CheckpointOp::grad` is a stub; this rule is the correct behaviour.
    //    `SmartCheckpointOp` is no longer listed here — its own `Op::grad` implements
    //    exactly this pass-through, and `ConditionalOp` now routes the cotangent to the
    //    branch that actually executed instead of to both.)
    // -----------------------------------------------------------
    else if op_type_id == TypeId::of::<crate::tensor_ops::CheckpointOp>() {
        first_only(Some(gy))
    }
    // -----------------------------------------------------------
    // 5) Symbolic backends.  `EmlOp::grad` / `EmlElementWiseOp::grad` are empty; the real
    //    rule needs the symbolic derivative of the lowered op, built here.
    // -----------------------------------------------------------
    else if is_eml_op {
        #[cfg(feature = "symbolic")]
        {
            use crate::symbolic_backend::EmlOp;
            use scirs2_symbolic::eml::grad as sym_grad;
            use std::sync::Arc;

            // Hold the inner `Ref` in a binding so the borrow lives long enough for the
            // downcast chain, and release it before building new tensors. The downcast
            // is guaranteed to succeed: the `TypeId` check above already proved it.
            let inner = y_tensor.inner();
            let eml_op_data: Option<(Arc<scirs2_symbolic::eml::LoweredOp>, usize)> = inner
                .get_op()
                .as_any()
                .and_then(|any| any.downcast_ref::<EmlOp>())
                .map(|eml| (Arc::clone(&eml.op), eml.num_vars));
            drop(inner);

            let (op_arc, num_vars) = eml_op_data?;
            let mut out = Vec::with_capacity(num_inputs);
            for i in 0..num_inputs {
                let g_lowered = sym_grad(&op_arc, i);
                // Build a new EmlOp tensor for d(f)/d(Var(i)) using the same original
                // inputs -- evaluates correctly with placeholder feeds.
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
                out.push(Some(T::mul(gy, gval_tensor)));
            }
            Some(out)
        }
        #[cfg(not(feature = "symbolic"))]
        {
            // Without the symbolic feature, EmlOp should not appear in the graph, but if
            // it does (e.g. from a pre-compiled dep), it carries no usable derivative.
            let _ = (y_tensor, gy, g);
            Some(vec![None; num_inputs])
        }
    } else if is_eml_elementwise_op {
        // Element-wise application of a LoweredOp to a 1-D input:
        //   gx[i] = gy[i] * d(op)/d(Var(0))|_{x[i]}
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

            let deriv_op = maybe_deriv?;
            let x_input = y_tensor.get_backprop_input(0);
            let deriv_tensor = crate::tensor::Tensor::builder(g)
                .append_input(x_input, false)
                .build(EmlElementWiseOp { op: deriv_op });
            first_only(Some(T::mul(gy, deriv_tensor)))
        }
        #[cfg(not(feature = "symbolic"))]
        {
            let _ = (y_tensor, gy, g);
            Some(vec![None; num_inputs])
        }
    } else {
        // No override: dispatch to `Op::grad`.
        None
    }
}

/// Builds the gradient of every backprop input of `y_tensor` from the op's own
/// `Op::grad` implementation.
///
/// The node's op is reached through a cloned `Rc` handle so that no `RefCell` borrow on
/// the node set is held while `Op::grad` installs new nodes into the same graph. The node
/// keeps its own op throughout, which matters because several `Op::grad` implementations
/// evaluate `ctx.output()` — i.e. the very node being differentiated.
///
/// Inputs the op does not produce a gradient for stay `None`.  That is deliberate: an
/// absent gradient is loud (`grad()` substitutes an explicit zero and higher-level code
/// can detect a non-differentiable path), whereas the pass-through default this replaced
/// silently pretended that every unrecognised op was the identity function.
fn compute_grads_via_op<'graph, F: Float>(
    y_tensor: Tensor<'graph, F>,
    gy: Tensor<'graph, F>,
    num_inputs: usize,
    g: &'graph Graph<F>,
) -> Vec<Option<Tensor<'graph, F>>> {
    let cloned = g.access_inner(y_tensor.id).clone_op();
    let op = match cloned {
        Some(op) => op,
        // The node has no op (a placeholder-like node): nothing to differentiate.
        None => return vec![None; num_inputs],
    };

    let xs_owned: Vec<Tensor<'graph, F>> = (0..num_inputs)
        .map(|i| y_tensor.get_backprop_input(i))
        .collect();
    let zs_owned = [y_tensor];
    let gzs_owned = [gy];

    let xs_refs: Vec<&Tensor<'graph, F>> = xs_owned.iter().collect();
    let zs_refs: Vec<&Tensor<'graph, F>> = zs_owned.iter().collect();
    let gzs_refs: Vec<&Tensor<'graph, F>> = gzs_owned.iter().collect();

    let mut results: Vec<Option<Tensor<'graph, F>>> = Vec::with_capacity(num_inputs);
    {
        let mut ctx =
            crate::op::GradientContext::new(&zs_refs, &xs_refs, &gzs_refs, g, &mut results);
        op.grad(&mut ctx);
    }

    results.resize(num_inputs, None);
    results
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
