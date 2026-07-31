//! # Implementing differentiable operations
//!
//! Many of well-known ops are pre-defined in [crate::tensor_ops], but you can also
//! implement custom ops by hand.
//! See also [crate::tensor::TensorBuilder].
//!
//! ```
//! use scirs2_core::ndarray;
//! use scirs2_autograd as ag;
//! use ag::error::OpError;
//! use ag::tensor_ops::*;
//!
//! type NdArray<T: ag::Float> = scirs2_core::ndarray::Array<T, scirs2_core::ndarray::IxDyn>;
//!
//! // Implements `Op` trait for `Sigmoid`.
//! struct Sigmoid;
//!
//! impl<T: ag::Float> ag::op::Op<T> for Sigmoid {
//!     fn compute(
//!         &self,
//!         ctx: &mut ag::op::ComputeContext<T>,
//!     ) -> Result<(), OpError> {
//!         let x: &ag::NdArrayView<_> = &ctx.input(0);
//!         // Use `scirs2_core::ndarray::Array::mapv` for element-wise computation.
//!         let half = T::from(0.5).expect("Operation failed");
//!         let y = x.mapv(move |a| ((a * half).tanh() * half) + half);
//!         ctx.append_output(y);
//!         Ok(())
//!     }
//!
//!     fn grad(&self, ctx: &mut ag::op::GradientContext<T>) {
//!         // gradient of the output of Sigmoid
//!         let gy = ctx.output_grad();
//!         let y = ctx.output();
//!         // gradient of the input of Sigmoid
//!         let gx = gy * (y - square(y));
//!         ctx.append_input_grad(0, Some(gx));
//!     }
//! }
//!
//! // `sigmoid` function for end-user.
//! fn sigmoid<'graph, F: ag::Float>(x: &ag::Tensor<'graph, F>, g: &'graph ag::Context<F>)
//! -> ag::Tensor<'graph, F> {
//!     ag::Tensor::builder(g)
//!            .append_input(x, false)
//!            .build(Sigmoid)
//! }
//! ```
//!
use std::any::{type_name, TypeId};
use std::marker::PhantomData;

pub use crate::error::OpError;
use crate::ndarray_ext::{NdArrayView, NdArrayViewMut};
use crate::tensor::Tensor;
use crate::{Float, NdArray};

/// Trait for tensor operations. `Tensor` structs wrap this.
///
/// The `'static` supertrait bound is not new in practice: the only place an `Op` is ever
/// installed into a graph ([`crate::tensor::TensorBuilder::build`]) already requires
/// `O: Op<F> + 'static`, and every node stores its op as `Rc<dyn Op<F>>` (implicitly
/// `+ 'static`). Declaring the bound on the trait itself just makes that existing fact
/// available to default methods such as [`Op::concrete_type_id`].
pub trait Op<F: Float>: 'static {
    /// Name of this op
    fn name(&self) -> &'static str {
        type_name::<Self>()
    }

    /// Runs this op with `ComputeContext`.
    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError>;

    /// Returns gradients for input nodes by use of output's gradients etc.
    ///
    /// This is the **live** backward pass: the crate-private
    /// `gradient::compute_gradients` builds a
    /// [`GradientContext`] for every node on the backprop path and calls this method.
    /// An implementation must call [`GradientContext::append_input_grad`] once per
    /// backprop input; inputs that are genuinely non-differentiable (shapes, axes,
    /// indices, ...) must be given an explicit `None` rather than being left out.
    ///
    /// The two lifetimes are deliberately independent: `'a` is the (short) borrow of the
    /// per-node scratch buffers owned by the backprop loop, while `'graph` is the lifetime
    /// of the computation graph that the newly-built gradient tensors belong to.
    fn grad<'a, 'graph>(&self, ctx: &mut GradientContext<'a, 'graph, F>);

    /// Returns self as Any for downcasting. Default returns None.
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        None
    }

    /// Returns the `TypeId` of the concrete Rust type implementing this op.
    ///
    /// [`Op::name`] is **not** a safe way to identify a specific built-in op and must
    /// never be used to select behaviour (e.g. which VJP to run): it can be overridden
    /// to return an arbitrary caller-supplied string (see
    /// [`crate::custom_gradient::CustomGradientOp::name`], which forwards the *user's*
    /// name verbatim through `CustomGradientWrapper::name`), and several unrelated
    /// internal op structs intentionally share the same `name()` string across modules
    /// (for example two distinct `Cholesky`-family ops). A caller-supplied name must
    /// never be able to select another op's gradient rule.
    ///
    /// This method is the safe replacement: it is a default method with no per-op
    /// override, so it always reflects the *actual* concrete type behind `&dyn Op<F>`
    /// regardless of anything the op chooses to return from `name()`. Code that must
    /// recognise a specific built-in op type (such as the backward-pass override table
    /// in the crate-private `gradient` module) should compare `concrete_type_id()` against
    /// `TypeId::of::<TheConcreteStruct>()`, never `name()` strings.
    fn concrete_type_id(&self) -> TypeId {
        TypeId::of::<Self>()
    }
}

#[allow(dead_code)]
pub(crate) enum OpInput<'graph, F: Float> {
    Variable(crate::variable::VariableID),
    NonVariable(usize, &'graph Tensor<'graph, F>),
}

/// Variable or non-variable tensor input.
#[allow(dead_code)]
pub(crate) struct OpInputGetter<'a, F: Float> {
    f: F,
    _marker: PhantomData<&'a ()>,
}

impl<F: Float> OpInputGetter<'_, F> {
    #[allow(dead_code)]
    pub fn new(_: F) -> Self {
        Self {
            f: F::zero(),
            _marker: PhantomData,
        }
    }
}

impl<'a, 'graph, F: Float> From<&'a OpInput<'graph, F>> for OpInputGetter<'a, F> {
    fn from(x: &'a OpInput<'graph, F>) -> Self {
        let _ = x;
        Self {
            f: F::zero(),
            _marker: PhantomData,
        }
    }
}

/// Context given to `Op::compute`.
pub struct ComputeContext<F: Float> {
    pub(crate) inputs: Vec<NdArray<F>>,
    pub(crate) outputs: Vec<NdArray<F>>,
}

impl<F: Float> ComputeContext<F> {
    /// Creates new ComputeContext.
    pub fn new(inputs: &[NdArray<F>], outputs: &mut [NdArray<F>]) -> Self {
        // Clone all inputs to own the data
        let input_arrays = inputs.to_vec();
        Self {
            inputs: input_arrays,
            outputs: Vec::new(),
        }
    }

    /// Creates a new ComputeContext with prepared inputs.
    pub fn with_inputs(input_arrays: Vec<NdArray<F>>) -> Self {
        Self {
            inputs: input_arrays,
            outputs: Vec::new(),
        }
    }

    /// Returns `i`-th input array.
    /// If index is out of bounds, returns an empty scalar array.
    /// This can happen when operations are created dynamically during gradient computation.
    pub fn input(&self, i: usize) -> NdArrayView<F> {
        if i >= self.inputs.len() {
            // Return an empty scalar instead of panicking or warning
            // This handles the case where some operations may not have all inputs during evaluation
            static DUMMY_SCALAR: once_cell::sync::Lazy<NdArray<f32>> =
                once_cell::sync::Lazy::new(|| crate::ndarray_ext::zeros::<f32>(&[]));

            #[allow(clippy::transmute_ptr_to_ref)]
            unsafe {
                std::mem::transmute::<
                    scirs2_core::ndarray::ArrayBase<
                        scirs2_core::ndarray::ViewRepr<&f32>,
                        scirs2_core::ndarray::Dim<scirs2_core::ndarray::IxDynImpl>,
                    >,
                    scirs2_core::ndarray::ArrayBase<
                        scirs2_core::ndarray::ViewRepr<&F>,
                        scirs2_core::ndarray::Dim<scirs2_core::ndarray::IxDynImpl>,
                    >,
                >(DUMMY_SCALAR.view())
            }
        } else {
            self.inputs[i].view()
        }
    }

    /// Note: This method is deprecated and will panic.
    /// With the new architecture, inputs are immutable.
    pub fn input_mut(&mut self, i: usize) -> NdArrayViewMut<'_, F> {
        let _ = i; // Suppress unused parameter warning
        panic!("input_mut is not supported in the new ComputeContext implementation");
    }

    /// Returns all input array views.
    pub fn inputs(&self) -> Vec<NdArrayView<F>> {
        self.inputs.iter().map(|arr| arr.view()).collect()
    }

    /// Appends an output array.
    pub fn append_output<A>(&mut self, output: A)
    where
        A: Into<NdArray<F>>,
    {
        self.outputs.push(output.into());
    }

    /// Get all outputs
    pub fn get_outputs(&self) -> &[NdArray<F>] {
        &self.outputs
    }
}

/// Context given to `Op::grad`.
///
/// `'a` borrows the per-node scratch buffers owned by the backprop loop in
/// the crate-private `gradient` module; `'graph` is the lifetime of the computation graph the newly-built
/// gradient tensors belong to. They are intentionally **not** unified — the scratch
/// buffers are locals of the backprop loop and can never live as long as the graph
/// borrow, which is exactly what previously made this type impossible to construct.
pub struct GradientContext<'a, 'graph, F: Float> {
    /// tensor outputs. No owned data.
    pub(crate) zs: &'a [&'a Tensor<'graph, F>],

    /// tensor inputs. No owned data.
    pub(crate) xs: &'a [&'a Tensor<'graph, F>],

    /// Graph the gradient sub-graph is built into.
    pub(crate) context: &'graph crate::graph::Graph<F>,

    /// gradients of outputs. No owned data.
    pub(crate) gzs: &'a [&'a Tensor<'graph, F>],

    /// gradient tensors to be the result.
    pub(crate) results: &'a mut Vec<Option<Tensor<'graph, F>>>,

    /// Index of array field.
    pub(crate) array_field_id: usize,

    /// This is needed to constrain type parameters.
    pub(crate) _marker: PhantomData<&'a mut &'graph F>,
}

impl<'a, 'graph, F: Float> GradientContext<'a, 'graph, F> {
    /// Creates a context for a single backprop node.
    ///
    /// * `zs`  — the outputs of the node being differentiated
    /// * `xs`  — the node's *backprop* inputs, in backprop-input order
    /// * `gzs` — the upstream cotangents, one per entry of `zs`
    /// * `context` — the graph the gradient sub-graph is built into
    /// * `results` — output buffer, indexed by backprop-input position
    pub(crate) fn new(
        zs: &'a [&'a Tensor<'graph, F>],
        xs: &'a [&'a Tensor<'graph, F>],
        gzs: &'a [&'a Tensor<'graph, F>],
        context: &'graph crate::graph::Graph<F>,
        results: &'a mut Vec<Option<Tensor<'graph, F>>>,
    ) -> Self {
        GradientContext {
            zs,
            xs,
            gzs,
            context,
            results,
            array_field_id: 0,
            _marker: PhantomData,
        }
    }

    /// Compute input gradients
    pub fn compute_input_grads(&self) -> Vec<Option<Tensor<'graph, F>>> {
        self.results.clone().into_iter().collect()
    }
}

impl<'a, 'graph, F: Float> GradientContext<'a, 'graph, F> {
    /// Returns the output array.
    pub fn output(&self) -> &'a Tensor<'graph, F> {
        self.zs[self.array_field_id]
    }

    /// Returns the gradient of output array.
    pub fn output_grad(&self) -> &'a Tensor<'graph, F> {
        self.gzs[self.array_field_id]
    }

    /// Returns the `i`-th input array.
    pub fn input(&self, i: usize) -> &'a Tensor<'graph, F> {
        self.xs[i]
    }

    /// Returns the `i`-th input array, or `None` when `i` is out of range.
    ///
    /// Prefer this over [`Self::input`] in `Op::grad` implementations whose op can be
    /// built with a variable number of backprop inputs.
    pub fn try_input(&self, i: usize) -> Option<&'a Tensor<'graph, F>> {
        self.xs.get(i).copied()
    }

    /// Returns the number of inputs.
    pub fn num_inputs(&self) -> usize {
        self.xs.len()
    }

    /// Returns the number of outputs.
    pub fn num_outputs(&self) -> usize {
        self.zs.len()
    }

    /// Returns the graph the gradient sub-graph is built into.
    pub fn graph(&self) -> &'graph crate::graph::Graph<F> {
        self.context
    }

    /// Appends a gradient for the input indexed by `i`.
    pub fn append_input_grad(&mut self, i: usize, gx: Option<Tensor<'graph, F>>) {
        for _ in self.results.len()..=i {
            self.results.push(None);
        }
        self.results[i] = gx;
    }

    /// Appends a gradient for the input indexed by 0.
    /// Short-hand for `append_input_grad(0, gx)`.
    pub fn append_input_grad_by_ref(&mut self, gx: Option<&Tensor<'graph, F>>) {
        self.append_input_grad(0, gx.cloned());
    }

    /// Appends a gradient for the input indexed by 0.
    /// Short-hand for `append_input_grad(0, gx)`.
    pub fn append_input_grad_0(&mut self, gx: Option<Tensor<'graph, F>>) {
        self.append_input_grad(0, gx);
    }

    /// Returns all input tensors.
    pub fn inputs(&self) -> &'a [&'a Tensor<'graph, F>] {
        self.xs
    }
}

/// Output from op.
#[derive(Clone)]
#[allow(dead_code)]
pub struct OpOutput<F: Float> {
    pub(crate) output: NdArray<F>,
}

impl<F: Float> OpOutput<F> {
    #[allow(dead_code)]
    pub(crate) fn new(output: NdArray<F>) -> Self {
        Self { output }
    }
}
