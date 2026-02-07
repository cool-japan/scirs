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
use std::any::type_name;
use std::marker::PhantomData;

pub use crate::error::OpError;
use crate::ndarray_ext::{NdArrayView, NdArrayViewMut};
use crate::smallvec::SmallVec as RawSmallVec;
use crate::tensor::Tensor;
use crate::{Float, NdArray};

pub(crate) const DEFAULT_NUM_EDGES: usize = 2;

pub(crate) type SmallVec<T> = RawSmallVec<[T; DEFAULT_NUM_EDGES]>;

/// Trait for tensor operations. `Tensor` structs wrap this.
pub trait Op<F: Float> {
    /// Name of this op
    fn name(&self) -> &'static str {
        type_name::<Self>()
    }

    /// Runs this op with `ComputeContext`.
    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError>;

    /// Returns gradients for input nodes by use of output's gradients etc.
    fn grad(&self, ctx: &mut GradientContext<F>);
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
/// Follows the upstream rust-autograd design with a single lifetime parameter.
/// The "steal-call-restore" pattern in `compute_input_grads` allows proper
/// polymorphic dispatch to `Op::grad()` implementations.
pub struct GradientContext<'graph, F: Float> {
    /// The output tensor of this op.
    pub(crate) y: Tensor<'graph, F>,
    /// The gradient of the output tensor.
    pub(crate) gy: Tensor<'graph, F>,
    /// Reference to the computation graph.
    pub(crate) graph: &'graph crate::Graph<F>,
    /// Accumulated gradient results for each input.
    pub(crate) gxs: SmallVec<Option<Tensor<'graph, F>>>,
}

impl<'graph, F: Float> GradientContext<'graph, F> {
    /// Create a new GradientContext.
    pub(crate) fn new(
        y: Tensor<'graph, F>,
        gy: Tensor<'graph, F>,
        graph: &'graph crate::Graph<F>,
        num_inputs: usize,
    ) -> Self {
        let mut gxs = SmallVec::new();
        for _ in 0..num_inputs {
            gxs.push(None);
        }
        GradientContext { y, gy, graph, gxs }
    }

    /// Steal-call-restore: take the op out, call Op::grad(), put it back.
    pub(crate) fn compute_input_grads(mut self) -> SmallVec<Option<Tensor<'graph, F>>> {
        let id = self.y.id;
        // Steal the op from the tensor node (RefMut dropped at semicolon)
        let stolen = self.graph.access_inner_mut(id).op.take().unwrap();
        // Call Op::grad — no outstanding borrows on node_set at this point
        stolen.grad(&mut self);
        // Restore the op
        let mut slot = Some(stolen);
        std::mem::swap(&mut self.graph.access_inner_mut(id).op, &mut slot);
        self.gxs
    }

    /// Returns the output tensor.
    pub fn output(&self) -> Tensor<'graph, F> {
        self.y
    }

    /// Returns the gradient of the output tensor.
    pub fn output_grad(&self) -> Tensor<'graph, F> {
        self.gy
    }

    /// Returns the `i`-th input tensor.
    pub fn input(&self, i: usize) -> Tensor<'graph, F> {
        self.y.get_backprop_input(i)
    }

    /// Returns the number of inputs.
    pub fn num_inputs(&self) -> usize {
        self.y.num_backprop_inputs()
    }

    /// Returns the number of outputs.
    pub fn num_outputs(&self) -> usize {
        1
    }

    /// Returns a reference to the computation graph.
    pub fn graph(&self) -> &'graph crate::Graph<F> {
        self.graph
    }

    /// Appends a gradient for the input indexed by `i`.
    pub fn append_input_grad(&mut self, i: usize, gx: Option<Tensor<'graph, F>>) {
        while self.gxs.len() <= i {
            self.gxs.push(None);
        }
        self.gxs[i] = gx;
    }

    /// Appends a gradient for input 0.
    pub fn append_input_grad_by_ref(&mut self, gx: Option<&Tensor<'graph, F>>) {
        self.append_input_grad(0, gx.copied());
    }

    /// Appends a gradient for input 0.
    pub fn append_input_grad_0(&mut self, gx: Option<Tensor<'graph, F>>) {
        self.append_input_grad(0, gx);
    }

    /// Returns all input tensors as a vec.
    pub fn inputs(&self) -> Vec<Tensor<'graph, F>> {
        let n = self.num_inputs();
        (0..n).map(|i| self.input(i)).collect()
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
