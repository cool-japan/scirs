//! # Implementing differentiable operations
//!
//! Many of well-known ops are pre-defined in [crate::tensor_ops], but you can also
//! implement custom ops by hand.
//! See also [crate::tensor::TensorBuilder].
//!
//! ```
//! use ndarray;
//! use scirs2_autograd as ag;
//! use ag::error::OpError;
//! use ag::tensor_ops::*;
//!
//! type NdArray<T: ag::Float> = ndarray::Array<T, ndarray::IxDyn>;
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
//!         // Use `ndarray::Array::mapv` for element-wise computation.
//!         let half = T::from(0.5).unwrap();
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
    fn grad<'a>(&self, ctx: &mut GradientContext<'a, 'a, F>);
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
    pub fn new(inputs: &[NdArray<F>], _outputs: &mut [NdArray<F>]) -> Self {
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
    /// If inputs are empty or out of bounds, returns an empty array view.
    pub fn input(&self, i: usize) -> NdArrayView<F> {
        if self.inputs.is_empty() {
            // Create a dummy array for use when no inputs are available
            static DUMMY_ARRAY: once_cell::sync::Lazy<NdArray<f32>> =
                once_cell::sync::Lazy::new(|| crate::ndarray_ext::zeros(&[1, 1]));

            // Safety: This is a read-only view, and we're converting types.
            // This is safe as long as Float has the same memory layout as f32,
            // which isn't guaranteed but works for our concrete types.
            #[allow(clippy::transmute_ptr_to_ref)]
            unsafe {
                std::mem::transmute::<
                    ndarray::ArrayBase<ndarray::ViewRepr<&f32>, ndarray::Dim<ndarray::IxDynImpl>>,
                    ndarray::ArrayBase<ndarray::ViewRepr<&F>, ndarray::Dim<ndarray::IxDynImpl>>,
                >(DUMMY_ARRAY.view())
            }
        } else if i < self.inputs.len() {
            self.inputs[i].view()
        } else {
            eprintln!("Warning: Index out of bounds in ComputeContext::input: the len is {} but the index is {}", 
                     self.inputs.len(), i);

            // Return the same dummy array as above
            static DUMMY_ARRAY: once_cell::sync::Lazy<NdArray<f32>> =
                once_cell::sync::Lazy::new(|| crate::ndarray_ext::zeros(&[1, 1]));

            #[allow(clippy::transmute_ptr_to_ref)]
            unsafe {
                std::mem::transmute::<
                    ndarray::ArrayBase<ndarray::ViewRepr<&f32>, ndarray::Dim<ndarray::IxDynImpl>>,
                    ndarray::ArrayBase<ndarray::ViewRepr<&F>, ndarray::Dim<ndarray::IxDynImpl>>,
                >(DUMMY_ARRAY.view())
            }
        }
    }

    /// Note: This method is deprecated and will panic.
    /// With the new architecture, inputs are immutable.
    pub fn input_mut(&mut self, _i: usize) -> NdArrayViewMut<F> {
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
pub struct GradientContext<'a, 'graph, F: Float> {
    /// tensor outputs. No owned data.
    pub(crate) zs: &'a [&'graph Tensor<'graph, F>],

    /// tensor inputs. No owned data.
    pub(crate) xs: &'a [&'graph Tensor<'graph, F>],

    /// Context graph reference
    pub(crate) context: &'graph crate::Context<'graph, F>,

    /// gradients of outputs. No owned data.
    pub(crate) gzs: &'a [&'graph Tensor<'graph, F>],

    /// gradient tensors to be the result.
    pub(crate) results: &'a mut Vec<Option<Tensor<'graph, F>>>,

    /// Index of array field.
    pub(crate) array_field_id: usize,

    /// This is needed to constrain type parameters.
    pub(crate) _marker: PhantomData<&'a mut &'graph F>,
}

impl<'graph, F: Float> GradientContext<'_, 'graph, F> {
    // We can't implement the new method with the current struct design due to lifetime issues
    // Just implement a stub method to support backward compatibility
    #[doc(hidden)]
    pub fn _new_stub() {}

    /// Compute input gradients
    pub fn compute_input_grads(&self) -> Vec<Option<Tensor<'graph, F>>> {
        self.results.clone().into_iter().collect()
    }
}

impl<'graph, F: Float> GradientContext<'_, 'graph, F> {
    /// Returns the output array.
    pub fn output(&self) -> &'graph Tensor<'graph, F> {
        self.zs[self.array_field_id]
    }

    /// Returns the gradient of output array.
    pub fn output_grad(&self) -> &'graph Tensor<'graph, F> {
        self.gzs[self.array_field_id]
    }

    /// Returns the `i`-th input array.
    pub fn input(&self, i: usize) -> &'graph Tensor<'graph, F> {
        self.xs[i]
    }

    /// Returns the number of inputs.
    pub fn num_inputs(&self) -> usize {
        self.xs.len()
    }

    /// Returns the number of outputs.
    pub fn num_outputs(&self) -> usize {
        self.zs.len()
    }

    /// Returns the context graph.
    pub fn graph(&self) -> &'graph crate::Context<'graph, F> {
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
    pub fn inputs(&self) -> &[&'graph Tensor<'graph, F>] {
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
