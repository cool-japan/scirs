//! Adam optimizer

use crate::optimizers::Optimizer;
use crate::tensor::Tensor;
use crate::tensor_ops::gradient_descent_ops::adam;
use crate::variable::VariableID;
use crate::{Context, Float, VariableEnvironment};

/// Adam optimizer
///
/// This implementation is based on <https://arxiv.org/abs/1412.6980v8>.
///
/// # Example
///
/// ```no_run
/// use scirs2_autograd as ag;
/// use ag::prelude::*;
/// use ag::optimizers::{Adam, Optimizer};
/// use ag::variable::NamespaceTrait;
/// use ag::tensor_ops as T;
///
/// // Define parameters to optimize
/// let mut env = ag::VariableEnvironment::new();
/// let mut rng = ag::ndarray_ext::ArrayRng::<f32>::default();
///
/// let w = env.slot().set(rng.glorot_uniform(&[28 * 28, 10]));
/// let b = env.slot().set(ag::ndarray_ext::zeros(&[1, 10]));
///
/// // Create Adam optimizer with default parameters
/// // State arrays (momentum buffers) are stored in the "my_adam" namespace
/// let adam = Adam::default("my_adam", env.default_namespace().current_var_ids(), &mut env);
///
/// // Training loop
/// for epoch in 0..100 {
///     env.run(|ctx| {
///         let w = ctx.variable(w);
///         let b = ctx.variable(b);
///
///         // Forward pass (example)
///         // let predictions = ...;
///         // let loss = ...;
///
///         // Compute gradients
///         // let grads = T::grad(&[loss], &[w, b]);
///
///         // Update parameters (this now actually updates variables!)
///         // adam.update(&[w, b], &grads, ctx, ag::Feeder::new());
///     });
/// }
/// ```
///
/// # Manual Update Control
///
/// For advanced use cases where you need explicit control over when updates are applied:
///
/// ```no_run
/// # use scirs2_autograd as ag;
/// # use ag::optimizers::{Adam, Optimizer};
/// # let mut env = ag::VariableEnvironment::new();
/// # let w = env.slot().set(ag::ndarray_ext::zeros::<f32>(&[10]));
/// # let adam = Adam::default("my_adam", vec![w], &mut env);
/// env.run(|ctx| {
///     # let w_tensor = ctx.variable(w);
///     # let grads = vec![w_tensor]; // Dummy gradient
///     // Get update tensors without applying them
///     let update_tensors = adam.get_update_tensors(&[w_tensor], &grads, ctx);
///
///     // Evaluate updates
///     let results = ctx.evaluator().extend(&update_tensors).run();
///     let values: Vec<_> = results.into_iter().map(|r| r.expect("Evaluation failed")).collect();
///
///     // Apply updates manually
///     Adam::apply_update_tensors(&[w_tensor], &values, ctx.env());
/// });
/// ```
///
/// See also <https://github.com/raskr/rust-autograd/blob/master/examples/>
pub struct Adam<F: Float> {
    pub alpha: F,
    pub eps: F,
    pub b1: F,
    pub b2: F,
    pub adam_namespace_id: &'static str,
}

impl<F: Float> Adam<F> {
    /// Instantiates `Adam` optimizer with the recommended parameters in the original paper.
    pub fn default(
        unique_namespace_id: &'static str,
        var_id_list: impl IntoIterator<Item = VariableID>,
        env_handle: &mut VariableEnvironment<F>,
    ) -> Adam<F> {
        Adam::new(
            F::from(0.001).expect("Failed to convert constant to float"),
            F::from(1e-08).expect("Failed to convert constant to float"),
            F::from(0.9).expect("Failed to convert constant to float"),
            F::from(0.999).expect("Failed to convert constant to float"),
            var_id_list,
            env_handle,
            unique_namespace_id,
        )
    }

    /// Instantiates `Adam` optimizer with given params.
    pub fn new(
        alpha: F,
        eps: F,
        b1: F,
        b2: F,
        var_id_list: impl IntoIterator<Item = VariableID>,
        env: &mut VariableEnvironment<F>,
        adam_namespace_id: &'static str,
    ) -> Adam<F> {
        for vid in var_id_list.into_iter() {
            let m_name = format!("{vid}m");
            let v_name = format!("{vid}v");
            let t_name = format!("{vid}t");
            let (m, v, t) = {
                let target_var = env
                    .get_array_by_id(vid)
                    .expect("variable array not found")
                    .borrow();
                let varshape = target_var.shape();
                (
                    crate::ndarray_ext::zeros(varshape),
                    crate::ndarray_ext::zeros(varshape),
                    crate::ndarray_ext::from_scalar(F::one()),
                )
            };
            let mut adam_ns = env.namespace_mut(adam_namespace_id);
            adam_ns.slot().name(m_name).set(m);
            adam_ns.slot().name(v_name).set(v);
            adam_ns.slot().name(t_name).set(t);
        }
        Adam {
            alpha,
            eps,
            b1,
            b2,
            adam_namespace_id,
        }
    }
}

impl<F: Float> Optimizer<F> for Adam<F> {
    fn compute_updates<'g, A, B>(
        &self,
        params: &[A],
        grads: &[B],
        g: &'g Context<F>,
    ) -> Vec<Tensor<'g, F>>
    where
        A: AsRef<Tensor<'g, F>> + Copy,
        B: AsRef<Tensor<'g, F>> + Copy,
    {
        let num_params = params.len();
        assert_eq!(num_params, grads.len());
        let mut ret = Vec::with_capacity(num_params);
        for i in 0..num_params {
            let param = params[i].as_ref();
            let namespace = g.namespace(self.adam_namespace_id);
            let var_id = param.get_variable_id().expect("Got non-variable tensor");
            let m = g.variable_by_name(format!("{var_id}m"), &namespace);
            let v = g.variable_by_name(format!("{var_id}v"), &namespace);
            let t = g.variable_by_name(format!("{var_id}t"), &namespace);

            // Create the Adam operation, which will return multiple outputs
            // Only the first output (updated parameter) is what we need to return
            let adam_op = Tensor::builder(g)
                .append_input(param, true)
                .append_input(grads[i].as_ref(), false)
                .append_input(m, true)
                .append_input(v, true)
                .append_input(t, true)
                .build(adam::AdamOp {
                    alpha: self.alpha,
                    eps: self.eps,
                    b1: self.b1,
                    b2: self.b2,
                });

            // Add the updated parameter to the result
            ret.push(adam_op);
        }
        ret
    }
}
