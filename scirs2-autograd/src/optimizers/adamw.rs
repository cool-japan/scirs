//! AdamW optimizer

use crate::optimizers::Optimizer;
use crate::tensor::Tensor;
use crate::tensor_ops::gradient_descent_ops::adamw;
use crate::variable::VariableID;
use crate::{Context, Float, VariableEnvironment};

/// AdamW optimizer with decoupled weight decay
///
/// This implementation is based on the paper "Decoupled Weight Decay Regularization"
/// (https://arxiv.org/abs/1711.05101). AdamW fixes the issue with Adam where L2 regularization
/// and weight decay are not equivalent. AdamW decouples weight decay from gradient-based updates.
///
/// # Example
/// ```
/// use scirs2_autograd as ag;
/// use ag::prelude::*;
/// use ag::optimizers::AdamW;
/// use ag::variable::NamespaceTrait;
/// use ag::tensor_ops::*;
///
/// // Define parameters to optimize.
/// let mut env = ag::VariableEnvironment::new();
/// let mut rng = ag::ndarray_ext::ArrayRng::<f32>::default();
///
/// let w = env.slot().set(rng.glorot_uniform(&[4, 2]));
/// let b = env.slot().set(ag::ndarray_ext::zeros(&[1, 2]));
///
/// // AdamW optimizer with default params.
/// let _adamw = AdamW::default("my_adamw", env.default_namespace().current_var_ids(), &mut env);
///
/// env.run(|g| {
///     let w = g.variable(w);
///     let b = g.variable(b);
///     
///     // Simple operations using w and b
///     let x = ones(&[1, 4], g);
///     let y = add(matmul(x, w), b);
///     let loss = sum_all(y);
///     
///     let _grads = grad(&[loss], &[w, b]);
///     // Optimizer usage would happen here in a real training loop
/// });
/// ```
///
/// See also <https://github.com/raskr/rust-autograd/blob/master/examples/>
pub struct AdamW<F: Float> {
    pub alpha: F,
    pub eps: F,
    pub b1: F,
    pub b2: F,
    pub weight_decay: F,
    pub adamw_namespace_id: &'static str,
}

impl<F: Float> AdamW<F> {
    /// Instantiates `AdamW` optimizer with the recommended parameters.
    ///
    /// # Arguments
    /// * `unique_namespace_id` - Unique identifier for the optimizer's variable namespace
    /// * `var_id_list` - List of variable IDs to optimize
    /// * `env_handle` - Mutable reference to the variable environment
    ///
    /// # Default parameters
    /// * `alpha` (learning rate): 0.001
    /// * `eps`: 1e-8
    /// * `b1` (beta1): 0.9
    /// * `b2` (beta2): 0.999
    /// * `weight_decay`: 0.01
    pub fn default(
        unique_namespace_id: &'static str,
        var_id_list: impl IntoIterator<Item = VariableID>,
        env_handle: &mut VariableEnvironment<F>,
    ) -> AdamW<F> {
        AdamW::new(
            F::from(0.001).unwrap(), // alpha (learning rate)
            F::from(1e-08).unwrap(), // eps
            F::from(0.9).unwrap(),   // b1 (beta1)
            F::from(0.999).unwrap(), // b2 (beta2)
            F::from(0.01).unwrap(),  // weight_decay
            var_id_list,
            env_handle,
            unique_namespace_id,
        )
    }

    /// Instantiates `AdamW` optimizer with given parameters.
    ///
    /// # Arguments
    /// * `alpha` - Learning rate (default: 0.001)
    /// * `eps` - Term added to denominator to improve numerical stability (default: 1e-8)
    /// * `b1` - Coefficient for computing running averages of gradient (default: 0.9)
    /// * `b2` - Coefficient for computing running averages of squared gradient (default: 0.999)
    /// * `weight_decay` - Weight decay coefficient (default: 0.01)
    /// * `var_id_list` - List of variable IDs to optimize
    /// * `env` - Mutable reference to the variable environment
    /// * `adamw_namespace_id` - Unique identifier for the optimizer's variable namespace
    pub fn new(
        alpha: F,
        eps: F,
        b1: F,
        b2: F,
        weight_decay: F,
        var_id_list: impl IntoIterator<Item = VariableID>,
        env: &mut VariableEnvironment<F>,
        adamw_namespace_id: &'static str,
    ) -> AdamW<F> {
        // Initialize state variables for each parameter
        for vid in var_id_list.into_iter() {
            let m_name = format!("{}m", vid);
            let v_name = format!("{}v", vid);
            let t_name = format!("{}t", vid);
            let (m, v, t) = {
                let target_var = env
                    .get_array_by_id(vid)
                    .expect("variable array not found")
                    .borrow();
                let var_shape = target_var.shape();
                (
                    crate::ndarray_ext::zeros(var_shape), // First moment estimate
                    crate::ndarray_ext::zeros(var_shape), // Second moment estimate
                    crate::ndarray_ext::from_scalar(F::one()), // Timestep
                )
            };
            let mut adamw_ns = env.namespace_mut(adamw_namespace_id);
            adamw_ns.slot().name(m_name).set(m);
            adamw_ns.slot().name(v_name).set(v);
            adamw_ns.slot().name(t_name).set(t);
        }
        AdamW {
            alpha,
            eps,
            b1,
            b2,
            weight_decay,
            adamw_namespace_id,
        }
    }

    /// Creates a new AdamW optimizer with custom weight decay.
    ///
    /// This is a convenience method for quickly creating an AdamW optimizer
    /// with default Adam parameters but custom weight decay.
    pub fn with_weight_decay(
        weight_decay: F,
        unique_namespace_id: &'static str,
        var_id_list: impl IntoIterator<Item = VariableID>,
        env_handle: &mut VariableEnvironment<F>,
    ) -> AdamW<F> {
        AdamW::new(
            F::from(0.001).unwrap(), // alpha (learning rate)
            F::from(1e-08).unwrap(), // eps
            F::from(0.9).unwrap(),   // b1 (beta1)
            F::from(0.999).unwrap(), // b2 (beta2)
            weight_decay,            // custom weight_decay
            var_id_list,
            env_handle,
            unique_namespace_id,
        )
    }

    /// Creates a new AdamW optimizer with custom learning rate and weight decay.
    ///
    /// This is a convenience method for quickly creating an AdamW optimizer
    /// with custom learning rate and weight decay, using default beta values.
    pub fn with_lr_and_weight_decay(
        alpha: F,
        weight_decay: F,
        unique_namespace_id: &'static str,
        var_id_list: impl IntoIterator<Item = VariableID>,
        env_handle: &mut VariableEnvironment<F>,
    ) -> AdamW<F> {
        AdamW::new(
            alpha,                   // custom learning rate
            F::from(1e-08).unwrap(), // eps
            F::from(0.9).unwrap(),   // b1 (beta1)
            F::from(0.999).unwrap(), // b2 (beta2)
            weight_decay,            // custom weight_decay
            var_id_list,
            env_handle,
            unique_namespace_id,
        )
    }
}

impl<F: Float> Optimizer<F> for AdamW<F> {
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
            let namespace = g.namespace(self.adamw_namespace_id);
            let var_id = param.get_variable_id().expect("Got non-variable tensor");
            let m = g.variable_by_name(format!("{}m", var_id), &namespace);
            let v = g.variable_by_name(format!("{}v", var_id), &namespace);
            let t = g.variable_by_name(format!("{}t", var_id), &namespace);

            // Create the AdamW operation, which will return multiple outputs
            // Only the first output (updated parameter) is what we need to return
            let adamw_op = Tensor::builder(g)
                .append_input(param, true)
                .append_input(grads[i].as_ref(), false)
                .append_input(m, true)
                .append_input(v, true)
                .append_input(t, true)
                .build(adamw::AdamWOp {
                    alpha: self.alpha,
                    eps: self.eps,
                    b1: self.b1,
                    b2: self.b2,
                    weight_decay: self.weight_decay,
                });

            // Log AdamW operation construction
            eprintln!("Created AdamWOp with all 5 inputs");

            // Add the updated parameter to the result
            ret.push(adamw_op);
        }
        ret
    }
}
