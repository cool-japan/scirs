//! Simplified L-BFGS optimizer implementation

use ndarray::{Array, Array1, Dimension, ScalarOperand};
use num_traits::Float;
use std::collections::VecDeque;
use std::fmt::Debug;

use crate::error::Result;
use crate::optimizers::Optimizer;

/// Simplified L-BFGS optimizer for testing
///
/// This is a simpler implementation that focuses on correctness over performance.
#[derive(Debug, Clone)]
pub struct SimpleLBFGS<A: Float + ScalarOperand + Debug> {
    /// Learning rate
    learning_rate: A,
    /// History size
    history_size: usize,
    /// Step history
    s_list: VecDeque<Array1<A>>,
    /// Gradient difference history
    y_list: VecDeque<Array1<A>>,
    /// Previous parameters
    prev_params: Option<Array1<A>>,
    /// Previous gradient
    prev_grad: Option<Array1<A>>,
}

impl<A: Float + ScalarOperand + Debug> SimpleLBFGS<A> {
    pub fn new(learning_rate: A) -> Self {
        Self {
            learning_rate,
            history_size: 20,
            s_list: VecDeque::new(),
            y_list: VecDeque::new(),
            prev_params: None,
            prev_grad: None,
        }
    }

    pub fn learning_rate(&self) -> A {
        self.learning_rate
    }

    pub fn set_lr(&mut self, lr: A) {
        self.learning_rate = lr;
    }

    pub fn reset(&mut self) {
        self.s_list.clear();
        self.y_list.clear();
        self.prev_params = None;
        self.prev_grad = None;
    }

    fn compute_direction(&self, grad: &Array1<A>) -> Array1<A> {
        let mut q = grad.clone();
        let m = self.s_list.len();
        
        if m == 0 {
            return q.mapv(|x| -x);
        }
        
        let mut alpha = vec![A::zero(); m];
        
        // First loop
        for i in (0..m).rev() {
            let s_i = &self.s_list[i];
            let y_i = &self.y_list[i];
            let rho_i = A::one() / s_i.dot(y_i);
            alpha[i] = rho_i * s_i.dot(&q);
            q = &q - &(y_i * alpha[i]);
        }
        
        // Scale by approximate Hessian
        let s = &self.s_list[m-1];
        let y = &self.y_list[m-1];
        let gamma = s.dot(y) / y.dot(y);
        let mut z = q * gamma;
        
        // Second loop
        for i in 0..m {
            let s_i = &self.s_list[i];
            let y_i = &self.y_list[i];
            let rho_i = A::one() / s_i.dot(y_i);
            let beta = rho_i * y_i.dot(&z);
            z = &z + &(s_i * (alpha[i] - beta));
        }
        
        z.mapv(|x| -x)
    }
}

impl<A, D> Optimizer<A, D> for SimpleLBFGS<A>
where
    A: Float + ScalarOperand + Debug,
    D: Dimension,
{
    fn step(&mut self, params: &Array<A, D>, gradients: &Array<A, D>) -> Result<Array<A, D>> {
        let params_flat = params.to_owned().into_shape_with_order(params.len()).unwrap();
        let grad_flat = gradients.to_owned().into_shape_with_order(gradients.len()).unwrap();
        
        // Update history if we have previous values
        if let (Some(prev_p), Some(prev_g)) = (&self.prev_params, &self.prev_grad) {
            let s = &params_flat - prev_p;
            let y = &grad_flat - prev_g;
            
            // Check if update is valid
            let sy = s.dot(&y);
            if sy > A::from(1e-10).unwrap() {
                if self.s_list.len() >= self.history_size {
                    self.s_list.pop_front();
                    self.y_list.pop_front();
                }
                self.s_list.push_back(s);
                self.y_list.push_back(y);
            }
        }
        
        // Compute search direction
        let direction = self.compute_direction(&grad_flat);
        
        // Update parameters
        let new_params_flat = &params_flat + &(&direction * self.learning_rate);
        
        // Store current values for next iteration
        self.prev_params = Some(params_flat);
        self.prev_grad = Some(grad_flat);
        
        // Reshape and return
        Ok(new_params_flat.into_shape_with_order(params.raw_dim()).unwrap())
    }

    fn get_learning_rate(&self) -> A {
        self.learning_rate
    }

    fn set_learning_rate(&mut self, learning_rate: A) {
        self.learning_rate = learning_rate;
    }
}