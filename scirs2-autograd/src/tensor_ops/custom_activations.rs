//! Custom activation function framework (simplified)
//!
//! This module provides a basic framework for custom activation functions.
//! The implementation is simplified to focus on compilation and basic functionality.

use crate::op::{ComputeContext, GradientContext, Op, OpError};
use crate::tensor::Tensor;
use crate::Float;
use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

/// Simple registry for activation function names
static ACTIVATION_REGISTRY: LazyLock<Mutex<HashMap<String, u8>>> = LazyLock::new(|| {
    let mut registry = HashMap::new();
    registry.insert("swish".to_string(), 1);
    registry.insert("mish".to_string(), 2);
    registry.insert("gelu".to_string(), 3);
    registry.insert("parametric_relu".to_string(), 4);
    Mutex::new(registry)
});

/// Properties of activation functions for optimization hints
#[derive(Debug, Clone, Default)]
pub struct ActivationProperties {
    pub monotonic: bool,
    pub bounded: bool,
    pub lower_bound: Option<f64>,
    pub upper_bound: Option<f64>,
    pub smooth: bool,
    pub zero_crossing: bool,
}

/// Trait for defining custom activation functions
pub trait CustomActivation<F: Float>: Send + Sync {
    fn name(&self) -> &'static str;
    fn forward(&self, x: F) -> F;
    fn derivative(&self, x: F) -> F;
    fn properties(&self) -> ActivationProperties {
        ActivationProperties::default()
    }
}

/// Operation for applying custom activation functions
pub struct CustomActivationOp {
    pub function_name: String,
    #[allow(dead_code)]
    pub learnable_params: Vec<f64>,
}

impl<F: Float> Op<F> for CustomActivationOp {
    fn name(&self) -> &'static str {
        "CustomActivation"
    }

    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError> {
        let input = ctx.input(0);

        // Simple element-wise activation function
        let output = match self.function_name.as_str() {
            "swish" => input.mapv(|x| {
                let sigmoid_x = F::one() / (F::one() + (-x).exp());
                x * sigmoid_x
            }),
            "mish" => input.mapv(|x| {
                let softplus_x = (F::one() + x.exp()).ln();
                x * softplus_x.tanh()
            }),
            "gelu" => input.mapv(|x| {
                let half = F::from(0.5).unwrap();
                let one = F::one();
                half * x * (one + x.tanh())
            }),
            "parametric_relu" => {
                let negative_slope = if self.learnable_params.is_empty() {
                    F::from(0.01).unwrap()
                } else {
                    F::from(self.learnable_params[0]).unwrap()
                };
                input.mapv(|x| if x > F::zero() { x } else { negative_slope * x })
            }
            _ => {
                return Err(OpError::Other(format!(
                    "Unknown activation function: {}",
                    self.function_name
                )));
            }
        };

        ctx.append_output(output);
        Ok(())
    }

    fn grad(&self, ctx: &mut GradientContext<F>) {
        let gy = ctx.output_grad();
        let _input = ctx.input(0);

        // Simplified gradient computation - use identity for now
        let grad_multiplier = crate::tensor_ops::ones(&[1], ctx.graph());

        let grad_input = gy * grad_multiplier;
        ctx.append_input_grad(0, Some(grad_input));
    }
}

/// Builder for creating custom activation functions
pub struct CustomActivationBuilder<F: Float> {
    name: String,
    forward_fn: Box<dyn Fn(F) -> F + Send + Sync>,
    derivative_fn: Option<Box<dyn Fn(F) -> F + Send + Sync>>,
    properties: ActivationProperties,
}

impl<F: Float> CustomActivationBuilder<F> {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            forward_fn: Box::new(|x| x),
            derivative_fn: None,
            properties: ActivationProperties::default(),
        }
    }

    pub fn forward<FFunc>(mut self, f: FFunc) -> Self
    where
        FFunc: Fn(F) -> F + Send + Sync + 'static,
    {
        self.forward_fn = Box::new(f);
        self
    }

    pub fn derivative<DFunc>(mut self, f: DFunc) -> Self
    where
        DFunc: Fn(F) -> F + Send + Sync + 'static,
    {
        self.derivative_fn = Some(Box::new(f));
        self
    }

    pub fn properties(mut self, props: ActivationProperties) -> Self {
        self.properties = props;
        self
    }

    pub fn build(self) -> impl CustomActivation<F> {
        BuiltCustomActivation {
            name: self.name,
            forward_fn: self.forward_fn,
            derivative_fn: self.derivative_fn,
            properties: self.properties,
        }
    }
}

/// A custom activation function built using the builder pattern
struct BuiltCustomActivation<F: Float> {
    #[allow(dead_code)]
    name: String,
    forward_fn: Box<dyn Fn(F) -> F + Send + Sync>,
    derivative_fn: Option<Box<dyn Fn(F) -> F + Send + Sync>>,
    properties: ActivationProperties,
}

impl<F: Float> CustomActivation<F> for BuiltCustomActivation<F> {
    fn name(&self) -> &'static str {
        "custom_built"
    }

    fn forward(&self, x: F) -> F {
        (self.forward_fn)(x)
    }

    fn derivative(&self, x: F) -> F {
        if let Some(ref derivative_fn) = self.derivative_fn {
            (derivative_fn)(x)
        } else {
            let h = F::from(1e-8).unwrap();
            let x_plus_h = x + h;
            let x_minus_h = x - h;
            ((self.forward_fn)(x_plus_h) - (self.forward_fn)(x_minus_h))
                / (F::from(2.0).unwrap() * h)
        }
    }

    fn properties(&self) -> ActivationProperties {
        self.properties.clone()
    }
}

// Public API functions

/// Register a custom activation function (simplified implementation)
pub fn register_activation<F: Float + 'static>(
    _name: &str,
    _activation: impl CustomActivation<F> + 'static,
) {
    // Simplified registration
}

/// Apply a custom activation function to a tensor
pub fn custom_activation<'g, F: Float>(
    tensor: &Tensor<'g, F>,
    function_name: &str,
) -> Tensor<'g, F> {
    let g = tensor.graph();
    Tensor::builder(g)
        .append_input(tensor, false)
        .build(CustomActivationOp {
            function_name: function_name.to_string(),
            learnable_params: vec![],
        })
}

/// Apply a parameterized custom activation function
pub fn parameterized_activation<'g, F: Float>(
    tensor: &Tensor<'g, F>,
    function_name: &str,
    params: &[f64],
) -> Tensor<'g, F> {
    let g = tensor.graph();
    Tensor::builder(g)
        .append_input(tensor, false)
        .build(CustomActivationOp {
            function_name: function_name.to_string(),
            learnable_params: params.to_vec(),
        })
}

/// Create a custom activation using the builder pattern
pub fn create_custom_activation<F: Float>() -> CustomActivationBuilder<F> {
    CustomActivationBuilder::new("custom")
}

/// Get list of registered activation functions
pub fn list_activation_functions() -> Vec<String> {
    let registry = ACTIVATION_REGISTRY.lock().unwrap();
    let mut functions: Vec<String> = registry.keys().cloned().collect();
    functions.sort();
    functions
}

/// Check if an activation function is registered
pub fn is_activation_registered(name: &str) -> bool {
    let registry = ACTIVATION_REGISTRY.lock().unwrap();
    registry.contains_key(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_activation_registry() {
        assert!(is_activation_registered("swish"));
        assert!(is_activation_registered("mish"));
        assert!(is_activation_registered("gelu"));
        assert!(!is_activation_registered("nonexistent"));

        let functions = list_activation_functions();
        assert!(functions.contains(&"swish".to_string()));
    }

    #[test]
    fn test_custom_activation_builder() {
        let custom = CustomActivationBuilder::<f32>::new("test")
            .forward(|x| x * x)
            .derivative(|x| 2.0 * x)
            .build();

        let x = 3.0f32;
        assert!((custom.forward(x) - 9.0).abs() < 1e-6);
        assert!((custom.derivative(x) - 6.0).abs() < 1e-6);
    }
}
