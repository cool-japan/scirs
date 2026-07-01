//! Python bindings for scirs2-neural
//!
//! This module provides Python bindings for neural network activation functions
//! and utilities. Full layer-based training requires scirs2-autograd's computational
//! graph system. For comprehensive neural network training, use PyTorch or TensorFlow.

use pyo3::prelude::*;
use scirs2_neural::activations_minimal::{Activation, ReLU, Sigmoid, Softmax, Tanh, GELU};
use scirs2_numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods};

// ============================================================================
// Activation Function Classes
// ============================================================================

/// ReLU activation function
///
/// Applies: f(x) = max(0, x)
///
/// Example:
///     relu = scirs2.ReLU()
///     output = relu.forward(input_array)
#[pyclass(name = "ReLU")]
pub struct PyReLU {
    inner: ReLU,
}

#[pymethods]
impl PyReLU {
    #[new]
    fn new() -> Self {
        Self { inner: ReLU::new() }
    }

    /// Forward pass
    ///
    /// Args:
    ///     input (np.ndarray): Input array (any shape)
    ///
    /// Returns:
    ///     np.ndarray: Activated output
    fn forward(&self, py: Python, input: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        apply_activation(&self.inner, py, input)
    }

    /// Backward pass (gradient computation)
    ///
    /// Args:
    ///     grad_output (np.ndarray): Gradient from next layer
    ///     input (np.ndarray): Original input to forward pass
    ///
    /// Returns:
    ///     np.ndarray: Gradient with respect to input
    fn backward(
        &self,
        py: Python,
        grad_output: &Bound<'_, PyAny>,
        input: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        apply_activation_backward(&self.inner, py, grad_output, input)
    }
}

/// Sigmoid activation function
///
/// Applies: f(x) = 1 / (1 + exp(-x))
///
/// Example:
///     sigmoid = scirs2.Sigmoid()
///     output = sigmoid.forward(input_array)
#[pyclass(name = "Sigmoid")]
pub struct PySigmoid {
    inner: Sigmoid,
}

#[pymethods]
impl PySigmoid {
    #[new]
    fn new() -> Self {
        Self {
            inner: Sigmoid::new(),
        }
    }

    /// Forward pass
    fn forward(&self, py: Python, input: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        apply_activation(&self.inner, py, input)
    }

    /// Backward pass
    fn backward(
        &self,
        py: Python,
        grad_output: &Bound<'_, PyAny>,
        input: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        apply_activation_backward(&self.inner, py, grad_output, input)
    }
}

/// Tanh activation function
///
/// Applies: f(x) = tanh(x)
///
/// Example:
///     tanh = scirs2.Tanh()
///     output = tanh.forward(input_array)
#[pyclass(name = "Tanh")]
pub struct PyTanh {
    inner: Tanh,
}

#[pymethods]
impl PyTanh {
    #[new]
    fn new() -> Self {
        Self { inner: Tanh::new() }
    }

    /// Forward pass
    fn forward(&self, py: Python, input: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        apply_activation(&self.inner, py, input)
    }

    /// Backward pass
    fn backward(
        &self,
        py: Python,
        grad_output: &Bound<'_, PyAny>,
        input: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        apply_activation_backward(&self.inner, py, grad_output, input)
    }
}

/// GELU activation function
///
/// Gaussian Error Linear Unit activation.
///
/// Example:
///     gelu = scirs2.GELU()
///     output = gelu.forward(input_array)
#[pyclass(name = "GELU")]
pub struct PyGELU {
    inner: GELU,
}

#[pymethods]
impl PyGELU {
    #[new]
    #[pyo3(signature = (fast=false))]
    fn new(fast: bool) -> Self {
        Self {
            inner: if fast { GELU::fast() } else { GELU::new() },
        }
    }

    /// Forward pass
    fn forward(&self, py: Python, input: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        apply_activation(&self.inner, py, input)
    }

    /// Backward pass
    fn backward(
        &self,
        py: Python,
        grad_output: &Bound<'_, PyAny>,
        input: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        apply_activation_backward(&self.inner, py, grad_output, input)
    }
}

/// Softmax activation function
///
/// Applies: f(x)_i = exp(x_i) / sum(exp(x_j))
///
/// Example:
///     softmax = scirs2.Softmax(axis=-1)
///     output = softmax.forward(input_array)
#[pyclass(name = "Softmax")]
pub struct PySoftmax {
    inner: Softmax,
}

#[pymethods]
impl PySoftmax {
    #[new]
    #[pyo3(signature = (axis=-1))]
    fn new(axis: isize) -> Self {
        Self {
            inner: Softmax::new(axis),
        }
    }

    /// Forward pass
    fn forward(&self, py: Python, input: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        apply_activation(&self.inner, py, input)
    }

    /// Backward pass
    fn backward(
        &self,
        py: Python,
        grad_output: &Bound<'_, PyAny>,
        input: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyAny>> {
        apply_activation_backward(&self.inner, py, grad_output, input)
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Apply activation function to NumPy array
fn apply_activation<A: Activation<f64>>(
    activation: &A,
    py: Python,
    input: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    // Try 1D array
    if let Ok(arr1d) = input.cast::<PyArray1<f64>>() {
        let binding = arr1d.readonly();
        let data = binding.as_array().to_owned();
        let dyn_input = data.into_dyn();

        let output = activation.forward(&dyn_input).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Activation error: {}", e))
        })?;

        let out1d = output
            .into_dimensionality::<scirs2_core::ndarray::Ix1>()
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Dimension error: {}", e))
            })?;

        return Ok(out1d.into_pyarray(py).unbind().into());
    }

    // Try 2D array
    if let Ok(arr2d) = input.cast::<PyArray2<f64>>() {
        let binding = arr2d.readonly();
        let data = binding.as_array().to_owned();
        let dyn_input = data.into_dyn();

        let output = activation.forward(&dyn_input).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Activation error: {}", e))
        })?;

        let out2d = output
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Dimension error: {}", e))
            })?;

        return Ok(out2d.into_pyarray(py).unbind().into());
    }

    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
        "Input must be 1D or 2D float64 numpy array",
    ))
}

/// Apply activation backward pass
fn apply_activation_backward<A: Activation<f64>>(
    activation: &A,
    py: Python,
    grad_output: &Bound<'_, PyAny>,
    input: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    // Try 1D arrays
    if let (Ok(grad1d), Ok(inp1d)) = (
        grad_output.cast::<PyArray1<f64>>(),
        input.cast::<PyArray1<f64>>(),
    ) {
        let grad_binding = grad1d.readonly();
        let grad_data = grad_binding.as_array().to_owned().into_dyn();

        let inp_binding = inp1d.readonly();
        let inp_data = inp_binding.as_array().to_owned().into_dyn();

        let grad_input = activation.backward(&grad_data, &inp_data).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Activation backward error: {}",
                e
            ))
        })?;

        let out1d = grad_input
            .into_dimensionality::<scirs2_core::ndarray::Ix1>()
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Dimension error: {}", e))
            })?;

        return Ok(out1d.into_pyarray(py).unbind().into());
    }

    // Try 2D arrays
    if let (Ok(grad2d), Ok(inp2d)) = (
        grad_output.cast::<PyArray2<f64>>(),
        input.cast::<PyArray2<f64>>(),
    ) {
        let grad_binding = grad2d.readonly();
        let grad_data = grad_binding.as_array().to_owned().into_dyn();

        let inp_binding = inp2d.readonly();
        let inp_data = inp_binding.as_array().to_owned().into_dyn();

        let grad_input = activation.backward(&grad_data, &inp_data).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Activation backward error: {}",
                e
            ))
        })?;

        let out2d = grad_input
            .into_dimensionality::<scirs2_core::ndarray::Ix2>()
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Dimension error: {}", e))
            })?;

        return Ok(out2d.into_pyarray(py).unbind().into());
    }

    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
        "Inputs must be 1D or 2D float64 numpy arrays",
    ))
}

// ============================================================================
// Module Registration
// ============================================================================

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register activation function classes
    m.add_class::<PyReLU>()?;
    m.add_class::<PySigmoid>()?;
    m.add_class::<PyTanh>()?;
    m.add_class::<PyGELU>()?;
    m.add_class::<PySoftmax>()?;

    // Add module documentation
    m.add(
        "__doc__",
        "Neural network activation functions and utilities\n\n\
        This module provides standalone activation functions that can be used\n\
        with NumPy arrays for neural network inference and custom training loops.\n\n\
        Available activations:\n\
        - ReLU: Rectified Linear Unit\n\
        - Sigmoid: Logistic sigmoid\n\
        - Tanh: Hyperbolic tangent\n\
        - GELU: Gaussian Error Linear Unit\n\
        - Softmax: Softmax normalization\n\n\
        Each activation provides:\n\
        - forward(input): Forward pass\n\
        - backward(grad_output, input): Backward pass for gradient computation\n\n\
        For comprehensive neural network training with automatic differentiation,\n\
        we recommend using PyTorch or TensorFlow, which integrate seamlessly\n\
        with scirs2 via NumPy array compatibility.",
    )?;

    Ok(())
}
