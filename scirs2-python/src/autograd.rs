//! Python bindings for scirs2-autograd
//!
//! This module provides Python bindings for variable management and model persistence
//! from scirs2-autograd. Due to Rust lifetime complexities, full computational graph
//! APIs are not exposed. For comprehensive automatic differentiation, use PyTorch
//! or TensorFlow which integrate seamlessly with scirs2 via NumPy arrays.

use pyo3::prelude::*;
use scirs2_autograd::variable::NamespaceTrait;
use scirs2_autograd::VariableEnvironment;
use scirs2_numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods};

/// Variable environment for managing trainable parameters
///
/// Provides save/load functionality for model persistence. For training,
/// use PyTorch/TensorFlow and transfer weights via NumPy arrays.
///
/// Example:
///     env = scirs2.VariableEnvironment()
///     # Set variables using NumPy arrays
///     var_id = env.set_variable("weights", np.random.randn(784, 128))
///     # Save trained model
///     env.save("model.json")
///     # Load model
///     loaded_env = scirs2.VariableEnvironment.load("model.json")
#[pyclass(name = "VariableEnvironment", unsendable)]
pub struct PyVariableEnvironment {
    inner: VariableEnvironment<f64>,
}

#[pymethods]
impl PyVariableEnvironment {
    /// Create a new variable environment
    #[new]
    fn new() -> Self {
        Self {
            inner: VariableEnvironment::new(),
        }
    }

    /// Set a named variable from a NumPy array
    ///
    /// Args:
    ///     name (str): Variable name
    ///     array (np.ndarray): Variable value (1D or 2D float64 array)
    fn set_variable(&mut self, name: &str, array: &Bound<'_, PyAny>) -> PyResult<()> {
        // Try 1D array first
        if let Ok(arr1d) = array.cast::<PyArray1<f64>>() {
            let binding = arr1d.readonly();
            let data = binding.as_array().to_owned();
            self.inner.name(name).set(data);
            return Ok(());
        }

        // Try 2D array
        if let Ok(arr2d) = array.cast::<PyArray2<f64>>() {
            let binding = arr2d.readonly();
            let data = binding.as_array().to_owned();
            self.inner.name(name).set(data);
            return Ok(());
        }

        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "Array must be 1D or 2D float64 numpy array",
        ))
    }

    /// Get a named variable as a NumPy array
    ///
    /// Args:
    ///     name (str): Variable name
    ///
    /// Returns:
    ///     np.ndarray: Variable value
    fn get_variable(&self, py: Python, name: &str) -> PyResult<Py<PyAny>> {
        let namespace = self.inner.default_namespace();
        let array_ref = namespace.get_array_by_name(name).ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyKeyError, _>(format!("Variable '{}' not found", name))
        })?;

        let array = array_ref.borrow();

        // Return based on dimensionality
        match array.ndim() {
            1 => {
                let arr1d = array
                    .view()
                    .into_dimensionality::<scirs2_core::ndarray::Ix1>()
                    .map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                            "Dimension error: {}",
                            e
                        ))
                    })?;
                Ok(arr1d.to_owned().into_pyarray(py).unbind().into())
            }
            2 => {
                let arr2d = array
                    .view()
                    .into_dimensionality::<scirs2_core::ndarray::Ix2>()
                    .map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                            "Dimension error: {}",
                            e
                        ))
                    })?;
                Ok(arr2d.to_owned().into_pyarray(py).unbind().into())
            }
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Only 1D and 2D arrays are currently supported",
            )),
        }
    }

    /// List all variable names in the default namespace
    ///
    /// Returns:
    ///     list: List of variable names
    fn list_variables(&self) -> Vec<String> {
        self.inner
            .default_namespace()
            .current_var_names()
            .into_iter()
            .map(|s: &str| s.to_string())
            .collect()
    }

    /// Save the variable environment to a file
    ///
    /// Args:
    ///     path (str): Path to save file (.json format)
    fn save(&self, path: &str) -> PyResult<()> {
        self.inner
            .save(path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Save error: {}", e)))
    }

    /// Load a variable environment from a file
    ///
    /// Args:
    ///     path (str): Path to load file (.json format)
    ///
    /// Returns:
    ///     VariableEnvironment: Loaded environment
    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let inner = VariableEnvironment::<f64>::load(path).map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("Load error: {}", e))
        })?;

        Ok(Self { inner })
    }

    /// Get the number of variables in the environment
    fn __len__(&self) -> usize {
        self.inner.default_namespace().current_var_names().len()
    }

    /// String representation
    fn __repr__(&self) -> String {
        format!(
            "VariableEnvironment({} variables)",
            self.inner.default_namespace().current_var_names().len()
        )
    }
}

// ============================================================================
// Module Registration
// ============================================================================

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register VariableEnvironment class
    m.add_class::<PyVariableEnvironment>()?;

    // Add module documentation
    m.add("__doc__", "Automatic differentiation and variable management\n\n\
        This module provides model parameter storage and persistence from scirs2-autograd.\n\
        Due to Rust lifetime complexities in the computational graph API, full autodiff\n\
        functionality is not exposed. For neural network training, we recommend:\n\n\
        - PyTorch: Industry-standard deep learning framework\n\
        - TensorFlow: Comprehensive ML platform\n\n\
        scirs2 arrays are NumPy-compatible, enabling seamless integration with these frameworks.\n\n\
        Use VariableEnvironment for:\n\
        - Storing and managing model parameters\n\
        - Saving/loading trained model weights\n\
        - Transferring weights between scirs2 and PyTorch/TensorFlow")?;

    Ok(())
}
