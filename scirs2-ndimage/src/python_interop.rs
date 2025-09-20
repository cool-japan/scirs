//! Python Interoperability Support
//!
//! This module provides the foundation for Python bindings, including data exchange,
//! error conversion, and API compatibility layers that would be used with PyO3.
//!
//! Note: This module provides the foundation but requires PyO3 dependency to create
//! actual Python bindings. See the documentation for setup instructions.

use ndarray::Dimension;
use std::collections::HashMap;

use crate::error::NdimageError;

/// Python-compatible array metadata
#[derive(Debug, Clone)]
pub struct PyArrayInfo {
    /// Array shape
    pub shape: Vec<usize>,
    /// Data type name (f32, f64, i32, etc.)
    pub dtype: String,
    /// Stride information for memory layout
    pub strides: Vec<isize>,
    /// Whether the array is contiguous in memory
    pub contiguous: bool,
}

/// Python-compatible error information
#[derive(Debug, Clone)]
pub struct PyError {
    /// Error type (ValueError, RuntimeError, etc.)
    pub error_type: String,
    /// Error message
    pub message: String,
    /// Optional error context
    pub context: Option<HashMap<String, String>>,
}

impl From<NdimageError> for PyError {
    fn from(error: NdimageError) -> Self {
        match error {
            NdimageError::InvalidInput(msg) => PyError {
                error_type: "ValueError".to_string(),
                message: msg,
                context: None,
            },
            NdimageError::DimensionError(msg) => PyError {
                error_type: "ValueError".to_string(),
                message: format!("Dimension , error: {}", msg),
                context: None,
            },
            NdimageError::ComputationError(msg) => PyError {
                error_type: "RuntimeError".to_string(),
                message: msg,
                context: None,
            },
            NdimageError::MemoryError(msg) => PyError {
                error_type: "MemoryError".to_string(),
                message: msg,
                context: None,
            },
            // Catch-all for remaining error types
            _ => PyError {
                error_type: "RuntimeError".to_string(),
                message: format!("{}", error),
                context: None,
            },
        }
    }
}

/// Python-compatible function parameter specification
#[derive(Debug, Clone)]
pub struct PyParameter {
    /// Parameter name
    pub name: String,
    /// Parameter type
    pub param_type: String,
    /// Default value (if any)
    pub default: Option<String>,
    /// Parameter description
    pub description: String,
    /// Whether the parameter is required
    pub required: bool,
}

/// Python-compatible function specification
#[derive(Debug, Clone)]
pub struct PyFunction {
    /// Function name
    pub name: String,
    /// Function description/docstring
    pub description: String,
    /// Input parameters
    pub parameters: Vec<PyParameter>,
    /// Return type description
    pub return_type: String,
    /// Usage examples
    pub examples: Vec<String>,
}

/// Array conversion utilities for Python interop
pub mod array_conversion {
    use super::*;

    /// Convert array metadata to Python-compatible format
    pub fn array_to_py_info<T, D>(
        array: &ndarray::ArrayBase<ndarray::OwnedRepr<T>, D>,
    ) -> PyArrayInfo
    where
        T: 'static,
        D: Dimension,
    {
        let shape = array.shape().to_vec();
        let strides = array.strides().iter().map(|&s| s as isize).collect();

        let dtype = if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            "float32".to_string()
        } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
            "float64".to_string()
        } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<i32>() {
            "int32".to_string()
        } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<i64>() {
            "int64".to_string()
        } else {
            "unknown".to_string()
        };

        PyArrayInfo {
            shape,
            dtype,
            strides,
            contiguous: array.is_standard_layout(),
        }
    }

    /// Validate array compatibility for Python interop
    pub fn validate_array_compatibility<T>(info: &PyArrayInfo) -> Result<(), PyError>
    where
        T: 'static,
    {
        // Check if the dtype matches the expected type
        let expected_dtype = if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            "float32"
        } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
            "float64"
        } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<i32>() {
            "int32"
        } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<i64>() {
            "int64"
        } else {
            return Err(PyError {
                error_type: "TypeError".to_string(),
                message: "Unsupported array data type".to_string(),
                context: None,
            });
        };

        if info.dtype != expected_dtype {
            return Err(PyError {
                error_type: "TypeError".to_string(),
                message: format!("Expected dtype '{}', got '{}'", expected_dtype, info.dtype),
                context: None,
            });
        }

        // Check for reasonable array sizes
        let total_elements: usize = info.shape.iter().product();
        if total_elements > 1_000_000_000 {
            return Err(PyError {
                error_type: "MemoryError".to_string(),
                message: "Array too large for processing".to_string(),
                context: None,
            });
        }

        Ok(())
    }
}

/// Python API specification generation
pub mod api_spec {
    use super::*;

    /// Generate Python API specifications for ndimage functions
    pub fn generate_filter_api_specs() -> Vec<PyFunction> {
        vec![
            PyFunction {
                name: "gaussian_filter".to_string(),
                description: "Apply Gaussian filter to n-dimensional array.".to_string(),
                parameters: vec![
                    PyParameter {
                        name: "input".to_string(),
                        param_type: "array_like".to_string(),
                        default: None,
                        description: "Input array to filter".to_string(),
                        required: true,
                    },
                    PyParameter {
                        name: "sigma".to_string(),
                        param_type: "float or sequence of floats".to_string(),
                        default: None,
                        description: "Standard deviation for Gaussian kernel".to_string(),
                        required: true,
                    },
                    PyParameter {
                        name: "mode".to_string(),
                        param_type: "str".to_string(),
                        default: Some("'reflect'".to_string()),
                        description:
                            "Boundary mode ('reflect', 'constant', 'nearest', 'mirror', 'wrap')"
                                .to_string(),
                        required: false,
                    },
                ],
                return_type: "ndarray".to_string(),
                examples: vec![
                    ">>> import scirs2_ndimage as ndi".to_string(),
                    ">>> result = ndi.gaussian_filter(image, sigma=1.0)".to_string(),
                ],
            },
            PyFunction {
                name: "median_filter".to_string(),
                description: "Apply median filter to n-dimensional array.".to_string(),
                parameters: vec![
                    PyParameter {
                        name: "input".to_string(),
                        param_type: "array_like".to_string(),
                        default: None,
                        description: "Input array to filter".to_string(),
                        required: true,
                    },
                    PyParameter {
                        name: "size".to_string(),
                        param_type: "int or sequence of ints".to_string(),
                        default: None,
                        description: "Size of the median filter window".to_string(),
                        required: true,
                    },
                ],
                return_type: "ndarray".to_string(),
                examples: vec![">>> result = ndi.median_filter(image, size=3)".to_string()],
            },
        ]
    }

    /// Generate Python API specifications for morphology functions
    pub fn generate_morphology_api_specs() -> Vec<PyFunction> {
        vec![PyFunction {
            name: "binary_erosion".to_string(),
            description: "Multidimensional binary erosion with given structuring element."
                .to_string(),
            parameters: vec![
                PyParameter {
                    name: "input".to_string(),
                    param_type: "array_like".to_string(),
                    default: None,
                    description: "Binary array to be eroded".to_string(),
                    required: true,
                },
                PyParameter {
                    name: "structure".to_string(),
                    param_type: "array_like, optional".to_string(),
                    default: Some("None".to_string()),
                    description: "Structuring element for erosion".to_string(),
                    required: false,
                },
            ],
            return_type: "ndarray".to_string(),
            examples: vec![">>> result = ndi.binary_erosion(binary_image)".to_string()],
        }]
    }

    /// Generate comprehensive API documentation
    pub fn generate_python_docs() -> String {
        let mut docs = String::new();

        docs.push_str("# SciRS2 NDImage Python API\n\n");
        docs.push_str("## Filters\n\n");

        for func in generate_filter_api_specs() {
            docs.push_str(&format!("### {}\n\n", func.name));
            docs.push_str(&format!("{}\n\n", func.description));
            docs.push_str("**Parameters:**\n\n");

            for param in &func.parameters {
                let req_str = if param.required {
                    " (required)"
                } else {
                    " (optional)"
                };
                let default_str = param
                    .default
                    .as_ref()
                    .map(|d| format!(", default: {}", d))
                    .unwrap_or_default();

                docs.push_str(&format!(
                    "- `{}` (*{}*{}{}) - {}\n",
                    param.name, param.param_type, req_str, default_str, param.description
                ));
            }

            docs.push_str(&format!("\n**Returns:** {}\n\n", func.return_type));

            if !func.examples.is_empty() {
                docs.push_str("**Examples:**\n\n```python\n");
                for example in &func.examples {
                    docs.push_str(&format!("{}\n", example));
                }
                docs.push_str("```\n\n");
            }
        }

        docs.push_str("## Morphology\n\n");

        for func in generate_morphology_api_specs() {
            docs.push_str(&format!("### {}\n\n", func.name));
            docs.push_str(&format!("{}\n\n", func.description));
            // ... similar formatting as above
        }

        docs
    }
}

/// Example Python binding signatures (would be used with PyO3)
pub mod binding_examples {

    /// Example binding signature for Gaussian filter
    /// This shows what the actual PyO3 binding would look like
    pub fn example_gaussian_filter_binding() -> String {
        r#"
#[pyfunction]
#[pyo3(signature = (input, sigma, mode="reflect"))]
#[allow(dead_code)]
fn gaussian_filter(
    py: Python,
    input: &PyArray<f64, Ix2>,
    sigma: f64,
    mode: Option<&str>,
) -> PyResult<Py<PyArray<f64, Ix2>>> {
    let input_array = input.readonly();
    let input_view = input_array.as_array();
    
    let boundary_mode = match mode.unwrap_or("reflect") {
        "reflect" => BoundaryMode::Reflect,
        "constant" => BoundaryMode::Constant(0.0),
        "nearest" => BoundaryMode::Nearest,
        "mirror" => BoundaryMode::Mirror,
        "wrap" => BoundaryMode::Wrap_ => return Err(PyValueError::new_err("Invalid boundary mode")),
    };
    
    let result = crate::filters::gaussian_filter(&input_view, sigma, Some(boundary_mode))
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    
    Ok(result.to_pyarray(py).to_owned())
}
"#
        .to_string()
    }

    /// Example binding signature for median filter
    pub fn example_median_filter_binding() -> String {
        r#"
#[pyfunction]
#[allow(dead_code)]
fn median_filter(
    py: Python,
    input: &PyArray<f64, Ix2>,
    size: usize,
) -> PyResult<Py<PyArray<f64, Ix2>>> {
    let input_array = input.readonly();
    let input_view = input_array.as_array();
    
    let result = crate::filters::median_filter(&input_view, size)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    
    Ok(result.to_pyarray(py).to_owned())
}
"#
        .to_string()
    }

    /// Generate module definition for PyO3
    pub fn generate_module_definition() -> String {
        r#"
#[pymodule]
#[allow(dead_code)]
fn scirs2_ndimage(py: Python, m: &PyModule) -> PyResult<()> {
    // Filters submodule
    let filters_module = PyModule::new(py, "filters")?;
    filters_module.add_function(wrap_pyfunction!(gaussian_filter, filters_module)?)?;
    filters_module.add_function(wrap_pyfunction!(median_filter, filters_module)?)?;
    m.add_submodule(filters_module)?;
    
    // Morphology submodule
    let morphology_module = PyModule::new(py, "morphology")?;
    morphology_module.add_function(wrap_pyfunction!(binary_erosion, morphology_module)?)?;
    morphology_module.add_function(wrap_pyfunction!(binary_dilation, morphology_module)?)?;
    m.add_submodule(morphology_module)?;
    
    // Measurements submodule
    let measurements_module = PyModule::new(py, "measurements")?;
    measurements_module.add_function(wrap_pyfunction!(label, measurements_module)?)?;
    measurements_module.add_function(wrap_pyfunction!(center_of_mass, measurements_module)?)?;
    m.add_submodule(measurements_module)?;
    
    Ok(())
}
"#
        .to_string()
    }
}

/// Setup and installation utilities
pub mod setup {

    /// Generate setup.py content for Python package
    pub fn generate_setup_py() -> String {
        r#"
from setuptools import setup
from pyo3_setuptools_rust import Pyo3RustExtension, build_rust

setup(
    name="scirs2-ndimage",
    version="0.1.0-beta.2",
    author="SciRS2 Team",
    author_email="contact@scirs2.org",
    description="High-performance N-dimensional image processing library",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/cool-japan/scirs",
    rust_extensions=[
        Pyo3RustExtension(
            "scirs2_ndimage._rust",
            binding="pyo3",
            debug=False,
        )
    ],
    packages=["scirs2_ndimage"],
    zip_safe=False,
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Rust",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    cmdclass={"build_rust": build_rust},
)
"#
        .to_string()
    }

    /// Generate _init__.py for Python package
    pub fn generate_init_py() -> String {
        r#"
'"'
SciRS2 NDImage - High-performance N-dimensional image processing
==============================================================

A comprehensive library for n-dimensional image processing with SciPy-compatible APIs
and Rust performance.

Submodules
----------
filters : Filtering operations (Gaussian, median, etc.)
morphology : Morphological operations (erosion, dilation, etc.)
measurements : Measurements and analysis functions
interpolation : Interpolation and geometric transformations
segmentation : Image segmentation algorithms
features : Feature detection algorithms

Examples
--------
>>> import scirs2_ndimage as ndi
>>> import numpy as np
>>> image = np.random.random((100, 100))
>>> filtered = ndi.gaussian_filter(image, sigma=1.0)
>>> binary = image > 0.5
>>> eroded = ndi.binary_erosion(binary)
'"'

from ._rust import *
from . import filters, morphology, measurements, interpolation, segmentation, features

__version__ = "0.1.0-beta.2"
__author__ = "SciRS2 Team"

# Expose commonly used functions at the top level
from .filters import gaussian_filter, median_filter, uniform_filter
from .morphology import binary_erosion, binary_dilation, binary_opening, binary_closing
from .measurements import label, center_of_mass, find_objects
from .features import canny, sobel_edges

__all__ = [
    "gaussian_filter",
    "median_filter", 
    "uniform_filter",
    "binary_erosion",
    "binary_dilation",
    "binary_opening", 
    "binary_closing",
    "label",
    "center_of_mass",
    "find_objects",
    "canny",
    "sobel_edges",
]
"#
        .to_string()
    }

    /// Generate installation instructions
    pub fn generate_install_instructions() -> String {
        r#"
# SciRS2 NDImage Python Installation Guide

## Prerequisites

1. **Rust**: Install Rust from https://rustup.rs/
2. **Python**: Python 3.7 or later
3. **Dependencies**: 
   ```bash
   pip install setuptools-rust pyo3-setuptools-rust numpy
   ```

## Building from Source

1. Clone the repository:
   ```bash
   git clone https://github.com/cool-japan/scirs.git
   cd scirs/scirs2-ndimage
   ```

2. Build and install:
   ```bash
   pip install .
   ```

3. For development installation:
   ```bash
   pip install -e .
   ```

## Usage

```python
import scirs2_ndimage as ndi
import numpy as np

# Create sample data
image = np.random.random((100, 100))

# Apply Gaussian filter
filtered = ndi.gaussian_filter(image, sigma=1.0)

# Binary morphology
binary = image > 0.5
eroded = ndi.binary_erosion(binary)

# Feature detection
edges = ndi.canny(image, sigma=1.0, low_threshold=0.1, high_threshold=0.2)
```

## Performance Notes

- SciRS2 NDImage leverages Rust's performance for computational kernels
- SIMD optimizations are automatically enabled when available
- Parallel processing is used for large arrays
- Memory usage is optimized through zero-copy operations where possible

## Compatibility

This package provides a SciPy-compatible API, making it a drop-in replacement
for many `scipy.ndimage` functions with improved performance.
"#
        .to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_array_to_py_info() {
        let arr = array![[1.0f64, 2.0], [3.0, 4.0]];
        let info = array_conversion::array_to_py_info(&arr);

        assert_eq!(info.shape, vec![2, 2]);
        assert_eq!(info.dtype, "float64");
        assert!(info.contiguous);
    }

    #[test]
    fn test_validate_array_compatibility() {
        let info = PyArrayInfo {
            shape: vec![10, 10],
            dtype: "float64".to_string(),
            strides: vec![8, 80],
            contiguous: true,
        };

        let result = array_conversion::validate_array_compatibility::<f64>(&info);
        assert!(result.is_ok());
    }

    #[test]
    fn test_invalid_dtype_compatibility() {
        let info = PyArrayInfo {
            shape: vec![10, 10],
            dtype: "float32".to_string(),
            strides: vec![4, 40],
            contiguous: true,
        };

        let result = array_conversion::validate_array_compatibility::<f64>(&info);
        assert!(result.is_err());
    }

    #[test]
    fn test_error_conversion() {
        let ndimage_error = NdimageError::InvalidInput("Test error".to_string());
        let py_error: PyError = ndimage_error.into();

        assert_eq!(py_error.error_type, "ValueError");
        assert_eq!(py_error.message, "Test error");
    }

    #[test]
    fn test_api_spec_generation() {
        let specs = api_spec::generate_filter_api_specs();
        assert!(!specs.is_empty());

        let gaussian_spec = specs.iter().find(|s| s.name == "gaussian_filter");
        assert!(gaussian_spec.is_some());

        let spec = gaussian_spec.unwrap();
        assert!(!spec.parameters.is_empty());
        assert!(spec.parameters.iter().any(|p| p.name == "input"));
        assert!(spec.parameters.iter().any(|p| p.name == "sigma"));
    }

    #[test]
    fn test_python_docs_generation() {
        let docs = api_spec::generate_python_docs();
        assert!(docs.contains("# SciRS2 NDImage Python API"));
        assert!(docs.contains("gaussian_filter"));
        assert!(docs.contains("Parameters:"));
    }
}
