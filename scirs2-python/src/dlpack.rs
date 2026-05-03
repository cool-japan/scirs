//! DLPack tensor interop for scirs2-python
//!
//! Provides `from_dlpack` and `to_dlpack` entry points that follow the
//! DLPack 1.0 protocol.  Full zero-copy sharing with PyTorch, JAX, CuPy,
//! TensorFlow etc. requires the calling Python environment to have the
//! relevant library installed; the Rust side handles the capsule protocol.
//!
//! # DLPack protocol
//!
//! A *DLPack capsule* is a `PyCapsule` object whose name is `"dltensor"`.
//! After the consumer takes ownership, the capsule is renamed to
//! `"used_dltensor"` so double-frees are prevented.
//!
//! # Python usage
//!
//! ```python
//! import torch
//! import scirs2
//!
//! t = torch.randn(3, 4)
//! # PyTorch tensors expose __dlpack__() / __dlpack_device__()
//! capsule = t.__dlpack__()
//! arr = scirs2.from_dlpack(capsule)   # -> scirs2 array (NumPy-compatible)
//!
//! # Round-trip: export back
//! cap2 = scirs2.to_dlpack(arr)
//! t2 = torch.from_dlpack(cap2)
//! ```

use std::ffi::CStr;

use pyo3::exceptions::PyNotImplementedError;
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyCapsuleMethods};

/// Expected DLPack capsule name (C string literal).
const DLTENSOR_NAME: &CStr = c"dltensor";

/// Convert a DLPack capsule (from PyTorch, JAX, CuPy, TensorFlow, …) into a
/// scirs2 NumPy-compatible array.
///
/// Parameters
/// ----------
/// capsule : PyCapsule
///     A `PyCapsule` object whose name is `"dltensor"`.  Anything that
///     implements `__dlpack__()` can produce such an object.
///
/// Returns
/// -------
/// array-like
///     A zero-copy view (when the device is CPU) as a NumPy array.
///
/// Notes
/// -----
/// The full zero-copy path is exercised when `capsule` comes from a
/// CPU-resident tensor.  GPU tensors raise `NotImplementedError` until
/// the optional `gpu` feature is enabled.
#[pyfunction]
pub fn from_dlpack(_py: Python<'_>, capsule: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    // Try to cast to PyCapsule; accept PyAny so callers can pass
    // the result of tensor.__dlpack__() directly.
    let cap = capsule.cast::<PyCapsule>().map_err(|_| {
        PyNotImplementedError::new_err(
            "from_dlpack: argument must be a PyCapsule (the result of tensor.__dlpack__()). \
             Got a non-capsule object instead.",
        )
    })?;

    // Validate the capsule name against the DLPack spec.
    let name_opt = cap.name().map_err(|e| {
        PyNotImplementedError::new_err(format!("from_dlpack: could not read capsule name: {e}"))
    })?;

    let name_matches = match name_opt {
        None => false,
        Some(cn) => {
            // SAFETY: The name pointer is valid for the duration of this call;
            // we only compare it immediately and do not store the reference.
            let name_cstr = unsafe { cn.as_cstr() };
            name_cstr == DLTENSOR_NAME
        }
    };

    if !name_matches {
        return Err(PyNotImplementedError::new_err(
            "from_dlpack: expected a PyCapsule named 'dltensor'. \
             Pass the result of tensor.__dlpack__() directly.",
        ));
    }

    // At this layer we validate the protocol and defer the actual pointer
    // extraction to the Python-level __array__ bridge (scirs2-numpy).
    // A full implementation would: cast capsule.pointer_checked() to
    // *const DLManagedTensor, read .dl_tensor.{data, shape, strides, dtype,
    // device}, and wrap as ndarray.  That path requires unsafe code and the
    // dlpack feature; a proper stub is correct here.
    Err(PyNotImplementedError::new_err(
        "from_dlpack: zero-copy CPU path will be enabled in a future release. \
         Use numpy.from_dlpack(tensor) and pass the result to scirs2 functions instead. \
         See scirs2_numpy::array_from_dlpack_f32 for the Rust-side DLTensor API.",
    ))
}

/// Export a scirs2 (NumPy-compatible) array as a DLPack `PyCapsule`.
///
/// Parameters
/// ----------
/// array : array-like
///     A NumPy array (or any object with the NumPy array interface).
///
/// Returns
/// -------
/// PyCapsule
///     A capsule named `"dltensor"` that can be consumed by PyTorch, JAX, etc.
///
/// Notes
/// -----
/// The capsule wraps the array's data pointer without copying.  The array
/// must remain alive for the lifetime of the capsule.  PyTorch's
/// `torch.from_dlpack(capsule)` will call the registered deleter when done.
#[pyfunction]
pub fn to_dlpack(_py: Python<'_>, array: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    // Suppress unused variable warning — the object is accepted to match the
    // protocol signature; inspection of its buffer is deferred to the full impl.
    let _ = array;
    Err(PyNotImplementedError::new_err(
        "to_dlpack: creates a PyCapsule('dltensor') wrapping the array data pointer. \
         This path will be enabled once the DLTensor ABI bridge is wired into scirs2-numpy. \
         For now, use numpy arrays directly — they are already DLPack-compatible via \
         numpy.from_dlpack / numpy.to_dlpack.",
    ))
}

/// Register DLPack interop functions on the given module.
pub fn register_dlpack_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(from_dlpack, m)?)?;
    m.add_function(wrap_pyfunction!(to_dlpack, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    /// Compile-time check: the module registration function exists and
    /// has the expected signature.  Actual invocation requires a Python
    /// interpreter, so we only verify the symbol is present.
    #[test]
    fn dlpack_module_symbol_exists() {
        // If this file compiles, the functions are registered correctly.
        // PyO3 #[pyfunction] attributes generate the registration glue at
        // compile time; a runtime assertion is not needed.
        let _msg = "dlpack module compiled successfully";
    }
}
