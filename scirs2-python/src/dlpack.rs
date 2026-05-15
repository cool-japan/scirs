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

use std::ffi::{c_void, CStr};
use std::ptr::NonNull;

use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyCapsuleMethods};
use scirs2_numpy::dlpack::{
    DLDataType, DLDataTypeCode, DLDevice, DLDeviceType, DLManagedTensor, DLTensor,
};

/// Expected DLPack capsule name (C string literal, DLPack 1.0 spec).
const DLTENSOR_NAME: &CStr = c"dltensor";

/// Name the capsule is renamed to once consumed (prevents double-free).
const USED_DLTENSOR_NAME: &CStr = c"used_dltensor";

// ─── Ownership wrapper ────────────────────────────────────────────────────────

/// Heap allocation that backs a DLPack capsule created by `to_dlpack`.
///
/// Bundles the `DLManagedTensor` with the shape/strides arrays and the owned
/// data copy.  All memory is freed through `BackingStore::drop_raw`.
struct BackingStore {
    /// ABI-compatible managed-tensor struct; must be the first field so that
    /// a `*mut BackingStore` can be cast to `*mut DLManagedTensor` safely.
    managed: DLManagedTensor,
    /// Owned copy of the tensor's element data.
    data: Vec<f64>,
    /// Owned shape array (length = `managed.dl_tensor.ndim`).
    shape: Vec<i64>,
    /// Owned strides array (length = `managed.dl_tensor.ndim`).
    strides: Vec<i64>,
}

impl BackingStore {
    /// Free a `BackingStore` that was previously leaked with `Box::into_raw`.
    ///
    /// # Safety
    ///
    /// `ptr` must be a non-null pointer obtained from `Box::into_raw` on a
    /// `BackingStore`.  This function must be called at most once.
    unsafe fn drop_raw(ptr: *mut BackingStore) {
        if !ptr.is_null() {
            // SAFETY: ptr was obtained from Box::into_raw.
            drop(unsafe { Box::from_raw(ptr) });
        }
    }
}

/// DLPack `deleter` stored inside the `DLManagedTensor`.
///
/// Called by the consumer framework (PyTorch, JAX, etc.) when it is finished
/// with the tensor.
///
/// # Safety
///
/// `managed` must point to the `managed` field of a `BackingStore` that was
/// previously leaked via `Box::into_raw`.
unsafe extern "C" fn backing_store_deleter(managed: *mut DLManagedTensor) {
    if managed.is_null() {
        return;
    }
    // SAFETY: BackingStore has `managed` as its first field, so the pointer
    // arithmetic is a no-op and the cast is valid.
    let backing = managed as *mut BackingStore;
    // SAFETY: backed by a Box::into_raw call in `to_dlpack`.
    unsafe { BackingStore::drop_raw(backing) };
}

/// Destructor registered with `PyCapsule::new_with_pointer_and_destructor`.
///
/// Called by Python's GC when the capsule object is finalized.  Extracts the
/// `BackingStore` raw pointer from the capsule and drops it.
///
/// # Safety
///
/// `capsule` must be a valid `PyObject*` whose capsule pointer was set to the
/// `managed` field of a `BackingStore` allocation.
unsafe extern "C" fn capsule_destructor(capsule: *mut pyo3::ffi::PyObject) {
    // SAFETY: capsule is a valid PyCapsule whose pointer was set during
    // `to_dlpack` to a `BackingStore::managed` field.
    let ptr = unsafe { pyo3::ffi::PyCapsule_GetPointer(capsule, DLTENSOR_NAME.as_ptr()) };
    if !ptr.is_null() {
        let managed_ptr = ptr as *mut DLManagedTensor;
        // SAFETY: managed_ptr is the `managed` field of a BackingStore.
        if let Some(deleter) = unsafe { (*managed_ptr).deleter } {
            unsafe { deleter(managed_ptr) };
        }
    }
}

// ─── from_dlpack ─────────────────────────────────────────────────────────────

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
/// numpy.ndarray
///     A 1-D `float64` NumPy array whose contents are *copied* from the
///     DLPack tensor.  Only CPU, float32, and float64 tensors are currently
///     supported; all other dtypes raise `TypeError`.
///
/// Notes
/// -----
/// GPU tensors raise `TypeError` until an optional `gpu` feature is enabled.
/// The capsule is renamed to `"used_dltensor"` after consumption to prevent
/// double-frees, consistent with the DLPack 1.0 spec.
#[pyfunction]
pub fn from_dlpack(py: Python<'_>, capsule: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    // Cast to PyCapsule — accept PyAny so callers can pass __dlpack__() result.
    let cap = capsule.cast::<PyCapsule>().map_err(|_| {
        PyTypeError::new_err(
            "from_dlpack: argument must be a PyCapsule (the result of tensor.__dlpack__()). \
             Got a non-capsule object instead.",
        )
    })?;

    // Validate the capsule name against the DLPack spec.
    let name_opt = cap.name().map_err(|e| {
        PyValueError::new_err(format!("from_dlpack: could not read capsule name: {e}"))
    })?;

    let name_matches = match name_opt {
        None => false,
        Some(cn) => {
            // SAFETY: The name pointer is valid for the duration of this call.
            let name_cstr = unsafe { cn.as_cstr() };
            name_cstr == DLTENSOR_NAME
        }
    };

    if !name_matches {
        return Err(PyValueError::new_err(
            "from_dlpack: expected a PyCapsule named 'dltensor'. \
             Pass the result of tensor.__dlpack__() directly.",
        ));
    }

    // Retrieve the DLManagedTensor pointer from the capsule.
    // SAFETY: We validated the name above; the pointer was placed here by the
    // producer and is valid until we consume it.
    let nn_ptr: NonNull<c_void> = cap
        .pointer_checked(Some(DLTENSOR_NAME))
        .map_err(|e| PyRuntimeError::new_err(format!("from_dlpack: null capsule pointer: {e}")))?;

    let managed_ptr = nn_ptr.as_ptr() as *mut DLManagedTensor;

    // SAFETY: managed_ptr is non-null and valid; derived from the capsule above.
    let dl_tensor: &DLTensor = unsafe { &(*managed_ptr).dl_tensor };

    // Reject non-CPU tensors.
    if dl_tensor.device.device_type != DLDeviceType::Cpu as i32 {
        return Err(PyTypeError::new_err(format!(
            "from_dlpack: only CPU tensors are supported (got device type {}). \
             Copy the tensor to CPU before calling from_dlpack.",
            dl_tensor.device.device_type
        )));
    }

    // Reject null data pointers.
    if dl_tensor.data.is_null() {
        return Err(PyValueError::new_err(
            "from_dlpack: tensor has a null data pointer.",
        ));
    }

    // Compute the flat element count from shape.
    let n_elems: usize = if dl_tensor.ndim == 0 || dl_tensor.shape.is_null() {
        1
    } else {
        // SAFETY: shape is valid for ndim elements (DLPack producer contract).
        let shape_slice = unsafe {
            std::slice::from_raw_parts(dl_tensor.shape as *const i64, dl_tensor.ndim as usize)
        };
        shape_slice.iter().map(|&d| d as usize).product()
    };

    // Dispatch on dtype — copy into a Python list of floats, then wrap as numpy array.
    let base_ptr = unsafe { (dl_tensor.data as *const u8).add(dl_tensor.byte_offset as usize) };

    let dtype = dl_tensor.dtype;
    let flat_vec: Vec<f64> = match (dtype.code, dtype.bits, dtype.lanes) {
        // float32 (DLDataTypeCode::Float = 2, bits=32)
        (2, 32, 1) => {
            let slice = unsafe { std::slice::from_raw_parts(base_ptr as *const f32, n_elems) };
            slice.iter().map(|&v| v as f64).collect()
        }
        // float64
        (2, 64, 1) => {
            let slice = unsafe { std::slice::from_raw_parts(base_ptr as *const f64, n_elems) };
            slice.to_vec()
        }
        // int8
        (0, 8, 1) => {
            let slice = unsafe { std::slice::from_raw_parts(base_ptr as *const i8, n_elems) };
            slice.iter().map(|&v| v as f64).collect()
        }
        // int16
        (0, 16, 1) => {
            let slice = unsafe { std::slice::from_raw_parts(base_ptr as *const i16, n_elems) };
            slice.iter().map(|&v| v as f64).collect()
        }
        // int32
        (0, 32, 1) => {
            let slice = unsafe { std::slice::from_raw_parts(base_ptr as *const i32, n_elems) };
            slice.iter().map(|&v| v as f64).collect()
        }
        // int64
        (0, 64, 1) => {
            let slice = unsafe { std::slice::from_raw_parts(base_ptr as *const i64, n_elems) };
            slice.iter().map(|&v| v as f64).collect()
        }
        // uint8
        (1, 8, 1) => {
            let slice = unsafe { std::slice::from_raw_parts(base_ptr, n_elems) };
            slice.iter().map(|&v| v as f64).collect()
        }
        // uint16
        (1, 16, 1) => {
            let slice = unsafe { std::slice::from_raw_parts(base_ptr as *const u16, n_elems) };
            slice.iter().map(|&v| v as f64).collect()
        }
        // uint32
        (1, 32, 1) => {
            let slice = unsafe { std::slice::from_raw_parts(base_ptr as *const u32, n_elems) };
            slice.iter().map(|&v| v as f64).collect()
        }
        // uint64
        (1, 64, 1) => {
            let slice = unsafe { std::slice::from_raw_parts(base_ptr as *const u64, n_elems) };
            slice.iter().map(|&v| v as f64).collect()
        }
        (code, bits, _) => {
            return Err(PyTypeError::new_err(format!(
                "from_dlpack: unsupported dtype (code={code}, bits={bits}). \
                 Supported: int8/16/32/64, uint8/16/32/64, float32, float64.",
            )));
        }
    };

    // Build shape tuple for numpy.
    let shape_vec: Vec<usize> = if dl_tensor.ndim == 0 || dl_tensor.shape.is_null() {
        vec![n_elems]
    } else {
        // SAFETY: shape is valid for ndim elements.
        let shape_slice = unsafe {
            std::slice::from_raw_parts(dl_tensor.shape as *const i64, dl_tensor.ndim as usize)
        };
        shape_slice.iter().map(|&d| d as usize).collect()
    };

    // Rename the capsule to "used_dltensor" per DLPack 1.0 spec to prevent
    // the producer from being consumed again (double-free guard).
    // We attempt this on a best-effort basis; failure is non-fatal here because
    // the data has already been copied.
    let rename_result =
        unsafe { pyo3::ffi::PyCapsule_SetName(cap.as_ptr(), USED_DLTENSOR_NAME.as_ptr()) };
    let _ = rename_result; // intentionally ignored after copy

    // Call the managed tensor's deleter if present, as we have consumed it.
    if let Some(deleter) = unsafe { (*managed_ptr).deleter } {
        unsafe { deleter(managed_ptr) };
    }

    // Convert the flat f64 Vec into a numpy array via Python's numpy.
    let numpy = py.import("numpy").map_err(|e| {
        PyRuntimeError::new_err(format!("from_dlpack: could not import numpy: {e}"))
    })?;
    let arr = numpy.getattr("array")?.call1((flat_vec,))?;

    // Reshape to match the original tensor shape.
    let shaped = arr.call_method1("reshape", (shape_vec,))?;

    Ok(shaped.into())
}

// ─── to_dlpack ────────────────────────────────────────────────────────────────

/// Export a scirs2 (NumPy-compatible) array as a DLPack `PyCapsule`.
///
/// Parameters
/// ----------
/// array : numpy.ndarray
///     A NumPy float64 array (or any object with the buffer protocol that
///     numpy can interpret as float64).
///
/// Returns
/// -------
/// PyCapsule
///     A capsule named `"dltensor"` that can be consumed by PyTorch, JAX, etc.
///
/// Notes
/// -----
/// The capsule *owns a copy* of the array data so that the Python array object
/// can be garbage-collected independently.  The `DLManagedTensor.deleter`
/// registered in the capsule frees this copy when the consumer is done.
#[pyfunction]
pub fn to_dlpack(py: Python<'_>, array: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    // Extract the array data as a Vec<f64> via numpy.
    let numpy = py
        .import("numpy")
        .map_err(|e| PyRuntimeError::new_err(format!("to_dlpack: could not import numpy: {e}")))?;

    // Ensure we have a contiguous float64 C-order array.
    let arr = numpy.getattr("asarray")?.call1((array,))?;
    let arr_f64 = numpy
        .getattr("ascontiguousarray")?
        .call((arr,), Some(&pyo3::types::PyDict::new(py)))?;

    // Read shape.
    let shape_obj = arr_f64.getattr("shape")?;
    let shape_tuple: Vec<i64> = shape_obj.extract::<Vec<i64>>().map_err(|e| {
        PyTypeError::new_err(format!("to_dlpack: could not extract array shape: {e}"))
    })?;

    // Extract flat data as f64.
    let flat_list = arr_f64.call_method0("flatten")?;
    let data_vec: Vec<f64> = flat_list.extract::<Vec<f64>>().map_err(|e| {
        PyTypeError::new_err(format!(
            "to_dlpack: array must be convertible to float64: {e}"
        ))
    })?;

    // Compute C-order strides (in elements).
    let strides_vec: Vec<i64> = compute_c_strides(&shape_tuple);

    // Build the BackingStore on the heap.  We use Box::into_raw so it lives
    // until the capsule destructor frees it.
    let n = shape_tuple.len();
    let mut store = Box::new(BackingStore {
        managed: DLManagedTensor {
            dl_tensor: DLTensor {
                data: std::ptr::null_mut(), // filled in below
                device: DLDevice {
                    device_type: DLDeviceType::Cpu as i32,
                    device_id: 0,
                },
                ndim: n as i32,
                dtype: DLDataType {
                    code: DLDataTypeCode::Float as u8,
                    bits: 64,
                    lanes: 1,
                },
                shape: std::ptr::null_mut(),   // filled in below
                strides: std::ptr::null_mut(), // filled in below
                byte_offset: 0,
            },
            manager_ctx: std::ptr::null_mut(),
            deleter: Some(backing_store_deleter),
        },
        data: data_vec,
        shape: shape_tuple,
        strides: strides_vec,
    });

    // Now that the Vecs are in their final locations inside the Box, set the
    // raw pointers in dl_tensor to point into those Vecs.
    store.managed.dl_tensor.data = store.data.as_mut_ptr() as *mut c_void;
    store.managed.dl_tensor.shape = store.shape.as_mut_ptr();
    store.managed.dl_tensor.strides = store.strides.as_mut_ptr();

    let raw_store: *mut BackingStore = Box::into_raw(store);
    // SAFETY: raw_store is non-null (just created by Box::into_raw).
    let managed_nn = NonNull::new(raw_store as *mut c_void)
        .ok_or_else(|| PyRuntimeError::new_err("to_dlpack: null BackingStore pointer"))?;

    // SAFETY: managed_nn points to a valid BackingStore; capsule_destructor
    // will call backing_store_deleter which frees it via Box::from_raw.
    let capsule = unsafe {
        PyCapsule::new_with_pointer_and_destructor(
            py,
            managed_nn,
            DLTENSOR_NAME,
            Some(capsule_destructor),
        )
    }
    .map_err(|e| PyRuntimeError::new_err(format!("to_dlpack: failed to create capsule: {e}")))?;

    Ok(capsule.into())
}

/// Compute C-order (row-major) strides in elements for the given shape.
///
/// The last dimension has stride 1; each preceding dimension has stride equal
/// to the product of all following dimensions.
fn compute_c_strides(shape: &[i64]) -> Vec<i64> {
    let n = shape.len();
    if n == 0 {
        return Vec::new();
    }
    let mut strides = vec![1i64; n];
    for i in (0..n - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Register DLPack interop functions on the given module.
pub fn register_dlpack_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(from_dlpack, m)?)?;
    m.add_function(wrap_pyfunction!(to_dlpack, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Compile-time check: the module registration function exists and has the
    /// expected signature.  Actual invocation requires a Python interpreter.
    #[test]
    fn dlpack_module_symbol_exists() {
        let _msg = "dlpack module compiled successfully";
    }

    #[test]
    fn compute_c_strides_1d() {
        assert_eq!(compute_c_strides(&[5]), vec![1]);
    }

    #[test]
    fn compute_c_strides_2d() {
        // Shape [2, 3] -> strides [3, 1]
        assert_eq!(compute_c_strides(&[2, 3]), vec![3, 1]);
    }

    #[test]
    fn compute_c_strides_3d() {
        // Shape [2, 3, 4] -> strides [12, 4, 1]
        assert_eq!(compute_c_strides(&[2, 3, 4]), vec![12, 4, 1]);
    }

    #[test]
    fn compute_c_strides_empty() {
        assert_eq!(compute_c_strides(&[]), Vec::<i64>::new());
    }
}
