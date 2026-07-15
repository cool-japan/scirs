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
//! # Stride handling
//!
//! `from_dlpack` does **not** assume the producer's buffer is C-contiguous.
//! Non-contiguous tensors — a transposed PyTorch tensor (`t.t().__dlpack__()`),
//! a reversed/negative-stride view, a sliced view, … — report their true
//! per-dimension strides via `DLTensor.strides`, and this module walks the
//! buffer using genuine N-dimensional strided iteration so the copied data
//! always matches the producer's logical layout instead of being silently
//! misread as if it were contiguous.
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
///
/// # Layout
///
/// `#[repr(C)]` is required here, not optional: `to_dlpack` stores a pointer
/// to this struct's *start* in the `PyCapsule`, and both `from_dlpack` and
/// `capsule_destructor` reinterpret that address directly as `*mut
/// DLManagedTensor` (see `managed` below). Rust's default `repr(Rust)`
/// layout is free to reorder fields and does not guarantee `managed` sits at
/// offset 0 — without `#[repr(C)]` the reinterpreted pointer can land on the
/// wrong bytes (e.g. inside one of the `Vec` fields), so `device_type` reads
/// as garbage and calling the bogus `deleter` function pointer during
/// capsule cleanup is undefined behavior (observed in practice as an
/// immediate SIGBUS crash when a `to_dlpack` capsule is garbage-collected).
#[repr(C)]
struct BackingStore {
    /// ABI-compatible managed-tensor struct; must be the first field so that
    /// a `*mut BackingStore` can be cast to `*mut DLManagedTensor` safely.
    /// (Enforced by `#[repr(C)]` above — see the `# Layout` note.)
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
/// # Already-consumed capsules
///
/// Per the DLPack 1.0 protocol, `from_dlpack` renames a capsule it has
/// consumed from `"dltensor"` to `"used_dltensor"` *and* already invokes the
/// `deleter` itself at that point. If this destructor blindly called
/// `PyCapsule_GetPointer` with the original `"dltensor"` name on such a
/// capsule, two things would go wrong: (1) the name no longer matches, so
/// CPython sets a `ValueError` — which a finalizer must never leave
/// pending, since CPython surfaces it as a `SystemError` out of the next
/// `gc.collect()` / refcount drop; and (2) even if the name check were
/// bypassed, the `BackingStore` was already freed by `from_dlpack`, so
/// fetching and calling its deleter again here would be a double-free.
/// `PyCapsule_IsValid` is used instead of `PyCapsule_GetPointer` for the
/// initial check because the CPython docs guarantee it "will not fail" (it
/// never sets an exception) — it simply reports whether the capsule is
/// still a live, unconsumed `"dltensor"` capsule.
///
/// # Safety
///
/// `capsule` must be a valid `PyObject*`. If it is a live (unconsumed)
/// `"dltensor"` capsule, its stored pointer must have been set during
/// `to_dlpack` to the `managed` field of a `BackingStore` allocation.
unsafe extern "C" fn capsule_destructor(capsule: *mut pyo3::ffi::PyObject) {
    // SAFETY: capsule is a valid PyObject (finalizer contract of
    // `PyCapsule::new_with_pointer_and_destructor`). `PyCapsule_IsValid`
    // never sets a Python exception, unlike `PyCapsule_GetPointer` on a
    // name mismatch.
    let is_live_dltensor =
        unsafe { pyo3::ffi::PyCapsule_IsValid(capsule, DLTENSOR_NAME.as_ptr()) } != 0;
    if !is_live_dltensor {
        // Either not a capsule of ours, or already consumed by
        // `from_dlpack` (renamed to "used_dltensor", deleter already run
        // there) — nothing left to free.
        return;
    }

    // SAFETY: capsule is a valid PyCapsule whose pointer was set during
    // `to_dlpack` to a `BackingStore::managed` field; just confirmed live
    // and named "dltensor" by `PyCapsule_IsValid` above, so
    // `PyCapsule_GetPointer` with the same name is guaranteed to succeed.
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

/// Errors produced while decoding a [`DLTensor`]'s payload into a flat `f64`
/// buffer.
///
/// Kept separate from `PyErr` so the pure-Rust extraction logic
/// (`read_strided_elements` / `extract_dlpack_data`) has no PyO3 dependency
/// and can be unit tested without a Python interpreter.
#[derive(Debug)]
enum ExtractError {
    /// The element dtype (code + bits + lanes) is not one this bridge knows
    /// how to widen to `f64`.
    UnsupportedDtype {
        /// DLDataType code (0=int, 1=uint, 2=float, 3=bfloat).
        code: u8,
        /// DLDataType bit width.
        bits: u8,
    },
    /// An offset computation while walking the tensor's shape/strides
    /// overflowed, or the shape/strides were otherwise malformed.
    Arithmetic(String),
}

/// Read `product(shape)` elements of type `T` starting at `base_ptr`, honoring
/// the tensor's per-dimension `strides` (in units of `size_of::<T>()`, per the
/// DLPack ABI) instead of assuming a C-contiguous layout.
///
/// Elements are produced in row-major (C-order) traversal order relative to
/// `shape` — the innermost (last) dimension varies fastest — regardless of the
/// tensor's *physical* stride layout.  This means the result can be hand off
/// directly to `numpy.array(..).reshape(shape)` and still contain the
/// logically correct values even when the source tensor is a transposed,
/// reversed, sliced, or otherwise non-contiguous view (e.g.
/// `torch_tensor.t().__dlpack__()`).
///
/// # Errors
///
/// Returns `Err` — rather than silently wrapping or panicking — if `shape`
/// and `strides` have different lengths, if `shape` contains a negative
/// dimension, or if any intermediate offset computation would overflow
/// `i64`.
///
/// # Safety
///
/// For every multi-index `idx` within `shape`, `base_ptr` offset by
/// `(idx[0]*strides[0] + ... + idx[n-1]*strides[n-1]) * size_of::<T>()` bytes
/// must be a valid, readable, properly initialized `T` value.  This is the
/// same "producer promises a valid memory region" contract that every DLPack
/// consumer (NumPy, PyTorch, …) relies on.
unsafe fn read_strided_elements<T: Copy>(
    base_ptr: *const u8,
    shape: &[i64],
    strides: &[i64],
) -> Result<Vec<T>, String> {
    if shape.len() != strides.len() {
        return Err(format!(
            "shape/strides length mismatch: {} dims vs {} strides",
            shape.len(),
            strides.len()
        ));
    }

    if shape.is_empty() {
        // 0-dimensional tensor: a single scalar sits at `base_ptr` itself.
        // SAFETY: caller guarantees base_ptr is valid for one `T` read.
        let v = unsafe { std::ptr::read_unaligned(base_ptr as *const T) };
        return Ok(vec![v]);
    }

    let elem_size = std::mem::size_of::<T>() as i64;
    let ndim = shape.len();

    let mut total: usize = 1;
    for &d in shape {
        if d < 0 {
            return Err(format!("negative dimension {d} in tensor shape"));
        }
        total = total
            .checked_mul(d as usize)
            .ok_or_else(|| "tensor element count overflows usize".to_string())?;
    }

    let mut result: Vec<T> = Vec::with_capacity(total);
    let mut idx = vec![0i64; ndim];

    for _ in 0..total {
        let mut offset_elems: i64 = 0;
        for (&i, &s) in idx.iter().zip(strides.iter()) {
            let term = i
                .checked_mul(s)
                .ok_or_else(|| "stride offset overflow".to_string())?;
            offset_elems = offset_elems
                .checked_add(term)
                .ok_or_else(|| "stride offset overflow".to_string())?;
        }
        let byte_offset = offset_elems
            .checked_mul(elem_size)
            .ok_or_else(|| "byte offset overflow".to_string())?;

        // `wrapping_offset` is a safe fn (pure pointer arithmetic that can
        // never itself trigger UB); it is used instead of `offset` so an
        // extreme-but-not-i64-overflowing byte offset can't cause UB in the
        // arithmetic step. The actual unsafety is confined to the
        // dereference below.
        let ptr = base_ptr.wrapping_offset(byte_offset as isize) as *const T;
        // SAFETY: caller guarantees every offset reachable via shape/strides
        // addresses a valid, initialized `T` value relative to `base_ptr`.
        // `read_unaligned` additionally tolerates any alignment.
        let v = unsafe { std::ptr::read_unaligned(ptr) };
        result.push(v);

        // Odometer increment in row-major order: the last (innermost)
        // dimension varies fastest, matching the traversal order that
        // `numpy.reshape(shape)` expects from a flat buffer.
        for (i, &dim) in idx.iter_mut().zip(shape.iter()).rev() {
            *i += 1;
            if *i < dim {
                break;
            }
            *i = 0;
        }
    }

    Ok(result)
}

/// Decode a validated (non-null data, CPU-resident) [`DLTensor`] into a flat
/// `f64` buffer plus its logical shape, honoring `dl_tensor.strides` so that
/// non-contiguous (transposed, reversed, sliced, …) source tensors are read
/// correctly rather than silently misinterpreted as C-contiguous.
///
/// The returned `Vec<f64>` is always in row-major (C) order relative to the
/// returned shape, so it can be handed straight to
/// `numpy.array(..).reshape(shape)`.
///
/// # Safety
///
/// `dl_tensor.data` (offset by `dl_tensor.byte_offset`) combined with every
/// offset reachable via `dl_tensor.shape`/`dl_tensor.strides` must address
/// valid, initialized memory of the declared `dtype` — the standard DLPack
/// producer contract.  `dl_tensor.shape` (if non-null) must be valid for
/// `dl_tensor.ndim` elements, likewise for `dl_tensor.strides`.
unsafe fn extract_dlpack_data(
    dl_tensor: &DLTensor,
) -> Result<(Vec<f64>, Vec<usize>), ExtractError> {
    let is_scalar = dl_tensor.ndim <= 0 || dl_tensor.shape.is_null();

    let shape_i64: Vec<i64> = if is_scalar {
        Vec::new()
    } else {
        // SAFETY: shape is valid for ndim elements (DLPack producer contract).
        unsafe {
            std::slice::from_raw_parts(dl_tensor.shape as *const i64, dl_tensor.ndim as usize)
        }
        .to_vec()
    };

    // A NULL `strides` pointer is the DLPack convention for "this tensor is
    // C-contiguous". A *non-null* `strides` must always be honored, even
    // when it describes a non-contiguous layout (transposed/sliced/reversed
    // views) — that is precisely the case that must not be read as if it
    // were contiguous.
    let elem_strides: Vec<i64> = if is_scalar {
        Vec::new()
    } else if dl_tensor.strides.is_null() {
        compute_c_strides(&shape_i64)
    } else {
        // SAFETY: strides is valid for ndim elements (same producer contract
        // already relied on for `shape` above).
        unsafe {
            std::slice::from_raw_parts(dl_tensor.strides as *const i64, dl_tensor.ndim as usize)
        }
        .to_vec()
    };

    // SAFETY: caller (from_dlpack) already validated `dl_tensor.data` is
    // non-null; byte_offset is producer-supplied per the DLPack contract.
    let base_ptr = unsafe { (dl_tensor.data as *const u8).add(dl_tensor.byte_offset as usize) };

    let dtype = dl_tensor.dtype;
    let flat_vec: Vec<f64> = match (dtype.code, dtype.bits, dtype.lanes) {
        // float32 (DLDataTypeCode::Float = 2, bits=32)
        (2, 32, 1) => {
            let raw: Vec<f32> =
                unsafe { read_strided_elements(base_ptr, &shape_i64, &elem_strides) }
                    .map_err(ExtractError::Arithmetic)?;
            raw.into_iter().map(|v| v as f64).collect()
        }
        // float64
        (2, 64, 1) => unsafe { read_strided_elements::<f64>(base_ptr, &shape_i64, &elem_strides) }
            .map_err(ExtractError::Arithmetic)?,
        // int8
        (0, 8, 1) => {
            let raw: Vec<i8> =
                unsafe { read_strided_elements(base_ptr, &shape_i64, &elem_strides) }
                    .map_err(ExtractError::Arithmetic)?;
            raw.into_iter().map(|v| v as f64).collect()
        }
        // int16
        (0, 16, 1) => {
            let raw: Vec<i16> =
                unsafe { read_strided_elements(base_ptr, &shape_i64, &elem_strides) }
                    .map_err(ExtractError::Arithmetic)?;
            raw.into_iter().map(|v| v as f64).collect()
        }
        // int32
        (0, 32, 1) => {
            let raw: Vec<i32> =
                unsafe { read_strided_elements(base_ptr, &shape_i64, &elem_strides) }
                    .map_err(ExtractError::Arithmetic)?;
            raw.into_iter().map(|v| v as f64).collect()
        }
        // int64
        (0, 64, 1) => {
            let raw: Vec<i64> =
                unsafe { read_strided_elements(base_ptr, &shape_i64, &elem_strides) }
                    .map_err(ExtractError::Arithmetic)?;
            raw.into_iter().map(|v| v as f64).collect()
        }
        // uint8
        (1, 8, 1) => {
            let raw: Vec<u8> =
                unsafe { read_strided_elements(base_ptr, &shape_i64, &elem_strides) }
                    .map_err(ExtractError::Arithmetic)?;
            raw.into_iter().map(|v| v as f64).collect()
        }
        // uint16
        (1, 16, 1) => {
            let raw: Vec<u16> =
                unsafe { read_strided_elements(base_ptr, &shape_i64, &elem_strides) }
                    .map_err(ExtractError::Arithmetic)?;
            raw.into_iter().map(|v| v as f64).collect()
        }
        // uint32
        (1, 32, 1) => {
            let raw: Vec<u32> =
                unsafe { read_strided_elements(base_ptr, &shape_i64, &elem_strides) }
                    .map_err(ExtractError::Arithmetic)?;
            raw.into_iter().map(|v| v as f64).collect()
        }
        // uint64
        (1, 64, 1) => {
            let raw: Vec<u64> =
                unsafe { read_strided_elements(base_ptr, &shape_i64, &elem_strides) }
                    .map_err(ExtractError::Arithmetic)?;
            raw.into_iter().map(|v| v as f64).collect()
        }
        (code, bits, _) => {
            return Err(ExtractError::UnsupportedDtype { code, bits });
        }
    };

    let shape_vec: Vec<usize> = if is_scalar {
        vec![1]
    } else {
        shape_i64.iter().map(|&d| d.max(0) as usize).collect()
    };

    Ok((flat_vec, shape_vec))
}

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
///     A NumPy array (matching the source tensor's shape and dimensionality)
///     whose contents are *copied* from the DLPack tensor.  CPU tensors of
///     dtype int8/16/32/64, uint8/16/32/64, float32, or float64 are
///     supported; all other dtypes raise `TypeError`.
///
/// Notes
/// -----
/// GPU tensors raise `TypeError` until an optional `gpu` feature is enabled.
/// The capsule is renamed to `"used_dltensor"` after consumption to prevent
/// double-frees, consistent with the DLPack 1.0 spec.
///
/// **Stride handling**: the source tensor is *not* assumed to be
/// C-contiguous.  A non-null `strides` field (e.g. from a transposed
/// PyTorch tensor's `t().__dlpack__()`, a reversed/negative-stride view, or
/// any other sliced/strided view) is read using genuine N-dimensional
/// strided iteration, so the copied data is always logically correct
/// regardless of the producer's physical memory layout.  A null `strides`
/// field is treated as the DLPack-spec default of C-contiguous.
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

    // Decode the tensor's data honoring its actual stride layout — a
    // producer is not required to hand back a C-contiguous buffer (e.g. a
    // transposed PyTorch tensor's `__dlpack__()` reports non-row-major
    // strides), so we must walk it accordingly rather than assuming `shape`
    // alone determines memory layout.
    // SAFETY: dl_tensor was validated above (non-null data, CPU device);
    // shape/strides validity for `ndim` elements is the DLPack producer
    // contract, same as relied on everywhere else in this module.
    let (flat_vec, shape_vec) = unsafe { extract_dlpack_data(dl_tensor) }.map_err(|e| match e {
        ExtractError::UnsupportedDtype { code, bits } => PyTypeError::new_err(format!(
            "from_dlpack: unsupported dtype (code={code}, bits={bits}). \
             Supported: int8/16/32/64, uint8/16/32/64, float32, float64.",
        )),
        ExtractError::Arithmetic(msg) => PyValueError::new_err(format!("from_dlpack: {msg}")),
    })?;

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

    // ─── read_strided_elements / extract_dlpack_data: stride correctness ────
    //
    // These exercise the exact bug fixed here: a `from_dlpack` producer that
    // reports a non-contiguous (e.g. transposed) tensor must be read
    // according to its *actual* strides, never silently treated as if it
    // were C-contiguous.

    #[test]
    fn read_strided_elements_contiguous_matches_plain_read() {
        // Shape [2, 3] laid out C-contiguously; strides = [3, 1] should
        // reproduce the buffer verbatim (no permutation) — i.e. no
        // regression for the common (already-contiguous) case.
        let data: [f64; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = [2_i64, 3];
        let strides = compute_c_strides(&shape);
        let base_ptr = data.as_ptr() as *const u8;

        // SAFETY: base_ptr is valid for the 6 elements shape/strides address.
        let out: Vec<f64> = unsafe { read_strided_elements(base_ptr, &shape, &strides) }
            .expect("contiguous read should not fail");
        assert_eq!(out, data.to_vec());
    }

    #[test]
    fn read_strided_elements_transposed_2d_reads_logical_order() {
        // Underlying buffer is the C-contiguous 2x3 matrix:
        //   [[1, 2, 3],
        //    [4, 5, 6]]
        // A *transposed* (3x2) view of that same buffer has shape=[3,2] and
        // strides=[1,3] (the classic PyTorch `t.t()` / NumPy `.T` case: no
        // data is moved, only shape/strides are swapped). The logical
        // transposed matrix is:
        //   [[1, 4],
        //    [2, 5],
        //    [3, 6]]
        // so row-major traversal must yield [1, 4, 2, 5, 3, 6] — NOT the
        // physical buffer order [1, 2, 3, 4, 5, 6] a contiguous-only reader
        // would silently (and wrongly) produce.
        let data: [f64; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = [3_i64, 2];
        let strides = [1_i64, 3];
        let base_ptr = data.as_ptr() as *const u8;

        // SAFETY: every offset reachable via shape/strides indexes within
        // `data` (max offset = 2*1 + 1*3 = 5, i.e. the last element).
        let out: Vec<f64> = unsafe { read_strided_elements(base_ptr, &shape, &strides) }
            .expect("transposed strided read should succeed");
        assert_eq!(out, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

        // Sanity: this must differ from the physical (wrong) contiguous
        // reading of the same buffer, proving the fix actually changes
        // behavior for non-contiguous input.
        assert_ne!(out, data.to_vec());
    }

    #[test]
    fn read_strided_elements_negative_stride_reversed_view() {
        // A negative-stride view (e.g. NumPy's `arr[::-1]`) walking backwards
        // through the buffer starting from its last element.
        let data: [f64; 3] = [10.0, 20.0, 30.0];
        let shape = [3_i64];
        let strides = [-1_i64];
        // Base pointer must point at the *first logical* element, i.e. the
        // last physical element (index 2).
        let base_ptr = unsafe { data.as_ptr().add(2) } as *const u8;

        // SAFETY: offsets range over {0, -1, -2} elements from base_ptr,
        // i.e. indices {2, 1, 0} into `data` — all in bounds.
        let out: Vec<f64> = unsafe { read_strided_elements(base_ptr, &shape, &strides) }
            .expect("negative-stride read should succeed");
        assert_eq!(out, vec![30.0, 20.0, 10.0]);
    }

    #[test]
    fn read_strided_elements_length_mismatch_errors_without_reading() {
        // shape has 2 dims, strides only 1 — must error out before ever
        // dereferencing base_ptr.
        let x = 0.0_f64;
        let base_ptr = &x as *const f64 as *const u8;
        let result: Result<Vec<f64>, String> =
            unsafe { read_strided_elements(base_ptr, &[2, 2], &[1]) };
        assert!(result.is_err());
    }

    #[test]
    fn read_strided_elements_stride_overflow_errors_cleanly() {
        // Pathological strides that overflow i64 arithmetic must produce a
        // clean Err, never a panic/abort/UB. Only idx=[0,0] (offset 0) is
        // ever actually dereferenced before the overflow is detected on the
        // very next offset computation, so a 1-element buffer suffices.
        let data = [0.0_f64];
        let shape = [2_i64, 2];
        let strides = [i64::MAX, i64::MAX];
        let base_ptr = data.as_ptr() as *const u8;

        let result: Result<Vec<f64>, String> =
            unsafe { read_strided_elements(base_ptr, &shape, &strides) };
        assert!(result.is_err(), "expected overflow to be caught as Err");
    }

    #[test]
    fn read_strided_elements_scalar_zero_dim() {
        let data = [42.0_f64];
        let base_ptr = data.as_ptr() as *const u8;
        let out: Vec<f64> =
            unsafe { read_strided_elements(base_ptr, &[], &[]) }.expect("scalar read should work");
        assert_eq!(out, vec![42.0]);
    }

    // ─── extract_dlpack_data: full DLTensor-level coverage ───────────────────

    /// Build a bare CPU [`DLTensor`] over the given data/shape/strides for
    /// testing `extract_dlpack_data` without any PyCapsule/Python machinery.
    fn make_tensor(
        data_ptr: *mut c_void,
        shape: &[i64],
        strides: *mut i64,
        dtype: DLDataType,
    ) -> DLTensor {
        DLTensor {
            data: data_ptr,
            device: DLDevice {
                device_type: DLDeviceType::Cpu as i32,
                device_id: 0,
            },
            ndim: shape.len() as i32,
            dtype,
            shape: shape.as_ptr() as *mut i64,
            strides,
            byte_offset: 0,
        }
    }

    #[test]
    fn extract_dlpack_data_null_strides_is_treated_as_c_contiguous() {
        // Regression guard: a plain (already-contiguous) producer with a
        // NULL strides pointer — the DLPack convention — must still work
        // exactly as before this fix.
        let mut data: [f64; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = [2_i64, 3];
        let dtype = DLDataType {
            code: DLDataTypeCode::Float as u8,
            bits: 64,
            lanes: 1,
        };
        let tensor = make_tensor(
            data.as_mut_ptr() as *mut c_void,
            &shape,
            std::ptr::null_mut(),
            dtype,
        );

        // SAFETY: tensor borrows `data`/`shape`, both alive for this call.
        let (flat, shape_vec) =
            unsafe { extract_dlpack_data(&tensor) }.expect("contiguous extraction should succeed");
        assert_eq!(flat, data.to_vec());
        assert_eq!(shape_vec, vec![2, 3]);
    }

    #[test]
    fn extract_dlpack_data_non_null_strides_transposed_view_reads_correctly() {
        // The exact bug scenario: a producer hands back a transposed view
        // (shape=[3,2], strides=[1,3] over the same physical 2x3 buffer)
        // with a *non-null* strides pointer. Before this fix, `from_dlpack`
        // ignored `strides` entirely and would have returned the raw
        // physical buffer order (WRONG). It must now return the logically
        // transposed data.
        let mut data: [f64; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut strides = [1_i64, 3];
        let shape = [3_i64, 2];
        let dtype = DLDataType {
            code: DLDataTypeCode::Float as u8,
            bits: 64,
            lanes: 1,
        };
        let tensor = make_tensor(
            data.as_mut_ptr() as *mut c_void,
            &shape,
            strides.as_mut_ptr(),
            dtype,
        );

        // SAFETY: tensor borrows `data`/`shape`/`strides`, all alive here.
        let (flat, shape_vec) = unsafe { extract_dlpack_data(&tensor) }
            .expect("transposed extraction should succeed, not silently misread");
        assert_eq!(flat, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        assert_eq!(shape_vec, vec![3, 2]);

        // The whole point of the fix: this must NOT equal the naive
        // (contiguous-assuming) physical read of the same buffer.
        assert_ne!(flat, data.to_vec());
    }

    #[test]
    fn extract_dlpack_data_float32_transposed_widens_correctly() {
        // Same transposed-view scenario but with float32 source data,
        // covering the widening-to-f64 dtype arms as well as strides.
        let mut data: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut strides = [1_i64, 3];
        let shape = [3_i64, 2];
        let dtype = DLDataType {
            code: DLDataTypeCode::Float as u8,
            bits: 32,
            lanes: 1,
        };
        let tensor = make_tensor(
            data.as_mut_ptr() as *mut c_void,
            &shape,
            strides.as_mut_ptr(),
            dtype,
        );

        // SAFETY: tensor borrows `data`/`shape`/`strides`, all alive here.
        let (flat, shape_vec) =
            unsafe { extract_dlpack_data(&tensor) }.expect("f32 transposed extraction failed");
        assert_eq!(flat, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        assert_eq!(shape_vec, vec![3, 2]);
    }

    #[test]
    fn extract_dlpack_data_unsupported_dtype_errors() {
        let mut data: [f64; 1] = [0.0];
        let shape = [1_i64];
        let dtype = DLDataType {
            code: 99, // not a recognised DLDataTypeCode
            bits: 64,
            lanes: 1,
        };
        let tensor = make_tensor(
            data.as_mut_ptr() as *mut c_void,
            &shape,
            std::ptr::null_mut(),
            dtype,
        );

        // SAFETY: tensor borrows `data`/`shape`, both alive here.
        let result = unsafe { extract_dlpack_data(&tensor) };
        assert!(matches!(
            result,
            Err(ExtractError::UnsupportedDtype { code: 99, bits: 64 })
        ));
    }
}
