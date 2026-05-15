//! CUDA / GPU tensor passthrough via DLPack.
//!
//! When a DLPack capsule contains a tensor resident on CUDA (or another
//! non-CPU device), naively trying to consume it as a CPU `ndarray` would
//! either panic or silently trigger an unacceptable host-device copy.
//!
//! This module provides:
//!
//! - [`CudaTensorInfo`] — metadata extracted from a CUDA DLPack tensor
//!   without triggering any data copy.
//! - [`cuda_tensor_info_from_dltensor`] — pure-Rust function operating
//!   directly on a [`DLTensor`]; no Python runtime needed.
//! - `dlpack_auto_dispatch_f32` / `dlpack_auto_dispatch_f64` — device-
//!   aware dispatch that returns an `ndarray` view for CPU tensors, or
//!   `CudaTensorInfo` for GPU tensors, with **no host-device copy**.
//!
//! # Design
//!
//! The DLPack standard defines device type codes:
//!
//! | Code | Device |
//! |------|--------|
//! | 1    | CPU    |
//! | 2    | CUDA   |
//! | 3    | CUDA pinned host |
//! | 4    | OpenCL |
//! | 7    | Vulkan |
//! | 8    | Metal  |
//! | 10   | ROCm   |
//!
//! For CPU tensors (type 1) the existing zero-copy `array_from_dlpack_f32/f64`
//! functions are used directly.  For CUDA tensors (type 2) we extract shape,
//! dtype, device_id, and byte_offset without touching the data pointer.
//!
//! # CUDA runtime linkage
//!
//! Full GPU-to-GPU processing (e.g. copying the tensor buffer to a
//! `cudarc`-managed allocation) requires the `cuda_special` cargo feature
//! and CUDA runtime linkage, which is deliberately kept **out of default
//! features** to preserve the Pure Rust build.  With default features only
//! the metadata extraction path is available.

use std::ffi::CStr;

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::dlpack::{
    array_from_dlpack_f32, array_from_dlpack_f64, DLDataType, DLDeviceType, DLManagedTensor,
    DLTensor, DlpackError,
};

/// DLPack capsule name used when the capsule has not yet been consumed.
const DLTENSOR_NAME: &CStr = c"dltensor";

/// DLPack capsule name used after the capsule has been consumed by a framework.
///
/// Attempting to call [`cuda_tensor_info`] on an already-consumed capsule raises
/// [`pyo3::exceptions::PyValueError`].
const USED_DLTENSOR_NAME: &CStr = c"used_dltensor";

/// Metadata extracted from a CUDA-resident DLPack tensor.
///
/// Accessing this struct **does not** copy tensor data from GPU to CPU.
/// It records only the shape, dtype, device index, and byte offset that are
/// safe to read from the capsule header.
#[derive(Debug, Clone)]
pub struct CudaTensorInfo {
    /// Zero-based index of the CUDA device (e.g., 0 for the first GPU).
    pub device_id: i32,
    /// Tensor dimensions in row-major (C) order.
    pub shape: Vec<usize>,
    /// DLPack element data-type descriptor.
    pub dtype: DLDataType,
    /// Byte offset from the data pointer to the first element.
    pub byte_offset: u64,
    /// Raw device-type code (2 = CUDA, 10 = ROCm, etc.).
    pub device_type_code: i32,
}

impl CudaTensorInfo {
    /// Return the total number of elements (product of shape dimensions).
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Return the element bit-width.
    pub fn dtype_bits(&self) -> u8 {
        self.dtype.bits
    }

    /// Return a human-readable device string (e.g. `"cuda:0"`).
    pub fn device_str(&self) -> String {
        let name = match self.device_type_code {
            2 => "cuda",
            3 => "cuda_host",
            4 => "opencl",
            7 => "vulkan",
            8 => "metal",
            10 => "rocm",
            _ => "unknown",
        };
        format!("{}:{}", name, self.device_id)
    }
}

/// The result of auto-dispatching a DLPack tensor based on its device.
///
/// No host-device copy is performed for [`DLPackDispatchResult::Gpu`].
pub enum DLPackDispatchResult<'a, T> {
    /// The tensor is on CPU and has been zero-copy viewed as an `ndarray`.
    Cpu(ndarray::ArrayViewD<'a, T>),
    /// The tensor is on a GPU (or other accelerator) — metadata is returned
    /// without touching the data buffer.
    Gpu(CudaTensorInfo),
    /// The tensor is on an unrecognised device.
    OtherDevice {
        /// DLPack device-type code.
        device_type: i32,
        /// Device index.
        device_id: i32,
    },
}

/// Extract [`CudaTensorInfo`] from a raw [`DLTensor`] pointer.
///
/// This is the **pure-Rust** implementation that does not require a Python
/// runtime; the PyO3 wrappers delegate to this function after extracting the
/// `DLTensor` from the capsule.
///
/// # Errors
///
/// Returns [`DlpackError::NonCpuDevice`] when `tensor.device.device_type == 1`
/// (CPU), because `CudaTensorInfo` is only meaningful for non-CPU devices.
///
/// Returns [`DlpackError::NullPointer`] when `tensor.data` is null.
///
/// # Safety
///
/// `tensor.shape` must be valid for `tensor.ndim` elements.
pub fn cuda_tensor_info_from_dltensor(tensor: &DLTensor) -> Result<CudaTensorInfo, DlpackError> {
    if tensor.data.is_null() {
        return Err(DlpackError::NullPointer);
    }
    // Reject CPU tensors — this function is for non-CPU devices.
    if tensor.device.device_type == DLDeviceType::Cpu as i32 {
        return Err(DlpackError::NonCpuDevice);
    }
    let ndim = tensor.ndim.max(0) as usize;
    let shape = if ndim == 0 || tensor.shape.is_null() {
        Vec::new()
    } else {
        // SAFETY: caller guarantees shape is valid for ndim elements.
        unsafe { std::slice::from_raw_parts(tensor.shape as *const i64, ndim) }
            .iter()
            .map(|&d| d as usize)
            .collect()
    };
    Ok(CudaTensorInfo {
        device_id: tensor.device.device_id,
        shape,
        dtype: tensor.dtype,
        byte_offset: tensor.byte_offset,
        device_type_code: tensor.device.device_type,
    })
}

/// Dispatch an `f32` DLPack tensor to CPU or GPU path without a CPU roundtrip.
///
/// - **CPU (device_type=1)**: zero-copy `ndarray::ArrayViewD<f32>`.
/// - **GPU/accelerator (device_type≠1)**: metadata only, no data copy.
///
/// # Safety
///
/// `tensor` must be a valid, aligned, non-null pointer to a [`DLTensor`]
/// whose `shape` field is valid for `ndim` elements.  The tensor and its data
/// must remain live and unmodified for the lifetime `'a` of the returned view.
pub unsafe fn dlpack_auto_dispatch_f32<'a>(
    tensor: *const DLTensor,
) -> Result<DLPackDispatchResult<'a, f32>, DlpackError> {
    // SAFETY: caller guarantees tensor is valid.
    let t = unsafe { &*tensor };
    match t.device.device_type {
        dt if dt == DLDeviceType::Cpu as i32 => {
            // SAFETY: forwarded from caller invariants.
            let view = unsafe { array_from_dlpack_f32(tensor)? };
            Ok(DLPackDispatchResult::Cpu(view))
        }
        _ => {
            let info = cuda_tensor_info_from_dltensor(t)?;
            Ok(DLPackDispatchResult::Gpu(info))
        }
    }
}

/// Dispatch an `f64` DLPack tensor to CPU or GPU path without a CPU roundtrip.
///
/// Same semantics as [`dlpack_auto_dispatch_f32`] but for 64-bit floats.
///
/// # Safety
///
/// Same invariants as [`dlpack_auto_dispatch_f32`].
pub unsafe fn dlpack_auto_dispatch_f64<'a>(
    tensor: *const DLTensor,
) -> Result<DLPackDispatchResult<'a, f64>, DlpackError> {
    // SAFETY: caller guarantees tensor is valid.
    let t = unsafe { &*tensor };
    match t.device.device_type {
        dt if dt == DLDeviceType::Cpu as i32 => {
            // SAFETY: forwarded from caller invariants.
            let view = unsafe { array_from_dlpack_f64(tensor)? };
            Ok(DLPackDispatchResult::Cpu(view))
        }
        _ => {
            let info = cuda_tensor_info_from_dltensor(t)?;
            Ok(DLPackDispatchResult::Gpu(info))
        }
    }
}

// ─── Python-facing capsule API ───────────────────────────────────────────────

/// Extract [`CudaTensorInfo`] from a Python DLPack capsule object.
///
/// Accepts any Python object that wraps a `"dltensor"` PyCapsule — typically
/// the return value of `tensor.__dlpack__()` on a PyTorch, JAX, or CuPy GPU
/// tensor.
///
/// # Errors
///
/// Returns `PyValueError` when:
/// - the object is not a PyCapsule named `"dltensor"`,
/// - the capsule has already been consumed (name `"used_dltensor"`),
/// - the underlying tensor is CPU-resident (use the regular DLPack CPU path),
/// - the data pointer inside the managed tensor is null.
///
/// # Example (Python side)
///
/// ```python
/// import torch
/// t = torch.zeros(4, 4, device="cuda")
/// capsule = t.__dlpack__()
/// info = scirs2_numpy.get_cuda_tensor_info(capsule)
/// # info == {"device_id": 0, "shape": [4, 4], "device_type": 2, "device_str": "cuda:0"}
/// ```
pub fn cuda_tensor_info(capsule: &Bound<'_, PyAny>) -> PyResult<CudaTensorInfo> {
    // ── Step 1: obtain the raw PyObject* ────────────────────────────────────
    let raw_obj: *mut pyo3::ffi::PyObject = capsule.as_ptr();

    // ── Step 2: detect "used_dltensor" and give a clear error ───────────────
    // PyCapsule_IsValid returns 1 when the capsule exists and its name matches.
    // We check `used_dltensor` first so the error message is actionable.
    let is_used =
        unsafe { pyo3::ffi::PyCapsule_IsValid(raw_obj, USED_DLTENSOR_NAME.as_ptr()) == 1 };
    if is_used {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "DLPack capsule has already been consumed ('used_dltensor'). \
             Call __dlpack__() again on the original tensor.",
        ));
    }

    // ── Step 3: retrieve the dltensor pointer ───────────────────────────────
    let raw_ptr = unsafe { pyo3::ffi::PyCapsule_GetPointer(raw_obj, DLTENSOR_NAME.as_ptr()) };

    if raw_ptr.is_null() {
        // PyCapsule_GetPointer sets a Python exception when it fails.
        // Propagate it as a PyErr.
        return Err(PyErr::fetch(capsule.py()));
    }

    // ── Step 4: dereference the managed tensor to get DLTensor ─────────────
    // SAFETY: `raw_ptr` was returned by PyCapsule_GetPointer with the correct
    // capsule name; it points to the DLManagedTensor stored when the capsule
    // was created.  The capsule (and therefore this allocation) remains live
    // as long as `capsule` is live.
    let managed = unsafe { &*(raw_ptr as *const DLManagedTensor) };
    let dl_tensor = &managed.dl_tensor;

    // ── Step 5: delegate to pure-Rust extractor ─────────────────────────────
    cuda_tensor_info_from_dltensor(dl_tensor).map_err(|e| match e {
        DlpackError::NonCpuDevice => pyo3::exceptions::PyValueError::new_err(
            "cuda_tensor_info requires a non-CPU DLPack tensor. \
             Use the standard DLPack CPU path for CPU tensors.",
        ),
        DlpackError::NullPointer => {
            pyo3::exceptions::PyValueError::new_err("DLPack tensor has a null data pointer.")
        }
        other => pyo3::exceptions::PyValueError::new_err(format!("DLPack error: {other}")),
    })
}

/// Python-facing function: extract GPU tensor metadata from a DLPack capsule.
///
/// Accepts the capsule returned by `tensor.__dlpack__()` and returns a dict:
///
/// ```text
/// {
///   "device_id":   int,   # zero-based GPU index
///   "shape":       list,  # tensor dimensions
///   "device_type": int,   # raw DLPack device code (2=CUDA, 10=ROCm, …)
///   "device_str":  str,   # human-readable, e.g. "cuda:0"
/// }
/// ```
///
/// Raises `ValueError` for CPU tensors, consumed capsules, or null data pointers.
#[pyfunction]
pub fn get_cuda_tensor_info(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Py<PyDict>> {
    let info = cuda_tensor_info(obj)?;

    let dict = PyDict::new(py);
    dict.set_item("device_id", info.device_id)?;
    dict.set_item("shape", info.shape.clone())?;
    dict.set_item("device_type", info.device_type_code)?;
    dict.set_item("device_str", info.device_str())?;
    Ok(dict.into())
}

/// Register the `get_cuda_tensor_info` function into a PyO3 module.
///
/// Call this from your `#[pymodule]` init function to expose the function to Python.
pub fn register_dlpack_cuda_module(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_cuda_tensor_info, m)?)?;
    Ok(())
}

// ─── Testing helpers ─────────────────────────────────────────────────────────

/// Build a mock non-CPU [`DLTensor`] for testing (never accesses data pointer).
///
/// The `data` pointer is set to a sentinel non-null value; **do not** read from it.
#[cfg(test)]
fn make_non_cpu_dltensor(device_type: i32, device_id: i32, shape: &[i64]) -> DLTensor {
    use crate::dlpack::{DLDataTypeCode, DLDevice};
    use std::ffi::c_void;
    // Sentinel non-null data pointer (never dereferenced in metadata-only path).
    static SENTINEL: u8 = 0;
    DLTensor {
        data: &SENTINEL as *const u8 as *mut c_void,
        device: DLDevice {
            device_type,
            device_id,
        },
        ndim: shape.len() as i32,
        dtype: DLDataType {
            code: DLDataTypeCode::Float as u8,
            bits: 32,
            lanes: 1,
        },
        shape: shape.as_ptr() as *mut i64,
        strides: std::ptr::null_mut(),
        byte_offset: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dlpack::{dlpack_from_slice, DLDeviceType};
    use std::ffi::c_void;

    // ─── cuda_tensor_info_from_dltensor ──────────────────────────────────────

    #[test]
    fn test_cuda_tensor_info_rejects_cpu_device() {
        let data = [1.0_f64, 2.0, 3.0];
        let shape = [3_i64];
        let tensor = dlpack_from_slice(&data, &shape);
        // CPU tensor (device_type=1) must be rejected.
        let result = cuda_tensor_info_from_dltensor(&tensor);
        assert!(
            matches!(result, Err(DlpackError::NonCpuDevice)),
            "CPU tensor should be rejected by cuda_tensor_info_from_dltensor"
        );
    }

    #[test]
    fn test_cuda_tensor_info_rejects_null_data() {
        let shape = [4_i64, 4];
        let mut tensor = make_non_cpu_dltensor(2, 0, &shape);
        tensor.data = std::ptr::null_mut();
        let result = cuda_tensor_info_from_dltensor(&tensor);
        assert!(
            matches!(result, Err(DlpackError::NullPointer)),
            "null data pointer should be rejected"
        );
    }

    #[test]
    fn test_cuda_tensor_info_extracts_shape() {
        let shape = [3_i64, 4, 5];
        let tensor = make_non_cpu_dltensor(2, 0, &shape);
        let info = cuda_tensor_info_from_dltensor(&tensor)
            .expect("CUDA tensor should produce CudaTensorInfo");
        assert_eq!(info.shape, vec![3, 4, 5], "shape mismatch");
        assert_eq!(info.numel(), 60, "numel mismatch");
    }

    #[test]
    fn test_cuda_tensor_info_extracts_device_id() {
        let shape = [8_i64];
        let tensor = make_non_cpu_dltensor(2, 3, &shape); // device_id = 3 (4th GPU)
        let info = cuda_tensor_info_from_dltensor(&tensor).expect("should produce CudaTensorInfo");
        assert_eq!(info.device_id, 3, "device_id mismatch");
        assert_eq!(
            info.device_type_code, 2,
            "device_type_code should be CUDA (2)"
        );
    }

    #[test]
    fn test_cuda_tensor_info_device_str() {
        let shape = [1_i64];
        let tensor = make_non_cpu_dltensor(2, 0, &shape);
        let info = cuda_tensor_info_from_dltensor(&tensor).expect("should succeed");
        assert_eq!(info.device_str(), "cuda:0");
    }

    #[test]
    fn test_rocm_tensor_info_device_str() {
        let shape = [1_i64];
        let tensor = make_non_cpu_dltensor(10, 1, &shape); // ROCm device 1
        let info = cuda_tensor_info_from_dltensor(&tensor).expect("should succeed");
        assert_eq!(info.device_str(), "rocm:1");
    }

    #[test]
    fn test_cuda_tensor_info_zero_dim_tensor() {
        // ndim=0, shape ptr null.
        use crate::dlpack::{DLDataType, DLDataTypeCode};
        static SENTINEL: u8 = 0;
        let tensor = DLTensor {
            data: &SENTINEL as *const u8 as *mut c_void,
            device: crate::dlpack::DLDevice {
                device_type: 2,
                device_id: 0,
            },
            ndim: 0,
            dtype: DLDataType {
                code: DLDataTypeCode::Float as u8,
                bits: 32,
                lanes: 1,
            },
            shape: std::ptr::null_mut(),
            strides: std::ptr::null_mut(),
            byte_offset: 0,
        };
        let info = cuda_tensor_info_from_dltensor(&tensor).expect("zero-dim should succeed");
        assert!(info.shape.is_empty(), "zero-dim shape should be empty");
        assert_eq!(info.numel(), 1, "empty product is 1");
    }

    // ─── dlpack_auto_dispatch_f32 ────────────────────────────────────────────

    #[test]
    fn test_dlpack_auto_dispatch_cpu_f32_returns_array() {
        let data = [1.0_f32, 2.0, 3.0, 4.0];
        let shape = [2_i64, 2];
        let tensor = crate::dlpack::DLTensor {
            data: data.as_ptr() as *mut c_void,
            device: crate::dlpack::DLDevice {
                device_type: DLDeviceType::Cpu as i32,
                device_id: 0,
            },
            ndim: 2,
            dtype: crate::dlpack::DLDataType {
                code: crate::dlpack::DLDataTypeCode::Float as u8,
                bits: 32,
                lanes: 1,
            },
            shape: shape.as_ptr() as *mut i64,
            strides: std::ptr::null_mut(),
            byte_offset: 0,
        };
        // SAFETY: tensor is valid; data and shape are alive.
        let result = unsafe { dlpack_auto_dispatch_f32(&tensor as *const _) }
            .expect("CPU dispatch should succeed");
        assert!(
            matches!(result, DLPackDispatchResult::Cpu(_)),
            "CPU tensor should return Cpu variant"
        );
        if let DLPackDispatchResult::Cpu(view) = result {
            assert_eq!(view.shape(), &[2, 2]);
            assert_eq!(view[[0, 0]], 1.0_f32);
        }
    }

    #[test]
    fn test_dlpack_auto_dispatch_cuda_f32_returns_gpu_info() {
        let shape = [8_i64];
        let tensor = make_non_cpu_dltensor(DLDeviceType::Cuda as i32, 0, &shape);
        // SAFETY: tensor is valid; shape is alive.
        let result = unsafe { dlpack_auto_dispatch_f32(&tensor as *const _) }
            .expect("CUDA dispatch should succeed");
        assert!(
            matches!(result, DLPackDispatchResult::Gpu(_)),
            "CUDA tensor should return Gpu variant"
        );
        if let DLPackDispatchResult::Gpu(info) = result {
            assert_eq!(info.shape, vec![8]);
            assert_eq!(info.device_type_code, 2);
        }
    }

    #[test]
    fn test_dlpack_auto_dispatch_cpu_f64_returns_array() {
        let data = [10.0_f64, 20.0, 30.0];
        let shape = [3_i64];
        let tensor = dlpack_from_slice(&data, &shape);
        // SAFETY: tensor is valid; data and shape are alive.
        let result = unsafe { dlpack_auto_dispatch_f64(&tensor as *const _) }
            .expect("CPU f64 dispatch should succeed");
        assert!(
            matches!(result, DLPackDispatchResult::Cpu(_)),
            "CPU f64 tensor should return Cpu variant"
        );
    }

    #[test]
    fn test_dlpack_auto_dispatch_cuda_f64_returns_gpu_info() {
        let shape = [4_i64, 4];
        let tensor = make_non_cpu_dltensor(DLDeviceType::Cuda as i32, 1, &shape);
        // SAFETY: tensor is valid; shape is alive.
        let result = unsafe { dlpack_auto_dispatch_f64(&tensor as *const _) }
            .expect("CUDA f64 dispatch should succeed");
        if let DLPackDispatchResult::Gpu(info) = result {
            assert_eq!(info.shape, vec![4, 4]);
            assert_eq!(info.device_id, 1);
        } else {
            panic!("expected Gpu variant");
        }
    }

    #[test]
    fn test_dlpack_other_device_passthrough() {
        // Metal (device_type=8) should also return Gpu variant.
        let shape = [16_i64];
        let tensor = make_non_cpu_dltensor(DLDeviceType::Metal as i32, 0, &shape);
        // SAFETY: tensor is valid; shape is alive.
        let result = unsafe { dlpack_auto_dispatch_f32(&tensor as *const _) }
            .expect("Metal dispatch should succeed");
        assert!(
            matches!(result, DLPackDispatchResult::Gpu(_)),
            "Metal tensor should return Gpu variant"
        );
        if let DLPackDispatchResult::Gpu(info) = result {
            assert_eq!(info.device_str(), "metal:0");
        }
    }

    #[test]
    fn test_dlpack_rocm_passthrough() {
        let shape = [32_i64];
        let tensor = make_non_cpu_dltensor(DLDeviceType::Rocm as i32, 2, &shape);
        // SAFETY: tensor is valid; shape is alive.
        let result = unsafe { dlpack_auto_dispatch_f64(&tensor as *const _) }
            .expect("ROCm dispatch should succeed");
        if let DLPackDispatchResult::Gpu(info) = result {
            assert_eq!(info.device_type_code, 10);
            assert_eq!(info.device_id, 2);
        } else {
            panic!("expected Gpu variant for ROCm device");
        }
    }

    #[test]
    fn test_cuda_tensor_numel_empty_shape() {
        let shape: [i64; 0] = [];
        let tensor = make_non_cpu_dltensor(2, 0, &shape);
        // ndim=0, shape ptr points to empty slice.
        let info = cuda_tensor_info_from_dltensor(&tensor).expect("should succeed");
        assert_eq!(info.numel(), 1, "empty shape product is 1");
    }

    #[test]
    fn test_cuda_tensor_dtype_bits() {
        let shape = [4_i64];
        let tensor = make_non_cpu_dltensor(2, 0, &shape);
        let info = cuda_tensor_info_from_dltensor(&tensor).expect("should succeed");
        assert_eq!(info.dtype_bits(), 32, "dtype bits should be 32");
    }
}
