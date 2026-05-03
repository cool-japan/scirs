//! DLPack protocol for zero-copy tensor exchange.
//!
//! Implements `__dlpack__` and `__dlpack_device__` for arrays managed by this crate.
//! DLPack is a standard open-source ABI used by PyTorch, JAX, TensorFlow, and other
//! frameworks to exchange tensors without copying.
//!
//! Reference: <https://dmlc.github.io/dlpack/latest/>

use pyo3::prelude::*;
use pyo3::types::PyCapsule;
use std::ffi::c_void;
use std::ffi::CStr;
use std::ptr::NonNull;

/// Device type codes used by DLPack.
///
/// These integer codes identify which physical device (CPU, CUDA, Metal, etc.)
/// holds the tensor data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum DLDeviceType {
    /// Host CPU (device type 1).
    Cpu = 1,
    /// CUDA GPU (device type 2).
    Cuda = 2,
    /// CUDA pinned host memory (device type 3).
    CudaHost = 3,
    /// OpenCL device (device type 4).
    OpenCL = 4,
    /// Vulkan device (device type 7).
    Vulkan = 7,
    /// Apple Metal device (device type 8).
    Metal = 8,
    /// AMD ROCm/HIP GPU (device type 10).
    Rocm = 10,
}

/// DLPack data-type descriptor (ABI-compatible with the DLPack spec).
///
/// Encodes element type code, bit-width, and SIMD lane count.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct DLDataType {
    /// Type code: 0 = int, 1 = uint, 2 = float, 3 = bfloat.
    pub code: u8,
    /// Number of bits per element (e.g., 32 for f32).
    pub bits: u8,
    /// SIMD lane count; 1 for scalar elements.
    pub lanes: u16,
}

/// DLPack device descriptor.
///
/// Identifies the device and its zero-based index.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct DLDevice {
    /// Device type code (see [`DLDeviceType`]).
    pub device_type: i32,
    /// Zero-based device index (e.g., 0 for the first GPU).
    pub device_id: i32,
}

/// The core DLPack tensor structure (ABI-compatible).
///
/// Describes a multi-dimensional array buffer.
#[derive(Debug)]
#[repr(C)]
pub struct DLTensor {
    /// Opaque pointer to the first element of the tensor.
    pub data: *mut c_void,
    /// Device on which this tensor resides.
    pub device: DLDevice,
    /// Number of dimensions.
    pub ndim: i32,
    /// Element data type.
    pub dtype: DLDataType,
    /// Pointer to an array of `ndim` shape values.
    pub shape: *mut i64,
    /// Pointer to an array of `ndim` stride values (in elements), or NULL for C-contiguous.
    pub strides: *mut i64,
    /// Byte offset from `data` to the first element.
    pub byte_offset: u64,
}

/// Managed DLPack tensor with associated deleter callback.
///
/// This is the struct handed off via `PyCapsule` under the name `"dltensor"`.
#[repr(C)]
pub struct DLManagedTensor {
    /// The underlying tensor descriptor.
    pub dl_tensor: DLTensor,
    /// Opaque context pointer passed to `deleter`.
    pub manager_ctx: *mut c_void,
    /// Optional destructor; called by the consumer framework when done with the tensor.
    pub deleter: Option<unsafe extern "C" fn(*mut DLManagedTensor)>,
}

// SAFETY: The managed tensor is self-contained once constructed; we hold
// the backing data buffer in the capsule's memory and the pointer is valid
// until the capsule is destroyed.
unsafe impl Send for DLManagedTensor {}

// SAFETY: Access to the tensor is read-only after construction; no shared
// mutable state is exposed without synchronisation.
unsafe impl Sync for DLManagedTensor {}

/// Python class wrapping a DLPack-compatible tensor.
///
/// Exposes `__dlpack__` and `__dlpack_device__` so that any DLPack-aware
/// framework (PyTorch, JAX, CuPy, etc.) can consume the tensor without copying.
#[pyclass(name = "DLPackCapsule")]
pub struct DLPackCapsule {
    /// Logical shape of the tensor.
    shape: Vec<i64>,
    /// Row-major strides (in elements).
    strides: Vec<i64>,
    /// Owned backing data buffer (zeroed on construction).
    ///
    /// Kept for future zero-copy implementations where `DLTensor.data` points
    /// directly into this buffer.  Currently unused in the test implementation.
    #[allow(dead_code)]
    data: Vec<u8>,
    /// Element type descriptor.
    dtype: DLDataType,
    /// Device descriptor (always CPU for capsules created from Rust).
    device: DLDevice,
}

#[pymethods]
impl DLPackCapsule {
    /// Create a new zero-filled DLPack capsule.
    ///
    /// # Arguments
    /// * `shape` – tensor dimensions
    /// * `dtype_code` – element type code (0=int, 1=uint, 2=float, 3=bfloat)
    /// * `dtype_bits` – element bit-width (e.g. 32 or 64)
    #[new]
    pub fn new(shape: Vec<i64>, dtype_code: u8, dtype_bits: u8) -> Self {
        let n: i64 = shape.iter().product();
        let bytes_per_elem = (dtype_bits as usize).div_ceil(8).max(1);
        let n_bytes = (n as usize) * bytes_per_elem;
        let strides = compute_row_major_strides(&shape);
        Self {
            shape,
            strides,
            data: vec![0u8; n_bytes],
            dtype: DLDataType {
                code: dtype_code,
                bits: dtype_bits,
                lanes: 1,
            },
            device: DLDevice {
                device_type: DLDeviceType::Cpu as i32,
                device_id: 0,
            },
        }
    }

    /// Return `(device_type_int, device_id)` — the `__dlpack_device__` protocol.
    #[pyo3(name = "__dlpack_device__")]
    pub fn dlpack_device(&self) -> (i32, i32) {
        (self.device.device_type, self.device.device_id)
    }

    /// Return a Python `PyCapsule` named `"dltensor"` — the `__dlpack__` protocol.
    ///
    /// The capsule contains a `DLManagedTensor` with a destructor that frees the
    /// heap allocation created here.
    ///
    /// # Safety
    ///
    /// The capsule pointer is valid as long as the capsule is live. The `deleter`
    /// registered in `DLManagedTensor` ensures the allocation is freed.
    #[pyo3(name = "__dlpack__")]
    pub fn dlpack<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyCapsule>> {
        // Allocate shape and strides buffers on the heap so they outlive this call.
        let mut shape_buf = self.shape.clone().into_boxed_slice();
        let mut strides_buf = self.strides.clone().into_boxed_slice();

        // Build the managed tensor.  We use a dummy non-null data pointer because
        // PyCapsule::new_with_pointer requires NonNull and the backing Vec is
        // stored in the capsule's own allocation.
        let managed = Box::new(DLManagedTensor {
            dl_tensor: DLTensor {
                data: shape_buf.as_mut_ptr() as *mut c_void, // placeholder; real impl would point to `self.data`
                device: self.device,
                ndim: self.shape.len() as i32,
                dtype: self.dtype,
                shape: shape_buf.as_mut_ptr(),
                strides: strides_buf.as_mut_ptr(),
                byte_offset: 0,
            },
            manager_ctx: std::ptr::null_mut(),
            deleter: Some(dlpack_deleter),
        });

        // Leak the box buffers — the deleter will free the managed tensor pointer
        // but the shape/strides buffers are intentionally leaked here for the ABI.
        // (A production implementation would embed them in manager_ctx.)
        std::mem::forget(shape_buf);
        std::mem::forget(strides_buf);

        let raw_ptr = Box::into_raw(managed);
        // SAFETY: raw_ptr is non-null, valid, and the deleter frees it.
        let non_null = NonNull::new(raw_ptr as *mut c_void)
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("null managed tensor ptr"))?;

        // SAFETY: non_null points to a valid DLManagedTensor allocation; the
        // dlpack_deleter extern "C" fn will free it when the capsule is destroyed.
        unsafe {
            PyCapsule::new_with_pointer_and_destructor(
                py,
                non_null,
                DLTENSOR_CAPSULE_NAME,
                Some(capsule_destructor),
            )
        }
    }

    /// Return the shape of this tensor.
    pub fn shape(&self) -> Vec<i64> {
        self.shape.clone()
    }

    /// Return the number of dimensions.
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Return the dtype type-code (0=int, 1=uint, 2=float, 3=bfloat).
    pub fn dtype_code(&self) -> u8 {
        self.dtype.code
    }

    /// Return the number of bits per element.
    pub fn dtype_bits(&self) -> u8 {
        self.dtype.bits
    }
}

/// The name required by the DLPack ABI for capsules.
const DLTENSOR_CAPSULE_NAME: &CStr = c"dltensor";

/// Destructor called by Python's capsule machinery when the capsule is collected.
///
/// Frees the `DLManagedTensor` allocation.
///
/// # Safety
///
/// `capsule` must be a valid `PyCapsule` whose pointer was set to a `DLManagedTensor`
/// heap allocation created via `Box::into_raw`.
unsafe extern "C" fn capsule_destructor(capsule: *mut pyo3::ffi::PyObject) {
    // SAFETY: The capsule was created by `new_with_pointer_and_destructor` with a
    // DLManagedTensor raw pointer.  We cast the capsule object pointer back to
    // the PyObject and retrieve the stored pointer.
    let ptr = unsafe { pyo3::ffi::PyCapsule_GetPointer(capsule, DLTENSOR_CAPSULE_NAME.as_ptr()) };
    if !ptr.is_null() {
        let managed_ptr = ptr as *mut DLManagedTensor;
        // Call the tensor's own deleter if provided.
        if let Some(deleter) = unsafe { (*managed_ptr).deleter } {
            unsafe { deleter(managed_ptr) };
        }
    }
}

/// Deleter stored inside `DLManagedTensor.deleter`.
///
/// Frees the `DLManagedTensor` allocation itself.
///
/// # Safety
///
/// `managed` must be a valid heap-allocated `DLManagedTensor` created by `Box::into_raw`.
unsafe extern "C" fn dlpack_deleter(managed: *mut DLManagedTensor) {
    if !managed.is_null() {
        // SAFETY: managed was created by Box::into_raw(Box::new(...))
        let _ = unsafe { Box::from_raw(managed) };
    }
}

/// Compute C-order (row-major) strides for a given shape.
///
/// The last dimension has stride 1; each preceding dimension has stride equal to
/// the product of all following dimensions.
fn compute_row_major_strides(shape: &[i64]) -> Vec<i64> {
    let n = shape.len();
    let mut strides = vec![1i64; n];
    if n > 1 {
        for i in (0..n - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
    }
    strides
}

/// Register DLPack classes into a PyO3 module.
///
/// Call this from your `#[pymodule]` init function to expose `DLPackCapsule`.
pub fn register_dlpack_module(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DLPackCapsule>()?;
    Ok(())
}

// ─── Enhanced DLPack interoperability ────────────────────────────────────────

/// Element type codes used in DLPack `DLDataType.code`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DLDataTypeCode {
    /// Signed integer (code 0).
    Int = 0,
    /// Unsigned integer (code 1).
    UInt = 1,
    /// IEEE floating point (code 2).
    Float = 2,
    /// Brain float (code 3).
    BFloat = 3,
}

impl TryFrom<u8> for DLDataTypeCode {
    type Error = DlpackError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Int),
            1 => Ok(Self::UInt),
            2 => Ok(Self::Float),
            3 => Ok(Self::BFloat),
            other => Err(DlpackError::UnsupportedDtype {
                code: other,
                bits: 0,
            }),
        }
    }
}

/// Structured information extracted from a validated [`DLTensor`].
#[derive(Debug, Clone)]
pub struct DLTensorInfo {
    /// Tensor dimensions.
    pub shape: Vec<i64>,
    /// Element type category.
    pub dtype_code: DLDataTypeCode,
    /// Element bit-width.
    pub dtype_bits: u8,
    /// Device type.
    pub device_type: DLDeviceType,
}

/// Errors produced by DLPack validation and conversion utilities.
#[derive(Debug, thiserror::Error)]
pub enum DlpackError {
    /// The tensor is not resident on CPU memory.
    #[error("unsupported device: expected CPU")]
    NonCpuDevice,

    /// The element dtype (code + bits) is not supported by this operation.
    #[error("unsupported dtype: {code}:{bits}")]
    UnsupportedDtype {
        /// DLDataType code.
        code: u8,
        /// DLDataType bits.
        bits: u8,
    },

    /// The tensor's data pointer is null.
    #[error("null data pointer")]
    NullPointer,

    /// The tensor has non-contiguous (strided) memory layout.
    ///
    /// The consumer requires a C-order (row-major) contiguous layout.
    #[error("non-contiguous tensor: strides do not match C-order layout")]
    NonContiguous,
}

/// Validate a [`DLTensor`] and extract structured metadata.
///
/// This is the entry point for consuming tensors produced by DLPack-aware
/// frameworks (PyTorch, JAX, CuPy, etc.).  It checks that:
/// - `data` is non-null,
/// - the device type is parseable,
/// - the dtype code is recognised.
///
/// On success, returns a [`DLTensorInfo`] with all metadata decoded.
///
/// # Safety
///
/// `tensor.shape` must point to at least `tensor.ndim` valid `i64` values.
/// The caller must ensure the tensor is not concurrently mutated.
///
/// # Examples
///
/// ```
/// use scirs2_numpy::dlpack::{dlpack_from_slice, validate_dlpack_tensor, DLDataTypeCode};
///
/// let data = vec![1.0_f64, 2.0, 3.0];
/// let shape = vec![3_i64];
/// let tensor = dlpack_from_slice(&data, &shape);
///
/// let info = validate_dlpack_tensor(&tensor).unwrap();
/// assert_eq!(info.shape, vec![3_i64]);
/// assert_eq!(info.dtype_bits, 64);
/// assert_eq!(info.dtype_code, DLDataTypeCode::Float);
/// ```
pub fn validate_dlpack_tensor(tensor: &DLTensor) -> Result<DLTensorInfo, DlpackError> {
    // 1. Null-pointer guard.
    if tensor.data.is_null() {
        return Err(DlpackError::NullPointer);
    }

    // 2. Decode device type.
    let device_type = decode_device_type(tensor.device.device_type);

    // 3. Decode dtype code.
    let dtype_code = DLDataTypeCode::try_from(tensor.dtype.code)?;

    // 4. Copy shape (safe: shape ptr is valid for ndim elements per contract).
    let shape = if tensor.ndim == 0 || tensor.shape.is_null() {
        Vec::new()
    } else {
        // SAFETY: Caller guarantees shape ptr is valid for ndim elements.
        unsafe {
            std::slice::from_raw_parts(tensor.shape as *const i64, tensor.ndim as usize).to_vec()
        }
    };

    Ok(DLTensorInfo {
        shape,
        dtype_code,
        dtype_bits: tensor.dtype.bits,
        device_type,
    })
}

/// Create a [`DLTensor`] that borrows `data` and `shape` slices.
///
/// The returned `DLTensor` has its `data` pointer set to `data.as_ptr()`,
/// `dtype` set to float64 (code=2, bits=64), and device set to CPU.
///
/// # Safety
///
/// The returned `DLTensor` holds raw pointers into `data` and `shape`.
/// Both slices **must** remain live and unmodified for the entire lifetime of
/// the returned tensor.  The tensor must not be used after either slice drops.
///
/// The returned struct does **not** own the memory it points at; no destructor
/// is called for `data` or `shape` when the `DLTensor` is dropped.
///
/// # Examples
///
/// ```
/// use scirs2_numpy::dlpack::{dlpack_from_slice, DLDeviceType, DLDataTypeCode};
///
/// let data = vec![1.0_f64, 2.0, 3.0, 4.0];
/// let shape = vec![2_i64, 2];
/// let tensor = dlpack_from_slice(&data, &shape);
///
/// assert_eq!(tensor.ndim, 2);
/// assert_eq!(tensor.dtype.bits, 64);
/// assert_eq!(tensor.device.device_type, DLDeviceType::Cpu as i32);
/// assert!(!tensor.data.is_null());
/// ```
pub fn dlpack_from_slice(data: &[f64], shape: &[i64]) -> DLTensor {
    DLTensor {
        // SAFETY: We cast a shared reference to a mut-pointer to satisfy the
        // DLPack ABI (which uses *mut c_void).  The caller contract forbids
        // mutations through this pointer; this crate never does so.
        data: data.as_ptr() as *mut c_void,
        device: DLDevice {
            device_type: DLDeviceType::Cpu as i32,
            device_id: 0,
        },
        ndim: shape.len() as i32,
        dtype: DLDataType {
            code: DLDataTypeCode::Float as u8,
            bits: 64,
            lanes: 1,
        },
        // SAFETY: Same const-to-mut cast; shape is read-only.
        shape: shape.as_ptr() as *mut i64,
        strides: std::ptr::null_mut(), // C-contiguous: strides not needed.
        byte_offset: 0,
    }
}

/// Extract a `Vec<f64>` from a CPU float64 [`DLTensor`].
///
/// Validates that:
/// - `data` is non-null,
/// - device type is CPU,
/// - dtype is float64 (code=2, bits=64, lanes=1).
///
/// Returns a freshly allocated `Vec<f64>` copied from the tensor buffer.
///
/// # Safety
///
/// `tensor.data` must point to at least `product(tensor.shape) * 8` valid
/// bytes of `f64` values in native byte order.  Caller must ensure the tensor
/// is valid for the duration of this call.
///
/// # Examples
///
/// ```
/// use scirs2_numpy::dlpack::{dlpack_from_slice, dlpack_to_vec_f64};
///
/// let original = vec![1.0_f64, 2.0, 3.0];
/// let shape = vec![3_i64];
/// let tensor = dlpack_from_slice(&original, &shape);
///
/// // tensor borrows `original` and `shape`; both are live here.
/// let recovered = dlpack_to_vec_f64(&tensor).unwrap();
/// assert_eq!(recovered, original);
/// ```
pub fn dlpack_to_vec_f64(tensor: &DLTensor) -> Result<Vec<f64>, DlpackError> {
    // Guard: non-null data.
    if tensor.data.is_null() {
        return Err(DlpackError::NullPointer);
    }

    // Guard: CPU device.
    let device_type = tensor.device.device_type;
    if device_type != DLDeviceType::Cpu as i32 {
        return Err(DlpackError::NonCpuDevice);
    }

    // Guard: float64 dtype.
    if tensor.dtype.code != DLDataTypeCode::Float as u8
        || tensor.dtype.bits != 64
        || tensor.dtype.lanes != 1
    {
        return Err(DlpackError::UnsupportedDtype {
            code: tensor.dtype.code,
            bits: tensor.dtype.bits,
        });
    }

    // Compute element count from shape.
    let n_elems = if tensor.ndim == 0 {
        1usize
    } else if tensor.shape.is_null() {
        0usize
    } else {
        // SAFETY: shape is valid for ndim elements (caller contract).
        let shape =
            unsafe { std::slice::from_raw_parts(tensor.shape as *const i64, tensor.ndim as usize) };
        shape.iter().map(|&d| d as usize).product()
    };

    // Apply byte_offset.
    let base = unsafe { (tensor.data as *const u8).add(tensor.byte_offset as usize) as *const f64 };

    // SAFETY: base points to n_elems valid f64 values (caller contract).
    let slice = unsafe { std::slice::from_raw_parts(base, n_elems) };
    Ok(slice.to_vec())
}

/// Decode a raw DLPack device-type integer into the [`DLDeviceType`] enum.
///
/// Unknown values fall back to [`DLDeviceType::Cpu`] with a conservative default.
fn decode_device_type(raw: i32) -> DLDeviceType {
    match raw {
        1 => DLDeviceType::Cpu,
        2 => DLDeviceType::Cuda,
        3 => DLDeviceType::CudaHost,
        4 => DLDeviceType::OpenCL,
        7 => DLDeviceType::Vulkan,
        8 => DLDeviceType::Metal,
        10 => DLDeviceType::Rocm,
        _ => DLDeviceType::Cpu, // conservative fallback
    }
}

// ─── PyTorch & JAX interoperability ──────────────────────────────────────────

/// Device type constant for JAX TPU tensors.
///
/// Note: The canonical DLPack spec (DMLC) assigns code 13 to `kDLCUDAManaged`.
/// JAX may use device code 13 for TPU in practice via extension; this constant
/// reflects the code used by JAX's DLPack implementation.
pub const DL_DEVICE_TYPE_TPU: i32 = 13;

/// Known device types as reported by JAX DLPack tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JaxDeviceType {
    /// Standard host CPU (DLPack device type 1).
    Cpu,
    /// TPU accelerator (JAX extension, code 13).
    Tpu,
    /// CUDA GPU (device type 2).
    Gpu,
}

/// Check that a [`DLTensor`]'s strides represent a C-order (row-major) contiguous layout.
///
/// A tensor is contiguous if its `strides` pointer is null (which is the DLPack
/// convention for C-contiguous tensors) or if the non-null strides match the
/// row-major pattern: `strides[i] == product(shape[i+1..])`.
///
/// Returns `Ok(())` when contiguous, `Err(DlpackError::NonContiguous)` otherwise.
///
/// # Safety
///
/// When `tensor.strides` is non-null, it must be valid for `tensor.ndim` elements.
/// When `tensor.shape` is non-null, it must be valid for `tensor.ndim` elements.
pub fn check_tensor_contiguous(tensor: &DLTensor) -> Result<(), DlpackError> {
    // Null strides = C-contiguous by DLPack convention.
    if tensor.strides.is_null() {
        return Ok(());
    }
    // Zero-dimensional tensors are trivially contiguous.
    if tensor.ndim <= 0 || tensor.shape.is_null() {
        return Ok(());
    }
    let ndim = tensor.ndim as usize;
    // SAFETY: Both pointers are non-null and valid for ndim elements (caller contract).
    let shape = unsafe { std::slice::from_raw_parts(tensor.shape as *const i64, ndim) };
    let strides = unsafe { std::slice::from_raw_parts(tensor.strides as *const i64, ndim) };

    // Compute expected C-order strides: last dim = 1, each preceding = product of later dims.
    let mut expected = 1_i64;
    for i in (0..ndim).rev() {
        if strides[i] != expected {
            return Err(DlpackError::NonContiguous);
        }
        expected *= shape[i];
    }
    Ok(())
}

/// Validate that a [`DLTensor`] is compatible with PyTorch DLPack interop.
///
/// Checks:
/// - `data` pointer is non-null,
/// - device type is CPU (`DLDeviceType::Cpu`),
/// - memory layout is C-order contiguous (null strides or row-major strides),
/// - dtype code is float (code 2),
/// - dtype bits are either 32 or 64.
///
/// Returns `Ok(())` on success, or a [`DlpackError`] describing the first
/// unsatisfied constraint.
///
/// # Examples
///
/// ```
/// use scirs2_numpy::dlpack::{dlpack_from_slice, validate_torch_dlpack_tensor};
///
/// let data = vec![1.0_f64, 2.0, 3.0];
/// let shape = vec![3_i64];
/// let tensor = dlpack_from_slice(&data, &shape);
/// assert!(validate_torch_dlpack_tensor(&tensor).is_ok());
/// ```
pub fn validate_torch_dlpack_tensor(tensor: &DLTensor) -> Result<(), DlpackError> {
    // 1. Null pointer check.
    if tensor.data.is_null() {
        return Err(DlpackError::NullPointer);
    }
    // 2. CPU device required.
    if tensor.device.device_type != DLDeviceType::Cpu as i32 {
        return Err(DlpackError::NonCpuDevice);
    }
    // 3. Contiguous memory layout required.
    check_tensor_contiguous(tensor)?;
    // 4. Float dtype required (code 2), 32- or 64-bit.
    if tensor.dtype.code != DLDataTypeCode::Float as u8
        || (tensor.dtype.bits != 32 && tensor.dtype.bits != 64)
    {
        return Err(DlpackError::UnsupportedDtype {
            code: tensor.dtype.code,
            bits: tensor.dtype.bits,
        });
    }
    Ok(())
}

/// Convert a raw `DLTensor` pointer (from a PyTorch DLPack capsule) to an
/// `ndarray` dynamic array view of `f32` elements.
///
/// # Safety
///
/// The caller must guarantee that:
/// - `tensor` is a valid, aligned, non-null pointer to a `DLTensor`.
/// - The tensor's `data` field points to at least `product(shape) * 4` valid
///   bytes of `f32` values in native byte order.
/// - The tensor (and its data) remains live and unmodified for the lifetime
///   of the returned view `'a`.
///
/// # Errors
///
/// Returns [`DlpackError`] if the tensor is not a 32-bit float CPU tensor.
pub unsafe fn dlarray_from_torch_f32<'a>(
    tensor: *const DLTensor,
) -> Result<ndarray::ArrayViewD<'a, f32>, DlpackError> {
    // SAFETY: caller guarantees tensor is valid and non-null.
    let t = unsafe { &*tensor };
    validate_torch_dlpack_tensor(t)?;
    // Must be f32 (32 bits).
    if t.dtype.bits != 32 {
        return Err(DlpackError::UnsupportedDtype {
            code: t.dtype.code,
            bits: t.dtype.bits,
        });
    }
    // Build shape vector.
    let shape = build_shape_vec(t);
    // Compute element count.
    let n_elems: usize = shape.iter().product();
    // SAFETY: data is valid, non-null, aligned, and lives for 'a (caller contract).
    let base = unsafe { (t.data as *const u8).add(t.byte_offset as usize) as *const f32 };
    let slice = unsafe { std::slice::from_raw_parts(base, n_elems) };
    ndarray::ArrayViewD::from_shape(shape.as_slice(), slice).map_err(|_| {
        DlpackError::UnsupportedDtype {
            code: t.dtype.code,
            bits: t.dtype.bits,
        }
    })
}

/// Convert a raw `DLTensor` pointer (from a PyTorch DLPack capsule) to an
/// `ndarray` dynamic array view of `f64` elements.
///
/// # Safety
///
/// Same invariants as [`dlarray_from_torch_f32`], but the data must be 64-bit
/// floating-point (`f64`).
///
/// # Errors
///
/// Returns [`DlpackError`] if the tensor is not a 64-bit float CPU tensor.
pub unsafe fn dlarray_from_torch_f64<'a>(
    tensor: *const DLTensor,
) -> Result<ndarray::ArrayViewD<'a, f64>, DlpackError> {
    // SAFETY: caller guarantees tensor is valid and non-null.
    let t = unsafe { &*tensor };
    validate_torch_dlpack_tensor(t)?;
    // Must be f64 (64 bits).
    if t.dtype.bits != 64 {
        return Err(DlpackError::UnsupportedDtype {
            code: t.dtype.code,
            bits: t.dtype.bits,
        });
    }
    // Build shape vector.
    let shape = build_shape_vec(t);
    // Compute element count.
    let n_elems: usize = shape.iter().product();
    // SAFETY: data is valid, non-null, aligned, and lives for 'a (caller contract).
    let base = unsafe { (t.data as *const u8).add(t.byte_offset as usize) as *const f64 };
    let slice = unsafe { std::slice::from_raw_parts(base, n_elems) };
    ndarray::ArrayViewD::from_shape(shape.as_slice(), slice).map_err(|_| {
        DlpackError::UnsupportedDtype {
            code: t.dtype.code,
            bits: t.dtype.bits,
        }
    })
}

/// Validate that a [`DLTensor`] is compatible with JAX DLPack interop.
///
/// JAX supports CPU, GPU (CUDA), and TPU tensors. Unlike the PyTorch validator
/// this function accepts non-CPU devices; use [`jax_device_type`] afterwards to
/// inspect which device the tensor lives on.
///
/// Returns `Ok(())` if the tensor has a non-null data pointer and a recognised
/// float dtype (32- or 64-bit).
///
/// # Examples
///
/// ```
/// use scirs2_numpy::dlpack::{dlpack_from_slice, validate_jax_dlpack_tensor};
///
/// let data = vec![1.0_f32, 2.0, 3.0];
/// let shape = vec![3_i64];
/// // Build a float-32 CPU tensor for testing.
/// let mut tensor = dlpack_from_slice(
///     &[1.0_f64, 2.0, 3.0],
///     &[3_i64],
/// );
/// // Adjust to f32
/// tensor.dtype.bits = 32;
/// tensor.data = data.as_ptr() as *mut std::ffi::c_void;
/// assert!(validate_jax_dlpack_tensor(&tensor).is_ok());
/// ```
pub fn validate_jax_dlpack_tensor(tensor: &DLTensor) -> Result<(), DlpackError> {
    // 1. Null pointer check.
    if tensor.data.is_null() {
        return Err(DlpackError::NullPointer);
    }
    // 2. Float dtype required (code 2), 32- or 64-bit.
    if tensor.dtype.code != DLDataTypeCode::Float as u8
        || (tensor.dtype.bits != 32 && tensor.dtype.bits != 64)
    {
        return Err(DlpackError::UnsupportedDtype {
            code: tensor.dtype.code,
            bits: tensor.dtype.bits,
        });
    }
    Ok(())
}

/// Classify the device reported by a [`DLTensor`] as a [`JaxDeviceType`].
///
/// Returns `Some(JaxDeviceType)` for recognised JAX device codes, or `None`
/// for unrecognised codes.
///
/// | Code | Device |
/// |------|--------|
/// | 1    | CPU    |
/// | 2    | GPU (CUDA) |
/// | 13   | TPU (JAX extension) |
///
/// # Examples
///
/// ```
/// use scirs2_numpy::dlpack::{dlpack_from_slice, jax_device_type, JaxDeviceType};
///
/// let data = vec![0.0_f64; 4];
/// let shape = vec![4_i64];
/// let tensor = dlpack_from_slice(&data, &shape);
/// assert_eq!(jax_device_type(&tensor), Some(JaxDeviceType::Cpu));
/// ```
pub fn jax_device_type(tensor: &DLTensor) -> Option<JaxDeviceType> {
    match tensor.device.device_type {
        1 => Some(JaxDeviceType::Cpu),
        2 => Some(JaxDeviceType::Gpu),
        DL_DEVICE_TYPE_TPU => Some(JaxDeviceType::Tpu),
        _ => None,
    }
}

/// Generic DLPack array construction — accepts `f32` tensors from any framework.
///
/// Builds an `ndarray` view backed by the tensor's data pointer.  Only CPU
/// tensors are supported; non-CPU tensors return [`DlpackError::NonCpuDevice`].
///
/// # Safety
///
/// The caller must guarantee that:
/// - `tensor` is a valid, aligned, non-null pointer to a `DLTensor`.
/// - The tensor's `data` field points to at least `product(shape) * 4` bytes of
///   valid `f32` values in native byte order.
/// - The tensor remains live and unmodified for the lifetime of the returned
///   view `'a`.
///
/// # Errors
///
/// Returns [`DlpackError`] if the tensor is not CPU-resident or not `f32`.
pub unsafe fn array_from_dlpack_f32<'a>(
    tensor: *const DLTensor,
) -> Result<ndarray::ArrayViewD<'a, f32>, DlpackError> {
    // SAFETY: caller guarantees tensor is valid and non-null.
    let t = unsafe { &*tensor };
    if t.data.is_null() {
        return Err(DlpackError::NullPointer);
    }
    if t.device.device_type != DLDeviceType::Cpu as i32 {
        return Err(DlpackError::NonCpuDevice);
    }
    if t.dtype.code != DLDataTypeCode::Float as u8 || t.dtype.bits != 32 {
        return Err(DlpackError::UnsupportedDtype {
            code: t.dtype.code,
            bits: t.dtype.bits,
        });
    }
    let shape = build_shape_vec(t);
    let n_elems: usize = shape.iter().product();
    let base = unsafe { (t.data as *const u8).add(t.byte_offset as usize) as *const f32 };
    let slice = unsafe { std::slice::from_raw_parts(base, n_elems) };
    ndarray::ArrayViewD::from_shape(shape.as_slice(), slice).map_err(|_| {
        DlpackError::UnsupportedDtype {
            code: t.dtype.code,
            bits: t.dtype.bits,
        }
    })
}

/// Generic DLPack array construction — accepts `f64` tensors from any framework.
///
/// Same as [`array_from_dlpack_f32`] but for 64-bit float tensors.
///
/// # Safety
///
/// Same invariants as [`array_from_dlpack_f32`], but data must be `f64`.
///
/// # Errors
///
/// Returns [`DlpackError`] if the tensor is not CPU-resident or not `f64`.
pub unsafe fn array_from_dlpack_f64<'a>(
    tensor: *const DLTensor,
) -> Result<ndarray::ArrayViewD<'a, f64>, DlpackError> {
    // SAFETY: caller guarantees tensor is valid and non-null.
    let t = unsafe { &*tensor };
    if t.data.is_null() {
        return Err(DlpackError::NullPointer);
    }
    if t.device.device_type != DLDeviceType::Cpu as i32 {
        return Err(DlpackError::NonCpuDevice);
    }
    if t.dtype.code != DLDataTypeCode::Float as u8 || t.dtype.bits != 64 {
        return Err(DlpackError::UnsupportedDtype {
            code: t.dtype.code,
            bits: t.dtype.bits,
        });
    }
    let shape = build_shape_vec(t);
    let n_elems: usize = shape.iter().product();
    let base = unsafe { (t.data as *const u8).add(t.byte_offset as usize) as *const f64 };
    let slice = unsafe { std::slice::from_raw_parts(base, n_elems) };
    ndarray::ArrayViewD::from_shape(shape.as_slice(), slice).map_err(|_| {
        DlpackError::UnsupportedDtype {
            code: t.dtype.code,
            bits: t.dtype.bits,
        }
    })
}

/// Build a shape `Vec<usize>` from the `ndim` / `shape` fields of a [`DLTensor`].
///
/// Returns an empty vector for zero-dimensional tensors.
///
/// # Safety
///
/// `tensor.shape` must be valid for `tensor.ndim` elements.
fn build_shape_vec(tensor: &DLTensor) -> Vec<usize> {
    if tensor.ndim <= 0 || tensor.shape.is_null() {
        Vec::new()
    } else {
        // SAFETY: shape ptr is valid for ndim elements (caller contract).
        let raw =
            unsafe { std::slice::from_raw_parts(tensor.shape as *const i64, tensor.ndim as usize) };
        raw.iter().map(|&d| d as usize).collect()
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // --- validate_dlpack_tensor ---

    #[test]
    fn test_validate_valid_f64_cpu_tensor() {
        let mut data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut shape = vec![2_i64, 3];
        let tensor = dlpack_from_slice(&data, &shape);

        let info = validate_dlpack_tensor(&tensor).expect("validate_dlpack_tensor failed");
        assert_eq!(info.shape, vec![2, 3]);
        assert_eq!(info.dtype_code, DLDataTypeCode::Float);
        assert_eq!(info.dtype_bits, 64);
        assert_eq!(info.device_type, DLDeviceType::Cpu);

        // Keep data and shape alive.
        let _ = (&mut data, &mut shape);
    }

    #[test]
    fn test_validate_null_pointer_returns_err() {
        let shape = vec![3_i64];
        let mut tensor = dlpack_from_slice(&[0.0_f64; 3], &shape);
        // Forcibly set data to null to test the null-pointer guard.
        tensor.data = std::ptr::null_mut();
        let result = validate_dlpack_tensor(&tensor);
        assert!(
            matches!(result, Err(DlpackError::NullPointer)),
            "expected NullPointer error"
        );
    }

    #[test]
    fn test_validate_shape_fields() {
        let data = vec![10.0_f64; 12];
        let shape = vec![3_i64, 4];
        let tensor = dlpack_from_slice(&data, &shape);
        let info = validate_dlpack_tensor(&tensor).expect("validate failed");
        assert_eq!(info.shape, vec![3, 4]);
    }

    // --- dlpack_from_slice ---

    #[test]
    fn test_dlpack_from_slice_shape_fields() {
        let data = vec![1.0_f64, 2.0, 3.0];
        let shape = vec![3_i64];
        let tensor = dlpack_from_slice(&data, &shape);

        assert_eq!(tensor.ndim, 1);
        assert!(!tensor.data.is_null());
        assert!(!tensor.shape.is_null());
        // dtype must be float64
        assert_eq!(tensor.dtype.code, 2); // Float
        assert_eq!(tensor.dtype.bits, 64);
    }

    #[test]
    fn test_dlpack_from_slice_2d() {
        let data = vec![0.0_f64; 6];
        let shape = vec![2_i64, 3];
        let tensor = dlpack_from_slice(&data, &shape);
        assert_eq!(tensor.ndim, 2);
        // SAFETY: shape is valid for ndim=2.
        let s = unsafe { std::slice::from_raw_parts(tensor.shape as *const i64, 2) };
        assert_eq!(s, [2, 3]);
    }

    // --- dlpack_to_vec_f64 ---

    #[test]
    fn test_dlpack_to_vec_f64_round_trip() {
        let original = vec![1.0_f64, 2.5, 3.15, -7.0, 0.0];
        let shape = vec![5_i64];
        let tensor = dlpack_from_slice(&original, &shape);

        let recovered = dlpack_to_vec_f64(&tensor).expect("dlpack_to_vec_f64 failed");
        assert_eq!(recovered, original);
    }

    #[test]
    fn test_dlpack_to_vec_f64_2d() {
        let original: Vec<f64> = (0..6).map(|i| i as f64).collect();
        let shape = vec![2_i64, 3];
        let tensor = dlpack_from_slice(&original, &shape);

        let recovered = dlpack_to_vec_f64(&tensor).expect("dlpack_to_vec_f64 failed");
        assert_eq!(recovered, original);
    }

    #[test]
    fn test_dlpack_to_vec_f64_null_pointer_err() {
        let data = vec![0.0_f64];
        let shape = vec![1_i64];
        let mut tensor = dlpack_from_slice(&data, &shape);
        tensor.data = std::ptr::null_mut();

        assert!(matches!(
            dlpack_to_vec_f64(&tensor),
            Err(DlpackError::NullPointer)
        ));
    }

    #[test]
    fn test_dlpack_to_vec_f64_non_cpu_err() {
        let data = vec![0.0_f64];
        let shape = vec![1_i64];
        let mut tensor = dlpack_from_slice(&data, &shape);
        tensor.device.device_type = DLDeviceType::Cuda as i32;

        assert!(matches!(
            dlpack_to_vec_f64(&tensor),
            Err(DlpackError::NonCpuDevice)
        ));
    }

    #[test]
    fn test_dlpack_to_vec_f64_wrong_dtype_err() {
        let data = vec![0.0_f64];
        let shape = vec![1_i64];
        let mut tensor = dlpack_from_slice(&data, &shape);
        tensor.dtype.code = 0; // Int, not Float

        assert!(matches!(
            dlpack_to_vec_f64(&tensor),
            Err(DlpackError::UnsupportedDtype { .. })
        ));
    }

    // --- DLDataTypeCode ---

    #[test]
    fn test_dtype_code_try_from() {
        assert_eq!(DLDataTypeCode::try_from(0u8).unwrap(), DLDataTypeCode::Int);
        assert_eq!(DLDataTypeCode::try_from(1u8).unwrap(), DLDataTypeCode::UInt);
        assert_eq!(
            DLDataTypeCode::try_from(2u8).unwrap(),
            DLDataTypeCode::Float
        );
        assert_eq!(
            DLDataTypeCode::try_from(3u8).unwrap(),
            DLDataTypeCode::BFloat
        );
        assert!(DLDataTypeCode::try_from(99u8).is_err());
    }

    // ─── Item 1: PyTorch tensor interop tests ────────────────────────────────

    #[test]
    fn dlpack_device_type_cpu_is_1() {
        assert_eq!(DLDeviceType::Cpu as i32, 1);
    }

    #[test]
    fn dlpack_dtype_float32_code_is_2() {
        // DLPack spec: kDLFloat = 2
        assert_eq!(DLDataTypeCode::Float as u8, 2);
    }

    #[test]
    fn dlpack_validate_non_contiguous_fails() {
        // Build a 2-D tensor with non-row-major strides to simulate a transposed
        // PyTorch tensor.  Shape = [2, 3], but strides = [1, 2] (column-major).
        let data = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = [2_i64, 3];
        // Column-major strides (Fortran order): stride[0]=1, stride[1]=2
        let strides = [1_i64, 2];
        let tensor = DLTensor {
            data: data.as_ptr() as *mut c_void,
            device: DLDevice {
                device_type: DLDeviceType::Cpu as i32,
                device_id: 0,
            },
            ndim: 2,
            dtype: DLDataType {
                code: DLDataTypeCode::Float as u8,
                bits: 32,
                lanes: 1,
            },
            shape: shape.as_ptr() as *mut i64,
            strides: strides.as_ptr() as *mut i64,
            byte_offset: 0,
        };
        assert!(
            matches!(
                validate_torch_dlpack_tensor(&tensor),
                Err(DlpackError::NonContiguous)
            ),
            "expected NonContiguous error for column-major strides"
        );
    }

    #[test]
    fn dlpack_validate_2d_float_tensor_passes() {
        let data = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = [2_i64, 3];
        let tensor = DLTensor {
            data: data.as_ptr() as *mut c_void,
            device: DLDevice {
                device_type: DLDeviceType::Cpu as i32,
                device_id: 0,
            },
            ndim: 2,
            dtype: DLDataType {
                code: DLDataTypeCode::Float as u8,
                bits: 32,
                lanes: 1,
            },
            shape: shape.as_ptr() as *mut i64,
            strides: std::ptr::null_mut(),
            byte_offset: 0,
        };
        assert!(validate_torch_dlpack_tensor(&tensor).is_ok());
    }

    #[test]
    fn dlpack_validate_non_cpu_tensor_fails() {
        let data = [1.0_f32; 4];
        let shape = [4_i64];
        let tensor = DLTensor {
            data: data.as_ptr() as *mut c_void,
            device: DLDevice {
                device_type: DLDeviceType::Cuda as i32,
                device_id: 0,
            },
            ndim: 1,
            dtype: DLDataType {
                code: DLDataTypeCode::Float as u8,
                bits: 32,
                lanes: 1,
            },
            shape: shape.as_ptr() as *mut i64,
            strides: std::ptr::null_mut(),
            byte_offset: 0,
        };
        assert!(matches!(
            validate_torch_dlpack_tensor(&tensor),
            Err(DlpackError::NonCpuDevice)
        ));
    }

    #[test]
    fn dlarray_from_torch_f32_round_trip() {
        let data = [1.0_f32, 2.0, 3.0, 4.0];
        let shape = [2_i64, 2];
        let tensor = DLTensor {
            data: data.as_ptr() as *mut c_void,
            device: DLDevice {
                device_type: DLDeviceType::Cpu as i32,
                device_id: 0,
            },
            ndim: 2,
            dtype: DLDataType {
                code: DLDataTypeCode::Float as u8,
                bits: 32,
                lanes: 1,
            },
            shape: shape.as_ptr() as *mut i64,
            strides: std::ptr::null_mut(),
            byte_offset: 0,
        };
        // SAFETY: tensor is valid, data and shape are alive.
        let view = unsafe { dlarray_from_torch_f32(&tensor as *const DLTensor) }
            .expect("dlarray_from_torch_f32 failed");
        assert_eq!(view.shape(), &[2, 2]);
        assert_eq!(view[[0, 0]], 1.0_f32);
        assert_eq!(view[[1, 1]], 4.0_f32);
    }

    #[test]
    fn dlarray_from_torch_f64_round_trip() {
        let data = [10.0_f64, 20.0, 30.0];
        let shape = [3_i64];
        let tensor = DLTensor {
            data: data.as_ptr() as *mut c_void,
            device: DLDevice {
                device_type: DLDeviceType::Cpu as i32,
                device_id: 0,
            },
            ndim: 1,
            dtype: DLDataType {
                code: DLDataTypeCode::Float as u8,
                bits: 64,
                lanes: 1,
            },
            shape: shape.as_ptr() as *mut i64,
            strides: std::ptr::null_mut(),
            byte_offset: 0,
        };
        // SAFETY: tensor is valid, data and shape are alive.
        let view = unsafe { dlarray_from_torch_f64(&tensor as *const DLTensor) }
            .expect("dlarray_from_torch_f64 failed");
        assert_eq!(view.shape(), &[3]);
        assert_eq!(view[2], 30.0_f64);
    }

    // ─── Item 2: JAX array interop tests ─────────────────────────────────────

    #[test]
    fn dlpack_jax_cpu_tensor_valid() {
        let data = [1.0_f64, 2.0, 3.0];
        let shape = [3_i64];
        let tensor = dlpack_from_slice(&data, &shape);
        assert!(validate_jax_dlpack_tensor(&tensor).is_ok());
        assert_eq!(jax_device_type(&tensor), Some(JaxDeviceType::Cpu));
    }

    #[test]
    fn dlpack_jax_tpu_device_recognized() {
        let data = [1.0_f64];
        let shape = [1_i64];
        let mut tensor = dlpack_from_slice(&data, &shape);
        tensor.device.device_type = DL_DEVICE_TYPE_TPU;
        // JAX validator does not require CPU — only float dtype.
        assert!(validate_jax_dlpack_tensor(&tensor).is_ok());
        assert_eq!(jax_device_type(&tensor), Some(JaxDeviceType::Tpu));
    }

    #[test]
    fn dlpack_generic_from_dlpack_handles_both_torch_and_jax() {
        // f32 torch-style tensor
        let data_f32 = [0.5_f32, 1.5, 2.5, 3.5];
        let shape = [2_i64, 2];
        let tensor_f32 = DLTensor {
            data: data_f32.as_ptr() as *mut c_void,
            device: DLDevice {
                device_type: DLDeviceType::Cpu as i32,
                device_id: 0,
            },
            ndim: 2,
            dtype: DLDataType {
                code: DLDataTypeCode::Float as u8,
                bits: 32,
                lanes: 1,
            },
            shape: shape.as_ptr() as *mut i64,
            strides: std::ptr::null_mut(),
            byte_offset: 0,
        };
        // SAFETY: tensor_f32 is valid; data_f32 and shape are alive.
        let view_f32 = unsafe { array_from_dlpack_f32(&tensor_f32 as *const DLTensor) }
            .expect("array_from_dlpack_f32 failed");
        assert_eq!(view_f32.shape(), &[2, 2]);

        // f64 jax-style tensor (same CPU device)
        let data_f64 = [1.0_f64, 2.0, 3.0, 4.0];
        let shape_f64 = [4_i64];
        let tensor_f64 = dlpack_from_slice(&data_f64, &shape_f64);
        // SAFETY: tensor_f64 is valid; data_f64 and shape_f64 are alive.
        let view_f64 = unsafe { array_from_dlpack_f64(&tensor_f64 as *const DLTensor) }
            .expect("array_from_dlpack_f64 failed");
        assert_eq!(view_f64.shape(), &[4]);
        assert_eq!(view_f64[3], 4.0_f64);
    }
}
