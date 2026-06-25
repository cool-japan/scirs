//! GPU-accelerated matrix operations: core types and error definitions
//!
//! This module provides a pure-Rust simulated GPU abstraction that models the
//! GPU programming model (buffers, capabilities, backends) without requiring
//! any C/Fortran dependencies or hardware-specific feature flags.  When real
//! OxiBLAS GPU backends become available they can slot in behind the same
//! public API.
//!
//! ## Design
//!
//! - [`GpuBackendKind`]: selects between the CPU fallback, the pure-Rust
//!   simulated-GPU tile engine, or the OxiBLAS GPU backend (future).
//! - [`GpuMatrixBuffer`]: a CPU-backed representation of a matrix "on device",
//!   tracking shape metadata alongside the flat data vector.
//! - [`GpuError`] / [`GpuResult`]: error type hierarchy for all GPU operations.
//! - [`GpuCapabilities`]: describes the (simulated) device capabilities.
//! - [`GpuMatrixConfig`]: tuning knobs for GEMM and dispatch.

use std::fmt;

// ─── Backend selection ────────────────────────────────────────────────────────

/// Selects the compute backend for GPU-style matrix operations.
///
/// The enum is `#[non_exhaustive]` so future backends (real OxiBLAS GPU,
/// Vulkan compute, WebGPU, …) can be added without a breaking change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum GpuBackendKind {
    /// Pure CPU path — uses the 3-level cache-blocked GEMM from `gpu_gemm`.
    /// Always available and chosen automatically for small matrices.
    Cpu,
    /// Pure-Rust simulated GPU — models the tiled GEMM execution pattern
    /// (32×32 tiles) on top of Vecs.  Useful for testing the GPU code path
    /// without hardware and as a functional reference.
    Simulated,
    /// OxiBLAS GPU backend.  Activated when the `gpu` feature flag is enabled
    /// and a compatible OxiBLAS GPU runtime is present.  Falls back silently to
    /// `Simulated` when the runtime is absent.
    OxiBlasGpu,
}

impl Default for GpuBackendKind {
    fn default() -> Self {
        Self::Simulated
    }
}

impl fmt::Display for GpuBackendKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu => write!(f, "CPU"),
            Self::Simulated => write!(f, "Simulated-GPU"),
            Self::OxiBlasGpu => write!(f, "OxiBLAS-GPU"),
        }
    }
}

// ─── Configuration ────────────────────────────────────────────────────────────

/// Tuning configuration for GPU-style matrix operations.
///
/// All fields have sensible defaults via [`Default`].
#[derive(Clone, Debug)]
pub struct GpuMatrixConfig {
    /// Which backend to use.  Defaults to [`GpuBackendKind::Simulated`].
    pub backend: GpuBackendKind,
    /// Tile edge length for the tiled GEMM kernel (both row and column tiles).
    /// Must be ≥ 1.  Larger tiles improve cache reuse up to a point; 32 is a
    /// good default for simulated execution.
    pub tile_size: usize,
    /// Hint that the workload may benefit from tensor-core style accumulation.
    /// Currently advisory only; reserved for future OxiBLAS GPU integration.
    pub use_tensor_cores: bool,
    /// Soft limit on device memory the operation is allowed to use, in MiB.
    /// Zero means "no limit".
    pub memory_limit_mb: u64,
    /// Minimum `m * n * k` product below which [`GpuBackendKind::Cpu`] is
    /// chosen even when `backend` is set to `Simulated` or `OxiBlasGpu`.
    pub gpu_threshold: usize,
}

impl Default for GpuMatrixConfig {
    fn default() -> Self {
        Self {
            backend: GpuBackendKind::default(),
            tile_size: 32,
            use_tensor_cores: false,
            memory_limit_mb: 0,
            gpu_threshold: 1_000_000, // 10³ elements per side
        }
    }
}

// ─── GPU buffer ───────────────────────────────────────────────────────────────

/// A CPU-backed matrix buffer that models a GPU device allocation.
///
/// Data is stored in row-major (C) order as a flat `Vec<T>`.  The type
/// intentionally does not implement `Copy` — callers should pass references
/// or use [`clone`](GpuMatrixBuffer::clone) explicitly.
#[derive(Clone, Debug)]
pub struct GpuMatrixBuffer<T> {
    /// Flat row-major data: element (i, j) is at `data[i * cols + j]`.
    pub(crate) data: Vec<T>,
    /// Number of rows.
    pub rows: usize,
    /// Number of columns.
    pub cols: usize,
}

impl<T: Clone + Default> GpuMatrixBuffer<T> {
    /// Allocate a zeroed buffer of shape `rows × cols`.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![T::default(); rows * cols],
            rows,
            cols,
        }
    }

    /// Create a buffer from existing row-major flat data.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::DimensionMismatch`] when `data.len() != rows * cols`.
    pub fn from_slice(data: &[T], rows: usize, cols: usize) -> GpuResult<Self> {
        let expected = rows.checked_mul(cols).ok_or(GpuError::OutOfMemory {
            requested_bytes: u64::MAX,
            available_bytes: 0,
        })?;
        if data.len() != expected {
            return Err(GpuError::DimensionMismatch {
                expected,
                got: data.len(),
                context: "GpuMatrixBuffer::from_slice: data length mismatch".to_string(),
            });
        }
        Ok(Self {
            data: data.to_vec(),
            rows,
            cols,
        })
    }

    /// Number of elements stored in the buffer.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` if the buffer contains no elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Immutable view of the flat row-major data.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Mutable view of the flat row-major data.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Read element at position `(row, col)`.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::DimensionMismatch`] if either index is out of range.
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> GpuResult<&T> {
        if row >= self.rows || col >= self.cols {
            return Err(GpuError::DimensionMismatch {
                expected: self.rows * self.cols,
                got: row * self.cols + col,
                context: format!(
                    "GpuMatrixBuffer::get index ({row},{col}) out of bounds for {}×{}",
                    self.rows, self.cols
                ),
            });
        }
        Ok(&self.data[row * self.cols + col])
    }

    /// Mutable reference to element at `(row, col)`.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::DimensionMismatch`] if either index is out of range.
    #[inline]
    pub fn get_mut(&mut self, row: usize, col: usize) -> GpuResult<&mut T> {
        if row >= self.rows || col >= self.cols {
            return Err(GpuError::DimensionMismatch {
                expected: self.rows * self.cols,
                got: row * self.cols + col,
                context: format!(
                    "GpuMatrixBuffer::get_mut index ({row},{col}) out of bounds for {}×{}",
                    self.rows, self.cols
                ),
            });
        }
        Ok(&mut self.data[row * self.cols + col])
    }
}

// ─── Error types ─────────────────────────────────────────────────────────────

/// Error type for GPU-style matrix operations.
///
/// `#[non_exhaustive]` — new variants may be added in future releases.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum GpuError {
    /// The operation would exceed available (simulated) device memory.
    OutOfMemory {
        /// Number of bytes the operation requested.
        requested_bytes: u64,
        /// Number of bytes currently available.
        available_bytes: u64,
    },
    /// The requested operation is not supported by the selected backend.
    UnsupportedOperation {
        /// Human-readable description of what was attempted.
        operation: String,
        /// Which backend rejected the operation.
        backend: String,
    },
    /// No suitable GPU backend could be found or initialised.
    BackendUnavailable {
        /// Human-readable reason.
        reason: String,
    },
    /// Matrix or vector dimension mismatch.
    DimensionMismatch {
        /// Expected size / product.
        expected: usize,
        /// Actual size / product received.
        got: usize,
        /// Context string for debugging.
        context: String,
    },
    /// An integer overflow occurred while computing a buffer size.
    SizeOverflow {
        /// Description of what overflowed.
        detail: String,
    },
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfMemory {
                requested_bytes,
                available_bytes,
            } => write!(
                f,
                "GPU out-of-memory: requested {requested_bytes} bytes, \
                 only {available_bytes} bytes available"
            ),
            Self::UnsupportedOperation { operation, backend } => {
                write!(
                    f,
                    "GPU unsupported operation '{operation}' on backend '{backend}'"
                )
            }
            Self::BackendUnavailable { reason } => {
                write!(f, "GPU backend unavailable: {reason}")
            }
            Self::DimensionMismatch {
                expected,
                got,
                context,
            } => write!(
                f,
                "GPU dimension mismatch (expected {expected}, got {got}): {context}"
            ),
            Self::SizeOverflow { detail } => {
                write!(f, "GPU size overflow: {detail}")
            }
        }
    }
}

impl std::error::Error for GpuError {}

/// Convenience `Result` alias for GPU operations.
pub type GpuResult<T> = Result<T, GpuError>;

// ─── Capabilities ─────────────────────────────────────────────────────────────

/// Simulated GPU device capabilities.
#[derive(Clone, Debug)]
pub struct GpuCapabilities {
    /// Whether the device supports IEEE 754 half-precision (FP16).
    pub has_fp16: bool,
    /// Whether the device exposes tensor-core style mixed-precision units.
    pub has_tensor_cores: bool,
    /// Simulation working-memory budget in gigabytes (host RAM, not GPU VRAM).
    pub vram_gb: f64,
    /// Human-readable device name.
    pub name: String,
    /// Number of compute units; for the CPU-backed model this is the host core count.
    pub compute_units: u32,
    /// Preferred warp / wavefront width (elements processed together).
    pub warp_size: u32,
    /// Maximum total elements in a single kernel launch.
    pub max_buffer_elements: usize,
}

impl Default for GpuCapabilities {
    fn default() -> Self {
        detect_gpu_capabilities()
    }
}

/// Report the capabilities of the CPU-backed "GPU" execution model.
///
/// This backend is a pure-Rust simulation of the GPU programming model that runs
/// entirely on the host CPU — there is **no physical GPU**. The reported figures
/// therefore describe the host-backed simulation rather than real device
/// hardware: `compute_units` reflects the number of logical CPU cores, and the
/// memory / width figures are the simulation's working limits, not the VRAM and
/// warp width of a discrete graphics card.
pub fn detect_gpu_capabilities() -> GpuCapabilities {
    // Real host core count instead of a fabricated SM count.
    let compute_units = num_cpus::get().max(1) as u32;
    GpuCapabilities {
        has_fp16: true,
        has_tensor_cores: false, // No tensor-core hardware; not emulated.
        // Host-backed buffers live in system RAM; this is the simulation's
        // working memory budget, not the VRAM of a physical GPU.
        vram_gb: 8.0,
        name: "SciRS2 CPU-backed GEMM (simulated GPU model, no physical GPU)".to_string(),
        compute_units,
        warp_size: 32, // Tile width used by the simulated tiled-GEMM kernel.
        max_buffer_elements: 1 << 30, // Simulation cap (~1 billion elements).
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_backend_kind_default() {
        assert_eq!(GpuBackendKind::default(), GpuBackendKind::Simulated);
    }

    #[test]
    fn test_gpu_matrix_config_default() {
        let cfg = GpuMatrixConfig::default();
        assert_eq!(cfg.tile_size, 32);
        assert!(!cfg.use_tensor_cores);
        assert_eq!(cfg.memory_limit_mb, 0);
        assert_eq!(cfg.gpu_threshold, 1_000_000);
    }

    #[test]
    fn test_gpu_buffer_zeros() {
        let buf = GpuMatrixBuffer::<f64>::zeros(3, 4);
        assert_eq!(buf.rows, 3);
        assert_eq!(buf.cols, 4);
        assert_eq!(buf.len(), 12);
        assert!(buf.as_slice().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_gpu_buffer_from_slice_ok() {
        let data: Vec<f64> = (0..6).map(|i| i as f64).collect();
        let buf = GpuMatrixBuffer::from_slice(&data, 2, 3).unwrap();
        assert_eq!(buf.rows, 2);
        assert_eq!(buf.cols, 3);
        assert_eq!(*buf.get(0, 2).unwrap(), 2.0);
        assert_eq!(*buf.get(1, 0).unwrap(), 3.0);
    }

    #[test]
    fn test_gpu_buffer_from_slice_mismatch() {
        let data = vec![1.0_f64; 5];
        let result = GpuMatrixBuffer::from_slice(&data, 2, 3);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            GpuError::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn test_gpu_buffer_get_out_of_bounds() {
        let buf = GpuMatrixBuffer::<f64>::zeros(2, 2);
        assert!(buf.get(2, 0).is_err());
        assert!(buf.get(0, 2).is_err());
    }

    #[test]
    fn test_gpu_error_display_out_of_memory() {
        let err = GpuError::OutOfMemory {
            requested_bytes: 1024,
            available_bytes: 512,
        };
        let msg = err.to_string();
        assert!(msg.contains("out-of-memory"));
        assert!(msg.contains("1024"));
    }

    #[test]
    fn test_gpu_error_display_dimension_mismatch() {
        let err = GpuError::DimensionMismatch {
            expected: 6,
            got: 4,
            context: "test".to_string(),
        };
        let msg = err.to_string();
        assert!(msg.contains("dimension mismatch"));
        assert!(msg.contains("6"));
        assert!(msg.contains("4"));
    }

    #[test]
    fn test_detect_gpu_capabilities() {
        let caps = detect_gpu_capabilities();
        assert!(caps.vram_gb > 0.0);
        assert!(caps.compute_units > 0);
        assert!(!caps.name.is_empty());
    }

    #[test]
    fn test_gpu_backend_kind_display() {
        assert_eq!(GpuBackendKind::Cpu.to_string(), "CPU");
        assert_eq!(GpuBackendKind::Simulated.to_string(), "Simulated-GPU");
        assert_eq!(GpuBackendKind::OxiBlasGpu.to_string(), "OxiBLAS-GPU");
    }
}
