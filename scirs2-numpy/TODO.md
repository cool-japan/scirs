# scirs2-numpy TODO

## Status: v0.4.3 Released (2026-05-03)

## v0.3.3 Completed

### Core Array Types
- [x] `PyArray<T, D>` - Generic N-dimensional NumPy array wrapper
- [x] Fixed-rank aliases: `PyArray0` through `PyArray6`
- [x] `PyArrayDyn<T>` - Dynamic-rank arrays
- [x] `PyReadonlyArray` - Shared, read-only borrow of NumPy memory
- [x] `PyReadwriteArray` - Exclusive mutable borrow of NumPy memory
- [x] `PyArray0Methods` and `PyArrayMethods` traits

### Type Support
- [x] `f32`, `f64` - Single and double precision float
- [x] `i32`, `i64`, `u32`, `u64`, `i16`, `u16`, `i8`, `u8`
- [x] `Complex<f32>`, `Complex<f64>` (via `num-complex`)
- [x] Bool arrays
- [x] `datetime64`, `timedelta64` (via `datetime` module)

### Conversions
- [x] `ToPyArray` trait: `ndarray::Array<T,D>` -> `PyArray<T,D>` (zero-copy)
- [x] `as_array()`: `PyArray<T,D>` -> `ndarray::ArrayView<T,D>` (zero-copy when contiguous)
- [x] `as_array_mut()`: mutable ArrayViewMut via `PyReadwriteArray`
- [x] Automatic contiguous-buffer fallback for strided/non-contiguous arrays
- [x] C-order and Fortran-order layout detection and handling

### Type Coercion
- [x] `PyArrayLike<T, D, TypeConstraint>` - Accept array-like Python objects
- [x] `AllowTypeChange` - Implicit type coercion at boundaries
- [x] `TypeMustMatch` - Strict dtype checking
- [x] Fixed-rank `PyArrayLike0` through `PyArrayLike6` and `PyArrayLikeDyn`

### ndarray 0.17 Compatibility (Core Purpose)
- [x] All internal type references use ndarray 0.17 APIs
- [x] No ndarray 0.16 dependencies
- [x] Eliminates version conflict with `scirs2-linalg`, `scirs2-stats`, `scirs2-fft`, etc.
- [x] Direct `ArrayView`/`Array` interop with all SciRS2 crates

### Optional Features
- [x] `nalgebra` feature: `as_matrix()` / `as_matrix_mut()` for `nalgebra::MatrixView`
- [x] `ToPyArray` impl for `nalgebra::Matrix`

### Unsafe FFI
- [x] `npyffi` module: raw bindings to NumPy C API (`ndarraytypes.h`, `npy_common.h`, etc.)
- [x] `get_array_module()` - access the `numpy.core` Python module
- [x] Array metadata: `shape()`, `strides()`, `ndim()`, `len()`, `is_empty()`
- [x] Memory flags: `is_c_contiguous()`, `is_fortran_contiguous()`

### Strings
- [x] `strings` module: fixed-width string array support (`|S<N>` dtype)
- [x] `PyFixedUnicode` and `PyFixedString` for structured array string fields

### Error Handling
- [x] `NotContiguousError` for non-contiguous array access
- [x] `FromVecError` for invalid shape construction
- [x] All errors surface as Python exceptions via `PyErr`

## v0.4.0 Roadmap

### Full numpy API Compatibility
- [x] `numpy.array_protocol` (`__array__` / `__array_interface__`) support
- [x] Structured dtype support (record arrays with named fields)
- [x] Masked array support (`numpy.ma`)
- [x] Array subclass support (e.g., accepting `pandas.Series`)

### DLPack Support
- [x] `__dlpack__` and `__dlpack_device__` protocol for zero-copy GPU/CPU tensor exchange
- [x] DLPack interoperability: `validate_dlpack_tensor`, `dlpack_from_slice`, `dlpack_to_vec_f64`
- [x] PyTorch tensor interop via DLPack
- [x] JAX array interop via DLPack
- [x] Enable GPU tensors (CUDA) to be passed without CPU roundtrip — Implemented in v0.4.3 (`dlpack_cuda.rs`: `cuda_tensor_info_from_dltensor`, `dlpack_auto_dispatch_f32/f64`, `CudaTensorInfo`, 15 tests)

### Performance
- [x] SIMD-accelerated copy for non-contiguous-to-contiguous coercion
- [x] Benchmark conversion overhead vs upstream rust-numpy
- [x] Profile allocation patterns for large array round-trips

### Type System
- [x] `PyUntypedArray` for accepting arrays without known element type at compile time
- [x] Runtime dtype inspection: `array.dtype_name()`, `array.itemsize()`
- [x] Extended integer types: `i128`, `u128` where NumPy supports them

### Documentation and Tests
- [x] Comprehensive doctests for all public APIs
- [x] Python-side pytest tests comparing output with upstream rust-numpy behavior
- [x] Migration guide: `rust-numpy` -> `scirs2-numpy`

## Known Issues

- Non-contiguous arrays (e.g., transposed NumPy arrays with non-standard strides) trigger a silent copy; this is correct behavior but worth documenting clearly.
- The `datetime64` support is incomplete for all NumPy datetime units; only `ns` (nanoseconds) is fully tested.
- `nalgebra` integration requires square-shaped arrays for `Matrix` types; dynamic-shape nalgebra matrices are supported via `DMatrix`.
- Mutable borrow (`PyReadwriteArray`) requires exclusive access; passing the same NumPy array to two `readwrite()` calls simultaneously will panic (this is intentional and enforced by Rust's borrow checker through PyO3's lifetime tracking).
