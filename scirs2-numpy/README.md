# scirs2-numpy

A SciRS2-maintained fork of [rust-numpy](https://github.com/PyO3/rust-numpy) providing Rust interfaces for the [NumPy C API](https://numpy.org/doc/stable/reference/c-api) with native support for **ndarray 0.17**.

[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](../LICENSE)
[![Version](https://img.shields.io/badge/version-0.4.4-green)]()

## What Is This Crate?

`scirs2-numpy` is the bridge between Rust's [ndarray](https://github.com/rust-ndarray/ndarray) crate and Python's [NumPy](https://numpy.org/) library. It enables:

1. **Zero-copy conversion** of NumPy arrays to Rust `ndarray` arrays (where memory layout permits)
2. **Returning Rust arrays to Python** as NumPy arrays without data copying
3. **Complex number support** for scientific computing workloads
4. **Multi-dimensional array support** up to dynamic rank (`PyArrayDyn`)

The key motivation for maintaining this fork is **ndarray 0.17 support**. The upstream `rust-numpy` crate pins to ndarray 0.16, which creates a version conflict when used alongside the rest of the SciRS2 ecosystem (which uses ndarray 0.17 throughout). This fork eliminates that conflict, enabling direct zero-copy interop in `scirs2-python`.

## Relationship to rust-numpy

| Feature | `rust-numpy` (upstream) | `scirs2-numpy` (this crate) |
|---------|------------------------|----------------------------|
| ndarray version | 0.16 | **0.17** |
| Zero-copy interop | Yes | Yes |
| Complex arrays | Yes | Yes |
| nalgebra integration | Yes | Yes |
| PyO3 version | Latest | Latest |
| SciRS2 compatibility | Needs conversion | **Native, no conversion** |

If you do not use SciRS2 internals and only need NumPy interop in your own PyO3 extension, you can use the upstream `rust-numpy`. Use `scirs2-numpy` when you need compatibility with `scirs2-*` crates.

## When to Use This Crate vs scirs2-python

| Scenario | Recommended |
|----------|-------------|
| Writing a Python extension that calls SciRS2 internals | `scirs2-numpy` + PyO3 |
| Using SciRS2 from Python | `scirs2-python` (wraps this crate) |
| Pure Python user, no Rust code | `pip install scirs2` |
| Standalone Rust project needing NumPy interop | `rust-numpy` (upstream) |

## Features

- **Zero-copy array sharing**: Rust reads/writes NumPy array memory directly when alignment and contiguity permit
- **Numeric type support**: `f32`, `f64`, `i32`, `i64`, `u32`, `u64`, `c32` (complex float), `c64` (complex double)
- **Multi-dimensional arrays**: `PyArray0` through `PyArray6` and `PyArrayDyn` for dynamic rank
- **Read/write access control**: `PyReadonlyArray` and `PyReadwriteArray` for safe borrow semantics
- **Memory layout handling**: C-order (row-major) and Fortran-order (column-major) both supported
- **Type coercion**: `PyArrayLike` for accepting arrays with automatic type conversion
- **nalgebra integration**: Optional `nalgebra` feature converts between NumPy arrays and `nalgebra::Matrix`
- **`AllowTypeChange`/`TypeMustMatch`**: Control strictness of type checking at array boundaries

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
scirs2-numpy = { path = "../scirs2-numpy" }
# or from crates.io once published:
# scirs2-numpy = "0.4.4"
pyo3 = { version = "0.22", features = ["extension-module"] }
```

### Basic Example: Accepting a NumPy Array and Returning One

```rust
use pyo3::prelude::*;
use scirs2_numpy::{PyArray1, PyArrayMethods, ToPyArray};
use ndarray::Array1;

#[pyfunction]
fn double_array<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyArray1<f64>>,
) -> Bound<'py, PyArray1<f64>> {
    // Zero-copy read access to the NumPy array
    let readonly = x.readonly();
    let arr: ndarray::ArrayView1<f64> = readonly.as_array();

    // Compute using ndarray 0.17
    let result: Array1<f64> = &arr * 2.0;

    // Convert back to NumPy (zero-copy when possible)
    result.to_pyarray(py)
}

#[pymodule]
fn my_extension(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(double_array, m)?)?;
    Ok(())
}
```

### Matrix Operations (2D Arrays)

```rust
use pyo3::prelude::*;
use scirs2_numpy::{PyArray2, PyArrayMethods, ToPyArray};
use ndarray::Array2;

#[pyfunction]
fn matrix_square<'py>(
    py: Python<'py>,
    m: &Bound<'py, PyArray2<f64>>,
) -> Bound<'py, PyArray2<f64>> {
    let read = m.readonly();
    let mat: ndarray::ArrayView2<f64> = read.as_array();

    // mat.dot(&mat) computes M * M using ndarray 0.17
    let result: Array2<f64> = mat.dot(&mat);
    result.to_pyarray(py)
}
```

### Dynamic Rank Arrays

```rust
use pyo3::prelude::*;
use scirs2_numpy::{PyArrayDyn, PyArrayMethods, ToPyArray};

#[pyfunction]
fn sum_all<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyArrayDyn<f64>>,
) -> f64 {
    x.readonly().as_array().sum()
}
```

### Complex Number Support

```rust
use pyo3::prelude::*;
use scirs2_numpy::{PyArray1, PyArrayMethods, ToPyArray};
use num_complex::Complex64;

#[pyfunction]
fn conjugate<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyArray1<Complex64>>,
) -> Bound<'py, PyArray1<Complex64>> {
    let arr = x.readonly();
    let view = arr.as_array();
    let conj: ndarray::Array1<Complex64> = view.mapv(|c| c.conj());
    conj.to_pyarray(py)
}
```

### Read/Write Mutable Access

```rust
use pyo3::prelude::*;
use scirs2_numpy::{PyArray1, PyArrayMethods};

#[pyfunction]
fn scale_inplace<'py>(
    x: &Bound<'py, PyArray1<f64>>,
    factor: f64,
) {
    // Mutates the NumPy array in-place (zero-copy)
    let mut rw = x.readwrite();
    rw.as_array_mut().mapv_inplace(|v| v * factor);
}
```

### nalgebra Integration (Optional Feature)

```rust
// Cargo.toml: scirs2-numpy = { version = "0.4.4", features = ["nalgebra"] }
use pyo3::prelude::*;
use scirs2_numpy::{PyArray2, PyArrayMethods};

#[pyfunction]
fn determinant<'py>(
    m: &Bound<'py, PyArray2<f64>>,
) -> f64 {
    let read = m.readonly();
    // Convert NumPy 2D array to nalgebra DMatrix
    let na_mat = read.as_matrix();
    na_mat.determinant()
}
```

## Type Aliases

```rust
use scirs2_numpy::{
    PyArray0, PyArray1, PyArray2, PyArray3, PyArray4, PyArray5, PyArray6,
    PyArrayDyn,
    PyArray0Methods, PyArrayMethods,
};
```

## Memory Layout

NumPy supports both C-order (row-major) and Fortran-order (column-major) arrays. `scirs2-numpy` handles both:

```rust
use scirs2_numpy::{PyArray2, PyArrayMethods};

// Check if an array is C-contiguous (row-major)
let is_c_order = array.is_c_contiguous();
// Check if Fortran-contiguous (column-major)
let is_f_order = array.is_fortran_contiguous();
```

For non-contiguous arrays (strides that are not standard), the crate automatically copies into a contiguous buffer before providing a Rust view.

## Type Coercion with PyArrayLike

`PyArrayLike` accepts Python lists, NumPy arrays, and other array-like objects, performing type coercion:

```rust
use pyo3::prelude::*;
use scirs2_numpy::{PyArrayLike1, AllowTypeChange};

#[pyfunction]
fn mean_of<'py>(
    x: PyArrayLike1<'py, f64, AllowTypeChange>,
) -> f64 {
    // x is automatically coerced to Array1<f64> even if the input was int32
    let arr = x.as_array();
    arr.sum() / arr.len() as f64
}
```

## API Overview

| Type / Function | Description |
|----------------|-------------|
| `PyArray<T, D>` | Core type wrapping a NumPy array of element type `T` and ndim `D` |
| `PyArrayDyn<T>` | Dynamic-rank NumPy array |
| `PyArray0..6<T>` | Fixed-rank aliases (0D through 6D) |
| `PyReadonlyArray` | Shared read-only borrow of array memory |
| `PyReadwriteArray` | Exclusive read-write borrow of array memory |
| `ToPyArray` | Trait to convert `ndarray::Array` → `PyArray` |
| `PyArrayMethods` | Core method trait: `shape()`, `strides()`, `as_array()`, `readonly()`, `readwrite()` |
| `PyArrayLike<T>` | Accept array-like Python objects with type coercion |
| `AllowTypeChange` | Marker: allow implicit type conversion when receiving arrays |
| `TypeMustMatch` | Marker: require exact dtype match |
| `get_array_module` | Get the `numpy` Python module object |

## Integration with SciRS2-Python

`scirs2-python` uses `scirs2-numpy` internally for all its Python binding modules. When you call `scirs2.fft_py(data)` from Python, the data flow is:

```
Python ndarray (NumPy)
    |
    v  zero-copy (via scirs2-numpy, PyReadonlyArray)
Rust ndarray::ArrayView<f64>  (ndarray 0.17)
    |
    v  SciRS2 computation (scirs2-fft, etc.)
Rust ndarray::Array<f64>
    |
    v  zero-copy (via ToPyArray)
Python ndarray (NumPy)
```

This eliminates the ndarray 0.16/0.17 conversion overhead that affected earlier versions.

## Building

This crate is a standard Rust library. Build with:

```bash
cargo build
cargo test
```

To run the Python integration tests, a Python environment with NumPy installed is required:

```bash
pip install pytest numpy
cargo test --features pyo3/auto-initialize
```

## License

Licensed under the Apache License 2.0. See [LICENSE](../LICENSE) for details.

## Authors

COOLJAPAN OU (Team KitaSan)

## Acknowledgments

Based on the upstream [rust-numpy](https://github.com/PyO3/rust-numpy) project by the PyO3 contributors, modified for ndarray 0.17 compatibility and SciRS2 ecosystem integration.
