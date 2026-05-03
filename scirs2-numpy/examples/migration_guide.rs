//! rust-numpy to scirs2-numpy migration guide.
//!
//! Prints a comprehensive table showing the equivalent API between
//! `rust-numpy` (the upstream crate) and `scirs2-numpy` (this fork).
//!
//! Run with:
//! ```bash
//! cargo run --example migration_guide -p scirs2-numpy
//! ```

fn main() {
    // Each entry: (rust-numpy expression, scirs2-numpy expression, notes)
    let entries: &[(&str, &str, &str)] = &[
        // ── Imports ──────────────────────────────────────────────────────
        (
            "use numpy::PyArray1",
            "use scirs2_numpy::PyArray1",
            "Direct drop-in replacement",
        ),
        (
            "use numpy::PyArray2",
            "use scirs2_numpy::PyArray2",
            "Direct drop-in replacement",
        ),
        (
            "use numpy::PyArrayDyn",
            "use scirs2_numpy::PyArrayDyn",
            "Dynamic-dimension array",
        ),
        (
            "use numpy::PyArrayMethods",
            "use scirs2_numpy::PyArrayMethods",
            "Method-trait for PyArray",
        ),
        (
            "use numpy::PyReadonlyArray1",
            "use scirs2_numpy::PyReadonlyArray1",
            "Read-only borrow guard",
        ),
        (
            "use numpy::PyReadwriteArray1",
            "use scirs2_numpy::PyReadwriteArray1",
            "Read-write borrow guard",
        ),
        (
            "use numpy::ToPyArray",
            "use scirs2_numpy::ToPyArray",
            "ndarray -> PyArray conversion",
        ),
        (
            "use numpy::IntoPyArray",
            "use scirs2_numpy::IntoPyArray",
            "Consuming ndarray -> PyArray",
        ),
        (
            "use numpy::Element",
            "use scirs2_numpy::Element",
            "NumPy element type trait",
        ),
        (
            "use numpy::Complex32",
            "use scirs2_numpy::Complex32",
            "Complex f32 element type",
        ),
        (
            "use numpy::Complex64",
            "use scirs2_numpy::Complex64",
            "Complex f64 element type",
        ),
        (
            "use numpy::dtype",
            "use scirs2_numpy::dtype",
            "Runtime dtype introspection",
        ),
        (
            "use numpy::PyArrayDescr",
            "use scirs2_numpy::PyArrayDescr",
            "Array descriptor / dtype object",
        ),
        (
            "use numpy::PyArrayDescrMethods",
            "use scirs2_numpy::PyArrayDescrMethods",
            "Methods on PyArrayDescr",
        ),
        (
            "use numpy::PyUntypedArray",
            "use scirs2_numpy::PyUntypedArray",
            "Type-erased PyArray reference",
        ),
        (
            "use numpy::PyUntypedArrayMethods",
            "use scirs2_numpy::PyUntypedArrayMethods",
            "Methods on PyUntypedArray",
        ),
        // ── Borrow API ───────────────────────────────────────────────────
        (
            "array.readonly()",
            "array.readonly()",
            "Unchanged — returns PyReadonlyArray",
        ),
        (
            "array.readwrite()",
            "array.readwrite()",
            "Unchanged — returns PyReadwriteArray",
        ),
        (
            "readonly.as_array()",
            "readonly.as_array()",
            "ndarray ArrayView via borrow",
        ),
        (
            "readwrite.as_array_mut()",
            "readwrite.as_array_mut()",
            "Mutable ArrayViewMut via borrow",
        ),
        // ── Conversion ───────────────────────────────────────────────────
        (
            "arr.to_pyarray(py)",
            "arr.to_pyarray(py)",
            "Copies ndarray into Python heap",
        ),
        (
            "arr.into_pyarray(py)",
            "arr.into_pyarray(py)",
            "Moves ndarray into Python heap",
        ),
        (
            "PyArray1::from_vec(py, v)",
            "PyArray1::from_vec(py, v)",
            "Vec<T> into 1-D PyArray",
        ),
        (
            "PyArray1::from_slice(py, s)",
            "PyArray1::from_slice(py, s)",
            "Slice into 1-D PyArray (copies)",
        ),
        // ── DLPack (NEW in scirs2-numpy) ─────────────────────────────────
        (
            "(not available)",
            "use scirs2_numpy::DLTensor",
            "DLPack C ABI tensor struct",
        ),
        (
            "(not available)",
            "use scirs2_numpy::DlpackError",
            "DLPack validation error type",
        ),
        (
            "(not available)",
            "validate_dlpack_tensor(&t)",
            "Validate any DLPack tensor",
        ),
        (
            "(not available)",
            "validate_torch_dlpack_tensor(&t)",
            "Validate PyTorch-origin tensor",
        ),
        (
            "(not available)",
            "validate_jax_dlpack_tensor(&t)",
            "Validate JAX-origin tensor (CPU/GPU/TPU)",
        ),
        (
            "(not available)",
            "dlpack_from_slice(data, shape)",
            "Build a DLTensor borrowing a slice",
        ),
        (
            "(not available)",
            "dlpack_to_vec_f64(&t)",
            "Extract f64 Vec from DLTensor",
        ),
        (
            "(not available)",
            "dlarray_from_torch_f32(&t)",
            "ArrayViewD<f32> from Torch tensor",
        ),
        (
            "(not available)",
            "dlarray_from_torch_f64(&t)",
            "ArrayViewD<f64> from Torch tensor",
        ),
        (
            "(not available)",
            "array_from_dlpack_f32(&t)",
            "Generic ArrayViewD<f32> from DLPack",
        ),
        (
            "(not available)",
            "array_from_dlpack_f64(&t)",
            "Generic ArrayViewD<f64> from DLPack",
        ),
        (
            "(not available)",
            "jax_device_type(&t)",
            "Identify JAX device (CPU/GPU/TPU)",
        ),
        (
            "(not available)",
            "DLPackCapsule::new(shape, code, bits)",
            "Create a DLPack PyCapsule",
        ),
        // ── Array protocol (NEW in scirs2-numpy) ─────────────────────────
        (
            "(not available)",
            "use scirs2_numpy::ArrayProtocol",
            "__array_interface__ support",
        ),
        (
            "(not available)",
            "use scirs2_numpy::NdArrayWrapper",
            "ndarray implementing __array__",
        ),
        // ── Masked arrays (NEW in scirs2-numpy) ──────────────────────────
        (
            "(not available)",
            "use scirs2_numpy::MaskedArray",
            "numpy.ma-compatible masked array",
        ),
        (
            "(not available)",
            "masked_array(data, mask)",
            "Construct a masked array",
        ),
        (
            "(not available)",
            "masked_less(arr, thresh)",
            "Mask elements less than threshold",
        ),
        // ── Structured arrays (NEW in scirs2-numpy) ──────────────────────
        (
            "(not available)",
            "use scirs2_numpy::StructuredArray",
            "Record array with named fields",
        ),
        (
            "(not available)",
            "use scirs2_numpy::StructuredDtype",
            "Dtype for record arrays",
        ),
        // ── Prelude ──────────────────────────────────────────────────────
        (
            "use numpy::prelude::*",
            "use scirs2_numpy::prelude::*",
            "Convenience glob import",
        ),
        // ── ndarray version change ────────────────────────────────────────
        (
            "ndarray 0.15 / 0.16",
            "ndarray 0.17",
            "scirs2-numpy targets ndarray 0.17",
        ),
        // ── pyo3 compatibility ───────────────────────────────────────────
        (
            "pyo3 0.20 – 0.22",
            "pyo3 0.23+",
            "scirs2-numpy follows latest pyo3",
        ),
    ];

    let col_w = 44usize;
    let note_w = 40usize;
    println!("{:<col_w$} {:<col_w$} Notes", "rust-numpy", "scirs2-numpy");
    println!("{}", "-".repeat(col_w * 2 + note_w + 2));
    for (old, new, note) in entries {
        println!("{:<col_w$} {:<col_w$} {}", old, new, note);
    }
    println!();
    println!(
        "Total entries: {}. See scirs2-numpy docs for full API reference.",
        entries.len()
    );
}
