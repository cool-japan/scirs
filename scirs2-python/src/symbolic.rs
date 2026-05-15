//! Python bindings for scirs2-symbolic.
//!
//! Exposes the EML substrate (`EmlTree`, `Canonical`, `LoweredOp`),
//! evaluation (`eval_real`), gradient (`grad`), and the symbolic-regression
//! API (`discover`) under the Python sub-namespace `scirs2.symbolic`.
//!
//! # Example (Python)
//!
//! ```python
//! import scirs2 as s2
//! import numpy as np
//!
//! # Build an EML tree: sin(x²)
//! x = s2.symbolic.EmlTree.var(0)
//! formula = s2.symbolic.Canonical.sin(s2.symbolic.Canonical.mul(x, x))
//! lowered = s2.symbolic.lower(formula)
//!
//! # Evaluate at x = 0.5
//! result = s2.symbolic.eval_real(lowered, [0.5])
//! print(result)  # ~0.247
//!
//! # Symbolic gradient with respect to variable 0
//! grad_op = s2.symbolic.grad(lowered, 0)
//!
//! # Symbolic regression
//! features = np.array([[1.0], [2.0], [3.0]])
//! targets = np.array([1.0, 4.0, 9.0])
//! results = s2.symbolic.discover(features, targets)
//! ```
//!
//! Note: `Canonical::sin` produces a 543-node-deep canonical EML tree;
//! evaluation is iterative (no stack blowup).

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use scirs2_numpy::{PyReadonlyArray1, PyReadonlyArray2};

use scirs2_symbolic::eml::eval::{eval_real as rust_eval_real, EvalCtx};
use scirs2_symbolic::eml::{
    grad as rust_grad, lower as rust_lower, simplify_op as rust_simplify_op,
    Canonical as RustCanonical, EmlTree as RustEmlTree, LoweredOp as RustLoweredOp,
};
use scirs2_symbolic::regression::{discover as rust_discover, SrConfig as RustSrConfig};

// ============================================================================
// EmlTree wrapper
// ============================================================================

/// Python wrapper for [`scirs2_symbolic::eml::EmlTree`].
#[pyclass(name = "EmlTree", module = "scirs2.symbolic", skip_from_py_object)]
#[derive(Clone)]
pub struct PyEmlTree {
    inner: RustEmlTree,
}

#[pymethods]
impl PyEmlTree {
    /// Construct the constant `1` — the only EML leaf.
    #[staticmethod]
    fn one() -> Self {
        Self {
            inner: RustEmlTree::one(),
        }
    }

    /// Construct a variable at index `idx`.
    #[staticmethod]
    fn var(idx: usize) -> Self {
        Self {
            inner: RustEmlTree::var(idx),
        }
    }

    /// Construct `eml(left, right) = exp(left) - ln(right)`.
    #[staticmethod]
    fn eml(left: &Self, right: &Self) -> Self {
        Self {
            inner: RustEmlTree::eml(&left.inner, &right.inner),
        }
    }

    /// Tree depth.
    fn depth(&self) -> usize {
        self.inner.depth()
    }

    /// Total node count.
    fn size(&self) -> usize {
        self.inner.size()
    }

    /// Number of distinct variables (max var index + 1, or 0 if none).
    fn num_vars(&self) -> usize {
        self.inner.num_vars()
    }

    /// Structural hash returned as `(high_u64, low_u64)`, since Python lacks
    /// a native `u128` type.
    fn structural_hash(&self) -> (u64, u64) {
        let h = self.inner.structural_hash();
        ((h >> 64) as u64, (h & 0xFFFF_FFFF_FFFF_FFFF) as u64)
    }

    fn __repr__(&self) -> String {
        format!(
            "EmlTree(depth={}, size={}, num_vars={})",
            self.depth(),
            self.size(),
            self.num_vars()
        )
    }
}

// ============================================================================
// Canonical namespace
// ============================================================================

/// Namespace for canonical EML constructors.
///
/// Mirrors `scirs2_symbolic::eml::Canonical`. Every method returns a
/// canonical [`PyEmlTree`] for the named elementary function.
#[pyclass(name = "Canonical", module = "scirs2.symbolic")]
pub struct PyCanonical;

#[pymethods]
impl PyCanonical {
    // ----- Basic operations -----
    /// `exp(x)`.
    #[staticmethod]
    fn exp(x: &PyEmlTree) -> PyEmlTree {
        PyEmlTree {
            inner: RustCanonical::exp(&x.inner),
        }
    }
    /// `ln(x)`.
    #[staticmethod]
    fn ln(x: &PyEmlTree) -> PyEmlTree {
        PyEmlTree {
            inner: RustCanonical::ln(&x.inner),
        }
    }
    /// Euler's number `e`.
    #[staticmethod]
    fn euler() -> PyEmlTree {
        PyEmlTree {
            inner: RustCanonical::euler(),
        }
    }
    /// `pi` (encoded such that complex evaluation yields `iπ`).
    #[staticmethod]
    fn pi() -> PyEmlTree {
        PyEmlTree {
            inner: RustCanonical::pi(),
        }
    }
    /// Negation `-x`.
    #[staticmethod]
    fn neg(x: &PyEmlTree) -> PyEmlTree {
        PyEmlTree {
            inner: RustCanonical::neg(&x.inner),
        }
    }

    // ----- Arithmetic -----
    /// `a + b`.
    #[staticmethod]
    fn add(a: &PyEmlTree, b: &PyEmlTree) -> PyEmlTree {
        PyEmlTree {
            inner: RustCanonical::add(&a.inner, &b.inner),
        }
    }
    /// `a - b`.
    #[staticmethod]
    fn sub(a: &PyEmlTree, b: &PyEmlTree) -> PyEmlTree {
        PyEmlTree {
            inner: RustCanonical::sub(&a.inner, &b.inner),
        }
    }
    /// `a * b`.
    #[staticmethod]
    fn mul(a: &PyEmlTree, b: &PyEmlTree) -> PyEmlTree {
        PyEmlTree {
            inner: RustCanonical::mul(&a.inner, &b.inner),
        }
    }
    /// `a / b`.
    #[staticmethod]
    fn div(a: &PyEmlTree, b: &PyEmlTree) -> PyEmlTree {
        PyEmlTree {
            inner: RustCanonical::div(&a.inner, &b.inner),
        }
    }
    /// `a ^ b` (power).
    #[staticmethod]
    fn pow(a: &PyEmlTree, b: &PyEmlTree) -> PyEmlTree {
        PyEmlTree {
            inner: RustCanonical::pow(&a.inner, &b.inner),
        }
    }

    // ----- Trig -----
    /// `sin(x)`.
    #[staticmethod]
    fn sin(x: &PyEmlTree) -> PyEmlTree {
        PyEmlTree {
            inner: RustCanonical::sin(&x.inner),
        }
    }
    /// `cos(x)`.
    #[staticmethod]
    fn cos(x: &PyEmlTree) -> PyEmlTree {
        PyEmlTree {
            inner: RustCanonical::cos(&x.inner),
        }
    }
    /// `tan(x)`.
    #[staticmethod]
    fn tan(x: &PyEmlTree) -> PyEmlTree {
        PyEmlTree {
            inner: RustCanonical::tan(&x.inner),
        }
    }

    // ----- Inverse trig -----
    /// `arcsin(x)`.
    #[staticmethod]
    fn arcsin(x: &PyEmlTree) -> PyEmlTree {
        PyEmlTree {
            inner: RustCanonical::arcsin(&x.inner),
        }
    }
    /// `arccos(x)`.
    #[staticmethod]
    fn arccos(x: &PyEmlTree) -> PyEmlTree {
        PyEmlTree {
            inner: RustCanonical::arccos(&x.inner),
        }
    }
    /// `arctan(x)`.
    #[staticmethod]
    fn arctan(x: &PyEmlTree) -> PyEmlTree {
        PyEmlTree {
            inner: RustCanonical::arctan(&x.inner),
        }
    }

    // ----- Hyperbolic -----
    /// `sinh(x)`.
    #[staticmethod]
    fn sinh(x: &PyEmlTree) -> PyEmlTree {
        PyEmlTree {
            inner: RustCanonical::sinh(&x.inner),
        }
    }
    /// `cosh(x)`.
    #[staticmethod]
    fn cosh(x: &PyEmlTree) -> PyEmlTree {
        PyEmlTree {
            inner: RustCanonical::cosh(&x.inner),
        }
    }
    /// `tanh(x)`.
    #[staticmethod]
    fn tanh(x: &PyEmlTree) -> PyEmlTree {
        PyEmlTree {
            inner: RustCanonical::tanh(&x.inner),
        }
    }

    // ----- Inverse hyperbolic -----
    /// `arcsinh(x)`.
    #[staticmethod]
    fn arcsinh(x: &PyEmlTree) -> PyEmlTree {
        PyEmlTree {
            inner: RustCanonical::arcsinh(&x.inner),
        }
    }
    /// `arccosh(x)`.
    #[staticmethod]
    fn arccosh(x: &PyEmlTree) -> PyEmlTree {
        PyEmlTree {
            inner: RustCanonical::arccosh(&x.inner),
        }
    }
    /// `arctanh(x)`.
    #[staticmethod]
    fn arctanh(x: &PyEmlTree) -> PyEmlTree {
        PyEmlTree {
            inner: RustCanonical::arctanh(&x.inner),
        }
    }

    // ----- Powers, roots, abs -----
    /// `sqrt(x)`.
    #[staticmethod]
    fn sqrt(x: &PyEmlTree) -> PyEmlTree {
        PyEmlTree {
            inner: RustCanonical::sqrt(&x.inner),
        }
    }
    /// `|x|`.
    #[staticmethod]
    fn abs(x: &PyEmlTree) -> PyEmlTree {
        PyEmlTree {
            inner: RustCanonical::abs(&x.inner),
        }
    }
    /// `x²`.
    #[staticmethod]
    fn square(x: &PyEmlTree) -> PyEmlTree {
        PyEmlTree {
            inner: RustCanonical::square(&x.inner),
        }
    }
    /// `1 / x`.
    #[staticmethod]
    fn reciprocal(x: &PyEmlTree) -> PyEmlTree {
        PyEmlTree {
            inner: RustCanonical::reciprocal(&x.inner),
        }
    }

    // ----- Constants -----
    /// Natural number `n >= 1`. Raises `ValueError` on `n == 0`
    /// (use `zero()` for the additive identity).
    #[staticmethod]
    fn nat(n: u64) -> PyResult<PyEmlTree> {
        RustCanonical::nat(n)
            .map(|t| PyEmlTree { inner: t })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Additive identity `0`.
    #[staticmethod]
    fn zero() -> PyEmlTree {
        PyEmlTree {
            inner: RustCanonical::zero(),
        }
    }

    /// Negative one `-1`.
    #[staticmethod]
    fn neg_one() -> PyEmlTree {
        PyEmlTree {
            inner: RustCanonical::neg_one(),
        }
    }

    /// Imaginary unit `i = exp(iπ/2)` (purely imaginary; `eval_real` errors).
    #[staticmethod]
    fn imag_unit() -> PyEmlTree {
        PyEmlTree {
            inner: RustCanonical::imag_unit(),
        }
    }
}

// ============================================================================
// LoweredOp wrapper
// ============================================================================

/// Python wrapper for [`scirs2_symbolic::eml::LoweredOp`] — the flat
/// operator IR produced by `lower`.
#[pyclass(name = "LoweredOp", module = "scirs2.symbolic", skip_from_py_object)]
#[derive(Clone)]
pub struct PyLoweredOp {
    inner: RustLoweredOp,
}

#[pymethods]
impl PyLoweredOp {
    /// Number of distinct variables (max var index + 1, or 0 if none).
    fn count_vars(&self) -> usize {
        self.inner.count_vars()
    }

    /// Structural hash returned as `(high_u64, low_u64)`.
    fn structural_hash(&self) -> (u64, u64) {
        let h = self.inner.structural_hash();
        ((h >> 64) as u64, (h & 0xFFFF_FFFF_FFFF_FFFF) as u64)
    }

    fn __repr__(&self) -> String {
        format!("LoweredOp(count_vars={})", self.count_vars())
    }
}

// ============================================================================
// Top-level functions
// ============================================================================

/// Lower an [`PyEmlTree`] to a [`PyLoweredOp`].
#[pyfunction]
fn lower(tree: &PyEmlTree) -> PyLoweredOp {
    PyLoweredOp {
        inner: rust_lower(&tree.inner),
    }
}

/// Algebraically simplify a [`PyLoweredOp`].
#[pyfunction]
fn simplify(op: &PyLoweredOp) -> PyLoweredOp {
    PyLoweredOp {
        inner: rust_simplify_op(&op.inner),
    }
}

/// Symbolic gradient with respect to variable `wrt`.
#[pyfunction]
fn grad(op: &PyLoweredOp, wrt: usize) -> PyLoweredOp {
    PyLoweredOp {
        inner: rust_grad(&op.inner, wrt),
    }
}

/// Evaluate a [`PyLoweredOp`] at the given real variable values.
///
/// `vars[i]` binds variable index `i`. Raises `RuntimeError` on
/// numerical-domain failures (e.g. `ln(0)`).
#[pyfunction]
fn eval_real(op: &PyLoweredOp, vars: Vec<f64>) -> PyResult<f64> {
    let ctx = EvalCtx::new(&vars);
    rust_eval_real(&op.inner, &ctx).map_err(|e| PyRuntimeError::new_err(e.to_string()))
}

// ============================================================================
// Symbolic regression
// ============================================================================

/// Beam-search symbolic regression — discovers formulas approximating
/// `targets ≈ f(features)`.
///
/// `features` has shape `(n_samples, n_features)`; `targets` has shape
/// `(n_samples,)`. Returns up to `top_n` formulas, ranked by combined
/// fitness (lower is better).
#[pyfunction]
#[pyo3(signature = (
    features,
    targets,
    max_iter = 50,
    top_n = 3,
    beam_width = 32,
    max_depth = 6,
    max_nodes = 20,
))]
#[allow(clippy::too_many_arguments)]
fn discover(
    py: Python<'_>,
    features: PyReadonlyArray2<f64>,
    targets: PyReadonlyArray1<f64>,
    max_iter: usize,
    top_n: usize,
    beam_width: usize,
    max_depth: usize,
    max_nodes: usize,
) -> PyResult<Vec<PyDiscoveredFormula>> {
    let features_arr = features.as_array();
    let targets_arr = targets.as_array();

    let config = RustSrConfig::default()
        .with_max_iter(max_iter)
        .with_top_n(top_n)
        .with_beam_width(beam_width)
        .with_max_depth(max_depth)
        .with_max_nodes(max_nodes);

    // Release the GIL while running the beam search (PyO3 0.28: detach == old allow_threads).
    let results = py.detach(|| rust_discover(features_arr, targets_arr, &config));

    Ok(results
        .into_iter()
        .map(|f| PyDiscoveredFormula {
            op: PyLoweredOp { inner: f.op },
            mse: f.fitness.mse,
            r_squared: f.fitness.r_squared,
            combined: f.fitness.combined,
            node_count: f.node_count,
            n_vars: f.n_vars,
        })
        .collect())
}

/// Python view of a discovered formula returned by `discover`.
#[pyclass(
    name = "DiscoveredFormula",
    module = "scirs2.symbolic",
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyDiscoveredFormula {
    /// The lowered operator IR.
    #[pyo3(get)]
    pub op: PyLoweredOp,
    /// Mean-squared error on the training data.
    #[pyo3(get)]
    pub mse: f64,
    /// Coefficient of determination `R²`.
    #[pyo3(get)]
    pub r_squared: f64,
    /// Combined fitness (MSE + parsimony penalty); lower is better.
    #[pyo3(get)]
    pub combined: f64,
    /// Total node count of the formula.
    #[pyo3(get)]
    pub node_count: usize,
    /// Number of distinct variables used.
    #[pyo3(get)]
    pub n_vars: usize,
}

#[pymethods]
impl PyDiscoveredFormula {
    fn __repr__(&self) -> String {
        format!(
            "DiscoveredFormula(mse={:.6}, r_squared={:.6}, n_nodes={}, n_vars={})",
            self.mse, self.r_squared, self.node_count, self.n_vars
        )
    }
}

// ============================================================================
// Module registration
// ============================================================================

/// Register the `symbolic` sub-namespace on the parent `scirs2` module.
pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    let symbolic = PyModule::new(py, "symbolic")?;

    symbolic.add_class::<PyEmlTree>()?;
    symbolic.add_class::<PyCanonical>()?;
    symbolic.add_class::<PyLoweredOp>()?;
    symbolic.add_class::<PyDiscoveredFormula>()?;

    symbolic.add_function(wrap_pyfunction!(lower, &symbolic)?)?;
    symbolic.add_function(wrap_pyfunction!(simplify, &symbolic)?)?;
    symbolic.add_function(wrap_pyfunction!(grad, &symbolic)?)?;
    symbolic.add_function(wrap_pyfunction!(eval_real, &symbolic)?)?;
    symbolic.add_function(wrap_pyfunction!(discover, &symbolic)?)?;

    symbolic.add(
        "__doc__",
        "Symbolic mathematics — EML substrate, evaluation, gradient, and \
         beam-search symbolic regression.\n\nClasses:\n  - EmlTree: uniform \
         binary EML tree (constant 1 + var leaves + binary eml nodes).\n  - \
         Canonical: namespace of elementary-function constructors.\n  - \
         LoweredOp: flat operator IR produced by lower(tree).\n  - \
         DiscoveredFormula: result of discover().\n\nFunctions:\n  - \
         lower(tree) -> LoweredOp\n  - simplify(op) -> LoweredOp\n  - grad(op, wrt) \
         -> LoweredOp\n  - eval_real(op, vars) -> float\n  - discover(features, \
         targets, ...) -> list[DiscoveredFormula]",
    )?;

    m.add_submodule(&symbolic)?;
    Ok(())
}
