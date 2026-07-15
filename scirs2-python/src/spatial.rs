//! Python bindings for scirs2-spatial
//!
//! Provides spatial algorithms similar to scipy.spatial

use pyo3::prelude::*;
use pyo3::types::PyDict;
use scirs2_core::ndarray::Array2 as Array2_17;
use scirs2_core::python::numpy_compat::{
    scirs_to_numpy_array1, scirs_to_numpy_array2, Array1, Array2,
};
use scirs2_numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};

// Import KDTree
use scirs2_spatial::distance::EuclideanDistance;
use scirs2_spatial::KDTree;

// Import ConvexHull
use scirs2_spatial::convex_hull::ConvexHull;

// =============================================================================
// Distance Functions
// =============================================================================

/// Euclidean distance between two points
#[pyfunction]
fn euclidean_py(u: PyReadonlyArray1<f64>, v: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let u_arr = u.as_array();
    let v_arr = v.as_array();

    if u_arr.len() != v_arr.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Arrays must have same length",
        ));
    }

    let dist: f64 = u_arr
        .iter()
        .zip(v_arr.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt();

    Ok(dist)
}

/// Manhattan (city block) distance between two points
#[pyfunction]
fn cityblock_py(u: PyReadonlyArray1<f64>, v: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let u_arr = u.as_array();
    let v_arr = v.as_array();

    if u_arr.len() != v_arr.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Arrays must have same length",
        ));
    }

    let dist: f64 = u_arr
        .iter()
        .zip(v_arr.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();

    Ok(dist)
}

/// Chebyshev distance between two points
#[pyfunction]
fn chebyshev_py(u: PyReadonlyArray1<f64>, v: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let u_arr = u.as_array();
    let v_arr = v.as_array();

    if u_arr.len() != v_arr.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Arrays must have same length",
        ));
    }

    let dist: f64 = u_arr
        .iter()
        .zip(v_arr.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);

    Ok(dist)
}

/// Minkowski distance between two points
#[pyfunction]
fn minkowski_py(u: PyReadonlyArray1<f64>, v: PyReadonlyArray1<f64>, p: f64) -> PyResult<f64> {
    let u_arr = u.as_array();
    let v_arr = v.as_array();

    if u_arr.len() != v_arr.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Arrays must have same length",
        ));
    }

    let dist: f64 = u_arr
        .iter()
        .zip(v_arr.iter())
        .map(|(a, b)| (a - b).abs().powf(p))
        .sum::<f64>()
        .powf(1.0 / p);

    Ok(dist)
}

/// Cosine distance between two points
#[pyfunction]
fn cosine_py(u: PyReadonlyArray1<f64>, v: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let u_arr = u.as_array();
    let v_arr = v.as_array();

    if u_arr.len() != v_arr.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Arrays must have same length",
        ));
    }

    let dot: f64 = u_arr.iter().zip(v_arr.iter()).map(|(a, b)| a * b).sum();
    let norm_u: f64 = u_arr.iter().map(|a| a.powi(2)).sum::<f64>().sqrt();
    let norm_v: f64 = v_arr.iter().map(|a| a.powi(2)).sum::<f64>().sqrt();

    if norm_u == 0.0 || norm_v == 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err("Zero vector"));
    }

    Ok(1.0 - dot / (norm_u * norm_v))
}

/// Hamming distance between two points
///
/// Returns the proportion of coordinates at which `u` and `v` differ, matching
/// `scipy.spatial.distance.hamming`.
///
/// This is intentionally *not* named `hamming_py` or `hamming_distance_py` to
/// avoid two separate naming collisions:
/// - `hamming_py` (in the `signal` bindings) is the Hamming *window* function
///   used in signal processing, not a distance metric (see
///   `tests/scipy_comparison/test_spatial_vs_scipy.py`).
/// - `hamming_distance_py` is reserved for a string/character Hamming distance
///   over equal-length strings (backed by
///   `scirs2_text::simd_ops::basic_ops::SimdStringOps::hamming_distance`, see
///   `tests/test_text.py::test_hamming_distance`), which counts differing
///   characters rather than returning a normalized proportion over arrays.
#[pyfunction]
fn spatial_hamming_distance_py(
    u: PyReadonlyArray1<f64>,
    v: PyReadonlyArray1<f64>,
) -> PyResult<f64> {
    let u_arr = u.as_array();
    let v_arr = v.as_array();

    if u_arr.len() != v_arr.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Arrays must have same length",
        ));
    }

    if u_arr.is_empty() {
        return Ok(0.0);
    }

    let n_diff = u_arr
        .iter()
        .zip(v_arr.iter())
        .filter(|&(a, b)| (a - b).abs() > f64::EPSILON)
        .count();

    Ok(n_diff as f64 / u_arr.len() as f64)
}

// =============================================================================
// Pairwise Distance Matrix
// =============================================================================

/// Compute pairwise distances between observations
#[pyfunction]
#[pyo3(signature = (x, metric="euclidean"))]
fn pdist_py(py: Python, x: PyReadonlyArray2<f64>, metric: &str) -> PyResult<Py<PyArray1<f64>>> {
    let x_arr = x.as_array();
    let n = x_arr.nrows();

    // Number of pairwise distances
    let n_dist = n * (n - 1) / 2;
    let mut result = Vec::with_capacity(n_dist);

    for i in 0..n {
        for j in (i + 1)..n {
            let dist = match metric {
                "euclidean" => x_arr
                    .row(i)
                    .iter()
                    .zip(x_arr.row(j).iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt(),
                "cityblock" | "manhattan" => x_arr
                    .row(i)
                    .iter()
                    .zip(x_arr.row(j).iter())
                    .map(|(a, b)| (a - b).abs())
                    .sum(),
                "chebyshev" => x_arr
                    .row(i)
                    .iter()
                    .zip(x_arr.row(j).iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f64::max),
                "hamming" => {
                    let n_cols = x_arr.ncols();
                    if n_cols == 0 {
                        0.0
                    } else {
                        let n_diff = x_arr
                            .row(i)
                            .iter()
                            .zip(x_arr.row(j).iter())
                            .filter(|&(a, b)| (a - b).abs() > f64::EPSILON)
                            .count();
                        n_diff as f64 / n_cols as f64
                    }
                }
                _ => x_arr
                    .row(i)
                    .iter()
                    .zip(x_arr.row(j).iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt(),
            };
            result.push(dist);
        }
    }

    scirs_to_numpy_array1(Array1::from_vec(result), py)
}

/// Compute pairwise distances between two sets of observations
#[pyfunction]
#[pyo3(signature = (xa, xb, metric="euclidean"))]
fn cdist_py(
    py: Python,
    xa: PyReadonlyArray2<f64>,
    xb: PyReadonlyArray2<f64>,
    metric: &str,
) -> PyResult<Py<PyArray2<f64>>> {
    let xa_arr = xa.as_array();
    let xb_arr = xb.as_array();
    let na = xa_arr.nrows();
    let nb = xb_arr.nrows();

    if xa_arr.ncols() != xb_arr.ncols() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Arrays must have same number of columns",
        ));
    }

    let mut result = Vec::with_capacity(na * nb);

    for i in 0..na {
        for j in 0..nb {
            let dist = match metric {
                "euclidean" => xa_arr
                    .row(i)
                    .iter()
                    .zip(xb_arr.row(j).iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt(),
                "cityblock" | "manhattan" => xa_arr
                    .row(i)
                    .iter()
                    .zip(xb_arr.row(j).iter())
                    .map(|(a, b)| (a - b).abs())
                    .sum(),
                "chebyshev" => xa_arr
                    .row(i)
                    .iter()
                    .zip(xb_arr.row(j).iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f64::max),
                "hamming" => {
                    let n_cols = xa_arr.ncols();
                    if n_cols == 0 {
                        0.0
                    } else {
                        let n_diff = xa_arr
                            .row(i)
                            .iter()
                            .zip(xb_arr.row(j).iter())
                            .filter(|&(a, b)| (a - b).abs() > f64::EPSILON)
                            .count();
                        n_diff as f64 / n_cols as f64
                    }
                }
                _ => xa_arr
                    .row(i)
                    .iter()
                    .zip(xb_arr.row(j).iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt(),
            };
            result.push(dist);
        }
    }

    // Reshape to 2D array
    let arr = Array2::from_shape_vec((na, nb), result)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;

    scirs_to_numpy_array2(arr, py)
}

/// Convert condensed distance matrix to square form
#[pyfunction]
fn squareform_py(py: Python, x: PyReadonlyArray1<f64>) -> PyResult<Py<PyArray2<f64>>> {
    let x_arr = x.as_array();
    let n_dist = x_arr.len();

    // Solve n*(n-1)/2 = n_dist for n
    let n = ((1.0 + (1.0 + 8.0 * n_dist as f64).sqrt()) / 2.0) as usize;

    let mut result = Array2::zeros((n, n));

    let mut idx = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            result[[i, j]] = x_arr[idx];
            result[[j, i]] = x_arr[idx];
            idx += 1;
        }
    }

    scirs_to_numpy_array2(result, py)
}

// =============================================================================
// Convex Hull
// =============================================================================

/// Compute the convex hull of a set of points
///
/// Returns indices of points that form the convex hull vertices
#[pyfunction]
fn convex_hull_py(py: Python, points: PyReadonlyArray2<f64>) -> PyResult<Py<PyAny>> {
    let points_arr = points.as_array();
    let n = points_arr.nrows();
    let k = points_arr.ncols();

    // Convert to Array2_17
    let mut pts = Vec::with_capacity(n * k);
    for row in points_arr.rows() {
        for &val in row.iter() {
            pts.push(val);
        }
    }
    let arr = Array2_17::from_shape_vec((n, k), pts)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))?;

    let hull = ConvexHull::new(&arr.view())
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;

    // Get vertices and simplices
    let vertices: Vec<i64> = hull.vertex_indices().iter().map(|&i| i as i64).collect();
    let simplices: Vec<Vec<i64>> = hull
        .simplices()
        .iter()
        .map(|s| s.iter().map(|&i| i as i64).collect())
        .collect();

    // Calculate volume and area
    let volume = hull.volume().unwrap_or(0.0);
    let area = hull.area().unwrap_or(0.0);

    let dict = PyDict::new(py);
    dict.set_item(
        "vertices",
        scirs_to_numpy_array1(Array1::from_vec(vertices), py)?,
    )?;

    // Convert simplices to a flat representation for Python
    let simplices_py: Vec<Vec<i64>> = simplices;
    dict.set_item("simplices", simplices_py)?;
    dict.set_item("volume", volume)?;
    dict.set_item("area", area)?;

    Ok(dict.into())
}

/// ConvexHull class for working with convex hulls
#[pyclass(name = "ConvexHullPy", unsendable)]
pub struct PyConvexHull {
    hull: ConvexHull,
}

#[pymethods]
impl PyConvexHull {
    /// Create a new ConvexHull from a 2D array of points
    ///
    /// Parameters:
    /// - points: Array of shape (n, k) containing n points in k dimensions
    #[new]
    fn new(points: PyReadonlyArray2<f64>) -> PyResult<Self> {
        let points_arr = points.as_array();
        let n = points_arr.nrows();
        let k = points_arr.ncols();

        // Convert to Array2_17
        let mut pts = Vec::with_capacity(n * k);
        for row in points_arr.rows() {
            for &val in row.iter() {
                pts.push(val);
            }
        }
        let arr = Array2_17::from_shape_vec((n, k), pts)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))?;

        let hull = ConvexHull::new(&arr.view())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;

        Ok(PyConvexHull { hull })
    }

    /// Get the indices of vertices that form the convex hull
    fn vertices(&self, py: Python) -> PyResult<Py<PyArray1<i64>>> {
        let vertices: Vec<i64> = self
            .hull
            .vertex_indices()
            .iter()
            .map(|&i| i as i64)
            .collect();
        scirs_to_numpy_array1(Array1::from_vec(vertices), py)
    }

    /// Get the simplices (facets) of the convex hull
    fn simplices(&self) -> Vec<Vec<i64>> {
        self.hull
            .simplices()
            .iter()
            .map(|s| s.iter().map(|&i| i as i64).collect())
            .collect()
    }

    /// Calculate the volume of the convex hull
    fn volume(&self) -> PyResult<f64> {
        self.hull
            .volume()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))
    }

    /// Calculate the surface area of the convex hull
    fn area(&self) -> PyResult<f64> {
        self.hull
            .area()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))
    }

    /// Check if a point is inside the convex hull
    fn contains(&self, point: PyReadonlyArray1<f64>) -> PyResult<bool> {
        let point_vec: Vec<f64> = point.as_array().to_vec();
        self.hull
            .contains(&point_vec)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))
    }
}

// =============================================================================
// KD-Tree
// =============================================================================

/// KD-Tree for efficient nearest neighbor searches
#[pyclass(name = "KDTree")]
pub struct PyKDTree {
    tree: KDTree<f64, EuclideanDistance<f64>>,
}

#[pymethods]
impl PyKDTree {
    /// Create a new KD-Tree from a 2D array of points
    ///
    /// Parameters:
    /// - data: Array of shape (n, k) containing n points in k dimensions
    #[new]
    fn new(data: PyReadonlyArray2<f64>) -> PyResult<Self> {
        let data_arr = data.as_array();
        let n = data_arr.nrows();
        let k = data_arr.ncols();

        // Convert to Array2_17
        let mut points = Vec::with_capacity(n * k);
        for row in data_arr.rows() {
            for &val in row.iter() {
                points.push(val);
            }
        }
        let arr = Array2_17::from_shape_vec((n, k), points)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))?;

        let tree = KDTree::new(&arr)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;

        Ok(PyKDTree { tree })
    }

    /// Query the tree for the k nearest neighbors to a point
    ///
    /// Parameters:
    /// - point: Query point
    /// - k: Number of nearest neighbors to find
    ///
    /// Returns:
    /// - Tuple of (indices, distances) arrays
    fn query(&self, py: Python, point: PyReadonlyArray1<f64>, k: usize) -> PyResult<Py<PyAny>> {
        let point_vec: Vec<f64> = point.as_array().to_vec();

        let (indices, distances) = self
            .tree
            .query(&point_vec, k)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;

        let dict = PyDict::new(py);
        dict.set_item(
            "indices",
            scirs_to_numpy_array1(
                Array1::from_vec(indices.iter().map(|&i| i as i64).collect()),
                py,
            )?,
        )?;
        dict.set_item(
            "distances",
            scirs_to_numpy_array1(Array1::from_vec(distances), py)?,
        )?;

        Ok(dict.into())
    }

    /// Query the tree for all points within a given radius
    ///
    /// Parameters:
    /// - point: Query point
    /// - r: Radius
    ///
    /// Returns:
    /// - Tuple of (indices, distances) arrays
    fn query_radius(
        &self,
        py: Python,
        point: PyReadonlyArray1<f64>,
        r: f64,
    ) -> PyResult<Py<PyAny>> {
        let point_vec: Vec<f64> = point.as_array().to_vec();

        let (indices, distances) = self
            .tree
            .query_radius(&point_vec, r)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;

        let dict = PyDict::new(py);
        dict.set_item(
            "indices",
            scirs_to_numpy_array1(
                Array1::from_vec(indices.iter().map(|&i| i as i64).collect()),
                py,
            )?,
        )?;
        dict.set_item(
            "distances",
            scirs_to_numpy_array1(Array1::from_vec(distances), py)?,
        )?;

        Ok(dict.into())
    }
}

/// Python module registration
pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Distance functions
    m.add_function(wrap_pyfunction!(euclidean_py, m)?)?;
    m.add_function(wrap_pyfunction!(cityblock_py, m)?)?;
    m.add_function(wrap_pyfunction!(chebyshev_py, m)?)?;
    m.add_function(wrap_pyfunction!(minkowski_py, m)?)?;
    m.add_function(wrap_pyfunction!(cosine_py, m)?)?;
    m.add_function(wrap_pyfunction!(spatial_hamming_distance_py, m)?)?;

    // Pairwise distances
    m.add_function(wrap_pyfunction!(pdist_py, m)?)?;
    m.add_function(wrap_pyfunction!(cdist_py, m)?)?;
    m.add_function(wrap_pyfunction!(squareform_py, m)?)?;

    // Convex hull
    m.add_function(wrap_pyfunction!(convex_hull_py, m)?)?;
    m.add_class::<PyConvexHull>()?;

    // Spatial data structures
    m.add_class::<PyKDTree>()?;

    Ok(())
}
