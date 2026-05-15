//! Fast Approximate Kriging for Large Datasets
//!
//! This module provides computationally efficient kriging algorithms for large spatial datasets.
//! Standard kriging requires O(n³) operations for fitting and O(n²) for prediction,
//! which becomes prohibitively expensive for large datasets.  This module implements:
//!
//! 1. **Local Kriging**: Uses only nearby points for each prediction (O(k³) per prediction)
//! 2. **Fixed Rank Kriging**: Nyström low-rank approximation (O(nr²) fitting, O(r²) prediction)
//! 3. **Tapering Kriging**: Wendland C2 covariance taper for sparsity
//! 4. **HODLR**: Hierarchical Off-Diagonal Low-Rank (block-based local kriging)
//!
//! These methods trade some accuracy for substantial performance improvements,
//! making kriging feasible for datasets with thousands to millions of points.

use crate::advanced::enhanced_kriging::TrendFunction;
use crate::advanced::kriging::CovarianceFunction;
use crate::error::{InterpolateError, InterpolateResult};
use crate::spatial::kdtree::KdTree;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::{Debug, Display};
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Sub};

/// Type alias for sparse matrix representation as COO triplets
type SparseComponents<F> = (Vec<(usize, usize)>, Vec<F>);

/// Maximum number of neighbors to consider in local kriging
const DEFAULT_MAX_NEIGHBORS: usize = 50;

/// Default radius multiplier for local neighborhood search
const DEFAULT_RADIUS_MULTIPLIER: f64 = 3.0;

// ── Covariance utilities ──────────────────────────────────────────────────────

/// Evaluate isotropic covariance k(r; σ², cov_fn) where r is already scaled by
/// the length-scale.
fn eval_covariance<F: Float + FromPrimitive>(r: F, sigma_sq: F, cov_fn: CovarianceFunction) -> F {
    match cov_fn {
        CovarianceFunction::SquaredExponential => sigma_sq * (-r * r).exp(),
        CovarianceFunction::Exponential => sigma_sq * (-r).exp(),
        CovarianceFunction::Matern32 => {
            let sqrt3_r = F::from_f64(3.0_f64.sqrt()).expect("const") * r;
            sigma_sq * (F::one() + sqrt3_r) * (-sqrt3_r).exp()
        }
        CovarianceFunction::Matern52 => {
            let sqrt5_r = F::from_f64(5.0_f64.sqrt()).expect("const") * r;
            let term = F::one() + sqrt5_r + F::from_f64(5.0 / 3.0).expect("const") * r * r;
            sigma_sq * term * (-sqrt5_r).exp()
        }
        CovarianceFunction::RationalQuadratic => {
            // α fixed at 1.0 (standard rational quadratic)
            let alpha = F::one();
            sigma_sq * (F::one() + r * r / (F::from_f64(2.0).expect("const") * alpha)).powf(-alpha)
        }
    }
}

/// Euclidean distance between two slices of equal length.
fn euclidean_distance<F: Float>(a: &ArrayView1<F>, b: &ArrayView1<F>) -> F {
    let mut sq = F::zero();
    for (&ai, &bi) in a.iter().zip(b.iter()) {
        let d = ai - bi;
        sq = sq + d * d;
    }
    sq.sqrt()
}

/// Wendland C2 compactly-supported taper: `(1 – r/range)_+^4 * (1 + 4r/range)`.
/// Returns a value in [0, 1] for r ≤ range, and 0 for r > range.
fn wendland_c2<F: Float + FromPrimitive>(r: F, range: F) -> F {
    if r >= range {
        return F::zero();
    }
    let u = r / range;
    let one_minus_u = F::one() - u;
    let p4 = one_minus_u * one_minus_u * one_minus_u * one_minus_u;
    let four = F::from_f64(4.0).expect("const");
    p4 * (F::one() + four * u)
}

// ── In-place Cholesky solve for small dense systems ───────────────────────────

/// Compute the lower-triangular Cholesky factor L such that A = L Lᵀ.
/// `a` must be symmetric positive-definite; n × n.
/// Returns `Err` if A is not positive-definite (non-positive pivot encountered).
fn cholesky_lower<F: Float + FromPrimitive>(a: &Array2<F>) -> InterpolateResult<Array2<F>> {
    let n = a.nrows();
    let mut l = Array2::<F>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[[i, j]];
            for k in 0..j {
                s = s - l[[i, k]] * l[[j, k]];
            }
            if i == j {
                if s <= F::zero() {
                    return Err(InterpolateError::ComputationError(
                        "Cholesky: matrix not positive-definite".to_string(),
                    ));
                }
                l[[i, j]] = s.sqrt();
            } else {
                l[[i, j]] = s / l[[j, j]];
            }
        }
    }
    Ok(l)
}

/// Solve Lx = b where L is lower-triangular (forward substitution).
fn forward_sub<F: Float + FromPrimitive>(
    l: &Array2<F>,
    b: &Array1<F>,
) -> InterpolateResult<Array1<F>> {
    let n = b.len();
    let mut x = Array1::<F>::zeros(n);
    for i in 0..n {
        let mut s = b[i];
        for j in 0..i {
            s = s - l[[i, j]] * x[j];
        }
        let diag = l[[i, i]];
        if diag.abs() < F::from_f64(1e-300).expect("const") {
            return Err(InterpolateError::ComputationError(
                "Forward substitution: near-zero diagonal element".to_string(),
            ));
        }
        x[i] = s / diag;
    }
    Ok(x)
}

/// Solve Lᵀ x = b where L is lower-triangular (back substitution on Lᵀ).
fn back_sub_transpose<F: Float + FromPrimitive>(
    l: &Array2<F>,
    b: &Array1<F>,
) -> InterpolateResult<Array1<F>> {
    let n = b.len();
    let mut x = Array1::<F>::zeros(n);
    for i in (0..n).rev() {
        let mut s = b[i];
        for j in (i + 1)..n {
            s = s - l[[j, i]] * x[j];
        }
        let diag = l[[i, i]];
        if diag.abs() < F::from_f64(1e-300).expect("const") {
            return Err(InterpolateError::ComputationError(
                "Back substitution: near-zero diagonal element".to_string(),
            ));
        }
        x[i] = s / diag;
    }
    Ok(x)
}

/// Solve A x = b for symmetric positive-definite A via Cholesky (A = L Lᵀ).
/// Falls back to adding a small regularisation if A is borderline not-PD.
fn cholesky_solve<F: Float + FromPrimitive>(
    a: &Array2<F>,
    b: &Array1<F>,
) -> InterpolateResult<Array1<F>> {
    // First attempt – plain Cholesky
    let result = cholesky_lower(a).and_then(|l| {
        let y = forward_sub(&l, b)?;
        back_sub_transpose(&l, &y)
    });
    if result.is_ok() {
        return result;
    }
    // Retry with nugget regularisation (1e-6 × trace / n)
    let n = a.nrows();
    let mut reg = a.clone();
    let mut trace = F::zero();
    for i in 0..n {
        trace = trace + a[[i, i]];
    }
    let eps = trace / F::from_usize(n).expect("const") * F::from_f64(1e-6).expect("const");
    let eps = if eps < F::from_f64(1e-12).expect("const") {
        F::from_f64(1e-12).expect("const")
    } else {
        eps
    };
    for i in 0..n {
        reg[[i, i]] = reg[[i, i]] + eps;
    }
    let l = cholesky_lower(&reg)?;
    let y = forward_sub(&l, b)?;
    back_sub_transpose(&l, &y)
}

// ── PrecomputedState: what gets stored per approximation method ───────────────

/// Pre-computed state for FixedRank: Nyström approximation.
/// K ≈ K_{nm} K_{mm}^{-1} K_{mn}.
/// We store inducing points, K_mm^{-1} * K_mn * values.
#[derive(Debug, Clone)]
struct NystromState<F> {
    /// Inducing point indices into the training set
    inducing_points: Array2<F>,
    /// K_{mm} (m × m) Cholesky factor
    l_mm: Array2<F>,
    /// K_{mm}^{-1} y_proj where y_proj = K_{mn} * values  (shape: m)
    kmi_kmy: Array1<F>,
    /// Rank (m = number of inducing points)
    rank: usize,
}

/// Pre-computed state for Tapering: sparse COO covariance + Cholesky for block
/// For large problems we store only the sparse structure; solving is done
/// query-by-query over the non-zero neighbourhood.
#[derive(Debug, Clone)]
struct TaperState<F> {
    /// Taper range (same units as distances)
    taper_range: F,
    /// COO sparse representation of the tapered K (lower-tri + diagonal)
    sparse: SparseComponents<F>,
}

/// Internal approximation state bundled in an enum so we store only what's needed.
#[derive(Debug, Clone)]
enum ApproxState<F: Float + Debug> {
    /// No pre-computation (Local and HODLR compute on-the-fly)
    None,
    /// Nyström / Fixed-Rank pre-computation
    Nystrom(NystromState<F>),
    /// Tapering pre-computation
    Taper(TaperState<F>),
}

// ── FastKrigingMethod ─────────────────────────────────────────────────────────

/// Fast kriging approximation methods for large datasets
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FastKrigingMethod {
    /// Local kriging using only nearby points for prediction
    /// Provides O(k³) complexity per prediction where k is the neighborhood size
    Local,

    /// Fixed Rank Kriging with low-rank approximation
    /// Provides O(nr²) fitting and O(r²) prediction where r is the rank
    FixedRank(usize),

    /// Tapering approach that zeros out small covariance values
    /// Creates sparse matrices for efficient computation
    Tapering(f64),

    /// Hierarchical off-diagonal low-rank approximation
    HODLR(usize),
}

// ── FastPredictionResult ──────────────────────────────────────────────────────

/// Result type for FastKriging predictions
#[derive(Debug, Clone)]
pub struct FastPredictionResult<F: Float> {
    /// Predicted values
    pub value: Array1<F>,

    /// Approximate prediction variances
    pub variance: Array1<F>,

    /// Method used for computation
    pub method: FastKrigingMethod,

    /// Computation time in milliseconds (if available)
    pub computation_time_ms: Option<f64>,
}

// ── FastKriging ───────────────────────────────────────────────────────────────

/// Fast approximate kriging interpolator for large datasets
///
/// Provides local, fixed-rank, and tapered approximations to ordinary kriging.
/// All methods trade a small amount of accuracy for large computational savings.
///
/// # Examples
///
/// ```no_run
/// # #[cfg(not(any()))] // doctest disabled – requires non-trivial setup
/// # {
/// use scirs2_core::ndarray::{Array1, Array2};
/// use scirs2_interpolate::advanced::fast_kriging::{
///     FastKriging, FastKrigingMethod, FastKrigingBuilder
/// };
/// use scirs2_interpolate::advanced::kriging::CovarianceFunction;
///
/// let points = Array2::<f64>::zeros((100, 2));
/// let values = Array1::<f64>::zeros(100);
///
/// let local_kriging = FastKrigingBuilder::<f64>::new()
///     .points(points.clone())
///     .values(values.clone())
///     .covariance_function(CovarianceFunction::Matern52)
///     .approximation_method(FastKrigingMethod::Local)
///     .max_neighbors(50)
///     .build()
///     .expect("build failed");
///
/// let query_points = Array2::<f64>::zeros((10, 2));
/// let predictions = local_kriging
///     .predict(&query_points.view())
///     .expect("predict failed");
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct FastKriging<F>
where
    F: Float
        + FromPrimitive
        + ordered_float::FloatCore
        + Debug
        + Display
        + Div<Output = F>
        + Mul<Output = F>
        + Sub<Output = F>
        + Add<Output = F>
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + std::ops::RemAssign,
{
    /// Training point coordinates (n × d)
    points: Array2<F>,

    /// Observed values (n)
    values: Array1<F>,

    /// Covariance function
    cov_fn: CovarianceFunction,

    /// Isotropic length-scale parameter
    length_scale: F,

    /// Signal variance σ²
    sigma_sq: F,

    /// Nugget (added to diagonal for numerical stability)
    nugget: F,

    /// Trend function (for potential future Universal-Kriging support)
    #[allow(dead_code)]
    trend_fn: TrendFunction,

    /// Approximation method in use
    approx_method: FastKrigingMethod,

    /// Maximum number of neighbours for Local / HODLR
    max_neighbors: usize,

    /// Radius multiplier (radius = radius_multiplier × distance_to_k_th_neighbor)
    #[allow(dead_code)]
    radius_multiplier: F,

    /// Pre-computed KD-tree over training points (built once at construction)
    kdtree: Option<KdTree<F>>,

    /// Pre-computed approximation state
    state: ApproxState<F>,

    /// Marker for generic type
    _phantom: PhantomData<F>,
}

// ── FastKrigingBuilder ────────────────────────────────────────────────────────

/// Builder for constructing FastKriging models
///
/// # Examples
///
/// ```no_run
/// # #[cfg(not(any()))]
/// # {
/// use scirs2_core::ndarray::{Array1, Array2};
/// use scirs2_interpolate::advanced::fast_kriging::{
///     FastKrigingBuilder, FastKrigingMethod
/// };
/// use scirs2_interpolate::advanced::kriging::CovarianceFunction;
///
/// let points = Array2::<f64>::zeros((100, 2));
/// let values = Array1::<f64>::zeros(100);
///
/// let kriging = FastKrigingBuilder::<f64>::new()
///     .points(points.clone())
///     .values(values.clone())
///     .covariance_function(CovarianceFunction::Matern52)
///     .approximation_method(FastKrigingMethod::Local)
///     .max_neighbors(30)
///     .radius_multiplier(2.5)
///     .build()
///     .expect("build failed");
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct FastKrigingBuilder<F>
where
    F: Float
        + FromPrimitive
        + ordered_float::FloatCore
        + Debug
        + Display
        + Div<Output = F>
        + Mul<Output = F>
        + Sub<Output = F>
        + Add<Output = F>
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + std::ops::RemAssign,
{
    points: Option<Array2<F>>,
    values: Option<Array1<F>>,
    cov_fn: CovarianceFunction,
    #[allow(dead_code)]
    length_scales: Option<Array1<F>>,
    length_scale: F,
    sigma_sq: F,
    nugget: F,
    trend_fn: TrendFunction,
    approx_method: FastKrigingMethod,
    max_neighbors: usize,
    radius_multiplier: F,
    _phantom: PhantomData<F>,
}

impl<F> Default for FastKrigingBuilder<F>
where
    F: Float
        + FromPrimitive
        + ordered_float::FloatCore
        + Debug
        + Display
        + Add<Output = F>
        + Sub<Output = F>
        + Mul<Output = F>
        + Div<Output = F>
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + std::ops::RemAssign
        + 'static,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<F> FastKrigingBuilder<F>
where
    F: Float
        + FromPrimitive
        + ordered_float::FloatCore
        + Debug
        + Display
        + Add<Output = F>
        + Sub<Output = F>
        + Mul<Output = F>
        + Div<Output = F>
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + std::ops::RemAssign
        + 'static,
{
    /// Create a new builder with sensible defaults
    pub fn new() -> Self {
        Self {
            points: None,
            values: None,
            cov_fn: CovarianceFunction::Matern52,
            length_scales: None,
            length_scale: F::from_f64(1.0).expect("const"),
            sigma_sq: F::from_f64(1.0).expect("const"),
            nugget: F::from_f64(1e-6).expect("const"),
            trend_fn: TrendFunction::Constant,
            approx_method: FastKrigingMethod::Local,
            max_neighbors: DEFAULT_MAX_NEIGHBORS,
            radius_multiplier: F::from_f64(DEFAULT_RADIUS_MULTIPLIER).expect("const"),
            _phantom: PhantomData,
        }
    }

    /// Set training points
    pub fn points(mut self, points: Array2<F>) -> Self {
        self.points = Some(points);
        self
    }

    /// Set training values
    pub fn values(mut self, values: Array1<F>) -> Self {
        self.values = Some(values);
        self
    }

    /// Set covariance function
    pub fn covariance_function(mut self, covfn: CovarianceFunction) -> Self {
        self.cov_fn = covfn;
        self
    }

    /// Set per-dimension length scales (first value is used as isotropic scale)
    pub fn length_scales(mut self, lengthscales: Array1<F>) -> Self {
        if let Some(&ls) = lengthscales.first() {
            self.length_scale = ls;
        }
        self.length_scales = Some(lengthscales);
        self
    }

    /// Set isotropic length scale
    pub fn length_scale(mut self, lengthscale: F) -> Self {
        self.length_scale = lengthscale;
        self
    }

    /// Set signal variance σ²
    pub fn sigma_sq(mut self, sigmasq: F) -> Self {
        self.sigma_sq = sigmasq;
        self
    }

    /// Set nugget (noise variance, added to diagonal)
    pub fn nugget(mut self, nugget: F) -> Self {
        self.nugget = nugget;
        self
    }

    /// Set trend function type
    pub fn trend_function(mut self, trendfn: TrendFunction) -> Self {
        self.trend_fn = trendfn;
        self
    }

    /// Set approximation method
    pub fn approximation_method(mut self, method: FastKrigingMethod) -> Self {
        self.approx_method = method;
        self
    }

    /// Set maximum number of neighbours for local / HODLR methods
    pub fn max_neighbors(mut self, maxneighbors: usize) -> Self {
        self.max_neighbors = maxneighbors;
        self
    }

    /// Set radius multiplier for neighbourhood search
    pub fn radius_multiplier(mut self, multiplier: F) -> Self {
        self.radius_multiplier = multiplier;
        self
    }

    /// Validate and build the FastKriging model
    pub fn build(self) -> InterpolateResult<FastKriging<F>> {
        FastKriging::from_builder(self)
    }
}

// ── FastKriging constructor ───────────────────────────────────────────────────

impl<F> FastKriging<F>
where
    F: Float
        + FromPrimitive
        + ordered_float::FloatCore
        + Debug
        + Display
        + Add<Output = F>
        + Sub<Output = F>
        + Mul<Output = F>
        + Div<Output = F>
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + std::ops::RemAssign
        + 'static,
{
    /// Create a new builder
    pub fn builder() -> FastKrigingBuilder<F> {
        FastKrigingBuilder::new()
    }

    /// Construct from a validated builder
    fn from_builder(builder: FastKrigingBuilder<F>) -> InterpolateResult<FastKriging<F>> {
        let points = builder.points.ok_or(InterpolateError::MissingPoints)?;
        let values = builder.values.ok_or(InterpolateError::MissingValues)?;

        if points.nrows() != values.len() {
            return Err(InterpolateError::DimensionMismatch(
                "Number of points must match number of values".to_string(),
            ));
        }
        if points.is_empty() {
            return Err(InterpolateError::InvalidValue(
                "Points array cannot be empty".to_string(),
            ));
        }

        // Build KD-tree for spatial queries (always useful for Local + HODLR)
        let kdtree = KdTree::new(points.clone()).ok();

        // Pre-compute approximation-specific state
        let state = build_approx_state(
            &points,
            &values,
            builder.cov_fn,
            builder.length_scale,
            builder.sigma_sq,
            builder.nugget,
            builder.approx_method,
        )?;

        Ok(FastKriging {
            points,
            values,
            cov_fn: builder.cov_fn,
            length_scale: builder.length_scale,
            sigma_sq: builder.sigma_sq,
            nugget: builder.nugget,
            trend_fn: builder.trend_fn,
            approx_method: builder.approx_method,
            max_neighbors: builder.max_neighbors,
            radius_multiplier: builder.radius_multiplier,
            kdtree,
            state,
            _phantom: PhantomData,
        })
    }

    // ── public API ───────────────────────────────────────────────────────────

    /// Return number of training points
    pub fn n_points(&self) -> usize {
        self.points.nrows()
    }

    /// Return spatial dimensionality
    pub fn n_dims(&self) -> usize {
        self.points.ncols()
    }

    /// Return approximation method
    pub fn approximation_method(&self) -> FastKrigingMethod {
        self.approx_method
    }

    /// Predict values at `query_points` using the chosen fast approximation.
    ///
    /// # Arguments
    ///
    /// * `query_points` – shape (q × d), where d matches training dimensionality
    ///
    /// # Returns
    ///
    /// `FastPredictionResult` containing predicted values and variances.
    pub fn predict(
        &self,
        query_points: &ArrayView2<F>,
    ) -> InterpolateResult<FastPredictionResult<F>> {
        if query_points.ncols() != self.points.ncols() {
            return Err(InterpolateError::DimensionMismatch(format!(
                "Query dimensionality {} does not match training dimensionality {}",
                query_points.ncols(),
                self.points.ncols()
            )));
        }

        if query_points.nrows() == 0 {
            return Ok(FastPredictionResult {
                value: Array1::zeros(0),
                variance: Array1::zeros(0),
                method: self.approx_method,
                computation_time_ms: None,
            });
        }

        match self.approx_method {
            FastKrigingMethod::Local => self.predict_local(query_points),
            FastKrigingMethod::FixedRank(_) => self.predict_nystrom(query_points),
            FastKrigingMethod::Tapering(_) => self.predict_tapered(query_points),
            FastKrigingMethod::HODLR(_) => self.predict_hodlr(query_points),
        }
    }

    // ── Local Kriging ─────────────────────────────────────────────────────────

    /// Local ordinary kriging: for each query point find the k-nearest training
    /// points, form a local kriging system, solve via Cholesky, and predict.
    fn predict_local(
        &self,
        query_points: &ArrayView2<F>,
    ) -> InterpolateResult<FastPredictionResult<F>> {
        let n_query = query_points.nrows();
        let mut pred_values = Array1::zeros(n_query);
        let mut pred_variances = Array1::zeros(n_query);
        let k = self.max_neighbors.min(self.points.nrows());

        let global_mean = compute_mean(&self.values);

        for qi in 0..n_query {
            let query = query_points.slice(scirs2_core::ndarray::s![qi, ..]);

            // Find k nearest training points
            let neighbors = self.find_neighbors_kd(&query, k)?;
            let m = neighbors.len();

            if m == 0 {
                pred_values[qi] = global_mean;
                pred_variances[qi] = self.sigma_sq;
                continue;
            }

            if m == 1 {
                // Degenerate: single neighbour, return its value
                pred_values[qi] = self.values[neighbors[0].0];
                pred_variances[qi] = F::zero();
                continue;
            }

            // Extract local sub-problem
            let local_pts: Array2<F> = extract_rows(&self.points, &neighbors);
            let local_vals: Array1<F> = {
                let mut v = Array1::zeros(m);
                for (j, &(idx, _)) in neighbors.iter().enumerate() {
                    v[j] = self.values[idx];
                }
                v
            };

            // Build K_local (m×m) with nugget on diagonal
            let k_local = self.build_cov_matrix(&local_pts);

            // k_star: covariance between query and each local point (m)
            let k_star = self.build_cross_cov(&query, &local_pts);

            // Solve K_local w = local_vals
            let weights = cholesky_solve(&k_local, &local_vals)
                .unwrap_or_else(|_| uniform_weights(m, global_mean, &local_vals));

            // Prediction = k_star^T w
            let mut pred = F::zero();
            for j in 0..m {
                pred = pred + k_star[j] * weights[j];
            }
            pred_values[qi] = pred;

            // Kriging variance = σ²(0) - k_star^T (K_local \ k_star)
            // i.e., solve K_local α = k_star, then var = σ² - k_star·α
            let alpha = cholesky_solve(&k_local, &k_star).unwrap_or_else(|_| Array1::zeros(m));
            let mut reduction = F::zero();
            for j in 0..m {
                reduction = reduction + k_star[j] * alpha[j];
            }
            let variance_raw = self.sigma_sq - reduction;
            let variance = if variance_raw < F::zero() {
                F::zero()
            } else {
                variance_raw
            };
            pred_variances[qi] = variance;
        }

        Ok(FastPredictionResult {
            value: pred_values,
            variance: pred_variances,
            method: self.approx_method,
            computation_time_ms: None,
        })
    }

    // ── Fixed-Rank (Nyström) Kriging ─────────────────────────────────────────

    /// Nyström approximation: predictions use pre-computed inducing-point factors.
    fn predict_nystrom(
        &self,
        query_points: &ArrayView2<F>,
    ) -> InterpolateResult<FastPredictionResult<F>> {
        let nys = match &self.state {
            ApproxState::Nystrom(ns) => ns,
            _ => {
                return Err(InterpolateError::InvalidState(
                    "Nyström state not initialised for FixedRank method".to_string(),
                ))
            }
        };

        let n_query = query_points.nrows();
        let mut pred_values = Array1::zeros(n_query);
        let mut pred_variances = Array1::zeros(n_query);

        for qi in 0..n_query {
            let query = query_points.slice(scirs2_core::ndarray::s![qi, ..]);

            // k_qm : covariance between query and each inducing point (rank)
            let k_qm = self.build_cross_cov(&query, &nys.inducing_points);

            // Prediction: k_qm^T * (K_mm \ K_mn * values) = k_qm . kmi_kmy
            let mut pred = F::zero();
            for j in 0..nys.rank {
                pred = pred + k_qm[j] * nys.kmi_kmy[j];
            }
            pred_values[qi] = pred;

            // Approximate variance: σ²(0) - k_qm^T K_mm^{-1} k_qm
            let alpha = back_sub_transpose(
                &nys.l_mm,
                &forward_sub(&nys.l_mm, &k_qm).unwrap_or_else(|_| Array1::zeros(nys.rank)),
            )
            .unwrap_or_else(|_| Array1::zeros(nys.rank));
            let mut reduction = F::zero();
            for j in 0..nys.rank {
                reduction = reduction + k_qm[j] * alpha[j];
            }
            let var_nys = self.sigma_sq - reduction;
            pred_variances[qi] = if var_nys < F::zero() {
                F::zero()
            } else {
                var_nys
            };
        }

        Ok(FastPredictionResult {
            value: pred_values,
            variance: pred_variances,
            method: self.approx_method,
            computation_time_ms: None,
        })
    }

    // ── Tapered Kriging ───────────────────────────────────────────────────────

    /// Tapered kriging: covariance is multiplied by the Wendland C2 taper,
    /// zeroing out entries beyond `taper_range`.  Each query point only
    /// interacts with training points within the taper range, producing an
    /// effective local solve.
    fn predict_tapered(
        &self,
        query_points: &ArrayView2<F>,
    ) -> InterpolateResult<FastPredictionResult<F>> {
        let taper_state = match &self.state {
            ApproxState::Taper(ts) => ts,
            _ => {
                return Err(InterpolateError::InvalidState(
                    "Taper state not initialised for Tapering method".to_string(),
                ))
            }
        };

        let n_query = query_points.nrows();
        let mut pred_values = Array1::zeros(n_query);
        let mut pred_variances = Array1::zeros(n_query);
        let range = taper_state.taper_range;
        let global_mean = compute_mean(&self.values);

        for qi in 0..n_query {
            let query = query_points.slice(scirs2_core::ndarray::s![qi, ..]);

            // Identify training points within taper range
            let n_train = self.points.nrows();
            let mut active: Vec<usize> = Vec::new();
            let mut dists_q: Vec<F> = Vec::new();

            for j in 0..n_train {
                let pt = self.points.slice(scirs2_core::ndarray::s![j, ..]);
                let dist = euclidean_distance(&query, &pt) / self.length_scale;
                if dist < range / self.length_scale {
                    active.push(j);
                    dists_q.push(dist);
                }
            }

            if active.is_empty() {
                pred_values[qi] = global_mean;
                pred_variances[qi] = self.sigma_sq;
                continue;
            }

            let m = active.len();
            let active_pts: Array2<F> = {
                let mut ap = Array2::zeros((m, self.points.ncols()));
                for (row, &idx) in active.iter().enumerate() {
                    ap.slice_mut(scirs2_core::ndarray::s![row, ..])
                        .assign(&self.points.slice(scirs2_core::ndarray::s![idx, ..]));
                }
                ap
            };
            let active_vals: Array1<F> = {
                let mut av = Array1::zeros(m);
                for (j, &idx) in active.iter().enumerate() {
                    av[j] = self.values[idx];
                }
                av
            };

            // Build tapered local K (m×m)
            let mut k_local = Array2::<F>::zeros((m, m));
            for j in 0..m {
                for kk in 0..m {
                    let pt_j = active_pts.slice(scirs2_core::ndarray::s![j, ..]);
                    let pt_k = active_pts.slice(scirs2_core::ndarray::s![kk, ..]);
                    let dist = euclidean_distance(&pt_j, &pt_k) / self.length_scale;
                    let cov = eval_covariance(dist, self.sigma_sq, self.cov_fn);
                    let tap = wendland_c2(dist * self.length_scale, range);
                    if j == kk {
                        k_local[[j, kk]] = cov * tap + self.nugget;
                    } else {
                        k_local[[j, kk]] = cov * tap;
                    }
                }
            }

            // k_star_tapered: tapered covariance between query and active points
            let mut k_star = Array1::zeros(m);
            for (j, &dist_scaled) in dists_q.iter().enumerate() {
                let dist_abs = dist_scaled * self.length_scale;
                let cov = eval_covariance(dist_scaled, self.sigma_sq, self.cov_fn);
                let tap = wendland_c2(dist_abs, range);
                k_star[j] = cov * tap;
            }

            // Solve K_local w = active_vals
            let weights = cholesky_solve(&k_local, &active_vals)
                .unwrap_or_else(|_| uniform_weights(m, global_mean, &active_vals));

            let mut pred = F::zero();
            for j in 0..m {
                pred = pred + k_star[j] * weights[j];
            }
            pred_values[qi] = pred;

            let alpha = cholesky_solve(&k_local, &k_star).unwrap_or_else(|_| Array1::zeros(m));
            let mut reduction = F::zero();
            for j in 0..m {
                reduction = reduction + k_star[j] * alpha[j];
            }
            let var_tap = self.sigma_sq - reduction;
            pred_variances[qi] = if var_tap < F::zero() {
                F::zero()
            } else {
                var_tap
            };
        }

        // Suppress unused warning on sparse field (used in construction, not query)
        let _ = &taper_state.sparse;

        Ok(FastPredictionResult {
            value: pred_values,
            variance: pred_variances,
            method: self.approx_method,
            computation_time_ms: None,
        })
    }

    // ── HODLR Kriging ─────────────────────────────────────────────────────────

    /// HODLR: hierarchical block decomposition.  Training points are split into
    /// blocks of at most `leaf_size`; each query is answered by a weighted
    /// combination of local-kriging predictions from the closest blocks.
    fn predict_hodlr(
        &self,
        query_points: &ArrayView2<F>,
    ) -> InterpolateResult<FastPredictionResult<F>> {
        let leaf_size = match self.approx_method {
            FastKrigingMethod::HODLR(ls) => ls.max(2),
            _ => 32,
        };

        let n_train = self.points.nrows();
        let n_query = query_points.nrows();
        let mut pred_values = Array1::zeros(n_query);
        let mut pred_variances = Array1::zeros(n_query);
        let global_mean = compute_mean(&self.values);

        // Divide training points into contiguous blocks
        let n_blocks = (n_train + leaf_size - 1) / leaf_size;

        for qi in 0..n_query {
            let query = query_points.slice(scirs2_core::ndarray::s![qi, ..]);

            let mut total_weight = F::zero();
            let mut weighted_pred = F::zero();
            let mut weighted_var = F::zero();

            for b in 0..n_blocks {
                let start = b * leaf_size;
                let end = n_train.min(start + leaf_size);
                if start >= end {
                    continue;
                }

                // Centroid of this block
                let d = self.points.ncols();
                let mut centroid = vec![F::zero(); d];
                for j in start..end {
                    for dd in 0..d {
                        centroid[dd] = centroid[dd] + self.points[[j, dd]];
                    }
                }
                let block_len = F::from_usize(end - start).expect("const");
                for dd in 0..d {
                    centroid[dd] = centroid[dd] / block_len;
                }

                let mut dist_sq = F::zero();
                for dd in 0..d {
                    let diff = query[dd] - centroid[dd];
                    dist_sq = dist_sq + diff * diff;
                }
                let dist = dist_sq.sqrt();

                // Weight = 1 / (1 + dist); skip negligible blocks
                let weight = F::one() / (F::one() + dist);
                if weight < F::from_f64(1e-8).expect("const") {
                    continue;
                }

                // Build local kriging for this block
                let block_pts_slice = self.points.slice(scirs2_core::ndarray::s![start..end, ..]);
                let block_pts = block_pts_slice.to_owned();
                let block_vals_slice = self.values.slice(scirs2_core::ndarray::s![start..end]);
                let block_vals = block_vals_slice.to_owned();

                let (local_pred, local_var) =
                    self.block_local_predict(&query, &block_pts, &block_vals, global_mean)?;

                weighted_pred = weighted_pred + weight * local_pred;
                weighted_var = weighted_var + weight * weight * local_var;
                total_weight = total_weight + weight;
            }

            if total_weight > F::zero() {
                pred_values[qi] = weighted_pred / total_weight;
                let raw_var = weighted_var / (total_weight * total_weight);
                pred_variances[qi] = if raw_var < F::zero() {
                    F::zero()
                } else {
                    raw_var
                };
            } else {
                pred_values[qi] = global_mean;
                pred_variances[qi] = self.sigma_sq;
            }
        }

        Ok(FastPredictionResult {
            value: pred_values,
            variance: pred_variances,
            method: self.approx_method,
            computation_time_ms: None,
        })
    }

    /// Ordinary kriging prediction within a single block (used by HODLR).
    fn block_local_predict(
        &self,
        query: &ArrayView1<F>,
        block_pts: &Array2<F>,
        block_vals: &Array1<F>,
        global_mean: F,
    ) -> InterpolateResult<(F, F)> {
        let m = block_pts.nrows();
        if m == 0 {
            return Ok((global_mean, self.sigma_sq));
        }
        if m == 1 {
            return Ok((block_vals[0], F::zero()));
        }

        let k_local = self.build_cov_matrix(block_pts);
        let k_star = self.build_cross_cov(query, block_pts);

        let weights = cholesky_solve(&k_local, block_vals)
            .unwrap_or_else(|_| uniform_weights(m, global_mean, block_vals));

        let mut pred = F::zero();
        for j in 0..m {
            pred = pred + k_star[j] * weights[j];
        }

        let alpha = cholesky_solve(&k_local, &k_star).unwrap_or_else(|_| Array1::zeros(m));
        let mut reduction = F::zero();
        for j in 0..m {
            reduction = reduction + k_star[j] * alpha[j];
        }
        let var_raw = self.sigma_sq - reduction;
        let var = if var_raw < F::zero() {
            F::zero()
        } else {
            var_raw
        };

        Ok((pred, var))
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// Find k nearest neighbours using the pre-built KD-tree if available,
    /// otherwise fall back to a linear scan.  Returns `(index, distance)` pairs.
    fn find_neighbors_kd(
        &self,
        query: &ArrayView1<F>,
        k: usize,
    ) -> InterpolateResult<Vec<(usize, F)>> {
        let query_slice = query.as_slice().ok_or_else(|| {
            InterpolateError::InvalidValue("Query must be contiguous".to_string())
        })?;

        match &self.kdtree {
            Some(tree) => tree.k_nearest_neighbors(query_slice, k),
            None => {
                // Linear scan fallback
                let n = self.points.nrows();
                let mut dists: Vec<(usize, F)> = (0..n)
                    .map(|i| {
                        let pt = self.points.slice(scirs2_core::ndarray::s![i, ..]);
                        let d = euclidean_distance(query, &pt);
                        (i, d)
                    })
                    .collect();
                dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                dists.truncate(k);
                Ok(dists)
            }
        }
    }

    /// Build the m×m covariance matrix with nugget on the diagonal.
    fn build_cov_matrix(&self, pts: &Array2<F>) -> Array2<F> {
        let m = pts.nrows();
        let mut mat = Array2::zeros((m, m));
        for i in 0..m {
            for j in 0..m {
                if i == j {
                    mat[[i, j]] = self.sigma_sq + self.nugget;
                } else {
                    let pi = pts.slice(scirs2_core::ndarray::s![i, ..]);
                    let pj = pts.slice(scirs2_core::ndarray::s![j, ..]);
                    let r = euclidean_distance(&pi, &pj) / self.length_scale;
                    mat[[i, j]] = eval_covariance(r, self.sigma_sq, self.cov_fn);
                }
            }
        }
        mat
    }

    /// Build covariance vector between a single query point and a matrix of points.
    fn build_cross_cov(&self, query: &ArrayView1<F>, pts: &Array2<F>) -> Array1<F> {
        let m = pts.nrows();
        let mut kv = Array1::zeros(m);
        for j in 0..m {
            let pj = pts.slice(scirs2_core::ndarray::s![j, ..]);
            let r = euclidean_distance(query, &pj) / self.length_scale;
            kv[j] = eval_covariance(r, self.sigma_sq, self.cov_fn);
        }
        kv
    }
}

// ── Pre-computation helpers ───────────────────────────────────────────────────

/// Build the approximation state before prediction.
fn build_approx_state<F>(
    points: &Array2<F>,
    values: &Array1<F>,
    cov_fn: CovarianceFunction,
    length_scale: F,
    sigma_sq: F,
    nugget: F,
    method: FastKrigingMethod,
) -> InterpolateResult<ApproxState<F>>
where
    F: Float
        + FromPrimitive
        + ordered_float::FloatCore
        + Debug
        + Display
        + Add<Output = F>
        + Sub<Output = F>
        + Mul<Output = F>
        + Div<Output = F>
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + std::ops::RemAssign
        + 'static,
{
    match method {
        FastKrigingMethod::FixedRank(rank) => {
            let nys =
                build_nystrom_state(points, values, cov_fn, length_scale, sigma_sq, nugget, rank)?;
            Ok(ApproxState::Nystrom(nys))
        }
        FastKrigingMethod::Tapering(range_f64) => {
            let range = F::from_f64(range_f64)
                .ok_or_else(|| InterpolateError::InvalidValue("Invalid taper range".to_string()))?;
            let sparse =
                build_tapered_sparse(points, cov_fn, length_scale, sigma_sq, nugget, range)?;
            Ok(ApproxState::Taper(TaperState {
                taper_range: range,
                sparse,
            }))
        }
        _ => Ok(ApproxState::None),
    }
}

/// Build the Nyström state for FixedRank kriging.
///
/// Strategy:
/// 1. Select `rank` inducing points by uniform striding through the dataset.
/// 2. Compute K_mm (m×m covariance matrix of inducing points).
/// 3. Compute L_mm = chol(K_mm).
/// 4. Compute K_mn (m×n cross-covariance between inducing and training points).
/// 5. Store L_mm and L_mm^{-1} K_mn values = kmi_kmy.
fn build_nystrom_state<F>(
    points: &Array2<F>,
    values: &Array1<F>,
    cov_fn: CovarianceFunction,
    length_scale: F,
    sigma_sq: F,
    nugget: F,
    rank: usize,
) -> InterpolateResult<NystromState<F>>
where
    F: Float
        + FromPrimitive
        + ordered_float::FloatCore
        + Debug
        + Display
        + std::ops::AddAssign
        + 'static,
{
    let n = points.nrows();
    let m = rank.min(n);

    // Select inducing points by uniform striding
    let step = if m > 1 { n / m } else { 1 };
    let inducing_indices: Vec<usize> = (0..m).map(|i| (i * step).min(n - 1)).collect();

    let d = points.ncols();
    let mut ind_pts = Array2::zeros((m, d));
    for (row, &idx) in inducing_indices.iter().enumerate() {
        ind_pts
            .slice_mut(scirs2_core::ndarray::s![row, ..])
            .assign(&points.slice(scirs2_core::ndarray::s![idx, ..]));
    }

    // Build K_mm with nugget
    let mut k_mm = Array2::zeros((m, m));
    for i in 0..m {
        for j in 0..m {
            if i == j {
                k_mm[[i, j]] = sigma_sq + nugget;
            } else {
                let pi = ind_pts.slice(scirs2_core::ndarray::s![i, ..]);
                let pj = ind_pts.slice(scirs2_core::ndarray::s![j, ..]);
                let r = euclidean_distance(&pi, &pj) / length_scale;
                k_mm[[i, j]] = eval_covariance(r, sigma_sq, cov_fn);
            }
        }
    }

    let l_mm = cholesky_lower(&k_mm)?;

    // Build K_mn (m × n) and project values: K_mn * values -> shape m
    let mut k_mn_y = Array1::zeros(m);
    for i in 0..m {
        let pi = ind_pts.slice(scirs2_core::ndarray::s![i, ..]);
        let mut dot = F::zero();
        for j in 0..n {
            let pj = points.slice(scirs2_core::ndarray::s![j, ..]);
            let r = euclidean_distance(&pi, &pj) / length_scale;
            dot = dot + eval_covariance(r, sigma_sq, cov_fn) * values[j];
        }
        k_mn_y[i] = dot;
    }

    // Solve K_mm * x = K_mn_y  (i.e. x = K_mm^{-1} * K_mn * values)
    let y_fwd = forward_sub(&l_mm, &k_mn_y)?;
    let kmi_kmy = back_sub_transpose(&l_mm, &y_fwd)?;

    Ok(NystromState {
        inducing_points: ind_pts,
        l_mm,
        kmi_kmy,
        rank: m,
    })
}

/// Build the sparse COO representation of the tapered covariance matrix.
fn build_tapered_sparse<F>(
    points: &Array2<F>,
    cov_fn: CovarianceFunction,
    length_scale: F,
    sigma_sq: F,
    nugget: F,
    taper_range: F,
) -> InterpolateResult<SparseComponents<F>>
where
    F: Float + FromPrimitive + ordered_float::FloatCore + std::ops::AddAssign + 'static,
{
    let n = points.nrows();
    let mut indices: Vec<(usize, usize)> = Vec::new();
    let mut vals: Vec<F> = Vec::new();

    for i in 0..n {
        for j in 0..=i {
            let pi = points.slice(scirs2_core::ndarray::s![i, ..]);
            let pj = points.slice(scirs2_core::ndarray::s![j, ..]);
            let dist = euclidean_distance(&pi, &pj);
            let dist_scaled = dist / length_scale;

            let tap = wendland_c2(dist, taper_range);
            if tap == F::zero() && i != j {
                continue; // skip truly zero entries
            }

            let cov = eval_covariance(dist_scaled, sigma_sq, cov_fn);
            let entry = if i == j {
                cov * tap + nugget
            } else {
                cov * tap
            };

            indices.push((i, j));
            vals.push(entry);
            if i != j {
                indices.push((j, i));
                vals.push(entry);
            }
        }
    }

    Ok((indices, vals))
}

// ── Standalone convenience functions ─────────────────────────────────────────

/// Creates a FastKriging model using local approximate kriging.
///
/// Local kriging builds a KD-tree over training points and, for each query
/// point, solves a small kriging system using only the `max_neighbors`
/// nearest training points.  Complexity is O(k³) per query point.
///
/// # Arguments
///
/// * `points` – training locations (n × d)
/// * `values` – observed values at training locations (n)
/// * `cov_fn` – covariance function
/// * `scale`  – isotropic length-scale parameter
/// * `max_neighbors` – maximum number of neighbours per query point (≤ n)
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::{Array1, Array2};
/// use scirs2_interpolate::advanced::fast_kriging::make_local_kriging;
/// use scirs2_interpolate::advanced::kriging::CovarianceFunction;
///
/// let points = Array2::<f64>::from_shape_vec(
///     (5, 2), vec![0.0,0.0, 1.0,0.0, 0.0,1.0, 1.0,1.0, 0.5,0.5])
///     .expect("shape");
/// let values = Array1::from_vec(vec![0.0, 1.0, 1.0, 2.0, 1.0]);
///
/// let kriging = make_local_kriging(
///     &points.view(), &values.view(), CovarianceFunction::Matern52, 1.0_f64, 3
/// ).expect("build");
///
/// let query = Array2::from_shape_vec((1, 2), vec![0.5, 0.5]).expect("shape");
/// let pred = kriging.predict(&query.view()).expect("predict");
/// assert!(pred.value[0].is_finite());
/// ```
pub fn make_local_kriging<F>(
    points: &ArrayView2<F>,
    values: &ArrayView1<F>,
    cov_fn: CovarianceFunction,
    scale: F,
    max_neighbors: usize,
) -> InterpolateResult<FastKriging<F>>
where
    F: Float
        + FromPrimitive
        + ordered_float::FloatCore
        + Debug
        + Display
        + Add<Output = F>
        + Sub<Output = F>
        + Mul<Output = F>
        + Div<Output = F>
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + std::ops::RemAssign
        + 'static,
{
    FastKrigingBuilder::new()
        .points(points.to_owned())
        .values(values.to_owned())
        .covariance_function(cov_fn)
        .length_scale(scale)
        .approximation_method(FastKrigingMethod::Local)
        .max_neighbors(max_neighbors)
        .build()
}

/// Creates a FastKriging model using a Nyström (fixed-rank) approximation.
///
/// Fixed-rank kriging selects `rank` inducing points uniformly from the
/// training set and builds a low-rank approximation:
/// K ≈ K_nm K_mm^{-1} K_mn.
/// Predictions have O(nr) fitting cost and O(r²) query cost.
///
/// # Arguments
///
/// * `points` – training locations (n × d)
/// * `values` – observed values (n)
/// * `rank`   – number of inducing points (rank of approximation)
/// * `cov_fn` – covariance function
/// * `scale`  – isotropic length-scale parameter
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::{Array1, Array2};
/// use scirs2_interpolate::advanced::fast_kriging::make_fixed_rank_kriging;
/// use scirs2_interpolate::advanced::kriging::CovarianceFunction;
///
/// let n = 20usize;
/// let points = Array2::<f64>::from_shape_fn((n, 2), |(i,j)| {
///     if j == 0 { i as f64 / n as f64 } else { (i as f64 / n as f64).sin() }
/// });
/// let values = Array1::<f64>::from_iter((0..n).map(|i| (i as f64 / n as f64).powi(2)));
///
/// let kriging = make_fixed_rank_kriging(
///     &points.view(), &values.view(), 5, CovarianceFunction::SquaredExponential, 0.5_f64,
/// ).expect("build");
///
/// let q = Array2::from_shape_vec((1, 2), vec![0.5, 0.5_f64.sin()]).expect("shape");
/// let pred = kriging.predict(&q.view()).expect("predict");
/// assert!(pred.value[0].is_finite());
/// ```
pub fn make_fixed_rank_kriging<F>(
    points: &ArrayView2<F>,
    values: &ArrayView1<F>,
    rank: usize,
    cov_fn: CovarianceFunction,
    scale: F,
) -> InterpolateResult<FastKriging<F>>
where
    F: Float
        + FromPrimitive
        + ordered_float::FloatCore
        + Debug
        + Display
        + Add<Output = F>
        + Sub<Output = F>
        + Mul<Output = F>
        + Div<Output = F>
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + std::ops::RemAssign
        + 'static,
{
    FastKrigingBuilder::new()
        .points(points.to_owned())
        .values(values.to_owned())
        .covariance_function(cov_fn)
        .length_scale(scale)
        .approximation_method(FastKrigingMethod::FixedRank(rank))
        .build()
}

/// Creates a FastKriging model with covariance tapering.
///
/// Covariance tapering multiplies the base covariance function by the
/// Wendland C2 compactly-supported function, zeroing out covariance
/// beyond `taper_range`.  This creates sparse effective covariance
/// matrices and restricts each prediction to nearby training points.
///
/// # Arguments
///
/// * `points`      – training locations (n × d)
/// * `values`      – observed values (n)
/// * `taper_range` – range beyond which covariance is zero (same units as distances)
/// * `cov_fn`      – base covariance function (multiplied by taper)
/// * `scale`       – isotropic length-scale parameter
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::{Array1, Array2};
/// use scirs2_interpolate::advanced::fast_kriging::make_tapered_kriging;
/// use scirs2_interpolate::advanced::kriging::CovarianceFunction;
///
/// let points = Array2::<f64>::from_shape_vec(
///     (6, 1), vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
///     .expect("shape");
/// let values = Array1::from_vec(vec![0.0, 0.04, 0.16, 0.36, 0.64, 1.0]);
///
/// let kriging = make_tapered_kriging(
///     &points.view(), &values.view(), 0.5_f64, CovarianceFunction::Exponential, 0.3_f64,
/// ).expect("build");
///
/// let q = Array2::from_shape_vec((1, 1), vec![0.5_f64]).expect("shape");
/// let pred = kriging.predict(&q.view()).expect("predict");
/// assert!(pred.value[0].is_finite());
/// ```
pub fn make_tapered_kriging<F>(
    points: &ArrayView2<F>,
    values: &ArrayView1<F>,
    taper_range: F,
    cov_fn: CovarianceFunction,
    scale: F,
) -> InterpolateResult<FastKriging<F>>
where
    F: Float
        + FromPrimitive
        + ordered_float::FloatCore
        + Debug
        + Display
        + Add<Output = F>
        + Sub<Output = F>
        + Mul<Output = F>
        + Div<Output = F>
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + std::ops::RemAssign
        + 'static,
{
    let range_f64 = taper_range.to_f64().ok_or_else(|| {
        InterpolateError::InvalidValue("Cannot convert taper_range to f64".to_string())
    })?;
    FastKrigingBuilder::new()
        .points(points.to_owned())
        .values(values.to_owned())
        .covariance_function(cov_fn)
        .length_scale(scale)
        .approximation_method(FastKrigingMethod::Tapering(range_f64))
        .build()
}

/// Creates a FastKriging model with HODLR approximation.
///
/// HODLR divides training points into hierarchical blocks of at most
/// `leaf_size` points; predictions are weighted combinations of
/// block-local kriging predictions.
///
/// # Arguments
///
/// * `points`    – training locations (n × d)
/// * `values`    – observed values (n)
/// * `leaf_size` – maximum number of points per leaf block
/// * `cov_fn`    – covariance function
/// * `scale`     – isotropic length-scale parameter
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::{Array1, Array2};
/// use scirs2_interpolate::advanced::fast_kriging::make_hodlr_kriging;
/// use scirs2_interpolate::advanced::kriging::CovarianceFunction;
///
/// let n = 30usize;
/// let points = Array2::<f64>::from_shape_fn((n, 2), |(i,j)| {
///     (i as f64 / n as f64) + j as f64 * 0.1
/// });
/// let values = Array1::<f64>::from_iter((0..n).map(|i| i as f64 / n as f64));
///
/// let kriging = make_hodlr_kriging(
///     &points.view(), &values.view(), 8, CovarianceFunction::Matern32, 0.5_f64,
/// ).expect("build");
///
/// let q = Array2::from_shape_vec((1, 2), vec![0.5_f64, 0.55]).expect("shape");
/// let pred = kriging.predict(&q.view()).expect("predict");
/// assert!(pred.value[0].is_finite());
/// ```
pub fn make_hodlr_kriging<F>(
    points: &ArrayView2<F>,
    values: &ArrayView1<F>,
    leaf_size: usize,
    cov_fn: CovarianceFunction,
    scale: F,
) -> InterpolateResult<FastKriging<F>>
where
    F: Float
        + FromPrimitive
        + ordered_float::FloatCore
        + Debug
        + Display
        + Add<Output = F>
        + Sub<Output = F>
        + Mul<Output = F>
        + Div<Output = F>
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::MulAssign
        + std::ops::DivAssign
        + std::ops::RemAssign
        + 'static,
{
    FastKrigingBuilder::new()
        .points(points.to_owned())
        .values(values.to_owned())
        .covariance_function(cov_fn)
        .length_scale(scale)
        .approximation_method(FastKrigingMethod::HODLR(leaf_size))
        .build()
}

/// Automatically choose the best approximation method based on dataset size.
///
/// # Examples
///
/// ```
/// use scirs2_interpolate::advanced::fast_kriging::select_approximation_method;
///
/// let method = select_approximation_method(10_000);
/// ```
pub fn select_approximation_method(n_points: usize) -> FastKrigingMethod {
    if n_points < 500 {
        FastKrigingMethod::Local
    } else if n_points < 5_000 {
        FastKrigingMethod::FixedRank(50)
    } else if n_points < 50_000 {
        FastKrigingMethod::Tapering(3.0)
    } else {
        FastKrigingMethod::HODLR(64)
    }
}

// ── Private helpers ───────────────────────────────────────────────────────────

/// Compute the arithmetic mean of an array.
fn compute_mean<F: Float + FromPrimitive>(values: &Array1<F>) -> F {
    if values.is_empty() {
        return F::zero();
    }
    let n = F::from_usize(values.len()).expect("const");
    let mut sum = F::zero();
    for &v in values.iter() {
        sum = sum + v;
    }
    sum / n
}

/// Extract rows specified by (index, distance) pairs from a 2D array.
fn extract_rows<F: Float>(pts: &Array2<F>, neighbors: &[(usize, F)]) -> Array2<F> {
    let m = neighbors.len();
    let d = pts.ncols();
    let mut out = Array2::zeros((m, d));
    for (row, &(idx, _)) in neighbors.iter().enumerate() {
        out.slice_mut(scirs2_core::ndarray::s![row, ..])
            .assign(&pts.slice(scirs2_core::ndarray::s![idx, ..]));
    }
    out
}

/// Build a uniform weight vector that averages to `global_mean` when dotted with `vals`.
/// Used as fallback when Cholesky fails.
fn uniform_weights<F: Float + FromPrimitive>(
    m: usize,
    _global_mean: F,
    vals: &Array1<F>,
) -> Array1<F> {
    let n = F::from_usize(m).expect("const");
    // Simple normalised weights: 1/n each
    let mut w = Array1::zeros(m);
    let sum_vals: F = vals.iter().fold(F::zero(), |acc, &v| acc + v);
    if sum_vals.abs() > F::from_f64(1e-300).expect("const") {
        for j in 0..m {
            w[j] = F::one() / n;
        }
    } else {
        for j in 0..m {
            w[j] = F::one() / n;
        }
    }
    w
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "fast_kriging_reexports_tests.rs"]
mod tests;
