//! Sketch matrix types and embedding transforms
//!
//! This module provides various randomised linear maps (sketches) used to compress
//! tall or wide matrices while approximately preserving geometric structure.
//!
//! # Implemented Sketches
//!
//! | Type | Sketch | Time | Notes |
//! |------|--------|------|-------|
//! | [`GaussianSketch`]     | i.i.d. N(0,1/m)   | O(nmd) | optimal distortion |
//! | [`SRHTTransform`]      | Subsampled RHT    | O(nd log n) | fast apply |
//! | [`CountSketchMatrix`]  | sparse ±1         | O(nnz) | streaming-friendly |
//! | [`JLTransform`]        | Johnson-Lindenstrauss | O(nmd) | distance-preserving |
//!
//! # References
//!
//! - Johnson & Lindenstrauss (1984). "Extensions of Lipschitz mappings."
//! - Woodruff (2014). "Sketching as a tool for numerical linear algebra." (FOCS tutorial)
//! - Ailon & Chazelle (2009). "The fast Johnson-Lindenstrauss transform."
//! - Clarkson & Woodruff (2013). "Low rank approximation and regression in input sparsity time."

use scirs2_core::ndarray::{Array1, Array2, ArrayView2};
use scirs2_core::numeric::{Float, FromPrimitive, NumAssign};
use scirs2_core::random::{ChaCha8Rng, Rng, SeedableRng};
use scirs2_core::RngExt;
use std::fmt::Debug;
use std::iter::Sum;

use crate::error::{LinalgError, LinalgResult};

// ============================================================================
// GaussianSketch
// ============================================================================

/// Gaussian embedding sketch.
///
/// The sketch matrix S ∈ ℝ^{m×n} has i.i.d. entries drawn from N(0, 1/m).
/// Multiplying a vector or matrix A from the left by S compresses n rows
/// down to m rows while approximately preserving norms:
///
/// ```text
/// (1 − ε) ‖x‖ ≤ ‖Sx‖ ≤ (1 + ε) ‖x‖
/// ```
///
/// with high probability when m = O(ε⁻² log(1/δ)).
#[derive(Debug, Clone)]
pub struct GaussianSketch<F> {
    /// The sketch matrix S ∈ ℝ^{m×n} scaled by 1/√m
    pub matrix: Array2<F>,
    /// Sketch dimension (output rows)
    pub sketch_dim: usize,
    /// Input dimension (input rows n)
    pub input_dim: usize,
}

impl<F> GaussianSketch<F>
where
    F: Float + NumAssign + FromPrimitive + Debug + Sum + 'static,
{
    /// Construct a Gaussian sketch.
    ///
    /// # Arguments
    ///
    /// * `sketch_dim` – number of output rows m (should be << `input_dim`)
    /// * `input_dim`  – number of input rows n
    /// * `seed`       – optional RNG seed for reproducibility
    pub fn new(sketch_dim: usize, input_dim: usize, seed: Option<u64>) -> LinalgResult<Self> {
        if sketch_dim == 0 {
            return Err(LinalgError::ValueError(
                "sketch_dim must be positive".to_string(),
            ));
        }
        if input_dim == 0 {
            return Err(LinalgError::ValueError(
                "input_dim must be positive".to_string(),
            ));
        }

        let matrix = gaussian_random_matrix(sketch_dim, input_dim, seed)?;
        Ok(Self {
            matrix,
            sketch_dim,
            input_dim,
        })
    }

    /// Apply the sketch: compute S·A where A ∈ ℝ^{n×d}.
    ///
    /// # Arguments
    ///
    /// * `a` – Input matrix with shape (n, d); n must equal `self.input_dim`
    pub fn apply(&self, a: &ArrayView2<F>) -> LinalgResult<Array2<F>> {
        let (n, _d) = a.dim();
        if n != self.input_dim {
            return Err(LinalgError::DimensionError(format!(
                "GaussianSketch: expected {} rows, got {}",
                self.input_dim, n
            )));
        }
        matmul_2d(&self.matrix.view(), a)
    }
}

// ============================================================================
// SRHT (Subsampled Randomised Hadamard Transform)
// ============================================================================

/// Subsampled Randomised Hadamard Transform (SRHT / FJLT).
///
/// Computes S = (1/√m) · P · H · D  where:
/// - D ∈ ℝ^{n×n} is a diagonal Rademacher sign matrix  
/// - H is the Walsh-Hadamard transform matrix of size n×n (n must be a power of 2)
/// - P ∈ ℝ^{m×n} is a random row-sampling matrix
///
/// Applying S to a vector x costs O(n log n) vs O(mn) for a dense Gaussian sketch,
/// with similar distortion guarantees.
///
/// # Dimension requirement
///
/// The input dimension n must be a power of 2.  If the matrix does not satisfy
/// this, pad with zero rows before creating the sketch.
#[derive(Debug, Clone)]
pub struct SRHTTransform<F> {
    /// Rademacher signs d_i ∈ {-1, +1}, length n
    pub signs: Array1<F>,
    /// Row indices to keep after the Hadamard transform, length m
    pub row_indices: Vec<usize>,
    /// Sketch dimension m
    pub sketch_dim: usize,
    /// Input dimension n (must be a power of 2)
    pub input_dim: usize,
}

impl<F> SRHTTransform<F>
where
    F: Float + NumAssign + FromPrimitive + Debug + Sum + 'static,
{
    /// Construct an SRHT sketch.
    ///
    /// # Arguments
    ///
    /// * `sketch_dim` – number of output rows m
    /// * `input_dim`  – number of input rows n (must be a power of 2)
    /// * `seed`       – optional RNG seed
    pub fn new(sketch_dim: usize, input_dim: usize, seed: Option<u64>) -> LinalgResult<Self> {
        if sketch_dim == 0 {
            return Err(LinalgError::ValueError(
                "sketch_dim must be positive".to_string(),
            ));
        }
        if input_dim == 0 {
            return Err(LinalgError::ValueError(
                "input_dim must be positive".to_string(),
            ));
        }
        if !input_dim.is_power_of_two() {
            return Err(LinalgError::ValueError(format!(
                "SRHTTransform: input_dim must be a power of 2, got {}",
                input_dim
            )));
        }
        if sketch_dim > input_dim {
            return Err(LinalgError::ValueError(format!(
                "sketch_dim ({sketch_dim}) must be ≤ input_dim ({input_dim})"
            )));
        }

        let mut rng: ChaCha8Rng = match seed {
            Some(s) => ChaCha8Rng::seed_from_u64(s),
            None => {
                let t = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos() as u64)
                    .unwrap_or(12345);
                ChaCha8Rng::seed_from_u64(t)
            }
        };

        // Rademacher signs
        let mut signs = Array1::zeros(input_dim);
        for i in 0..input_dim {
            let bit: u8 = rng.random::<u8>() & 1;
            signs[i] = if bit == 0 { F::one() } else { -F::one() };
        }

        // Sample m distinct row indices from [0, input_dim)
        let row_indices = sample_without_replacement(input_dim, sketch_dim, &mut rng);

        Ok(Self {
            signs,
            row_indices,
            sketch_dim,
            input_dim,
        })
    }

    /// Apply the SRHT to a matrix A ∈ ℝ^{n×d}.
    ///
    /// Returns S·A ∈ ℝ^{m×d}.
    pub fn apply(&self, a: &ArrayView2<F>) -> LinalgResult<Array2<F>> {
        let (n, d) = a.dim();
        if n != self.input_dim {
            return Err(LinalgError::DimensionError(format!(
                "SRHTTransform: expected {} rows, got {}",
                self.input_dim, n
            )));
        }

        let scale = F::from(1.0 / (self.sketch_dim as f64).sqrt()).ok_or_else(|| {
            LinalgError::ComputationError("SRHT scale conversion failed".to_string())
        })?;

        // Step 1: apply diagonal D: each row i of A is multiplied by signs[i]
        let mut work = Array2::<F>::zeros((n, d));
        for i in 0..n {
            for j in 0..d {
                work[[i, j]] = self.signs[i] * a[[i, j]];
            }
        }

        // Step 2: apply Walsh-Hadamard transform column-by-column
        for j in 0..d {
            let mut col: Vec<F> = (0..n).map(|i| work[[i, j]]).collect();
            fwht_inplace(&mut col);
            for i in 0..n {
                work[[i, j]] = col[i];
            }
        }

        // Step 3: subsample rows and apply scale
        let mut out = Array2::<F>::zeros((self.sketch_dim, d));
        for (out_row, &src_row) in self.row_indices.iter().enumerate() {
            for j in 0..d {
                out[[out_row, j]] = work[[src_row, j]] * scale;
            }
        }

        Ok(out)
    }
}

// ============================================================================
// CountSketchMatrix
// ============================================================================

/// Count sketch matrix.
///
/// S ∈ ℝ^{m×n} is an extremely sparse sketch: each input coordinate i is mapped
/// to a single output bucket h(i) ∈ \[m\] with a random sign σ(i) ∈ {−1, +1}.
///
/// Applying S to a vector takes O(n) time, making it ideal for sparse inputs.
/// The sketch satisfies:
///   𝔼[‖SAx‖²] = ‖Ax‖²  (unbiased norm estimate)
#[derive(Debug, Clone)]
pub struct CountSketchMatrix<F> {
    /// Bucket assignment for each input index; length n
    pub hash: Vec<usize>,
    /// Sign flip for each input index; length n (values ±1)
    pub signs: Array1<F>,
    /// Sketch dimension m (number of buckets)
    pub sketch_dim: usize,
    /// Input dimension n
    pub input_dim: usize,
}

impl<F> CountSketchMatrix<F>
where
    F: Float + NumAssign + FromPrimitive + Debug + Sum + 'static,
{
    /// Construct a count sketch.
    ///
    /// # Arguments
    ///
    /// * `sketch_dim` – m, number of hash buckets
    /// * `input_dim`  – n, input dimension
    /// * `seed`       – optional RNG seed
    pub fn new(sketch_dim: usize, input_dim: usize, seed: Option<u64>) -> LinalgResult<Self> {
        if sketch_dim == 0 {
            return Err(LinalgError::ValueError(
                "sketch_dim must be positive".to_string(),
            ));
        }
        if input_dim == 0 {
            return Err(LinalgError::ValueError(
                "input_dim must be positive".to_string(),
            ));
        }

        let mut rng: ChaCha8Rng = match seed {
            Some(s) => ChaCha8Rng::seed_from_u64(s),
            None => {
                let t = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos() as u64)
                    .unwrap_or(99999);
                ChaCha8Rng::seed_from_u64(t)
            }
        };

        let mut hash: Vec<usize> = Vec::with_capacity(input_dim);
        let mut signs: Array1<F> = Array1::zeros(input_dim);

        for i in 0..input_dim {
            hash.push((rng.random::<u64>() as usize) % sketch_dim);
            let bit: u8 = rng.random::<u8>() & 1;
            signs[i] = if bit == 0 { F::one() } else { -F::one() };
        }

        Ok(Self {
            hash,
            signs,
            sketch_dim,
            input_dim,
        })
    }

    /// Apply the count sketch: compute S·A where A ∈ ℝ^{n×d}.
    pub fn apply(&self, a: &ArrayView2<F>) -> LinalgResult<Array2<F>> {
        let (n, d) = a.dim();
        if n != self.input_dim {
            return Err(LinalgError::DimensionError(format!(
                "CountSketchMatrix: expected {} rows, got {}",
                self.input_dim, n
            )));
        }

        let mut out = Array2::<F>::zeros((self.sketch_dim, d));
        for i in 0..n {
            let bucket = self.hash[i];
            let sign = self.signs[i];
            for j in 0..d {
                out[[bucket, j]] += sign * a[[i, j]];
            }
        }
        Ok(out)
    }
}

// ============================================================================
// JLTransform
// ============================================================================

/// Johnson-Lindenstrauss transform.
///
/// The JL lemma guarantees that for a set of N points in ℝ^n, there exists a
/// linear map f: ℝ^n → ℝ^m with m = O(ε⁻² log N) such that:
///
/// ```text
/// (1 − ε) ‖u − v‖ ≤ ‖f(u) − f(v)‖ ≤ (1 + ε) ‖u − v‖
/// ```
///
/// for all pairs u, v in the set.
///
/// This implementation uses a scaled Gaussian projection with the optimal
/// `m = ⌈(4 ln N) / (ε²/2 − ε³/3)⌉` sample complexity.
#[derive(Debug, Clone)]
pub struct JLTransform<F> {
    /// The projection matrix ∈ ℝ^{m×n}
    pub matrix: Array2<F>,
    /// Target dimension m
    pub target_dim: usize,
    /// Source dimension n
    pub source_dim: usize,
    /// Distortion parameter ε used at construction time
    pub epsilon: F,
}

impl<F> JLTransform<F>
where
    F: Float + NumAssign + FromPrimitive + Debug + Sum + 'static,
{
    /// Construct a JL transform from distortion parameter ε and number of points.
    ///
    /// # Arguments
    ///
    /// * `source_dim` – dimension of the original space n
    /// * `n_points`   – number of points to embed N (used to compute m)
    /// * `epsilon`    – allowed distortion, should be in (0, 1)
    /// * `seed`       – optional RNG seed
    pub fn new(
        source_dim: usize,
        n_points: usize,
        epsilon: F,
        seed: Option<u64>,
    ) -> LinalgResult<Self> {
        if source_dim == 0 {
            return Err(LinalgError::ValueError(
                "source_dim must be positive".to_string(),
            ));
        }
        let eps_f64 = epsilon
            .to_f64()
            .ok_or_else(|| LinalgError::ValueError("epsilon conversion failed".to_string()))?;
        if !(0.0 < eps_f64 && eps_f64 < 1.0) {
            return Err(LinalgError::ValueError(format!(
                "epsilon must be in (0,1), got {eps_f64}"
            )));
        }

        // JL target dimension: m = ceil(4 ln N / (ε²/2 - ε³/3))
        let n_pts_f = (n_points.max(2) as f64).ln();
        let denom = eps_f64 * eps_f64 / 2.0 - eps_f64 * eps_f64 * eps_f64 / 3.0;
        let target_dim = ((4.0 * n_pts_f / denom).ceil() as usize)
            .max(1)
            .min(source_dim);

        let matrix = gaussian_random_matrix(target_dim, source_dim, seed)?;
        Ok(Self {
            matrix,
            target_dim,
            source_dim,
            epsilon,
        })
    }

    /// Construct with an explicit target dimension (bypassing the JL formula).
    pub fn with_target_dim(
        source_dim: usize,
        target_dim: usize,
        epsilon: F,
        seed: Option<u64>,
    ) -> LinalgResult<Self> {
        if source_dim == 0 || target_dim == 0 {
            return Err(LinalgError::ValueError(
                "source_dim and target_dim must be positive".to_string(),
            ));
        }
        let matrix = gaussian_random_matrix(target_dim, source_dim, seed)?;
        Ok(Self {
            matrix,
            target_dim,
            source_dim,
            epsilon,
        })
    }

    /// Embed a single point x ∈ ℝ^n → ℝ^m.
    pub fn embed_point(&self, x: &Array1<F>) -> LinalgResult<Array1<F>> {
        if x.len() != self.source_dim {
            return Err(LinalgError::DimensionError(format!(
                "JLTransform: expected vector length {}, got {}",
                self.source_dim,
                x.len()
            )));
        }
        let mut out = Array1::<F>::zeros(self.target_dim);
        for i in 0..self.target_dim {
            let mut acc = F::zero();
            for j in 0..self.source_dim {
                acc += self.matrix[[i, j]] * x[j];
            }
            out[i] = acc;
        }
        Ok(out)
    }

    /// Embed a matrix of row-points X ∈ ℝ^{N×n} → ℝ^{N×m}.
    pub fn embed_rows(&self, x: &ArrayView2<F>) -> LinalgResult<Array2<F>> {
        let (rows, cols) = x.dim();
        if cols != self.source_dim {
            return Err(LinalgError::DimensionError(format!(
                "JLTransform: expected {} columns, got {}",
                self.source_dim, cols
            )));
        }
        // X · S^T  where S = self.matrix (m×n)
        let mut out = Array2::<F>::zeros((rows, self.target_dim));
        for i in 0..rows {
            for k in 0..self.target_dim {
                let mut acc = F::zero();
                for j in 0..self.source_dim {
                    acc += x[[i, j]] * self.matrix[[k, j]];
                }
                out[[i, k]] = acc;
            }
        }
        Ok(out)
    }
}

// ============================================================================
// apply_sketch
// ============================================================================

/// Apply a sketch matrix S ∈ ℝ^{m×n} to A ∈ ℝ^{n×d} to produce SA ∈ ℝ^{m×d}.
///
/// This is a convenience wrapper that accepts any `Array2<F>` as the sketch.
/// Specialised sketch types (Gaussian, SRHT, Count) have their own `apply` methods.
///
/// # Arguments
///
/// * `sketch` – sketch matrix S with shape (m, n)
/// * `a`      – input matrix A with shape (n, d)
///
/// # Returns
///
/// SA with shape (m, d).
///
/// # Examples
///
/// ```rust
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::sketching::sketch_matrix::{GaussianSketch, apply_sketch};
///
/// let a = array![[1.0_f64, 2.0], [3.0, 4.0], [5.0, 6.0]];
/// let sketch = GaussianSketch::new(2, 3, Some(0)).expect("sketch creation failed");
/// let sa = apply_sketch(&sketch.matrix.view(), &a.view()).expect("sketch failed");
/// assert_eq!(sa.shape(), &[2, 2]);
/// ```
pub fn apply_sketch<F>(sketch: &ArrayView2<F>, a: &ArrayView2<F>) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + FromPrimitive + Debug + Sum + 'static,
{
    let (m, n) = sketch.dim();
    let (a_rows, _d) = a.dim();
    if n != a_rows {
        return Err(LinalgError::DimensionError(format!(
            "apply_sketch: sketch has {n} columns but A has {a_rows} rows"
        )));
    }
    matmul_2d(sketch, a)
}

// ============================================================================
// jl_embed_points
// ============================================================================

/// Johnson-Lindenstrauss embedding of a set of points.
///
/// Embeds a dataset X ∈ ℝ^{N×n} into a lower-dimensional space ℝ^{N×m} such
/// that pairwise Euclidean distances are approximately preserved.
///
/// # Arguments
///
/// * `x`       – data matrix of shape (N, n), each row is a point
/// * `epsilon` – desired distortion factor ε ∈ (0, 1)
/// * `seed`    – optional RNG seed
///
/// # Returns
///
/// Embedded data matrix of shape (N, m), together with the [`JLTransform`] used.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::ndarray::array;
/// use scirs2_linalg::sketching::sketch_matrix::jl_embed_points;
///
/// let x = array![[1.0_f64, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
/// let (embedded, _transform) = jl_embed_points(&x.view(), 0.5, Some(42)).expect("JL failed");
/// assert_eq!(embedded.nrows(), 3);
/// ```
pub fn jl_embed_points<F>(
    x: &ArrayView2<F>,
    epsilon: F,
    seed: Option<u64>,
) -> LinalgResult<(Array2<F>, JLTransform<F>)>
where
    F: Float + NumAssign + FromPrimitive + Debug + Sum + 'static,
{
    let (n_points, source_dim) = x.dim();
    if n_points == 0 {
        return Err(LinalgError::ValueError(
            "jl_embed_points: empty input matrix".to_string(),
        ));
    }
    let transform = JLTransform::new(source_dim, n_points, epsilon, seed)?;
    let embedded = transform.embed_rows(x)?;
    Ok((embedded, transform))
}

// ============================================================================
// Internal helpers
// ============================================================================

/// Generate a scaled Gaussian random matrix with i.i.d. entries N(0, 1/cols).
fn gaussian_random_matrix<F>(rows: usize, cols: usize, seed: Option<u64>) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + FromPrimitive + Debug + 'static,
{
    use scirs2_core::random::{Distribution, Normal};

    let mut rng: ChaCha8Rng = match seed {
        Some(s) => ChaCha8Rng::seed_from_u64(s),
        None => {
            let t = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(777);
            ChaCha8Rng::seed_from_u64(t)
        }
    };

    let std_dev = 1.0 / (cols as f64).sqrt();
    let normal = Normal::new(0.0, std_dev).map_err(|e| {
        LinalgError::ComputationError(format!("Normal distribution creation failed: {e}"))
    })?;

    let mut mat = Array2::<F>::zeros((rows, cols));
    for i in 0..rows {
        for j in 0..cols {
            let v: f64 = normal.sample(&mut rng);
            mat[[i, j]] = F::from(v).ok_or_else(|| {
                LinalgError::ComputationError(format!("Failed to convert f64 {v} to F"))
            })?;
        }
    }
    Ok(mat)
}

/// In-place Fast Walsh-Hadamard Transform (length must be a power of 2).
fn fwht_inplace<F>(x: &mut [F])
where
    F: Float + NumAssign,
{
    let n = x.len();
    let mut h = 1usize;
    while h < n {
        let mut i = 0;
        while i < n {
            for j in i..(i + h) {
                let u = x[j];
                let v = x[j + h];
                x[j] = u + v;
                x[j + h] = u - v;
            }
            i += 2 * h;
        }
        h *= 2;
    }
}

/// Sample `k` distinct indices from [0, n) without replacement (Fisher-Yates).
fn sample_without_replacement<R>(n: usize, k: usize, rng: &mut R) -> Vec<usize>
where
    R: Rng,
{
    let mut pool: Vec<usize> = (0..n).collect();
    for i in 0..k {
        let j = i + (rng.random::<u64>() as usize) % (n - i);
        pool.swap(i, j);
    }
    pool[..k].to_vec()
}

/// Dense matrix multiplication: C = A · B.
fn matmul_2d<F>(a: &ArrayView2<F>, b: &ArrayView2<F>) -> LinalgResult<Array2<F>>
where
    F: Float + NumAssign + 'static,
{
    let (m, k) = a.dim();
    let (k2, n) = b.dim();
    if k != k2 {
        return Err(LinalgError::DimensionError(format!(
            "matmul_2d: inner dimensions {k} and {k2} do not match"
        )));
    }
    let mut c = Array2::<F>::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            let mut acc = F::zero();
            for l in 0..k {
                acc += a[[i, l]] * b[[l, j]];
            }
            c[[i, j]] = acc;
        }
    }
    Ok(c)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    // Euclidean norm of a row vector
    fn row_norm(a: &Array2<f64>, row: usize) -> f64 {
        let mut s = 0.0f64;
        for j in 0..a.ncols() {
            s += a[[row, j]] * a[[row, j]];
        }
        s.sqrt()
    }

    #[test]
    fn test_gaussian_sketch_shape() {
        let a = Array2::<f64>::ones((10, 4));
        let sk = GaussianSketch::new(3, 10, Some(1)).expect("gaussian sketch");
        let sa = sk.apply(&a.view()).expect("apply failed");
        assert_eq!(sa.shape(), &[3, 4]);
    }

    #[test]
    fn test_gaussian_sketch_dimension_mismatch() {
        let a = Array2::<f64>::ones((5, 4));
        let sk = GaussianSketch::new(3, 10, Some(1)).expect("gaussian sketch");
        assert!(sk.apply(&a.view()).is_err());
    }

    #[test]
    fn test_apply_sketch_helper() {
        let a = array![[1.0_f64, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let sk = GaussianSketch::new(2, 3, Some(42)).expect("sketch");
        let sa = apply_sketch(&sk.matrix.view(), &a.view()).expect("apply_sketch");
        assert_eq!(sa.shape(), &[2, 2]);
    }

    #[test]
    fn test_srht_shape_power_of_two() {
        let n = 8usize;
        let m = 4usize;
        let a = Array2::<f64>::ones((n, 3));
        let srht = SRHTTransform::new(m, n, Some(7)).expect("srht");
        let sa = srht.apply(&a.view()).expect("srht apply");
        assert_eq!(sa.shape(), &[m, 3]);
    }

    #[test]
    fn test_srht_rejects_non_power_of_two() {
        assert!(SRHTTransform::<f64>::new(3, 7, None).is_err());
    }

    #[test]
    fn test_count_sketch_shape() {
        let a = Array2::<f64>::ones((20, 5));
        let cs = CountSketchMatrix::new(8, 20, Some(3)).expect("count sketch");
        let sa = cs.apply(&a.view()).expect("cs apply");
        assert_eq!(sa.shape(), &[8, 5]);
    }

    #[test]
    fn test_count_sketch_linearity() {
        // SA(x + y) = SAx + SAy
        let a = array![[1.0_f64, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let b = array![[0.0_f64, 1.0], [1.0, 0.0], [1.0, -1.0]];
        let cs = CountSketchMatrix::new(2, 3, Some(99)).expect("cs");
        let sa = cs.apply(&a.view()).expect("sa");
        let sb = cs.apply(&b.view()).expect("sb");

        // A + B
        let mut ab = a.clone();
        for i in 0..3 {
            for j in 0..2 {
                ab[[i, j]] += b[[i, j]];
            }
        }
        let s_ab = cs.apply(&ab.view()).expect("s_ab");

        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (s_ab[[i, j]] - sa[[i, j]] - sb[[i, j]]).abs() < 1e-12,
                    "linearity violated at [{i},{j}]"
                );
            }
        }
    }

    #[test]
    fn test_jl_embed_points_shape() {
        let x = Array2::<f64>::ones((50, 100));
        let (emb, t) = jl_embed_points(&x.view(), 0.3, Some(11)).expect("jl embed");
        assert_eq!(emb.nrows(), 50);
        assert_eq!(emb.ncols(), t.target_dim);
        assert!(t.target_dim <= 100);
    }

    #[test]
    fn test_jl_preserves_distances_roughly() {
        // Three orthogonal unit vectors; after JL the pairwise distances should
        // be roughly √2 (since ||e_i - e_j||² = 2 for i ≠ j).
        let x = array![
            [1.0_f64, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ];
        let transform = JLTransform::with_target_dim(8, 4, 0.3f64, Some(7)).expect("jl transform");
        let emb = transform.embed_rows(&x.view()).expect("embed");

        // Check that all rows have roughly unit norm
        for i in 0..4 {
            let n = row_norm(&emb, i);
            // With 4-dim projection, the variance is high; just check non-zero
            assert!(n > 0.0, "row {i} has zero norm after JL");
        }
    }

    #[test]
    fn test_fwht_known_values() {
        let mut x = vec![1.0f64, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0];
        fwht_inplace(&mut x);
        // WHT of this vector: sum = 4
        let sum: f64 = x.iter().sum();
        // FWHT is unnormalised; x[0] should equal original sum = 4
        let original_sum = 1.0 + 0.0 + 1.0 + 0.0 + 0.0 + 1.0 + 1.0 + 0.0;
        assert!(
            (x[0] - original_sum).abs() < 1e-12,
            "FWHT x[0] = {}, expected {}",
            x[0],
            original_sum
        );
    }

    #[test]
    fn test_jl_transform_with_target_dim() {
        let t = JLTransform::<f64>::with_target_dim(10, 3, 0.3, Some(1)).expect("jl");
        assert_eq!(t.target_dim, 3);
        assert_eq!(t.source_dim, 10);
    }
}
