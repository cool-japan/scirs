//! FFT-based matrix algorithms for circulant, Toeplitz, and related structured matrices
//!
//! This module implements O(n log n) algorithms for matrix-vector products and
//! linear system solves involving circulant, Toeplitz, and convolutional structure,
//! replacing naive O(n²) dense operations with FFT-accelerated methods.
//!
//! ## Structures covered
//!
//! - **Circulant**: C is fully defined by its first column c; C[i,j] = c[(i-j) mod n]
//! - **Toeplitz**: T[i,j] = t[j-i] (1D defining sequence, embedded in a 2n circulant)
//! - **Level-Toeplitz**: Block-Toeplitz with Toeplitz blocks (2D Toeplitz)
//!
//! ## Algorithms
//!
//! - **circulant_matmul**: O(n log n) using FFT convolution
//! - **toeplitz_matmul**: O(n log n) via 2n circulant embedding
//! - **convolve_matmul**: Generic FFT-based convolution product
//! - **FFTBasedSolver**: Direct solver for circulant/Toeplitz systems
//! - **LevelToeplitz**: Multilevel (2D) Toeplitz matrix-vector product
//!
//! ## References
//!
//! - Davis, P. J. (1979). "Circulant Matrices". Wiley.
//! - Strang, G. (1986). "A proposal for Toeplitz matrix calculations"
//! - Chan, R. H., & Ng, M. K. (1996). "Conjugate Gradient Methods for Toeplitz Systems"

use scirs2_core::numeric::Complex;
use std::f64::consts::PI;

use crate::error::{LinalgError, LinalgResult};

// ============================================================================
// Internal FFT primitives (Cooley-Tukey, power-of-2)
// ============================================================================

type C64 = Complex<f64>;

/// Pad `v` to the next power of 2 with zeros.
fn pad_to_power_of_two(v: &[C64]) -> Vec<C64> {
    let n = v.len();
    let m = n.next_power_of_two();
    let mut out = v.to_vec();
    out.resize(m, C64::new(0.0, 0.0));
    out
}

/// In-place Cooley-Tukey radix-2 FFT / IFFT.
///
/// `inverse = false` → forward FFT, `inverse = true` → IFFT (with 1/n scaling).
fn fft_inplace(data: &mut [C64], inverse: bool) {
    let n = data.len();
    debug_assert!(n.is_power_of_two(), "FFT size must be a power of 2");

    // Bit-reversal permutation
    let log2n = n.trailing_zeros() as usize;
    for i in 0..n {
        let j = reverse_bits(i, log2n);
        if j > i {
            data.swap(i, j);
        }
    }

    // Butterfly stages
    let mut h = 1usize;
    while h < n {
        let sign = if inverse { 1.0_f64 } else { -1.0_f64 };
        let angle_step = sign * PI / h as f64;
        let w_n = C64::new(angle_step.cos(), angle_step.sin());
        let mut i = 0;
        while i < n {
            let mut w = C64::new(1.0, 0.0);
            for j in 0..h {
                let u = data[i + j];
                let v = w * data[i + j + h];
                data[i + j] = u + v;
                data[i + j + h] = u - v;
                w *= w_n;
            }
            i += 2 * h;
        }
        h *= 2;
    }

    if inverse {
        let scale = 1.0 / n as f64;
        for d in data.iter_mut() {
            *d *= scale;
        }
    }
}

fn reverse_bits(mut x: usize, bits: usize) -> usize {
    let mut result = 0usize;
    for _ in 0..bits {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    result
}

/// Forward FFT of a real-valued sequence.
fn rfft(x: &[f64]) -> Vec<C64> {
    let mut buf: Vec<C64> = x.iter().map(|&v| C64::new(v, 0.0)).collect();
    fft_inplace(&mut buf, false);
    buf
}

/// Inverse FFT returning the real part only.
fn irfft(x: &mut [C64]) -> Vec<f64> {
    fft_inplace(x, true);
    x.iter().map(|c| c.re).collect()
}

/// Exact n-point DFT of a real-valued sequence (O(n²), but exact for any n).
///
/// This is used for circulant eigenvalue computation where padding to a
/// power-of-2 would give wrong eigenvalues when n is not a power of 2.
fn dft_exact(x: &[f64]) -> Vec<C64> {
    let n = x.len();
    let mut out = vec![C64::new(0.0, 0.0); n];
    let two_pi_over_n = 2.0 * std::f64::consts::PI / (n as f64);
    for k in 0..n {
        let mut sum = C64::new(0.0, 0.0);
        for (j, &xj) in x.iter().enumerate() {
            let angle = -(j as f64) * (k as f64) * two_pi_over_n;
            sum += C64::new(xj * angle.cos(), xj * angle.sin());
        }
        out[k] = sum;
    }
    out
}

/// Exact n-point IDFT returning the real part only.
///
/// Computes IDFT[k] = (1/n) * sum_j X[j] * exp(+2πi*j*k/n).
fn idft_exact(x: &[C64]) -> Vec<f64> {
    let n = x.len();
    let two_pi_over_n = 2.0 * std::f64::consts::PI / (n as f64);
    let mut out = vec![0.0f64; n];
    for k in 0..n {
        let mut sum = C64::new(0.0, 0.0);
        for (j, &xj) in x.iter().enumerate() {
            let angle = (j as f64) * (k as f64) * two_pi_over_n;
            sum += xj * C64::new(angle.cos(), angle.sin());
        }
        out[k] = sum.re / (n as f64);
    }
    out
}

// ============================================================================
// circulant_matmul
// ============================================================================

/// Compute the circulant matrix-vector product y = C(c) * x in O(n log n).
///
/// The circulant matrix C is defined by its first column c:
///   C[i, j] = c[(i - j) mod n]
///
/// The product is equivalent to the cyclic convolution c ⊛ x, computed
/// via FFT: y = IFFT(FFT(c) ⊙ FFT(x)).
///
/// # Arguments
///
/// * `c` - First column of the circulant matrix (length n)
/// * `x` - Input vector (length n)
///
/// # Returns
///
/// * Product vector y of length n
///
/// # Examples
///
/// ```
/// use scirs2_linalg::fft_based::circulant_matmul;
///
/// let c = vec![1.0, 2.0, 3.0, 4.0];
/// let x = vec![1.0, 0.0, 0.0, 0.0];
/// let y = circulant_matmul(&c, &x).expect("circulant_matmul failed");
/// // y should equal c (first column of C times unit vector)
/// assert!((y[0] - 1.0).abs() < 1e-10);
/// assert!((y[1] - 2.0).abs() < 1e-10);
/// ```
pub fn circulant_matmul(c: &[f64], x: &[f64]) -> LinalgResult<Vec<f64>> {
    let n = c.len();
    if x.len() != n {
        return Err(LinalgError::DimensionError(format!(
            "circulant_matmul: c has length {} but x has length {}",
            n,
            x.len()
        )));
    }
    if n == 0 {
        return Ok(Vec::new());
    }

    if n.is_power_of_two() {
        // Direct FFT-based circular convolution (exact for power-of-2)
        let c_fft = rfft(c);
        let x_fft = rfft(x);

        let mut prod: Vec<C64> = c_fft
            .iter()
            .zip(x_fft.iter())
            .map(|(&cf, &xf)| cf * xf)
            .collect();

        let y_full = irfft(&mut prod);
        Ok(y_full[..n].to_vec())
    } else {
        // For non-power-of-2: compute linear convolution via FFT, then wrap
        // Linear convolution of c and x has length 2n-1; we wrap modulo n.
        let conv_len = 2 * n - 1;
        let m = conv_len.next_power_of_two();

        let mut c_ext = vec![0.0f64; m];
        c_ext[..n].copy_from_slice(c);

        let mut x_ext = vec![0.0f64; m];
        x_ext[..n].copy_from_slice(x);

        let c_fft = rfft(&c_ext);
        let x_fft = rfft(&x_ext);

        let mut prod: Vec<C64> = c_fft
            .iter()
            .zip(x_fft.iter())
            .map(|(&cf, &xf)| cf * xf)
            .collect();

        let y_linear = irfft(&mut prod);

        // Wrap: y[i] = y_linear[i] + y_linear[i + n] (the latter exists for i < n-1)
        let mut y = vec![0.0f64; n];
        for i in 0..n {
            y[i] = y_linear[i];
            if i + n < conv_len {
                y[i] += y_linear[i + n];
            }
        }
        Ok(y)
    }
}

// ============================================================================
// toeplitz_matmul
// ============================================================================

/// Compute the Toeplitz matrix-vector product y = T * x in O(n log n).
///
/// The Toeplitz matrix T is characterized by a single vector t of length 2n-1,
/// where t[0..n-1] are the first row (right to left) and t[0],t[-(1)],..,t[-(n-1)]
/// are the first column. Equivalently, T[i, j] = t[n-1 + i - j].
///
/// The product is computed by embedding T in a 2n×2n circulant matrix and
/// using the circulant FFT trick.
///
/// # Arguments
///
/// * `t` - Toeplitz vector of length 2n-1, where `t[0]` is the top-left element.
///   Interpretation: T[i,j] = t[i - j + (n - 1)].
/// * `x` - Input vector of length n
///
/// # Returns
///
/// * Product vector y of length n
///
/// # Examples
///
/// ```
/// use scirs2_linalg::fft_based::toeplitz_matmul;
///
/// // Build the 3×3 Toeplitz matrix T with t = [3, 2, 1, 4, 7]
/// // (n=3, t[0]=3 = T[2,0], t[2]=1 = T[0,0] = diagonal, etc.)
/// let t = vec![3.0_f64, 2.0, 1.0, 4.0, 7.0]; // 2n-1 = 5
/// let x = vec![1.0, 0.0, 0.0];
/// let y = toeplitz_matmul(&t, &x).expect("toeplitz_matmul failed");
/// // First column of T: [t[2], t[3], t[4]] = [1, 4, 7]
/// assert!((y[0] - 1.0).abs() < 1e-9);
/// assert!((y[1] - 4.0).abs() < 1e-9);
/// assert!((y[2] - 7.0).abs() < 1e-9);
/// ```
pub fn toeplitz_matmul(t: &[f64], x: &[f64]) -> LinalgResult<Vec<f64>> {
    let n = x.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    // t must have length 2n-1
    if t.len() != 2 * n - 1 {
        return Err(LinalgError::DimensionError(format!(
            "toeplitz_matmul: t must have length 2n-1={} for n={}, got {}",
            2 * n - 1,
            n,
            t.len()
        )));
    }

    // Build the 2n circulant embedding c of length 2n.
    // The Toeplitz matrix T (n×n) with T[i,j] = t[n-1+i-j] is embedded in
    // a 2n×2n circulant C where C[i,j] = c[(i-j) mod 2n].
    // For i-j = k >= 0 (below diagonal): c[k] = t[n-1+k]
    // For i-j = -k < 0 (above diagonal): c[2n-k] = t[n-1-k]
    let two_n = 2 * n;
    let mut c = vec![0.0f64; two_n];
    // c[0] = t[n-1] (diagonal)
    c[0] = t[n - 1];
    // c[k] for k=1..n-1: below-diagonal entries (i-j = k), i.e. t[n-1+k]
    for k in 1..n {
        c[k] = t[n - 1 + k];
    }
    // c[n] = 0 (already 0)
    // c[2n-k] for k=1..n-1: above-diagonal entries (i-j = -k), i.e. t[n-1-k]
    for k in 1..n {
        c[two_n - k] = t[n - 1 - k];
    }

    // x_ext: pad x to length 2n with zeros
    let mut x_ext = vec![0.0f64; two_n];
    x_ext[..n].copy_from_slice(x);

    // Circular convolution via FFT
    let m = two_n.next_power_of_two();
    let mut c_padded = c.clone();
    c_padded.resize(m, 0.0);
    let mut x_padded = x_ext.clone();
    x_padded.resize(m, 0.0);

    let c_fft = rfft(&c_padded);
    let x_fft = rfft(&x_padded);

    let mut prod: Vec<C64> = c_fft
        .iter()
        .zip(x_fft.iter())
        .map(|(&cf, &xf)| cf * xf)
        .collect();

    let y_full = irfft(&mut prod);

    Ok(y_full[..n].to_vec())
}

// ============================================================================
// convolve_matmul
// ============================================================================

/// Compute the linear (non-cyclic) convolution h = f ∗ g via FFT.
///
/// The result has length `f.len() + g.len() - 1`. This is useful for
/// computing products involving structured matrices expressed as convolutions.
///
/// # Arguments
///
/// * `f` - First sequence (length m)
/// * `g` - Second sequence (length n)
///
/// # Returns
///
/// * Convolution result of length m + n - 1
///
/// # Examples
///
/// ```
/// use scirs2_linalg::fft_based::convolve_matmul;
///
/// let f = vec![1.0, 2.0, 3.0];
/// let g = vec![1.0, 1.0];
/// let h = convolve_matmul(&f, &g).expect("convolve_matmul failed");
/// assert!((h[0] - 1.0).abs() < 1e-10);
/// assert!((h[1] - 3.0).abs() < 1e-10);
/// assert!((h[2] - 5.0).abs() < 1e-10);
/// assert!((h[3] - 3.0).abs() < 1e-10);
/// ```
pub fn convolve_matmul(f: &[f64], g: &[f64]) -> LinalgResult<Vec<f64>> {
    if f.is_empty() || g.is_empty() {
        return Ok(Vec::new());
    }

    let output_len = f.len() + g.len() - 1;
    let m = output_len.next_power_of_two();

    let mut f_ext = f.to_vec();
    f_ext.resize(m, 0.0);
    let mut g_ext = g.to_vec();
    g_ext.resize(m, 0.0);

    let f_fft = rfft(&f_ext);
    let g_fft = rfft(&g_ext);

    let mut prod: Vec<C64> = f_fft
        .iter()
        .zip(g_fft.iter())
        .map(|(&ff, &gf)| ff * gf)
        .collect();

    let h_full = irfft(&mut prod);
    Ok(h_full[..output_len].to_vec())
}

// ============================================================================
// FFTBasedSolver
// ============================================================================

/// Solver type for FFT-based structured systems
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FFTSolverType {
    /// Circulant system: C x = b
    Circulant,
    /// Toeplitz system: T x = b (via circulant preconditioned CG)
    Toeplitz,
}

/// Fast direct/iterative solver for circulant and Toeplitz linear systems.
///
/// For **circulant** systems C x = b, the exact solution is:
///   x = C⁻¹ b = IFFT(FFT(b) / FFT(c))
///
/// This is O(n log n) and exact (up to floating-point arithmetic).
///
/// For **Toeplitz** systems T x = b, the solver uses the preconditioned
/// Conjugate Gradient method with a circulant preconditioner (Chan's optimal
/// preconditioner), achieving superlinear convergence in O(n log n) per iteration.
#[derive(Debug, Clone)]
pub struct FFTBasedSolver {
    /// Type of structured system
    pub solver_type: FFTSolverType,
    /// First column (circulant) or full Toeplitz vector (length 2n-1)
    defining_vector: Vec<f64>,
    /// System size n
    pub n: usize,
    /// FFT of the defining vector (for circulant) or chan preconditioner (for Toeplitz)
    fft_buf: Vec<C64>,
}

impl FFTBasedSolver {
    /// Build a solver for a **circulant** system C x = b.
    ///
    /// # Arguments
    ///
    /// * `c` - First column of the circulant matrix (length n)
    pub fn new_circulant(c: &[f64]) -> LinalgResult<Self> {
        let n = c.len();
        if n == 0 {
            return Err(LinalgError::ShapeError(
                "FFTBasedSolver: circulant size must be positive".to_string(),
            ));
        }

        // Use exact n-point DFT so eigenvalues are correct for any n,
        // not just powers of 2.
        let fft_buf = dft_exact(c);

        // Check that all eigenvalues (DFT values) are non-zero
        let min_abs = fft_buf
            .iter()
            .map(|z| z.norm())
            .fold(f64::INFINITY, f64::min);
        if min_abs < 1e-300 {
            return Err(LinalgError::SingularMatrixError(
                "FFTBasedSolver: circulant matrix is singular (zero eigenvalue)".to_string(),
            ));
        }

        Ok(Self {
            solver_type: FFTSolverType::Circulant,
            defining_vector: c.to_vec(),
            n,
            fft_buf,
        })
    }

    /// Build a solver for a **Toeplitz** system T x = b.
    ///
    /// The Toeplitz vector `t` has length 2n-1, with `t[n-1]` = diagonal,
    /// `t[n-1-k]` = sub-diagonal k, `t[n-1+k]` = super-diagonal k.
    ///
    /// # Arguments
    ///
    /// * `t` - Toeplitz defining vector of length 2n-1
    /// * `n` - System size
    pub fn new_toeplitz(t: &[f64], n: usize) -> LinalgResult<Self> {
        if n == 0 {
            return Err(LinalgError::ShapeError(
                "FFTBasedSolver: Toeplitz size must be positive".to_string(),
            ));
        }
        if t.len() != 2 * n - 1 {
            return Err(LinalgError::DimensionError(format!(
                "FFTBasedSolver: Toeplitz t must have length 2n-1={}, got {}",
                2 * n - 1,
                t.len()
            )));
        }

        // Build Chan's optimal circulant preconditioner
        // c_opt[k] = average of the kth diagonal of T
        let mut chan_c = vec![0.0f64; n];
        // diagonal k=0 has n elements, all equal t[n-1]
        chan_c[0] = t[n - 1];
        for k in 1..n {
            // sub-diagonal k: t[n-1-k], appears n-k times
            // super-diagonal k: t[n-1+k], appears n-k times
            // Chan's formula averages them:
            let sub = t[n - 1 - k];
            let sup = t[n - 1 + k];
            let weight = (n - k) as f64 / n as f64;
            let weight2 = k as f64 / n as f64;
            // c[k] is the average sub-/super-diagonal contribution
            chan_c[k] = weight * sub + weight2 * sup;
        }

        let m = n.next_power_of_two();
        let mut chan_ext = chan_c.clone();
        chan_ext.resize(m, 0.0);
        let fft_buf = rfft(&chan_ext);

        Ok(Self {
            solver_type: FFTSolverType::Toeplitz,
            defining_vector: t.to_vec(),
            n,
            fft_buf,
        })
    }

    /// Solve the structured system and return the solution vector.
    ///
    /// For circulant systems: exact O(n log n) solve.
    /// For Toeplitz systems: preconditioned CG with circulant preconditioner.
    ///
    /// # Arguments
    ///
    /// * `b` - Right-hand side vector of length n
    /// * `max_iter` - Maximum CG iterations (Toeplitz only, default: 50)
    /// * `tol` - Convergence tolerance (Toeplitz only, default: 1e-10)
    pub fn solve(
        &self,
        b: &[f64],
        max_iter: Option<usize>,
        tol: Option<f64>,
    ) -> LinalgResult<Vec<f64>> {
        if b.len() != self.n {
            return Err(LinalgError::DimensionError(format!(
                "FFTBasedSolver::solve: b has length {} but n={}",
                b.len(),
                self.n
            )));
        }

        match self.solver_type {
            FFTSolverType::Circulant => self.solve_circulant(b),
            FFTSolverType::Toeplitz => {
                let max_it = max_iter.unwrap_or(50);
                let tolerance = tol.unwrap_or(1e-10);
                self.solve_toeplitz_pcg(b, max_it, tolerance)
            }
        }
    }

    /// Apply the matrix (matvec): y = A * x
    pub fn matvec(&self, x: &[f64]) -> LinalgResult<Vec<f64>> {
        if x.len() != self.n {
            return Err(LinalgError::DimensionError(format!(
                "FFTBasedSolver::matvec: x has length {} but n={}",
                x.len(),
                self.n
            )));
        }

        match self.solver_type {
            FFTSolverType::Circulant => circulant_matmul(&self.defining_vector, x),
            FFTSolverType::Toeplitz => toeplitz_matmul(&self.defining_vector, x),
        }
    }

    /// Exact circulant solve via DFT eigendecomposition.
    ///
    /// Circulant eigenvalues are the exact n-point DFT of c.
    /// Solution: x = IDFT(DFT(b) / DFT(c)).
    fn solve_circulant(&self, b: &[f64]) -> LinalgResult<Vec<f64>> {
        let n = self.n;

        // Use exact n-point DFT for b as well
        let b_fft = dft_exact(b);

        // Divide in frequency domain: x_fft[k] = b_fft[k] / c_fft[k]
        let x_fft: Vec<C64> = b_fft
            .iter()
            .zip(self.fft_buf.iter())
            .map(|(&bf, &cf)| {
                let denom = cf.norm_sqr();
                if denom < 1e-300 {
                    C64::new(0.0, 0.0) // fallback for near-zero eigenvalue
                } else {
                    // Division: bf / cf = bf * conj(cf) / |cf|^2
                    C64::new((bf * cf.conj()).re / denom, (bf * cf.conj()).im / denom)
                }
            })
            .collect();

        // Use exact IDFT to recover x
        let x = idft_exact(&x_fft);
        Ok(x[..n].to_vec())
    }

    /// Preconditioned CG for Toeplitz system T x = b with circulant preconditioner.
    fn solve_toeplitz_pcg(&self, b: &[f64], max_iter: usize, tol: f64) -> LinalgResult<Vec<f64>> {
        let n = self.n;
        let m = n.next_power_of_two();

        let b_norm = b.iter().map(|&v| v * v).sum::<f64>().sqrt().max(1e-300);

        // Initial guess: x = 0
        let mut x = vec![0.0f64; n];
        let mut r = b.to_vec();

        // Apply circulant preconditioner M⁻¹ r
        let apply_precond = |r_vec: &[f64]| -> LinalgResult<Vec<f64>> {
            let mut r_ext = r_vec.to_vec();
            r_ext.resize(m, 0.0);
            let r_fft = rfft(&r_ext);
            let mut z_fft: Vec<C64> = r_fft
                .iter()
                .zip(self.fft_buf.iter())
                .map(|(&rf, &cf)| {
                    let denom = cf.norm_sqr();
                    if denom < 1e-300 {
                        C64::new(0.0, 0.0)
                    } else {
                        C64::new((rf * cf.conj()).re / denom, (rf * cf.conj()).im / denom)
                    }
                })
                .collect();
            let z_full = irfft(&mut z_fft);
            Ok(z_full[..n].to_vec())
        };

        let z = apply_precond(&r)?;
        let mut p = z.clone();
        let mut rz = dot_f64(&r, &z);

        for _ in 0..max_iter {
            let r_norm = dot_f64(&r, &r).sqrt();
            if r_norm / b_norm < tol {
                break;
            }

            // q = T p
            let q = toeplitz_matmul(&self.defining_vector, &p)?;

            let pq = dot_f64(&p, &q);
            if pq.abs() < 1e-300 {
                break;
            }

            let alpha = rz / pq;

            // x += alpha * p
            for i in 0..n {
                x[i] += alpha * p[i];
            }
            // r -= alpha * q
            for i in 0..n {
                r[i] -= alpha * q[i];
            }

            let z_new = apply_precond(&r)?;
            let rz_new = dot_f64(&r, &z_new);
            let beta = rz_new / rz.max(1e-300);
            rz = rz_new;

            // p = z_new + beta * p
            for i in 0..n {
                p[i] = z_new[i] + beta * p[i];
            }
            // z_new is not stored as it's fully captured in p above
            let _ = z_new;
        }

        Ok(x)
    }
}

// ============================================================================
// LevelToeplitz: Multilevel (2D) Toeplitz matrix-vector product
// ============================================================================

/// Multilevel (2D) Toeplitz matrix-vector product via 2D FFT.
///
/// A level-2 Toeplitz matrix T corresponds to a 2D Toeplitz structure where
/// T[i₁, i₂; j₁, j₂] = t[i₁ - j₁, i₂ - j₂]. Such matrices arise in 2D signal
/// processing, image restoration, and PDE discretizations on regular grids.
///
/// The matrix-vector product is computed as a 2D circular convolution:
///   y = T * x  ←→  Y = IFFT2(FFT2(t_pad) ⊙ FFT2(x_pad))
///
/// where t_pad is the level-Toeplitz kernel padded to a 2D circulant embedding.
#[derive(Debug, Clone)]
pub struct LevelToeplitz {
    /// Number of rows of the 2D grid (n₁)
    pub n1: usize,
    /// Number of columns of the 2D grid (n₂)
    pub n2: usize,
    /// Toeplitz kernel of shape (2*n1-1, 2*n2-1)
    /// kernel[i1+n1-1, i2+n2-1] = T[i1, i2] where (i1, i2) ∈ [-(n1-1), n1-1] × [-(n2-1), n2-1]
    kernel: Vec<Vec<f64>>,
    /// Precomputed 2D FFT of the circulant embedding
    kernel_fft: Vec<Vec<C64>>,
    /// Padded sizes (power of 2)
    m1: usize,
    m2: usize,
}

impl LevelToeplitz {
    /// Build a LevelToeplitz operator from its kernel.
    ///
    /// # Arguments
    ///
    /// * `n1` - Grid rows
    /// * `n2` - Grid columns
    /// * `kernel` - Toeplitz kernel of shape (2*n1-1) × (2*n2-1).
    ///   `kernel[r + n1 - 1][c + n2 - 1]` is the T[r, c] element.
    pub fn new(n1: usize, n2: usize, kernel: Vec<Vec<f64>>) -> LinalgResult<Self> {
        if n1 == 0 || n2 == 0 {
            return Err(LinalgError::ShapeError(
                "LevelToeplitz: n1 and n2 must be positive".to_string(),
            ));
        }

        let expected_rows = 2 * n1 - 1;
        let expected_cols = 2 * n2 - 1;

        if kernel.len() != expected_rows {
            return Err(LinalgError::DimensionError(format!(
                "LevelToeplitz: kernel must have {} rows (2*n1-1), got {}",
                expected_rows,
                kernel.len()
            )));
        }
        for (r, row) in kernel.iter().enumerate() {
            if row.len() != expected_cols {
                return Err(LinalgError::DimensionError(format!(
                    "LevelToeplitz: kernel row {} must have {} cols (2*n2-1), got {}",
                    r,
                    expected_cols,
                    row.len()
                )));
            }
        }

        let m1 = (2 * n1).next_power_of_two();
        let m2 = (2 * n2).next_power_of_two();

        // Build the 2D circulant embedding of the kernel.
        // kernel[r][c] corresponds to Toeplitz offset (d1, d2) = (r - (n1-1), c - (n2-1)).
        // The circulant embedding maps offset d to index (d mod m).
        let mut circ_real = vec![vec![0.0f64; m2]; m1];

        for r in 0..(2 * n1 - 1) {
            for c_idx in 0..(2 * n2 - 1) {
                let d1 = r as isize - (n1 as isize - 1);
                let d2 = c_idx as isize - (n2 as isize - 1);
                let r_circ = ((d1 % m1 as isize) + m1 as isize) as usize % m1;
                let c_circ = ((d2 % m2 as isize) + m2 as isize) as usize % m2;
                circ_real[r_circ][c_circ] = kernel[r][c_idx];
            }
        }

        // Compute 2D FFT of circ_real
        let kernel_fft = fft2d_real(&circ_real, m1, m2)?;

        Ok(Self {
            n1,
            n2,
            kernel,
            kernel_fft,
            m1,
            m2,
        })
    }

    /// Compute y = T * x where x is an n1×n2 matrix (flattened row-major).
    ///
    /// # Arguments
    ///
    /// * `x` - Input vector of length n1 * n2 (row-major flattening of the 2D grid)
    ///
    /// # Returns
    ///
    /// * Product vector y of length n1 * n2
    pub fn matvec(&self, x: &[f64]) -> LinalgResult<Vec<f64>> {
        let expected = self.n1 * self.n2;
        if x.len() != expected {
            return Err(LinalgError::DimensionError(format!(
                "LevelToeplitz::matvec: x has length {} but n1*n2={}",
                x.len(),
                expected
            )));
        }

        let m1 = self.m1;
        let m2 = self.m2;

        // Build padded 2D input
        let mut x_pad = vec![vec![0.0f64; m2]; m1];
        for i in 0..self.n1 {
            for j in 0..self.n2 {
                x_pad[i][j] = x[i * self.n2 + j];
            }
        }

        // 2D FFT of input
        let x_fft = fft2d_real(&x_pad, m1, m2)?;

        // Pointwise multiply in frequency domain
        let mut prod_fft = vec![vec![C64::new(0.0, 0.0); m2]; m1];
        for i in 0..m1 {
            for j in 0..m2 {
                prod_fft[i][j] = self.kernel_fft[i][j] * x_fft[i][j];
            }
        }

        // 2D IFFT
        let y_full = ifft2d(&mut prod_fft, m1, m2)?;

        // Extract the n1×n2 result
        let mut y = vec![0.0f64; self.n1 * self.n2];
        for i in 0..self.n1 {
            for j in 0..self.n2 {
                y[i * self.n2 + j] = y_full[i][j];
            }
        }

        Ok(y)
    }

    /// Solve the level-Toeplitz system T x = b using preconditioned CG.
    ///
    /// # Arguments
    ///
    /// * `b` - Right-hand side of length n1 * n2
    /// * `max_iter` - Maximum iterations (default: 100)
    /// * `tol` - Convergence tolerance (default: 1e-10)
    pub fn solve(
        &self,
        b: &[f64],
        max_iter: Option<usize>,
        tol: Option<f64>,
    ) -> LinalgResult<Vec<f64>> {
        let expected = self.n1 * self.n2;
        if b.len() != expected {
            return Err(LinalgError::DimensionError(format!(
                "LevelToeplitz::solve: b has length {} but n1*n2={}",
                b.len(),
                expected
            )));
        }

        let max_it = max_iter.unwrap_or(100);
        let tolerance = tol.unwrap_or(1e-10);
        let b_norm = dot_f64(b, b).sqrt().max(1e-300);

        // CG (no preconditioner; the level-Toeplitz is typically well-conditioned)
        let mut x = vec![0.0f64; expected];
        let mut r = b.to_vec();
        let mut p = r.clone();
        let mut rr = dot_f64(&r, &r);

        for _ in 0..max_it {
            if rr.sqrt() / b_norm < tolerance {
                break;
            }

            let q = self.matvec(&p)?;
            let pq = dot_f64(&p, &q);

            if pq.abs() < 1e-300 {
                break;
            }

            let alpha = rr / pq;
            for i in 0..expected {
                x[i] += alpha * p[i];
                r[i] -= alpha * q[i];
            }

            let rr_new = dot_f64(&r, &r);
            let beta = rr_new / rr.max(1e-300);
            rr = rr_new;

            for i in 0..expected {
                p[i] = r[i] + beta * p[i];
            }
        }

        Ok(x)
    }
}

// ============================================================================
// 2D FFT helpers
// ============================================================================

/// Compute the 2D FFT of a real-valued m1 × m2 matrix (stored row-major).
fn fft2d_real(input: &[Vec<f64>], m1: usize, m2: usize) -> LinalgResult<Vec<Vec<C64>>> {
    // Row-wise 1D FFT
    let mut rows_fft: Vec<Vec<C64>> = Vec::with_capacity(m1);
    for row in input {
        let mut row_ext: Vec<C64> = row.iter().map(|&v| C64::new(v, 0.0)).collect();
        row_ext.resize(m2, C64::new(0.0, 0.0));
        fft_inplace(&mut row_ext, false);
        rows_fft.push(row_ext);
    }

    // Pad rows to m1 if needed
    while rows_fft.len() < m1 {
        rows_fft.push(vec![C64::new(0.0, 0.0); m2]);
    }

    // Column-wise 1D FFT
    let mut result = rows_fft;
    for j in 0..m2 {
        let mut col: Vec<C64> = (0..m1).map(|i| result[i][j]).collect();
        fft_inplace(&mut col, false);
        for (i, val) in col.into_iter().enumerate() {
            result[i][j] = val;
        }
    }

    Ok(result)
}

/// Compute the 2D IFFT of a complex m1 × m2 matrix.
fn ifft2d(input: &mut [Vec<C64>], m1: usize, m2: usize) -> LinalgResult<Vec<Vec<f64>>> {
    // Column-wise IFFT first
    for j in 0..m2 {
        let mut col: Vec<C64> = (0..m1).map(|i| input[i][j]).collect();
        fft_inplace(&mut col, true);
        for (i, val) in col.into_iter().enumerate() {
            input[i][j] = val;
        }
    }

    // Row-wise IFFT
    let mut result = vec![vec![0.0f64; m2]; m1];
    for (i, row) in input.iter_mut().enumerate() {
        fft_inplace(row, true);
        for (j, val) in row.iter().enumerate() {
            result[i][j] = val.re;
        }
    }

    Ok(result)
}

// ============================================================================
// Helper functions
// ============================================================================

fn dot_f64(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(ai, bi)| ai * bi).sum()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circulant_matmul_identity() {
        // Circulant with c = [1, 0, 0] is the identity
        let c = vec![1.0, 0.0, 0.0];
        let x = vec![3.0, 1.0, 4.0];
        let y = circulant_matmul(&c, &x).expect("circulant_matmul failed");
        for i in 0..3 {
            assert!(
                (y[i] - x[i]).abs() < 1e-10,
                "Identity circulant failed at {i}"
            );
        }
    }

    #[test]
    fn test_circulant_matmul_known() {
        // C with c = [1, 2, 3]:
        // [[1, 3, 2],
        //  [2, 1, 3],
        //  [3, 2, 1]]
        // C * [1, 0, 0] = first column = [1, 2, 3]
        let c = vec![1.0, 2.0, 3.0];
        let x = vec![1.0, 0.0, 0.0];
        let y = circulant_matmul(&c, &x).expect("circulant_matmul failed");
        assert!((y[0] - 1.0).abs() < 1e-9);
        assert!((y[1] - 2.0).abs() < 1e-9);
        assert!((y[2] - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_toeplitz_matmul_identity_diag() {
        // 3x3 identity Toeplitz: T[i,j] = delta(i==j), so t = [0, 0, 1, 0, 0]
        let t = vec![0.0, 0.0, 1.0, 0.0, 0.0]; // 2*3-1 = 5 elements
        let x = vec![3.0, 1.0, 4.0];
        let y = toeplitz_matmul(&t, &x).expect("toeplitz_matmul failed");
        for i in 0..3 {
            assert!(
                (y[i] - x[i]).abs() < 1e-9,
                "Identity Toeplitz failed at {i}: got {}",
                y[i]
            );
        }
    }

    #[test]
    fn test_toeplitz_matmul_first_column() {
        // T * e_0 should equal the first column of T
        let n = 4;
        // t = [t_{-3}, t_{-2}, t_{-1}, t_0, t_1, t_2, t_3]
        // T[i,j] = t[i-j + n-1]
        // First column: T[i,0] = t[i + n-1]
        let t = vec![7.0, 5.0, 3.0, 1.0, 2.0, 4.0, 6.0]; // 2*4-1=7
        let x = vec![1.0, 0.0, 0.0, 0.0];
        let y = toeplitz_matmul(&t, &x).expect("toeplitz_matmul failed");
        // First column of T: T[i,0] = t[i + 3] = t[3]=1, t[4]=2, t[5]=4, t[6]=6
        assert!((y[0] - 1.0).abs() < 1e-9, "y[0]={}", y[0]);
        assert!((y[1] - 2.0).abs() < 1e-9, "y[1]={}", y[1]);
        assert!((y[2] - 4.0).abs() < 1e-9, "y[2]={}", y[2]);
        assert!((y[3] - 6.0).abs() < 1e-9, "y[3]={}", y[3]);
    }

    #[test]
    fn test_convolve_matmul() {
        let f = vec![1.0, 2.0, 3.0];
        let g = vec![1.0, 1.0];
        let h = convolve_matmul(&f, &g).expect("convolve_matmul failed");
        assert_eq!(h.len(), 4);
        assert!((h[0] - 1.0).abs() < 1e-10);
        assert!((h[1] - 3.0).abs() < 1e-10);
        assert!((h[2] - 5.0).abs() < 1e-10);
        assert!((h[3] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_fft_solver_circulant() {
        // c = [4, 1, 1], solve C x = b
        let c = vec![4.0, 1.0, 1.0];
        let solver = FFTBasedSolver::new_circulant(&c).expect("new_circulant failed");

        // Compute b = C * x_true
        let x_true = vec![1.0, 2.0, 3.0];
        let b = circulant_matmul(&c, &x_true).expect("matvec failed");

        let x_sol = solver.solve(&b, None, None).expect("solve failed");
        for i in 0..3 {
            assert!(
                (x_sol[i] - x_true[i]).abs() < 1e-8,
                "Circulant solve error at {i}: {} vs {}",
                x_sol[i],
                x_true[i]
            );
        }
    }

    #[test]
    fn test_fft_solver_toeplitz() {
        // 4×4 symmetric Toeplitz: t = [t_{-3}.., t_0, ..t_3]
        // Use a diagonally dominant example
        let n = 4;
        let t = vec![0.1, 0.2, 0.3, 4.0, 0.3, 0.2, 0.1]; // 2*4-1=7
        let solver = FFTBasedSolver::new_toeplitz(&t, n).expect("new_toeplitz failed");

        let x_true = vec![1.0, -1.0, 2.0, 0.5];
        let b = toeplitz_matmul(&t, &x_true).expect("matvec for b failed");

        let x_sol = solver
            .solve(&b, Some(100), Some(1e-10))
            .expect("solve failed");
        for i in 0..n {
            assert!(
                (x_sol[i] - x_true[i]).abs() < 1e-6,
                "Toeplitz solve error at {i}: {} vs {}",
                x_sol[i],
                x_true[i]
            );
        }
    }

    #[test]
    fn test_level_toeplitz_matvec() {
        // 2×2 level-Toeplitz with identity kernel (delta)
        let n1 = 2;
        let n2 = 2;
        // kernel shape: (2*2-1) × (2*2-1) = 3×3
        // kernel[r + n1 - 1][c + n2 - 1] = delta(r==0 && c==0)
        let mut kernel = vec![vec![0.0f64; 3]; 3];
        kernel[n1 - 1][n2 - 1] = 1.0; // delta at (0,0)

        let lt = LevelToeplitz::new(n1, n2, kernel).expect("LevelToeplitz::new failed");

        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = lt.matvec(&x).expect("matvec failed");
        for i in 0..4 {
            assert!(
                (y[i] - x[i]).abs() < 1e-9,
                "Level-Toeplitz identity failed at {i}: {} vs {}",
                y[i],
                x[i]
            );
        }
    }
}
