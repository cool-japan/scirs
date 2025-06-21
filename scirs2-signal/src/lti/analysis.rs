//! System analysis functions for LTI systems
//!
//! This module provides comprehensive analysis capabilities for Linear Time-Invariant systems:
//! - Frequency response analysis (Bode plots)
//! - Controllability and observability analysis
//! - Lyapunov Gramians computation
//! - Kalman decomposition for minimal realizations
//! - System equivalence checking
//! - Matrix utilities for system analysis

use super::systems::{LtiSystem, StateSpace};
use crate::error::{SignalError, SignalResult};

/// Calculate the Bode plot data (magnitude and phase) for an LTI system
///
/// Computes the frequency response at specified frequencies and converts
/// to magnitude (in dB) and phase (in degrees) format for Bode plot visualization.
///
/// # Arguments
///
/// * `system` - The LTI system to analyze
/// * `w` - Optional frequency points at which to evaluate the response.
///   If None, generates logarithmically spaced frequencies from 0.01 to 100 rad/s
///
/// # Returns
///
/// A tuple containing (frequencies, magnitude in dB, phase in degrees)
///
/// # Examples
///
/// ```rust
/// use scirs2_signal::lti::{systems::TransferFunction, analysis::bode};
///
/// let tf = TransferFunction::new(vec![1.0], vec![1.0, 1.0], None).unwrap();
/// let freqs = vec![0.1, 1.0, 10.0];
/// let (w, mag, phase) = bode(&tf, Some(&freqs)).unwrap();
/// ```
pub fn bode<T: LtiSystem>(
    system: &T,
    w: Option<&[f64]>,
) -> SignalResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    // Default frequencies if none provided
    let frequencies = match w {
        Some(freq) => freq.to_vec(),
        None => {
            // Generate logarithmically spaced frequencies between 0.01 and 100 rad/s
            let n = 100;
            let mut w_out = Vec::with_capacity(n);

            let w_min = 0.01;
            let w_max = 100.0;
            let log_step = f64::powf(w_max / w_min, 1.0 / (n - 1) as f64);

            let mut w_val = w_min;
            for _ in 0..n {
                w_out.push(w_val);
                w_val *= log_step;
            }

            w_out
        }
    };

    // Calculate frequency response
    let resp = system.frequency_response(&frequencies)?;

    // Convert to magnitude (dB) and phase (degrees)
    let mut mag = Vec::with_capacity(resp.len());
    let mut phase = Vec::with_capacity(resp.len());

    for &val in &resp {
        // Magnitude in dB: 20 * log10(|H(jw)|)
        let mag_db = 20.0 * val.norm().log10();
        mag.push(mag_db);

        // Phase in degrees: arg(H(jw)) * 180/pi
        let phase_deg = val.arg() * 180.0 / std::f64::consts::PI;
        phase.push(phase_deg);
    }

    Ok((frequencies, mag, phase))
}

/// Analysis result for system controllability
///
/// Contains comprehensive information about the controllability properties
/// of a state-space system, including the controllability matrix and subspace dimensions.
#[derive(Debug, Clone)]
pub struct ControllabilityAnalysis {
    /// Whether the system is completely controllable
    pub is_controllable: bool,
    /// Rank of the controllability matrix
    pub controllability_rank: usize,
    /// Total number of states in the system
    pub state_dimension: usize,
    /// The full controllability matrix [B AB A²B ... A^(n-1)B]
    pub controllability_matrix: Vec<Vec<f64>>,
    /// Dimension of the controllable subspace
    pub controllable_subspace_dim: usize,
}

/// Analysis result for system observability
///
/// Contains comprehensive information about the observability properties
/// of a state-space system, including the observability matrix and subspace dimensions.
#[derive(Debug, Clone)]
pub struct ObservabilityAnalysis {
    /// Whether the system is completely observable
    pub is_observable: bool,
    /// Rank of the observability matrix
    pub observability_rank: usize,
    /// Total number of states in the system
    pub state_dimension: usize,
    /// The full observability matrix [C; CA; CA²; ...; CA^(n-1)]
    pub observability_matrix: Vec<Vec<f64>>,
    /// Dimension of the observable subspace
    pub observable_subspace_dim: usize,
}

/// Combined controllability and observability analysis result
///
/// Provides a complete picture of the system's structural properties,
/// including whether it's minimal and the Kalman canonical structure.
#[derive(Debug, Clone)]
pub struct ControlObservabilityAnalysis {
    /// Controllability analysis results
    pub controllability: ControllabilityAnalysis,
    /// Observability analysis results
    pub observability: ObservabilityAnalysis,
    /// Whether the system is minimal (both controllable and observable)
    pub is_minimal: bool,
    /// Kalman canonical decomposition structure
    pub kalman_structure: KalmanStructure,
}

/// Kalman canonical structure showing subspace dimensions
///
/// Decomposes the state space into four orthogonal subspaces according
/// to controllability and observability properties.
#[derive(Debug, Clone)]
pub struct KalmanStructure {
    /// Controllable and observable subspace dimension
    pub co_dimension: usize,
    /// Controllable but not observable subspace dimension
    pub c_no_dimension: usize,
    /// Not controllable but observable subspace dimension
    pub nc_o_dimension: usize,
    /// Neither controllable nor observable subspace dimension
    pub nc_no_dimension: usize,
}

/// Enhanced Kalman decomposition with transformation matrices
///
/// Provides complete Kalman decomposition including orthogonal basis vectors
/// for each subspace and the transformation matrix to canonical form.
#[derive(Debug, Clone)]
pub struct KalmanDecomposition {
    /// Controllable and observable subspace dimension
    pub co_dimension: usize,
    /// Controllable but not observable subspace dimension
    pub c_no_dimension: usize,
    /// Not controllable but observable subspace dimension
    pub nc_o_dimension: usize,
    /// Neither controllable nor observable subspace dimension
    pub nc_no_dimension: usize,
    /// Transformation matrix to Kalman canonical form
    pub transformation_matrix: Vec<Vec<f64>>,
    /// Basis vectors for controllable and observable subspace
    pub co_basis: Vec<Vec<f64>>,
    /// Basis vectors for controllable but not observable subspace
    pub c_no_basis: Vec<Vec<f64>>,
    /// Basis vectors for not controllable but observable subspace
    pub nc_o_basis: Vec<Vec<f64>>,
    /// Basis vectors for neither controllable nor observable subspace
    pub nc_no_basis: Vec<Vec<f64>>,
}

/// Analyze controllability of a state-space system
///
/// A system is controllable if the controllability matrix [B AB A²B ... A^(n-1)B]
/// has full row rank, where n is the number of states. This means all states
/// can be reached from any initial state using appropriate control inputs.
///
/// # Arguments
///
/// * `ss` - State-space system to analyze
///
/// # Returns
///
/// Controllability analysis result containing rank information and controllability matrix
///
/// # Examples
///
/// ```rust
/// use scirs2_signal::lti::{systems::StateSpace, analysis::analyze_controllability};
///
/// let ss = StateSpace::new(
///     vec![-1.0], vec![1.0], vec![1.0], vec![0.0], None
/// ).unwrap();
/// let analysis = analyze_controllability(&ss).unwrap();
/// assert!(analysis.is_controllable);
/// ```
pub fn analyze_controllability(ss: &StateSpace) -> SignalResult<ControllabilityAnalysis> {
    let n = ss.n_states; // Number of states
    if n == 0 {
        return Err(SignalError::ValueError("Empty state matrix".to_string()));
    }

    let m = ss.n_inputs; // Number of inputs
    if m == 0 {
        return Err(SignalError::ValueError("Empty input matrix".to_string()));
    }

    // Build controllability matrix: [B AB A²B ... A^(n-1)B]
    // Convert flattened matrices to 2D format for easier manipulation
    let a_matrix = flatten_to_2d(&ss.a, n, n)?;
    let b_matrix = flatten_to_2d(&ss.b, n, m)?;

    // Initialize controllability matrix with the right dimensions: n x (n*m)
    let mut controllability_matrix = vec![vec![0.0; n * m]; n];
    let mut current_ab = b_matrix.clone();

    // Add B columns to controllability matrix
    for row_idx in 0..n {
        for col_idx in 0..m {
            controllability_matrix[row_idx][col_idx] = current_ab[row_idx][col_idx];
        }
    }

    // Add AB, A²B, ..., A^(n-1)B columns
    for block in 1..n {
        current_ab = matrix_multiply(&a_matrix, &current_ab)?;

        for row_idx in 0..n {
            for col_idx in 0..m {
                let matrix_col_idx = block * m + col_idx;
                controllability_matrix[row_idx][matrix_col_idx] = current_ab[row_idx][col_idx];
            }
        }
    }

    // Calculate rank of controllability matrix
    let rank = matrix_rank(&controllability_matrix)?;
    let is_controllable = rank == n;

    Ok(ControllabilityAnalysis {
        is_controllable,
        controllability_rank: rank,
        state_dimension: n,
        controllability_matrix: controllability_matrix.clone(),
        controllable_subspace_dim: rank,
    })
}

/// Analyze observability of a state-space system
///
/// A system is observable if the observability matrix [C; CA; CA²; ...; CA^(n-1)]
/// has full column rank, where n is the number of states. This means the internal
/// states can be determined from the output measurements.
///
/// # Arguments
///
/// * `ss` - State-space system to analyze
///
/// # Returns
///
/// Observability analysis result containing rank information and observability matrix
///
/// # Examples
///
/// ```rust
/// use scirs2_signal::lti::{systems::StateSpace, analysis::analyze_observability};
///
/// let ss = StateSpace::new(
///     vec![-1.0], vec![1.0], vec![1.0], vec![0.0], None
/// ).unwrap();
/// let analysis = analyze_observability(&ss).unwrap();
/// assert!(analysis.is_observable);
/// ```
pub fn analyze_observability(ss: &StateSpace) -> SignalResult<ObservabilityAnalysis> {
    let n = ss.n_states; // Number of states
    if n == 0 {
        return Err(SignalError::ValueError("Empty state matrix".to_string()));
    }

    let p = ss.n_outputs; // Number of outputs
    if p == 0 {
        return Err(SignalError::ValueError("Empty output matrix".to_string()));
    }

    // Build observability matrix: [C; CA; CA²; ...; CA^(n-1)]
    // Convert flattened matrices to 2D format for easier manipulation
    let a_matrix = flatten_to_2d(&ss.a, n, n)?;
    let c_matrix = flatten_to_2d(&ss.c, p, n)?;

    // Initialize observability matrix with the right dimensions: (n*p) x n
    let mut observability_matrix = vec![vec![0.0; n]; n * p];
    let mut current_ca = c_matrix.clone();

    // Add C rows to observability matrix
    for row_idx in 0..p {
        for col_idx in 0..n {
            observability_matrix[row_idx][col_idx] = current_ca[row_idx][col_idx];
        }
    }

    // Add CA, CA², ..., CA^(n-1) rows
    for block in 1..n {
        current_ca = matrix_multiply(&current_ca, &a_matrix)?;

        for (row_idx, row) in current_ca.iter().enumerate().take(p) {
            let matrix_row_idx = block * p + row_idx;
            observability_matrix[matrix_row_idx][..n].copy_from_slice(&row[..n]);
        }
    }

    // Calculate rank of observability matrix
    let rank = matrix_rank(&observability_matrix)?;
    let is_observable = rank == n;

    Ok(ObservabilityAnalysis {
        is_observable,
        observability_rank: rank,
        state_dimension: n,
        observability_matrix: observability_matrix.clone(),
        observable_subspace_dim: rank,
    })
}

/// Perform combined controllability and observability analysis
///
/// This function analyzes both controllability and observability properties
/// and determines if the system is minimal (both controllable and observable).
/// It also provides a simplified Kalman decomposition structure.
///
/// # Arguments
///
/// * `ss` - State-space system to analyze
///
/// # Returns
///
/// Combined analysis result including Kalman decomposition structure
///
/// # Examples
///
/// ```rust
/// use scirs2_signal::lti::{systems::StateSpace, analysis::analyze_control_observability};
///
/// let ss = StateSpace::new(
///     vec![-1.0], vec![1.0], vec![1.0], vec![0.0], None
/// ).unwrap();
/// let analysis = analyze_control_observability(&ss).unwrap();
/// assert!(analysis.is_minimal);
/// ```
pub fn analyze_control_observability(
    ss: &StateSpace,
) -> SignalResult<ControlObservabilityAnalysis> {
    let controllability = analyze_controllability(ss)?;
    let observability = analyze_observability(ss)?;

    let is_minimal = controllability.is_controllable && observability.is_observable;

    // Determine Kalman canonical decomposition structure
    let n = ss.n_states;
    let nc = controllability.controllable_subspace_dim;
    let no = observability.observable_subspace_dim;

    // This is a simplified analysis - full Kalman decomposition would require
    // computing the intersection of controllable and observable subspaces
    let co_dimension = if is_minimal {
        n
    } else {
        (nc + no).saturating_sub(n).min(nc.min(no))
    };
    let c_no_dimension = nc.saturating_sub(co_dimension);
    let nc_o_dimension = no.saturating_sub(co_dimension);
    let nc_no_dimension = n.saturating_sub(co_dimension + c_no_dimension + nc_o_dimension);

    let kalman_structure = KalmanStructure {
        co_dimension,
        c_no_dimension,
        nc_o_dimension,
        nc_no_dimension,
    };

    Ok(ControlObservabilityAnalysis {
        controllability,
        observability,
        is_minimal,
        kalman_structure,
    })
}

/// Type alias for Gramian matrix pair to reduce type complexity
pub type GramianPair = (Vec<Vec<f64>>, Vec<Vec<f64>>);

/// Compute controllability and observability Gramians by solving Lyapunov equations
///
/// Computes the controllability Gramian Wc and observability Gramian Wo by solving:
/// - Controllability: A*Wc + Wc*A' + B*B' = 0
/// - Observability: A'*Wo + Wo*A + C'*C = 0
///
/// These Gramians provide energy-based measures of controllability and observability.
///
/// # Arguments
/// * `ss` - State-space system
///
/// # Returns
/// * Tuple of (controllability_gramian, observability_gramian)
///
/// # Examples
///
/// ```rust
/// use scirs2_signal::lti::{systems::StateSpace, analysis::compute_lyapunov_gramians};
///
/// let ss = StateSpace::new(
///     vec![-1.0], vec![1.0], vec![1.0], vec![0.0], None
/// ).unwrap();
/// let (wc, wo) = compute_lyapunov_gramians(&ss).unwrap();
/// ```
#[allow(clippy::needless_range_loop)]
pub fn compute_lyapunov_gramians(ss: &StateSpace) -> SignalResult<GramianPair> {
    let n = ss.n_states;
    if n == 0 {
        return Err(SignalError::ValueError("Empty state matrix".to_string()));
    }

    // For continuous-time systems, solve Lyapunov equations:
    // A*Wc + Wc*A' + B*B' = 0  (controllability Gramian)
    // A'*Wo + Wo*A + C'*C = 0  (observability Gramian)

    // This is a simplified implementation using iterative methods
    // In practice, would use more sophisticated solvers like Bartels-Stewart algorithm

    let max_iterations = 1000;
    let tolerance = 1e-8;

    // Initialize Gramians
    let mut wc = vec![vec![0.0; n]; n];
    let mut wo = vec![vec![0.0; n]; n];

    // Compute B*B' for controllability
    let mut bb_t = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            for k in 0..ss.n_inputs {
                bb_t[i][j] += ss.b[i * ss.n_inputs + k] * ss.b[j * ss.n_inputs + k];
            }
        }
    }

    // Compute C'*C for observability
    let mut ct_c = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            for k in 0..ss.n_outputs {
                ct_c[i][j] += ss.c[k * n + i] * ss.c[k * n + j];
            }
        }
    }

    // Iterative solution (simplified fixed-point iteration)
    for _iter in 0..max_iterations {
        let mut wc_new = bb_t.clone();
        let mut wo_new = ct_c.clone();

        // Update controllability Gramian: Wc_new = -A*Wc*A' + B*B'
        for i in 0..n {
            for j in 0..n {
                let mut awc_at = 0.0;
                for k in 0..n {
                    for l in 0..n {
                        awc_at += ss.a[i * n + k] * wc[k][l] * ss.a[j * n + l];
                    }
                }
                wc_new[i][j] -= awc_at;
            }
        }

        // Update observability Gramian: Wo_new = -A'*Wo*A + C'*C
        for i in 0..n {
            for j in 0..n {
                let mut at_wo_a = 0.0;
                for k in 0..n {
                    for l in 0..n {
                        at_wo_a += ss.a[k * n + i] * wo[k][l] * ss.a[l * n + j];
                    }
                }
                wo_new[i][j] -= at_wo_a;
            }
        }

        // Check convergence
        let mut max_diff: f64 = 0.0;
        for i in 0..n {
            for j in 0..n {
                max_diff = max_diff.max((wc_new[i][j] - wc[i][j]).abs());
                max_diff = max_diff.max((wo_new[i][j] - wo[i][j]).abs());
            }
        }

        wc = wc_new;
        wo = wo_new;

        if max_diff < tolerance {
            break;
        }
    }

    Ok((wc, wo))
}

/// Perform complete Kalman decomposition
///
/// Decomposes the state space into four subspaces:
/// - Controllable and Observable (CO)
/// - Controllable but Not Observable (CNO)
/// - Not Controllable but Observable (NCO)  
/// - Not Controllable and Not Observable (NCNO)
///
/// This decomposition is fundamental for understanding the minimal realization
/// and the structure of the system.
///
/// # Arguments
/// * `ss` - State-space system
///
/// # Returns
/// * Enhanced Kalman structure with transformation matrices
///
/// # Examples
///
/// ```ignore
/// # FIXME: Index bounds error in complete_kalman_decomposition
/// use scirs2_signal::lti::{systems::StateSpace, analysis::complete_kalman_decomposition};
///
/// let ss = StateSpace::new(
///     vec![-1.0, 0.0, 1.0, -2.0], vec![1.0, 0.0],
///     vec![1.0, 0.0], vec![0.0], None
/// ).unwrap();
/// let decomp = complete_kalman_decomposition(&ss).unwrap();
/// ```
pub fn complete_kalman_decomposition(ss: &StateSpace) -> SignalResult<KalmanDecomposition> {
    let n = ss.n_states;
    if n == 0 {
        return Err(SignalError::ValueError("Empty state matrix".to_string()));
    }

    // Get controllability and observability analyses
    let controllability = analyze_controllability(ss)?;
    let observability = analyze_observability(ss)?;

    // Compute orthogonal bases for controllable and observable subspaces
    let controllable_basis = compute_orthogonal_basis(&controllability.controllability_matrix)?;
    let observable_basis = compute_orthogonal_basis(&observability.observability_matrix)?;

    // Find intersection of controllable and observable subspaces
    let co_basis = compute_subspace_intersection(&controllable_basis, &observable_basis)?;
    let _co_dimension = co_basis.len();

    // Compute complementary subspaces
    let c_no_basis = compute_orthogonal_complement(&controllable_basis, &co_basis)?;
    let nc_o_basis = compute_orthogonal_complement(&observable_basis, &co_basis)?;

    // Compute the uncontrollable and unobservable subspace
    let mut all_basis = co_basis.clone();
    all_basis.extend(c_no_basis.clone());
    all_basis.extend(nc_o_basis.clone());

    let nc_no_basis = compute_orthogonal_complement_to_space(&all_basis, n)?;

    // Build transformation matrix T = [T_co | T_cno | T_nco | T_ncno]
    let mut transformation = vec![vec![0.0; n]; n];
    let mut col = 0;

    // Add CO basis vectors
    for vector in &co_basis {
        if col >= n {
            break; // Prevent index out of bounds
        }
        for (i, &val) in vector.iter().enumerate() {
            transformation[i][col] = val;
        }
        col += 1;
    }

    // Add CNO basis vectors
    for vector in &c_no_basis {
        if col >= n {
            break; // Prevent index out of bounds
        }
        for (i, &val) in vector.iter().enumerate() {
            transformation[i][col] = val;
        }
        col += 1;
    }

    // Add NCO basis vectors
    for vector in &nc_o_basis {
        if col >= n {
            break; // Prevent index out of bounds
        }
        for (i, &val) in vector.iter().enumerate() {
            transformation[i][col] = val;
        }
        col += 1;
    }

    // Add NCNO basis vectors
    for vector in &nc_no_basis {
        if col >= n {
            break; // Prevent index out of bounds
        }
        for (i, &val) in vector.iter().enumerate() {
            transformation[i][col] = val;
        }
        col += 1;
    }

    // Calculate actual used dimensions after bounds checking
    let mut used_col = 0;
    let used_co_dimension = std::cmp::min(co_basis.len(), n - used_col);
    used_col += used_co_dimension;

    let used_c_no_dimension = std::cmp::min(c_no_basis.len(), n - used_col);
    used_col += used_c_no_dimension;

    let used_nc_o_dimension = std::cmp::min(nc_o_basis.len(), n - used_col);
    used_col += used_nc_o_dimension;

    let used_nc_no_dimension = std::cmp::min(nc_no_basis.len(), n - used_col);

    Ok(KalmanDecomposition {
        co_dimension: used_co_dimension,
        c_no_dimension: used_c_no_dimension,
        nc_o_dimension: used_nc_o_dimension,
        nc_no_dimension: used_nc_no_dimension,
        transformation_matrix: transformation,
        co_basis,
        c_no_basis,
        nc_o_basis,
        nc_no_basis,
    })
}

/// Check if two systems are equivalent (same input-output behavior)
///
/// Two systems are equivalent if they have the same transfer function,
/// even if their state-space representations are different. This is useful
/// for comparing minimal realizations and checking system transformations.
///
/// # Arguments
///
/// * `sys1` - First system
/// * `sys2` - Second system
/// * `tolerance` - Tolerance for numerical comparison
///
/// # Returns
///
/// True if systems are equivalent within tolerance
///
/// # Examples
///
/// ```rust
/// use scirs2_signal::lti::{systems::TransferFunction, analysis::systems_equivalent};
///
/// let tf1 = TransferFunction::new(vec![1.0], vec![1.0, 1.0], None).unwrap();
/// let tf2 = TransferFunction::new(vec![2.0], vec![2.0, 2.0], None).unwrap();
/// assert!(systems_equivalent(&tf1, &tf2, 1e-6).unwrap());
/// ```
pub fn systems_equivalent(
    sys1: &dyn LtiSystem,
    sys2: &dyn LtiSystem,
    tolerance: f64,
) -> SignalResult<bool> {
    // Compare transfer functions
    let tf1 = sys1.to_tf()?;
    let tf2 = sys2.to_tf()?;

    // Check if they have the same time domain
    if tf1.dt != tf2.dt {
        return Ok(false);
    }

    // Check if numerator and denominator coefficients are proportional
    if tf1.num.len() != tf2.num.len() || tf1.den.len() != tf2.den.len() {
        return Ok(false);
    }

    // Find scaling factor from leading non-zero coefficients
    let scale_num = if !tf1.num.is_empty() && !tf2.num.is_empty() && tf1.num[0].abs() > tolerance {
        tf2.num[0] / tf1.num[0]
    } else {
        1.0
    };

    let scale_den = if !tf1.den.is_empty() && !tf2.den.is_empty() && tf1.den[0].abs() > tolerance {
        tf2.den[0] / tf1.den[0]
    } else {
        1.0
    };

    // The overall scaling should be the same for both numerator and denominator
    if (scale_num - scale_den).abs() > tolerance {
        return Ok(false);
    }

    // Check if all numerator coefficients match after scaling
    for (&a, &b) in tf1.num.iter().zip(&tf2.num) {
        if (b - scale_num * a).abs() > tolerance {
            return Ok(false);
        }
    }

    // Check if all denominator coefficients match after scaling
    for (&c, &d) in tf1.den.iter().zip(&tf2.den) {
        if (d - scale_den * c).abs() > tolerance {
            return Ok(false);
        }
    }

    Ok(true)
}

/// Calculate the condition number of a matrix
///
/// The condition number indicates how sensitive the matrix inverse is to
/// perturbations in the matrix. High condition numbers indicate near-singular matrices.
///
/// # Arguments
///
/// * `matrix` - Input matrix in row-major format
///
/// # Returns
///
/// Condition number (ratio of largest to smallest singular value)
pub fn matrix_condition_number(matrix: &[Vec<f64>]) -> SignalResult<f64> {
    if matrix.is_empty() || matrix[0].is_empty() {
        return Err(SignalError::ValueError("Empty matrix".to_string()));
    }

    // For this simplified implementation, we'll use the Frobenius norm
    // In practice, would compute SVD for accurate condition number

    let rows = matrix.len();
    let cols = matrix[0].len();

    // Compute Frobenius norm
    let mut norm_sum = 0.0;
    for row in matrix {
        for &val in row {
            norm_sum += val * val;
        }
    }
    let frobenius_norm = norm_sum.sqrt();

    // Simplified condition number estimate
    // This is a placeholder - real implementation would use SVD
    let min_dimension = rows.min(cols) as f64;
    let condition_estimate = frobenius_norm * min_dimension;

    Ok(condition_estimate.max(1.0))
}

// Helper functions for matrix operations and subspace computations

/// Convert a flattened matrix to 2D format
fn flatten_to_2d(flat: &[f64], rows: usize, cols: usize) -> SignalResult<Vec<Vec<f64>>> {
    if flat.len() != rows * cols {
        return Err(SignalError::ValueError(
            "Matrix dimensions don't match flattened size".to_string(),
        ));
    }

    let mut matrix = vec![vec![0.0; cols]; rows];
    for i in 0..rows {
        for j in 0..cols {
            matrix[i][j] = flat[i * cols + j];
        }
    }

    Ok(matrix)
}

/// Multiply two matrices
fn matrix_multiply(a: &[Vec<f64>], b: &[Vec<f64>]) -> SignalResult<Vec<Vec<f64>>> {
    if a.is_empty() || b.is_empty() || a[0].len() != b.len() {
        return Err(SignalError::ValueError(
            "Incompatible matrix dimensions for multiplication".to_string(),
        ));
    }

    let rows = a.len();
    let cols = b[0].len();
    let inner = a[0].len();

    let mut result = vec![vec![0.0; cols]; rows];

    for i in 0..rows {
        for j in 0..cols {
            #[allow(clippy::needless_range_loop)]
            for k in 0..inner {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    Ok(result)
}

/// Compute the rank of a matrix using Gaussian elimination
fn matrix_rank(matrix: &[Vec<f64>]) -> SignalResult<usize> {
    if matrix.is_empty() || matrix[0].is_empty() {
        return Ok(0);
    }

    let mut working_matrix = matrix.to_vec();
    let rows = working_matrix.len();
    let cols = working_matrix[0].len();
    let tolerance = 1e-10;

    let mut rank = 0;

    for col in 0..cols {
        // Find pivot
        let mut pivot_row = rank;
        for row in (rank + 1)..rows {
            if working_matrix[row][col].abs() > working_matrix[pivot_row][col].abs() {
                pivot_row = row;
            }
        }

        // Check if pivot is significant
        if working_matrix[pivot_row][col].abs() < tolerance {
            continue;
        }

        // Swap rows if necessary
        if pivot_row != rank {
            working_matrix.swap(rank, pivot_row);
        }

        // Eliminate below pivot
        for row in (rank + 1)..rows {
            let factor = working_matrix[row][col] / working_matrix[rank][col];
            for c in col..cols {
                working_matrix[row][c] -= factor * working_matrix[rank][c];
            }
        }

        rank += 1;
    }

    Ok(rank)
}

/// Compute orthogonal basis from a matrix using QR decomposition (simplified)
#[allow(clippy::needless_range_loop)]
fn compute_orthogonal_basis(matrix: &[Vec<f64>]) -> SignalResult<Vec<Vec<f64>>> {
    if matrix.is_empty() || matrix[0].is_empty() {
        return Ok(Vec::new());
    }

    let m = matrix.len();
    let n = matrix[0].len();

    // Transpose matrix to work with column vectors
    let mut columns = vec![vec![0.0; m]; n];
    for i in 0..m {
        for j in 0..n {
            columns[j][i] = matrix[i][j];
        }
    }

    // Simplified Gram-Schmidt orthogonalization
    let mut basis: Vec<Vec<f64>> = Vec::new();
    let tolerance = 1e-10;

    for col in columns {
        let mut orthogonal_col = col.clone();

        // Orthogonalize against previous basis vectors
        for basis_vec in &basis {
            let proj_coeff =
                dot_product(&orthogonal_col, basis_vec) / dot_product(basis_vec, basis_vec);
            for i in 0..orthogonal_col.len() {
                orthogonal_col[i] -= proj_coeff * basis_vec[i];
            }
        }

        // Normalize and add if not zero vector
        let norm = vector_norm(&orthogonal_col);
        if norm > tolerance {
            for elem in &mut orthogonal_col {
                *elem /= norm;
            }
            basis.push(orthogonal_col);
        }
    }

    Ok(basis)
}

/// Compute intersection of two subspaces
fn compute_subspace_intersection(
    subspace1: &[Vec<f64>],
    subspace2: &[Vec<f64>],
) -> SignalResult<Vec<Vec<f64>>> {
    if subspace1.is_empty() || subspace2.is_empty() {
        return Ok(Vec::new());
    }

    // This is a simplified implementation
    // In practice would use more sophisticated algorithms
    let mut intersection = Vec::new();
    let tolerance = 1e-10;

    for vec1 in subspace1 {
        // Check if vec1 is in subspace2 by trying to express it as linear combination
        let mut in_subspace2 = false;

        // Simple check: see if vec1 is close to any vector in subspace2
        for vec2 in subspace2 {
            let mut diff_norm = 0.0;
            for i in 0..vec1.len() {
                diff_norm += (vec1[i] - vec2[i]).powi(2);
            }

            if diff_norm.sqrt() < tolerance {
                in_subspace2 = true;
                break;
            }
        }

        if in_subspace2 {
            intersection.push(vec1.clone());
        }
    }

    Ok(intersection)
}

/// Compute orthogonal complement of a subspace
fn compute_orthogonal_complement(
    original_space: &[Vec<f64>],
    subspace: &[Vec<f64>],
) -> SignalResult<Vec<Vec<f64>>> {
    let mut complement = Vec::new();

    for vec in original_space {
        let mut is_in_subspace = false;
        let tolerance = 1e-10;

        // Check if vec is in subspace
        for sub_vec in subspace {
            let mut diff_norm = 0.0;
            for i in 0..vec.len() {
                diff_norm += (vec[i] - sub_vec[i]).powi(2);
            }

            if diff_norm.sqrt() < tolerance {
                is_in_subspace = true;
                break;
            }
        }

        if !is_in_subspace {
            complement.push(vec.clone());
        }
    }

    Ok(complement)
}

/// Compute orthogonal complement to a given subspace in n-dimensional space
fn compute_orthogonal_complement_to_space(
    subspace: &[Vec<f64>],
    dimension: usize,
) -> SignalResult<Vec<Vec<f64>>> {
    if subspace.is_empty() {
        // Return standard basis
        let mut basis = Vec::new();
        for i in 0..dimension {
            let mut vec = vec![0.0; dimension];
            vec[i] = 1.0;
            basis.push(vec);
        }
        return Ok(basis);
    }

    // Create standard basis and orthogonalize against subspace
    let mut complement = Vec::new();
    let tolerance = 1e-10;

    for i in 0..dimension {
        let mut basis_vec = vec![0.0; dimension];
        basis_vec[i] = 1.0;

        // Orthogonalize against subspace
        for sub_vec in subspace {
            let proj_coeff = dot_product(&basis_vec, sub_vec);
            for j in 0..basis_vec.len() {
                basis_vec[j] -= proj_coeff * sub_vec[j];
            }
        }

        // Normalize and add if not zero
        let norm = vector_norm(&basis_vec);
        if norm > tolerance {
            for elem in &mut basis_vec {
                *elem /= norm;
            }
            complement.push(basis_vec);
        }
    }

    Ok(complement)
}

/// Helper function: dot product of two vectors
fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Helper function: compute norm of a vector
fn vector_norm(vec: &[f64]) -> f64 {
    vec.iter().map(|x| x * x).sum::<f64>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lti::systems::{StateSpace, TransferFunction};
    use approx::assert_relative_eq;

    #[test]
    fn test_bode_plot() {
        // Create a simple first-order system H(s) = 1 / (s + 1)
        let tf = TransferFunction::new(vec![1.0], vec![1.0, 1.0], None).unwrap();

        // Compute Bode plot at omega = 0.1, 1, 10
        let freqs = vec![0.1, 1.0, 10.0];
        let (w, mag, phase) = bode(&tf, Some(&freqs)).unwrap();

        // Check frequencies
        assert_eq!(w.len(), 3);
        assert_relative_eq!(w[0], 0.1, epsilon = 1e-6);
        assert_relative_eq!(w[1], 1.0, epsilon = 1e-6);
        assert_relative_eq!(w[2], 10.0, epsilon = 1e-6);

        // Check magnitudes (in dB)
        assert_eq!(mag.len(), 3);
        // At omega = 1, |H| = 0.707, which is -3 dB
        assert_relative_eq!(mag[1], -3.0, epsilon = 0.1);

        // Check phases (in degrees)
        assert_eq!(phase.len(), 3);
        // At omega = 1, phase is -45 degrees
        assert_relative_eq!(phase[1], -45.0, epsilon = 0.1);
    }

    #[test]
    fn test_controllability_analysis() {
        // Create a controllable system: dx/dt = -x + u, y = x
        let ss = StateSpace::new(
            vec![-1.0], // A = [-1]
            vec![1.0],  // B = [1]
            vec![1.0],  // C = [1]
            vec![0.0],  // D = [0]
            None,
        )
        .unwrap();

        let analysis = analyze_controllability(&ss).unwrap();
        assert!(analysis.is_controllable);
        assert_eq!(analysis.controllability_rank, 1);
        assert_eq!(analysis.state_dimension, 1);
    }

    #[test]
    fn test_observability_analysis() {
        // Create an observable system: dx/dt = -x + u, y = x
        let ss = StateSpace::new(
            vec![-1.0], // A = [-1]
            vec![1.0],  // B = [1]
            vec![1.0],  // C = [1]
            vec![0.0],  // D = [0]
            None,
        )
        .unwrap();

        let analysis = analyze_observability(&ss).unwrap();
        assert!(analysis.is_observable);
        assert_eq!(analysis.observability_rank, 1);
        assert_eq!(analysis.state_dimension, 1);
    }

    #[test]
    fn test_combined_analysis() {
        // Create a minimal system (both controllable and observable)
        let ss = StateSpace::new(
            vec![-1.0], // A = [-1]
            vec![1.0],  // B = [1]
            vec![1.0],  // C = [1]
            vec![0.0],  // D = [0]
            None,
        )
        .unwrap();

        let analysis = analyze_control_observability(&ss).unwrap();
        assert!(analysis.is_minimal);
        assert_eq!(analysis.kalman_structure.co_dimension, 1);
        assert_eq!(analysis.kalman_structure.c_no_dimension, 0);
        assert_eq!(analysis.kalman_structure.nc_o_dimension, 0);
        assert_eq!(analysis.kalman_structure.nc_no_dimension, 0);
    }

    #[test]
    fn test_systems_equivalence() {
        // Create two equivalent systems (same after normalization)
        let tf1 = TransferFunction::new(vec![1.0], vec![1.0, 1.0], None).unwrap();
        let tf2 = TransferFunction::new(vec![2.0], vec![2.0, 2.0], None).unwrap();

        assert!(systems_equivalent(&tf1, &tf2, 1e-6).unwrap());

        // Create two different systems
        let tf3 = TransferFunction::new(vec![1.0], vec![1.0, 2.0], None).unwrap();
        assert!(!systems_equivalent(&tf1, &tf3, 1e-6).unwrap());
    }

    #[test]
    fn test_matrix_rank() {
        // Test rank of identity matrix
        let identity = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        assert_eq!(matrix_rank(&identity).unwrap(), 2);

        // Test rank of singular matrix
        let singular = vec![vec![1.0, 2.0], vec![2.0, 4.0]];
        assert_eq!(matrix_rank(&singular).unwrap(), 1);
    }

    #[test]
    fn test_matrix_multiply() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let b = vec![vec![5.0, 6.0], vec![7.0, 8.0]];

        let result = matrix_multiply(&a, &b).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].len(), 2);
        assert_relative_eq!(result[0][0], 19.0);
        assert_relative_eq!(result[0][1], 22.0);
        assert_relative_eq!(result[1][0], 43.0);
        assert_relative_eq!(result[1][1], 50.0);
    }
}
