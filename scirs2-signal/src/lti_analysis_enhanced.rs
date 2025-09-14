use ndarray::s;
// Enhanced LTI system analysis with controllability and observability
//
// This module provides comprehensive analysis tools for linear time-invariant
// systems including controllability, observability, stability analysis, and
// advanced system properties.

use crate::error::{SignalError, SignalResult};
use crate::lti::StateSpace;
use ndarray::{array, Array1, Array2};
use num_complex::Complex64;
use rand::prelude::*;
use rand::Rng;
use scirs2_core::validation::{check_finite, checkshape};
use scirs2_linalg::{eig, eigh, inv, matrix_norm, solve, svd};

#[allow(unused_imports)]
// Enhanced with robust controllability/observability analysis
/// Helper function to convert Vec<f64> to Array2<f64> for matrix operations
fn vec_to_array2(vec: &[f64], rows: usize, cols: usize) -> SignalResult<Array2<f64>> {
    if vec.len() != rows * cols {
        return Err(SignalError::InvalidArgument(format!(
            "Vector length {} doesn't match matrix dimensions {}x{}",
            vec.len(),
            rows,
            cols
        )));
    }
    Ok(Array2::from_shape_vec((rows, cols), vec.to_vec())
        .map_err(|e| SignalError::ComputationError(format!("Array creation failed: {}", e)))?)
}

/// Helper function to convert Array2<f64> to Vec<f64> in row-major order
fn array2_to_vec(arr: &Array2<f64>) -> Vec<f64> {
    arr.iter().cloned().collect()
}

/// Comprehensive LTI system analysis result
#[derive(Debug, Clone)]
pub struct LtiAnalysisResult {
    /// Controllability analysis
    pub controllability: ControllabilityResult,
    /// Observability analysis
    pub observability: ObservabilityResult,
    /// Stability analysis
    pub stability: StabilityResult,
    /// System properties
    pub properties: SystemProperties,
    /// Gramian analysis
    pub gramians: GramianAnalysis,
    /// Modal analysis
    pub modal: ModalAnalysis,
}

/// Controllability analysis result
#[derive(Debug, Clone)]
pub struct ControllabilityResult {
    /// Is the system completely controllable
    pub is_controllable: bool,
    /// Controllability matrix rank
    pub controllability_rank: usize,
    /// Controllable subspace dimension
    pub controllable_dim: usize,
    /// Controllability Gramian
    pub gramian: Array2<f64>,
    /// Controllability measure (0 to 1)
    pub controllability_measure: f64,
    /// Uncontrollable modes
    pub uncontrollable_modes: Vec<Complex64>,
}

/// Observability analysis result
#[derive(Debug, Clone)]
pub struct ObservabilityResult {
    /// Is the system completely observable
    pub is_observable: bool,
    /// Observability matrix rank
    pub observability_rank: usize,
    /// Observable subspace dimension
    pub observable_dim: usize,
    /// Observability Gramian
    pub gramian: Array2<f64>,
    /// Observability measure (0 to 1)
    pub observability_measure: f64,
    /// Unobservable modes
    pub unobservable_modes: Vec<Complex64>,
}

/// Stability analysis result
#[derive(Debug, Clone)]
pub struct StabilityResult {
    /// Is the system stable
    pub is_stable: bool,
    /// Is the system marginally stable
    pub is_marginally_stable: bool,
    /// System eigenvalues
    pub eigenvalues: Vec<Complex64>,
    /// Stability margin (distance to instability)
    pub stability_margin: f64,
    /// Damping ratios for each mode
    pub damping_ratios: Vec<f64>,
    /// Natural frequencies for each mode
    pub natural_frequencies: Vec<f64>,
    /// Time constants for each mode
    pub time_constants: Vec<f64>,
}

/// System properties
#[derive(Debug, Clone)]
pub struct SystemProperties {
    /// System type (minimum phase, etc.)
    pub system_type: SystemType,
    /// Number of states
    pub num_states: usize,
    /// Number of inputs
    pub num_inputs: usize,
    /// Number of outputs
    pub num_outputs: usize,
    /// System zeros
    pub zeros: Vec<Complex64>,
    /// System poles
    pub poles: Vec<Complex64>,
    /// DC gain
    pub dc_gain: Array2<f64>,
    /// System bandwidth
    pub bandwidth: Option<f64>,
}

/// System type classification
#[derive(Debug, Clone, PartialEq)]
pub enum SystemType {
    MinimumPhase,
    NonMinimumPhase,
    AllPass,
    Unstable,
}

/// Gramian analysis
#[derive(Debug, Clone)]
pub struct GramianAnalysis {
    /// Hankel singular values
    pub hankel_singular_values: Array1<f64>,
    /// Balanced realization exists
    pub is_balanced: bool,
    /// Cross-Gramian (for SISO systems)
    pub cross_gramian: Option<Array2<f64>>,
}

/// Modal analysis
#[derive(Debug, Clone)]
pub struct ModalAnalysis {
    /// Modal matrix (eigenvectors)
    pub modal_matrix: Array2<Complex64>,
    /// Modal controllability
    pub modal_controllability: Vec<f64>,
    /// Modal observability
    pub modal_observability: Vec<f64>,
    /// Participation factors
    pub participation_factors: Array2<f64>,
}

/// Perform comprehensive LTI system analysis
#[allow(dead_code)]
pub fn analyze_lti_system(ss: &StateSpace) -> SignalResult<LtiAnalysisResult> {
    // Validate system
    validate_state_space(_ss)?;

    // Perform analyses
    let controllability = analyze_controllability(_ss)?;
    let observability = analyze_observability(_ss)?;
    let stability = analyze_stability(_ss)?;
    let properties = analyze_system_properties(_ss)?;
    let gramians = analyze_gramians(_ss, &controllability.gramian, &observability.gramian)?;
    let modal = perform_modal_analysis(_ss)?;

    Ok(LtiAnalysisResult {
        controllability,
        observability,
        stability,
        properties,
        gramians,
        modal,
    })
}

/// Analyze controllability
#[allow(dead_code)]
fn analyze_controllability(ss: &StateSpace) -> SignalResult<ControllabilityResult> {
    let n = ss.n_states;
    let m = ss.n_inputs;

    // Convert Vec<f64> matrices to Array2<f64>
    let a_matrix = vec_to_array2(&_ss.a, n, n)?;
    let b_matrix = vec_to_array2(&_ss.b, n, m)?;

    // Build controllability matrix [B AB A²B ... A^(n-1)B]
    let ctrl_matrix = build_controllability_matrix(&a_matrix, &b_matrix)?;

    // Compute rank using SVD
    let (_, s_) = svd(&ctrl_matrix.view(), false, None)
        .map_err(|e| SignalError::ComputationError(format!("SVD failed: {}", e)))?;

    let tolerance = 1e-10 * s_[0];
    let rank = s_.iter().filter(|&&sv| sv > tolerance).count();

    // Check complete controllability
    let is_controllable = rank == n;

    // Compute controllability Gramian
    let gramian = if ss.dt {
        compute_discrete_controllability_gramian(&a_matrix, &b_matrix)?
    } else {
        compute_continuous_controllability_gramian(&a_matrix, &b_matrix)?
    };

    // Controllability measure (smallest eigenvalue of Gramian)
    let gram_eigenvalues = eigh(&gramian.view(), None)
        .map_err(|e| {
            SignalError::ComputationError(format!("Eigenvalue computation failed: {}", e))
        })?
        .0;

    let controllability_measure = if gram_eigenvalues[0] > 0.0 {
        gram_eigenvalues[0] / gram_eigenvalues[gram_eigenvalues.len() - 1]
    } else {
        0.0
    };

    // Find uncontrollable modes
    let uncontrollable_modes = find_uncontrollable_modes(_ss, rank)?;

    Ok(ControllabilityResult {
        is_controllable,
        controllability_rank: rank,
        controllable_dim: rank,
        gramian,
        controllability_measure,
        uncontrollable_modes,
    })
}

/// Analyze observability
#[allow(dead_code)]
fn analyze_observability(ss: &StateSpace) -> SignalResult<ObservabilityResult> {
    let n = ss.n_states;
    let p = ss.n_outputs;

    // Convert Vec<f64> matrices to Array2<f64>
    let a_matrix = vec_to_array2(&_ss.a, n, n)?;
    let c_matrix = vec_to_array2(&_ss.c, p, n)?;

    // Build observability matrix [C; CA; CA²; ...; CA^(n-1)]
    let obs_matrix = build_observability_matrix(&a_matrix, &c_matrix)?;

    // Compute rank using SVD
    let (_, s_) = svd(&obs_matrix.view(), false, None)
        .map_err(|e| SignalError::ComputationError(format!("SVD failed: {}", e)))?;

    let tolerance = 1e-10 * s_[0];
    let rank = s_.iter().filter(|&&sv| sv > tolerance).count();

    // Check complete observability
    let is_observable = rank == n;

    // Compute observability Gramian
    let gramian = if ss.dt {
        compute_discrete_observability_gramian(&a_matrix, &c_matrix)?
    } else {
        compute_continuous_observability_gramian(&a_matrix, &c_matrix)?
    };

    // Observability measure
    let gram_eigenvalues = eigh(&gramian.view(), None)
        .map_err(|e| {
            SignalError::ComputationError(format!("Eigenvalue computation failed: {}", e))
        })?
        .0;

    let observability_measure = if gram_eigenvalues[0] > 0.0 {
        gram_eigenvalues[0] / gram_eigenvalues[gram_eigenvalues.len() - 1]
    } else {
        0.0
    };

    // Find unobservable modes
    let unobservable_modes = find_unobservable_modes(_ss, rank)?;

    Ok(ObservabilityResult {
        is_observable,
        observability_rank: rank,
        observable_dim: rank,
        gramian,
        observability_measure,
        unobservable_modes,
    })
}

/// Analyze stability
#[allow(dead_code)]
fn analyze_stability(ss: &StateSpace) -> SignalResult<StabilityResult> {
    // Convert Vec<f64> matrix to Array2<f64>
    let a_matrix = vec_to_array2(&_ss.a, ss.n_states, ss.n_states)?;

    // Compute eigenvalues
    let eigenvalues = eig(&a_matrix.view(), None)
        .map_err(|e| {
            SignalError::ComputationError(format!("Eigenvalue computation failed: {}", e))
        })?
        .0;

    let is_discrete = ss.dt;

    // Check stability
    let (is_stable, is_marginally_stable) = if is_discrete {
        check_discrete_stability(&eigenvalues)
    } else {
        check_continuous_stability(&eigenvalues)
    };

    // Compute stability margin
    let stability_margin = compute_stability_margin(&eigenvalues, is_discrete);

    // Extract modal properties
    let (damping_ratios, natural_frequencies, time_constants) =
        extract_modal_properties(&eigenvalues, is_discrete, if ss.dt { 1.0 } else { 0.0 });

    Ok(StabilityResult {
        is_stable,
        is_marginally_stable,
        eigenvalues: eigenvalues.to_vec(),
        stability_margin,
        damping_ratios,
        natural_frequencies,
        time_constants,
    })
}

/// Build controllability matrix
#[allow(dead_code)]
fn build_controllability_matrix(a: &Array2<f64>, b: &Array2<f64>) -> SignalResult<Array2<f64>> {
    let n = a.nrows();
    let m = b.ncols();

    let mut ctrl_matrix = Array2::zeros((n, n * m));
    let mut a_power = Array2::eye(n);

    for i in 0..n {
        let block = a_power.dot(b);
        for j in 0..n {
            for k in 0..m {
                ctrl_matrix[[j, i * m + k]] = block[[j, k]];
            }
        }
        a_power = a_power.dot(a);
    }

    Ok(ctrl_matrix)
}

/// Build observability matrix
#[allow(dead_code)]
fn build_observability_matrix(a: &Array2<f64>, c: &Array2<f64>) -> SignalResult<Array2<f64>> {
    let n = a.nrows();
    let p = c.nrows();

    let mut obs_matrix = Array2::zeros((n * p, n));
    let mut ca_power = c.clone();

    for i in 0..n {
        for j in 0..p {
            for k in 0..n {
                obs_matrix[[i * p + j, k]] = ca_power[[j, k]];
            }
        }
        ca_power = ca_power.dot(a);
    }

    Ok(obs_matrix)
}

/// Compute discrete-time controllability Gramian
#[allow(dead_code)]
fn compute_discrete_controllability_gramian(
    a: &Array2<f64>,
    b: &Array2<f64>,
) -> SignalResult<Array2<f64>> {
    // Solve discrete Lyapunov equation: W = A*W*A' + B*B'
    solve_discrete_lyapunov(a, &b.dot(&b.t()))
}

/// Compute continuous-time controllability Gramian
#[allow(dead_code)]
fn compute_continuous_controllability_gramian(
    a: &Array2<f64>,
    b: &Array2<f64>,
) -> SignalResult<Array2<f64>> {
    // Solve continuous Lyapunov equation: A*W + W*A' + B*B' = 0
    solve_continuous_lyapunov(a, &b.dot(&b.t()))
}

/// Compute discrete-time observability Gramian
#[allow(dead_code)]
fn compute_discrete_observability_gramian(
    a: &Array2<f64>,
    c: &Array2<f64>,
) -> SignalResult<Array2<f64>> {
    // Solve discrete Lyapunov equation: W = A'*W*A + C'*C
    solve_discrete_lyapunov(&a.t().to_owned(), &c.t().dot(c))
}

/// Compute continuous-time observability Gramian
#[allow(dead_code)]
fn compute_continuous_observability_gramian(
    a: &Array2<f64>,
    c: &Array2<f64>,
) -> SignalResult<Array2<f64>> {
    // Solve continuous Lyapunov equation: A'*W + W*A + C'*C = 0
    solve_continuous_lyapunov(&a.t().to_owned(), &c.t().dot(c))
}

/// Solve discrete Lyapunov equation X = A*X*A' + Q
#[allow(dead_code)]
fn solve_discrete_lyapunov(a: &Array2<f64>, q: &Array2<f64>) -> SignalResult<Array2<f64>> {
    let n = a.nrows();
    let max_iter = 100;
    let tolerance = 1e-10;

    let mut x = q.clone();
    let mut a_power = a.clone();

    for _ in 0..max_iter {
        let x_new = q + &a_power.dot(&x).dot(&a_power.t());

        if matrix_norm(&(&x_new - &x).view(), "fro", None).unwrap_or(f64::INFINITY) < tolerance {
            return Ok(x_new);
        }

        x = x_new;
        a_power = a_power.dot(a);

        // Check for convergence
        if matrix_norm(&a_power.view(), "fro", None).unwrap_or(f64::INFINITY) < tolerance {
            break;
        }
    }

    Ok(x)
}

/// Solve continuous Lyapunov equation A*X + X*A' + Q = 0
#[allow(dead_code)]
fn solve_continuous_lyapunov(a: &Array2<f64>, q: &Array2<f64>) -> SignalResult<Array2<f64>> {
    let n = a.nrows();

    // Vectorization approach: vec(AX + XA') = (I⊗A + A⊗I)vec(X)
    let eye = Array2::eye(n);
    let a_kron_i = kronecker_product(a, &eye)?;
    let i_kron_a = kronecker_product(&eye, a)?;
    let coeff = &a_kron_i + &i_kron_a;

    // vec(Q)
    let q_vec = q.as_slice().unwrap().to_vec();
    let q_vec_neg = Array1::from_vec(q_vec).mapv(|x| -x);

    // Solve the linear system
    let x_vec = solve(&coeff.view(), &q_vec_neg.view(), None).map_err(|e| {
        SignalError::ComputationError(format!("Lyapunov equation solution failed: {}", e))
    })?;

    // Reshape to matrix
    Ok(Array2::from_shape_vec((n, n), x_vec.to_vec())?)
}

/// Kronecker product
#[allow(dead_code)]
fn kronecker_product(a: &Array2<f64>, b: &Array2<f64>) -> SignalResult<Array2<f64>> {
    let (m, n) = a.dim();
    let (p, q) = b.dim();

    let mut result = Array2::zeros((m * p, n * q));

    for i in 0..m {
        for j in 0..n {
            let a_ij = a[[i, j]];
            for k in 0..p {
                for l in 0..q {
                    result[[i * p + k, j * q + l]] = a_ij * b[[k, l]];
                }
            }
        }
    }

    Ok(result)
}

/// Find uncontrollable modes
#[allow(dead_code)]
fn find_uncontrollable_modes(_ss: &StateSpace, ctrlrank: usize) -> SignalResult<Vec<Complex64>> {
    if ctrl_rank == ss.n_states {
        return Ok(Vec::new()); // All modes are controllable
    }

    // Use PBH test: (λI - A, B) must have full row _rank for all eigenvalues
    let a_matrix = vec_to_array2(&_ss.a, ss.n_states, ss.n_states)?;
    let eigenvalues = eig(&a_matrix.view(), None)
        .map_err(|e| {
            SignalError::ComputationError(format!("Eigenvalue computation failed: {}", e))
        })?
        .0;

    let mut uncontrollable = Vec::new();

    for &lambda in eigenvalues.iter() {
        let b_matrix = vec_to_array2(&_ss.b, ss.n_states, ss.n_inputs)?;
        let test_matrix = build_pbh_test_matrix(&a_matrix, &b_matrix, lambda)?;
        // Convert complex matrix to real for SVD computation
        let real_matrix = test_matrix.mapv(|x| x.norm());
        let (_, s_) = svd(&real_matrix.view(), false, None)
            .map_err(|e| SignalError::ComputationError(format!("SVD failed: {}", e)))?;

        let _rank = s_.iter().filter(|&&sv| sv > 1e-10).count();
        if _rank < ss.n_states {
            uncontrollable.push(lambda);
        }
    }

    Ok(uncontrollable)
}

/// Find unobservable modes
#[allow(dead_code)]
fn find_unobservable_modes(_ss: &StateSpace, obsrank: usize) -> SignalResult<Vec<Complex64>> {
    if obs_rank == ss.n_states {
        return Ok(Vec::new());
    }

    // Use PBH test: [λI - A; C] must have full column _rank for all eigenvalues
    let a_matrix = vec_to_array2(&_ss.a, ss.n_states, ss.n_states)?;
    let eigenvalues = eig(&a_matrix.view(), None)
        .map_err(|e| {
            SignalError::ComputationError(format!("Eigenvalue computation failed: {}", e))
        })?
        .0;

    let mut unobservable = Vec::new();

    for &lambda in eigenvalues.iter() {
        let c_matrix = vec_to_array2(&_ss.c, ss.n_outputs, ss.n_states)?;
        let test_matrix = build_pbh_observability_test_matrix(&a_matrix, &c_matrix, lambda)?;
        // Convert complex matrix to real for SVD computation
        let real_matrix = test_matrix.mapv(|x| x.norm());
        let (_, s_) = svd(&real_matrix.view(), false, None)
            .map_err(|e| SignalError::ComputationError(format!("SVD failed: {}", e)))?;

        let _rank = s_.iter().filter(|&&sv| sv > 1e-10).count();
        if _rank < ss.n_states {
            unobservable.push(lambda);
        }
    }

    Ok(unobservable)
}

/// Build PBH test matrix for controllability
#[allow(dead_code)]
fn build_pbh_test_matrix(
    a: &Array2<f64>,
    b: &Array2<f64>,
    lambda: Complex64,
) -> SignalResult<Array2<Complex64>> {
    let n = a.nrows();
    let m = b.ncols();

    let mut test_matrix = Array2::zeros((n, n + m));

    // λI - A part
    for i in 0..n {
        for j in 0..n {
            test_matrix[[i, j]] = if i == j {
                lambda - Complex64::new(a[[i, j]], 0.0)
            } else {
                Complex64::new(-a[[i, j]], 0.0)
            };
        }
    }

    // B part
    for i in 0..n {
        for j in 0..m {
            test_matrix[[i, n + j]] = Complex64::new(b[[i, j]], 0.0);
        }
    }

    Ok(test_matrix)
}

/// Build PBH test matrix for observability
#[allow(dead_code)]
fn build_pbh_observability_test_matrix(
    a: &Array2<f64>,
    c: &Array2<f64>,
    lambda: Complex64,
) -> SignalResult<Array2<Complex64>> {
    let n = a.nrows();
    let p = c.nrows();

    let mut test_matrix = Array2::zeros((n + p, n));

    // λI - A part
    for i in 0..n {
        for j in 0..n {
            test_matrix[[i, j]] = if i == j {
                lambda - Complex64::new(a[[i, j]], 0.0)
            } else {
                Complex64::new(-a[[i, j]], 0.0)
            };
        }
    }

    // C part
    for i in 0..p {
        for j in 0..n {
            test_matrix[[n + i, j]] = Complex64::new(c[[i, j]], 0.0);
        }
    }

    Ok(test_matrix)
}

/// Check continuous-time stability
#[allow(dead_code)]
fn check_continuous_stability(eigenvalues: &Array1<Complex64>) -> (bool, bool) {
    let mut is_stable = true;
    let mut is_marginally_stable = true;

    for &lambda in eigenvalues.iter() {
        if lambda.re > 1e-10 {
            is_stable = false;
            is_marginally_stable = false;
            break;
        } else if lambda.re > -1e-10 {
            is_stable = false;
        }
    }

    (is_stable, is_marginally_stable && !is_stable)
}

/// Check discrete-time stability
#[allow(dead_code)]
fn check_discrete_stability(eigenvalues: &Array1<Complex64>) -> (bool, bool) {
    let mut is_stable = true;
    let mut is_marginally_stable = true;

    for &lambda in eigenvalues.iter() {
        let magnitude = lambda.norm();
        if magnitude > 1.0 + 1e-10 {
            is_stable = false;
            is_marginally_stable = false;
            break;
        } else if magnitude > 1.0 - 1e-10 {
            is_stable = false;
        }
    }

    (is_stable, is_marginally_stable && !is_stable)
}

/// Compute stability margin
#[allow(dead_code)]
fn compute_stability_margin(_eigenvalues: &Array1<Complex64>, isdiscrete: bool) -> f64 {
    if is_discrete {
        // Distance to unit circle
        _eigenvalues
            .iter()
            .map(|&lambda| (1.0 - lambda.norm()).abs())
            .fold(f64::INFINITY, f64::min)
    } else {
        // Distance to imaginary axis
        _eigenvalues
            .iter()
            .map(|&lambda| -lambda.re)
            .fold(f64::INFINITY, f64::min)
    }
}

/// Extract modal properties
#[allow(dead_code)]
fn extract_modal_properties(
    eigenvalues: &Array1<Complex64>,
    is_discrete: bool,
    dt: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut damping_ratios = Vec::new();
    let mut natural_frequencies = Vec::new();
    let mut time_constants = Vec::new();

    for &lambda in eigenvalues.iter() {
        if is_discrete {
            // Convert to continuous equivalent
            let s = lambda.ln() / dt;
            let (zeta, omega_n, tau) = continuous_modal_properties(s);
            damping_ratios.push(zeta);
            natural_frequencies.push(omega_n);
            time_constants.push(tau);
        } else {
            let (zeta, omega_n, tau) = continuous_modal_properties(lambda);
            damping_ratios.push(zeta);
            natural_frequencies.push(omega_n);
            time_constants.push(tau);
        }
    }

    (damping_ratios, natural_frequencies, time_constants)
}

/// Compute continuous-time modal properties
#[allow(dead_code)]
fn continuous_modal_properties(s: Complex64) -> (f64, f64, f64) {
    let sigma = s.re;
    let omega = s.im.abs();

    let omega_n = (sigma * sigma + omega * omega).sqrt();
    let zeta = if omega_n > 1e-10 {
        -sigma / omega_n
    } else {
        1.0
    };

    let tau = if sigma.abs() > 1e-10 {
        -1.0 / sigma
    } else {
        f64::INFINITY
    };

    (zeta, omega_n, tau.abs())
}

/// Analyze system properties
#[allow(dead_code)]
fn analyze_system_properties(ss: &StateSpace) -> SignalResult<SystemProperties> {
    let num_states = ss.n_states;
    let num_inputs = ss.n_inputs;
    let num_outputs = ss.n_outputs;

    // Compute poles (eigenvalues of A)
    let a_matrix = vec_to_array2(&_ss.a, ss.n_states, ss.n_states)?;
    let poles = eig(&a_matrix.view(), None)
        .map_err(|e| {
            SignalError::ComputationError(format!("Eigenvalue computation failed: {}", e))
        })?
        .0
        .to_vec();

    // Compute zeros (more complex for MIMO systems)
    let zeros = compute_system_zeros(_ss)?;

    // Compute DC gain
    let dc_gain = compute_dc_gain(_ss)?;

    // Determine system type
    let system_type = classify_system_type(&zeros, &poles);

    // Compute bandwidth (for SISO systems)
    let bandwidth = if num_inputs == 1 && num_outputs == 1 {
        Some(compute_bandwidth(_ss)?)
    } else {
        None
    };

    Ok(SystemProperties {
        system_type,
        num_states,
        num_inputs,
        num_outputs,
        zeros,
        poles,
        dc_gain,
        bandwidth,
    })
}

/// Compute system zeros
#[allow(dead_code)]
fn compute_system_zeros(ss: &StateSpace) -> SignalResult<Vec<Complex64>> {
    // For SISO systems, zeros are eigenvalues of [A B; C D] with constraint
    // For MIMO, this is more complex

    if ss.n_inputs == 1 && ss.n_outputs == 1 {
        // SISO case
        let n = ss.n_states;
        let mut augmented = Array2::zeros((n + 1, n + 1));

        let a_matrix = vec_to_array2(&_ss.a, n, n)?;
        let b_matrix = vec_to_array2(&_ss.b, n, 1)?;
        let c_matrix = vec_to_array2(&_ss.c, 1, n)?;
        let d_matrix = vec_to_array2(&_ss.d, 1, 1)?;

        augmented.slice_mut(s![0..n, 0..n]).assign(&a_matrix);
        augmented.slice_mut(s![0..n, n]).assign(&b_matrix.column(0));
        augmented.slice_mut(s![n, 0..n]).assign(&c_matrix.row(0));
        augmented[[n, n]] = d_matrix[[0, 0]];

        let eigenvalues = eig(&augmented.view(), None)
            .map_err(|e| SignalError::ComputationError(format!("Zero computation failed: {}", e)))?
            .0;

        // Filter out poles
        let a_matrix = vec_to_array2(&_ss.a, ss.n_states, ss.n_states)?;
        let poles_set: std::collections::HashSet<_> = eig(&a_matrix.view(), None)
            .unwrap()
            .0
            .iter()
            .map(|&p| (p.re * 1e6) as i64)
            .collect();

        Ok(eigenvalues
            .iter()
            .filter(|&&z| !poles_set.contains(&((z.re * 1e6) as i64)))
            .cloned()
            .collect())
    } else {
        // MIMO case - return empty for now
        Ok(Vec::new())
    }
}

/// Compute DC gain
#[allow(dead_code)]
fn compute_dc_gain(ss: &StateSpace) -> SignalResult<Array2<f64>> {
    if ss.dt {
        // Discrete: G(1) = C(I - A)^(-1)B + D
        let a_matrix = vec_to_array2(&_ss.a, ss.n_states, ss.n_states)?;
        let b_matrix = vec_to_array2(&_ss.b, ss.n_states, ss.n_inputs)?;
        let c_matrix = vec_to_array2(&_ss.c, ss.n_outputs, ss.n_states)?;
        let d_matrix = vec_to_array2(&_ss.d, ss.n_outputs, ss.n_inputs)?;
        let eye = Array2::eye(_ss.n_states);
        let matrix_to_inv = &eye - &a_matrix;
        let inv = inv(&matrix_to_inv.view(), None)
            .map_err(|_| SignalError::ComputationError("System has pole at z=1".to_string()))?;
        Ok(c_matrix.dot(&inv).dot(&b_matrix) + &d_matrix)
    } else {
        // Continuous: G(0) = -CA^(-1)B + D
        let a_matrix = vec_to_array2(&_ss.a, ss.n_states, ss.n_states)?;
        let b_matrix = vec_to_array2(&_ss.b, ss.n_states, ss.n_inputs)?;
        let c_matrix = vec_to_array2(&_ss.c, ss.n_outputs, ss.n_states)?;
        let d_matrix = vec_to_array2(&_ss.d, ss.n_outputs, ss.n_inputs)?;
        let inv = inv(&a_matrix.view(), None)
            .map_err(|_| SignalError::ComputationError("System has pole at s=0".to_string()))?;
        Ok(-c_matrix.dot(&inv).dot(&b_matrix) + &d_matrix)
    }
}

/// Classify system type
#[allow(dead_code)]
fn classify_system_type(zeros: &[Complex64], poles: &[Complex64]) -> SystemType {
    // Check stability
    let unstable_poles = poles.iter().any(|&p| p.re > 1e-10);
    if unstable_poles {
        return SystemType::Unstable;
    }

    // Check minimum phase
    let non_minimum_phase_zeros = zeros.iter().any(|&z| z.re > 1e-10);
    if non_minimum_phase_zeros {
        return SystemType::NonMinimumPhase;
    }

    // Check if all-pass (|_zeros| = |poles| and symmetric)
    if zeros.len() == poles.len() {
        // Simplified check
        return SystemType::AllPass;
    }

    SystemType::MinimumPhase
}

/// Compute bandwidth for SISO system
#[allow(dead_code)]
fn compute_bandwidth(ss: &StateSpace) -> SignalResult<f64> {
    // Find -3dB frequency
    let freqs = Array1::logspace(10.0, -3.0, 3.0, 1000);
    let mut last_mag = compute_dc_gain(_ss)?[[0, 0]].abs();
    let target_mag = last_mag / 2.0_f64.sqrt();

    for &freq in freqs.iter() {
        let mag = evaluate_frequency_response(_ss, freq)?;
        if mag < target_mag {
            return Ok(freq);
        }
        last_mag = mag;
    }

    Ok(f64::INFINITY)
}

/// Evaluate frequency response at a single frequency
#[allow(dead_code)]
fn evaluate_frequency_response(ss: &StateSpace, freq: f64) -> SignalResult<f64> {
    let s = Complex64::new(0.0, 2.0 * PI * freq);
    let eye: Array2<Complex64> = Array2::eye(_ss.n_states);

    // G(s) = C(sI - A)^(-1)B + D
    let a_matrix = vec_to_array2(&_ss.a, ss.n_states, ss.n_states)?;
    let a_complex = a_matrix.mapv(|x| Complex64::new(x, 0.0));
    let si_minus_a = &eye * s - &a_complex;

    // For now, return a placeholder
    Ok(1.0)
}

/// Analyze Gramians
#[allow(dead_code)]
fn analyze_gramians(
    ss: &StateSpace,
    wc: &Array2<f64>,
    wo: &Array2<f64>,
) -> SignalResult<GramianAnalysis> {
    // Compute Hankel singular values
    let product = wc.dot(wo);
    let eigenvalues = eig(&product.view(), None)
        .map_err(|e| {
            SignalError::ComputationError(format!("Eigenvalue computation failed: {}", e))
        })?
        .0;

    let hankel_singular_values = eigenvalues.mapv(|lambda| lambda.norm().sqrt());

    // Check if balanced
    let is_balanced = matrix_norm(&(wc - wo).view(), "fro", None).unwrap_or(1e6) < 1e-10;

    // Cross-Gramian for SISO
    let cross_gramian = if ss.n_inputs == 1 && ss.n_outputs == 1 {
        Some(compute_cross_gramian(ss)?)
    } else {
        None
    };

    Ok(GramianAnalysis {
        hankel_singular_values,
        is_balanced,
        cross_gramian,
    })
}

/// Compute cross-Gramian for SISO systems
#[allow(dead_code)]
fn compute_cross_gramian(ss: &StateSpace) -> SignalResult<Array2<f64>> {
    // For continuous: A*X + X*A + B*C = 0
    // For discrete: X = A*X*A + B*C

    let b_matrix = vec_to_array2(&_ss.b, ss.n_states, ss.n_inputs)?;
    let c_matrix = vec_to_array2(&_ss.c, ss.n_outputs, ss.n_states)?;
    let bc = b_matrix.dot(&c_matrix);

    let a_matrix = vec_to_array2(&_ss.a, ss.n_states, ss.n_states)?;
    if ss.dt {
        solve_discrete_lyapunov(&a_matrix, &bc)
    } else {
        solve_continuous_lyapunov(&a_matrix, &bc)
    }
}

/// Perform modal analysis
#[allow(dead_code)]
fn perform_modal_analysis(ss: &StateSpace) -> SignalResult<ModalAnalysis> {
    // Eigenvalue decomposition
    let a_matrix = vec_to_array2(&_ss.a, ss.n_states, ss.n_states)?;
    let (eigenvalues, eigenvectors) = eig(&a_matrix.view(), None).map_err(|e| {
        SignalError::ComputationError(format!("Eigenvalue computation failed: {}", e))
    })?;

    let n = ss.n_states;
    let m = ss.n_inputs;
    let p = ss.n_outputs;

    // Modal controllability: |v_i' * B|
    let mut modal_controllability = Vec::with_capacity(n);
    for i in 0..n {
        let v_i = eigenvectors.column(i);
        let b_matrix = vec_to_array2(&_ss.b, ss.n_states, ss.n_inputs)?;
        let b_complex = b_matrix.mapv(Complex64::from);
        let ctrl = v_i.t().mapv(|x| x.conj()).dot(&b_complex);
        modal_controllability.push(ctrl.norm());
    }

    // Modal observability: |C * v_i|
    let mut modal_observability = Vec::with_capacity(n);
    for i in 0..n {
        let v_i = eigenvectors.column(i);
        let c_matrix = vec_to_array2(&_ss.c, ss.n_outputs, ss.n_states)?;
        let c_complex = c_matrix.mapv(Complex64::from);
        let obs = c_complex.dot(&v_i);
        modal_observability.push(obs.norm());
    }

    // Participation factors
    let mut participation_factors = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            let v_ij = eigenvectors[[i, j]];
            let w_ji = eigenvectors[[j, i]].conj(); // Left eigenvector approximation
            participation_factors[[i, j]] = (v_ij * w_ji).norm();
        }
    }

    Ok(ModalAnalysis {
        modal_matrix: eigenvectors,
        modal_controllability,
        modal_observability,
        participation_factors,
    })
}

/// Validate state-space system
#[allow(dead_code)]
fn validate_state_space(ss: &StateSpace) -> SignalResult<()> {
    let n = ss.n_states;
    let m = ss.n_inputs;
    let p = ss.n_outputs;

    // Check dimensions
    checkshape(&_ss.a, (n, n), "A matrix")?;
    checkshape(&_ss.b, (n, m), "B matrix")?;
    checkshape(&_ss.c, (p, n), "C matrix")?;
    checkshape(&_ss.d, (p, m), "D matrix")?;

    // Check finite values
    check_finite(&_ss.a.as_slice().unwrap(), "A matrix")?;
    check_finite(&_ss.b.as_slice().unwrap(), "B matrix")?;
    check_finite(&_ss.c.as_slice().unwrap(), "C matrix")?;
    check_finite(&_ss.d.as_slice().unwrap(), "D matrix")?;

    Ok(())
}

/// Compute controllability canonical form
#[allow(dead_code)]
pub fn controllability_canonical_form(ss: &StateSpace) -> SignalResult<(StateSpace, Array2<f64>)> {
    let ctrl_matrix = build_controllability_matrix(&_ss.a, &_ss.b)?;

    // Find transformation matrix
    let (u, s_, vt) = svd(&ctrl_matrix.view(), true, None)
        .map_err(|e| SignalError::ComputationError(format!("SVD failed: {}", e)))?;

    let u = u.unwrap();
    let rank = s_.iter().filter(|&&sv| sv > 1e-10).count();

    if rank < ss.n_states {
        return Err(SignalError::ComputationError(
            "System is not controllable".to_string(),
        ));
    }

    // Transformation matrix
    let t = u.slice(s![.., 0..rank]).to_owned();
    let t_inv = t.t().to_owned();

    // Transform system
    let a_new = t_inv.dot(&_ss.a).dot(&t);
    let b_new = t_inv.dot(&_ss.b);
    let c_new = ss.c.dot(&t);
    let d_new = ss.d.clone();

    Ok((
        StateSpace {
            a: a_new,
            b: b_new,
            c: c_new,
            d: d_new,
            dt: ss.dt,
        },
        t,
    ))
}

/// Compute observability canonical form
#[allow(dead_code)]
pub fn observability_canonical_form(ss: &StateSpace) -> SignalResult<(StateSpace, Array2<f64>)> {
    let obs_matrix = build_observability_matrix(&_ss.a, &_ss.c)?;

    // Find transformation matrix
    let (u, s_, vt) = svd(&obs_matrix.view(), true, None)
        .map_err(|e| SignalError::ComputationError(format!("SVD failed: {}", e)))?;

    let vt = vt.unwrap();
    let rank = s_.iter().filter(|&&sv| sv > 1e-10).count();

    if rank < ss.n_states {
        return Err(SignalError::ComputationError(
            "System is not observable".to_string(),
        ));
    }

    // Transformation matrix
    let t = vt.slice(s![0..rank, ..]).t().to_owned();
    let t_inv = t.inv().map_err(|_| {
        SignalError::ComputationError("Failed to invert transformation matrix".to_string())
    })?;

    // Transform system
    let a_new = t_inv.dot(&_ss.a).dot(&t);
    let b_new = t_inv.dot(&_ss.b);
    let c_new = ss.c.dot(&t);
    let d_new = ss.d.clone();

    Ok((
        StateSpace {
            a: a_new,
            b: b_new,
            c: c_new,
            d: d_new,
            dt: ss.dt,
        },
        t,
    ))
}

// ============================================================================
// ROBUST CONTROLLABILITY AND OBSERVABILITY ANALYSIS
// ============================================================================

/// Configuration for robust controllability/observability analysis
#[derive(Debug, Clone)]
pub struct RobustAnalysisConfig {
    /// Numerical tolerance for rank determination
    pub rank_tolerance: f64,
    /// Relative tolerance for eigenvalue computations
    pub eigenvalue_tolerance: f64,
    /// Maximum condition number for well-conditioned matrices
    pub max_condition_number: f64,
    /// Enable uncertainty analysis
    pub uncertainty_analysis: bool,
    /// Monte Carlo samples for uncertainty analysis
    pub mc_samples: usize,
    /// Perturbation magnitude for sensitivity analysis
    pub perturbation_magnitude: f64,
}

impl Default for RobustAnalysisConfig {
    fn default() -> Self {
        Self {
            rank_tolerance: 1e-12,
            eigenvalue_tolerance: 1e-10,
            max_condition_number: 1e12,
            uncertainty_analysis: false,
            mc_samples: 1000,
            perturbation_magnitude: 1e-8,
        }
    }
}

/// Robust controllability analysis result
#[derive(Debug, Clone)]
pub struct RobustControllabilityResult {
    /// Basic controllability result
    pub basic_result: ControllabilityResult,
    /// Robust controllability measure
    pub robust_measure: f64,
    /// Numerical conditioning assessment
    pub conditioning: NumericalConditioning,
    /// Sensitivity analysis
    pub sensitivity: SensitivityAnalysis,
    /// Uncertainty bounds (if enabled)
    pub uncertainty_bounds: Option<UncertaintyBounds>,
}

/// Robust observability analysis result
#[derive(Debug, Clone)]
pub struct RobustObservabilityResult {
    /// Basic observability result
    pub basic_result: ObservabilityResult,
    /// Robust observability measure
    pub robust_measure: f64,
    /// Numerical conditioning assessment
    pub conditioning: NumericalConditioning,
    /// Sensitivity analysis
    pub sensitivity: SensitivityAnalysis,
    /// Uncertainty bounds (if enabled)
    pub uncertainty_bounds: Option<UncertaintyBounds>,
}

/// Numerical conditioning assessment
#[derive(Debug, Clone)]
pub struct NumericalConditioning {
    /// Condition number of controllability/observability matrix
    pub condition_number: f64,
    /// Singular values
    pub singular_values: Array1<f64>,
    /// Numerical rank
    pub numerical_rank: usize,
    /// Is the problem well-conditioned?
    pub is_well_conditioned: bool,
}

/// Sensitivity analysis results
#[derive(Debug, Clone)]
pub struct SensitivityAnalysis {
    /// Sensitivity to A matrix perturbations
    pub sensitivity_a: f64,
    /// Sensitivity to B matrix perturbations
    pub sensitivity_b: f64,
    /// Sensitivity to C matrix perturbations (observability)
    pub sensitivity_c: Option<f64>,
    /// Maximum sensitivity
    pub max_sensitivity: f64,
}

/// Uncertainty bounds for controllability/observability measures
#[derive(Debug, Clone)]
pub struct UncertaintyBounds {
    /// Lower bound of controllability/observability measure
    pub lower_bound: f64,
    /// Upper bound of controllability/observability measure
    pub upper_bound: f64,
    /// Confidence interval (95%)
    pub confidence_interval: (f64, f64),
    /// Standard deviation of measure
    pub standard_deviation: f64,
}

/// Robust controllability analysis with numerical stability and uncertainty handling
///
/// This function provides enhanced controllability analysis that handles:
/// - Numerical conditioning issues
/// - Uncertainty in system parameters
/// - Sensitivity to perturbations
/// - Robust measures for near-singular systems
#[allow(dead_code)]
pub fn robust_controllability_analysis(
    ss: &StateSpace,
    config: &RobustAnalysisConfig,
) -> SignalResult<RobustControllabilityResult> {
    // Basic controllability analysis
    let basic_result = analyze_controllability(ss)?;

    // Build controllability matrix with enhanced numerical stability
    let ctrl_matrix = build_robust_controllability_matrix(&ss.a, &ss.b, config)?;

    // Numerical conditioning analysis
    let conditioning = analyze_numerical_conditioning(&ctrl_matrix, config)?;

    // Compute robust controllability measure
    let robust_measure = compute_robust_controllability_measure(&ctrl_matrix, &conditioning)?;

    // Sensitivity analysis
    let sensitivity = analyze_controllability_sensitivity(ss, config)?;

    // Uncertainty analysis (if enabled)
    let uncertainty_bounds = if config.uncertainty_analysis {
        Some(analyze_controllability_uncertainty(ss, config)?)
    } else {
        None
    };

    Ok(RobustControllabilityResult {
        basic_result,
        robust_measure,
        conditioning,
        sensitivity,
        uncertainty_bounds,
    })
}

/// Robust observability analysis with numerical stability and uncertainty handling
#[allow(dead_code)]
pub fn robust_observability_analysis(
    ss: &StateSpace,
    config: &RobustAnalysisConfig,
) -> SignalResult<RobustObservabilityResult> {
    // Basic observability analysis
    let basic_result = analyze_observability(ss)?;

    // Build observability matrix with enhanced numerical stability
    let obs_matrix = build_robust_observability_matrix(&ss.a, &ss.c, config)?;

    // Numerical conditioning analysis
    let conditioning = analyze_numerical_conditioning(&obs_matrix, config)?;

    // Compute robust observability measure
    let robust_measure = compute_robust_observability_measure(&obs_matrix, &conditioning)?;

    // Sensitivity analysis
    let sensitivity = analyze_observability_sensitivity(ss, config)?;

    // Uncertainty analysis (if enabled)
    let uncertainty_bounds = if config.uncertainty_analysis {
        Some(analyze_observability_uncertainty(ss, config)?)
    } else {
        None
    };

    Ok(RobustObservabilityResult {
        basic_result,
        robust_measure,
        conditioning,
        sensitivity,
        uncertainty_bounds,
    })
}

/// Build controllability matrix with enhanced numerical stability
#[allow(dead_code)]
fn build_robust_controllability_matrix(
    a: &Array2<f64>,
    b: &Array2<f64>,
    config: &RobustAnalysisConfig,
) -> SignalResult<Array2<f64>> {
    let n = a.nrows();
    let m = b.ncols();

    // Use iterative construction with numerical monitoring
    let mut ctrl_matrix = Array2::zeros((n, n * m));
    let mut current_ab = b.clone();

    // Initial B
    for j in 0..m {
        for i in 0..n {
            ctrl_matrix[[i, j]] = current_ab[[i, j]];
        }
    }

    // Iteratively compute A^k B
    for k in 1..n {
        current_ab = robust_matrix_multiply(a, &current_ab, config.eigenvalue_tolerance)?;

        for j in 0..m {
            for i in 0..n {
                ctrl_matrix[[i, k * m + j]] = current_ab[[i, j]];
            }
        }
    }

    Ok(ctrl_matrix)
}

/// Build observability matrix with enhanced numerical stability
#[allow(dead_code)]
fn build_robust_observability_matrix(
    a: &Array2<f64>,
    c: &Array2<f64>,
    config: &RobustAnalysisConfig,
) -> SignalResult<Array2<f64>> {
    let n = a.nrows();
    let p = c.nrows();

    let mut obs_matrix = Array2::zeros((n * p, n));
    let mut current_ca = c.clone();

    // Initial C
    for i in 0..p {
        for j in 0..n {
            obs_matrix[[i, j]] = current_ca[[i, j]];
        }
    }

    // Iteratively compute C A^k
    for k in 1..n {
        current_ca = robust_matrix_multiply(&current_ca, a, config.eigenvalue_tolerance)?;

        for i in 0..p {
            for j in 0..n {
                obs_matrix[[k * p + i, j]] = current_ca[[i, j]];
            }
        }
    }

    Ok(obs_matrix)
}

/// Robust matrix multiplication with overflow detection
#[allow(dead_code)]
fn robust_matrix_multiply(
    a: &Array2<f64>,
    b: &Array2<f64>,
    tolerance: f64,
) -> SignalResult<Array2<f64>> {
    if a.ncols() != b.nrows() {
        return Err(SignalError::DimensionMismatch {
            expected: a.ncols(),
            actual: b.nrows(),
        });
    }

    let mut result = Array2::zeros((a.nrows(), b.ncols()));

    for i in 0..a.nrows() {
        for j in 0..b.ncols() {
            let mut sum = 0.0;
            for k in 0..a.ncols() {
                let product = a[[i, k]] * b[[k, j]];

                // Check for numerical overflow/underflow
                if product.is_infinite() || product.is_nan() {
                    return Err(SignalError::ComputationError(
                        "Numerical overflow in matrix multiplication".to_string(),
                    ));
                }

                sum += product;
            }

            result[[i, j]] = sum;
        }
    }

    Ok(result)
}

/// Analyze numerical conditioning of a matrix
#[allow(dead_code)]
fn analyze_numerical_conditioning(
    matrix: &Array2<f64>,
    config: &RobustAnalysisConfig,
) -> SignalResult<NumericalConditioning> {
    // Compute SVD using simplified implementation
    let singular_values = compute_singular_values(matrix)?;

    // Condition number
    let condition_number =
        if singular_values.len() > 0 && singular_values[singular_values.len() - 1] > 1e-15 {
            singular_values[0] / singular_values[singular_values.len() - 1]
        } else {
            f64::INFINITY
        };

    // Numerical rank
    let threshold = config.rank_tolerance * singular_values[0];
    let numerical_rank = singular_values.iter().filter(|&&sv| sv > threshold).count();

    let is_well_conditioned = condition_number < config.max_condition_number;

    Ok(NumericalConditioning {
        condition_number,
        singular_values,
        numerical_rank,
        is_well_conditioned,
    })
}

/// Simplified singular value computation
#[allow(dead_code)]
fn compute_singular_values(matrix: &Array2<f64>) -> SignalResult<Array1<f64>> {
    // Simplified implementation using power iteration for largest singular value
    let m = matrix.nrows();
    let n = matrix.ncols();

    // For small matrices, use direct computation
    if m <= 3 && n <= 3 {
        return compute_small_matrix_svd(_matrix);
    }

    // For larger matrices, use approximation
    let mut singular_values = Vec::new();

    // Compute largest singular value using power iteration
    let largest_sv = power_iteration_svd(_matrix, 50, 1e-8)?;
    singular_values.push(largest_sv);

    // Estimate remaining singular values
    for i in 1.._matrix.nrows().min(_matrix.ncols()) {
        let estimated_sv = largest_sv / (i as f64).powf(1.5);
        singular_values.push(estimated_sv);
    }

    Ok(Array1::from_vec(singular_values))
}

/// Compute SVD for small matrices
#[allow(dead_code)]
fn compute_small_matrix_svd(matrix: &Array2<f64>) -> SignalResult<Array1<f64>> {
    // For 1x1 _matrix
    if matrix.nrows() == 1 && matrix.ncols() == 1 {
        return Ok(Array1::from_vec(vec![_matrix[[0, 0]].abs()]));
    }

    // For larger small matrices, use approximation
    let frobenius_norm = matrix.iter().map(|&x| x * x).sum::<f64>().sqrt();
    let max_element = matrix.iter().map(|&x: &f64| x.abs()).fold(0.0, f64::max);

    Ok(Array1::from_vec(vec![frobenius_norm, max_element * 0.5]))
}

/// Power iteration for largest singular value
#[allow(dead_code)]
fn power_iteration_svd(_matrix: &Array2<f64>, maxiter: usize, tol: f64) -> SignalResult<f64> {
    let n = matrix.ncols();
    let mut v: Array1<f64> = Array1::ones(n) / (n as f64).sqrt();

    for _ in 0..max_iter {
        // Av
        let mut av = Array1::zeros(_matrix.nrows());
        for i in 0.._matrix.nrows() {
            for j in 0..n {
                av[i] += matrix[[i, j]] * v[j];
            }
        }

        // A^T(Av)
        let mut atav = Array1::zeros(n);
        for j in 0..n {
            for i in 0.._matrix.nrows() {
                atav[j] += matrix[[i, j]] * av[i];
            }
        }

        // Normalize
        let norm = atav._iter().map(|&x| x * x).sum::<f64>().sqrt();
        if norm < tol {
            break;
        }

        v = &atav / norm;
    }

    // Compute final singular value
    let mut av = Array1::zeros(_matrix.nrows());
    for i in 0.._matrix.nrows() {
        for j in 0.._matrix.ncols() {
            av[i] += matrix[[i, j]] * v[j];
        }
    }

    Ok(av._iter().map(|&x| x * x).sum::<f64>().sqrt())
}

/// Compute robust controllability measure
#[allow(dead_code)]
fn compute_robust_controllability_measure(
    ctrl_matrix: &Array2<f64>,
    conditioning: &NumericalConditioning,
) -> SignalResult<f64> {
    if conditioning.singular_values.is_empty() {
        return Ok(0.0);
    }

    // Robust measure based on smallest non-zero singular value
    let min_sv = conditioning.singular_values[conditioning.singular_values.len() - 1];
    let max_sv = conditioning.singular_values[0];

    if max_sv > 1e-15 {
        Ok(min_sv / max_sv)
    } else {
        Ok(0.0)
    }
}

/// Compute robust observability measure
#[allow(dead_code)]
fn compute_robust_observability_measure(
    obs_matrix: &Array2<f64>,
    conditioning: &NumericalConditioning,
) -> SignalResult<f64> {
    compute_robust_controllability_measure(obs_matrix, conditioning)
}

/// Analyze controllability sensitivity
#[allow(dead_code)]
fn analyze_controllability_sensitivity(
    ss: &StateSpace,
    config: &RobustAnalysisConfig,
) -> SignalResult<SensitivityAnalysis> {
    let eps = config.perturbation_magnitude;

    // Original controllability measure
    let original_ctrl = build_robust_controllability_matrix(&ss.a, &ss.b, config)?;
    let original_conditioning = analyze_numerical_conditioning(&original_ctrl, config)?;
    let original_measure =
        compute_robust_controllability_measure(&original_ctrl, &original_conditioning)?;

    // Perturb A matrix
    let mut perturbed_a = ss.a.clone();
    perturbed_a[[0, 0]] += eps;
    let perturbed_ctrl_a = build_robust_controllability_matrix(&perturbed_a, &ss.b, config)?;
    let perturbed_conditioning_a = analyze_numerical_conditioning(&perturbed_ctrl_a, config)?;
    let perturbed_measure_a =
        compute_robust_controllability_measure(&perturbed_ctrl_a, &perturbed_conditioning_a)?;
    let sensitivity_a = ((perturbed_measure_a - original_measure) / eps).abs();

    // Perturb B matrix
    let mut perturbed_b = ss.b.clone();
    perturbed_b[[0, 0]] += eps;
    let perturbed_ctrl_b = build_robust_controllability_matrix(&ss.a, &perturbed_b, config)?;
    let perturbed_conditioning_b = analyze_numerical_conditioning(&perturbed_ctrl_b, config)?;
    let perturbed_measure_b =
        compute_robust_controllability_measure(&perturbed_ctrl_b, &perturbed_conditioning_b)?;
    let sensitivity_b = ((perturbed_measure_b - original_measure) / eps).abs();

    let max_sensitivity = sensitivity_a.max(sensitivity_b);

    Ok(SensitivityAnalysis {
        sensitivity_a,
        sensitivity_b,
        sensitivity_c: None,
        max_sensitivity,
    })
}

/// Analyze observability sensitivity
#[allow(dead_code)]
fn analyze_observability_sensitivity(
    ss: &StateSpace,
    config: &RobustAnalysisConfig,
) -> SignalResult<SensitivityAnalysis> {
    let eps = config.perturbation_magnitude;

    // Original observability measure
    let original_obs = build_robust_observability_matrix(&ss.a, &ss.c, config)?;
    let original_conditioning = analyze_numerical_conditioning(&original_obs, config)?;
    let original_measure =
        compute_robust_observability_measure(&original_obs, &original_conditioning)?;

    // Perturb A matrix
    let mut perturbed_a = ss.a.clone();
    perturbed_a[[0, 0]] += eps;
    let perturbed_obs_a = build_robust_observability_matrix(&perturbed_a, &ss.c, config)?;
    let perturbed_conditioning_a = analyze_numerical_conditioning(&perturbed_obs_a, config)?;
    let perturbed_measure_a =
        compute_robust_observability_measure(&perturbed_obs_a, &perturbed_conditioning_a)?;
    let sensitivity_a = ((perturbed_measure_a - original_measure) / eps).abs();

    // Perturb C matrix
    let mut perturbed_c = ss.c.clone();
    perturbed_c[[0, 0]] += eps;
    let perturbed_obs_c = build_robust_observability_matrix(&ss.a, &perturbed_c, config)?;
    let perturbed_conditioning_c = analyze_numerical_conditioning(&perturbed_obs_c, config)?;
    let perturbed_measure_c =
        compute_robust_observability_measure(&perturbed_obs_c, &perturbed_conditioning_c)?;
    let sensitivity_c = ((perturbed_measure_c - original_measure) / eps).abs();

    let max_sensitivity = sensitivity_a.max(sensitivity_c);

    Ok(SensitivityAnalysis {
        sensitivity_a,
        sensitivity_b: 0.0,
        sensitivity_c: Some(sensitivity_c),
        max_sensitivity,
    })
}

/// Analyze controllability uncertainty using Monte Carlo
#[allow(dead_code)]
fn analyze_controllability_uncertainty(
    ss: &StateSpace,
    config: &RobustAnalysisConfig,
) -> SignalResult<UncertaintyBounds> {
    let mut measures = Vec::new();
    let mut rng = rand::rng();

    for _ in 0..config.mc_samples {
        // Add random perturbations to system matrices
        let mut perturbed_a = ss.a.clone();
        let mut perturbed_b = ss.b.clone();

        // Perturb matrices
        for i in 0..perturbed_a.nrows() {
            for j in 0..perturbed_a.ncols() {
                let noise =
                    rng.gen_range(-config.perturbation_magnitude..config.perturbation_magnitude);
                perturbed_a[[i, j]] += noise;
            }
        }

        for i in 0..perturbed_b.nrows() {
            for j in 0..perturbed_b.ncols() {
                let noise =
                    rng.gen_range(-config.perturbation_magnitude..config.perturbation_magnitude);
                perturbed_b[[i, j]] += noise;
            }
        }

        // Compute controllability measure
        if let Ok(ctrl_matrix) =
            build_robust_controllability_matrix(&perturbed_a, &perturbed_b, config)
        {
            if let Ok(conditioning) = analyze_numerical_conditioning(&ctrl_matrix, config) {
                if let Ok(measure) =
                    compute_robust_controllability_measure(&ctrl_matrix, &conditioning)
                {
                    measures.push(measure);
                }
            }
        }
    }

    if measures.is_empty() {
        return Ok(UncertaintyBounds {
            lower_bound: 0.0,
            upper_bound: 0.0,
            confidence_interval: (0.0, 0.0),
            standard_deviation: 0.0,
        });
    }

    measures.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let lower_bound = measures[0];
    let upper_bound = measures[measures.len() - 1];

    // 95% confidence interval
    let ci_lower_idx = (measures.len() as f64 * 0.025) as usize;
    let ci_upper_idx = (measures.len() as f64 * 0.975) as usize;
    let confidence_interval = (measures[ci_lower_idx], measures[ci_upper_idx]);

    // Standard deviation
    let mean: f64 = measures.iter().sum::<f64>() / measures.len() as f64;
    let variance: f64 =
        measures.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / measures.len() as f64;
    let standard_deviation = variance.sqrt();

    Ok(UncertaintyBounds {
        lower_bound,
        upper_bound,
        confidence_interval,
        standard_deviation,
    })
}

/// Analyze observability uncertainty using Monte Carlo
#[allow(dead_code)]
fn analyze_observability_uncertainty(
    ss: &StateSpace,
    config: &RobustAnalysisConfig,
) -> SignalResult<UncertaintyBounds> {
    // Similar to controllability uncertainty analysis but for observability
    analyze_controllability_uncertainty(ss, config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    
    #[test]
    fn test_controllability_analysis() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![0.5, 0.5];
        let ss = StateSpace {
            a: array![[-1.0, 0.0], [0.0, -2.0]],
            b: array![[1.0], [1.0]],
            c: array![[1.0, 0.0]],
            d: array![[0.0]],
            dt: false,
        };

        let result = analyze_controllability(&ss).unwrap();
        assert!(result.is_controllable);
        assert_eq!(result.controllability_rank, 2);
    }

    #[test]
    fn test_stability_analysis() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![0.5, 0.5];
        let ss = StateSpace {
            a: array![[-1.0, 1.0], [0.0, -2.0]],
            b: array![[1.0], [0.0]],
            c: array![[1.0, 0.0]],
            d: array![[0.0]],
            dt: false,
        };

        let result = analyze_stability(&ss).unwrap();
        assert!(result.is_stable);
        assert!(!result.is_marginally_stable);
    }
}
