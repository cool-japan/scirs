//! Mamba: Selective State Space Model
//! (Gu & Dao, 2023)
//!
//! Input-dependent (selective) state-space model where the SSM parameters
//! B, C, and the discretization step delta are functions of the input,
//! enabling content-aware reasoning with linear-time complexity.
//!
//! The core operation is the **selective scan**:
//!
//!   x_t = diag(exp(A * delta_t)) * x_{t-1}  +  delta_t * B_t * u_t
//!   y_t = C_t * x_t  +  D * u_t
//!
//! where B_t, C_t, delta_t are all projected from the input at time t.

use crate::error::{Result, TimeSeriesError};

// ---------------------------------------------------------------------------
// MambaConfig
// ---------------------------------------------------------------------------

/// Mamba block configuration.
#[derive(Debug, Clone)]
pub struct MambaConfig {
    /// D: model dimension (input/output width).
    pub model_dim: usize,
    /// N: SSM state expansion dimension (default: 16).
    pub state_dim: usize,
    /// E: inner expansion factor (default: 2).
    pub expand_factor: usize,
    /// Rank of the delta (dt) projection (default: ceil(D/16) or 1).
    pub dt_rank: usize,
    /// Minimum discretization step.
    pub dt_min: f64,
    /// Maximum discretization step.
    pub dt_max: f64,
    /// 1-D depthwise convolution kernel size (default: 4).
    pub conv_dim: usize,
}

impl Default for MambaConfig {
    fn default() -> Self {
        Self {
            model_dim: 64,
            state_dim: 16,
            expand_factor: 2,
            dt_rank: 4,
            dt_min: 0.001,
            dt_max: 0.1,
            conv_dim: 4,
        }
    }
}

impl MambaConfig {
    /// The inner (expanded) dimension: E * D.
    pub fn inner_dim(&self) -> usize {
        self.expand_factor * self.model_dim
    }
}

// ---------------------------------------------------------------------------
// MambaBlock
// ---------------------------------------------------------------------------

/// A single Mamba block implementing selective state-space computation.
///
/// Weight matrices are stored as `Vec<Vec<f64>>` (row-major).
/// This is a reference implementation prioritising clarity over speed.
#[derive(Debug, Clone)]
pub struct MambaBlock {
    // -- Projection weights --
    /// Input projection: (2 * inner_dim) x model_dim
    pub in_proj_weight: Vec<Vec<f64>>,
    /// 1-D depthwise convolution weight: conv_dim x 1 (per channel)
    pub conv1d_weight: Vec<f64>,
    /// x -> (dt_rank + 2*N): (dt_rank + 2*state_dim) x inner_dim
    pub x_proj_weight: Vec<Vec<f64>>,
    /// dt projection: inner_dim x dt_rank
    pub dt_proj_weight: Vec<Vec<f64>>,
    /// dt projection bias: inner_dim
    pub dt_proj_bias: Vec<f64>,
    /// Log of SSM A matrix: state_dim x inner_dim  (A = -exp(a_log))
    pub a_log: Vec<Vec<f64>>,
    /// Skip-connection parameter D: inner_dim
    pub d_param: Vec<f64>,
    /// Output projection: model_dim x inner_dim
    pub out_proj_weight: Vec<Vec<f64>>,
    /// Configuration.
    pub config: MambaConfig,
}

impl MambaBlock {
    /// Create a new Mamba block with deterministic initialisation.
    ///
    /// Weights are initialised with small values derived from dimension
    /// scaling (Kaiming-like 1/sqrt(fan_in)).
    ///
    /// # Errors
    ///
    /// Returns an error if any dimension is zero.
    pub fn new(config: MambaConfig) -> Result<Self> {
        Self::new_with_seed(config, 42)
    }

    /// Create a new Mamba block with a specified seed for the
    /// pseudo-random weight initialisation.
    ///
    /// # Errors
    ///
    /// Returns an error if any dimension is zero.
    pub fn new_with_seed(config: MambaConfig, seed: u64) -> Result<Self> {
        let d = config.model_dim;
        let n = config.state_dim;
        let inner = config.inner_dim();
        let dt_rank = config.dt_rank;
        let conv_dim = config.conv_dim;

        if d == 0 || n == 0 || inner == 0 || dt_rank == 0 || conv_dim == 0 {
            return Err(TimeSeriesError::InvalidInput(
                "all Mamba dimensions must be > 0".into(),
            ));
        }

        // Simple deterministic PRNG (xorshift64) for reproducibility
        let mut rng_state = seed;

        let in_scale = 1.0 / (d as f64).sqrt();
        let inner_scale = 1.0 / (inner as f64).sqrt();
        let dt_scale = 1.0 / (dt_rank as f64).sqrt();

        let in_proj_weight = xorshift_matrix(&mut rng_state, 2 * inner, d, in_scale);
        let conv1d_weight = xorshift_vec(&mut rng_state, conv_dim, 1.0 / (conv_dim as f64).sqrt());
        let x_proj_weight = xorshift_matrix(&mut rng_state, dt_rank + 2 * n, inner, inner_scale);
        let dt_proj_weight = xorshift_matrix(&mut rng_state, inner, dt_rank, dt_scale);
        let dt_proj_bias = xorshift_vec(&mut rng_state, inner, 0.01);

        // A_log initialised so that A = -exp(a_log) has reasonable decay
        // Use log-spaced values: a_log[i][j] = log(i+1) (so A = -(i+1))
        let a_log: Vec<Vec<f64>> = (0..n).map(|i| vec![((i + 1) as f64).ln(); inner]).collect();

        // D parameter: ones (like a residual connection)
        let d_param = vec![1.0_f64; inner];

        let out_proj_weight = xorshift_matrix(&mut rng_state, d, inner, inner_scale);

        Ok(Self {
            in_proj_weight,
            conv1d_weight,
            x_proj_weight,
            dt_proj_weight,
            dt_proj_bias,
            a_log,
            d_param,
            out_proj_weight,
            config,
        })
    }

    /// Forward pass through the Mamba block.
    ///
    /// `input` has shape `[seq_len][model_dim]`.
    /// Returns output of shape `[seq_len][model_dim]`.
    ///
    /// # Errors
    ///
    /// Returns an error if input dimensions are inconsistent.
    pub fn forward(&self, input: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        let seq_len = input.len();
        if seq_len == 0 {
            return Ok(vec![]);
        }

        let d = self.config.model_dim;
        let inner = self.config.inner_dim();
        let n = self.config.state_dim;
        let dt_rank = self.config.dt_rank;

        // Validate input dimensions
        for (t, row) in input.iter().enumerate() {
            if row.len() != d {
                return Err(TimeSeriesError::DimensionMismatch {
                    expected: d,
                    actual: row.len(),
                });
            }
        }

        // 1. Input projection: [seq_len, D] -> [seq_len, 2*inner]
        let projected = linear_batch(input, &self.in_proj_weight)?;
        // Split into x_branch and z_branch (gate)
        let mut x_branch: Vec<Vec<f64>> = Vec::with_capacity(seq_len);
        let mut z_branch: Vec<Vec<f64>> = Vec::with_capacity(seq_len);
        for row in &projected {
            x_branch.push(row[..inner].to_vec());
            z_branch.push(row[inner..].to_vec());
        }

        // 2. 1-D depthwise convolution on x_branch (causal, per-channel)
        let x_conv = depthwise_conv1d(&x_branch, &self.conv1d_weight)?;

        // 3. SiLU activation on x_conv
        let x_act = silu_batch(&x_conv);

        // 4. Project x_act to get (delta, B, C):
        //    [seq_len, inner] -> [seq_len, dt_rank + 2*N]
        let x_dbc = linear_batch(&x_act, &self.x_proj_weight)?;

        // Split into delta_raw (dt_rank), B (N), C (N)
        let mut delta_raw: Vec<Vec<f64>> = Vec::with_capacity(seq_len);
        let mut b_seq: Vec<Vec<f64>> = Vec::with_capacity(seq_len);
        let mut c_seq: Vec<Vec<f64>> = Vec::with_capacity(seq_len);
        for row in &x_dbc {
            delta_raw.push(row[..dt_rank].to_vec());
            b_seq.push(row[dt_rank..(dt_rank + n)].to_vec());
            c_seq.push(row[(dt_rank + n)..(dt_rank + 2 * n)].to_vec());
        }

        // 5. dt projection: [seq_len, dt_rank] -> [seq_len, inner]
        //    then softplus and clamp to [dt_min, dt_max]
        let delta_proj = linear_batch_bias(&delta_raw, &self.dt_proj_weight, &self.dt_proj_bias)?;
        let delta: Vec<Vec<f64>> = delta_proj
            .iter()
            .map(|row| {
                row.iter()
                    .map(|&v| {
                        let sp = softplus(v);
                        sp.clamp(self.config.dt_min, self.config.dt_max)
                    })
                    .collect()
            })
            .collect();

        // 6. Compute A = -exp(a_log): [N, inner]
        let a: Vec<Vec<f64>> = self
            .a_log
            .iter()
            .map(|row| row.iter().map(|&v| -v.exp()).collect())
            .collect();

        // 7. Selective scan
        let y_ssm = selective_scan(&x_act, &delta, &a, &b_seq, &c_seq, &self.d_param)?;

        // 8. Gating: y = y_ssm * silu(z_branch)
        let z_act = silu_batch(&z_branch);
        let mut gated: Vec<Vec<f64>> = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let row: Vec<f64> = y_ssm[t]
                .iter()
                .zip(z_act[t].iter())
                .map(|(y, z)| y * z)
                .collect();
            gated.push(row);
        }

        // 9. Output projection: [seq_len, inner] -> [seq_len, D]
        let output = linear_batch(&gated, &self.out_proj_weight)?;

        Ok(output)
    }
}

// ---------------------------------------------------------------------------
// Xorshift PRNG helpers for weight initialisation
// ---------------------------------------------------------------------------

/// Advance xorshift64 state and return a value in [-1, 1].
fn xorshift_next(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    ((*state as f64) / (u64::MAX as f64)) * 2.0 - 1.0
}

/// Generate a random matrix with xorshift64.
fn xorshift_matrix(state: &mut u64, rows: usize, cols: usize, scale: f64) -> Vec<Vec<f64>> {
    (0..rows)
        .map(|_| (0..cols).map(|_| xorshift_next(state) * scale).collect())
        .collect()
}

/// Generate a random vector with xorshift64.
fn xorshift_vec(state: &mut u64, len: usize, scale: f64) -> Vec<f64> {
    (0..len).map(|_| xorshift_next(state) * scale).collect()
}

// ---------------------------------------------------------------------------
// Selective Scan
// ---------------------------------------------------------------------------

/// Selective scan algorithm -- the core recurrence of Mamba.
///
/// For each time step t and each channel j (0..dim):
///
///   x_t[i, j] = exp(A[i, j] * delta_t[j]) * x_{t-1}[i, j]
///             + delta_t[j] * B_t[i] * u_t[j]
///
///   y_t[j] = sum_i C_t[i] * x_t[i, j]  +  D[j] * u_t[j]
///
/// # Arguments
///
/// * `u`     - Input: `[seq_len][dim]`
/// * `delta` - Discretization steps: `[seq_len][dim]`
/// * `a`     - SSM A matrix: `[state_dim][dim]` (should be negative)
/// * `b`     - Input-dependent B: `[seq_len][state_dim]`
/// * `c`     - Input-dependent C: `[seq_len][state_dim]`
/// * `d`     - Skip connection: `[dim]`
///
/// # Returns
///
/// Output `[seq_len][dim]`.
///
/// # Errors
///
/// Returns an error on dimension mismatches.
pub fn selective_scan(
    u: &[Vec<f64>],
    delta: &[Vec<f64>],
    a: &[Vec<f64>],
    b: &[Vec<f64>],
    c: &[Vec<f64>],
    d: &[f64],
) -> Result<Vec<Vec<f64>>> {
    let seq_len = u.len();
    if seq_len == 0 {
        return Ok(vec![]);
    }

    let dim = u[0].len();
    let state_dim = a.len();

    // Validate dimensions
    if d.len() != dim {
        return Err(TimeSeriesError::DimensionMismatch {
            expected: dim,
            actual: d.len(),
        });
    }
    if delta.len() != seq_len || b.len() != seq_len || c.len() != seq_len {
        return Err(TimeSeriesError::InvalidInput(
            "delta, b, c must have same seq_len as u".into(),
        ));
    }

    // State: [state_dim][dim]
    let mut x = vec![vec![0.0_f64; dim]; state_dim];
    let mut output = Vec::with_capacity(seq_len);

    for t in 0..seq_len {
        let u_t = &u[t];
        let delta_t = &delta[t];
        let b_t = &b[t];
        let c_t = &c[t];

        // Update state
        for i in 0..state_dim {
            for j in 0..dim {
                // a_bar = exp(A[i][j] * delta_t[j])
                let a_bar = (a[i][j] * delta_t[j]).exp();
                // b_bar = delta_t[j] * B_t[i]
                let b_bar = delta_t[j] * b_t[i];
                x[i][j] = a_bar * x[i][j] + b_bar * u_t[j];
            }
        }

        // Compute output
        let mut y_t = vec![0.0_f64; dim];
        for j in 0..dim {
            let mut val = d[j] * u_t[j];
            for i in 0..state_dim {
                val += c_t[i] * x[i][j];
            }
            y_t[j] = val;
        }
        output.push(y_t);
    }

    Ok(output)
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Batch linear transform: output[t] = weight * input[t]
/// weight: [out_dim x in_dim], input[t]: [in_dim]
fn linear_batch(input: &[Vec<f64>], weight: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
    let out_dim = weight.len();
    if out_dim == 0 {
        return Ok(input.iter().map(|_| vec![]).collect());
    }
    let in_dim = weight[0].len();

    let mut output = Vec::with_capacity(input.len());
    for row in input {
        if row.len() != in_dim {
            return Err(TimeSeriesError::DimensionMismatch {
                expected: in_dim,
                actual: row.len(),
            });
        }
        let mut out = vec![0.0_f64; out_dim];
        for i in 0..out_dim {
            let mut s = 0.0_f64;
            for j in 0..in_dim {
                s += weight[i][j] * row[j];
            }
            out[i] = s;
        }
        output.push(out);
    }
    Ok(output)
}

/// Batch linear transform with bias.
fn linear_batch_bias(
    input: &[Vec<f64>],
    weight: &[Vec<f64>],
    bias: &[f64],
) -> Result<Vec<Vec<f64>>> {
    let mut output = linear_batch(input, weight)?;
    for row in &mut output {
        for (j, val) in row.iter_mut().enumerate() {
            if j < bias.len() {
                *val += bias[j];
            }
        }
    }
    Ok(output)
}

/// Causal depthwise 1-D convolution.
///
/// Each channel is convolved independently with the same kernel.
/// Input: `[seq_len][channels]`, kernel: `[kernel_size]`.
fn depthwise_conv1d(input: &[Vec<f64>], kernel: &[f64]) -> Result<Vec<Vec<f64>>> {
    let seq_len = input.len();
    if seq_len == 0 {
        return Ok(vec![]);
    }
    let channels = input[0].len();
    let k_size = kernel.len();

    let mut output = Vec::with_capacity(seq_len);
    for t in 0..seq_len {
        let mut row = vec![0.0_f64; channels];
        for (ki, &kv) in kernel.iter().enumerate() {
            if t >= ki {
                for ch in 0..channels {
                    row[ch] += kv * input[t - ki][ch];
                }
            }
        }
        output.push(row);
    }
    Ok(output)
}

/// SiLU (Sigmoid Linear Unit): x * sigmoid(x)
#[inline]
fn silu(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

/// Softplus: log(1 + exp(x)), numerically stable.
#[inline]
fn softplus(x: f64) -> f64 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        0.0
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// Apply SiLU element-wise to a batch.
fn silu_batch(input: &[Vec<f64>]) -> Vec<Vec<f64>> {
    input
        .iter()
        .map(|row| row.iter().map(|&v| silu(v)).collect())
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_selective_scan_output_shape() {
        let seq_len = 10;
        let dim = 4;
        let state_dim = 3;

        let u: Vec<Vec<f64>> = vec![vec![1.0; dim]; seq_len];
        let delta: Vec<Vec<f64>> = vec![vec![0.01; dim]; seq_len];
        let a: Vec<Vec<f64>> = vec![vec![-1.0; dim]; state_dim];
        let b: Vec<Vec<f64>> = vec![vec![1.0; state_dim]; seq_len];
        let c: Vec<Vec<f64>> = vec![vec![1.0; state_dim]; seq_len];
        let d = vec![0.5; dim];

        let output = selective_scan(&u, &delta, &a, &b, &c, &d).expect("should succeed");
        assert_eq!(output.len(), seq_len);
        assert_eq!(output[0].len(), dim);
    }

    #[test]
    fn test_selective_scan_zero_delta() {
        // With delta = 0, exp(A*0) = 1, b_bar = 0, so state doesn't change
        // from initial zero. Output = D * u only.
        let seq_len = 5;
        let dim = 3;
        let state_dim = 2;

        let u: Vec<Vec<f64>> = vec![vec![2.0; dim]; seq_len];
        let delta: Vec<Vec<f64>> = vec![vec![0.0; dim]; seq_len];
        let a: Vec<Vec<f64>> = vec![vec![-1.0; dim]; state_dim];
        let b: Vec<Vec<f64>> = vec![vec![1.0; state_dim]; seq_len];
        let c: Vec<Vec<f64>> = vec![vec![1.0; state_dim]; seq_len];
        let d = vec![0.5; dim];

        let output = selective_scan(&u, &delta, &a, &b, &c, &d).expect("should succeed");
        for t in 0..seq_len {
            for j in 0..dim {
                let expected = d[j] * u[t][j]; // = 0.5 * 2.0 = 1.0
                assert!(
                    (output[t][j] - expected).abs() < 1e-12,
                    "t={} j={}: got {} expected {}",
                    t,
                    j,
                    output[t][j],
                    expected
                );
            }
        }
    }

    #[test]
    fn test_selective_scan_empty() {
        let output = selective_scan(&[], &[], &vec![vec![-1.0; 2]; 3], &[], &[], &[1.0, 1.0])
            .expect("should succeed");
        assert!(output.is_empty());
    }

    #[test]
    fn test_selective_scan_dimension_mismatch() {
        let u = vec![vec![1.0, 2.0]];
        let delta = vec![vec![0.01, 0.01]];
        let a = vec![vec![-1.0, -1.0]];
        let b = vec![vec![1.0]];
        let c = vec![vec![1.0]];
        let d_wrong = vec![1.0, 2.0, 3.0]; // wrong dimension
        assert!(selective_scan(&u, &delta, &a, &b, &c, &d_wrong).is_err());
    }

    #[test]
    fn test_mamba_block_creation() {
        let config = MambaConfig {
            model_dim: 8,
            state_dim: 4,
            expand_factor: 2,
            dt_rank: 2,
            dt_min: 0.001,
            dt_max: 0.1,
            conv_dim: 4,
        };
        let block = MambaBlock::new(config).expect("should succeed");
        assert_eq!(block.config.model_dim, 8);
        assert_eq!(block.config.inner_dim(), 16);
    }

    #[test]
    fn test_mamba_block_forward_dimensions() {
        let config = MambaConfig {
            model_dim: 8,
            state_dim: 4,
            expand_factor: 2,
            dt_rank: 2,
            dt_min: 0.001,
            dt_max: 0.1,
            conv_dim: 4,
        };
        let block = MambaBlock::new(config).expect("should succeed");

        let seq_len = 6;
        let input: Vec<Vec<f64>> = vec![vec![0.1; 8]; seq_len];
        let output = block.forward(&input).expect("should succeed");

        assert_eq!(output.len(), seq_len);
        assert_eq!(output[0].len(), 8);
    }

    #[test]
    fn test_mamba_block_empty_input() {
        let block = MambaBlock::new(MambaConfig::default()).expect("should succeed");
        let output = block.forward(&[]).expect("should succeed");
        assert!(output.is_empty());
    }

    #[test]
    fn test_mamba_block_wrong_input_dim() {
        let config = MambaConfig {
            model_dim: 8,
            ..MambaConfig::default()
        };
        let block = MambaBlock::new(config).expect("should succeed");
        let input = vec![vec![1.0; 4]]; // wrong: should be 8
        assert!(block.forward(&input).is_err());
    }

    #[test]
    fn test_mamba_block_with_seed_deterministic() {
        let config = MambaConfig {
            model_dim: 8,
            state_dim: 4,
            expand_factor: 2,
            dt_rank: 2,
            dt_min: 0.001,
            dt_max: 0.1,
            conv_dim: 4,
        };
        let b1 = MambaBlock::new_with_seed(config.clone(), 123).expect("b1");
        let b2 = MambaBlock::new_with_seed(config, 123).expect("b2");

        // Same seed should produce same weights
        assert_eq!(b1.in_proj_weight.len(), b2.in_proj_weight.len());
        for (r1, r2) in b1.in_proj_weight.iter().zip(b2.in_proj_weight.iter()) {
            for (v1, v2) in r1.iter().zip(r2.iter()) {
                assert!((v1 - v2).abs() < 1e-15);
            }
        }
    }

    #[test]
    fn test_mamba_default_config() {
        let config = MambaConfig::default();
        assert_eq!(config.model_dim, 64);
        assert_eq!(config.state_dim, 16);
        assert_eq!(config.expand_factor, 2);
        assert_eq!(config.inner_dim(), 128);
    }

    #[test]
    fn test_silu_values() {
        assert!((silu(0.0) - 0.0).abs() < 1e-10);
        // silu(large) ≈ large
        assert!((silu(10.0) - 10.0).abs() < 0.001);
        // silu(x) > 0 for x > 0
        assert!(silu(1.0) > 0.0);
        // silu(x) < 0 for x < 0
        assert!(silu(-1.0) < 0.0);
    }

    #[test]
    fn test_softplus_values() {
        assert!((softplus(0.0) - (2.0_f64).ln()).abs() < 1e-10);
        // softplus(large) ≈ large
        assert!((softplus(30.0) - 30.0).abs() < 1e-10);
        // softplus(very negative) ≈ 0
        assert!(softplus(-30.0).abs() < 1e-10);
    }

    #[test]
    fn test_depthwise_conv1d() {
        let input = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let kernel = vec![1.0, 0.5]; // [k0, k1] -> causal: y[t] = k0*x[t] + k1*x[t-1]

        let output = depthwise_conv1d(&input, &kernel).expect("should succeed");
        assert_eq!(output.len(), 3);

        // t=0: k0*x[0] = [1.0, 2.0]
        assert!((output[0][0] - 1.0).abs() < 1e-12);
        assert!((output[0][1] - 2.0).abs() < 1e-12);

        // t=1: k0*x[1] + k1*x[0] = [3.0 + 0.5, 4.0 + 1.0] = [3.5, 5.0]
        assert!((output[1][0] - 3.5).abs() < 1e-12);
        assert!((output[1][1] - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_mamba_forward_finite_output() {
        let config = MambaConfig {
            model_dim: 4,
            state_dim: 2,
            expand_factor: 2,
            dt_rank: 1,
            dt_min: 0.001,
            dt_max: 0.1,
            conv_dim: 2,
        };
        let block = MambaBlock::new_with_seed(config, 99).expect("should succeed");

        let input = vec![vec![0.1, -0.2, 0.3, -0.4]; 8];
        let output = block.forward(&input).expect("forward");

        for (t, row) in output.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                assert!(
                    val.is_finite(),
                    "output[{}][{}] = {} is not finite",
                    t,
                    j,
                    val
                );
            }
        }
    }
}
