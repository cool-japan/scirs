//! Sensitivity analysis tools
//!
//! This module provides tools for analyzing how solutions depend on parameters,
//! including local sensitivity analysis and global sensitivity indices.

use crate::common::IntegrateFloat;
use crate::error::{IntegrateError, IntegrateResult};
use crate::ode::{solve_ivp, ODEOptions};
use ndarray::{Array1, Array2, ArrayView1};
use rand::Rng;
use std::collections::HashMap;

// Type alias for complex return type
type SensitivityResult<F> = IntegrateResult<(HashMap<usize, Array1<F>>, HashMap<usize, Array1<F>>)>;

/// Parameter sensitivity information
#[derive(Clone)]
pub struct ParameterSensitivity<F: IntegrateFloat> {
    /// Parameter name
    pub name: String,
    /// Parameter index
    pub index: usize,
    /// Nominal value
    pub nominal_value: F,
    /// Sensitivity matrix (∂y/∂p)
    pub sensitivity: Array2<F>,
    /// Time points
    pub t_eval: Array1<F>,
}

/// Sensitivity analysis results
pub struct SensitivityAnalysis<F: IntegrateFloat> {
    /// Solution at nominal parameters
    pub nominal_solution: Array2<F>,
    /// Time points
    pub t_eval: Array1<F>,
    /// Parameter sensitivities
    pub sensitivities: Vec<ParameterSensitivity<F>>,
    /// First-order sensitivity indices (if computed)
    pub first_order_indices: Option<HashMap<String, Array1<F>>>,
    /// Total sensitivity indices (if computed)
    pub total_indices: Option<HashMap<String, Array1<F>>>,
}

impl<F: IntegrateFloat> SensitivityAnalysis<F> {
    /// Get sensitivity for a specific parameter
    pub fn get_sensitivity(&self, paramname: &str) -> Option<&ParameterSensitivity<F>> {
        self.sensitivities.iter().find(|s| s.name == paramname)
    }

    /// Compute relative sensitivities
    pub fn relative_sensitivities(&self) -> IntegrateResult<HashMap<String, Array2<F>>> {
        let mut result = HashMap::new();

        for sens in &self.sensitivities {
            let mut rel_sens = sens.sensitivity.clone();

            // Compute S_ij = (p_j / y_i) * (∂y_i/∂p_j)
            for i in 0..rel_sens.nrows() {
                for j in 0..rel_sens.ncols() {
                    let y_nominal = self.nominal_solution[[i, j]];
                    if y_nominal.abs() > F::epsilon() {
                        rel_sens[[i, j]] *= sens.nominal_value / y_nominal;
                    }
                }
            }

            result.insert(sens.name.clone(), rel_sens);
        }

        Ok(result)
    }

    /// Compute time-averaged sensitivities
    pub fn time_averaged_sensitivities(&self) -> HashMap<String, Array1<F>> {
        let mut result = HashMap::new();
        let n_time = self.t_eval.len();

        for sens in &self.sensitivities {
            let n_states = sens.sensitivity.ncols();
            let mut avg_sens = Array1::zeros(n_states);

            // Compute time average for each state variable
            for j in 0..n_states {
                let mut sum = F::zero();
                for i in 0..n_time {
                    sum += sens.sensitivity[[i, j]].abs();
                }
                avg_sens[j] = sum / F::from(n_time).unwrap();
            }

            result.insert(sens.name.clone(), avg_sens);
        }

        result
    }
}

/// Compute sensitivities using forward sensitivity analysis
#[allow(dead_code)]
pub fn compute_sensitivities<F, SysFunc, ParamFunc>(
    system: SysFunc,
    _parameters: ParamFunc,
    param_names: Vec<String>,
    nominal_params: ArrayView1<F>,
    y0: ArrayView1<F>,
    t_span: (F, F),
    _t_eval: Option<ArrayView1<F>>,
    options: Option<ODEOptions<F>>,
) -> IntegrateResult<SensitivityAnalysis<F>>
where
    F: IntegrateFloat + std::default::Default,
    SysFunc: Fn(F, ArrayView1<F>, ArrayView1<F>) -> Array1<F> + Clone,
    ParamFunc: Fn(usize) -> Array1<F>,
{
    let n_states = y0.len();
    let n_params = nominal_params.len();

    if param_names.len() != n_params {
        return Err(IntegrateError::ValueError(
            "Number of parameter _names must match number of _parameters".to_string(),
        ));
    }

    // Solve nominal system
    let opts = options.clone().unwrap_or_default();

    let nominal_result = solve_ivp(
        |t, y| system(t, y, nominal_params),
        [t_span.0, t_span.1],
        y0.to_owned(),
        Some(opts),
    )?;

    let t_points = nominal_result.t.clone();

    // Compute sensitivities for each parameter
    let mut sensitivities = Vec::new();

    for (param_idx, param_name) in param_names.iter().enumerate() {
        // Create augmented system for sensitivity equations
        let augmented_dim = n_states * (1 + 1); // States + sensitivity matrix
        let mut y0_aug = Array1::zeros(augmented_dim);

        // Initial conditions: y(0) and S(0) = 0
        y0_aug.slice_mut(ndarray::s![0..n_states]).assign(&y0);

        let system_clone = system.clone();
        let params = nominal_params.to_owned();

        // Augmented system: [dy/dt; dS/dt]
        let augmented_system = move |t: F, y_aug: ArrayView1<F>| -> Array1<F> {
            let y = y_aug.slice(ndarray::s![0..n_states]);
            let s = y_aug
                .slice(ndarray::s![n_states..])
                .to_owned()
                .into_shape_with_order((n_states,))
                .unwrap();

            // Compute f(t, y, p)
            let f = system_clone(t, y, params.view());

            // Compute ∂f/∂y using finite differences
            let eps = F::from(1e-8).unwrap();
            let mut df_dy = Array2::zeros((n_states, n_states));

            for j in 0..n_states {
                let mut y_pert = y.to_owned();
                y_pert[j] += eps;
                let f_pert = system_clone(t, y_pert.view(), params.view());

                for i in 0..n_states {
                    df_dy[[i, j]] = (f_pert[i] - f[i]) / eps;
                }
            }

            // Compute ∂f/∂p for the current parameter
            let mut params_pert = params.to_owned();
            params_pert[param_idx] += eps;
            let f_pert = system_clone(t, y, params_pert.view());
            let df_dp = (f_pert - &f) / eps;

            // dS/dt = ∂f/∂y * S + ∂f/∂p
            let ds_dt = df_dy.dot(&s) + df_dp;

            // Combine derivatives
            let mut result = Array1::zeros(augmented_dim);
            result.slice_mut(ndarray::s![0..n_states]).assign(&f);
            result.slice_mut(ndarray::s![n_states..]).assign(&ds_dt);

            result
        };

        // Solve augmented system
        let aug_opts = options.clone().unwrap_or_default();

        let aug_result = solve_ivp(
            augmented_system,
            [t_span.0, t_span.1],
            y0_aug,
            Some(aug_opts),
        )?;

        // Extract sensitivity matrix
        let aug_time = aug_result.t.len();
        let mut sensitivity = Array2::zeros((aug_time, n_states));
        for (i, sol) in aug_result.y.iter().enumerate() {
            let s = sol.slice(ndarray::s![n_states..]);
            sensitivity.row_mut(i).assign(&s);
        }

        sensitivities.push(ParameterSensitivity {
            name: param_name.clone(),
            index: param_idx,
            nominal_value: nominal_params[param_idx],
            sensitivity,
            t_eval: Array1::from_vec(aug_result.t.clone()),
        });
    }

    // Convert Vec<Array1<F>> to Array2<F>
    let n_points = nominal_result.t.len();
    let mut nominal_solution = Array2::zeros((n_points, n_states));
    for (i, sol) in nominal_result.y.iter().enumerate() {
        nominal_solution.row_mut(i).assign(sol);
    }

    Ok(SensitivityAnalysis {
        nominal_solution,
        t_eval: Array1::from_vec(t_points),
        sensitivities,
        first_order_indices: None,
        total_indices: None,
    })
}

/// Compute local sensitivity indices at a specific time
#[allow(dead_code)]
pub fn local_sensitivity_indices<F: IntegrateFloat>(
    analysis: &SensitivityAnalysis<F>,
    time_index: usize,
) -> IntegrateResult<HashMap<String, Array1<F>>> {
    let n_states = analysis.nominal_solution.ncols();
    let mut indices = HashMap::new();

    for sens in &analysis.sensitivities {
        let mut param_indices = Array1::zeros(n_states);

        for j in 0..n_states {
            let y_nominal = analysis.nominal_solution[[time_index, j]];
            let s_ij = sens.sensitivity[[time_index, j]];

            if y_nominal.abs() > F::epsilon() {
                // Normalized sensitivity _index
                param_indices[j] = (s_ij * sens.nominal_value / y_nominal).abs();
            }
        }

        indices.insert(sens.name.clone(), param_indices);
    }

    Ok(indices)
}

/// Sobol indices for global sensitivity analysis
pub struct SobolIndices<F: IntegrateFloat> {
    /// First-order indices S_i
    pub first_order: HashMap<String, F>,
    /// Total indices S_Ti
    pub total: HashMap<String, F>,
    /// Second-order indices S_ij (optional)
    pub second_order: Option<HashMap<(String, String), F>>,
}

/// Variance-based sensitivity analysis using Sobol method
pub struct SobolAnalysis<F: IntegrateFloat> {
    /// Number of samples
    n_samples: usize,
    /// Parameter bounds
    param_bounds: Vec<(F, F)>,
    /// Random seed for reproducibility
    seed: Option<u64>,
}

impl<F: IntegrateFloat> SobolAnalysis<F> {
    /// Create a new Sobol analysis
    pub fn new(n_samples: usize, param_bounds: Vec<(F, F)>) -> Self {
        SobolAnalysis {
            n_samples,
            param_bounds,
            seed: None,
        }
    }

    /// Set random seed for reproducibility
    pub fn with_seed(&mut self, seed: u64) -> &mut Self {
        self.seed = Some(seed);
        self
    }

    /// Compute Sobol indices
    pub fn compute_indices<Func>(
        &self,
        model: Func,
        param_names: Vec<String>,
    ) -> IntegrateResult<SobolIndices<F>>
    where
        Func: Fn(ArrayView1<F>) -> IntegrateResult<F> + Sync + Send,
    {
        let n_params = self.param_bounds.len();
        if param_names.len() != n_params {
            return Err(IntegrateError::ValueError(
                "Number of parameter _names must match bounds".to_string(),
            ));
        }

        // Generate quasi-random samples using Sobol sequence
        let sample_matrix_a = self.generate_sample_matrix();
        let sample_matrix_b = self.generate_sample_matrix();

        // Evaluate model at base samples
        let y_a = SobolAnalysis::<F>::evaluate_model(&model, &sample_matrix_a)?;
        let y_b = SobolAnalysis::<F>::evaluate_model(&model, &sample_matrix_b)?;

        // Compute variance
        let var_y = SobolAnalysis::<F>::compute_variance(&y_a, &y_b, self.n_samples);

        let mut first_order = HashMap::new();
        let mut total = HashMap::new();

        // Compute indices for each parameter
        for (i, name) in param_names.iter().enumerate() {
            // Create matrix C_i where column i comes from B, rest from A
            let sample_matrix_ci = self.create_mixed_matrix(&sample_matrix_a, &sample_matrix_b, i);
            let y_ci = SobolAnalysis::<F>::evaluate_model(&model, &sample_matrix_ci)?;

            // First-order index: S_i = V(E(Y|X_i)) / V(Y)
            let s_i = SobolAnalysis::<F>::compute_first_order_index(
                &y_a,
                &y_b,
                &y_ci,
                var_y,
                self.n_samples,
            );
            first_order.insert(name.clone(), s_i);

            // Total index: S_Ti = 1 - V(E(Y|X_~i)) / V(Y)
            let s_ti = SobolAnalysis::<F>::compute_total_index(&y_a, &y_ci, var_y, self.n_samples);
            total.insert(name.clone(), s_ti);
        }

        Ok(SobolIndices {
            first_order,
            total,
            second_order: None,
        })
    }

    /// Generate sample matrix using quasi-random sequences
    fn generate_sample_matrix(&self) -> Vec<Array1<F>> {
        let n_params = self.param_bounds.len();
        let mut samples = Vec::with_capacity(self.n_samples);

        // Simple uniform random sampling (should use Sobol sequence for better coverage)
        for i in 0..self.n_samples {
            let mut sample = Array1::zeros(n_params);
            for j in 0..n_params {
                let (low, high) = self.param_bounds[j];
                let u = F::from(i).unwrap() / F::from(self.n_samples - 1).unwrap();
                sample[j] = low + (high - low) * u;
            }
            samples.push(sample);
        }

        samples
    }

    /// Evaluate model at all sample points
    fn evaluate_model<Func>(model: &Func, samples: &[Array1<F>]) -> IntegrateResult<Vec<F>>
    where
        Func: Fn(ArrayView1<F>) -> IntegrateResult<F> + Sync + Send,
    {
        // Evaluate _model at each sample point
        let mut results = Vec::with_capacity(samples.len());
        for sample in samples {
            results.push(model(sample.view())?);
        }
        Ok(results)
    }

    /// Create mixed sample matrix for computing indices
    fn create_mixed_matrix(
        &self,
        matrix_a: &[Array1<F>],
        matrix_b: &[Array1<F>],
        param_idx: usize,
    ) -> Vec<Array1<F>> {
        let mut mixed = Vec::with_capacity(self.n_samples);

        for i in 0..self.n_samples {
            let mut sample = matrix_a[i].clone();
            sample[param_idx] = matrix_b[i][param_idx];
            mixed.push(sample);
        }

        mixed
    }

    /// Compute variance of model outputs
    fn compute_variance(y_a: &[F], y_b: &[F], n_samples: usize) -> F {
        let n = F::from(n_samples).unwrap();
        let mut sum = F::zero();
        let mut sum_sq = F::zero();

        for i in 0..n_samples {
            let y = (y_a[i] + y_b[i]) / F::from(2.0).unwrap();
            sum += y;
            sum_sq += y * y;
        }

        let mean = sum / n;
        sum_sq / n - mean * mean
    }

    /// Compute first-order Sobol index
    fn compute_first_order_index(
        y_a: &[F],
        y_b: &[F],
        y_ci: &[F],
        var_y: F,
        n_samples: usize,
    ) -> F {
        let n = F::from(n_samples).unwrap();
        let mut sum = F::zero();

        for i in 0..n_samples {
            sum += y_b[i] * (y_ci[i] - y_a[i]);
        }

        let v_i = sum / n;
        (v_i / var_y).max(F::zero()).min(F::one())
    }

    /// Compute total Sobol index
    fn compute_total_index(y_a: &[F], y_ci: &[F], var_y: F, n_samples: usize) -> F {
        let n = F::from(n_samples).unwrap();
        let mut sum = F::zero();

        for i in 0..n_samples {
            let diff = y_a[i] - y_ci[i];
            sum += diff * diff;
        }

        let e_i = sum / (F::from(2.0).unwrap() * n);
        (e_i / var_y).max(F::zero()).min(F::one())
    }
}

/// Extended Fourier Amplitude Sensitivity Test (eFAST)
pub struct EFAST<F: IntegrateFloat> {
    /// Number of samples
    n_samples: usize,
    /// Parameter bounds
    param_bounds: Vec<(F, F)>,
    /// Interference factor
    interference_factor: usize,
}

impl<F: IntegrateFloat> EFAST<F> {
    /// Create a new eFAST analysis
    pub fn new(n_samples: usize, param_bounds: Vec<(F, F)>) -> Self {
        EFAST {
            n_samples,
            param_bounds,
            interference_factor: 4,
        }
    }

    /// Set interference factor
    pub fn with_interference_factor(&mut self, factor: usize) -> &mut Self {
        self.interference_factor = factor;
        self
    }

    /// Compute sensitivity indices using eFAST
    pub fn compute_indices<Func>(
        &self,
        model: Func,
        param_names: Vec<String>,
    ) -> IntegrateResult<HashMap<String, F>>
    where
        Func: Fn(ArrayView1<F>) -> IntegrateResult<F>,
    {
        let n_params = self.param_bounds.len();
        if param_names.len() != n_params {
            return Err(IntegrateError::ValueError(
                "Number of parameter _names must match bounds".to_string(),
            ));
        }

        let mut indices = HashMap::new();
        let omega_max = (self.n_samples - 1) / (2 * self.interference_factor);

        // Compute indices for each parameter
        for (i, name) in param_names.iter().enumerate() {
            let omega_i = omega_max;
            let samples = self.generate_samples(i, omega_i);

            // Evaluate model
            let mut y_values = Vec::with_capacity(self.n_samples);
            for sample in &samples {
                y_values.push(model(sample.view())?);
            }

            // Compute Fourier coefficients
            let sensitivity = self.compute_fourier_sensitivity(&y_values, omega_i);
            indices.insert(name.clone(), sensitivity);
        }

        Ok(indices)
    }

    /// Generate parameter samples using search curve
    fn generate_samples(&self, _param_index: usize, omega: usize) -> Vec<Array1<F>> {
        let n_params = self.param_bounds.len();
        let mut samples = Vec::with_capacity(self.n_samples);

        for k in 0..self.n_samples {
            let s = F::from(k).unwrap() / F::from(self.n_samples).unwrap();
            let mut sample = Array1::zeros(n_params);

            for j in 0..n_params {
                let (low, high) = self.param_bounds[j];

                if j == _param_index {
                    // Use higher frequency for parameter of interest
                    let angle = F::from(2.0 * std::f64::consts::PI * omega as f64).unwrap() * s;
                    let x = (F::one() + angle.sin()) / F::from(2.0).unwrap();
                    sample[j] = low + (high - low) * x;
                } else {
                    // Use lower frequencies for other parameters
                    let omega_j = if j < _param_index { j + 1 } else { j };
                    let angle = F::from(2.0 * std::f64::consts::PI * omega_j as f64).unwrap() * s;
                    let x = (F::one() + angle.sin()) / F::from(2.0).unwrap();
                    sample[j] = low + (high - low) * x;
                }
            }

            samples.push(sample);
        }

        samples
    }

    /// Compute Fourier-based sensitivity
    fn compute_fourier_sensitivity(&self, y_values: &[F], omega: usize) -> F {
        let n = self.n_samples;
        let mut a_omega = F::zero();
        let mut b_omega = F::zero();

        for (k, y_value) in y_values.iter().enumerate().take(n) {
            let angle =
                F::from(2.0 * std::f64::consts::PI * omega as f64 * k as f64 / n as f64).unwrap();
            a_omega += *y_value * angle.cos();
            b_omega += *y_value * angle.sin();
        }

        a_omega *= F::from(2.0).unwrap() / F::from(n).unwrap();
        b_omega *= F::from(2.0).unwrap() / F::from(n).unwrap();

        // Return normalized sensitivity
        (a_omega * a_omega + b_omega * b_omega).sqrt()
    }
}

/// Parameter sensitivity ranking
#[allow(dead_code)]
pub fn rank_parameters<F: IntegrateFloat>(analysis: &SensitivityAnalysis<F>) -> Vec<(String, F)> {
    let averaged = analysis.time_averaged_sensitivities();
    let mut rankings: Vec<(String, F)> = Vec::new();

    for (name, sens) in averaged {
        // Use norm of sensitivity vector as ranking metric
        let mut norm = F::zero();
        for &s in sens.iter() {
            norm += s * s;
        }
        norm = norm.sqrt();
        rankings.push((name, norm));
    }

    // Sort by sensitivity (descending)
    rankings.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    rankings
}

/// Compute sensitivity-based parameter subset selection
#[allow(dead_code)]
pub fn select_important_parameters<F: IntegrateFloat>(
    analysis: &SensitivityAnalysis<F>,
    threshold: F,
) -> Vec<String> {
    let rankings = rank_parameters(analysis);
    let mut important = Vec::new();

    // Compute total sensitivity
    let total: F = rankings
        .iter()
        .map(|(_, s)| *s)
        .fold(F::zero(), |acc, x| acc + x);

    if total > F::epsilon() {
        let mut cumulative = F::zero();

        for (name, sens) in rankings {
            cumulative += sens;
            important.push(name);

            // Stop when we've captured threshold fraction of total sensitivity
            if cumulative / total >= threshold {
                break;
            }
        }
    }

    important
}

/// Global sensitivity analysis using Sobol indices
pub struct SobolSensitivity<F: IntegrateFloat> {
    /// Number of parameters
    n_params: usize,
    /// Number of samples
    n_samples: usize,
    /// Parameter bounds
    param_bounds: Vec<(F, F)>,
}

impl<F: IntegrateFloat + std::default::Default> SobolSensitivity<F> {
    /// Create a new Sobol sensitivity analyzer
    pub fn new(param_bounds: Vec<(F, F)>, n_samples: usize) -> Self {
        SobolSensitivity {
            n_params: param_bounds.len(),
            n_samples,
            param_bounds,
        }
    }

    /// Generate Sobol sample matrices
    pub fn generate_samples(&self) -> (Array2<F>, Array2<F>) {
        use rand::Rng;
        let mut rng = rand::rng();

        // Generate base sample matrix A
        let mut a_matrix = Array2::zeros((self.n_samples, self.n_params));
        for i in 0..self.n_samples {
            for j in 0..self.n_params {
                let (lower, upper) = self.param_bounds[j];
                let u: f64 = rng.random();
                a_matrix[[i, j]] = lower + (upper - lower) * F::from(u).unwrap();
            }
        }

        // Generate alternative sample matrix B
        let mut b_matrix = Array2::zeros((self.n_samples, self.n_params));
        for i in 0..self.n_samples {
            for j in 0..self.n_params {
                let (lower, upper) = self.param_bounds[j];
                let u: f64 = rng.random();
                b_matrix[[i, j]] = lower + (upper - lower) * F::from(u).unwrap();
            }
        }

        (a_matrix, b_matrix)
    }

    /// Compute first-order and total Sobol indices
    pub fn compute_indices<Func, SysFunc>(
        &self,
        system: SysFunc,
        y0_func: Func,
        t_span: (F, F),
        t_eval: ArrayView1<F>,
        options: Option<ODEOptions<F>>,
    ) -> SensitivityResult<F>
    where
        Func: Fn(ArrayView1<F>) -> Array1<F>,
        SysFunc: Fn(F, ArrayView1<F>, ArrayView1<F>) -> Array1<F> + Clone,
    {
        let (a_matrix, b_matrix) = self.generate_samples();
        let n_states = y0_func(a_matrix.row(0)).len();
        let n_time = t_eval.len();

        // Compute model outputs for base samples
        let mut y_a = Array2::zeros((self.n_samples, n_states * n_time));
        let mut y_b = Array2::zeros((self.n_samples, n_states * n_time));

        for i in 0..self.n_samples {
            let params_a = a_matrix.row(i);
            let params_b = b_matrix.row(i);

            let y0_a = y0_func(params_a);
            let y0_b = y0_func(params_b);

            let sol_a = solve_ivp(
                |t, y| system(t, y, params_a),
                [t_span.0, t_span.1],
                y0_a,
                options.clone(),
            )?;

            let sol_b = solve_ivp(
                |t, y| system(t, y, params_b),
                [t_span.0, t_span.1],
                y0_b,
                options.clone(),
            )?;

            // Flatten solutions
            for (j, t) in t_eval.iter().enumerate() {
                let idx_a = sol_a
                    .t
                    .iter()
                    .position(|&t_sol| (t_sol - *t).abs() < F::epsilon())
                    .unwrap_or(0);
                let idx_b = sol_b
                    .t
                    .iter()
                    .position(|&t_sol| (t_sol - *t).abs() < F::epsilon())
                    .unwrap_or(0);

                for k in 0..n_states {
                    y_a[[i, j * n_states + k]] = sol_a.y[idx_a][k];
                    y_b[[i, j * n_states + k]] = sol_b.y[idx_b][k];
                }
            }
        }

        // Compute variance of outputs
        let _mean_y = y_a.mean_axis(ndarray::Axis(0)).unwrap();
        let var_y = y_a.var_axis(ndarray::Axis(0), F::zero());

        let mut first_order_indices = HashMap::new();
        let mut total_indices = HashMap::new();

        // Compute indices for each parameter
        for param_idx in 0..self.n_params {
            // Create C_i matrix (all columns from B except i-th from A)
            let mut y_c_i = Array2::zeros((self.n_samples, n_states * n_time));

            for sample in 0..self.n_samples {
                let mut params_c_i = b_matrix.row(sample).to_owned();
                params_c_i[param_idx] = a_matrix[[sample, param_idx]];

                let y0_c = y0_func(params_c_i.view());
                let sol_c = solve_ivp(
                    |t, y| system(t, y, params_c_i.view()),
                    [t_span.0, t_span.1],
                    y0_c,
                    options.clone(),
                )?;

                for (j, t) in t_eval.iter().enumerate() {
                    let idx = sol_c
                        .t
                        .iter()
                        .position(|&t_sol| (t_sol - *t).abs() < F::epsilon())
                        .unwrap_or(0);
                    for k in 0..n_states {
                        y_c_i[[sample, j * n_states + k]] = sol_c.y[idx][k];
                    }
                }
            }

            // First-order index: S_i = V[E(Y|X_i)] / V(Y)
            let mut s_i = Array1::zeros(n_states * n_time);
            for j in 0..(n_states * n_time) {
                let mut sum = F::zero();
                for sample in 0..self.n_samples {
                    sum += y_a[[sample, j]] * (y_c_i[[sample, j]] - y_b[[sample, j]]);
                }
                let v_i = sum / F::from(self.n_samples).unwrap();
                s_i[j] = v_i / var_y[j];
            }
            first_order_indices.insert(param_idx, s_i);

            // Total index: ST_i = 1 - V[E(Y|X_~i)] / V(Y)
            let mut st_i = Array1::zeros(n_states * n_time);
            for j in 0..(n_states * n_time) {
                let mut sum = F::zero();
                for sample in 0..self.n_samples {
                    sum += y_b[[sample, j]] * (y_c_i[[sample, j]] - y_a[[sample, j]]);
                }
                let v_not_i = sum / F::from(self.n_samples).unwrap();
                st_i[j] = F::one() - v_not_i / var_y[j];
            }
            total_indices.insert(param_idx, st_i);
        }

        Ok((first_order_indices, total_indices))
    }
}

/// Morris screening method for parameter sensitivity
pub struct MorrisScreening<F: IntegrateFloat> {
    /// Number of parameters
    n_params: usize,
    /// Number of trajectories
    n_trajectories: usize,
    /// Step size
    delta: F,
    /// Parameter bounds
    param_bounds: Vec<(F, F)>,
    /// Grid levels
    grid_levels: usize,
}

impl<F: IntegrateFloat> MorrisScreening<F> {
    /// Create a new Morris screening analyzer
    pub fn new(param_bounds: Vec<(F, F)>, n_trajectories: usize, delta: F) -> Self {
        MorrisScreening {
            n_params: param_bounds.len(),
            n_trajectories,
            delta,
            param_bounds,
            grid_levels: 4,
        }
    }

    /// Create a new Morris screening analysis (legacy compatibility)
    pub fn new_simple(n_trajectories: usize, param_bounds: Vec<(F, F)>) -> Self {
        MorrisScreening {
            n_params: param_bounds.len(),
            n_trajectories,
            delta: F::from(0.1).unwrap(),
            param_bounds,
            grid_levels: 4,
        }
    }

    /// Set number of grid levels
    pub fn with_grid_levels(mut self, levels: usize) -> Self {
        self.grid_levels = levels;
        self
    }

    /// Generate Morris trajectories
    pub fn generate_trajectories(&self) -> Vec<Array2<F>> {
        use rand::seq::SliceRandom;
        let mut rng = rand::rng();

        let mut trajectories = Vec::new();

        for _ in 0..self.n_trajectories {
            let mut trajectory = Array2::zeros((self.n_params + 1, self.n_params));

            // Generate base point
            for j in 0..self.n_params {
                let (lower, upper) = self.param_bounds[j];
                let u: f64 = rng.random();
                trajectory[[0, j]] = lower + (upper - lower) * F::from(u).unwrap();
            }

            // Generate trajectory by changing one parameter at a time
            let mut param_order: Vec<usize> = (0..self.n_params).collect();
            param_order.shuffle(&mut rng);

            for (i, &param_idx) in param_order.iter().enumerate() {
                // Copy previous point
                for j in 0..self.n_params {
                    trajectory[[i + 1, j]] = trajectory[[i, j]];
                }

                // Change one parameter
                let (lower, upper) = self.param_bounds[param_idx];
                let range = upper - lower;
                let direction = if rng.gen::<bool>() {
                    F::one()
                } else {
                    -F::one()
                };
                trajectory[[i + 1, param_idx]] += direction * self.delta * range;

                // Ensure within bounds
                trajectory[[i + 1, param_idx]] =
                    trajectory[[i + 1, param_idx]].max(lower).min(upper);
            }

            trajectories.push(trajectory);
        }

        trajectories
    }

    /// Compute elementary effects from pre-generated trajectories
    pub fn compute_effects<Func>(
        &self,
        model: Func,
        trajectories: &[Array2<F>],
    ) -> IntegrateResult<(Array1<F>, Array1<F>)>
    where
        Func: Fn(ArrayView1<F>) -> IntegrateResult<F>,
    {
        let mut elementary_effects = vec![Vec::new(); self.n_params];

        for trajectory in trajectories {
            for i in 0..self.n_params {
                let y_before = model(trajectory.row(i))?;
                let y_after = model(trajectory.row(i + 1))?;

                // Find which parameter changed
                for j in 0..self.n_params {
                    if (trajectory[[i + 1, j]] - trajectory[[i, j]]).abs() > F::epsilon() {
                        let effect =
                            (y_after - y_before) / (trajectory[[i + 1, j]] - trajectory[[i, j]]);
                        elementary_effects[j].push(effect);
                        break;
                    }
                }
            }
        }

        // Compute mean and standard deviation of elementary effects
        let mut mu = Array1::zeros(self.n_params);
        let mut sigma = Array1::zeros(self.n_params);

        for j in 0..self.n_params {
            let effects = &elementary_effects[j];
            let n = F::from(effects.len()).unwrap();

            // Mean of absolute effects (mu*)
            let sum_abs: F = effects
                .iter()
                .map(|&e| e.abs())
                .fold(F::zero(), |acc, x| acc + x);
            mu[j] = sum_abs / n;

            // Standard deviation
            let mean: F = effects.iter().fold(F::zero(), |acc, &x| acc + x) / n;
            let variance: F = effects
                .iter()
                .map(|&e| (e - mean) * (e - mean))
                .fold(F::zero(), |acc, x| acc + x)
                / n;
            sigma[j] = variance.sqrt();
        }

        Ok((mu, sigma))
    }

    /// Compute elementary effects with parameter names (legacy compatibility)
    pub fn compute_effects_named<Func>(
        &self,
        model: Func,
        param_names: Vec<String>,
    ) -> IntegrateResult<HashMap<String, (F, F)>>
    where
        Func: Fn(ArrayView1<F>) -> IntegrateResult<F>,
    {
        let n_params = self.param_bounds.len();
        if param_names.len() != n_params {
            return Err(IntegrateError::ValueError(
                "Number of parameter _names must match bounds".to_string(),
            ));
        }

        let mut effects = HashMap::new();
        for name in &param_names {
            effects.insert(name.clone(), (F::zero(), F::zero()));
        }

        // Generate trajectories and compute elementary effects
        for _ in 0..self.n_trajectories {
            let trajectory = self.generate_trajectory_legacy(n_params);

            for i in 0..n_params {
                let p1 = trajectory[i].view();
                let p2 = trajectory[i + 1].view();

                let y1 = model(p1)?;
                let y2 = model(p2)?;

                // Find which parameter changed
                let mut changed_param = None;
                for j in 0..n_params {
                    if (p1[j] - p2[j]).abs() > F::epsilon() {
                        changed_param = Some(j);
                        break;
                    }
                }

                if let Some(j) = changed_param {
                    let delta = p2[j] - p1[j];
                    let ee = (y2 - y1) / delta;

                    let name = &param_names[j];
                    let (sum, sum_sq) = effects.get_mut(name).unwrap();
                    *sum += ee;
                    *sum_sq += ee * ee;
                }
            }
        }

        // Compute mean and standard deviation
        let n_traj = F::from(self.n_trajectories).unwrap();
        let mut results = HashMap::new();

        for (name, (sum, sum_sq)) in effects {
            let mu = sum / n_traj;
            let sigma = ((sum_sq / n_traj) - mu * mu).sqrt();
            results.insert(name, (mu.abs(), sigma));
        }

        Ok(results)
    }

    /// Generate a Morris trajectory (legacy compatibility)
    fn generate_trajectory_legacy(&self, n_params: usize) -> Vec<Array1<F>> {
        // Simplified trajectory generation
        let mut trajectory = Vec::new();
        let mut current = Array1::zeros(n_params);

        // Random starting point
        for i in 0..n_params {
            let (low, high) = self.param_bounds[i];
            current[i] = low + (high - low) * F::from(0.5).unwrap();
        }
        trajectory.push(current.clone());

        // Change one parameter at a time
        for i in 0..n_params {
            let (low, high) = self.param_bounds[i];
            let delta = (high - low) / F::from((self.grid_levels - 1) as f64).unwrap();
            current[i] += delta;
            trajectory.push(current.clone());
        }

        trajectory
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parameter_sensitivity() {
        // Simple linear ODE: dy/dt = -a*y
        let system =
            |_t: f64, y: ArrayView1<f64>, p: ArrayView1<f64>| Array1::from_vec(vec![-p[0] * y[0]]);

        let param_names = vec!["a".to_string()];
        let nominal_params = Array1::from_vec(vec![1.0]);
        let y0 = Array1::from_vec(vec![1.0]);
        let t_span = (0.0, 1.0);

        let analysis = compute_sensitivities(
            system,
            |_| Array1::from_vec(vec![1.0]),
            param_names,
            nominal_params.view(),
            y0.view(),
            t_span,
            None,
            None,
        );

        // Should complete without errors
        assert!(analysis.is_ok());
    }
}
