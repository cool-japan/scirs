//! Vector Autoregressive (VAR) models for multivariate time series
//!
//! Implements VAR, VARMA, VECM and related multivariate time series models

use scirs2_core::ndarray::{s, Array1, Array2, ArrayBase, Axis, Data, Ix2, ScalarOperand};
use scirs2_core::numeric::{Float, FromPrimitive, NumAssign, ToPrimitive};
use std::fmt::{Debug, Display};
use std::iter::Sum;

use crate::error::{Result, TimeSeriesError};

/// Vector Autoregressive (VAR) model
#[derive(Debug, Clone)]
pub struct VARModel<F> {
    /// Order of the VAR model
    pub order: usize,
    /// Number of variables
    pub n_vars: usize,
    /// Coefficient matrices for each lag
    pub coefficients: Vec<Array2<F>>,
    /// Intercept vector
    pub intercept: Array1<F>,
    /// Covariance matrix of residuals
    pub covariance: Array2<F>,
    /// Whether the model has been fitted
    pub is_fitted: bool,
    /// Training data supplied to the most recent [`VARModel::fit`] call.
    ///
    /// Retained so that [`VARModel::granger_causality`] can estimate the
    /// restricted (causality-constrained) regression it needs for a genuine
    /// nested F-test; a fitted model's coefficients/covariance alone are not
    /// sufficient for that since the restricted model is a *different*
    /// regression (with the candidate cause's lags dropped).
    pub training_data: Option<Array2<F>>,
}

impl<F> VARModel<F>
where
    F: Float + FromPrimitive + Debug + Display + ScalarOperand,
{
    /// Create a new VAR model
    pub fn new(_order: usize, nvars: usize) -> Result<Self> {
        if _order == 0 {
            return Err(TimeSeriesError::InvalidInput(
                "VAR _order must be at least 1".to_string(),
            ));
        }
        if nvars == 0 {
            return Err(TimeSeriesError::InvalidInput(
                "Number of variables must be at least 1".to_string(),
            ));
        }

        let coefficients = vec![Array2::zeros((nvars, nvars)); _order];
        let intercept = Array1::zeros(nvars);
        let covariance = Array2::eye(nvars);

        Ok(Self {
            order: _order,
            n_vars: nvars,
            coefficients,
            intercept,
            covariance,
            is_fitted: false,
            training_data: None,
        })
    }

    /// Fit the VAR model using OLS
    pub fn fit<S>(&mut self, data: &ArrayBase<S, Ix2>) -> Result<()>
    where
        S: Data<Elem = F>,
    {
        scirs2_core::validation::checkarray_finite(data, "data")?;

        let (t, k) = data.dim();
        if k != self.n_vars {
            return Err(TimeSeriesError::InvalidInput(format!(
                "Data must have {} variables, got {}",
                self.n_vars, k
            )));
        }

        if t <= self.order {
            return Err(TimeSeriesError::InvalidInput(format!(
                "Time series length ({}) must be greater than VAR order ({})",
                t, self.order
            )));
        }

        // Construct design matrix and response matrix
        let n_obs = t - self.order;
        let n_regressors = self.order * self.n_vars + 1; // +1 for intercept

        let mut x = Array2::zeros((n_obs, n_regressors));
        let mut y = Array2::zeros((n_obs, self.n_vars));

        // Fill matrices
        for i in 0..n_obs {
            // Response variables
            for j in 0..self.n_vars {
                y[[i, j]] = data[[i + self.order, j]];
            }

            // Intercept
            x[[i, 0]] = F::one();

            // Lagged variables
            for lag in 0..self.order {
                for var in 0..self.n_vars {
                    let col_idx = 1 + lag * self.n_vars + var;
                    x[[i, col_idx]] = data[[i + self.order - lag - 1, var]];
                }
            }
        }

        // OLS estimation: β = (X'X)^(-1)X'Y
        let xtx = x.t().dot(&x);
        let xty = x.t().dot(&y);

        // Solve for coefficients (simplified - would use proper linear solver)
        let beta = solve_normal_equations(&xtx, &xty)?;

        // Extract coefficients
        self.intercept = beta.column(0).to_owned();

        for lag in 0..self.order {
            let mut coef_matrix = Array2::zeros((self.n_vars, self.n_vars));
            for i in 0..self.n_vars {
                for j in 0..self.n_vars {
                    let row_idx = 1 + lag * self.n_vars + j;
                    coef_matrix[[i, j]] = beta[[row_idx, i]];
                }
            }
            self.coefficients[lag] = coef_matrix;
        }

        // Calculate residuals and covariance
        let fitted = x.dot(&beta);
        let residuals = &y - &fitted;
        self.covariance = residuals.t().dot(&residuals)
            / F::from(n_obs - n_regressors).expect("Failed to convert to float");

        // Retained for Granger-causality testing, which needs to re-estimate
        // a restricted regression from the original (undifferenced) data.
        self.training_data = Some(data.to_owned());

        self.is_fitted = true;
        Ok(())
    }

    /// Make predictions
    pub fn predict(&self, values: &Array2<F>, steps: usize) -> Result<Array2<F>> {
        if !self.is_fitted {
            return Err(TimeSeriesError::InvalidInput(
                "Model must be fitted before prediction".to_string(),
            ));
        }

        let (n, k) = values.dim();
        if k != self.n_vars {
            return Err(TimeSeriesError::InvalidInput(format!(
                "Data must have {} variables, got {}",
                self.n_vars, k
            )));
        }

        if n < self.order {
            return Err(TimeSeriesError::InvalidInput(format!(
                "Need at least {} observations for prediction, got {}",
                self.order, n
            )));
        }

        let mut predictions = Array2::zeros((steps, self.n_vars));
        let mut history = values.slice(s![n - self.order.., ..]).to_owned();

        for t in 0..steps {
            let mut pred = self.intercept.clone();

            for lag in 0..self.order {
                let lag_values = history.row(history.nrows() - 1 - lag);
                pred = pred + self.coefficients[lag].dot(&lag_values);
            }

            predictions.row_mut(t).assign(&pred);

            // Update history for next prediction
            if t < steps - 1 {
                // Shift history and add new prediction
                for i in 0..self.order - 1 {
                    let next_row = history.row(i + 1).to_owned();
                    history.row_mut(i).assign(&next_row);
                }
                history.row_mut(self.order - 1).assign(&pred);
            }
        }

        Ok(predictions)
    }

    /// Calculate impulse response function
    pub fn impulse_response(&self, periods: usize, shockvar: usize) -> Result<Array2<F>> {
        if !self.is_fitted {
            return Err(TimeSeriesError::InvalidInput(
                "Model must be fitted before calculating impulse response".to_string(),
            ));
        }

        if shockvar >= self.n_vars {
            return Err(TimeSeriesError::InvalidInput(format!(
                "Shock variable {} out of range (0-{})",
                shockvar,
                self.n_vars - 1
            )));
        }

        let mut responses = Array2::zeros((periods, self.n_vars));

        // Initial shock
        let mut shock = Array1::zeros(self.n_vars);
        shock[shockvar] = F::one();
        responses.row_mut(0).assign(&shock);

        // Calculate responses
        for t in 1..periods {
            let mut response = Array1::zeros(self.n_vars);

            for lag in 0..self.order.min(t) {
                let past_response = responses.row(t - lag - 1);
                response = response + self.coefficients[lag].dot(&past_response);
            }

            responses.row_mut(t).assign(&response);
        }

        Ok(responses)
    }

    /// Forecast error variance decomposition
    pub fn variance_decomposition(&self, periods: usize) -> Result<Vec<Array2<F>>> {
        if !self.is_fitted {
            return Err(TimeSeriesError::InvalidInput(
                "Model must be fitted before variance decomposition".to_string(),
            ));
        }

        let mut decomposition = vec![Array2::zeros((self.n_vars, self.n_vars)); periods];

        // Get impulse responses for each variable
        let mut impulse_responses = Vec::new();
        for i in 0..self.n_vars {
            impulse_responses.push(self.impulse_response(periods, i)?);
        }

        // Calculate cumulative variance contributions
        for (h, decomp_h) in decomposition.iter_mut().enumerate().take(periods) {
            let mut total_variance = Array1::zeros(self.n_vars);

            for (shock_var, impulse_response) in impulse_responses.iter().enumerate() {
                for response_var in 0..self.n_vars {
                    let mut contribution = F::zero();

                    for t in 0..=h {
                        let response = impulse_response[[t, response_var]];
                        contribution = contribution + response * response;
                    }

                    decomp_h[[response_var, shock_var]] = contribution;
                    total_variance[response_var] = total_variance[response_var] + contribution;
                }
            }

            // Normalize to percentages
            for response_var in 0..self.n_vars {
                if total_variance[response_var] > F::epsilon() {
                    for shock_var in 0..self.n_vars {
                        decomp_h[[response_var, shock_var]] =
                            decomp_h[[response_var, shock_var]] / total_variance[response_var];
                    }
                }
            }
        }

        Ok(decomposition)
    }

    /// Test whether `cause_var` Granger-causes `effectvar`.
    ///
    /// This is the standard Granger F-test: the *unrestricted* regression of
    /// `effectvar` on lags of every variable (the same regressors as the
    /// already-fitted VAR equation for `effectvar`) is compared against a
    /// *restricted* regression with `cause_var`'s lagged terms dropped. Under
    /// H0 ("`cause_var` does not Granger-cause `effectvar`", i.e. all of
    /// `cause_var`'s lag coefficients in the `effectvar` equation are jointly
    /// zero):
    ///
    /// ```text
    /// F = ((RSS_restricted - RSS_unrestricted) / q) / (RSS_unrestricted / (n - k))  ~  F(q, n - k)
    /// ```
    ///
    /// where `q` is the number of restrictions (one per lag of `cause_var`,
    /// so `q = self.order`), `n` is the number of effective observations, and
    /// `k` is the number of regressors in the unrestricted model. A small
    /// `p_value` is evidence against H0, i.e. evidence that `cause_var` does
    /// Granger-cause `effectvar`.
    pub fn granger_causality(&self, cause_var: usize, effectvar: usize) -> Result<(F, F)> {
        if !self.is_fitted {
            return Err(TimeSeriesError::InvalidInput(
                "Model must be fitted before testing Granger causality".to_string(),
            ));
        }

        if cause_var >= self.n_vars || effectvar >= self.n_vars {
            return Err(TimeSeriesError::InvalidInput(
                "Variable indices out of range".to_string(),
            ));
        }

        let data = self.training_data.as_ref().ok_or_else(|| {
            TimeSeriesError::InvalidModel(
                "VAR model has no stored training data (fit() must be called on this model \
                 instance before Granger-causality testing can re-estimate the restricted \
                 regression)"
                    .to_string(),
            )
        })?;

        let (t, _k) = data.dim();
        let n_obs = t - self.order;
        let n_regressors_unrestricted = self.order * self.n_vars + 1;
        let n_regressors_restricted = n_regressors_unrestricted - self.order;

        let mut y = Array1::zeros(n_obs);
        let mut x_unrestricted = Array2::zeros((n_obs, n_regressors_unrestricted));
        let mut x_restricted = Array2::zeros((n_obs, n_regressors_restricted));

        for i in 0..n_obs {
            y[i] = data[[i + self.order, effectvar]];

            x_unrestricted[[i, 0]] = F::one();
            x_restricted[[i, 0]] = F::one();

            let mut restricted_col = 1;
            for lag in 0..self.order {
                for var in 0..self.n_vars {
                    let value = data[[i + self.order - lag - 1, var]];
                    x_unrestricted[[i, 1 + lag * self.n_vars + var]] = value;

                    if var != cause_var {
                        x_restricted[[i, restricted_col]] = value;
                        restricted_col += 1;
                    }
                }
            }
        }

        let rss_unrestricted = ols_residual_sum_of_squares(&x_unrestricted, &y)?;
        let rss_restricted = ols_residual_sum_of_squares(&x_restricted, &y)?;

        let df1 = self.order; // number of restrictions
        if n_obs <= n_regressors_unrestricted {
            return Err(TimeSeriesError::InsufficientData {
                message: "Not enough observations for Granger-causality F-test".to_string(),
                required: n_regressors_unrestricted + 1,
                actual: n_obs,
            });
        }
        let df2 = n_obs - n_regressors_unrestricted; // denominator degrees of freedom

        let q = F::from(df1).expect("Failed to convert to float");
        let df2_f = F::from(df2).expect("Failed to convert to float");
        let denom = rss_unrestricted / df2_f;

        let f_stat = if denom > F::zero() {
            (((rss_restricted - rss_unrestricted) / q) / denom).max(F::zero())
        } else {
            F::zero()
        };

        // Reuse the crate's shared (pure-Rust, incomplete-beta-based)
        // F-distribution tail probability rather than re-deriving one.
        let f_stat_f64 = f_stat.to_f64().unwrap_or(0.0);
        let p_value_f64 = crate::causality::f_distribution_p_value(f_stat_f64, df1, df2);
        let p_value = F::from(p_value_f64).expect("Failed to convert to float");

        Ok((f_stat, p_value))
    }
}

/// Ordinary-least-squares residual sum of squares for regressing `y` on `x`.
///
/// Shared by [`VARModel::granger_causality`]'s restricted and unrestricted
/// regressions; delegates to [`solve_normal_equations`] so both use the same
/// (Cholesky-with-LU-fallback) numerical method as [`VARModel::fit`].
fn ols_residual_sum_of_squares<F>(x: &Array2<F>, y: &Array1<F>) -> Result<F>
where
    F: Float + FromPrimitive + Debug + Display + ScalarOperand,
{
    let xtx = x.t().dot(x);
    let y_col = y.clone().insert_axis(Axis(1));
    let xty = x.t().dot(&y_col);

    let beta = solve_normal_equations(&xtx, &xty)?;
    let fitted = x.dot(&beta);

    let mut rss = F::zero();
    for i in 0..y.len() {
        let resid = y[i] - fitted[[i, 0]];
        rss = rss + resid * resid;
    }
    Ok(rss)
}

/// Vector Moving Average (VMA) model
#[derive(Debug, Clone)]
pub struct VMAModel<F> {
    /// Order of the VMA model
    pub order: usize,
    /// Number of variables
    pub n_vars: usize,
    /// MA coefficient matrices
    pub ma_coefficients: Vec<Array2<F>>,
    /// Intercept vector
    pub intercept: Array1<F>,
    /// Innovation covariance
    pub covariance: Array2<F>,
}

/// Vector ARMA (VARMA) model
#[derive(Debug, Clone)]
pub struct VARMAModel<F> {
    /// VAR component
    pub var: VARModel<F>,
    /// VMA component
    pub vma: VMAModel<F>,
}

/// Vector Error Correction Model (VECM)
#[derive(Debug, Clone)]
pub struct VECMModel<F> {
    /// Number of cointegrating relationships
    pub rank: usize,
    /// Adjustment coefficients (alpha)
    pub adjustment: Array2<F>,
    /// Cointegrating vectors (beta)
    pub cointegration: Array2<F>,
    /// Short-run dynamics
    pub short_run: Vec<Array2<F>>,
    /// Deterministic terms
    pub deterministic: Array2<F>,
    /// Residual covariance
    pub covariance: Array2<F>,
    /// Whether the model is fitted
    pub is_fitted: bool,
}

impl<F> VECMModel<F>
where
    F: Float
        + FromPrimitive
        + Debug
        + Display
        + NumAssign
        + Sum
        + Send
        + Sync
        + ScalarOperand
        + 'static,
{
    /// Create a new VECM model
    pub fn new(_n_vars: usize, rank: usize, lagorder: usize) -> Result<Self> {
        if rank >= _n_vars {
            return Err(TimeSeriesError::InvalidInput(
                "Cointegration rank must be less than number of variables".to_string(),
            ));
        }
        if lagorder == 0 {
            return Err(TimeSeriesError::InvalidInput(
                "VECM lag order must be at least 1 (the equivalent level-VAR order)".to_string(),
            ));
        }

        let adjustment = Array2::zeros((_n_vars, rank));
        let cointegration = Array2::zeros((_n_vars, rank));
        let short_run = vec![Array2::zeros((_n_vars, _n_vars)); lagorder - 1];
        let deterministic = Array2::zeros((_n_vars, 2)); // constant and trend
        let covariance = Array2::eye(_n_vars);

        Ok(Self {
            rank,
            adjustment,
            cointegration,
            short_run,
            deterministic,
            covariance,
            is_fitted: false,
        })
    }

    /// Fit VECM using the Johansen (1988, 1991) reduced-rank-regression procedure.
    ///
    /// Estimates the cointegrated VECM
    ///
    /// ```text
    /// ΔY_t = α β' Y_{t-1} + Σ_{i=1}^{p} Γ_i ΔY_{t-i} + μ + ε_t     (p = lag_order - 1)
    /// ```
    ///
    /// by:
    /// 1. Regressing `ΔY_t` and `Y_{t-1}` each on the short-run regressors
    ///    `Z_t = [1, ΔY_{t-1}, ..., ΔY_{t-p}]` (an *unrestricted constant*,
    ///    i.e. Johansen's deterministic-trend "Case 3"; no linear trend term
    ///    is estimated) to obtain residuals `R0`, `R1`.
    /// 2. Forming the residual product-moment matrices `S00, S01, S11` and
    ///    solving the generalized symmetric eigenvalue problem
    ///    `S10 S00⁻¹ S01 v = λ S11 v` (via a Cholesky-whitening reduction to a
    ///    standard symmetric eigenproblem, solved by `scirs2_linalg::eigh`).
    /// 3. Taking `β` as the eigenvectors for the `self.rank` largest
    ///    eigenvalues (S11-orthonormalized, so `α = S01 β` directly), then
    ///    jointly re-estimating `α`, `Γ_i`, `μ` by OLS regression of `ΔY_t` on
    ///    `[β'Y_{t-1}, Z_t]` (equivalent to the analytic `α` by the
    ///    Frisch–Waugh–Lovell theorem, and additionally yields `Γ_i`, `μ`).
    ///
    /// Reference: S. Johansen, "Statistical Analysis of Cointegration
    /// Vectors", Journal of Economic Dynamics and Control 12 (1988) 231-254.
    pub fn fit<S>(&mut self, data: &ArrayBase<S, Ix2>) -> Result<()>
    where
        S: Data<Elem = F>,
    {
        scirs2_core::validation::checkarray_finite(data, "data")?;

        let (t, k) = data.dim();
        let n = self.adjustment.nrows();
        if k != n {
            return Err(TimeSeriesError::InvalidInput(format!(
                "Data must have {n} variables, got {k}"
            )));
        }

        let lag_order = self.short_run.len() + 1;
        let p = self.short_run.len(); // number of lagged-difference regressors
        if t <= lag_order + n {
            return Err(TimeSeriesError::InsufficientData {
                message: "Time series too short for the Johansen procedure".to_string(),
                required: lag_order + n + 1,
                actual: t,
            });
        }

        let n_obs = t - lag_order;
        let n_short_run = 1 + p * n; // constant + p lagged-difference blocks

        // Build ΔY_t (response), Y_{t-1} (cointegration regressor), and
        // Z_t = [1, ΔY_{t-1}, ..., ΔY_{t-p}] (short-run regressors).
        let mut dy = Array2::<F>::zeros((n_obs, n));
        let mut y_lag = Array2::<F>::zeros((n_obs, n));
        let mut z = Array2::<F>::zeros((n_obs, n_short_run));

        for i in 0..n_obs {
            let time = lag_order + i;
            for var in 0..n {
                dy[[i, var]] = data[[time, var]] - data[[time - 1, var]];
                y_lag[[i, var]] = data[[time - 1, var]];
            }
            z[[i, 0]] = F::one();
            for lag in 1..=p {
                for var in 0..n {
                    z[[i, 1 + (lag - 1) * n + var]] =
                        data[[time - lag, var]] - data[[time - lag - 1, var]];
                }
            }
        }

        // Step 1: partial out the short-run dynamics.
        let r0 = ols_residuals(&z, &dy)?;
        let r1 = ols_residuals(&z, &y_lag)?;

        // Step 2: residual product-moment matrices.
        let n_obs_f = F::from(n_obs).expect("Failed to convert to float");
        let s00 = r0.t().dot(&r0) / n_obs_f;
        let s01 = r0.t().dot(&r1) / n_obs_f;
        let s11 = r1.t().dot(&r1) / n_obs_f;

        // A = S10 S00^{-1} S01 (symmetric): solve S00 X = S01, then A = S01' X.
        let s00_inv_s01 = solve_normal_equations(&s00, &s01)?;
        let mut a_mat = s01.t().dot(&s00_inv_s01);
        symmetrize(&mut a_mat);

        // Step 3: generalized eigenproblem A v = λ S11 v via Cholesky whitening:
        // S11 = L L', M = L^{-1} A L^{-T}, eigh(M) = (λ, U), V = L^{-T} U.
        // V is then S11-orthonormal (v_i' S11 v_j = δ_ij), matching Johansen's
        // normalization, so α = S01 β holds directly for β = leading columns of V.
        let l = cholesky_factor(&s11)?;
        let l_inv = invert_lower_triangular(&l)?;
        let mut m = l_inv.dot(&a_mat).dot(&l_inv.t());
        symmetrize(&mut m);

        let (eigenvalues, u) = scirs2_linalg::eigh(&m.view(), None).map_err(|e| {
            TimeSeriesError::ComputationError(format!(
                "Johansen procedure eigendecomposition failed: {e}"
            ))
        })?;
        let v_all = l_inv.t().dot(&u);

        // Sort by descending eigenvalue: the largest eigenvalues correspond to
        // the strongest cointegrating relationships.
        let mut order: Vec<usize> = (0..eigenvalues.len()).collect();
        order.sort_by(|&i, &j| {
            eigenvalues[j]
                .partial_cmp(&eigenvalues[i])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let rank = self.rank;
        let mut beta = Array2::<F>::zeros((n, rank));
        for (new_col, &old_col) in order.iter().take(rank).enumerate() {
            for row in 0..n {
                beta[[row, new_col]] = v_all[[row, old_col]];
            }
        }

        // Step 4: joint regression of ΔY_t on [β'Y_{t-1}, Z_t] gives (by
        // Frisch–Waugh–Lovell) the same α as S01·β, plus Γ_i and μ.
        let ect = y_lag.dot(&beta); // error-correction term(s), n_obs x rank
        let mut w = Array2::<F>::zeros((n_obs, rank + n_short_run));
        for i in 0..n_obs {
            for j in 0..rank {
                w[[i, j]] = ect[[i, j]];
            }
            for j in 0..n_short_run {
                w[[i, rank + j]] = z[[i, j]];
            }
        }

        let theta = solve_normal_equations(&w.t().dot(&w), &w.t().dot(&dy))?;

        let mut alpha = Array2::<F>::zeros((n, rank));
        for i in 0..n {
            for j in 0..rank {
                alpha[[i, j]] = theta[[j, i]];
            }
        }

        let mut short_run = vec![Array2::<F>::zeros((n, n)); p];
        for lag in 0..p {
            let mut gamma = Array2::<F>::zeros((n, n));
            for i in 0..n {
                for var in 0..n {
                    let row_idx = rank + 1 + lag * n + var;
                    gamma[[i, var]] = theta[[row_idx, i]];
                }
            }
            short_run[lag] = gamma;
        }

        let mut deterministic = Array2::<F>::zeros((n, 2));
        for i in 0..n {
            // Constant term (column 0); no linear trend is estimated (column 1
            // stays zero — only Case 3, "unrestricted constant", is supported).
            deterministic[[i, 0]] = theta[[rank, i]];
        }

        let fitted = w.dot(&theta);
        let residuals = &dy - &fitted;
        let dof = n_obs.saturating_sub(rank + n_short_run).max(1);
        let covariance =
            residuals.t().dot(&residuals) / F::from(dof).expect("Failed to convert to float");

        self.adjustment = alpha;
        self.cointegration = beta;
        self.short_run = short_run;
        self.deterministic = deterministic;
        self.covariance = covariance;
        self.is_fitted = true;
        Ok(())
    }

    /// Convert the fitted VECM to its equivalent level-VAR(`lag_order`)
    /// representation, using the standard algebraic identity
    ///
    /// ```text
    /// Φ_1 = I + Π + Γ_1,      Φ_j = Γ_j - Γ_{j-1}  (2 ≤ j ≤ p),      Φ_{p+1} = -Γ_p
    /// ```
    ///
    /// where `Π = α β'` and `p = lag_order - 1`, derived by substituting
    /// `Y_t = Y_{t-1} + ΔY_t` into the fitted VECM equation and collecting
    /// terms in `Y_{t-1}, ..., Y_{t-lag_order}`.
    pub fn to_var(&self) -> Result<VARModel<F>> {
        if !self.is_fitted {
            return Err(TimeSeriesError::InvalidInput(
                "VECM must be fitted before conversion to VAR".to_string(),
            ));
        }

        let n = self.adjustment.nrows();
        let p = self.short_run.len();
        let lag_order = p + 1;
        let mut var = VARModel::new(lag_order, n)?;

        let pi = self.adjustment.dot(&self.cointegration.t());
        let identity: Array2<F> = Array2::eye(n);

        var.coefficients[0] = match self.short_run.first() {
            Some(gamma_1) => &identity + &pi + gamma_1,
            None => &identity + &pi,
        };

        for j in 1..p {
            var.coefficients[j] = &self.short_run[j] - &self.short_run[j - 1];
        }

        if let Some(gamma_p) = self.short_run.last() {
            var.coefficients[lag_order - 1] = gamma_p.mapv(|x| -x);
        }

        var.intercept = self.deterministic.column(0).to_owned();
        var.covariance = self.covariance.clone();
        var.is_fitted = true;

        Ok(var)
    }
}

/// Multivariate OLS residuals of regressing each column of `y` on `x`.
fn ols_residuals<F>(x: &Array2<F>, y: &Array2<F>) -> Result<Array2<F>>
where
    F: Float + FromPrimitive + Debug + Display + ScalarOperand,
{
    let beta = solve_normal_equations(&x.t().dot(x), &x.t().dot(y))?;
    Ok(y - &x.dot(&beta))
}

/// Symmetrize a square matrix in place: `a := (a + a') / 2`.
///
/// Used to absorb floating-point asymmetry (e.g. from summation order)
/// before feeding a matrix that is symmetric in exact arithmetic to a
/// symmetric eigensolver, which requires exact symmetry.
fn symmetrize<F>(a: &mut Array2<F>)
where
    F: Float,
{
    let n = a.nrows();
    let two = F::one() + F::one();
    for i in 0..n {
        for j in (i + 1)..n {
            let avg = (a[[i, j]] + a[[j, i]]) / two;
            a[[i, j]] = avg;
            a[[j, i]] = avg;
        }
    }
}

/// Lower-triangular Cholesky factor `L` such that `a = L L'`.
///
/// A small ridge is added to the diagonal for numerical robustness, since
/// callers use this on residual product-moment matrices that are only
/// positive semi-definite in exact arithmetic (e.g. `S11` in the Johansen
/// procedure) and can be numerically singular for small samples.
fn cholesky_factor<F>(a: &Array2<F>) -> Result<Array2<F>>
where
    F: Float + FromPrimitive + Debug + Display + ScalarOperand,
{
    let n = a.nrows();
    let ridge = F::from(1e-10).expect("Failed to convert constant to float");
    let mut l = Array2::<F>::zeros((n, n));

    for i in 0..n {
        for j in 0..=i {
            let mut sum = F::zero();
            for m in 0..j {
                sum = sum + l[[i, m]] * l[[j, m]];
            }
            if i == j {
                let val = a[[j, j]] + ridge - sum;
                if val <= F::zero() {
                    return Err(TimeSeriesError::NumericalInstability(
                        "Matrix is not positive definite in Cholesky factorization".to_string(),
                    ));
                }
                l[[j, j]] = val.sqrt();
            } else {
                if l[[j, j]] == F::zero() {
                    return Err(TimeSeriesError::NumericalInstability(
                        "Zero pivot in Cholesky factorization".to_string(),
                    ));
                }
                l[[i, j]] = (a[[i, j]] - sum) / l[[j, j]];
            }
        }
    }

    Ok(l)
}

/// Invert a lower-triangular matrix by forward substitution.
fn invert_lower_triangular<F>(l: &Array2<F>) -> Result<Array2<F>>
where
    F: Float + FromPrimitive + Debug + Display + ScalarOperand,
{
    let n = l.nrows();
    let mut inv = Array2::<F>::zeros((n, n));

    for col in 0..n {
        for i in 0..n {
            let mut sum = if i == col { F::one() } else { F::zero() };
            for j in 0..i {
                sum = sum - l[[i, j]] * inv[[j, col]];
            }
            if l[[i, i]].abs() <= F::from(1e-14).expect("Failed to convert constant to float") {
                return Err(TimeSeriesError::NumericalInstability(
                    "Singular matrix while inverting Cholesky factor".to_string(),
                ));
            }
            inv[[i, col]] = sum / l[[i, i]];
        }
    }

    Ok(inv)
}

/// Helper function to solve normal equations (X'X)β = X'Y
#[allow(dead_code)]
fn solve_normal_equations<F>(xtx: &Array2<F>, xty: &Array2<F>) -> Result<Array2<F>>
where
    F: Float + FromPrimitive + Debug + Display + ScalarOperand,
{
    let n = xtx.nrows();
    let _k = xty.ncols();

    if n != xtx.ncols() {
        return Err(TimeSeriesError::InvalidInput(
            "X'X matrix must be square".to_string(),
        ));
    }

    if n != xty.nrows() {
        return Err(TimeSeriesError::InvalidInput(
            "Dimensions of X'X and X'Y do not match".to_string(),
        ));
    }

    // Try Cholesky decomposition first (for positive definite matrices)
    if let Ok(beta) = solve_cholesky(xtx, xty) {
        return Ok(beta);
    }

    // Fall back to LU decomposition with partial pivoting
    solve_lu_decomposition(xtx, xty)
}

/// Solve using Cholesky decomposition
#[allow(dead_code)]
fn solve_cholesky<F>(a: &Array2<F>, b: &Array2<F>) -> Result<Array2<F>>
where
    F: Float + FromPrimitive + Debug + Display + ScalarOperand,
{
    let n = a.nrows();
    let k = b.ncols();

    // Cholesky decomposition: A = LL^T
    let mut l = Array2::<F>::zeros((n, n));

    for i in 0..n {
        for j in 0..=i {
            if i == j {
                // Diagonal elements
                let mut sum = F::zero();
                for k in 0..j {
                    sum = sum + l[[j, k]] * l[[j, k]];
                }
                let val = a[[j, j]] - sum;
                if val <= F::zero() {
                    return Err(TimeSeriesError::NumericalInstability(
                        "Matrix is not positive definite for Cholesky decomposition".to_string(),
                    ));
                }
                l[[j, j]] = val.sqrt();
            } else {
                // Lower triangular elements
                let mut sum = F::zero();
                for k in 0..j {
                    sum = sum + l[[i, k]] * l[[j, k]];
                }
                if l[[j, j]] == F::zero() {
                    return Err(TimeSeriesError::NumericalInstability(
                        "Zero pivot in Cholesky decomposition".to_string(),
                    ));
                }
                l[[i, j]] = (a[[i, j]] - sum) / l[[j, j]];
            }
        }
    }

    // Solve Ly = b for each column of b
    let mut y = Array2::<F>::zeros((n, k));
    for col in 0..k {
        for i in 0..n {
            let mut sum = F::zero();
            for j in 0..i {
                sum = sum + l[[i, j]] * y[[j, col]];
            }
            y[[i, col]] = (b[[i, col]] - sum) / l[[i, i]];
        }
    }

    // Solve L^T x = y for each column
    let mut x = Array2::<F>::zeros((n, k));
    for col in 0..k {
        for i in (0..n).rev() {
            let mut sum = F::zero();
            for j in (i + 1)..n {
                sum = sum + l[[j, i]] * x[[j, col]];
            }
            x[[i, col]] = (y[[i, col]] - sum) / l[[i, i]];
        }
    }

    Ok(x)
}

/// Solve using LU decomposition with partial pivoting
#[allow(dead_code)]
fn solve_lu_decomposition<F>(a: &Array2<F>, b: &Array2<F>) -> Result<Array2<F>>
where
    F: Float + FromPrimitive + Debug + Display + ScalarOperand,
{
    let n = a.nrows();
    let k = b.ncols();

    // Create working copies
    let mut lu = a.clone();
    let mut b_work = b.clone();
    let mut perm = (0..n).collect::<Vec<_>>();

    // LU decomposition with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_val = lu[[col, col]].abs();
        let mut max_row = col;

        for row in (col + 1)..n {
            let val = lu[[row, col]].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }

        // Swap rows if needed
        if max_row != col {
            for j in 0..n {
                let temp = lu[[col, j]];
                lu[[col, j]] = lu[[max_row, j]];
                lu[[max_row, j]] = temp;
            }

            for j in 0..k {
                let temp = b_work[[col, j]];
                b_work[[col, j]] = b_work[[max_row, j]];
                b_work[[max_row, j]] = temp;
            }

            perm.swap(col, max_row);
        }

        // Check for near-zero pivot
        if lu[[col, col]].abs() < F::from(1e-12).expect("Failed to convert constant to float") {
            return Err(TimeSeriesError::NumericalInstability(
                "Near-zero pivot in LU decomposition".to_string(),
            ));
        }

        // Eliminate below pivot
        for row in (col + 1)..n {
            let factor = lu[[row, col]] / lu[[col, col]];
            lu[[row, col]] = factor; // Store multiplier

            for j in (col + 1)..n {
                lu[[row, j]] = lu[[row, j]] - factor * lu[[col, j]];
            }

            for j in 0..k {
                b_work[[row, j]] = b_work[[row, j]] - factor * b_work[[col, j]];
            }
        }
    }

    // Back substitution
    let mut x = Array2::<F>::zeros((n, k));
    for col in 0..k {
        // Copy solution
        for i in 0..n {
            x[[i, col]] = b_work[[i, col]];
        }

        // Solve Ux = y
        for i in (0..n).rev() {
            let mut sum = F::zero();
            for j in (i + 1)..n {
                sum = sum + lu[[i, j]] * x[[j, col]];
            }
            x[[i, col]] = (x[[i, col]] - sum) / lu[[i, i]];
        }
    }

    Ok(x)
}

/// Model selection criteria
#[derive(Debug, Clone, Copy)]
pub enum SelectionCriterion {
    /// Akaike Information Criterion
    AIC,
    /// Bayesian Information Criterion
    BIC,
    /// Hannan-Quinn Information Criterion
    HQC,
    /// Final Prediction Error
    FPE,
}

/// Select optimal VAR order
#[allow(dead_code)]
pub fn select_var_order<S, F>(
    data: &ArrayBase<S, Ix2>,
    max_order: usize,
    criterion: SelectionCriterion,
) -> Result<usize>
where
    S: Data<Elem = F>,
    F: Float + FromPrimitive + Debug + Display + ScalarOperand,
{
    let (t, k) = data.dim();
    let mut best_order = 1;
    let mut best_criterion = F::infinity();

    for _order in 1..=max_order {
        if t <= _order + 1 {
            break;
        }

        let mut model = VARModel::new(_order, k)?;
        model.fit(data)?;

        let log_det = matrix_log_determinant(&model.covariance);
        let n_params = _order * k * k + k;

        let criterion_value = match criterion {
            SelectionCriterion::AIC => {
                log_det
                    + F::from(2.0).expect("Failed to convert constant to float")
                        * F::from(n_params).expect("Failed to convert to float")
                        / F::from(t).expect("Failed to convert to float")
            }
            SelectionCriterion::BIC => {
                log_det
                    + F::from(n_params).expect("Failed to convert to float").ln()
                        * F::from(t).expect("Failed to convert to float")
                        / F::from(t).expect("Failed to convert to float")
            }
            SelectionCriterion::HQC => {
                log_det
                    + F::from(2.0).expect("Failed to convert constant to float")
                        * F::from(n_params).expect("Failed to convert to float").ln()
                        * F::from(t).expect("Failed to convert to float").ln()
                        / F::from(t).expect("Failed to convert to float")
            }
            SelectionCriterion::FPE => {
                let factor = (F::from(t).expect("Failed to convert to float")
                    + F::from(n_params).expect("Failed to convert to float"))
                    / (F::from(t).expect("Failed to convert to float")
                        - F::from(n_params).expect("Failed to convert to float"));
                log_det + factor.ln()
            }
        };

        if criterion_value < best_criterion {
            best_criterion = criterion_value;
            best_order = _order;
        }
    }

    Ok(best_order)
}

/// Calculate log determinant of a matrix using LU decomposition
#[allow(dead_code)]
fn matrix_log_determinant<F>(matrix: &Array2<F>) -> F
where
    F: Float + FromPrimitive + Debug + Display + ScalarOperand,
{
    let n = matrix.nrows();
    if n != matrix.ncols() {
        return F::neg_infinity(); // Invalid _matrix
    }

    if n == 0 {
        return F::zero();
    }

    // Create working copy for LU decomposition
    let mut lu = matrix.clone();
    let mut sign = F::one();

    // LU decomposition with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_val = lu[[col, col]].abs();
        let mut max_row = col;

        for row in (col + 1)..n {
            let val = lu[[row, col]].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }

        // Swap rows if needed
        if max_row != col {
            for j in col..n {
                let temp = lu[[col, j]];
                lu[[col, j]] = lu[[max_row, j]];
                lu[[max_row, j]] = temp;
            }
            sign = -sign; // Row swap changes determinant sign
        }

        // Check for zero pivot (singular matrix)
        if lu[[col, col]].abs() < F::from(1e-12).expect("Failed to convert constant to float") {
            return F::neg_infinity(); // log(0) = -infinity
        }

        // Eliminate below pivot
        for row in (col + 1)..n {
            let factor = lu[[row, col]] / lu[[col, col]];

            for j in (col + 1)..n {
                lu[[row, j]] = lu[[row, j]] - factor * lu[[col, j]];
            }
        }
    }

    // Calculate log determinant from diagonal elements
    let mut log_det = F::zero();
    for i in 0..n {
        let diag_element = lu[[i, i]];
        if diag_element.abs() < F::from(1e-12).expect("Failed to convert constant to float") {
            return F::neg_infinity(); // Singular _matrix
        }
        log_det = log_det + diag_element.abs().ln();
    }

    // Account for sign
    if sign < F::zero() {
        // For negative determinant, we return ln(|det|)
        // Note: This assumes we want the log of the absolute determinant
        log_det
    } else {
        log_det
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_var_creation() {
        let model = VARModel::<f64>::new(2, 3).expect("Operation failed");
        assert_eq!(model.order, 2);
        assert_eq!(model.n_vars, 3);
        assert_eq!(model.coefficients.len(), 2);
        assert!(!model.is_fitted);
    }

    #[test]
    fn test_var_fit() {
        // Create simple AR(1) data
        let mut data = Array2::zeros((100, 2));
        data[[0, 0]] = 1.0;
        data[[0, 1]] = 0.5;

        for t in 1..100 {
            data[[t, 0]] = 0.5 * data[[t - 1, 0]] + 0.1 * data[[t - 1, 1]];
            data[[t, 1]] = 0.2 * data[[t - 1, 0]] + 0.7 * data[[t - 1, 1]];
        }

        let mut model = VARModel::new(1, 2).expect("Operation failed");
        model.fit(&data).expect("Operation failed");
        assert!(model.is_fitted);
    }

    #[test]
    fn test_var_predict() {
        let mut model = VARModel::new(1, 2).expect("Operation failed");
        model.coefficients[0] = array![[0.5, 0.1], [0.2, 0.7]];
        model.intercept = array![0.0, 0.0];
        model.is_fitted = true;

        let initial = array![[1.0, 0.5]];
        let predictions = model.predict(&initial, 5).expect("Operation failed");
        assert_eq!(predictions.dim(), (5, 2));
    }

    #[test]
    fn test_impulse_response() {
        let mut model = VARModel::new(1, 2).expect("Operation failed");
        model.coefficients[0] = array![[0.5, 0.1], [0.2, 0.7]];
        model.is_fitted = true;

        let irf = model.impulse_response(10, 0).expect("Operation failed");
        assert_eq!(irf.dim(), (10, 2));
        assert_eq!(irf[[0, 0]], 1.0);
        assert_eq!(irf[[0, 1]], 0.0);
    }

    #[test]
    fn test_vecm_creation() {
        let model = VECMModel::<f64>::new(3, 2, 3).expect("Operation failed");
        assert_eq!(model.rank, 2);
        assert_eq!(model.short_run.len(), 2);
        assert!(!model.is_fitted);
    }

    #[test]
    fn test_var_order_selection() {
        // Create realistic VAR data with noise to avoid singular matrices
        let mut data = Array2::zeros((100, 2));
        data[[0, 0]] = 1.0;
        data[[0, 1]] = 0.5;

        // Generate AR(1) process with sufficient variation
        use scirs2_core::random::SeedableRng;
        let mut rng = scirs2_core::random::rngs::StdRng::seed_from_u64(42);

        for t in 1..100 {
            let noise1: f64 = scirs2_core::random::RngExt::random_range(&mut rng, -0.1..0.1);
            let noise2: f64 = scirs2_core::random::RngExt::random_range(&mut rng, -0.1..0.1);

            data[[t, 0]] = 0.3 * data[[t - 1, 0]] + 0.1 * data[[t - 1, 1]] + 0.1 + noise1;
            data[[t, 1]] = 0.2 * data[[t - 1, 0]] + 0.4 * data[[t - 1, 1]] + 0.05 + noise2;
        }

        let order = select_var_order(&data, 3, SelectionCriterion::AIC).expect("Operation failed");
        assert!((1..=3).contains(&order));
    }

    #[test]
    fn test_granger_causality_detects_real_cause() {
        // Variable 0 is an independent AR(1) process; variable 1 genuinely
        // depends on variable 0's *lagged* value. So variable 0 should
        // Granger-cause variable 1, but not vice versa.
        //
        // The former hardcoded stub (f_stat=2.5, p_value=0.05 for every call)
        // would fail both assertions below (0.05 is neither < 0.01 nor > 0.05).
        use scirs2_core::random::SeedableRng;
        let mut rng = scirs2_core::random::rngs::StdRng::seed_from_u64(123);

        let n = 300;
        let mut data = Array2::<f64>::zeros((n, 2));
        for t in 1..n {
            let noise_x: f64 = scirs2_core::random::RngExt::random_range(&mut rng, -0.3..0.3);
            let noise_y: f64 = scirs2_core::random::RngExt::random_range(&mut rng, -0.3..0.3);

            data[[t, 0]] = 0.4 * data[[t - 1, 0]] + noise_x;
            data[[t, 1]] = 0.3 * data[[t - 1, 1]] + 0.8 * data[[t - 1, 0]] + noise_y;
        }

        let mut model = VARModel::new(1, 2).expect("VAR model creation should succeed");
        model.fit(&data).expect("VAR fit should succeed");

        let (f_forward, p_forward) = model
            .granger_causality(0, 1)
            .expect("Granger test 0->1 should succeed");
        assert!(
            p_forward < 0.01,
            "variable 0 genuinely drives variable 1: expected a small p-value, got f={f_forward}, p={p_forward}"
        );

        let (_, p_reverse) = model
            .granger_causality(1, 0)
            .expect("Granger test 1->0 should succeed");
        assert!(
            p_reverse > 0.05,
            "variable 1 does not drive variable 0: expected a large p-value, got p={p_reverse}"
        );
    }

    #[test]
    fn test_granger_causality_rejects_independent_series() {
        // Two independent AR(1) processes with independent noise: neither
        // should show significant Granger causality on the other.
        use scirs2_core::random::SeedableRng;
        let mut rng = scirs2_core::random::rngs::StdRng::seed_from_u64(456);

        let n = 300;
        let mut data = Array2::<f64>::zeros((n, 2));
        for t in 1..n {
            let noise_x: f64 = scirs2_core::random::RngExt::random_range(&mut rng, -0.3..0.3);
            let noise_y: f64 = scirs2_core::random::RngExt::random_range(&mut rng, -0.3..0.3);

            data[[t, 0]] = 0.5 * data[[t - 1, 0]] + noise_x;
            data[[t, 1]] = 0.5 * data[[t - 1, 1]] + noise_y;
        }

        let mut model = VARModel::new(1, 2).expect("VAR model creation should succeed");
        model.fit(&data).expect("VAR fit should succeed");

        let (_, p_value) = model
            .granger_causality(0, 1)
            .expect("Granger test should succeed");
        assert!(
            p_value > 0.05,
            "independent series should not show significant Granger causality, got p={p_value}"
        );
    }

    /// Build a bivariate series with a genuine, known cointegrating
    /// relationship: a common stochastic trend `w` (a random walk) drives
    /// both variables, so `y1 - 0.5*y2` is stationary (true cointegrating
    /// vector `[1, -0.5]`) while `y1` and `y2` individually are unit-root
    /// (non-stationary).
    fn cointegrated_series(seed: u64, n: usize) -> Array2<f64> {
        use scirs2_core::random::SeedableRng;
        let mut rng = scirs2_core::random::rngs::StdRng::seed_from_u64(seed);

        let mut w = 0.0_f64;
        let mut data = Array2::<f64>::zeros((n, 2));
        for t in 0..n {
            let step: f64 = scirs2_core::random::RngExt::random_range(&mut rng, -1.0..1.0);
            w += step;
            let noise1: f64 = scirs2_core::random::RngExt::random_range(&mut rng, -0.5..0.5);
            let noise2: f64 = scirs2_core::random::RngExt::random_range(&mut rng, -0.5..0.5);
            data[[t, 0]] = w + noise1;
            data[[t, 1]] = 2.0 * w + noise2;
        }
        data
    }

    #[test]
    fn test_vecm_fit_recovers_known_cointegration() {
        // Regression guard: the former stub left `cointegration` at its
        // `new()`-time value of all zeros, which fails the `scale.abs() >
        // 1e-8` sanity check below immediately.
        let n = 300;
        let data = cointegrated_series(7, n);

        let mut model = VECMModel::new(2, 1, 2).expect("VECM creation should succeed");
        model.fit(&data).expect("Johansen fit should succeed");
        assert!(model.is_fitted);

        // Normalize the estimated cointegrating vector so its first entry is
        // 1, then compare against the analytically-known [1, -0.5].
        let beta = &model.cointegration;
        let scale = beta[[0, 0]];
        assert!(
            scale.abs() > 1e-8,
            "degenerate cointegrating vector: {beta:?}"
        );
        let normalized = [beta[[0, 0]] / scale, beta[[1, 0]] / scale];

        assert!(
            (normalized[0] - 1.0).abs() < 1e-6,
            "normalization should force the first entry to 1, got {normalized:?}"
        );
        assert!(
            (normalized[1] - (-0.5)).abs() < 0.05,
            "expected cointegrating vector close to [1, -0.5], got {normalized:?}"
        );

        // The recovered cointegrating combination should be near-stationary
        // (much lower variance) unlike the raw, random-walk-driven series.
        let mut combo_values = Vec::with_capacity(n);
        for t in 0..n {
            combo_values.push(data[[t, 0]] * normalized[0] + data[[t, 1]] * normalized[1]);
        }
        let combo_mean = combo_values.iter().sum::<f64>() / n as f64;
        let combo_var = combo_values
            .iter()
            .map(|v| (v - combo_mean).powi(2))
            .sum::<f64>()
            / n as f64;

        let y1_mean = (0..n).map(|t| data[[t, 0]]).sum::<f64>() / n as f64;
        let y1_var = (0..n)
            .map(|t| (data[[t, 0]] - y1_mean).powi(2))
            .sum::<f64>()
            / n as f64;

        assert!(
            combo_var < y1_var * 0.1,
            "cointegrating combination should be far less volatile than the raw \
             (unit-root) series: combo_var={combo_var}, y1_var={y1_var}"
        );
    }

    #[test]
    fn test_vecm_to_var_matches_algebraic_identity() {
        // Regression guard: the former stub's `to_var()` built a fresh,
        // all-zero `VARModel` regardless of the (also-stubbed, all-zero)
        // VECM parameters, so `actual_sum` below was the zero matrix while
        // `expected_sum` was the identity -- failing immediately.
        let n = 300;
        let data = cointegrated_series(99, n);

        let mut vecm = VECMModel::new(2, 1, 2).expect("VECM creation should succeed");
        vecm.fit(&data).expect("Johansen fit should succeed");

        let var = vecm
            .to_var()
            .expect("VECM -> VAR conversion should succeed");
        assert_eq!(var.order, 2);
        assert!(var.is_fitted);

        // Algebraic identity: summing all level-VAR coefficient matrices must
        // reproduce I + Pi (this telescopes out of the correct Phi_j
        // definitions used to convert a VECM to its level-VAR form).
        let pi = vecm.adjustment.dot(&vecm.cointegration.t());
        let identity = Array2::<f64>::eye(2);
        let expected_sum = &identity + &pi;

        let mut actual_sum = Array2::<f64>::zeros((2, 2));
        for coef in &var.coefficients {
            actual_sum = actual_sum + coef;
        }

        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (actual_sum[[i, j]] - expected_sum[[i, j]]).abs() < 1e-8,
                    "sum of VAR coefficients should equal I + Pi at [{i},{j}]: {} vs {}",
                    actual_sum[[i, j]],
                    expected_sum[[i, j]]
                );
            }
        }
    }
}
