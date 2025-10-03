//! Portfolio optimization algorithms
//!
//! This module implements various portfolio optimization techniques based on
//! Modern Portfolio Theory (MPT) and alternative approaches. These algorithms
//! help construct optimal portfolios under different risk and return objectives.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Float;

use crate::error::{Result, TimeSeriesError};

/// Modern Portfolio Theory: Calculate efficient frontier point
///
/// Constructs a portfolio on the efficient frontier for a given target return.
/// This simplified implementation uses iterative adjustment toward the target
/// return. In practice, quadratic programming should be used for precision.
///
/// # Arguments
///
/// * `expected_returns` - Expected returns for each asset
/// * `covariance_matrix` - Asset return covariance matrix
/// * `target_return` - Desired portfolio expected return
///
/// # Returns
///
/// * `Result<Array1<F>>` - Optimal portfolio weights
///
/// # Examples
///
/// ```rust
/// use scirs2_series::financial::portfolio::optimization::calculate_efficient_portfolio;
/// use scirs2_core::ndarray::{array, Array2};
///
/// let expected_returns = array![0.10, 0.12, 0.08];
/// let cov_matrix = Array2::from_shape_vec((3, 3),
///     vec![0.01, 0.002, 0.001, 0.002, 0.015, 0.003, 0.001, 0.003, 0.008]).unwrap();
/// let target_return = 0.10;
///
/// let weights = calculate_efficient_portfolio(&expected_returns, &cov_matrix, target_return).unwrap();
/// ```
pub fn calculate_efficient_portfolio<F: Float + Clone>(
    expected_returns: &Array1<F>,
    covariance_matrix: &Array2<F>,
    target_return: F,
) -> Result<Array1<F>> {
    let n = expected_returns.len();

    if covariance_matrix.nrows() != n || covariance_matrix.ncols() != n {
        return Err(TimeSeriesError::DimensionMismatch {
            expected: n,
            actual: covariance_matrix.nrows(),
        });
    }

    if n == 0 {
        return Err(TimeSeriesError::InvalidInput(
            "Cannot optimize empty portfolio".to_string(),
        ));
    }

    // Equal weight as starting point
    let mut weights = Array1::from_elem(n, F::one() / F::from(n).unwrap());

    // Simple iterative adjustment toward target return
    let max_iterations = 1000;
    let convergence_tolerance = F::from(1e-6).unwrap();
    let learning_rate = F::from(0.01).unwrap();

    for iteration in 0..max_iterations {
        let current_return = weights
            .iter()
            .zip(expected_returns.iter())
            .map(|(&w, &r)| w * r)
            .fold(F::zero(), |acc, x| acc + x);

        let return_diff = target_return - current_return;

        if return_diff.abs() < convergence_tolerance {
            break;
        }

        // Adjust weights toward higher/lower return assets
        for i in 0..n {
            let adjustment = if expected_returns[i] > current_return {
                return_diff * learning_rate
            } else {
                -return_diff * learning_rate
            };

            weights[i] = weights[i] + adjustment;
            weights[i] = weights[i].max(F::zero()); // Ensure non-negative
        }

        // Normalize weights to sum to 1
        let weight_sum = weights.sum();
        if weight_sum > F::zero() {
            weights.mapv_inplace(|w| w / weight_sum);
        } else {
            // Reset to equal weights if all became zero
            weights = Array1::from_elem(n, F::one() / F::from(n).unwrap());
        }

        // Add damping for later iterations to prevent oscillation
        if iteration > max_iterations / 2 {
            let damping_factor = F::from(0.95).unwrap();
            weights.mapv_inplace(|w| {
                w * damping_factor + (F::one() / F::from(n).unwrap()) * (F::one() - damping_factor)
            });
        }
    }

    Ok(weights)
}

/// Risk parity portfolio optimization
///
/// Constructs a portfolio where each asset contributes equally to portfolio risk.
/// This implementation uses inverse volatility weighting as an approximation,
/// which is computationally efficient and often performs well in practice.
///
/// # Arguments
///
/// * `covariance_matrix` - Asset return covariance matrix
///
/// # Returns
///
/// * `Result<Array1<F>>` - Risk parity portfolio weights
///
/// # Examples
///
/// ```rust
/// use scirs2_series::financial::portfolio::optimization::risk_parity_portfolio;
/// use scirs2_core::ndarray::Array2;
///
/// let cov_matrix = Array2::from_shape_vec((2, 2), vec![0.01, 0.002, 0.002, 0.015]).unwrap();
/// let weights = risk_parity_portfolio(&cov_matrix).unwrap();
/// ```
pub fn risk_parity_portfolio<F: Float + Clone>(covariance_matrix: &Array2<F>) -> Result<Array1<F>> {
    let n = covariance_matrix.nrows();

    if covariance_matrix.ncols() != n {
        return Err(TimeSeriesError::DimensionMismatch {
            expected: n,
            actual: covariance_matrix.ncols(),
        });
    }

    if n == 0 {
        return Err(TimeSeriesError::InvalidInput(
            "Cannot optimize empty portfolio".to_string(),
        ));
    }

    // Calculate inverse volatility weights
    let mut weights = Array1::zeros(n);
    let mut total_inv_vol = F::zero();

    for i in 0..n {
        let variance = covariance_matrix[[i, i]];
        if variance > F::zero() {
            let inv_volatility = F::one() / variance.sqrt();
            weights[i] = inv_volatility;
            total_inv_vol = total_inv_vol + inv_volatility;
        } else {
            // Handle zero variance by assigning small weight
            weights[i] = F::from(1e-6).unwrap();
            total_inv_vol = total_inv_vol + weights[i];
        }
    }

    // Normalize weights
    if total_inv_vol > F::zero() {
        weights.mapv_inplace(|w| w / total_inv_vol);
    } else {
        // Fallback to equal weights if all assets have zero variance
        weights = Array1::from_elem(n, F::one() / F::from(n).unwrap());
    }

    Ok(weights)
}

/// Minimum variance portfolio optimization
///
/// Constructs the portfolio with the lowest possible variance (risk).
/// This implementation uses inverse variance weighting, which provides
/// a good approximation to the true minimum variance portfolio.
///
/// # Arguments
///
/// * `covariance_matrix` - Asset return covariance matrix
///
/// # Returns
///
/// * `Result<Array1<F>>` - Minimum variance portfolio weights
///
/// # Examples
///
/// ```rust
/// use scirs2_series::financial::portfolio::optimization::minimum_variance_portfolio;
/// use scirs2_core::ndarray::Array2;
///
/// let cov_matrix = Array2::from_shape_vec((2, 2), vec![0.01, 0.002, 0.002, 0.015]).unwrap();
/// let weights = minimum_variance_portfolio(&cov_matrix).unwrap();
/// ```
pub fn minimum_variance_portfolio<F: Float + Clone>(
    covariance_matrix: &Array2<F>,
) -> Result<Array1<F>> {
    let n = covariance_matrix.nrows();

    if covariance_matrix.ncols() != n {
        return Err(TimeSeriesError::DimensionMismatch {
            expected: n,
            actual: covariance_matrix.ncols(),
        });
    }

    if n == 0 {
        return Err(TimeSeriesError::InvalidInput(
            "Cannot optimize empty portfolio".to_string(),
        ));
    }

    // Calculate inverse variance weights
    let mut weights = Array1::zeros(n);
    let mut total_inv_var = F::zero();

    for i in 0..n {
        let variance = covariance_matrix[[i, i]];
        if variance > F::zero() {
            let inv_variance = F::one() / variance;
            weights[i] = inv_variance;
            total_inv_var = total_inv_var + inv_variance;
        } else {
            return Err(TimeSeriesError::InvalidInput(format!(
                "Asset {} has zero variance",
                i
            )));
        }
    }

    // Normalize weights
    if total_inv_var > F::zero() {
        weights.mapv_inplace(|w| w / total_inv_var);
    } else {
        return Err(TimeSeriesError::InvalidInput(
            "All assets have zero variance".to_string(),
        ));
    }

    Ok(weights)
}

/// Maximum diversification portfolio
///
/// Constructs a portfolio that maximizes the diversification ratio, defined as
/// the weighted average of individual volatilities divided by portfolio volatility.
/// This approach tends to favor assets with low correlations.
///
/// # Arguments
///
/// * `expected_returns` - Expected returns (used for tiebreaking)
/// * `covariance_matrix` - Asset return covariance matrix
///
/// # Returns
///
/// * `Result<Array1<F>>` - Maximum diversification portfolio weights
pub fn maximum_diversification_portfolio<F: Float + Clone>(
    expected_returns: &Array1<F>,
    covariance_matrix: &Array2<F>,
) -> Result<Array1<F>> {
    let n = expected_returns.len();

    if covariance_matrix.nrows() != n || covariance_matrix.ncols() != n {
        return Err(TimeSeriesError::DimensionMismatch {
            expected: n,
            actual: covariance_matrix.nrows(),
        });
    }

    // Start with inverse volatility weights (good approximation)
    let mut weights = Array1::zeros(n);

    for i in 0..n {
        let variance = covariance_matrix[[i, i]];
        if variance > F::zero() {
            weights[i] = F::one() / variance.sqrt();
        } else {
            weights[i] = F::from(1e-6).unwrap();
        }
    }

    // Normalize weights
    let weight_sum = weights.sum();
    if weight_sum > F::zero() {
        weights.mapv_inplace(|w| w / weight_sum);
    }

    // Iterative improvement (simplified)
    for _ in 0..50 {
        let mut new_weights = weights.clone();

        // Calculate portfolio volatility
        let portfolio_var = calculate_portfolio_variance(&weights, covariance_matrix)?;
        let portfolio_vol = portfolio_var.sqrt();

        if portfolio_vol <= F::zero() {
            break;
        }

        // Adjust weights to improve diversification
        for i in 0..n {
            let individual_vol = covariance_matrix[[i, i]].sqrt();
            let contribution = weights[i] * individual_vol / portfolio_vol;

            // Increase weight if asset has high individual vol but low portfolio contribution
            if individual_vol > contribution {
                let adjustment = F::from(0.01).unwrap() * (individual_vol - contribution);
                new_weights[i] = new_weights[i] + adjustment;
            }
        }

        // Normalize and update
        let new_sum = new_weights.sum();
        if new_sum > F::zero() {
            new_weights.mapv_inplace(|w| w / new_sum);
            weights = new_weights;
        }
    }

    Ok(weights)
}

/// Maximum Sharpe ratio portfolio (tangency portfolio)
///
/// Constructs the portfolio with the highest Sharpe ratio, which represents
/// the optimal risk-return trade-off on the efficient frontier.
///
/// # Arguments
///
/// * `expected_returns` - Expected asset returns
/// * `covariance_matrix` - Asset return covariance matrix
/// * `risk_free_rate` - Risk-free rate for Sharpe ratio calculation
///
/// # Returns
///
/// * `Result<Array1<F>>` - Maximum Sharpe ratio portfolio weights
pub fn maximum_sharpe_portfolio<F: Float + Clone>(
    expected_returns: &Array1<F>,
    covariance_matrix: &Array2<F>,
    risk_free_rate: F,
) -> Result<Array1<F>> {
    let n = expected_returns.len();

    if covariance_matrix.nrows() != n || covariance_matrix.ncols() != n {
        return Err(TimeSeriesError::DimensionMismatch {
            expected: n,
            actual: covariance_matrix.nrows(),
        });
    }

    // Calculate excess returns
    let excess_returns: Array1<F> = expected_returns.mapv(|r| r - risk_free_rate);

    // Start with equal weights
    let mut weights = Array1::from_elem(n, F::one() / F::from(n).unwrap());
    let mut best_sharpe = F::neg_infinity();

    // Grid search with refinement (simplified optimization)
    for _iteration in 0..100 {
        let mut best_weights = weights.clone();

        // Try small perturbations
        for i in 0..n {
            for delta in [-0.05, -0.01, 0.01, 0.05].iter() {
                let mut test_weights = weights.clone();

                // Adjust weight i
                test_weights[i] = test_weights[i] + F::from(*delta).unwrap();
                test_weights[i] = test_weights[i].max(F::zero());

                // Normalize
                let sum = test_weights.sum();
                if sum > F::zero() {
                    test_weights.mapv_inplace(|w| w / sum);

                    // Calculate Sharpe ratio
                    if let Ok(sharpe) = calculate_portfolio_sharpe_ratio(
                        &test_weights,
                        &excess_returns,
                        covariance_matrix,
                    ) {
                        if sharpe > best_sharpe {
                            best_sharpe = sharpe;
                            best_weights = test_weights;
                        }
                    }
                }
            }
        }

        weights = best_weights;
    }

    Ok(weights)
}

/// Calculate portfolio variance given weights and covariance matrix
///
/// # Arguments
///
/// * `weights` - Portfolio weights
/// * `covariance_matrix` - Asset return covariance matrix
///
/// # Returns
///
/// * `Result<F>` - Portfolio variance
pub fn calculate_portfolio_variance<F: Float + Clone>(
    weights: &Array1<F>,
    covariance_matrix: &Array2<F>,
) -> Result<F> {
    let n = weights.len();

    if covariance_matrix.nrows() != n || covariance_matrix.ncols() != n {
        return Err(TimeSeriesError::DimensionMismatch {
            expected: n,
            actual: covariance_matrix.nrows(),
        });
    }

    let mut variance = F::zero();

    for i in 0..n {
        for j in 0..n {
            variance = variance + weights[i] * weights[j] * covariance_matrix[[i, j]];
        }
    }

    Ok(variance)
}

/// Calculate portfolio Sharpe ratio
///
/// # Arguments
///
/// * `weights` - Portfolio weights
/// * `excess_returns` - Asset excess returns (return - risk_free_rate)
/// * `covariance_matrix` - Asset return covariance matrix
///
/// # Returns
///
/// * `Result<F>` - Portfolio Sharpe ratio
fn calculate_portfolio_sharpe_ratio<F: Float + Clone>(
    weights: &Array1<F>,
    excess_returns: &Array1<F>,
    covariance_matrix: &Array2<F>,
) -> Result<F> {
    // Calculate portfolio expected excess return
    let portfolio_excess_return = weights
        .iter()
        .zip(excess_returns.iter())
        .map(|(&w, &r)| w * r)
        .fold(F::zero(), |acc, x| acc + x);

    // Calculate portfolio variance
    let portfolio_variance = calculate_portfolio_variance(weights, covariance_matrix)?;

    if portfolio_variance <= F::zero() {
        return Ok(F::zero());
    }

    let portfolio_volatility = portfolio_variance.sqrt();

    Ok(portfolio_excess_return / portfolio_volatility)
}

/// Calculate correlation matrix from returns
///
/// Computes the correlation matrix from a matrix of asset returns.
/// This is often needed as input for portfolio optimization.
///
/// # Arguments
///
/// * `returns` - Asset return matrix (rows: time, cols: assets)
///
/// # Returns
///
/// * `Result<Array2<F>>` - Correlation matrix
///
/// # Examples
///
/// ```rust
/// use scirs2_series::financial::portfolio::optimization::calculate_correlation_matrix;
/// use scirs2_core::ndarray::Array2;
///
/// let returns = Array2::from_shape_vec((5, 2),
///     vec![0.01, 0.02, -0.01, 0.005, 0.015, -0.008, 0.005, 0.012, -0.002, 0.008]).unwrap();
/// let corr_matrix = calculate_correlation_matrix(&returns).unwrap();
/// ```
pub fn calculate_correlation_matrix<F: Float + Clone>(
    returns: &Array2<F>, // rows: time, cols: assets
) -> Result<Array2<F>> {
    let n_assets = returns.ncols();
    let n_periods = returns.nrows();

    if n_periods < 2 {
        return Err(TimeSeriesError::InsufficientData {
            message: "Need at least 2 periods for correlation calculation".to_string(),
            required: 2,
            actual: n_periods,
        });
    }

    let mut correlation_matrix = Array2::zeros((n_assets, n_assets));

    // Calculate means
    let means: Array1<F> = (0..n_assets)
        .map(|i| {
            let col = returns.column(i);
            col.sum() / F::from(n_periods).unwrap()
        })
        .collect();

    // Calculate correlation coefficients
    for i in 0..n_assets {
        for j in 0..n_assets {
            if i == j {
                correlation_matrix[[i, j]] = F::one();
            } else {
                let col_i = returns.column(i);
                let col_j = returns.column(j);

                let mut numerator = F::zero();
                let mut sum_sq_i = F::zero();
                let mut sum_sq_j = F::zero();

                for t in 0..n_periods {
                    let dev_i = col_i[t] - means[i];
                    let dev_j = col_j[t] - means[j];

                    numerator = numerator + dev_i * dev_j;
                    sum_sq_i = sum_sq_i + dev_i * dev_i;
                    sum_sq_j = sum_sq_j + dev_j * dev_j;
                }

                let denominator = (sum_sq_i * sum_sq_j).sqrt();
                if denominator > F::zero() {
                    correlation_matrix[[i, j]] = numerator / denominator;
                } else {
                    correlation_matrix[[i, j]] = F::zero();
                }
            }
        }
    }

    Ok(correlation_matrix)
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{arr1, Array2};

    #[test]
    fn test_efficient_portfolio() {
        let expected_returns = arr1(&[0.10, 0.12, 0.08]);
        let cov_matrix = Array2::from_shape_vec(
            (3, 3),
            vec![0.01, 0.002, 0.001, 0.002, 0.015, 0.003, 0.001, 0.003, 0.008],
        )
        .unwrap();
        let target_return = 0.10;

        let result = calculate_efficient_portfolio(&expected_returns, &cov_matrix, target_return);
        assert!(result.is_ok());

        let weights = result.unwrap();
        assert_eq!(weights.len(), 3);

        // Weights should sum to approximately 1.0
        let weight_sum: f64 = weights.sum();
        assert!((weight_sum - 1.0).abs() < 1e-6);

        // All weights should be non-negative
        for &weight in weights.iter() {
            assert!(weight >= 0.0);
        }
    }

    #[test]
    fn test_risk_parity_portfolio() {
        let cov_matrix = Array2::from_shape_vec((2, 2), vec![0.01, 0.002, 0.002, 0.015]).unwrap();

        let result = risk_parity_portfolio(&cov_matrix);
        assert!(result.is_ok());

        let weights = result.unwrap();
        assert_eq!(weights.len(), 2);

        // Weights should sum to 1.0
        let weight_sum: f64 = weights.sum();
        assert!((weight_sum - 1.0).abs() < 1e-10);

        // Asset with lower variance should have higher weight
        assert!(weights[0] > weights[1]); // 0.01 < 0.015
    }

    #[test]
    fn test_minimum_variance_portfolio() {
        let cov_matrix = Array2::from_shape_vec((2, 2), vec![0.01, 0.002, 0.002, 0.015]).unwrap();

        let result = minimum_variance_portfolio(&cov_matrix);
        assert!(result.is_ok());

        let weights = result.unwrap();
        assert_eq!(weights.len(), 2);

        // Weights should sum to 1.0
        let weight_sum: f64 = weights.sum();
        assert!((weight_sum - 1.0).abs() < 1e-10);

        // Asset with lower variance should have higher weight
        assert!(weights[0] > weights[1]);
    }

    #[test]
    fn test_portfolio_variance() {
        let weights = arr1(&[0.6, 0.4]);
        let cov_matrix = Array2::from_shape_vec((2, 2), vec![0.01, 0.002, 0.002, 0.015]).unwrap();

        let result = calculate_portfolio_variance(&weights, &cov_matrix);
        assert!(result.is_ok());

        let variance = result.unwrap();
        assert!(variance > 0.0);

        // Manual calculation: 0.6² * 0.01 + 0.4² * 0.015 + 2 * 0.6 * 0.4 * 0.002
        let expected = 0.36 * 0.01 + 0.16 * 0.015 + 2.0 * 0.6 * 0.4 * 0.002;
        assert!((variance - expected).abs() < 1e-10);
    }

    #[test]
    fn test_correlation_matrix() {
        let returns = Array2::from_shape_vec(
            (5, 2),
            vec![
                0.01, 0.02, -0.01, 0.005, 0.015, -0.008, 0.005, 0.012, -0.002, 0.008,
            ],
        )
        .unwrap();

        let result = calculate_correlation_matrix(&returns);
        assert!(result.is_ok());

        let corr_matrix = result.unwrap();
        assert_eq!(corr_matrix.shape(), &[2, 2]);

        // Diagonal should be 1.0
        assert!((corr_matrix[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((corr_matrix[[1, 1]] - 1.0).abs() < 1e-10);

        // Matrix should be symmetric
        assert!((corr_matrix[[0, 1]] - corr_matrix[[1, 0]]).abs() < 1e-10);

        // Correlations should be between -1 and 1
        assert!(corr_matrix[[0, 1]].abs() <= 1.0);
    }

    #[test]
    fn test_dimension_mismatch() {
        let expected_returns = arr1(&[0.10, 0.12]);
        let cov_matrix = Array2::from_shape_vec((3, 3), vec![0.0; 9]).unwrap();
        let target_return = 0.10;

        let result = calculate_efficient_portfolio(&expected_returns, &cov_matrix, target_return);
        assert!(result.is_err());
    }

    #[test]
    fn test_zero_variance_error() {
        let cov_matrix = Array2::from_shape_vec(
            (2, 2),
            vec![0.0, 0.0, 0.0, 0.015], // First asset has zero variance
        )
        .unwrap();

        let result = minimum_variance_portfolio(&cov_matrix);
        assert!(result.is_err());
    }

    #[test]
    fn test_maximum_sharpe_portfolio() {
        let expected_returns = arr1(&[0.10, 0.12]);
        let cov_matrix = Array2::from_shape_vec((2, 2), vec![0.01, 0.002, 0.002, 0.015]).unwrap();
        let risk_free_rate = 0.02;

        let result = maximum_sharpe_portfolio(&expected_returns, &cov_matrix, risk_free_rate);
        assert!(result.is_ok());

        let weights = result.unwrap();
        assert_eq!(weights.len(), 2);

        // Weights should sum to approximately 1.0
        let weight_sum: f64 = weights.sum();
        assert!((weight_sum - 1.0).abs() < 1e-6);
    }
}
