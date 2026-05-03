//! Time Series Cross-Validation
//!
//! Specialized cross-validation methods that respect temporal ordering.
//! Standard k-fold CV is not appropriate for time series because it breaks
//! the temporal structure. This module provides:
//!
//! - **Expanding window**: Training set grows, test window moves forward
//! - **Sliding window**: Fixed-size training window slides forward
//! - **Blocked split**: Non-overlapping contiguous blocks
//! - **Purged CV**: Gap (embargo) between train and test to prevent leakage
//! - **Walk-forward**: One-step-ahead expanding validation
//! - **Summary statistics**: Aggregate CV results across folds
//!
//! # References
//!
//! - Bergmeir, C. & Benitez, J.M. (2012) "On the use of cross-validation
//!   for time series predictor evaluation"
//! - de Prado, M.L. (2018) "Advances in Financial Machine Learning", Ch. 7

use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::{Float, FromPrimitive};
use std::fmt::{Debug, Display};

use crate::error::{Result, TimeSeriesError};

// ---------------------------------------------------------------------------
// Fold representation
// ---------------------------------------------------------------------------

/// A single train-test split (fold) defined by index ranges
#[derive(Debug, Clone)]
pub struct TimeSeriesFold {
    /// Start index of training set (inclusive)
    pub train_start: usize,
    /// End index of training set (exclusive)
    pub train_end: usize,
    /// Start index of test set (inclusive)
    pub test_start: usize,
    /// End index of test set (exclusive)
    pub test_end: usize,
}

impl TimeSeriesFold {
    /// Length of the training set
    pub fn train_size(&self) -> usize {
        self.train_end - self.train_start
    }

    /// Length of the test set
    pub fn test_size(&self) -> usize {
        self.test_end - self.test_start
    }

    /// Extract training data from an array
    pub fn train_data<F: Float>(&self, data: &Array1<F>) -> Array1<F> {
        data.slice(scirs2_core::ndarray::s![self.train_start..self.train_end])
            .to_owned()
    }

    /// Extract test data from an array
    pub fn test_data<F: Float>(&self, data: &Array1<F>) -> Array1<F> {
        data.slice(scirs2_core::ndarray::s![self.test_start..self.test_end])
            .to_owned()
    }
}

impl std::fmt::Display for TimeSeriesFold {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Fold(train=[{}..{}], test=[{}..{}])",
            self.train_start, self.train_end, self.test_start, self.test_end
        )
    }
}

// ---------------------------------------------------------------------------
// Expanding window cross-validation
// ---------------------------------------------------------------------------

/// Configuration for expanding window cross-validation
#[derive(Debug, Clone)]
pub struct ExpandingWindowConfig {
    /// Minimum training set size (initial window)
    pub initial_train_size: usize,
    /// Test set size at each fold
    pub test_size: usize,
    /// Step size between successive folds (default = test_size)
    pub step_size: Option<usize>,
}

/// Generate expanding window cross-validation folds
///
/// The training set starts at `initial_train_size` and grows by `step_size`
/// at each fold. The test window always has `test_size` observations
/// immediately following the training set.
///
/// ```text
/// Fold 1: [TRAIN........] [TEST]
/// Fold 2: [TRAIN...............] [TEST]
/// Fold 3: [TRAIN....................] [TEST]
/// ```
///
/// # Arguments
/// * `n` - Total number of observations
/// * `config` - Configuration parameters
///
/// # Returns
/// Vector of TimeSeriesFold
pub fn expanding_window(n: usize, config: &ExpandingWindowConfig) -> Result<Vec<TimeSeriesFold>> {
    if config.initial_train_size == 0 {
        return Err(TimeSeriesError::InvalidParameter {
            name: "initial_train_size".to_string(),
            message: "Must be positive".to_string(),
        });
    }
    if config.test_size == 0 {
        return Err(TimeSeriesError::InvalidParameter {
            name: "test_size".to_string(),
            message: "Must be positive".to_string(),
        });
    }

    let min_total = config.initial_train_size + config.test_size;
    if n < min_total {
        return Err(TimeSeriesError::InsufficientData {
            message: "for expanding window CV".to_string(),
            required: min_total,
            actual: n,
        });
    }

    let step = config.step_size.unwrap_or(config.test_size);
    if step == 0 {
        return Err(TimeSeriesError::InvalidParameter {
            name: "step_size".to_string(),
            message: "Must be positive".to_string(),
        });
    }

    let mut folds = Vec::new();
    let mut train_end = config.initial_train_size;

    while train_end + config.test_size <= n {
        folds.push(TimeSeriesFold {
            train_start: 0,
            train_end,
            test_start: train_end,
            test_end: train_end + config.test_size,
        });
        train_end += step;
    }

    if folds.is_empty() {
        return Err(TimeSeriesError::InsufficientData {
            message: "No valid folds for expanding window CV".to_string(),
            required: min_total,
            actual: n,
        });
    }

    Ok(folds)
}

// ---------------------------------------------------------------------------
// Sliding window cross-validation
// ---------------------------------------------------------------------------

/// Configuration for sliding window cross-validation
#[derive(Debug, Clone)]
pub struct SlidingWindowConfig {
    /// Fixed training window size
    pub train_size: usize,
    /// Test set size at each fold
    pub test_size: usize,
    /// Step size between folds (default = test_size)
    pub step_size: Option<usize>,
}

/// Generate sliding window cross-validation folds
///
/// Both training and test windows slide forward. Training window size
/// is fixed (unlike expanding window).
///
/// ```text
/// Fold 1: [TRAIN........] [TEST]
/// Fold 2:    [TRAIN........] [TEST]
/// Fold 3:       [TRAIN........] [TEST]
/// ```
///
/// # Arguments
/// * `n` - Total number of observations
/// * `config` - Configuration parameters
///
/// # Returns
/// Vector of TimeSeriesFold
pub fn sliding_window(n: usize, config: &SlidingWindowConfig) -> Result<Vec<TimeSeriesFold>> {
    if config.train_size == 0 {
        return Err(TimeSeriesError::InvalidParameter {
            name: "train_size".to_string(),
            message: "Must be positive".to_string(),
        });
    }
    if config.test_size == 0 {
        return Err(TimeSeriesError::InvalidParameter {
            name: "test_size".to_string(),
            message: "Must be positive".to_string(),
        });
    }

    let min_total = config.train_size + config.test_size;
    if n < min_total {
        return Err(TimeSeriesError::InsufficientData {
            message: "for sliding window CV".to_string(),
            required: min_total,
            actual: n,
        });
    }

    let step = config.step_size.unwrap_or(config.test_size);
    if step == 0 {
        return Err(TimeSeriesError::InvalidParameter {
            name: "step_size".to_string(),
            message: "Must be positive".to_string(),
        });
    }

    let mut folds = Vec::new();
    let mut start = 0;

    while start + config.train_size + config.test_size <= n {
        folds.push(TimeSeriesFold {
            train_start: start,
            train_end: start + config.train_size,
            test_start: start + config.train_size,
            test_end: start + config.train_size + config.test_size,
        });
        start += step;
    }

    if folds.is_empty() {
        return Err(TimeSeriesError::InsufficientData {
            message: "No valid folds for sliding window CV".to_string(),
            required: min_total,
            actual: n,
        });
    }

    Ok(folds)
}

// ---------------------------------------------------------------------------
// Blocked time series split
// ---------------------------------------------------------------------------

/// Configuration for blocked time series split
#[derive(Debug, Clone)]
pub struct BlockedSplitConfig {
    /// Number of blocks (folds)
    pub n_blocks: usize,
}

/// Generate blocked time series cross-validation folds
///
/// Divides the series into `n_blocks` contiguous blocks of approximately
/// equal size. Each block serves as a test set, with all preceding blocks
/// forming the training set.
///
/// ```text
/// Block 1: [TEST  ] [----] [----] [----]  (no training data)
/// Block 2: [TRAIN ] [TEST] [----] [----]
/// Block 3: [TRAIN ] [TRAI] [TEST] [----]
/// Block 4: [TRAIN ] [TRAI] [TRAI] [TEST]
/// ```
///
/// Note: The first block cannot be used for evaluation (no training data).
///
/// # Arguments
/// * `n` - Total number of observations
/// * `config` - Configuration parameters
///
/// # Returns
/// Vector of TimeSeriesFold (n_blocks - 1 folds)
pub fn blocked_split(n: usize, config: &BlockedSplitConfig) -> Result<Vec<TimeSeriesFold>> {
    if config.n_blocks < 2 {
        return Err(TimeSeriesError::InvalidParameter {
            name: "n_blocks".to_string(),
            message: format!("Must be >= 2, got {}", config.n_blocks),
        });
    }

    if n < config.n_blocks {
        return Err(TimeSeriesError::InsufficientData {
            message: "for blocked CV".to_string(),
            required: config.n_blocks,
            actual: n,
        });
    }

    let block_size = n / config.n_blocks;
    if block_size == 0 {
        return Err(TimeSeriesError::InvalidParameter {
            name: "n_blocks".to_string(),
            message: "Too many blocks for the data size".to_string(),
        });
    }

    let mut folds = Vec::new();

    // Skip first block (no training data)
    for b in 1..config.n_blocks {
        let test_start = b * block_size;
        let test_end = if b == config.n_blocks - 1 {
            n // Last block takes remaining data
        } else {
            (b + 1) * block_size
        };

        folds.push(TimeSeriesFold {
            train_start: 0,
            train_end: test_start,
            test_start,
            test_end,
        });
    }

    Ok(folds)
}

// ---------------------------------------------------------------------------
// Purged cross-validation
// ---------------------------------------------------------------------------

/// Configuration for purged cross-validation
#[derive(Debug, Clone)]
pub struct PurgedCVConfig {
    /// Number of folds
    pub n_folds: usize,
    /// Embargo period: number of observations to skip between train and test
    pub embargo_size: usize,
}

/// Generate purged cross-validation folds with embargo
///
/// Similar to blocked CV but adds a gap (embargo) between the training
/// and test sets to prevent information leakage. This is especially
/// important for financial time series with overlapping labels.
///
/// ```text
/// Fold 1: [TRAIN ] [EMBARGO] [TEST] [----]
/// Fold 2: [TRAIN ] [----] [EMBARGO] [TEST]
/// ```
///
/// # Arguments
/// * `n` - Total number of observations
/// * `config` - Configuration parameters
///
/// # Returns
/// Vector of TimeSeriesFold
pub fn purged_cv(n: usize, config: &PurgedCVConfig) -> Result<Vec<TimeSeriesFold>> {
    if config.n_folds < 2 {
        return Err(TimeSeriesError::InvalidParameter {
            name: "n_folds".to_string(),
            message: format!("Must be >= 2, got {}", config.n_folds),
        });
    }

    let block_size = n / config.n_folds;
    if block_size == 0 {
        return Err(TimeSeriesError::InsufficientData {
            message: "for purged CV".to_string(),
            required: config.n_folds,
            actual: n,
        });
    }

    let min_total = config.n_folds + config.embargo_size * (config.n_folds - 1);
    if n < min_total {
        return Err(TimeSeriesError::InsufficientData {
            message: "for purged CV with embargo".to_string(),
            required: min_total,
            actual: n,
        });
    }

    let mut folds = Vec::new();

    for b in 1..config.n_folds {
        let test_start = b * block_size;
        let test_end = if b == config.n_folds - 1 {
            n
        } else {
            (b + 1) * block_size
        };

        // Training ends embargo_size before the test start
        let train_end = if test_start > config.embargo_size {
            test_start - config.embargo_size
        } else {
            0
        };

        if train_end > 0 {
            folds.push(TimeSeriesFold {
                train_start: 0,
                train_end,
                test_start,
                test_end,
            });
        }
    }

    if folds.is_empty() {
        return Err(TimeSeriesError::InsufficientData {
            message: "No valid folds for purged CV (embargo too large)".to_string(),
            required: config.n_folds + config.embargo_size,
            actual: n,
        });
    }

    Ok(folds)
}

// ---------------------------------------------------------------------------
// Walk-forward validation
// ---------------------------------------------------------------------------

/// Configuration for walk-forward validation
#[derive(Debug, Clone)]
pub struct WalkForwardConfig {
    /// Minimum training set size
    pub initial_train_size: usize,
    /// Number of steps to forecast at each fold (typically 1)
    pub forecast_horizon: usize,
    /// Whether to use an expanding (true) or fixed (false) training window
    pub expanding: bool,
}

/// Generate walk-forward validation folds
///
/// At each step, the model is trained on historical data and tested on the
/// next `forecast_horizon` observations. This is the most rigorous approach
/// for evaluating forecasting models.
///
/// ```text
/// Step 1: [TRAIN........] [F]
/// Step 2: [TRAIN.........] [F]     (expanding)
/// Step 2:  [TRAIN........] [F]     (fixed window)
/// Step 3: [TRAIN..........] [F]
/// ...
/// ```
///
/// # Arguments
/// * `n` - Total number of observations
/// * `config` - Configuration parameters
///
/// # Returns
/// Vector of TimeSeriesFold
pub fn walk_forward(n: usize, config: &WalkForwardConfig) -> Result<Vec<TimeSeriesFold>> {
    if config.initial_train_size == 0 {
        return Err(TimeSeriesError::InvalidParameter {
            name: "initial_train_size".to_string(),
            message: "Must be positive".to_string(),
        });
    }
    if config.forecast_horizon == 0 {
        return Err(TimeSeriesError::InvalidParameter {
            name: "forecast_horizon".to_string(),
            message: "Must be positive".to_string(),
        });
    }

    let min_total = config.initial_train_size + config.forecast_horizon;
    if n < min_total {
        return Err(TimeSeriesError::InsufficientData {
            message: "for walk-forward validation".to_string(),
            required: min_total,
            actual: n,
        });
    }

    let mut folds = Vec::new();
    let mut train_end = config.initial_train_size;

    while train_end + config.forecast_horizon <= n {
        let train_start = if config.expanding {
            0
        } else {
            // Fixed window: slide forward
            if train_end > config.initial_train_size {
                train_end - config.initial_train_size
            } else {
                0
            }
        };

        folds.push(TimeSeriesFold {
            train_start,
            train_end,
            test_start: train_end,
            test_end: train_end + config.forecast_horizon,
        });

        train_end += config.forecast_horizon;
    }

    if folds.is_empty() {
        return Err(TimeSeriesError::InsufficientData {
            message: "No valid folds for walk-forward validation".to_string(),
            required: min_total,
            actual: n,
        });
    }

    Ok(folds)
}

// ---------------------------------------------------------------------------
// CV runner and summary statistics
// ---------------------------------------------------------------------------

/// Result of a single fold evaluation
#[derive(Debug, Clone)]
pub struct FoldResult<F: Float> {
    /// Fold index
    pub fold_index: usize,
    /// Training set size
    pub train_size: usize,
    /// Test set size
    pub test_size: usize,
    /// Error metric (e.g., MAE, RMSE) for this fold
    pub error: F,
}

/// Summary statistics across cross-validation folds
#[derive(Debug, Clone)]
pub struct CVSummary<F: Float> {
    /// Number of folds
    pub n_folds: usize,
    /// Mean error across folds
    pub mean_error: F,
    /// Standard deviation of error across folds
    pub std_error: F,
    /// Minimum error (best fold)
    pub min_error: F,
    /// Maximum error (worst fold)
    pub max_error: F,
    /// Median error
    pub median_error: F,
    /// Individual fold results
    pub fold_results: Vec<FoldResult<F>>,
}

impl<F: Float + Display> std::fmt::Display for CVSummary<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Cross-Validation Summary ({} folds)", self.n_folds)?;
        writeln!(f, "================================")?;
        writeln!(f, "Mean error:   {:.6}", self.mean_error)?;
        writeln!(f, "Std error:    {:.6}", self.std_error)?;
        writeln!(f, "Min error:    {:.6}", self.min_error)?;
        writeln!(f, "Max error:    {:.6}", self.max_error)?;
        writeln!(f, "Median error: {:.6}", self.median_error)?;
        Ok(())
    }
}

/// Run cross-validation with a user-provided evaluation function
///
/// # Arguments
/// * `data` - Full time series
/// * `folds` - Cross-validation folds (from any of the generators above)
/// * `evaluate` - Function: (train_data, test_data) -> error_metric
///
/// # Returns
/// CVSummary with aggregate statistics
pub fn run_cv<F, Func>(
    data: &Array1<F>,
    folds: &[TimeSeriesFold],
    evaluate: Func,
) -> Result<CVSummary<F>>
where
    F: Float + FromPrimitive + Debug + Display,
    Func: Fn(&Array1<F>, &Array1<F>) -> Result<F>,
{
    if folds.is_empty() {
        return Err(TimeSeriesError::InvalidInput(
            "No folds provided for cross-validation".to_string(),
        ));
    }

    let n = data.len();
    let mut fold_results = Vec::with_capacity(folds.len());

    for (idx, fold) in folds.iter().enumerate() {
        // Validate fold indices
        if fold.train_end > n || fold.test_end > n {
            return Err(TimeSeriesError::InvalidInput(format!(
                "Fold {} indices exceed data length {}",
                idx, n
            )));
        }

        let train_data = fold.train_data(data);
        let test_data = fold.test_data(data);

        let error = evaluate(&train_data, &test_data)?;

        fold_results.push(FoldResult {
            fold_index: idx,
            train_size: fold.train_size(),
            test_size: fold.test_size(),
            error,
        });
    }

    compute_summary(fold_results)
}

/// Compute summary statistics from fold results
pub fn compute_summary<F>(fold_results: Vec<FoldResult<F>>) -> Result<CVSummary<F>>
where
    F: Float + FromPrimitive + Debug,
{
    let n_folds = fold_results.len();
    if n_folds == 0 {
        return Err(TimeSeriesError::InvalidInput(
            "No fold results to summarize".to_string(),
        ));
    }

    let n_f = F::from_usize(n_folds).ok_or_else(|| {
        TimeSeriesError::ComputationError("Failed to convert fold count".to_string())
    })?;

    // Mean
    let mean_error = fold_results
        .iter()
        .map(|r| r.error)
        .fold(F::zero(), |a, x| a + x)
        / n_f;

    // Std dev
    let variance = fold_results
        .iter()
        .map(|r| {
            let d = r.error - mean_error;
            d * d
        })
        .fold(F::zero(), |a, x| a + x)
        / n_f;
    let std_error = variance.sqrt();

    // Min/Max
    let min_error = fold_results
        .iter()
        .map(|r| r.error)
        .fold(F::infinity(), |a, x| a.min(x));
    let max_error = fold_results
        .iter()
        .map(|r| r.error)
        .fold(F::neg_infinity(), |a, x| a.max(x));

    // Median
    let mut errors: Vec<F> = fold_results.iter().map(|r| r.error).collect();
    errors.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_error = if n_folds % 2 == 1 {
        errors[n_folds / 2]
    } else {
        let two = F::from_f64(2.0).unwrap_or(F::one() + F::one());
        (errors[n_folds / 2 - 1] + errors[n_folds / 2]) / two
    };

    Ok(CVSummary {
        n_folds,
        mean_error,
        std_error,
        min_error,
        max_error,
        median_error,
        fold_results,
    })
}

// ---------------------------------------------------------------------------
// CrossValidator — high-level wrapper
// ---------------------------------------------------------------------------

/// High-level cross-validation wrapper for time series
///
/// Provides a scikit-learn-style API for performing time series cross-validation
/// with expanding windows and an optional held-out test set.
///
/// # Examples
///
/// ```rust
/// use scirs2_series::ts_cv::CrossValidator;
/// use scirs2_core::ndarray::Array1;
///
/// let data = Array1::linspace(0.0_f64, 1.0, 100);
/// let cv = CrossValidator::new(5, 0.2);
/// let folds = cv.time_series_split(&data).expect("should succeed");
/// assert_eq!(folds.len(), 5);
/// ```
#[derive(Debug, Clone)]
pub struct CrossValidator {
    /// Number of folds for cross-validation
    n_folds: usize,
    /// Holdout fraction reserved as final test set (0.0 = no holdout)
    holdout_fraction: f64,
}

impl CrossValidator {
    /// Create a new `CrossValidator`
    ///
    /// # Arguments
    ///
    /// * `n_folds` — Number of cross-validation folds (must be ≥ 2)
    /// * `holdout_fraction` — Fraction of data to hold out at the end as a final
    ///   test set; must be in `[0.0, 1.0)`. Pass `0.0` to use all data for CV.
    pub fn new(n_folds: usize, holdout_fraction: f64) -> Self {
        Self {
            n_folds,
            holdout_fraction,
        }
    }

    /// Produce expanding-window time-series folds
    ///
    /// Reserves `holdout_fraction * n` observations at the end as a held-out
    /// test set, then applies an expanding-window split to the remaining data
    /// to produce exactly `n_folds` folds.
    ///
    /// # Returns
    ///
    /// `Vec<TimeSeriesFold>` with `len() == n_folds`
    pub fn time_series_split<F: Float>(&self, data: &Array1<F>) -> Result<Vec<TimeSeriesFold>> {
        let n = data.len();
        if n == 0 {
            return Err(TimeSeriesError::InvalidInput(
                "Data array is empty".to_string(),
            ));
        }
        if self.n_folds < 2 {
            return Err(TimeSeriesError::InvalidParameter {
                name: "n_folds".to_string(),
                message: format!("Must be >= 2, got {}", self.n_folds),
            });
        }
        if !(0.0..1.0).contains(&self.holdout_fraction) {
            return Err(TimeSeriesError::InvalidParameter {
                name: "holdout_fraction".to_string(),
                message: format!("Must be in [0.0, 1.0), got {}", self.holdout_fraction),
            });
        }

        // Compute the CV region (exclude the holdout tail)
        let holdout_size = (n as f64 * self.holdout_fraction).round() as usize;
        let cv_n = n.saturating_sub(holdout_size);

        if cv_n < self.n_folds + 1 {
            return Err(TimeSeriesError::InsufficientData {
                message: format!(
                    "Not enough data for {} folds after reserving holdout",
                    self.n_folds
                ),
                required: self.n_folds + 1 + holdout_size,
                actual: n,
            });
        }

        // Use blocked split to get exactly n_folds folds within [0, cv_n)
        let block_size = cv_n / (self.n_folds + 1); // +1 because first block = initial train
        if block_size == 0 {
            return Err(TimeSeriesError::InsufficientData {
                message: "Block size is zero; too many folds for the data size".to_string(),
                required: (self.n_folds + 1) * 2,
                actual: cv_n,
            });
        }

        // Produce exactly n_folds folds with an expanding training window
        let mut folds = Vec::with_capacity(self.n_folds);
        let initial_train = block_size;
        let test_size = block_size;

        for k in 0..self.n_folds {
            let train_end = initial_train + k * test_size;
            let test_start = train_end;
            let test_end = (test_start + test_size).min(cv_n);
            if test_end > test_start && train_end > 0 {
                folds.push(TimeSeriesFold {
                    train_start: 0,
                    train_end,
                    test_start,
                    test_end,
                });
            }
        }

        // Ensure we return exactly n_folds (pad/truncate if needed)
        folds.truncate(self.n_folds);

        if folds.len() < self.n_folds {
            return Err(TimeSeriesError::InsufficientData {
                message: format!(
                    "Could only generate {} folds, {} requested",
                    folds.len(),
                    self.n_folds
                ),
                required: self.n_folds,
                actual: folds.len(),
            });
        }

        Ok(folds)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    const TOL: f64 = 1e-10;

    // --- Expanding window tests ---

    #[test]
    fn test_expanding_window_basic() {
        let config = ExpandingWindowConfig {
            initial_train_size: 5,
            test_size: 2,
            step_size: None,
        };
        let folds = expanding_window(11, &config).expect("Should succeed");
        assert_eq!(folds.len(), 3);

        // Fold 0: train=[0..5], test=[5..7]
        assert_eq!(folds[0].train_start, 0);
        assert_eq!(folds[0].train_end, 5);
        assert_eq!(folds[0].test_start, 5);
        assert_eq!(folds[0].test_end, 7);

        // Fold 1: train=[0..7], test=[7..9]
        assert_eq!(folds[1].train_start, 0);
        assert_eq!(folds[1].train_end, 7);
        assert_eq!(folds[1].test_start, 7);
        assert_eq!(folds[1].test_end, 9);

        // Fold 2: train=[0..9], test=[9..11]
        assert_eq!(folds[2].train_start, 0);
        assert_eq!(folds[2].train_end, 9);
        assert_eq!(folds[2].test_start, 9);
        assert_eq!(folds[2].test_end, 11);
    }

    #[test]
    fn test_expanding_window_custom_step() {
        let config = ExpandingWindowConfig {
            initial_train_size: 5,
            test_size: 2,
            step_size: Some(1),
        };
        let folds = expanding_window(10, &config).expect("Should succeed");
        // Step=1 => more folds
        assert!(folds.len() > 1);

        // Training always starts at 0
        for fold in &folds {
            assert_eq!(fold.train_start, 0);
        }
    }

    #[test]
    fn test_expanding_window_insufficient_data() {
        let config = ExpandingWindowConfig {
            initial_train_size: 10,
            test_size: 5,
            step_size: None,
        };
        assert!(expanding_window(12, &config).is_err());
    }

    #[test]
    fn test_expanding_window_zero_params() {
        let config = ExpandingWindowConfig {
            initial_train_size: 0,
            test_size: 2,
            step_size: None,
        };
        assert!(expanding_window(10, &config).is_err());
    }

    // --- Sliding window tests ---

    #[test]
    fn test_sliding_window_basic() {
        let config = SlidingWindowConfig {
            train_size: 5,
            test_size: 2,
            step_size: None,
        };
        let folds = sliding_window(11, &config).expect("Should succeed");
        assert_eq!(folds.len(), 3);

        // Fold 0: train=[0..5], test=[5..7]
        assert_eq!(folds[0].train_start, 0);
        assert_eq!(folds[0].train_end, 5);
        assert_eq!(folds[0].test_end, 7);

        // Fold 1: train=[2..7], test=[7..9]
        assert_eq!(folds[1].train_start, 2);
        assert_eq!(folds[1].train_end, 7);

        // Training size is always fixed
        for fold in &folds {
            assert_eq!(fold.train_size(), 5);
            assert_eq!(fold.test_size(), 2);
        }
    }

    #[test]
    fn test_sliding_window_step_one() {
        let config = SlidingWindowConfig {
            train_size: 5,
            test_size: 2,
            step_size: Some(1),
        };
        let folds = sliding_window(10, &config).expect("Should succeed");
        assert!(folds.len() >= 3);

        // All train sizes must be 5
        for fold in &folds {
            assert_eq!(fold.train_size(), 5);
        }
    }

    #[test]
    fn test_sliding_window_insufficient() {
        let config = SlidingWindowConfig {
            train_size: 10,
            test_size: 5,
            step_size: None,
        };
        assert!(sliding_window(12, &config).is_err());
    }

    // --- Blocked split tests ---

    #[test]
    fn test_blocked_split_basic() {
        let config = BlockedSplitConfig { n_blocks: 4 };
        let folds = blocked_split(20, &config).expect("Should succeed");
        assert_eq!(folds.len(), 3); // n_blocks - 1

        // Each fold's test starts where the previous one's ended
        assert_eq!(folds[0].train_end, folds[0].test_start);
        assert_eq!(folds[0].train_start, 0);

        // Training grows with each fold
        assert!(folds[1].train_size() > folds[0].train_size());
        assert!(folds[2].train_size() > folds[1].train_size());
    }

    #[test]
    fn test_blocked_split_two_blocks() {
        let config = BlockedSplitConfig { n_blocks: 2 };
        let folds = blocked_split(10, &config).expect("Should succeed");
        assert_eq!(folds.len(), 1);
        assert_eq!(folds[0].train_start, 0);
        assert_eq!(folds[0].train_end, 5);
        assert_eq!(folds[0].test_start, 5);
        assert_eq!(folds[0].test_end, 10);
    }

    #[test]
    fn test_blocked_split_invalid() {
        assert!(blocked_split(10, &BlockedSplitConfig { n_blocks: 1 }).is_err());
        assert!(blocked_split(3, &BlockedSplitConfig { n_blocks: 5 }).is_err());
    }

    // --- Purged CV tests ---

    #[test]
    fn test_purged_cv_basic() {
        let config = PurgedCVConfig {
            n_folds: 4,
            embargo_size: 2,
        };
        let folds = purged_cv(40, &config).expect("Should succeed");

        // All folds should have a gap between train_end and test_start
        for fold in &folds {
            assert!(
                fold.test_start >= fold.train_end + config.embargo_size || fold.train_end == 0,
                "Embargo violated: train_end={}, test_start={}",
                fold.train_end,
                fold.test_start
            );
        }
    }

    #[test]
    fn test_purged_cv_no_embargo() {
        let config = PurgedCVConfig {
            n_folds: 3,
            embargo_size: 0,
        };
        let folds = purged_cv(30, &config).expect("Should succeed");

        // Without embargo, should behave like blocked split
        for fold in &folds {
            assert_eq!(fold.train_end, fold.test_start);
        }
    }

    #[test]
    fn test_purged_cv_large_embargo() {
        let config = PurgedCVConfig {
            n_folds: 3,
            embargo_size: 100,
        };
        assert!(purged_cv(10, &config).is_err());
    }

    // --- Walk-forward tests ---

    #[test]
    fn test_walk_forward_expanding() {
        let config = WalkForwardConfig {
            initial_train_size: 5,
            forecast_horizon: 1,
            expanding: true,
        };
        let folds = walk_forward(10, &config).expect("Should succeed");
        assert_eq!(folds.len(), 5);

        // All start at 0 (expanding)
        for fold in &folds {
            assert_eq!(fold.train_start, 0);
            assert_eq!(fold.test_size(), 1);
        }

        // Training grows
        assert!(folds[1].train_size() > folds[0].train_size());
    }

    #[test]
    fn test_walk_forward_fixed() {
        let config = WalkForwardConfig {
            initial_train_size: 5,
            forecast_horizon: 1,
            expanding: false,
        };
        let folds = walk_forward(10, &config).expect("Should succeed");

        // Training size should be fixed (5) after the first fold
        for fold in folds.iter().skip(1) {
            assert_eq!(fold.train_size(), 5);
        }
    }

    #[test]
    fn test_walk_forward_multi_step() {
        let config = WalkForwardConfig {
            initial_train_size: 5,
            forecast_horizon: 3,
            expanding: true,
        };
        let folds = walk_forward(20, &config).expect("Should succeed");

        for fold in &folds {
            assert_eq!(fold.test_size(), 3);
        }
    }

    #[test]
    fn test_walk_forward_insufficient() {
        let config = WalkForwardConfig {
            initial_train_size: 10,
            forecast_horizon: 5,
            expanding: true,
        };
        assert!(walk_forward(12, &config).is_err());
    }

    // --- Run CV tests ---

    #[test]
    fn test_run_cv_basic() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        let config = ExpandingWindowConfig {
            initial_train_size: 5,
            test_size: 2,
            step_size: None,
        };
        let folds = expanding_window(data.len(), &config).expect("Should get folds");

        // Simple evaluator: MAE of naive forecast (last training value)
        let eval_fn = |train: &Array1<f64>, test: &Array1<f64>| -> Result<f64> {
            let last_train = train[train.len() - 1];
            let n = test.len() as f64;
            let sum_err: f64 = test.iter().map(|&t| (t - last_train).abs()).sum();
            Ok(sum_err / n)
        };

        let summary = run_cv(&data, &folds, eval_fn).expect("CV should succeed");
        assert_eq!(summary.n_folds, folds.len());
        assert!(summary.mean_error >= 0.0);
        assert!(summary.min_error <= summary.max_error);
        assert!(summary.median_error >= summary.min_error);
        assert!(summary.median_error <= summary.max_error);
    }

    #[test]
    fn test_run_cv_zero_error() {
        let data = array![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let config = ExpandingWindowConfig {
            initial_train_size: 5,
            test_size: 2,
            step_size: None,
        };
        let folds = expanding_window(data.len(), &config).expect("Should get folds");

        let eval_fn = |train: &Array1<f64>, test: &Array1<f64>| -> Result<f64> {
            let last = train[train.len() - 1];
            let n = test.len() as f64;
            let err: f64 = test.iter().map(|&t| (t - last).abs()).sum();
            Ok(err / n)
        };

        let summary = run_cv(&data, &folds, eval_fn).expect("CV should succeed");
        assert!(summary.mean_error.abs() < TOL);
    }

    #[test]
    fn test_summary_statistics() {
        let fold_results = vec![
            FoldResult {
                fold_index: 0,
                train_size: 5,
                test_size: 2,
                error: 1.0,
            },
            FoldResult {
                fold_index: 1,
                train_size: 7,
                test_size: 2,
                error: 3.0,
            },
            FoldResult {
                fold_index: 2,
                train_size: 9,
                test_size: 2,
                error: 2.0,
            },
        ];

        let summary = compute_summary(fold_results).expect("Should succeed");
        assert_eq!(summary.n_folds, 3);
        assert!((summary.mean_error - 2.0).abs() < TOL);
        assert!((summary.min_error - 1.0).abs() < TOL);
        assert!((summary.max_error - 3.0).abs() < TOL);
        assert!((summary.median_error - 2.0).abs() < TOL);
    }

    #[test]
    fn test_summary_single_fold() {
        let fold_results = vec![FoldResult {
            fold_index: 0,
            train_size: 10,
            test_size: 5,
            error: 1.5,
        }];

        let summary = compute_summary(fold_results).expect("Should succeed");
        assert_eq!(summary.n_folds, 1);
        assert!((summary.mean_error - 1.5).abs() < TOL);
        assert!((summary.std_error).abs() < TOL);
    }

    #[test]
    fn test_summary_empty() {
        let fold_results: Vec<FoldResult<f64>> = vec![];
        assert!(compute_summary(fold_results).is_err());
    }

    #[test]
    fn test_fold_data_extraction() {
        let data = array![10.0, 20.0, 30.0, 40.0, 50.0];
        let fold = TimeSeriesFold {
            train_start: 0,
            train_end: 3,
            test_start: 3,
            test_end: 5,
        };

        let train = fold.train_data(&data);
        assert_eq!(train.len(), 3);
        assert!((train[0] - 10.0).abs() < TOL);
        assert!((train[2] - 30.0).abs() < TOL);

        let test = fold.test_data(&data);
        assert_eq!(test.len(), 2);
        assert!((test[0] - 40.0).abs() < TOL);
        assert!((test[1] - 50.0).abs() < TOL);
    }

    #[test]
    fn test_fold_display() {
        let fold = TimeSeriesFold {
            train_start: 0,
            train_end: 10,
            test_start: 10,
            test_end: 15,
        };
        let s = format!("{fold}");
        assert!(s.contains("train=[0..10]"));
        assert!(s.contains("test=[10..15]"));
    }

    #[test]
    fn test_no_overlap_expanding() {
        let config = ExpandingWindowConfig {
            initial_train_size: 3,
            test_size: 2,
            step_size: Some(2),
        };
        let folds = expanding_window(15, &config).expect("Should succeed");

        // Train end should equal test start (no gap, no overlap)
        for fold in &folds {
            assert_eq!(fold.train_end, fold.test_start);
        }
    }

    #[test]
    fn test_no_overlap_sliding() {
        let config = SlidingWindowConfig {
            train_size: 4,
            test_size: 2,
            step_size: Some(2),
        };
        let folds = sliding_window(12, &config).expect("Should succeed");

        for fold in &folds {
            assert_eq!(fold.train_end, fold.test_start);
        }
    }

    #[test]
    fn test_walk_forward_no_data_leak() {
        let config = WalkForwardConfig {
            initial_train_size: 5,
            forecast_horizon: 1,
            expanding: true,
        };
        let folds = walk_forward(20, &config).expect("Should succeed");

        // Ensure test data is always strictly after train data
        for fold in &folds {
            assert!(fold.test_start >= fold.train_end);
        }
    }

    #[test]
    fn test_cv_summary_display() {
        let summary = CVSummary {
            n_folds: 3,
            mean_error: 1.5,
            std_error: 0.5,
            min_error: 1.0,
            max_error: 2.0,
            median_error: 1.5,
            fold_results: vec![],
        };
        let s = format!("{summary}");
        assert!(s.contains("3 folds"));
        assert!(s.contains("Mean error"));
    }

    // --- CrossValidator tests ---

    #[test]
    fn test_cross_validator_basic() {
        let data = Array1::linspace(0.0_f64, 1.0, 200);
        let cv = CrossValidator::new(5, 0.2);
        let folds = cv.time_series_split(&data).expect("should succeed");
        assert_eq!(folds.len(), 5);
        // Each fold's train data ends where test data begins
        for fold in &folds {
            assert_eq!(fold.train_end, fold.test_start);
            assert!(fold.test_end > fold.test_start);
            assert!(fold.train_end > fold.train_start);
        }
    }

    #[test]
    fn test_cross_validator_no_holdout() {
        let data = Array1::linspace(0.0_f64, 1.0, 100);
        let cv = CrossValidator::new(3, 0.0);
        let folds = cv.time_series_split(&data).expect("should succeed");
        assert_eq!(folds.len(), 3);
    }

    #[test]
    fn test_cross_validator_invalid_n_folds() {
        let data = Array1::linspace(0.0_f64, 1.0, 100);
        let cv = CrossValidator::new(1, 0.0);
        assert!(cv.time_series_split(&data).is_err());
    }

    #[test]
    fn test_cross_validator_fold_indices_in_range() {
        let data = Array1::linspace(0.0_f64, 1.0, 200);
        let cv = CrossValidator::new(5, 0.2);
        let folds = cv.time_series_split(&data).expect("should succeed");
        for fold in &folds {
            assert!(fold.train_end <= data.len());
            assert!(fold.test_end <= data.len());
            assert!(fold.train_start < fold.train_end);
            assert!(fold.test_start < fold.test_end);
        }
    }
}
