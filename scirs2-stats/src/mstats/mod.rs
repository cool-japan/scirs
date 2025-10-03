//! Masked array statistics
//!
//! This module provides statistical functions that work with masked arrays,
//! following SciPy's `stats.mstats` module.

use crate::error::{StatsError, StatsResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};

/// Masked array structure
///
/// Represents an array with associated mask indicating which values are valid/invalid.
#[derive(Debug, Clone)]
pub struct MaskedArray<T> {
    /// The data array
    pub data: Array1<T>,
    /// The mask array (true = valid, false = masked/invalid)
    pub mask: Array1<bool>,
}

impl<T: Copy> MaskedArray<T> {
    /// Create a new masked array
    pub fn new(data: Array1<T>, mask: Array1<bool>) -> StatsResult<Self> {
        if data.len() != mask.len() {
            return Err(StatsError::DimensionMismatch(
                "Data and mask arrays must have the same length".to_string(),
            ));
        }

        Ok(Self { data, mask })
    }

    /// Create a masked array with all values unmasked (valid)
    pub fn fromdata(data: Array1<T>) -> Self {
        let mask = Array1::from_elem(data.len(), true);
        Self { data, mask }
    }

    /// Get the valid (unmasked) values
    pub fn valid_values(&self) -> Vec<T> {
        self.data
            .iter()
            .zip(self.mask.iter())
            .filter_map(|(&value, &is_valid)| if is_valid { Some(value) } else { None })
            .collect()
    }

    /// Count the number of valid values
    pub fn count_valid(&self) -> usize {
        self.mask.iter().filter(|&&is_valid| is_valid).count()
    }

    /// Check if the array has any valid values
    pub fn has_valid_values(&self) -> bool {
        self.count_valid() > 0
    }
}

/// Masked 2D array structure
#[derive(Debug, Clone)]
pub struct MaskedArray2<T> {
    /// The data array
    pub data: Array2<T>,
    /// The mask array (true = valid, false = masked/invalid)
    pub mask: Array2<bool>,
}

impl<T: Copy> MaskedArray2<T> {
    /// Create a new masked 2D array
    pub fn new(data: Array2<T>, mask: Array2<bool>) -> StatsResult<Self> {
        if data.shape() != mask.shape() {
            return Err(StatsError::DimensionMismatch(
                "Data and mask arrays must have the same shape".to_string(),
            ));
        }

        Ok(Self { data, mask })
    }

    /// Create a masked array with all values unmasked (valid)
    pub fn fromdata(data: Array2<T>) -> Self {
        let mask = Array2::from_elem(data.dim(), true);
        Self { data, mask }
    }
}

/// Compute the mean of a masked array
///
/// # Arguments
/// * `maskedarray` - The masked array
/// * `axis` - Axis along which to compute the mean (None for overall mean)
///
/// # Returns
/// * Mean of valid values
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_stats::mstats::{MaskedArray, masked_mean};
///
/// let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
/// let mask = array![true, true, false, true, true]; // 3.0 is masked
/// let masked_arr = MaskedArray::new(data, mask).unwrap();
///
/// let mean = masked_mean(&masked_arr, None).unwrap();
/// assert!((mean - 3.0).abs() < 1e-10); // Mean of [1, 2, 4, 5] = 3.0
/// ```
#[allow(dead_code)]
pub fn masked_mean<T>(maskedarray: &MaskedArray<T>, axis: Option<usize>) -> StatsResult<f64>
where
    T: Copy + Into<f64>,
{
    if !maskedarray.has_valid_values() {
        return Err(StatsError::InvalidArgument(
            "Array has no valid values".to_string(),
        ));
    }

    let valid_values = maskedarray.valid_values();
    let sum: f64 = valid_values.iter().map(|&x| x.into()).sum();
    Ok(sum / valid_values.len() as f64)
}

/// Compute the variance of a masked array
///
/// # Arguments
/// * `maskedarray` - The masked array
/// * `ddof` - Delta degrees of freedom (0 for population variance, 1 for sample variance)
/// * `axis` - Axis along which to compute the variance (None for overall variance)
///
/// # Returns
/// * Variance of valid values
#[allow(dead_code)]
pub fn masked_var<T>(
    maskedarray: &MaskedArray<T>,
    ddof: usize,
    axis: Option<usize>,
) -> StatsResult<f64>
where
    T: Copy + Into<f64>,
{
    if !maskedarray.has_valid_values() {
        return Err(StatsError::InvalidArgument(
            "Array has no valid values".to_string(),
        ));
    }

    let valid_values = maskedarray.valid_values();
    let n = valid_values.len();

    if n <= ddof {
        return Err(StatsError::InvalidArgument(
            "Number of valid values must be greater than ddof".to_string(),
        ));
    }

    let mean = masked_mean(maskedarray, axis)?;
    let sum_squared_diff: f64 = valid_values
        .iter()
        .map(|&x| {
            let diff = x.into() - mean;
            diff * diff
        })
        .sum();

    Ok(sum_squared_diff / (n - ddof) as f64)
}

/// Compute the standard deviation of a masked array
///
/// # Arguments
/// * `maskedarray` - The masked array
/// * `ddof` - Delta degrees of freedom (0 for population std, 1 for sample std)
/// * `axis` - Axis along which to compute the std (None for overall std)
///
/// # Returns
/// * Standard deviation of valid values
#[allow(dead_code)]
pub fn masked_std<T>(
    maskedarray: &MaskedArray<T>,
    ddof: usize,
    axis: Option<usize>,
) -> StatsResult<f64>
where
    T: Copy + Into<f64>,
{
    let variance = masked_var(maskedarray, ddof, axis)?;
    Ok(variance.sqrt())
}

/// Compute the median of a masked array
///
/// # Arguments
/// * `maskedarray` - The masked array
///
/// # Returns
/// * Median of valid values
#[allow(dead_code)]
pub fn masked_median<T>(maskedarray: &MaskedArray<T>) -> StatsResult<f64>
where
    T: Copy + Into<f64> + PartialOrd,
{
    if !maskedarray.has_valid_values() {
        return Err(StatsError::InvalidArgument(
            "Array has no valid values".to_string(),
        ));
    }

    let mut valid_values = maskedarray.valid_values();
    valid_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n = valid_values.len();
    let median = if n % 2 == 1 {
        valid_values[n / 2].into()
    } else {
        let mid1 = valid_values[n / 2 - 1].into();
        let mid2 = valid_values[n / 2].into();
        (mid1 + mid2) / 2.0
    };

    Ok(median)
}

/// Compute quantiles of a masked array
///
/// # Arguments
/// * `maskedarray` - The masked array
/// * `q` - Quantile or sequence of quantiles to compute (0.0 to 1.0)
///
/// # Returns
/// * Array of quantiles
#[allow(dead_code)]
pub fn masked_quantile<T>(
    maskedarray: &MaskedArray<T>,
    q: ArrayView1<f64>,
) -> StatsResult<Array1<f64>>
where
    T: Copy + Into<f64> + PartialOrd,
{
    if !maskedarray.has_valid_values() {
        return Err(StatsError::InvalidArgument(
            "Array has no valid values".to_string(),
        ));
    }

    for &quantile in q.iter() {
        if !(0.0..=1.0).contains(&quantile) {
            return Err(StatsError::InvalidArgument(
                "Quantiles must be between 0 and 1".to_string(),
            ));
        }
    }

    let mut valid_values = maskedarray.valid_values();
    valid_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n = valid_values.len() as f64;
    let mut quantiles = Array1::zeros(q.len());

    for (i, &quantile) in q.iter().enumerate() {
        let index = quantile * (n - 1.0);
        let lower = index.floor() as usize;
        let upper = index.ceil() as usize;
        let fraction = index - lower as f64;

        if lower == upper {
            quantiles[i] = valid_values[lower].into();
        } else {
            let lower_val = valid_values[lower].into();
            let upper_val = valid_values[upper].into();
            quantiles[i] = lower_val + fraction * (upper_val - lower_val);
        }
    }

    Ok(quantiles)
}

/// Compute the correlation coefficient between two masked arrays
///
/// # Arguments
/// * `x` - First masked array
/// * `y` - Second masked array
/// * `method` - Correlation method ("pearson", "spearman", or "kendall")
///
/// # Returns
/// * Correlation coefficient
#[allow(dead_code)]
pub fn masked_corrcoef<T>(x: &MaskedArray<T>, y: &MaskedArray<T>, method: &str) -> StatsResult<f64>
where
    T: Copy + Into<f64> + PartialOrd,
{
    if x.data.len() != y.data.len() {
        return Err(StatsError::DimensionMismatch(
            "Arrays must have the same length".to_string(),
        ));
    }

    // Combine masks (both values must be valid)
    let combined_mask: Array1<bool> = x
        .mask
        .iter()
        .zip(y.mask.iter())
        .map(|(&x_valid, &y_valid)| x_valid && y_valid)
        .collect();

    let valid_pairs: Vec<(T, T)> = x
        .data
        .iter()
        .zip(y.data.iter())
        .zip(combined_mask.iter())
        .filter_map(
            |((&x_val, &y_val), &is_valid)| {
                if is_valid {
                    Some((x_val, y_val))
                } else {
                    None
                }
            },
        )
        .collect();

    if valid_pairs.is_empty() {
        return Err(StatsError::InvalidArgument(
            "No valid pairs found".to_string(),
        ));
    }

    let n = valid_pairs.len() as f64;

    match method {
        "pearson" => {
            let x_values: Vec<f64> = valid_pairs.iter().map(|(x, _)| (*x).into()).collect();
            let y_values: Vec<f64> = valid_pairs.iter().map(|(_, y)| (*y).into()).collect();

            let x_mean: f64 = x_values.iter().sum::<f64>() / n;
            let y_mean: f64 = y_values.iter().sum::<f64>() / n;

            let mut numerator = 0.0;
            let mut x_var = 0.0;
            let mut y_var = 0.0;

            for (&x_val, &y_val) in x_values.iter().zip(y_values.iter()) {
                let x_diff = x_val - x_mean;
                let y_diff = y_val - y_mean;
                numerator += x_diff * y_diff;
                x_var += x_diff * x_diff;
                y_var += y_diff * y_diff;
            }

            if x_var == 0.0 || y_var == 0.0 {
                return Ok(0.0);
            }

            Ok(numerator / (x_var * y_var).sqrt())
        }
        "spearman" => {
            // Convert to ranks
            let mut x_values: Vec<(f64, usize)> = valid_pairs
                .iter()
                .enumerate()
                .map(|(i, (x, _))| ((*x).into(), i))
                .collect();
            let mut y_values: Vec<(f64, usize)> = valid_pairs
                .iter()
                .enumerate()
                .map(|(i, (_, y))| ((*y).into(), i))
                .collect();

            x_values.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            y_values.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            let mut x_ranks = vec![0.0; valid_pairs.len()];
            let mut y_ranks = vec![0.0; valid_pairs.len()];

            for (rank, (_, original_idx)) in x_values.iter().enumerate() {
                x_ranks[*original_idx] = rank as f64 + 1.0;
            }
            for (rank, (_, original_idx)) in y_values.iter().enumerate() {
                y_ranks[*original_idx] = rank as f64 + 1.0;
            }

            // Calculate Pearson correlation on ranks
            let x_rank_mean = x_ranks.iter().sum::<f64>() / n;
            let y_rank_mean = y_ranks.iter().sum::<f64>() / n;

            let mut numerator = 0.0;
            let mut x_var = 0.0;
            let mut y_var = 0.0;

            for (&x_rank, &y_rank) in x_ranks.iter().zip(y_ranks.iter()) {
                let x_diff = x_rank - x_rank_mean;
                let y_diff = y_rank - y_rank_mean;
                numerator += x_diff * y_diff;
                x_var += x_diff * x_diff;
                y_var += y_diff * y_diff;
            }

            if x_var == 0.0 || y_var == 0.0 {
                return Ok(0.0);
            }

            Ok(numerator / (x_var * y_var).sqrt())
        }
        "kendall" => {
            // Kendall's tau
            let mut concordant = 0;
            let mut discordant = 0;

            for i in 0..valid_pairs.len() {
                for j in (i + 1)..valid_pairs.len() {
                    let (x1, y1) = valid_pairs[i];
                    let (x2, y2) = valid_pairs[j];

                    let x1_f64 = x1.into();
                    let y1_f64 = y1.into();
                    let x2_f64 = x2.into();
                    let y2_f64 = y2.into();

                    let x_diff = x2_f64 - x1_f64;
                    let y_diff = y2_f64 - y1_f64;

                    if x_diff * y_diff > 0.0 {
                        concordant += 1;
                    } else if x_diff * y_diff < 0.0 {
                        discordant += 1;
                    }
                    // Ties contribute 0
                }
            }

            let total_pairs = valid_pairs.len() * (valid_pairs.len() - 1) / 2;
            Ok((concordant - discordant) as f64 / total_pairs as f64)
        }
        _ => Err(StatsError::InvalidArgument(
            "Method must be one of 'pearson', 'spearman', or 'kendall'".to_string(),
        )),
    }
}

/// Compute the covariance between two masked arrays
///
/// # Arguments
/// * `x` - First masked array
/// * `y` - Second masked array
/// * `ddof` - Delta degrees of freedom
///
/// # Returns
/// * Covariance
#[allow(dead_code)]
pub fn masked_cov<T>(x: &MaskedArray<T>, y: &MaskedArray<T>, ddof: usize) -> StatsResult<f64>
where
    T: Copy + Into<f64>,
{
    if x.data.len() != y.data.len() {
        return Err(StatsError::DimensionMismatch(
            "Arrays must have the same length".to_string(),
        ));
    }

    // Combine masks (both values must be valid)
    let combined_mask: Array1<bool> = x
        .mask
        .iter()
        .zip(y.mask.iter())
        .map(|(&x_valid, &y_valid)| x_valid && y_valid)
        .collect();

    let valid_pairs: Vec<(T, T)> = x
        .data
        .iter()
        .zip(y.data.iter())
        .zip(combined_mask.iter())
        .filter_map(
            |((&x_val, &y_val), &is_valid)| {
                if is_valid {
                    Some((x_val, y_val))
                } else {
                    None
                }
            },
        )
        .collect();

    if valid_pairs.len() <= ddof {
        return Err(StatsError::InvalidArgument(
            "Number of valid pairs must be greater than ddof".to_string(),
        ));
    }

    let n = valid_pairs.len() as f64;
    let x_values: Vec<f64> = valid_pairs.iter().map(|(x, _)| (*x).into()).collect();
    let y_values: Vec<f64> = valid_pairs.iter().map(|(_, y)| (*y).into()).collect();

    let x_mean: f64 = x_values.iter().sum::<f64>() / n;
    let y_mean: f64 = y_values.iter().sum::<f64>() / n;

    let covariance: f64 = x_values
        .iter()
        .zip(y_values.iter())
        .map(|(&x_val, &y_val)| (x_val - x_mean) * (y_val - y_mean))
        .sum::<f64>()
        / (n - ddof as f64);

    Ok(covariance)
}

/// Compute masked skewness
///
/// # Arguments
/// * `maskedarray` - The masked array
/// * `bias` - If false, use bias-corrected formula
///
/// # Returns
/// * Skewness of valid values
#[allow(dead_code)]
pub fn masked_skew<T>(maskedarray: &MaskedArray<T>, bias: bool) -> StatsResult<f64>
where
    T: Copy + Into<f64>,
{
    if !maskedarray.has_valid_values() {
        return Err(StatsError::InvalidArgument(
            "Array has no valid values".to_string(),
        ));
    }

    let valid_values = maskedarray.valid_values();
    let n = valid_values.len() as f64;

    if n < 3.0 {
        return Err(StatsError::InvalidArgument(
            "Skewness requires at least 3 valid values".to_string(),
        ));
    }

    let mean = masked_mean(maskedarray, None)?;
    let std_dev = masked_std(maskedarray, 1, None)?;

    if std_dev == 0.0 {
        return Ok(0.0);
    }

    let m3: f64 = valid_values
        .iter()
        .map(|&x| {
            let z = (x.into() - mean) / std_dev;
            z.powi(3)
        })
        .sum::<f64>()
        / n;

    if bias {
        Ok(m3)
    } else {
        // Bias-corrected skewness
        let correction = ((n * (n - 1.0)).sqrt()) / (n - 2.0);
        Ok(correction * m3)
    }
}

/// Compute masked kurtosis
///
/// # Arguments
/// * `maskedarray` - The masked array
/// * `fisher` - If true, return Fisher's kurtosis (excess kurtosis)
/// * `bias` - If false, use bias-corrected formula
///
/// # Returns
/// * Kurtosis of valid values
#[allow(dead_code)]
pub fn masked_kurtosis<T>(
    maskedarray: &MaskedArray<T>,
    fisher: bool,
    bias: bool,
) -> StatsResult<f64>
where
    T: Copy + Into<f64>,
{
    if !maskedarray.has_valid_values() {
        return Err(StatsError::InvalidArgument(
            "Array has no valid values".to_string(),
        ));
    }

    let valid_values = maskedarray.valid_values();
    let n = valid_values.len() as f64;

    if n < 4.0 {
        return Err(StatsError::InvalidArgument(
            "Kurtosis requires at least 4 valid values".to_string(),
        ));
    }

    let mean = masked_mean(maskedarray, None)?;
    let std_dev = masked_std(maskedarray, 1, None)?;

    if std_dev == 0.0 {
        return Err(StatsError::InvalidArgument(
            "Standard deviation is zero".to_string(),
        ));
    }

    let m4: f64 = valid_values
        .iter()
        .map(|&x| {
            let z = (x.into() - mean) / std_dev;
            z.powi(4)
        })
        .sum::<f64>()
        / n;

    let kurtosis = if bias {
        m4
    } else {
        // Bias-corrected kurtosis
        let term1 = (n - 1.0) / ((n - 2.0) * (n - 3.0));
        let term2 = (n + 1.0) * m4 - 3.0 * (n - 1.0);
        term1 * term2 + 3.0
    };

    if fisher {
        Ok(kurtosis - 3.0) // Excess kurtosis
    } else {
        Ok(kurtosis)
    }
}

/// Compute trimmed mean of a masked array
///
/// # Arguments
/// * `maskedarray` - The masked array
/// * `proportiontocut` - Fraction of values to trim from each end (0.0 to 0.5)
///
/// # Returns
/// * Trimmed mean of valid values
#[allow(dead_code)]
pub fn masked_tmean<T>(maskedarray: &MaskedArray<T>, proportiontocut: f64) -> StatsResult<f64>
where
    T: Copy + Into<f64> + PartialOrd,
{
    if !(0.0..0.5).contains(&proportiontocut) {
        return Err(StatsError::InvalidArgument(
            "proportiontocut must be between 0 and 0.5".to_string(),
        ));
    }

    if !maskedarray.has_valid_values() {
        return Err(StatsError::InvalidArgument(
            "Array has no valid values".to_string(),
        ));
    }

    let mut valid_values = maskedarray.valid_values();
    valid_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n = valid_values.len();
    let ncut = (n as f64 * proportiontocut).floor() as usize;

    if n <= 2 * ncut {
        return Err(StatsError::InvalidArgument(
            "Too many values would be trimmed".to_string(),
        ));
    }

    let trimmed_values = &valid_values[ncut..(n - ncut)];
    let sum: f64 = trimmed_values.iter().map(|&x| x.into()).sum();

    Ok(sum / trimmed_values.len() as f64)
}
