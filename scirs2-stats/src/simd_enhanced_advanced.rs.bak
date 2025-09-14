//! Advanced SIMD optimizations for complex statistical operations
//!
//! This module provides additional SIMD-accelerated implementations for advanced
//! statistical functions that can benefit significantly from vectorization.

use crate::error::{StatsError, StatsResult};
use ndarray::{Array1, Array2, ArrayBase, Axis, Data, Ix1, Ix2};
use num_traits::{Float, FromPrimitive, NumCast};
use scirs2_core::simd_ops::SimdUnifiedOps;

/// SIMD-optimized t-test computation
///
/// Performs a two-sample t-test using SIMD acceleration for large arrays.
#[allow(dead_code)]
pub fn ttest_ind_simd<F, D1, D2>(
    a: &ArrayBase<D1, Ix1>,
    b: &ArrayBase<D2, Ix1>,
    equal_var: bool,
) -> StatsResult<(F, F)>
where
    F: Float + NumCast + SimdUnifiedOps,
    D1: Data<Elem = F>,
    D2: Data<Elem = F>,
{
    if a.is_empty() || b.is_empty() {
        return Err(StatsError::invalid_argument("Arrays cannot be empty"));
    }

    let n1 = a.len();
    let n2 = b.len();

    // Compute means using SIMD
    let mean1 = if n1 > 16 {
        F::simd_sum(&a.view()) / F::from(n1).unwrap()
    } else {
        a.iter().fold(F::zero(), |acc, &x| acc + x) / F::from(n1).unwrap()
    };

    let mean2 = if n2 > 16 {
        F::simd_sum(&b.view()) / F::from(n2).unwrap()
    } else {
        b.iter().fold(F::zero(), |acc, &x| acc + x) / F::from(n2).unwrap()
    };

    // Compute variances using SIMD
    let var1 = if n1 > 16 {
        let mean_array = Array1::from_elem(n1, mean1);
        let diff = F::simd_sub(&a.view(), &mean_array.view());
        let sq_diff = F::simd_mul(&diff.view(), &diff.view());
        F::simd_sum(&sq_diff.view()) / F::from(n1 - 1).unwrap()
    } else {
        a.iter()
            .map(|&x| {
                let diff = x - mean1;
                diff * diff
            })
            .fold(F::zero(), |acc, x| acc + x)
            / F::from(n1 - 1).unwrap()
    };

    let var2 = if n2 > 16 {
        let mean_array = Array1::from_elem(n2, mean2);
        let diff = F::simd_sub(&b.view(), &mean_array.view());
        let sq_diff = F::simd_mul(&diff.view(), &diff.view());
        F::simd_sum(&sq_diff.view()) / F::from(n2 - 1).unwrap()
    } else {
        b.iter()
            .map(|&x| {
                let diff = x - mean2;
                diff * diff
            })
            .fold(F::zero(), |acc, x| acc + x)
            / F::from(n2 - 1).unwrap()
    };

    // Compute t-statistic
    let (t_stat, df) = if equal_var {
        // Pooled variance
        let pooled_var = ((F::from(n1 - 1).unwrap() * var1) + (F::from(n2 - 1).unwrap() * var2))
            / F::from(n1 + n2 - 2).unwrap();

        let se = (pooled_var * (F::one() / F::from(n1).unwrap() + F::one() / F::from(n2).unwrap()))
            .sqrt();
        let t = (mean1 - mean2) / se;
        let df = F::from(n1 + n2 - 2).unwrap();
        (t, df)
    } else {
        // Welch's t-test (unequal variances)
        let se1_sq = var1 / F::from(n1).unwrap();
        let se2_sq = var2 / F::from(n2).unwrap();
        let se = (se1_sq + se2_sq).sqrt();
        let t = (mean1 - mean2) / se;

        // Welch-Satterthwaite equation for degrees of freedom
        let num = (se1_sq + se2_sq) * (se1_sq + se2_sq);
        let den = (se1_sq * se1_sq) / F::from(n1 - 1).unwrap()
            + (se2_sq * se2_sq) / F::from(n2 - 1).unwrap();
        let df = num / den;
        (t, df)
    };

    // Simplified p-value calculation (two-tailed)
    let p_value = simplified_t_pvalue(t_stat.abs(), df);

    Ok((t_stat, p_value))
}

/// SIMD-optimized matrix multiplication for correlation matrices
///
/// Efficiently computes correlation matrices using SIMD operations.
#[allow(dead_code)]
pub fn corrcoef_matrix_simd<F, D>(data: &ArrayBase<D, Ix2>) -> StatsResult<Array2<F>>
where
    F: Float + NumCast + SimdUnifiedOps + FromPrimitive + Clone,
    D: Data<Elem = F>,
{
    let (n_samples_, n_features) = data.dim();

    if n_samples_ < 2 {
        return Err(StatsError::invalid_argument("Need at least 2 samples"));
    }

    // Center the data
    let means = data.mean_axis(Axis(0)).unwrap();
    let mut centered = data.to_owned();
    for i in 0..n_samples_ {
        for j in 0..n_features {
            centered[[i, j]] = centered[[i, j]] - means[j];
        }
    }

    let mut corr_matrix = Array2::zeros((n_features, n_features));

    // Compute correlation matrix using SIMD
    for i in 0..n_features {
        for j in i..n_features {
            let col_i = centered.column(i);
            let col_j = centered.column(j);

            let corr = if n_samples_ > 16 {
                // SIMD path
                let dot_product = F::simd_mul(&col_i.view(), &col_j.view());
                let dot_sum = F::simd_sum(&dot_product.view());

                let i_sq = F::simd_mul(&col_i.view(), &col_i.view());
                let i_sq_sum = F::simd_sum(&i_sq.view());

                let j_sq = F::simd_mul(&col_j.view(), &col_j.view());
                let j_sq_sum = F::simd_sum(&j_sq.view());

                let denominator = (i_sq_sum * j_sq_sum).sqrt();
                if denominator > F::epsilon() {
                    dot_sum / denominator
                } else {
                    F::zero()
                }
            } else {
                // Scalar fallback
                let mut dot_product = F::zero();
                let mut i_sq_sum = F::zero();
                let mut j_sq_sum = F::zero();

                for (&xi, &xj) in col_i.iter().zip(col_j.iter()) {
                    dot_product = dot_product + xi * xj;
                    i_sq_sum = i_sq_sum + xi * xi;
                    j_sq_sum = j_sq_sum + xj * xj;
                }

                let denominator = (i_sq_sum * j_sq_sum).sqrt();
                if denominator > F::epsilon() {
                    dot_product / denominator
                } else {
                    F::zero()
                }
            };

            corr_matrix[[i, j]] = corr;
            corr_matrix[[j, i]] = corr;
        }
    }

    Ok(corr_matrix)
}

/// SIMD-optimized robust statistics computation
///
/// Computes robust statistics using SIMD acceleration where applicable.
#[allow(dead_code)]
pub fn robust_statistics_simd<F, D>(data: &ArrayBase<D, Ix1>) -> StatsResult<(F, F, F)>
// (median, mad, iqr)
where
    F: Float + NumCast + SimdUnifiedOps,
    D: Data<Elem = F>,
{
    if data.is_empty() {
        return Err(StatsError::invalid_argument("Data cannot be empty"));
    }

    let n = data.len();
    let mut sorteddata = data.to_vec();
    sorteddata.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Compute median
    let median = if n % 2 == 0 {
        (sorteddata[n / 2 - 1] + sorteddata[n / 2]) / F::from(2).unwrap()
    } else {
        sorteddata[n / 2]
    };

    // Compute MAD (Median Absolute Deviation)
    let mut deviations = Array1::zeros(n);
    for i in 0..n {
        deviations[i] = (data[i] - median).abs();
    }

    let mad = if n > 16 {
        // SIMD-accelerated MAD calculation would require custom median implementation
        // For now, use standard approach
        let mut dev_sorted = deviations.to_vec();
        dev_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        if n % 2 == 0 {
            (dev_sorted[n / 2 - 1] + dev_sorted[n / 2]) / F::from(2).unwrap()
        } else {
            dev_sorted[n / 2]
        }
    } else {
        let mut dev_sorted = deviations.to_vec();
        dev_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        if n % 2 == 0 {
            (dev_sorted[n / 2 - 1] + dev_sorted[n / 2]) / F::from(2).unwrap()
        } else {
            dev_sorted[n / 2]
        }
    };

    // Compute IQR (Interquartile Range)
    let q1_idx = n / 4;
    let q3_idx = 3 * n / 4;
    let q1 = sorteddata[q1_idx];
    let q3 = sorteddata[q3_idx];
    let iqr = q3 - q1;

    Ok((median, mad, iqr))
}

/// SIMD-optimized bootstrap statistics
///
/// Performs bootstrap resampling with SIMD-accelerated statistic computation.
#[allow(dead_code)]
pub fn bootstrap_mean_simd<F, D>(
    data: &ArrayBase<D, Ix1>,
    n_bootstrap: usize,
    seed: Option<u64>,
) -> StatsResult<Array1<F>>
where
    F: Float + NumCast + SimdUnifiedOps,
    D: Data<Elem = F>,
{
    if data.is_empty() {
        return Err(StatsError::invalid_argument("Data cannot be empty"));
    }

    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => {
            use std::time::{SystemTime, UNIX_EPOCH};
            let s = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs();
            StdRng::seed_from_u64(s)
        }
    };

    let n = data.len();
    let mut bootstrap_means = Array1::zeros(n_bootstrap);

    for i in 0..n_bootstrap {
        // Generate _bootstrap sample
        let mut bootstrap_sample = Array1::zeros(n);
        for j in 0..n {
            let idx = rng.gen_range(0..n);
            bootstrap_sample[j] = data[idx];
        }

        // Compute mean using SIMD if beneficial
        let bootstrap_mean = if n > 16 {
            F::simd_sum(&bootstrap_sample.view()) / F::from(n).unwrap()
        } else {
            bootstrap_sample.iter().fold(F::zero(), |acc, &x| acc + x) / F::from(n).unwrap()
        };

        bootstrap_means[i] = bootstrap_mean;
    }

    Ok(bootstrap_means)
}

/// SIMD-optimized linear regression
///
/// Computes linear regression coefficients using SIMD acceleration.
#[allow(dead_code)]
pub fn linear_regression_simd<F, D1, D2>(
    x: &ArrayBase<D1, Ix1>,
    y: &ArrayBase<D2, Ix1>,
) -> StatsResult<(F, F)>
// (slope, intercept)
where
    F: Float + NumCast + SimdUnifiedOps,
    D1: Data<Elem = F>,
    D2: Data<Elem = F>,
{
    if x.len() != y.len() {
        return Err(StatsError::dimension_mismatch(
            "x and y must have same length",
        ));
    }

    if x.len() < 2 {
        return Err(StatsError::invalid_argument("Need at least 2 data points"));
    }

    let n = x.len();

    // Compute means using SIMD
    let (mean_x, mean_y) = if n > 16 {
        let sum_x = F::simd_sum(&x.view());
        let sum_y = F::simd_sum(&y.view());
        (sum_x / F::from(n).unwrap(), sum_y / F::from(n).unwrap())
    } else {
        let sum_x = x.iter().fold(F::zero(), |acc, &val| acc + val);
        let sum_y = y.iter().fold(F::zero(), |acc, &val| acc + val);
        (sum_x / F::from(n).unwrap(), sum_y / F::from(n).unwrap())
    };

    // Compute slope and intercept using SIMD
    let (numerator, denominator) = if n > 16 {
        let mean_x_array = Array1::from_elem(n, mean_x);
        let mean_y_array = Array1::from_elem(n, mean_y);

        let x_diff = F::simd_sub(&x.view(), &mean_x_array.view());
        let y_diff = F::simd_sub(&y.view(), &mean_y_array.view());

        let xy_prod = F::simd_mul(&x_diff.view(), &y_diff.view());
        let x_sq = F::simd_mul(&x_diff.view(), &x_diff.view());

        let numerator = F::simd_sum(&xy_prod.view());
        let denominator = F::simd_sum(&x_sq.view());

        (numerator, denominator)
    } else {
        let mut numerator = F::zero();
        let mut denominator = F::zero();

        for (&xi, &yi) in x.iter().zip(y.iter()) {
            let x_diff = xi - mean_x;
            let y_diff = yi - mean_y;
            numerator = numerator + x_diff * y_diff;
            denominator = denominator + x_diff * x_diff;
        }

        (numerator, denominator)
    };

    if denominator.abs() < F::epsilon() {
        return Err(StatsError::computation(
            "Cannot compute slope: zero variance in x",
        ));
    }

    let slope = numerator / denominator;
    let intercept = mean_y - slope * mean_x;

    Ok((slope, intercept))
}

/// Simplified t-distribution p-value calculation
#[allow(dead_code)]
fn simplified_t_pvalue<F: Float>(t: F, df: F) -> F {
    // Very rough approximation - in practice would use proper t-distribution
    if df > F::from(30.0).unwrap() {
        // Use normal approximation for large df
        let z = t;
        F::from(2.0).unwrap() * (F::one() - normal_cdf(z.abs()))
    } else {
        // Simple approximation for small df
        let p = F::from(2.0).unwrap() * (F::one() + t * t / df).powf(-df / F::from(2.0).unwrap());
        p.min(F::one())
    }
}

/// Simplified normal CDF approximation
#[allow(dead_code)]
fn normal_cdf<F: Float>(x: F) -> F {
    // Very rough approximation using erf
    let sqrt2 = F::from(1.4142135623730951).unwrap();
    (F::one() + erf_approx(x / sqrt2)) / F::from(2.0).unwrap()
}

/// Error function approximation
#[allow(dead_code)]
fn erf_approx<F: Float>(x: F) -> F {
    // Simple rational approximation
    let a = F::from(0.3275911).unwrap();
    let p = F::from(0.254829592).unwrap();

    let t = F::one() / (F::one() + a * x.abs());
    let y = F::one() - p * t * (-x * x).exp();

    if x >= F::zero() {
        y
    } else {
        -y
    }
}
