use crate::error::MetricsError;
use scirs2_core::ndarray::{Array1, ArrayBase, Data, Dimension};

/// Compute calibration curve (also known as reliability diagram).
///
/// The calibration curve plots the mean predicted probability against
/// the fraction of positive samples.
///
/// # Arguments
///
/// * `y_true` - Ground truth (correct) labels
/// * `y_prob` - Predicted probabilities
/// * `n_bins` - Number of bins to use for the curve (default: 5)
///
/// # Returns
///
/// A tuple containing:
/// * `prob_true` - The fraction of positive samples in each bin
/// * `prob_pred` - The mean predicted probability in each bin
/// * `counts` - The number of samples in each bin
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_metrics::classification::curves::calibration_curve;
///
/// let y_true = array![0, 0, 1, 1, 1];
/// let y_prob = array![0.1, 0.4, 0.35, 0.8, 0.6];
///
/// let (prob_true, prob_pred, counts) = calibration_curve(&y_true, &y_prob, Some(5)).unwrap();
/// ```
/// Type alias for calibration curve result
pub type CalibrationCurveResult = (Array1<f64>, Array1<f64>, Array1<usize>);

#[allow(dead_code)]
pub fn calibration_curve<S1, S2, D1, D2>(
    y_true: &ArrayBase<S1, D1>,
    y_prob: &ArrayBase<S2, D2>,
    n_bins: Option<usize>,
) -> Result<CalibrationCurveResult, MetricsError>
where
    S1: Data,
    S2: Data,
    S1::Elem: PartialEq + Clone + Into<f64>,
    S2::Elem: PartialOrd + Clone + Into<f64>,
    D1: Dimension,
    D2: Dimension,
{
    // Ensure arrays have same shape
    if y_true.shape() != y_prob.shape() {
        return Err(MetricsError::ShapeMismatch {
            shape1: format!("{:?}", y_true.shape()),
            shape2: format!("{:?}", y_prob.shape()),
        });
    }

    let bins = n_bins.unwrap_or(5);
    if bins < 2 {
        return Err(MetricsError::InvalidArgument(
            "Number of bins must be at least 2".to_string(),
        ));
    }

    // Convert arrays to Vec for easier processing
    let y_true_vec: Vec<f64> = y_true
        .iter()
        .map(|x| {
            let val: f64 = x.clone().into();
            val
        })
        .collect();

    let y_prob_vec: Vec<f64> = y_prob
        .iter()
        .map(|x| {
            let val: f64 = x.clone().into();
            // Ensure probabilities are in [0, 1]
            if !(0.0..=1.0).contains(&val) {
                return Err(MetricsError::InvalidArgument(
                    "Probability values must be in range [0, 1]".to_string(),
                ));
            }
            Ok(val)
        })
        .collect::<Result<Vec<f64>, MetricsError>>()?;

    // Define bin edges
    let bin_width = 1.0 / bins as f64;
    let edges: Vec<f64> = (0..=bins).map(|i| i as f64 * bin_width).collect();

    // Initialize arrays for results
    let mut prob_true = vec![0.0; bins];
    let mut prob_pred = vec![0.0; bins];
    let mut counts = vec![0; bins];

    // Assign samples to bins and compute statistics
    for (true_val, prob_val) in y_true_vec.iter().zip(y_prob_vec.iter()) {
        // Find the bin index
        let mut bin_idx = bins - 1;
        for i in 0..bins {
            if *prob_val < edges[i + 1] {
                bin_idx = i;
                break;
            }
        }

        // Update bin statistics
        prob_true[bin_idx] += true_val;
        prob_pred[bin_idx] += prob_val;
        counts[bin_idx] += 1;
    }

    // Compute mean values for each bin
    for i in 0..bins {
        if counts[i] > 0 {
            prob_true[i] /= counts[i] as f64;
            prob_pred[i] /= counts[i] as f64;
        }
    }

    Ok((
        Array1::from_vec(prob_true),
        Array1::from_vec(prob_pred),
        Array1::from_vec(counts),
    ))
}
