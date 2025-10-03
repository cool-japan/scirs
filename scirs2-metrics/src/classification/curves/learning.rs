use crate::error::MetricsError;
use scirs2_core::ndarray::{Array1, ArrayBase, Data, Dimension};

/// Compute learning curve points.
///
/// A learning curve shows model performance (e.g., accuracy or loss)
/// as a function of the number of training examples.
///
/// # Arguments
///
/// * `train_sizes` - Array of training set sizes
/// * `train_scores` - Array of scores on training sets
/// * `test_scores` - Array of scores on test sets
///
/// # Returns
///
/// A tuple containing:
/// * `train_sizes_abs` - The absolute numbers of training examples
/// * `train_scores` - The training scores
/// * `test_scores` - The test scores
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_metrics::classification::curves::learning_curve;
///
/// let train_sizes = array![0.2, 0.4, 0.6, 0.8, 1.0];
/// let train_scores = array![0.7, 0.8, 0.85, 0.9, 0.92];
/// let test_scores = array![0.6, 0.7, 0.75, 0.8, 0.85];
///
/// let total_examples = 1000;
/// let (train_sizes_abs, train_scores_result, test_scores_result) =
///     learning_curve(&train_sizes, &train_scores, &test_scores, total_examples).unwrap();
/// ```
/// Type alias for learning curve result
pub type LearningCurveResult = (Array1<usize>, Array1<f64>, Array1<f64>);

#[allow(dead_code)]
pub fn learning_curve<S1, S2, S3, D1, D2, D3>(
    train_sizes: &ArrayBase<S1, D1>,
    train_scores: &ArrayBase<S2, D2>,
    test_scores: &ArrayBase<S3, D3>,
    total_examples: usize,
) -> Result<LearningCurveResult, MetricsError>
where
    S1: Data,
    S2: Data,
    S3: Data,
    S1::Elem: PartialOrd + Clone + Into<f64>,
    S2::Elem: Clone + Into<f64>,
    S3::Elem: Clone + Into<f64>,
    D1: Dimension,
    D2: Dimension,
    D3: Dimension,
{
    // Ensure arrays have same length
    if train_sizes.len() != train_scores.len() || train_sizes.len() != test_scores.len() {
        return Err(MetricsError::ShapeMismatch {
            shape1: format!("{:?}", train_sizes.shape()),
            shape2: format!("{:?}", train_scores.shape()),
        });
    }

    // Convert train_sizes to absolute values if they're fractions
    let train_sizes_abs: Vec<usize> = train_sizes
        .iter()
        .map(|x| {
            let val: f64 = x.clone().into();
            if val <= 0.0 {
                return Err(MetricsError::InvalidArgument(
                    "Training sizes must be positive".to_string(),
                ));
            }

            // If train_sizes are fractions (<=1.0), convert to absolute numbers
            if val <= 1.0 {
                Ok((val * total_examples as f64).round() as usize)
            } else {
                Ok(val as usize)
            }
        })
        .collect::<Result<Vec<usize>, MetricsError>>()?;

    // Convert scores to f64
    let train_scores_f64: Vec<f64> = train_scores
        .iter()
        .map(|x| {
            let val: f64 = x.clone().into();
            val
        })
        .collect();

    let test_scores_f64: Vec<f64> = test_scores
        .iter()
        .map(|x| {
            let val: f64 = x.clone().into();
            val
        })
        .collect();

    Ok((
        Array1::from_vec(train_sizes_abs),
        Array1::from_vec(train_scores_f64),
        Array1::from_vec(test_scores_f64),
    ))
}
