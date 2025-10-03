use crate::error::MetricsError;
use scirs2_core::ndarray::{Array1, ArrayBase, Data, Dimension};

/// Compute precision-recall curve points.
///
/// The precision-recall curve shows the relationship between precision and recall
/// at various threshold settings.
///
/// # Arguments
///
/// * `y_true` - Ground truth (correct) labels
/// * `y_score` - Target scores (estimated probabilities)
///
/// # Returns
///
/// A tuple containing:
/// * `precision` - Precision values
/// * `recall` - Recall values
/// * `thresholds` - Thresholds used to compute precision and recall
///
/// # Examples
///
/// ```
/// use scirs2_core::ndarray::array;
/// use scirs2_metrics::classification::curves::precision_recall_curve;
///
/// let y_true = array![0, 0, 1, 1, 1];
/// let y_score = array![0.1, 0.4, 0.35, 0.8, 0.6];
///
/// let (precision, recall, thresholds) = precision_recall_curve(&y_true, &y_score).unwrap();
/// ```
/// Type alias for precision-recall curve result
pub type PrecisionRecallCurveResult = (Array1<f64>, Array1<f64>, Array1<f64>);

#[allow(dead_code)]
pub fn precision_recall_curve<S1, S2, D1, D2>(
    y_true: &ArrayBase<S1, D1>,
    y_score: &ArrayBase<S2, D2>,
) -> Result<PrecisionRecallCurveResult, MetricsError>
where
    S1: Data,
    S2: Data,
    S1::Elem: PartialEq + Clone + Into<f64>,
    S2::Elem: PartialOrd + Clone + Into<f64>,
    D1: Dimension,
    D2: Dimension,
{
    // Ensure arrays have same shape
    if y_true.shape() != y_score.shape() {
        return Err(MetricsError::ShapeMismatch {
            shape1: format!("{:?}", y_true.shape()),
            shape2: format!("{:?}", y_score.shape()),
        });
    }

    // Convert arrays to Vec for easier processing
    let mut y_true_vec: Vec<f64> = y_true
        .iter()
        .map(|x| {
            let val: f64 = x.clone().into();
            val
        })
        .collect();

    let mut y_score_vec: Vec<f64> = y_score
        .iter()
        .map(|x| {
            let val: f64 = x.clone().into();
            val
        })
        .collect();

    // Create pairs and sort by decreasing score
    let mut pairs: Vec<(f64, f64)> = y_true_vec
        .iter()
        .cloned()
        .zip(y_score_vec.iter().cloned())
        .collect();

    pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Extract sorted values
    for (i, (label, score)) in pairs.iter().enumerate() {
        y_true_vec[i] = *label;
        y_score_vec[i] = *score;
    }

    // Get unique thresholds
    let mut thresholds: Vec<f64> = y_score_vec.clone();
    thresholds.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    thresholds.dedup();

    // Compute precision and recall for each threshold
    let pos_count: f64 = y_true_vec.iter().filter(|&&x| x > 0.5).count() as f64;

    if pos_count == 0.0 {
        return Err(MetricsError::InvalidArgument(
            "Precision-recall curve requires positive samples".to_string(),
        ));
    }

    let mut precision: Vec<f64> = Vec::with_capacity(thresholds.len() + 1);
    let mut recall: Vec<f64> = Vec::with_capacity(thresholds.len() + 1);

    for t in &thresholds {
        let tp = y_true_vec
            .iter()
            .zip(y_score_vec.iter())
            .filter(|&(&y, &s)| y > 0.5 && s >= *t)
            .count() as f64;

        let fp = y_true_vec
            .iter()
            .zip(y_score_vec.iter())
            .filter(|&(&y, &s)| y <= 0.5 && s >= *t)
            .count() as f64;

        let prec = if tp + fp > 0.0 { tp / (tp + fp) } else { 1.0 };
        let rec = tp / pos_count;

        precision.push(prec);
        recall.push(rec);
    }

    // Add a point at recall=0, precision=1
    precision.push(1.0);
    recall.push(0.0);

    Ok((
        Array1::from_vec(precision),
        Array1::from_vec(recall),
        Array1::from_vec(thresholds),
    ))
}
